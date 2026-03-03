import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats
import ee
import requests
from io import BytesIO
import os
import argparse
import random

# Initialize GEE
try:
    ee.Initialize(project='ee-gsingh')
except Exception as e:
    print(f"GEE Init Error: {e}")
    # Fallback if already initialized or auth needs to happen
    try:
        ee.Initialize()
    except:
        pass

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_analysis_results.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

class TrajectoryVerifier:
    def __init__(self):
        self.con = duckdb.connect()
        try:
            self.results = pd.read_parquet(RESULTS_PATH)
            print(f"Loaded results for {len(self.results)} pixels.")
        except Exception as e:
            print(f"Error loading results: {e}")
            self.results = pd.DataFrame()

    def get_pixel_data(self, pixel_id):
        """Fetch GPP time series for a pixel."""
        # Use DuckDB to get text data efficiently
        try:
            query = f"SELECT * FROM '{DATA_PATH}' WHERE pixel_id = {pixel_id}"
            df = self.con.query(query).df()
            if df.empty:
                return None
            return df.iloc[0]
        except Exception as e:
            print(f"Error fetching pixel data: {e}")
            return None

    def generate_kml(self, pixel_id, lat, lon):
        """Generate a KML file for the pixel location."""
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Placemark>
    <name>Pixel {pixel_id}</name>
    <description>
      <![CDATA[
        <h3>Pixel Analysis Report</h3>
        <p><b>Latitude:</b> {lat}</p>
        <p><b>Longitude:</b> {lon}</p>
        <p><b>ID:</b> {pixel_id}</p>
      ]]>
    </description>
    <Point>
      <coordinates>{lon},{lat},0</coordinates>
    </Point>
  </Placemark>
</kml>"""
        
        kml_path = os.path.join(OUTPUT_DIR, f"pixel_{pixel_id}.kml")
        with open(kml_path, "w") as f:
            f.write(kml_content)
        print(f"  [KML] Saved to: {kml_path}")
        return kml_path

    def print_pixel_report(self, pixel_id, res_row):
        """Print a detailed summary report for the pixel."""
        slope = res_row.get('sen_slope', np.nan)
        p_val = res_row.get('mk_p_value', np.nan)
        cv = res_row.get('cv_gpp', np.nan)
        mean_gpp = res_row.get('mean_gpp', np.nan)
        cluster = res_row.get('cluster', 'N/A')
        autocorr = res_row.get('autocorr_lag1', np.nan)

        # Interpretation logic
        trend_str = "Stable"
        if p_val < 0.05:
            if slope > 0: trend_str = "Significantly IMPROVING (Recovery)"
            else: trend_str = "Significantly DECLINING (Degradation)"
        elif slope > 0: trend_str = "Slight Improvement (Not Significant)"
        elif slope < 0: trend_str = "Slight Decline (Not Significant)"

        stability_str = "Stable"
        if cv > 0.2: stability_str = "Highly Variable (Unstable)"
        elif cv > 0.1: stability_str = "Moderately Variable"
        
        resilience_str = "Normal Resilience"
        if autocorr > 0.6: resilience_str = "Low Resilience (High Autocorrelation - Warning Signal)"

        print("\n" + "="*60)
        print(f"  PIXEL ANALYSIS REPORT | ID: {pixel_id}")
        print("="*60)
        print(f"  Coordinates:      {res_row.name if isinstance(res_row.name, tuple) else 'Lat/Lon in KML'}")
        print(f"  Cluster Type:     {cluster}")
        print("-" * 60)
        print(f"  1. TREND:         {trend_str}")
        print(f"     slope:         {slope:.2f} gC/m²/yr")
        print(f"     p-value:       {p_val:.4f}")
        print(f"  2. PRODUCTIVITY:  {mean_gpp:.0f} gC/m² (Mean)")
        print(f"  3. STABILITY:     {stability_str}")
        print(f"     CV:            {cv:.2f}")
        print(f"  4. RESILIENCE:    {resilience_str}")
        print(f"     Lag-1 AC:      {autocorr:.2f}")
        print("="*60)
        print("  WHAT TO LOOK FOR IN HISTORICAL IMAGERY:")
        if slope > 5:
            print("  -> Look for: bush encroachment, afforestation, or crop expansion.")
        elif slope < -5:
            print("  -> Look for: clearing, fire scars, drought impacts, or erosion.")
        else:
            print("  -> Look for: steady vegetation cover, seasonal fluctuations only.")
        if cv > 0.25:
             print("  -> Note: This area has wild swings in productivity (disturbance driven?)")
        print("="*60 + "\n")

    def fetch_satellite_image(self, lat, lon, year):
        """Fetch a thumbnail from GEE for a given location and year."""
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(500).bounds() # 1km box
        
        # Landsat Collection Selection
        if year >= 2013:
            col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            bands = ['SR_B4', 'SR_B3', 'SR_B2'] # RGB for L8
        elif year >= 1999:
            col = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            bands = ['SR_B3', 'SR_B2', 'SR_B1'] # RGB for L7
        else:
            col = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            bands = ['SR_B3', 'SR_B2', 'SR_B1'] # RGB for L5
            
        # Filter and Mosaic
        img = (col
               .filterBounds(point)
               .filterDate(f'{year}-01-01', f'{year}-12-31')
               .filter(ee.Filter.lt('CLOUD_COVER', 10))
               .median() # Use median to remove clouds
               .select(bands)
               .multiply(0.0000275).add(-0.2) # Scale factors for C2 L2
        )
        
        # Visualisation parameters
        vis_params = {
            'min': 0,
            'max': 0.3,
            'gamma': 1.4,
            'dimensions': 300,
            'region': region,
            'format': 'png'
        }
        
        try:
            url = img.getThumbURL(vis_params)
            response = requests.get(url)
            return mpimg.imread(BytesIO(response.content), format='png')
        except Exception as e:
            print(f"Error fetching image for {year}: {e}")
            return None

    def plot_pixel(self, pixel_id, show=False, generate_kml=False):
        row = self.get_pixel_data(pixel_id)
        if row is None:
            print(f"Pixel {pixel_id} not found in raw data.")
            return

        # Extract Time Series
        years = np.arange(2000, 2023)
        gpp = [row[f'GPP_{y}'] for y in years]
        
        # Get result metrics
        if self.results.empty:
             print("Warning: Results dataframe is empty. Metrics will be missing.")
             res_row = pd.Series({'sen_slope': 0, 'mk_p_value': 1, 'mean_gpp': np.mean(gpp), 'cv_gpp': 0, 'cluster': -1})
        else:
            res_matches = self.results[self.results['pixel_id'] == pixel_id]
            if res_matches.empty:
                print(f"Pixel {pixel_id} not found in analysis results (maybe not in sample?).")
                # compute on fly?
                res_row = pd.Series({'sen_slope': 0, 'mk_p_value': 1, 'mean_gpp': np.mean(gpp), 'cv_gpp': 0, 'cluster': -1})
            else:
                res_row = res_matches.iloc[0]
        
        # Print Summary
        self.print_pixel_report(pixel_id, res_row)
        
        # Generate KML
        if generate_kml:
            self.generate_kml(pixel_id, row['latitude'], row['longitude'])
        
        # Setup Plot
        fig = plt.figure(figsize=(18, 8))
        gs = fig.add_gridspec(2, 5) # 2 rows, 5 columns
        
        # 1. Time Series Plot (Top, spanning all columns)
        ax_ts = fig.add_subplot(gs[0, :])
        ax_ts.plot(years, gpp, 'o-', color='green', label='GPP', alpha=0.7)
        
        # Add Trend Line (Sen's Slope)
        # Recalculate full Theil-Sen for plotting to get intercept (metrics file only has slope)
        res = stats.theilslopes(gpp, years, alpha=0.95)
        ts_slope, ts_intercept, lo, hi = res
        
        ax_ts.plot(years, ts_intercept + ts_slope * years, 'r--', lw=2, label=f"Sen Slope: {ts_slope:.2f}")
        ax_ts.fill_between(years, ts_intercept + lo * years, ts_intercept + hi * years, color='red', alpha=0.1)
        
        # Title
        p_val = res_row.get('mk_p_value', 1.0)
        sig_star = "*" if p_val < 0.05 else "ns"
        title = (f"Pixel {pixel_id} | Cluster {res_row.get('cluster', '?')} | "
                 f"Slope: {ts_slope:.2f} ({sig_star}, p={p_val:.3f})")
        ax_ts.set_title(title, fontsize=14)
        ax_ts.set_ylabel("GPP (gC/m²/yr)")
        # Dynamically scale Y axis to minimize empty space while keeping 0 as baseline
        y_max = max(gpp) if max(gpp) > 0 else 1000
        ax_ts.set_ylim(0, y_max * 1.15) 
        ax_ts.legend()
        ax_ts.grid(True, alpha=0.3)
        
        # 2. Historical Imagery (Bottom row)
        img_years = [1990, 2000, 2010, 2022]
        
        for i, y in enumerate(img_years):
            ax_img = fig.add_subplot(gs[1, i]) 
            img = self.fetch_satellite_image(row['latitude'], row['longitude'], y)
            if img is not None:
                ax_img.imshow(img)
                # Add red dot at center (pixel location)
                # Thumbnails are 300x300 by default (dimensions param in vis_params)
                ax_img.plot(150, 150, 'ro', markersize=6, markeredgecolor='white', markeredgewidth=1)
            ax_img.set_title(f"Landsat {y}")
            ax_img.axis('off')
            
        # 3. Text Info (Bottom right)
        ax_txt = fig.add_subplot(gs[1, 4])
        ax_txt.axis('off')
        info_text = (
            f"Lat: {row['latitude']:.4f}\n"
            f"Lon: {row['longitude']:.4f}\n\n"
            f"Mean GPP: {res_row.get('mean_gpp', 0):.0f}\n"
            f"CV: {res_row.get('cv_gpp', 0):.2f}\n"
            f"Autocorr: {res_row.get('autocorr_lag1', 0):.2f}\n"
        )
        ax_txt.text(0.1, 0.5, info_text, va='center', fontsize=12)
        
        plt.tight_layout()
        filename = os.path.join(IMAGE_DIR, f"pixel_{pixel_id}_cluster{res_row.get('cluster', 'X')}.png")
        plt.savefig(filename)
        print(f"  [PLOT] Saved to: {filename}")
        if show:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify trajectory analysis results.")
    parser.add_argument("--pixel_id", type=int, help="Specific pixel ID to verify")
    parser.add_argument("--samples", type=int, default=1, help="Number of random samples per cluster")
    parser.add_argument("--kml", action="store_true", help="Generate KML output for viewed pixels")
    args = parser.parse_args()

    verifier = TrajectoryVerifier()
    
    if args.pixel_id is not None:
        print(f"Verifying specific pixel: {args.pixel_id}")
        verifier.plot_pixel(args.pixel_id, show=True, generate_kml=args.kml)
    else:
        # Sample pixels from each cluster
        clusters = sorted(verifier.results['cluster'].unique())
        print(f"Clusters found: {clusters}")
        
        for c in clusters:
            cluster_data = verifier.results[verifier.results['cluster'] == c]
            if cluster_data.empty:
                continue
                
            # Pick random samples
            sample_ids = cluster_data.sample(n=min(len(cluster_data), args.samples))['pixel_id'].tolist()
            
            for pid in sample_ids:
                print(f"Verifying Cluster {c} -> Pixel {pid}")
                try:
                    verifier.plot_pixel(pid, generate_kml=args.kml)
                except Exception as e:
                    print(f"Error plotting {pid}: {e}")



if __name__ == "__main__":
    main()
