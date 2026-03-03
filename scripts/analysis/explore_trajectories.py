import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import geemap
import ee
import os
import argparse
import sys
import webbrowser

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_analysis_results.parquet")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
YEARS = list(range(2000, 2023))

def load_pixel_data(pixel_id=None, index=None, cluster=None):
    """
    Load data for a specific pixel.
    """
    con = duckdb.connect()
    
    # 1. Find the pixel in results
    if index is not None:
        # Get pixel_id from index in results
        query = f"SELECT pixel_id, latitude, longitude, sen_slope, mk_z, mk_p_value, mean_gpp, cluster FROM '{RESULTS_PATH}' LIMIT 1 OFFSET {index}"
        meta = con.execute(query).df()
        if meta.empty:
            print(f"Index {index} out of bounds.")
            return None, None
        pixel_id = meta['pixel_id'].iloc[0]
        
    elif pixel_id is not None:
        query = f"SELECT pixel_id, latitude, longitude, sen_slope, mk_z, mk_p_value, mean_gpp, cluster FROM '{RESULTS_PATH}' WHERE pixel_id = {pixel_id}"
        meta = con.execute(query).df()
        if meta.empty:
            print(f"Pixel ID {pixel_id} not found.")
            return None, None
            
    elif cluster is not None:
        # Pick a random pixel from this cluster
        query = f"SELECT pixel_id, latitude, longitude, sen_slope, mk_z, mk_p_value, mean_gpp, cluster FROM '{RESULTS_PATH}' WHERE cluster = {cluster} ORDER BY RANDOM() LIMIT 1"
        meta = con.execute(query).df()
        if meta.empty:
            print(f"No pixels found for cluster {cluster}.")
            return None, None
        pixel_id = meta['pixel_id'].iloc[0]
        
    else:
        print("Must provide index, pixel_id, or cluster.")
        return None, None

    # 2. Get full time series from raw data
    # We need GPP and SVH columns
    gpp_cols = [f"GPP_{y}" for y in YEARS]
    svh_cols = [f"SVH_{y}" for y in YEARS]
    cols = ",".join(gpp_cols + svh_cols)
    
    query_raw = f"SELECT {cols} FROM '{RAW_DATA_PATH}' WHERE pixel_id = {pixel_id}"
    raw = con.execute(query_raw).df()
    
    return meta.iloc[0], raw.iloc[0]

def plot_timeseries(meta, raw, output_path=None):
    """
    Plot GPP and SVH time series.
    """
    gpp = [raw[f"GPP_{y}"] for y in YEARS]
    svh = [raw[f"SVH_{y}"] for y in YEARS]
    
    # Simple linear trend for GPP
    slope = meta['sen_slope']
    intercept = meta['mean_gpp'] - slope * (len(YEARS) / 2.0) # Approx center
    # Recalculate robust trend line if we want it perfect, but let's just show raw first
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GPP (gC/m²/yr)', color=color)
    ax1.plot(YEARS, gpp, color=color, marker='o', label='GPP')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Add trendline annotation
    ax1.set_title(f"Pixel: {meta['pixel_id']} | Cluster: {meta['cluster']} | Slope: {meta['sen_slope']:.4f} | MK Z: {meta['mk_z']:.2f}")

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:brown'
    ax2.set_ylabel('Short Veg Height (m)', color=color)  # we already handled the x-label with ax1
    ax2.plot(YEARS, svh, color=color, linestyle='--', marker='x', label='SVH')
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    if output_path:
        plt.savefig(output_path)
        print(f"Data plot saved to {output_path}")
    else:
        plt.show()
    plt.close()

import leafmap.maplibregl as leafmap

# ... (imports remain the same)

def create_map(meta, output_html, years=None):
    """
    Create a Leaflet map with high-res imagery using leafmap and Esri Wayback.
    
    Args:
        meta: Pixel metadata (dict-like)
        output_html: Path to save the map HTML
        years: List of years for Wayback imagery, e.g. [2014, 2020]
    """
    try:
        # Initialize map centered on point
        m = leafmap.Map(center=[meta['latitude'], meta['longitude']], zoom=18)
        
        # Add Google Satellite as base
        m.add_basemap("SATELLITE")

        # ---------------------------------------------------------
        # WAYBACK IMAGERY LOGIC
        # ---------------------------------------------------------
        # If specific years requested, use add_wayback_layer
        if years:
            print(f"Adding Wayback imagery for years: {years}...")
            try:
                # Fetch Wayback configuration to find closest dates
                config_url = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
                import requests
                r = requests.get(config_url, timeout=10)
                if r.status_code == 200:
                    config_dict = r.json()
                    releases = [{'releaseNum': k, **v} for k, v in config_dict.items()]
                    
                    # Map requested years to closest release
                    for target_year in years:
                        target_date = pd.to_datetime(f"{target_year}-06-01")
                        
                        # Find closest
                        best_release = None
                        min_diff = float('inf')
                        
                        for rel in releases:
                            # Extract date from itemTitle: "World Imagery (Wayback 2014/02/20)"
                            import re
                            title = rel.get('itemTitle', '')
                            match = re.search(r'Wayback (\d{4}/\d{2}/\d{2})', title)
                            if match:
                                date_str = match.group(1)
                                rel_date = pd.to_datetime(date_str)
                                diff = abs((rel_date - target_date).days)
                                if diff < min_diff:
                                    min_diff = diff
                                    best_release = rel
                                    best_release['releaseDate'] = date_str
                        
                        if best_release:
                            # Convert date format from YYYY/MM/DD to YYYY-MM-DD
                            date_formatted = best_release['releaseDate'].replace('/', '-')
                            print(f"  Year {target_year} -> Wayback {date_formatted}")
                            m.add_wayback_layer(date=date_formatted)
                            
                else:
                    print(f"Failed to fetch Wayback config. Status: {r.status_code}")
            except Exception as e:
                print(f"Error fetching Wayback layers: {e}")

        # Add generic Esri layer too if no specific years or just as backup
        if not years:
             m.add_basemap("Esri.WorldImagery")

        # Save map
        m.to_html(output_html)
        print(f"Map saved to {output_html}")
        return [output_html]
        
    except Exception as e:
        print(f"Error creating map: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    parser = argparse.ArgumentParser(description="Explore Trajectory Analysis Results")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int, help="Row index in results file")
    group.add_argument("--id", type=int, help="Pixel ID")
    group.add_argument("--cluster", type=int, help="Random pixel from this cluster")
    parser.add_argument("--years", nargs='+', type=int, help="List of years for Wayback imagery (e.g. 2010 2015 2020)")
    
    args = parser.parse_args()
    
    print("Loading data...")
    meta, raw = load_pixel_data(index=args.index, pixel_id=args.id, cluster=args.cluster)
    
    if meta is None:
        return

    print("------------------------------------------------")
    print(f"Pixel ID: {meta['pixel_id']}")
    print(f"Lat/Lon: {meta['latitude']}, {meta['longitude']}")
    print(f"Cluster: {meta['cluster']}")
    print(f"GPP Slope: {meta['sen_slope']:.4f}")
    print(f"Mean GPP: {meta['mean_gpp']:.2f}")
    print("------------------------------------------------")

    # Outputs
    os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
    plot_file = os.path.join(BASE_DIR, "plots", f"pixel_{meta['pixel_id']}_plot.png")
    map_file = os.path.join(BASE_DIR, "plots", f"pixel_{meta['pixel_id']}_map.html")
    
    # Plot
    plot_timeseries(meta, raw, output_path=plot_file)
    
    # Map
    maps = create_map(meta, map_file, years=args.years)
    
    if maps:
        # Open in browser
        webbrowser.open('file://' + os.path.abspath(plot_file))
        for m in maps:
             webbrowser.open('file://' + os.path.abspath(m))

if __name__ == "__main__":
    main()
