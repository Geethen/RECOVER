import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os
import argparse
import leafmap

def main():
    parser = argparse.ArgumentParser(description="Generate Interactive Trajectory Map")
    parser.add_argument("--lat_min", type=float, default=-33.0)
    parser.add_argument("--lat_max", type=float, default=-32.5)
    parser.add_argument("--lon_min", type=float, default=18.5)
    parser.add_argument("--lon_max", type=float, default=19.5)
    parser.add_argument("--output_name", type=str, default="spatial_subset_trajectories")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_analysis_results.parquet")
    
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Analysis results not found at {RESULTS_PATH}")
        return

    print(f"Loading results...")
    df = pd.read_parquet(RESULTS_PATH)
    
    # Apply spatial subset
    print(f"Applying spatial filter: Lat[{args.lat_min}:{args.lat_max}], Lon[{args.lon_min}:{args.lon_max}]")
    subset = df[
        (df['latitude'] >= args.lat_min) & (df['latitude'] <= args.lat_max) &
        (df['longitude'] >= args.lon_min) & (df['longitude'] <= args.lon_max)
    ].copy()

    if subset.empty:
        print("No pixels found in the specified spatial bounds.")
        return
    
    print(f"Subsetting {len(subset):,} pixels...")

    # Limit size for interactive map performance if needed
    if len(subset) > 50000:
        print("Warning: Subset is large. Downsampling to 50,000 pixels for map performance.")
        subset = subset.sample(50000)

    # Convert to GeoDataFrame
    geometry = [Point(xy) for xy in zip(subset.longitude, subset.latitude)]
    gdf = gpd.GeoDataFrame(subset, geometry=geometry, crs="EPSG:4324")

    # Save as GeoJSON for leafmap
    geo_path = os.path.join(BASE_DIR, "data", f"{args.output_name}.geojson")
    gdf.to_file(geo_path, driver='GeoJSON')
    print(f"Saved GeoJSON to {geo_path}")

    # Create Interactive Map using Leafmap
    print("Generating interactive map...")
    m = leafmap.Map(center=[(args.lat_min + args.lat_max)/2, (args.lon_min + args.lon_max)/2], zoom=10)
    
    # We'll use a nicer palette for clusters
    m.add_gdf(gdf, layer_name="Trajectories", 
              column='cluster',
              cmap='viridis',
              info_mode='on_click',
              fields=['pixel_id', 'cluster', 'sen_slope', 'mk_p_value', 'mean_gpp'])

    # Save map to HTML
    html_path = os.path.join(BASE_DIR, "plots", f"{args.output_name}_map.html")
    m.to_html(html_path)
    print(f"Map saved to {html_path}")
    print("You can open this file in any web browser to explore interactively.")

if __name__ == "__main__":
    main()
