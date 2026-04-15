import duckdb
import pandas as pd
import numpy as np
try:
    from esda.moran import Moran
    from libpysal.weights import KNN
except ImportError:
    print("esda or libpysal not installed. Please install 'esda' and 'libpysal'.")
    # python -m pip install esda libpysal

import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")

def calculate_moran(k=5):
    if not os.path.exists(RESULTS_PATH):
        print("Results file not found.")
        return

    print("Loading results...")
    results = pd.read_parquet(RESULTS_PATH)
    
    print("Loading coordinates...")
    con = duckdb.connect()
    coords = con.execute(f"SELECT pixel_id, latitude, longitude FROM '{RAW_DATA_PATH}'").df()
    
    # Join
    merged = pd.merge(results, coords, on="pixel_id", how="inner")
    
    # Drop NaNs
    merged = merged.dropna(subset=['latitude', 'longitude', 'cluster'])
    
    if len(merged) > 50000:
        print(f"Subsampling 50k points from {len(merged)} for Moran's I (computational limit)...")
        merged = merged.sample(50000, random_state=42)
        
    print("Creating weights matrix (KNN)...")
    try:
        w = KNN.from_dataframe(merged, k=k, geom_col=None, ids='pixel_id') # geom_col None if lat/lon cols exist? 
        # KNN.from_dataframe often expects geometry column (geopandas).
        # Use simple coordinate array if not geopandas.
        
        coords_array = merged[['longitude', 'latitude']].values
        w = KNN.from_array(coords_array, k=k)
        w.transform = 'R'
        
        # Calculate Moran's I on Cluster ID (treating as ordinal/nominal? Risky)
        # Better to calculate on features or probability of "good" cluster.
        # Let's calculate on cluster_prob and maybe GPP_slope1 (if available)
        
        target_cols = ['cluster', 'cluster_prob']
        if 'GPP_slope1' in merged.columns:
            target_cols.append('GPP_slope1')
            
        for col in target_cols:
            y = merged[col].values
            mi = Moran(y, w)
            print(f"\nVariate: {col}")
            print(f"  Moran's I: {mi.I:.4f}")
            print(f"  p-value: {mi.p_sim:.4f}")
            
    except Exception as e:
        print(f"Error calculating Moran's I: {e}")

if __name__ == "__main__":
    calculate_moran()
