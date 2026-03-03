import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt

try:
    from esda.moran import Moran
    from libpysal.weights import KNN
except ImportError:
    print("esda or libpysal not installed. Please install 'esda' and 'libpysal'.")
    exit(1)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "validation")
OUT_FILE = os.path.join(OUT_DIR, "spatial_coherence_results.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots", "validation")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def calculate_moran(k=5):
    if not os.path.exists(RESULTS_PATH):
        print("Results file not found.")
        return

    print("Loading coordinates and cluster labels...")
    con = duckdb.connect()
    # trajectory_results already contains latitude/longitude - no JOIN needed
    query = f"""
    SELECT pixel_id, latitude, longitude, cluster, cluster_prob
    FROM '{RESULTS_PATH}'
    WHERE cluster != -1
    """
    merged = con.execute(query).df()
    
    # Drop NaNs
    merged = merged.dropna(subset=['latitude', 'longitude', 'cluster'])
    
    if len(merged) > 50000:
        print(f"Subsampling 50k points from {len(merged)} for Moran's I (computational limit)...")
        merged = merged.sample(50000, random_state=42)
        
    print("Creating weights matrix (KNN)...")
    try:
        coords_array = merged[['longitude', 'latitude']].values
        w = KNN.from_array(coords_array, k=k)
        w.transform = 'R'
        
        # Calculate Moran's I on cluster probability and binary presence per cluster
        clusters = merged['cluster'].unique()
        
        out_metrics = []
        
        # 1. Cluster probability
        y = merged['cluster_prob'].values
        mi = Moran(y, w)
        print(f"\nVariate: cluster_prob")
        print(f"  Moran's I: {mi.I:.4f}, p-value: {mi.p_sim:.4f}")
        out_metrics.append({
            'Target': 'cluster_prob',
            'Morans_I': mi.I,
            'p_value': mi.p_sim,
            'Significant': mi.p_sim < 0.05
        })
        
        # 2. Binary variables for each cluster presence
        for c in clusters:
            val = (merged['cluster'] == c).astype(float).values
            mi_c = Moran(val, w)
            print(f"Variate: Ind(Cluster == {c})")
            print(f"  Moran's I: {mi_c.I:.4f}, p-value: {mi_c.p_sim:.4f}")
            out_metrics.append({
                'Target': f'Cluster_{c}_Presence',
                'Morans_I': mi_c.I,
                'p_value': mi_c.p_sim,
                'Significant': mi_c.p_sim < 0.05
            })
            
        res_df = pd.DataFrame(out_metrics)
        res_df.to_csv(OUT_FILE, index=False)
        print(f"Saved to {OUT_FILE}")
            
    except Exception as e:
        print(f"Error calculating Moran's I: {e}")
        
    # Spatial Maps (Scatter plot of sampled points)
    print("Generating spatial map of clusters...")
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(merged['longitude'], merged['latitude'], c=merged['cluster'], cmap='tab10', s=2, alpha=0.5)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Spatial Distribution of Trajectory Clusters (Sample)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    out_img = os.path.join(PLOT_DIR, "spatial_cluster_map.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved {out_img}")

if __name__ == "__main__":
    calculate_moran()
