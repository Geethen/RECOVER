import os
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
try:
    from shapely.geometry import Polygon
    has_shapely = True
except ImportError:
    has_shapely = False
    print("Shapely not found, overlap calculations will be approximate or skipped.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "validation")
OUT_FILE = os.path.join(OUT_DIR, "visualization_metrics.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots", "validation")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    print("Validating Reduced-Space Separability (UMAP 2D)...")
    
    print(f"Loading results from {RESULTS_PATH}...")
    df = pd.read_parquet(RESULTS_PATH)
    
    # Subsample for visualization and fast UMAP calculation
    if len(df) > 50000:
        print("Subsampling to 50k points for UMAP 2D visualization...")
        df = df.sample(50000, random_state=42)
        
    feature_file = os.path.join(BASE_DIR, "data", "selected_features.txt")
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            feature_cols = [c for c in f.read().splitlines() if c in df.columns]
        if not feature_cols:
            print("selected_features.txt has no columns matching the results parquet. Falling back to auto-detect.")
    if not os.path.exists(feature_file) or not feature_cols:
        exclude_cols = ['pixel_id', 'eco_id', 'has_break', 'breakpoint', 'cluster', 'cluster_prob', 'latitude', 'longitude']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Feature matrix
    X = df[feature_cols].fillna(0).values
    labels = df['cluster'].values
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Computing UMAP 2D... (This will take a moment)")
    umap_model = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.1, random_state=42)
    embedding = umap_model.fit_transform(X_scaled)
    
    # Plotting
    plt.figure(figsize=(10, 8))
    
    # Plot noise first so it's on bottom
    noise_mask = (labels == -1)
    if noise_mask.sum() > 0:
        plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1], c='gray', s=1, alpha=0.3, label='Noise (-1)')
        
    clusters = np.unique(labels[~noise_mask])
    polygons = {}
    
    for c in clusters:
        c_mask = (labels == c)
        c_pts = embedding[c_mask]
        
        plt.scatter(c_pts[:, 0], c_pts[:, 1], s=2, alpha=0.7, label=f'Cluster {c}')
        
        # Centroid
        centroid = np.mean(c_pts, axis=0)
        plt.scatter(centroid[0], centroid[1], c='black', marker='x', s=100)
        
        if len(c_pts) >= 3:
            hull = ConvexHull(c_pts)
            hull_pts = c_pts[hull.vertices]
            # Close the polygon for plotting
            hull_pts_closed = np.vstack((hull_pts, hull_pts[0]))
            plt.plot(hull_pts_closed[:, 0], hull_pts_closed[:, 1], 'k--', lw=1)
            
            if has_shapely:
                polygons[c] = Polygon(hull_pts)

    plt.title('UMAP 2D Projection of Trajectory Clusters')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out_img = os.path.join(PLOT_DIR, "umap_2d_clusters.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved {out_img}")

    # Compute Overlaps
    metrics = []
    if has_shapely and len(polygons) > 1:
        for i, c1 in enumerate(clusters):
            p1 = polygons.get(c1)
            # Use 'is None' to avoid ValueError from Shapely 2.x boolean geometry evaluation
            if p1 is None or not p1.is_valid: continue
            area1 = p1.area

            for c2 in clusters[i+1:]:
                p2 = polygons.get(c2)
                if p2 is None or not p2.is_valid: continue
                area2 = p2.area
                
                try:
                    intersection = p1.intersection(p2).area
                    overlap_pct1 = (intersection / area1) * 100
                    overlap_pct2 = (intersection / area2) * 100
                    max_overlap = max(overlap_pct1, overlap_pct2)
                    
                    flag = "Flag (Heavy Overlap)" if max_overlap > 30 else "Pass"
                    
                    metrics.append({
                        'Cluster_Pair': f"{c1} vs {c2}",
                        'Intersection_Area': intersection,
                        'Overlap_Pct': max_overlap,
                        'Flag': flag
                    })
                except:
                    pass
    
    if metrics:
        res_df = pd.DataFrame(metrics)
        res_df.to_csv(OUT_FILE, index=False)
        print(f"Saved overlap metrics to {OUT_FILE}")
        print(res_df.to_string())
    else:
        print("No overlap metrics computed (possibly due to missing polygons or Shapely library).")

if __name__ == "__main__":
    main()
