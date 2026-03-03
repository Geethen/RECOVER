import gc
import os
import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "validation")
OUT_FILE = os.path.join(OUT_DIR, "stability_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print(f"Loading results from {RESULTS_PATH}...")
    df = pd.read_parquet(RESULTS_PATH)
    
    # Exclude noise points (-1) for stability metrics? 
    # Actually, the original clustering handled them. Replacing -1 with a separate class is fine,
    # or just filtering them out. We will keep them for the subsample but filter out for metrics.
    # To save time, we take a 100k sample as our "population" 
    if len(df) > 100000:
        print("Subsampling to 100,000 points for computational feasibility of stability iterations...")
        df = df.sample(100000, random_state=42)
        
    feature_file = os.path.join(BASE_DIR, "data", "selected_features.txt")
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            feature_cols = f.read().splitlines()
    else:
        exclude_cols = ['pixel_id', 'eco_id', 'has_break', 'breakpoint', 'cluster', 'cluster_prob', 'latitude', 'longitude']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Scale here because the whole population is scaled.
    scaler = StandardScaler()
    X_full_scaled = scaler.fit_transform(df[feature_cols].fillna(0).values)
    df_labels = df['cluster'].values
    
    n_iterations = 5
    subsample_frac = 0.4

    aris = []
    nmis = []

    for i in range(n_iterations):
        print(f"Iteration {i+1}/{n_iterations}...")

        # Subsample indices (use a different seed per iteration for varied subsamples)
        rng = np.random.RandomState(i)
        n_samples = int(len(df) * subsample_frac)
        idx = rng.choice(len(df), n_samples, replace=False)

        X_sub = X_full_scaled[idx]
        orig_labels_sub = df_labels[idx]

        # Create a fresh UMAP instance per iteration (fixed seed - stability comes from data, not randomness)
        umap_iter = umap.UMAP(n_neighbors=15, n_components=3, min_dist=0.1, random_state=42)
        emb_sub = umap_iter.fit_transform(X_sub)

        # Scale HDBSCAN params proportionally to the subsample size
        # Using 1% of subsample size (min 50) to match spirit of main classifier
        mc_sub = max(50, int(n_samples * 0.01))
        ms_sub = max(5, int(n_samples * 0.005))
        print(f"  HDBSCAN params for subsample: min_cluster_size={mc_sub}, min_samples={ms_sub}")
        hdb = hdbscan.HDBSCAN(min_cluster_size=mc_sub, min_samples=ms_sub)
        new_labels_sub = hdb.fit_predict(emb_sub)

        # Evaluate only on points assigned to clusters in both original and new labelling
        mask = (orig_labels_sub != -1) & (new_labels_sub != -1)

        if mask.sum() > 0:
            ari = adjusted_rand_score(orig_labels_sub[mask], new_labels_sub[mask])
            nmi = normalized_mutual_info_score(orig_labels_sub[mask], new_labels_sub[mask])
            aris.append(ari)
            nmis.append(nmi)
            print(f"  ARI: {ari:.3f}, NMI: {nmi:.3f}")
        else:
            print("  No valid non-noise clusters matched.")

        # Explicitly release large arrays before the next iteration
        del emb_sub, new_labels_sub, hdb, umap_iter, X_sub
        gc.collect()
            
    mean_ari = np.mean(aris) if aris else 0.0
    std_ari = np.std(aris) if aris else 0.0
    mean_nmi = np.mean(nmis) if nmis else 0.0
    
    if mean_ari > 0.7:
        stability_interp = "Strong"
    elif mean_ari >= 0.4:
        stability_interp = "Moderate"
    else:
        stability_interp = "Unstable"
        
    metrics = [
        {'Metric': 'Mean ARI', 'Value': mean_ari, 'Interpretation': 'Adjusted Rand Index across subsamples', 'Flag': stability_interp},
        {'Metric': 'Std ARI', 'Value': std_ari, 'Interpretation': 'Variance in ARI', 'Flag': '-'},
        {'Metric': 'Mean NMI', 'Value': mean_nmi, 'Interpretation': 'Normalized Mutual Information', 'Flag': '-'}
    ]
    
    res_df = pd.DataFrame(metrics)
    res_df.to_csv(OUT_FILE, index=False)
    print(f"\nStability metrics saved to {OUT_FILE}")
    print(res_df.to_string())

if __name__ == "__main__":
    main()
