import os
import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "validation")
OUT_FILE = os.path.join(OUT_DIR, "validation_metrics.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print(f"Loading results from {RESULTS_PATH}...")
    df = pd.read_parquet(RESULTS_PATH)
    
    # Exclude noise points (-1) for validation metrics (or at least separating them)
    # The specification says: "exclude -1 noise"
    df_valid = df[df['cluster'] != -1].copy()
    
    print(f"Valid points (excluding noise): {len(df_valid)}")
    
    if len(df_valid) > 100000:
        print("Subsampling to 100,000 points for computational feasibility of metric calculations...")
        df_valid = df_valid.sample(100000, random_state=42).reset_index(drop=True)
        
    # Reconstruct features
    feature_file = os.path.join(BASE_DIR, "data", "selected_features.txt")
    if os.path.exists(feature_file):
        with open(feature_file, "r") as f:
            feature_cols = f.read().splitlines()
    else:
        exclude_cols = ['pixel_id', 'eco_id', 'has_break', 'breakpoint', 'cluster', 'cluster_prob', 'latitude', 'longitude']
        feature_cols = [c for c in df_valid.columns if c not in exclude_cols]
    
    X = df_valid[feature_cols].fillna(0).values
    labels = df_valid['cluster'].values
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Re-computing UMAP (3D) embedding for validation metrics (matching classifier params)...")
    # Using same params as trajectory_classifier.py: n_components=3
    umap_model = umap.UMAP(n_neighbors=15, n_components=3, min_dist=0.1, random_state=42)
    embedding = umap_model.fit_transform(X_scaled)
    
    metrics = []

    # 1A. Silhouette Score
    print("Computing Silhouette Score...")
    global_sil = silhouette_score(embedding, labels, metric='euclidean', sample_size=min(50000, len(embedding)))

    # Compute per-cluster silhouette means
    # silhouette_samples is O(n^2) - cap to 10k to avoid memory/time blowup on large datasets
    SIL_CAP = 10000
    print(f"Computing per-cluster Silhouette Scores (capped at {SIL_CAP} points to avoid O(n^2) cost)...")
    if len(embedding) > SIL_CAP:
        sil_rng = np.random.RandomState(42)
        sil_idx = sil_rng.choice(len(embedding), SIL_CAP, replace=False)
        sil_embedding = embedding[sil_idx]
        sil_labels = labels[sil_idx]
    else:
        sil_idx = np.arange(len(embedding))
        sil_embedding = embedding
        sil_labels = labels

    sil_samples = silhouette_samples(sil_embedding, sil_labels, metric='euclidean')
    df_sil = df_valid.iloc[sil_idx].copy()
    df_sil['silhouette'] = sil_samples
    cluster_sil = df_sil.groupby('cluster')['silhouette'].mean().to_dict()
    
    metrics.append({
        'Metric': 'Global Silhouette Score',
        'Value': global_sil,
        'Interpretation': 'Higher is better (cohesion vs separation)',
        'Flag': 'Pass' if global_sil > 0.25 else 'Fail'
    })
    
    for c, s in cluster_sil.items():
        metrics.append({
            'Metric': f'Cluster {c} Mean Silhouette',
            'Value': s,
            'Interpretation': 'Mean cohesion for cluster',
            'Flag': 'Flag (Weak)' if s < 0.2 else 'Pass'
        })
        
    # 1B. Davies-Bouldin Index
    print("Computing Davies-Bouldin Index...")
    db_index = davies_bouldin_score(embedding, labels)
    db_interp = "<1.0 good, 1-2 moderate, >2 weak"
    db_flag = "Pass" if db_index < 1.5 else "Fail/Weak"
    metrics.append({
        'Metric': 'Davies-Bouldin Index',
        'Value': db_index,
        'Interpretation': db_interp,
        'Flag': db_flag
    })
    
    # 1C. Calinski-Harabasz Index
    print("Computing Calinski-Harabasz Index...")
    ch_score = calinski_harabasz_score(embedding, labels)
    
    # Random baseline
    np.random.seed(42)
    random_labels = np.random.permutation(labels)
    ch_score_random = calinski_harabasz_score(embedding, random_labels)
    
    metrics.append({
        'Metric': 'Calinski-Harabasz Index',
        'Value': ch_score,
        'Interpretation': 'Between vs Within dispersion (higher better)',
        'Flag': 'Pass' if ch_score > ch_score_random * 2 else 'Fail'
    })
    
    metrics.append({
        'Metric': 'Calinski-Harabasz Random Baseline',
        'Value': ch_score_random,
        'Interpretation': 'Expected score for random labels',
        'Flag': '-'
    })
    
    # Save to CSV
    res_df = pd.DataFrame(metrics)
    res_df.to_csv(OUT_FILE, index=False)
    print(f"\nValidation metrics saved to {OUT_FILE}")
    print(res_df.to_string())

if __name__ == "__main__":
    main()
