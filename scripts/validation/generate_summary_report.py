import os
import pandas as pd

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VAL_DIR = os.path.join(BASE_DIR, "data", "validation")
REPORT_PATH = os.path.join(VAL_DIR, "summary_report.md")

def load_csv_if_exists(filename):
    path = os.path.join(VAL_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def main():
    print("Generating Validation Summary Report...")
    
    internal_df = load_csv_if_exists("validation_metrics.csv")
    stability_df = load_csv_if_exists("stability_results.csv")
    eco_df = load_csv_if_exists("feature_separation_stats.csv")
    spatial_df = load_csv_if_exists("spatial_coherence_results.csv")
    external_df = load_csv_if_exists("external_validation_results.csv")
    vis_df = load_csv_if_exists("visualization_metrics.csv")
    
    lines = [
        "# Ecological Trajectory Clustering: Validation Report\n",
        "## 1. Internal Cluster Validity (Structure)",
        "Goal: Measure cohesion and separation of clusters in embedding space.\n"
    ]
    
    if internal_df is not None:
        lines.append(internal_df.to_markdown(index=False))
        lines.append("\n**Interpretation:** \n- *Silhouette Score* measures if clusters are distinct. Scores < 0.25 indicate that the lower dimensional embedding (UMAP space) is crowded.\n- *Davies-Bouldin* < 1.0 means clusters are well separated. \n- *Suggestion*: If internal validity is low, consider adjusting `min_dist` in UMAP during classification to enforce tighter grouping, or increasing `min_cluster_size` in HDBSCAN to reduce fragmented clusters.\n")
        
    lines.extend([
        "## 2. Stability Validation",
        "Goal: Test robustness to data perturbation (subsampling 40% and refitting).\n"
    ])
    
    if stability_df is not None:
        lines.append(stability_df.to_markdown(index=False))
        lines.append("\n**Interpretation:** \n- *Adjusted Rand Index (ARI)* compares the new clustering on subsampled data to the original labels. ARI > 0.6 is strong.\n- *Suggestion*: If ARI is low (<0.4), it means the clusters are highly sensitive to the specific pixels included. To improve stability, increase the `min_cluster_size` in HDBSCAN. Currently, it is dynamically set based on input size. Hardcoding a higher floor (e.g., `min_cluster_size=1000`) will force the algorithm to only identify the most prominent, stable ecological trajectories instead of fracturing into many small ones.\n")
        
    lines.extend([
        "## 3. Ecological Interpretability",
        "Goal: Confirm clusters differ meaningfully in physical and dynamical metrics.\n"
    ])
    
    sig_eco = 0  # Initialize before conditional block
    if eco_df is not None:
        sig_eco = len(eco_df[eco_df['Significant'] & eco_df['Effect_Magnitude'].isin(['Medium', 'Large'])])
        lines.append(f"**Significant Features (Medium/Large Effect Size):** {sig_eco}\n")
        lines.append(eco_df.to_markdown(index=False))
        lines.append("\n**Interpretation:** \n- This shows which engineered features (slopes, CVs, Catch22) actually drive the separation between the clusters.\n- *Suggestion*: If less than 5 features show medium/large effect sizes, the clustering might be splitting hairs. Review the feature set and consider dropping highly collinear metrics before UMAP reduction.\n")
        
    lines.extend([
        "## 4. Spatial Coherence",
        "Goal: Ensure clusters are not spatially random (Moran's I).\n"
    ])
    
    if spatial_df is not None:
        lines.append(spatial_df.to_markdown(index=False))
        lines.append("\n**Interpretation:** \n- A significant Moran's I means pixels of the same trajectory type tend to cluster together geographically, which is expected for natural land cover processes.\n- *Suggestion*: If clusters are randomly distributed (high p-value), they might represent arbitrary noise rather than real agricultural abandonment patterns.\n")
        
    lines.extend([
        "## 5. External Validation",
        "Goal: Ensure ecological grounding using independent environmental signals (SANLC 2022).\n"
    ])
    
    if external_df is not None:
        lines.append(external_df.to_markdown(index=False))
        lines.append("\n**Interpretation:** \n- This tests whether the generated trajectory classes map onto known 2022 land cover states in South Africa.\n- *Suggestion*: Use the generated bar chart in `plots/validation/` to manually assign human-readable labels to the numeric clusters (e.g., Cluster 0 = Secondary Grassland).\n")
        
    lines.extend([
        "## 6. Reduced-Space Separability Visualization",
        "Goal: Confirm cluster separation (convex hull overlap) in 2D embedding space.\n"
    ])
    
    if vis_df is not None and len(vis_df) > 0:
        lines.append(vis_df.to_markdown(index=False))
        lines.append("\n**Interpretation:** \n- High overlap percentages indicate that clusters bleed into one another in 2D space.\n- *Suggestion*: Combine heavily overlapping clusters post-hoc, or increase `min_cluster_size` in HDBSCAN.\n")
    else:
        lines.append("No overlap metrics to display (could mean no clusters overlapped heavily or Shapely was missing).\n")
        
    # Get current cluster count and pixel breakdown
    try:
        df_res = pd.read_parquet(os.path.join(BASE_DIR, "data", "trajectory_results.parquet"))
        total_pix = len(df_res)
        noise_pix = int((df_res['cluster'] == -1).sum())
        core_pix = total_pix - noise_pix
        current_clusters = len(df_res[df_res['cluster'] != -1]['cluster'].unique())
    except:
        current_clusters = "Unknown"
        total_pix = "Unknown"
        noise_pix = "Unknown"
        core_pix = "Unknown"

    lines.extend([
        "## 7. Cluster Count and Pixel Summary",
        f"**Total Pixels Processed:** {total_pix}",
        f"**Core Cluster Pixels (cluster != -1):** {core_pix}",
        f"**Noise Pixels (cluster == -1):** {noise_pix}",
        f"**Number of Core Clusters:** {current_clusters}",
        "\n**Observation:** The number of clusters has been reduced through a post-hoc merging process.",
        "**Interpretation:** This reduction is beneficial for ecological interpretability, ensuring each cluster represents a distinct structural regime.",
        "1. **Geographical Variance:** The full dataset naturally contains more diversity than the initial spatial subset.",
        "2. **Post-Hoc Refinement:** Clusters with >30% geometric overlap in the 2D embedding space have been merged into single entities to ensure statistical and physical separability.",
        "3. **Refinement Suggestion:** If the current count is still too high, increase the `min_cluster_size` in `trajectory_classifier.py` and re-run the classification, or increase the overlap threshold in the merging script.",
        ""
    ])
        
    lines.extend([
        "---",
        "## Final Evaluation Status",
        "Clusters are considered robust if all criteria pass:\n"
    ])
    
    wandb_run_id_path = os.path.join(VAL_DIR, "wandb_run_id.txt")
    run = None
    if HAS_WANDB and os.path.exists(wandb_run_id_path):
        with open(wandb_run_id_path, "r") as f:
            run_id = f.read().strip()
        try:
            os.environ["WANDB_ANONYMOUS"] = "allow"
            os.environ["WANDB_SILENT"] = "true"
            run = wandb.init(project="eco-trajectory", id=run_id, resume="must")
        except Exception as e:
            print(f"Failed to resume wandb run: {e}")
    
    # Check criteria
    sil_pass = False
    if internal_df is not None:
        global_sil = internal_df[internal_df['Metric'] == 'Global Silhouette Score']['Value'].values
        if len(global_sil) > 0 and global_sil[0] > 0.25:
            sil_pass = True
            
    db_pass = False
    if internal_df is not None:
        db = internal_df[internal_df['Metric'] == 'Davies-Bouldin Index']['Value'].values
        if len(db) > 0 and db[0] < 1.5:
            db_pass = True
            
    ari_pass = False
    if stability_df is not None:
        ari = stability_df[stability_df['Metric'] == 'Mean ARI']['Value'].values
        if len(ari) > 0 and ari[0] > 0.6:
            ari_pass = True
            
    eco_pass = False
    if eco_df is not None:
        if sig_eco >= 5:
            eco_pass = True
            
    spatial_pass = False
    if spatial_df is not None:
        if (spatial_df['Significant'] == True).all():
            spatial_pass = True
            
    ext_pass = False
    if external_df is not None:
        if (external_df['Significant'] == True).all():
            ext_pass = True
            
    lines.append(f"- **Silhouette > 0.25:** {'✅ PASS' if sil_pass else '❌ FAIL'}")
    lines.append(f"- **DB Index < 1.5:** {'✅ PASS' if db_pass else '❌ FAIL'}")
    lines.append(f"- **ARI Stability > 0.6:** {'✅ PASS' if ari_pass else '❌ FAIL'}")
    lines.append(f"- **≥ 5 features sig. differentiate:** {'✅ PASS' if eco_pass else '❌ FAIL'}")
    lines.append(f"- **Moran's I Significant:** {'✅ PASS' if spatial_pass else '❌ FAIL'}")
    lines.append(f"- **Environmental Gradients Significant:** {'✅ PASS' if ext_pass else '❌ FAIL'}")
    
    lines.append("\n**Visual Deliverables Located in `plots/validation/`:**")
    lines.append("- `trajectory_separation_GPP.png`")
    lines.append("- `trajectory_separation_SVH.png`")
    lines.append("- `spatial_cluster_map.png`")
    lines.append("- `land_cover_distribution.png`")
    lines.append("- `umap_2d_clusters.png`")
    
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
        
    if run is not None:
        metrics_to_log = {}
        if internal_df is not None:
            g_sil = internal_df[internal_df['Metric'] == 'Global Silhouette Score']['Value'].values
            db = internal_df[internal_df['Metric'] == 'Davies-Bouldin Index']['Value'].values
            if len(g_sil) > 0: metrics_to_log["validation/global_silhouette"] = g_sil[0]
            if len(db) > 0: metrics_to_log["validation/davies_bouldin"] = db[0]
            
        if stability_df is not None:
            ari = stability_df[stability_df['Metric'] == 'Mean ARI']['Value'].values
            if len(ari) > 0: metrics_to_log["validation/stability_ari"] = ari[0]
            
        if eco_df is not None:
            val = len(eco_df[eco_df['Significant'] & eco_df['Effect_Magnitude'].isin(['Medium', 'Large'])])
            metrics_to_log["validation/significant_features_count"] = val
            
        if spatial_df is not None:
            prob_moran = spatial_df[spatial_df['Target'] == 'cluster_prob']['Morans_I'].values
            if len(prob_moran) > 0: metrics_to_log["validation/spatial_morans_i"] = prob_moran[0]
            
        if current_clusters != "Unknown":
            metrics_to_log["results/final_cluster_count"] = current_clusters
            
        if metrics_to_log:
            wandb.log(metrics_to_log)
        try:
            wandb.finish()
        except:
            pass
            
    print(f"Summary Report generated at: {REPORT_PATH}")

if __name__ == "__main__":
    main()
