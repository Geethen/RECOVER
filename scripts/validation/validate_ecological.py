import os
import pandas as pd
import duckdb
from scipy.stats import kruskal
import matplotlib.pyplot as plt

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "validation")
OUT_FILE = os.path.join(OUT_DIR, "feature_separation_stats.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots", "validation")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
YEARS = list(range(2000, 2023))

# Epsilon-squared effect size for Kruskal-Wallis (Tomczak & Tomczak 2014)
# ε²_H = (H - k + 1) / (n - k), where k = number of groups
def epsilon_squared(H, n, k):
    if n == k:
        return 0.0
    return (H - k + 1) / (n - k)

def validate_features(df):
    print("Validating Feature Differentiation...")
    exclude_cols = ['pixel_id', 'eco_id', 'has_break', 'breakpoint', 'cluster', 'cluster_prob', 'latitude', 'longitude']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Exclude noise (-1) from feature distribution comparison between core clusters
    df_core = df[df['cluster'] != -1]
    clusters = df_core['cluster'].unique()
    
    if len(clusters) < 2:
        print("Not enough core clusters for feature differentiation tests.")
        return
        
    results = []
    
    for feature in feature_cols:
        # Get groups for this feature
        groups = [df_core[df_core['cluster'] == c][feature].dropna().values for c in clusters]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) < 2: continue
        
        # Kruskal-Wallis due to non-normality
        try:
            stat, p_val = kruskal(*groups)

            # Combine all group sizes; pass k (number of groups) for correct ε² formula
            n = sum(len(g) for g in groups)
            k = len(groups)
            e_sq = epsilon_squared(stat, n, k)

            # Interpretation thresholds from Tomczak & Tomczak (2014)
            if e_sq >= 0.26: effect = "Large"
            elif e_sq >= 0.08: effect = "Medium"
            else: effect = "Small"

            results.append({
                'Feature': feature,
                'H_statistic': stat,
                'p_value': p_val,
                'Effect_Size_epsilon2': e_sq,
                'Effect_Magnitude': effect,
                'Significant': p_val < 0.01
            })
        except Exception as e:
            print(f"Error computing KW for {feature}: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_FILE, index=False)
    
    sig_count = sum((res_df['Significant']) & (res_df['Effect_Magnitude'].isin(["Medium", "Large"])))
    print(f"\nFeature validation saved to {OUT_FILE}.")
    print(f"{sig_count} features showed significant differences with Medium/Large effect size.")
    
def validate_trajectory_separation():
    print("Validating Trajectory Separation (Medians)...")
    
    con = duckdb.connect()
    
    # We will compute the median GPP and SVH trajectory by loading a subset of data or looping
    print("Querying temporal medians per cluster on a 100k sample to prevent memory issues...")
    
    # Sample 100k pixels from the results
    res_sample = con.execute(f"SELECT pixel_id, cluster FROM '{RESULTS_PATH}' WHERE cluster != -1 USING SAMPLE 100000").df()
    
    # Load the raw data for these pixels using a temp table join (avoids huge IN-clause)
    gpp_cols = [f"GPP_{y}" for y in YEARS]
    svh_cols = [f"SVH_{y}" for y in YEARS]
    
    # Register the pandas DataFrame so DuckDB can reference it in SQL
    con.register('res_sample', res_sample)
    con.execute("CREATE OR REPLACE TEMP TABLE sample_pids AS SELECT pixel_id FROM res_sample")
    
    query = f"""
    SELECT r.pixel_id, {', '.join(gpp_cols + svh_cols)}
    FROM '{RAW_DATA_PATH}' r
    INNER JOIN sample_pids sp ON r.pixel_id = sp.pixel_id
    """
    try:
        raw_sample = con.execute(query).df()
    except Exception as e:
        print(f"Aggregation failed: {e}")
        return
        
    df = res_sample.merge(raw_sample, on='pixel_id')
    print(f"Merged {len(df)} sampled pixels for trajectory plots across {df['cluster'].nunique()} clusters...")
    
    for var_prefix, title in [('GPP', 'Gross Primary Productivity'), ('SVH', 'Short Vegetation Height')]:
        plt.figure(figsize=(10, 6))
        for c in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == c][[f"{var_prefix}_{y}" for y in YEARS]]
            meds = cluster_data.median()
            q25 = cluster_data.quantile(0.25)
            q75 = cluster_data.quantile(0.75)
            
            line, = plt.plot(YEARS, meds, label=f"Cluster {c} (n_sample={len(cluster_data)})", linewidth=2)
            plt.fill_between(YEARS, q25, q75, alpha=0.15, color=line.get_color())
            
        plt.title(f"{title} Trajectories by Cluster (Median ± IQR)")
        plt.xlabel("Year")
        plt.ylabel(title)
        plt.legend()
        plt.tight_layout()
        out_plot = os.path.join(PLOT_DIR, f"trajectory_separation_{var_prefix}.png")
        plt.savefig(out_plot, dpi=300)
        plt.close()
        print(f"Saved {out_plot}")


def main():
    print(f"Loading results sample from {RESULTS_PATH} for feature validation...")
    con = duckdb.connect()
    # Use a 100k stratified sample to avoid loading the full parquet into RAM
    df = con.execute(f"SELECT * FROM '{RESULTS_PATH}' USING SAMPLE 100000").df()
    con.close()

    validate_features(df)
    validate_trajectory_separation()

if __name__ == "__main__":
    main()
