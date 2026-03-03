import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "clusters")
os.makedirs(OUTPUT_DIR, exist_ok=True)
YEARS = list(range(2000, 2023))

def load_data(pixel_id):
    con = duckdb.connect()
    # Load raw data
    gpp_cols = [f"GPP_{y}" for y in YEARS]
    svh_cols = [f"SVH_{y}" for y in YEARS]
    cols = ",".join(gpp_cols + svh_cols)
    query = f"SELECT {cols} FROM '{RAW_DATA_PATH}' WHERE pixel_id = '{pixel_id}' LIMIT 1"
    df = con.execute(query).df()
    if df.empty: return None
    
    # Melt
    gpp = df[gpp_cols].values.flatten()
    svh = df[svh_cols].values.flatten()
    return gpp, svh

def plot_cluster_examples(n_examples=5):
    if not os.path.exists(RESULTS_PATH):
        print("Results file not found.")
        return

    # Load results
    results = pd.read_parquet(RESULTS_PATH)
    clusters = sorted(results['cluster'].unique())
    
    for cluster in clusters:
        print(f"Plotting Cluster {cluster}...")
        # Select random examples
        subset = results[results['cluster'] == cluster]
        if len(subset) > n_examples:
            examples = subset.sample(n_examples, random_state=42)
        else:
            examples = subset
            
        fig, axes = plt.subplots(n_examples, 1, figsize=(10, 3*n_examples), sharex=True)
        if n_examples == 1: axes = [axes]
        
        for i, (_, row) in enumerate(examples.iterrows()):
            pixel_id = row['pixel_id']
            # Load raw data
            data = load_data(pixel_id)
            if data:
                gpp, svh = data
                ax = axes[i]
                
                # Plot GPP
                color = 'tab:green'
                ax.set_ylabel('GPP', color=color)
                ax.plot(YEARS, gpp, color=color, label='GPP')
                ax.tick_params(axis='y', labelcolor=color)
                
                # Plot SVH
                ax2 = ax.twinx()
                color = 'tab:brown'
                ax2.set_ylabel('SVH', color=color)
                ax2.plot(YEARS, svh, color=color, linestyle='--', label='SVH')
                ax2.tick_params(axis='y', labelcolor=color)
                
                # Add title info
                title = f"Pixel {pixel_id} | Breakpoint: {row['breakpoint'] if row['has_break'] else 'None'}"
                # Add slope info if available in columns
                if 'GPP_slope1' in row:
                     title += f" | GPP Slope1: {row['GPP_slope1']:.2f}"
                ax.set_title(title, fontsize=10)
                
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"cluster_{cluster}_examples.png"))
        plt.close()
        print(f"Saved plot for Cluster {cluster}")

if __name__ == "__main__":
    plot_cluster_examples()
