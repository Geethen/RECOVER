import os
import pandas as pd
import numpy as np
import networkx as nx
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "data", "validation", "visualization_metrics.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
UNMERGED_PATH = os.path.join(BASE_DIR, "data", "trajectory_results_unmerged.parquet")

def main():
    if not os.path.exists(METRICS_PATH):
        print("No visualization metrics found. Run validate_visualization.py first.")
        return
        
    metrics = pd.read_csv(METRICS_PATH)
    
    # Filter for heavy overlap
    heavy = metrics[metrics['Flag'] == 'Flag (Heavy Overlap)']
    print(f"Found {len(heavy)} heavily overlapping cluster pairs.")
    
    # Create graph and find connected components
    G = nx.Graph()
    
    # Add all unique clusters from the pairs
    for _, row in heavy.iterrows():
        c1, c2 = row['Cluster_Pair'].split(' vs ')
        c1, c2 = int(c1), int(c2)
        G.add_edge(c1, c2)
        
    components = list(nx.connected_components(G))
    
    mapping = {}
    for comp in components:
        comp_list = sorted(list(comp))
        target_cluster = comp_list[0] # Merge into the lowest ID
        for c in comp_list:
            mapping[c] = target_cluster
            
    if not mapping:
        print("No overlaps detected > 30%. No merging needed.")
        return
        
    print("\nMerge Mapping:")
    for comp in components:
        comp_list = sorted(list(comp))
        print(f"  Clusters {comp_list} -> {comp_list[0]}")
        
    # Backup original before modifying
    if not os.path.exists(UNMERGED_PATH):
        shutil.copy2(RESULTS_PATH, UNMERGED_PATH)
        print(f"\nBacked up original results to {UNMERGED_PATH}")
        
    print("\nApplying remapping to results...")
    df = pd.read_parquet(RESULTS_PATH)
    original_count = df['cluster'].nunique()
    
    # Apply mapping. If cluster not in mapping, it keeps its original ID.
    df['cluster'] = df['cluster'].map(lambda x: mapping.get(x, x))
    
    final_count = df['cluster'].nunique()
    print(f"Cluster count reduced from {original_count} to {final_count} (including noise).")
    
    df.to_parquet(RESULTS_PATH)
    print("Saved merged results to trajectory_results.parquet.")

if __name__ == "__main__":
    main()
