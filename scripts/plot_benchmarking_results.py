import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_hcas_results(csv_path, output_dir):
    print(f"Loading benchmarked data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if new columns exist
    if 'HCAS_score' not in df.columns:
        print("Error: HCAS_score column not found. Please run the benchmarking script first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Distribution of HCAS Scores
    plt.figure(figsize=(10, 6))
    sns.histplot(df['HCAS_score'], bins=30, kde=True, color='forestgreen', alpha=0.7)
    plt.axvline(df['HCAS_score'].mean(), color='red', linestyle='--', label=f"Mean: {df['HCAS_score'].mean():.3f}")
    plt.axvline(df['HCAS_score'].median(), color='blue', linestyle=':', label=f"Median: {df['HCAS_score'].median():.3f}")
    
    plt.title('Distribution of Multivariate HCAS Condition Scores', fontsize=15)
    plt.xlabel('HCAS Score (1.0 = Pristine, 0.0 = Degraded)', fontsize=12)
    plt.ylabel('Number of Transformed Sites', fontsize=12)
    plt.legend()
    plt.xlim(0, 1.05)
    
    plot_path_hcas = os.path.join(output_dir, 'hcas_distribution.png')
    plt.savefig(plot_path_hcas, dpi=300)
    print(f"Saved HCAS distribution to {plot_path_hcas}")

    # 2. Distribution of Composite Departures
    plt.figure(figsize=(10, 6))
    sns.histplot(df['comp_dep'], bins=30, kde=True, color='darkorange', alpha=0.7)
    plt.axvline(df['comp_dep'].mean(), color='red', linestyle='--', label=f"Mean: {df['comp_dep'].mean():.3f}")
    
    plt.title('Distribution of Composite Spectral Departures', fontsize=15)
    plt.xlabel('Manhattan Distance (Standardized Multi-Index Space)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    
    plot_path_dep = os.path.join(output_dir, 'composite_departure_distribution.png')
    plt.savefig(plot_path_dep, dpi=300)
    print(f"Saved Departure distribution to {plot_path_dep}")

    # 3. Relationship: Departure vs HCAS Score
    plt.figure(figsize=(10, 6))
    # Sample if dataset is too large for scatter
    df_sample = df.sample(min(5000, len(df)))
    sns.scatterplot(data=df_sample, x='comp_dep', y='HCAS_score', alpha=0.4, color='teal', edgecolor=None)
    
    plt.title('Relationship: Spectral Departure vs Calibrated HCAS Score', fontsize=15)
    plt.xlabel('Composite Departure (Manhattan Distance)', fontsize=12)
    plt.ylabel('HCAS Score', fontsize=12)
    
    plot_path_rel = os.path.join(output_dir, 'hcas_vs_departure.png')
    plt.savefig(plot_path_rel, dpi=300)
    print(f"Saved Relationship plot to {plot_path_rel}")

if __name__ == "__main__":
    BASE_DIR = r"C:\Users\coach\myfiles\postdoc2\code"
    INPUT_FILE = os.path.join(BASE_DIR, "data", "benchmarked_condition.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "plots", "benchmarking")
    
    if os.path.exists(INPUT_FILE):
        plot_hcas_results(INPUT_FILE, OUTPUT_DIR)
    else:
        print(f"Input file not found: {INPUT_FILE}")
