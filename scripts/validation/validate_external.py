import os
import duckdb
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "validation")
OUT_FILE = os.path.join(OUT_DIR, "external_validation_results.csv")
PLOT_DIR = os.path.join(BASE_DIR, "plots", "validation")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def cramers_v(confusion_matrix):
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    if n == 0: return 0.0
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))

def validate_external():
    print("Validating External Environmental Signals (Land Cover 2022)...")
    
    con = duckdb.connect()
    
    query = f"""
    SELECT r.pixel_id, res.cluster, r.sanlc_2022
    FROM '{RESULTS_PATH}' res
    JOIN '{RAW_DATA_PATH}' r ON res.pixel_id = r.pixel_id
    WHERE res.cluster != -1 AND r.sanlc_2022 IS NOT NULL
    """
    
    try:
        df = con.execute(query).df()
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
        
    print(f"Loaded {len(df)} points for external validation.")
    
    if len(df) == 0:
        print("No valid external data found.")
        return
        
    # Contingency Table
    contingency = pd.crosstab(df['cluster'], df['sanlc_2022'])
    
    chi2, p_val, dof, expected = chi2_contingency(contingency.values)
    v = cramers_v(contingency)
    
    # Interpretation
    if v >= 0.25: effect = "Large"
    elif v >= 0.15: effect = "Medium"
    elif v >= 0.05: effect = "Small"
    else: effect = "Negligible"
    
    results = [{
        'Variable': 'SANLC_2022',
        'Test': 'Chi-Square',
        'Statistic': chi2,
        'p_value': p_val,
        'Effect_Size_CramersV': v,
        'Effect_Magnitude': effect,
        'Significant': p_val < 0.01
    }]
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_FILE, index=False)
    print(f"Saved to {OUT_FILE}")
    print(res_df.to_string())
    
    # Plotting
    print("Generating distribution plot...")
    plt.figure(figsize=(12, 6))
    
    # Calculate proportions
    props = contingency.div(contingency.sum(axis=1), axis=0)
    
    props.plot(kind='bar', stacked=True, colormap='tab20', ax=plt.gca())
    plt.title('SANLC 2022 Land Cover Distribution by Cluster')
    plt.ylabel('Proportion')
    plt.xlabel('Cluster')
    plt.legend(title='Land Cover Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    out_img = os.path.join(PLOT_DIR, "land_cover_distribution.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print(f"Saved {out_img}")

if __name__ == "__main__":
    validate_external()
