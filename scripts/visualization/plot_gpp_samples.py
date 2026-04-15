import os
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Constants
PARQUET_PATH = r'c:\Users\coach\myfiles\postdoc2\code\data\abandoned_ag_gpp_2000_2022_SA.parquet'
ECOREGION_CSV_PATH = r'c:\Users\coach\myfiles\postdoc2\code\data\ecoregion_sds.csv'
OUTPUT_PLOT_DIR = r'c:\Users\coach\myfiles\postdoc2\code\plots'
SAMPLES_PER_ECOREGION = 10

def main():
    if not os.path.exists(OUTPUT_PLOT_DIR):
        os.makedirs(OUTPUT_PLOT_DIR)

    print("Connecting to DuckDB and sampling data...")
    con = duckdb.connect(database=':memory:')
    
    # Get GPP column names
    gpp_cols = [f'GPP_{year}' for year in range(2000, 2023)]
    cols_to_select = ['eco_id'] + gpp_cols
    cols_str = ", ".join([f'"{c}"' for c in cols_to_select])
    
    # Subquery to sample rows per eco_id
    query = f"""
    SELECT {cols_str} FROM (
        SELECT {cols_str}, row_number() OVER (PARTITION BY eco_id ORDER BY random()) as rn
        FROM '{PARQUET_PATH}'
    ) WHERE rn <= {SAMPLES_PER_ECOREGION}
    """
    
    samples_df = con.execute(query).df()
    print(f"Sampled {len(samples_df)} rows across {samples_df['eco_id'].nunique()} ecoregions.")
    
    # Load ecoregion names
    print("Loading ecoregion metadata...")
    eco_meta = pd.read_csv(ECOREGION_CSV_PATH)
    eco_meta = eco_meta[['ECO_ID', 'ECO_NAME', 'BIOME_NAME']].drop_duplicates()
    
    # Merge names
    merged_df = samples_df.merge(eco_meta, left_on='eco_id', right_on='ECO_ID', how='left')
    
    # Assign Sample index to distinguish lines
    merged_df['SampleIdx'] = merged_df.groupby('ECO_NAME').cumcount()
    
    # Reshape for plotting: Wide to Long
    plot_df = merged_df.melt(
        id_vars=['ECO_NAME', 'BIOME_NAME', 'SampleIdx'],
        value_vars=gpp_cols,
        var_name='Year',
        value_name='GPP'
    )
    plot_df['Year'] = plot_df['Year'].str.replace('GPP_', '').astype(int)
    plot_df = plot_df.dropna(subset=['ECO_NAME', 'BIOME_NAME'])
    
    biomes = plot_df['BIOME_NAME'].unique()
    
    for biome in biomes:
        biome_data = plot_df[plot_df['BIOME_NAME'] == biome]
        ecos_in_biome = sorted(biome_data['ECO_NAME'].unique())
        
        n_ecos = len(ecos_in_biome)
        cols = 3
        rows = (n_ecos + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), squeeze=False)
        fig.suptitle(f'GPP Trajectories for {biome}', fontsize=20)
        axes_flat = axes.flatten()
        
        for i, eco in enumerate(ecos_in_biome):
            ax = axes_flat[i]
            eco_data = biome_data[biome_data['ECO_NAME'] == eco]
            
            sns.lineplot(
                data=eco_data, 
                x='Year', 
                y='GPP', 
                units='SampleIdx', 
                estimator=None, 
                ax=ax, 
                alpha=0.6,
                color='forestgreen'
            )
            ax.set_title(eco, fontsize=10)
            ax.set_ylabel('GPP')
            ax.set_xlabel('Year')
            ax.grid(True, alpha=0.3)
            
        # Hide empty axes
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        safe_biome_name = biome.replace(' ', '_').replace('/', '_').replace('&', 'and')
        output_path = os.path.join(OUTPUT_PLOT_DIR, f'gpp_samples_{safe_biome_name}.png')
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
        plt.close()

if __name__ == "__main__":
    main()
