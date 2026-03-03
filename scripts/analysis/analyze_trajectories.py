import duckdb
import pandas as pd
import numpy as np
import os
import sys
import argparse

# Try imports
try:
    from scipy import stats
except ImportError:
    print("Scipy not found. Please install scipy.")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Clustering will be skipped.")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "trajectory_analysis_results.parquet")
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]

def fast_vectorized_process(y_chunk):
    """
    Compute Theil-Sen slope and Mann-Kendall Z-score for a chunk of pixels (B, T)
    using vectorized broadcasting.
    """
    B, T = y_chunk.shape
    
    # generate pairs indices (i, j) where i < j
    i, j = np.triu_indices(T, k=1)
    dx = (j - i).astype(float)
    
    # Compute differences (B, Pairs)
    diffs = y_chunk[:, j] - y_chunk[:, i]
    
    # 1. Theil-Sen Slope: Median of slopes
    slopes = diffs / dx[None, :]
    sen_slopes = np.nanmedian(slopes, axis=1)
    
    # 2. Mann-Kendall Statistic S: Sum of signs of differences
    signs = np.sign(diffs)
    S = np.nansum(signs, axis=1)
    
    # Variance of S (ignoring ties for speed/simplicity given float data)
    # Var(S) = n(n-1)(2n+5)/18
    # Adjust for NaNs per row? That's hard vectorised.
    # Assuming mostly full data:
    n = np.sum(~np.isnan(y_chunk), axis=1) # Count valid per row
    
    # Filter where enough data
    valid_mask = n >= 5
    
    # Formula for variance with no ties
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    
    # Z-score
    # if S > 0: Z = (S - 1)/sqrt(VarS)
    # if S < 0: Z = (S + 1)/sqrt(VarS)
    # if S == 0: Z = 0
    sigma = np.sqrt(var_s)
    Z = np.zeros(B)
    
    pos = (S > 0) & valid_mask
    neg = (S < 0) & valid_mask
    
    Z[pos] = (S[pos] - 1) / sigma[pos]
    Z[neg] = (S[neg] + 1) / sigma[neg]
    
    # P-value (Two-tailed)
    # 2 * (1 - norm.cdf(|Z|))
    # using stats.norm.sf -> 2 * sf(abs(Z))
    try:
        p_values = 2 * stats.norm.sf(np.abs(Z))
    except:
        # manual sf approximation if needed, but scipy.stats is imported
        p_values = np.zeros(B) 
        pass 
        
    # Basic Stats
    means = np.nanmean(y_chunk, axis=1)
    stds = np.nanstd(y_chunk, axis=1)
    cvs = stds / (means + 1e-6)
    
    # Return DataFrame part
    return pd.DataFrame({
        'sen_slope': sen_slopes,
        'mk_z': Z,
        'mk_p_value': p_values,
        'mean_gpp': means,
        'cv_gpp': cvs
    })

def vectorized_analysis(df):
    """
    Process dataframe using chunked vectorization.
    """
    print(f"  Starting vectorized analysis on {len(df)} pixels...")
    
    gpp_data = df[GPP_COLS].values
    chunk_size = 100000 
    results_list = []
    
    # Simple chunk loop
    for k in range(0, len(gpp_data), chunk_size):
        chunk = gpp_data[k : k + chunk_size]
        res = fast_vectorized_process(chunk)
        results_list.append(res)
        print(f"    Processed {k + len(chunk)} / {len(gpp_data)}...")
        
    final_res = pd.concat(results_list, ignore_index=True)
    final_res.index = df.index  # Align index
    
    # Add Lag-1 Autocorrelation
    y_t = gpp_data[:, 1:]
    y_tm1 = gpp_data[:, :-1]
    
    y_t_dev = y_t - np.nanmean(y_t, axis=1, keepdims=True)
    y_tm1_dev = y_tm1 - np.nanmean(y_tm1, axis=1, keepdims=True)
    
    numer_acr = np.nansum(y_t_dev * y_tm1_dev, axis=1)
    denom_acr = np.sqrt(np.nansum(y_t_dev**2, axis=1) * np.nansum(y_tm1_dev**2, axis=1))
    
    autocorr = numer_acr / (denom_acr + 1e-9)
    final_res['autocorr_lag1'] = autocorr
    
    return final_res



def main():
    parser = argparse.ArgumentParser(description="Analyze GPP Trajectories")
    parser.add_argument("--sample_size", type=int, default=100000, help="Number of pixels to process (-1 for all)")
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return

    print(f"Connecting to DuckDB to read: {DATA_PATH}")
    con = duckdb.connect()
    
    # Get total count
    try:
        count = con.execute(f"SELECT COUNT(*) FROM '{DATA_PATH}'").fetchone()[0]
        print(f"Total pixels in dataset: {count:,}")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Columns to select
    cols = ['pixel_id', 'latitude', 'longitude'] + GPP_COLS
    col_str = ", ".join(cols)
    
    if args.sample_size > 0:
        print(f"Processing a sample of {args.sample_size:,} pixels...")
        query = f"SELECT {col_str} FROM '{DATA_PATH}' LIMIT {args.sample_size}"
    else:
        print("Processing ALL pixels (this may take a while)...")
        query = f"SELECT {col_str} FROM '{DATA_PATH}'"
    
    # Fetch data (using Pandas for the vectorization step)
    # Note: duckdb.query(q).df() loads result into memory.
    # For full dataset (33M rows), this might exceed RAM.
    # If so, we should iterate chunks from DuckDB cursor.
    
    if args.sample_size > 0 or count < 1000000:
        # Load all at once for small/medium
        print("Loading data into memory...")
        try:
            df = con.query(query).df()
        except Exception as e:
            print(f"Error loading data: {e}")
            return
            
        print(f"Loaded {len(df)} rows. Starting parallel analysis...")
        results = vectorized_analysis(df)
        final_df = pd.concat([df[['pixel_id', 'latitude', 'longitude']], results], axis=1)

    else:
        # Stream processing for massive dataset
        # Vectorized analysis works on one DF, so we must batch the load -> process -> save
        # This requires writing partial results or accumulating carefully.
        # Given "millions of rows", let's assume valid RAM (32GB+ for 33M rows * 25 cols * 8 bytes ≈ 6GB).
        # Actually 33M * 200 bytes = 6.6GB. Pandas overhead ~3x. So ~20GB.
        # It MIGHT fit. If safest, we chunk.
        
        print("Large dataset detected. Processing in chunks from DuckDB...")
        # Since vectorized_analysis expects a DF and returns a DF with aligned index,
        # we can just read in chunks here.
        
        chunk_size = 500000
        rel = con.query(query)
        processed_chunks = []
        
        while True:
            chunk_df = rel.fetch_df_chunk(vectors_per_chunk=1024*4) # approx 4k * vector size
            if chunk_df.empty and (chunk_df is None or len(chunk_df) == 0):
                # fetch_df_chunk returns empty DF when done? 
                # Actually DuckDB Python client behavior on fetch_df_chunk varies by version.
                # simpler:
                break
            
            if len(chunk_df) == 0:
                break
                
            print(f"  Processing chunk of {len(chunk_df)} rows...")
            res_chunk = vectorized_analysis(chunk_df)
            meta_chunk = chunk_df[['pixel_id', 'latitude', 'longitude']].reset_index(drop=True)
            res_chunk = res_chunk.reset_index(drop=True)
            
            combined = pd.concat([meta_chunk, res_chunk], axis=1)
            processed_chunks.append(combined)
            
            if len(processed_chunks) * chunk_size > 5000000:
                 # Safety flush to parquet to avoid OOM?
                 # For now, let's keep in memory if possible or write to disk
                 pass

        if processed_chunks:
            final_df = pd.concat(processed_chunks, ignore_index=True)
        else:
             # Fallback if chunking failed (older duckdb)
             print("Chunking yielded no results or failed. Trying full load...")
             df = con.query(query).df()
             results = vectorized_analysis(df)
             final_df = pd.concat([df[['pixel_id', 'latitude', 'longitude']], results], axis=1)

    # -------------------------------------------------------------------------

    
    # -------------------------------------------------------------------------
    # Clustering (Ecological Typology)
    # -------------------------------------------------------------------------
    if SKLEARN_AVAILABLE:
        print("Running K-Means Clustering (k=5)...")
        features = ['sen_slope', 'cv_gpp', 'autocorr_lag1', 'mean_gpp']
        
        # Simple imputation for NaN (if any pixels were all NaN)
        X = final_df[features].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        final_df['cluster'] = kmeans.fit_predict(X_scaled)
        
        # Summary
        print("\n=== Cluster Summaries ===")
        summary = final_df.groupby('cluster')[features].mean()
        print(summary)
        print("\nCounts:")
        print(final_df['cluster'].value_counts())
        
        # Try to interpret clusters automatically
        # Sort by slope to find "Recovery" vs "Degradation"
        sorted_clusters = summary.sort_values('sen_slope', ascending=False)
        print("\nInterpretation (Sorted by Slope):")
        print(sorted_clusters)

        
    # -------------------------------------------------------------------------
    # Save Results
    # -------------------------------------------------------------------------
    print(f"Saving results to {OUTPUT_PATH}")
    final_df.to_parquet(OUTPUT_PATH)
    print("Done.")

if __name__ == "__main__":
    main()
