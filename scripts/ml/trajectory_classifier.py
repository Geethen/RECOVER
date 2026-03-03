import gc
import duckdb
import numpy as np
import pandas as pd
import concurrent.futures
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import pycatch22
import wandb
import os
import argparse
import glob
import psutil

# ==============================================================================
# PROFILING UTILS
# ==============================================================================

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    return mem_mb

def log_mem(stage):
    print(f"[MEM_PROFILE] {stage: <30} | Memory Usage: {get_memory_usage():>8.2f} MB")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
SUMMARY_PATH = os.path.join(BASE_DIR, "data", "cluster_summary.csv")
TEMP_FEATURES_DIR = os.path.join(BASE_DIR, "data", "temp_features")
YEARS = list(range(2000, 2023))

os.makedirs(TEMP_FEATURES_DIR, exist_ok=True)

# ==============================================================================
# CORE PHYSICS / MATH
# ==============================================================================

# ==============================================================================
# VECTORIZED CORE
# ==============================================================================

def precompute_matrices(years, min_segment=6):
    """Precompute pseudo-inverses for all possible breakpoints to enable vectorization."""
    n = len(years)
    years = np.array(years)
    
    # 0-breakpoint (Linear)
    A0 = np.vstack([np.ones(n), years]).T
    P0 = np.linalg.pinv(A0)  # (2, T)
    
    # 1-breakpoint (Piecewise)
    pseudo_inverses = {}
    design_matrices = {}
    # Segment masks + pseudo-inverses for slope/coupling extraction (avoids per-pixel pinv calls)
    segment_data = {}   # bp -> (m1, m2, P1, P2)
    valid_indices = range(min_segment, n - min_segment + 1)
    for i in valid_indices:
        bp_x = years[i]
        lhs_term = np.maximum(0, years - bp_x)
        A = np.vstack([np.ones(n), years, lhs_term]).T
        design_matrices[bp_x] = A
        pseudo_inverses[bp_x] = np.linalg.pinv(A)  # (3, T)
        # Precompute segment masks and per-segment pseudo-inverses
        m1 = years <= bp_x
        m2 = years > bp_x
        A1 = np.vstack([np.ones(m1.sum()), years[m1]]).T
        A2 = np.vstack([np.ones(m2.sum()), years[m2]]).T
        segment_data[bp_x] = (m1, m2, np.linalg.pinv(A1), np.linalg.pinv(A2))

    return P0, A0, pseudo_inverses, design_matrices, segment_data

def vectorized_fit(Y, P, A):
    """Fit many pixels at once. Y: (N, T), P: (K, T), A: (T, K)."""
    # Beta: (N, K)
    beta = Y @ P.T
    # Pred: (N, T)
    pred = beta @ A.T
    # RSS: (N,)
    rss = np.sum((Y - pred)**2, axis=1)
    return beta, rss

def vectorized_pca1(G, S):
    """Analytic PCA1 for N pixels. G, S: (N, T). Already standardized."""
    # Compute 2x2 covariances per pixel
    # Var(G), Var(S), Cov(G,S)
    c11 = np.var(G, axis=1)
    c22 = np.var(S, axis=1)
    c12 = np.mean(G * S, axis=1) - np.mean(G, axis=1) * np.mean(S, axis=1)
    
    # Eigenvalues of [[c11, c12], [c12, c22]]
    # (L - c11)(L - c22) - c12^2 = 0
    # L^2 - (c11+c22)L + c11*c22 - c12^2 = 0
    tr = c11 + c22
    det = c11 * c22 - c12**2
    eig1 = tr/2 + np.sqrt((tr/2)**2 - det)
    
    # Eigenvector for eig1: (c11-eig1)x + c12y = 0  => y = -(c11-eig1)/c12 * x
    # Normalize
    w1 = c12
    w2 = eig1 - c11
    norm = np.sqrt(w1**2 + w2**2)
    w1 /= (norm + 1e-9)
    w2 /= (norm + 1e-9)
    
    return w1[:, None] * G + w2[:, None] * S

def process_worker(df_chunk, stats_dict, chunk_id, precomputed):
    P0, A0, pseudo_inverses, design_matrices, segment_data = precomputed
    
    # Sort by year to guarantee temporal order regardless of parquet column ordering
    gpp_cols = sorted([c for c in df_chunk.columns if c.startswith('GPP_')], key=lambda x: int(x.split('_')[1]))
    svh_cols = sorted([c for c in df_chunk.columns if c.startswith('SVH_')], key=lambda x: int(x.split('_')[1]))
    
    # Extract matrices — use to_numpy with na_value to handle pd.NA (nullable int/float columns)
    G_raw = df_chunk[gpp_cols].to_numpy(dtype=np.float32, na_value=0.0)
    S_raw = df_chunk[svh_cols].to_numpy(dtype=np.float32, na_value=0.0)
    
    # Standardize
    ecos = df_chunk['eco_id'].values
    G_mean = np.array([stats_dict.get(e, {'GPP_mean':0})['GPP_mean'] for e in ecos])[:, None]
    G_std = np.array([stats_dict.get(e, {'GPP_std':1})['GPP_std'] for e in ecos])[:, None]
    S_mean = np.array([stats_dict.get(e, {'SVH_mean':0})['SVH_mean'] for e in ecos])[:, None]
    S_std = np.array([stats_dict.get(e, {'SVH_std':1})['SVH_std'] for e in ecos])[:, None]
    
    G = (G_raw - G_mean) / (G_std + 1e-9)
    S = (S_raw - S_mean) / (S_std + 1e-9)
    
    # 1. PCA1
    PCA1 = vectorized_pca1(G, S)
    
    # 2. Linear Fits (Base) - use precomputed A0 directly
    _, rss0 = vectorized_fit(PCA1, P0, A0)

    # 3. Piecewise Search - only track RSS and best breakpoint (beta not needed)
    n_pix = G.shape[0]
    best_rss = np.full(n_pix, np.inf)
    best_bp = np.full(n_pix, -1.0)

    for bp, Pi in pseudo_inverses.items():
        Ai = design_matrices[bp]
        _, ri = vectorized_fit(PCA1, Pi, Ai)
        mask = ri < best_rss
        best_rss[mask] = ri[mask]
        best_bp[mask] = bp
        
    # 4. BIC Selection
    bic0 = 23 * np.log(rss0/23 + 1e-9) + 3 * np.log(23)
    bic1 = 23 * np.log(best_rss/23 + 1e-9) + 5 * np.log(23)
    has_break = (bic0 - bic1) > 2
    
    # 5. Feature Extraction
    results = []

    # Pre-extract to plain arrays - avoids repeated per-element .iloc[i] overhead in Python loops
    pixel_ids = df_chunk['pixel_id'].values
    lats = df_chunk['latitude'].values
    lons = df_chunk['longitude'].values

    for i in range(n_pix):
        bp = best_bp[i]
        hb = has_break[i]

        pixel_res = {
            'pixel_id': pixel_ids[i],
            'latitude': lats[i],
            'longitude': lons[i],
        }

        # Look up precomputed segment data once per pixel (avoids per-variable repetition)
        seg = segment_data[bp] if hb else None  # (m1, m2, P1, P2)

        for name, data in [('GPP', G[i]), ('SVH', S[i])]:
            # Catch22 (no vectorised alternative - sequential by design)
            try:
                c22 = pycatch22.catch22_all(data.tolist())
                for cn, cv in zip(c22['names'], c22['values']):
                    pixel_res[f'{name}_c22_{cn}'] = cv
            except: pass


        # Coupling features - reuse precomputed masks from seg
        gdata, sdata = G[i], S[i]
        if not hb:
            c1 = float(np.corrcoef(gdata, sdata)[0, 1]) if len(gdata) > 1 else 0.0
            pixel_res.update({'coupling_corr1': c1, 'coupling_corr2': c1, 'coupling_delta_corr': 0.0})
        else:
            m1, m2 = seg[0], seg[1]
            c1 = float(np.corrcoef(gdata[m1], sdata[m1])[0, 1]) if m1.sum() > 1 else 0.0
            c2 = float(np.corrcoef(gdata[m2], sdata[m2])[0, 1]) if m2.sum() > 1 else 0.0
            pixel_res.update({'coupling_corr1': c1, 'coupling_corr2': c2, 'coupling_delta_corr': c2 - c1})

        results.append(pixel_res)
        
    out = None
    if results:
        df_out = pd.DataFrame(results)
        # lat/lon need float32 (float16 gives only ~3km precision in SA)
        # all catch22/coupling features can safely use float16
        f32_cols = ['latitude', 'longitude']
        f16_cols = [c for c in df_out.select_dtypes(include='float64').columns if c not in f32_cols]
        df_out[f32_cols] = df_out[f32_cols].astype('float32')
        df_out[f16_cols] = df_out[f16_cols].astype('float16')
        out = os.path.join(TEMP_FEATURES_DIR, f"chunk_{chunk_id}.parquet")
        df_out.to_parquet(out, compression='zstd', compression_level=6)
    
    del results, G_raw, S_raw, G, S, PCA1
    gc.collect()
    return out

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=15000)
    parser.add_argument("--n_jobs", type=int, default=4, help="Number of worker processes. Set to 1 for sequential execution.")
    parser.add_argument("--lat_min", type=float, default=None)
    parser.add_argument("--lat_max", type=float, default=None)
    parser.add_argument("--lon_min", type=float, default=None)
    parser.add_argument("--lon_max", type=float, default=None)
    parser.add_argument("--geojson_out", type=str, default=None)
    args = parser.parse_args()
    
    where_clause = ""
    if args.lat_min is not None and args.lat_max is not None and args.lon_min is not None and args.lon_max is not None:
        where_clause = f" WHERE latitude >= {args.lat_min} AND latitude <= {args.lat_max} AND longitude >= {args.lon_min} AND longitude <= {args.lon_max}"
    
    log_mem("Startup")
    con = duckdb.connect()
    
    # 1. Biome Stats
    print("Step 1: Calculating Biome Stats (Streaming)...")
    log_mem("Before Biome Stats Query Definition")
    gpp_raw = [f"GPP_{y}" for y in YEARS]
    svh_raw = [f"SVH_{y}" for y in YEARS]
    
    # Simple subquery for stats to minimize memory
    stats_query = f"""
    SELECT eco_id, 
           AVG(mean_gpp) as GPP_mean, STDDEV(mean_gpp) as GPP_std,
           AVG(mean_svh) as SVH_mean, STDDEV(mean_svh) as SVH_std
    FROM (
        SELECT eco_id, 
               ({' + '.join([f'COALESCE({c},0)' for c in gpp_raw])}) / {len(YEARS)} as mean_gpp,
               ({' + '.join([f'COALESCE({c},0)' for c in svh_raw])}) / {len(YEARS)} as mean_svh
        FROM '{RAW_DATA_PATH}'
        {where_clause}
    ) GROUP BY eco_id
    """
    log_mem("Executing stats_query")
    stats_df = con.execute(stats_query).df().set_index('eco_id')
    stats_dict = stats_df.to_dict('index')
    log_mem("After Stats Calculation")
    
    # 2. Features via streaming
    print("Step 2: Extracting Features in Chunks...")
    total = con.execute(f"SELECT count(*) FROM '{RAW_DATA_PATH}' {where_clause}").fetchone()[0]
    if args.sample: total = min(total, args.sample)
    
    precomputed = precompute_matrices(YEARS)
    feature_files = []
    
    if args.n_jobs > 1:
        # To avoid BrokenProcessPool, we can use "spawn" or ensure the main process doesn't leak memory to workers.
        # But we will use ProcessPoolExecutor and periodically force gc.
        import gc
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            for offset in range(0, total, args.chunk_size):
                # Skip already-processed offsets (e.g. after a resumed run)
                expected = [os.path.join(TEMP_FEATURES_DIR, f"chunk_{offset}_{i}.parquet") for i in range(args.n_jobs)]
                if all(os.path.exists(f) for f in expected):
                    feature_files.extend(expected)
                    continue

                limit = min(args.chunk_size, total - offset)
                df = con.execute(f"SELECT * FROM '{RAW_DATA_PATH}' {where_clause} LIMIT {limit} OFFSET {offset}").df()
                log_mem(f"After fetching data batch (offset {offset})")
                
                # Split for worker
                subs = np.array_split(df, args.n_jobs)
                # Pass copies to ensure we don't hold references in the main thread after execution
                futs = [executor.submit(process_worker, sub.copy(), stats_dict, f"{offset}_{i}", precomputed) for i, sub in enumerate(subs)]
                
                # Help GC immediately
                del df
                del subs
                
                for f in concurrent.futures.as_completed(futs):
                    res = f.result()
                    if res: feature_files.append(res)
                
                gc.collect()
                print(f"  Progress: {100*(offset+limit)/total:.1f}%")
                log_mem(f"End of chunk processing (offset {offset})")
    else:
        for offset in range(0, total, args.chunk_size):
            # Skip already-processed offsets (e.g. after a resumed run)
            expected = os.path.join(TEMP_FEATURES_DIR, f"chunk_{offset}_0.parquet")
            if os.path.exists(expected):
                feature_files.append(expected)
                continue

            limit = min(args.chunk_size, total - offset)
            df = con.execute(f"SELECT * FROM '{RAW_DATA_PATH}' {where_clause} LIMIT {limit} OFFSET {offset}").df()
            log_mem(f"After fetching data batch (offset {offset})")

            res = process_worker(df, stats_dict, f"{offset}_0", precomputed)
            if res: feature_files.append(res)

            del df
            gc.collect()

            print(f"  Progress: {100*(offset+limit)/total:.1f}%")
            log_mem(f"End of chunk processing (offset {offset})")

    # 3. Clustering
    print("Step 3: Clustering (Subsampling for GMM Fit)...")
    all_f = sorted(glob.glob(os.path.join(TEMP_FEATURES_DIR, "chunk_*.parquet")))
    
    if not all_f:
        print("No feature files found.")
        return

    # Random sample of features for fitting (reservoir sampling for spatial representativeness)
    sample_feat = con.execute(f"SELECT * FROM read_parquet({all_f}) USING SAMPLE reservoir(100000 ROWS) REPEATABLE (42)").df()
    log_mem("After loading sample features for UMAP/HDBSCAN")

    # Exclude spatial coords and metadata - clustering must be driven by trajectory shape, not location
    exclude_meta = {'pixel_id', 'eco_id', 'has_break', 'breakpoint', 'latitude', 'longitude'}
    cols = [c for c in sample_feat.columns if c not in exclude_meta]
    
    print("  Identifying collinear features...")
    corr_matrix = sample_feat[cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
    cols_to_keep = [c for c in cols if c not in to_drop]
    print(f"  Dropping {len(to_drop)} features with correlation > 0.85. Keeping {len(cols_to_keep)}.")
    
    os.makedirs(os.path.join(BASE_DIR, "data", "validation"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "data", "selected_features.txt"), "w") as f:
        f.write("\n".join(cols_to_keep))
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(sample_feat[cols_to_keep].fillna(0))
    log_mem("After scaling features")
    
    os.environ["WANDB_ANONYMOUS"] = "allow"
    os.environ["WANDB_SILENT"] = "true"
    min_cluster = 2000
    min_samp = 50
    run = wandb.init(
        project="eco-trajectory",
        config={
            "collinear_threshold": 0.85,
            "features_kept": len(cols_to_keep),
            "umap_components": 3,
            "umap_min_dist": 0.1,
            "hdbscan_min_cluster": min_cluster,
            "hdbscan_min_samples": min_samp
        }
    )
    with open(os.path.join(BASE_DIR, "data", "validation", "wandb_run_id.txt"), "w") as f:
        f.write(run.id)
    
    # Advanced Clustering: UMAP + HDBSCAN
    print("  Fitting UMAP for dimensionality reduction...")
    umap_model = umap.UMAP(n_neighbors=15, n_components=3, min_dist=0.1, random_state=42)
    embedding = umap_model.fit_transform(X_train)
    log_mem("After UMAP fitting")
    
    print("  Fitting HDBSCAN on UMAP embeddings...")
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster, min_samples=min_samp, prediction_data=True)
    hdbscan_model.fit(embedding)
    log_mem("After HDBSCAN training")
    print(f"  Found {len(set(hdbscan_model.labels_)) - (1 if -1 in hdbscan_model.labels_ else 0)} clusters.")

    # Finalize in batches
    print("Step 4: Writing Results...")
    temp_results = []
    for f in all_f:
        out_f = f.replace("chunk_", "res_")
        # Skip already-predicted chunks (e.g. after a resumed run)
        if os.path.exists(out_f):
            temp_results.append(out_f)
            continue

        chunk = pd.read_parquet(f)
        X_chunk = scaler.transform(chunk[cols_to_keep].fillna(0))

        # Predict
        chunk_emb = umap_model.transform(X_chunk)
        labels, probs = hdbscan.approximate_predict(hdbscan_model, chunk_emb)

        slim = pd.DataFrame({
            'pixel_id': chunk['pixel_id'].values,
            'latitude': chunk['latitude'].values,
            'longitude': chunk['longitude'].values,
            'cluster': labels,
            'cluster_prob': probs.astype('float32'),
        })
        slim.to_parquet(out_f, compression='zstd', compression_level=6)
        temp_results.append(out_f)
        log_mem(f"Processed results for {os.path.basename(f)}")
    
    con.execute(f"COPY (SELECT * FROM read_parquet({temp_results})) TO '{OUTPUT_PATH}' (FORMAT PARQUET)")
    log_mem("After saving consolidated results")
    
    # Summary
    # Using GPP_delta_mean instead of GPP_delta_mu as defined in the code
    try:
        summary_query = f"SELECT cluster, COUNT(*) as count FROM '{OUTPUT_PATH}' GROUP BY cluster ORDER BY cluster"
        summary = con.execute(summary_query).df()
        summary.to_csv(SUMMARY_PATH)
    except Exception as e:
        print(f"Summary failed: {e}")
    
    # Cleanup
    for f in glob.glob(os.path.join(TEMP_FEATURES_DIR, "*.parquet")): 
        try: os.remove(f)
        except: pass
    try: os.rmdir(TEMP_FEATURES_DIR)
    except: pass
    
    if args.geojson_out:
        import geopandas as gpd
        from shapely.geometry import Point
        print("Saving GeoJSON output...")
        # Only load necessary columns to save memory
        final_df = con.execute(f"SELECT cluster, cluster_prob, latitude, longitude FROM '{OUTPUT_PATH}'").df()
        if not final_df.empty:
            geom = [Point(xy) for xy in zip(final_df.longitude, final_df.latitude)]
            # Keep only cluster and cluster_prob in attributes
            gdf = gpd.GeoDataFrame(final_df[['cluster', 'cluster_prob']], geometry=geom, crs="EPSG:4326")
            gdf.to_file(args.geojson_out, driver='GeoJSON')
            print(f"GeoJSON output saved to {args.geojson_out}")
            
        
    try:
        wandb.finish()
    except:
        pass
    log_mem("Final")
    print("Done.")

if __name__ == "__main__":
    main()
