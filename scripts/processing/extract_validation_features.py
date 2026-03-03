"""
Extract catch22 + coupling features for 5 × 100k validation subsets.

Uses the same seeds as export_qgis_subsets.py so the subsets are identical.
Replicates the classifier's exact per-ecoregion standardisation before catch22.
Coupling is the whole-series GPP/SVH Pearson correlation (simplified: no breakpoint).

Outputs
-------
  data/validation_features/subset_{1..5}.parquet   — per-subset features + labels
  data/validation_features/all_subsets.parquet     — merged, ready for validation scripts
"""

import gc
import os
import concurrent.futures

import duckdb
import numpy as np
import pandas as pd
import pycatch22

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_PATH = os.path.join(BASE_DIR, "data", "trajectory_results.parquet")
RAW_PATH     = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
OUT_DIR      = os.path.join(BASE_DIR, "data", "validation_features")

os.makedirs(OUT_DIR, exist_ok=True)

YEARS      = list(range(2000, 2023))
GPP_COLS   = [f"GPP_{y}" for y in YEARS]
SVH_COLS   = [f"SVH_{y}" for y in YEARS]
N_SUBSETS  = 5
SEEDS      = [i * 137 for i in range(1, N_SUBSETS + 1)]   # matches export_qgis_subsets.py
N_JOBS     = 4
CHUNK_SIZE = 5_000   # pixels per worker batch


# ── Worker ────────────────────────────────────────────────────────────────────
def _extract_chunk(args):
    """Run catch22 + coupling on a sub-chunk. Returns list of feature dicts."""
    df_chunk, stats_dict = args

    gpp_raw = df_chunk[GPP_COLS].to_numpy(dtype=np.float32, na_value=0.0)
    svh_raw = df_chunk[SVH_COLS].to_numpy(dtype=np.float32, na_value=0.0)
    ecos    = df_chunk['eco_id'].values

    # Per-ecoregion standardisation (identical to classifier)
    G_mean = np.array([stats_dict.get(e, {'GPP_mean': 0})['GPP_mean'] for e in ecos])[:, None]
    G_std  = np.array([stats_dict.get(e, {'GPP_std':  1})['GPP_std']  for e in ecos])[:, None]
    S_mean = np.array([stats_dict.get(e, {'SVH_mean': 0})['SVH_mean'] for e in ecos])[:, None]
    S_std  = np.array([stats_dict.get(e, {'SVH_std':  1})['SVH_std']  for e in ecos])[:, None]

    G = (gpp_raw - G_mean) / (G_std + 1e-9)   # (N, 23)
    S = (svh_raw - S_mean) / (S_std + 1e-9)

    pixel_ids    = df_chunk['pixel_id'].values
    lats         = df_chunk['latitude'].values
    lons         = df_chunk['longitude'].values
    clusters     = df_chunk['cluster'].values
    cluster_prob = df_chunk['cluster_prob'].values

    results = []
    for i in range(len(df_chunk)):
        row = {
            'pixel_id':     pixel_ids[i],
            'latitude':     lats[i],
            'longitude':    lons[i],
            'cluster':      int(clusters[i]),
            'cluster_prob': float(cluster_prob[i]),
        }

        for name, data in [('GPP', G[i]), ('SVH', S[i])]:
            try:
                c22 = pycatch22.catch22_all(data.tolist())
                for cn, cv in zip(c22['names'], c22['values']):
                    row[f'{name}_c22_{cn}'] = cv
            except Exception:
                pass

        # Whole-series GPP/SVH coupling correlation
        g, s = G[i], S[i]
        row['coupling_corr'] = float(np.corrcoef(g, s)[0, 1]) if len(g) > 1 else 0.0

        results.append(row)

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tmp_dir = os.path.join(BASE_DIR, "data", "duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    con.execute("SET threads=2")

    # ── Biome stats (identical formula to classifier) ──────────────────────
    print("Computing per-ecoregion standardisation stats...")
    gpp_sum = ' + '.join([f'COALESCE({c},0)' for c in GPP_COLS])
    svh_sum = ' + '.join([f'COALESCE({c},0)' for c in SVH_COLS])
    n = len(YEARS)

    stats_df = con.execute(f"""
        SELECT eco_id,
               AVG(mean_gpp) AS GPP_mean, STDDEV(mean_gpp) AS GPP_std,
               AVG(mean_svh) AS SVH_mean, STDDEV(mean_svh) AS SVH_std
        FROM (
            SELECT eco_id,
                   ({gpp_sum}) / {n} AS mean_gpp,
                   ({svh_sum}) / {n} AS mean_svh
            FROM '{RAW_PATH}'
        )
        GROUP BY eco_id
    """).df().set_index('eco_id')
    stats_dict = stats_df.to_dict('index')
    print(f"  Stats computed for {len(stats_dict)} ecoregions.")

    merged_parts = []

    for subset_idx, seed in enumerate(SEEDS, start=1):
        out_path = os.path.join(OUT_DIR, f"subset_{subset_idx}.parquet")
        if os.path.exists(out_path):
            print(f"\nSubset {subset_idx}: already exists, skipping.")
            merged_parts.append(out_path)
            continue

        print(f"\nSubset {subset_idx}/{N_SUBSETS} (seed={seed})...")

        # ── Sample pixel ids + labels from results ─────────────────────────
        res_sample = con.execute(f"""
            SELECT pixel_id, latitude, longitude, cluster, cluster_prob
            FROM (SELECT * FROM read_parquet('{RESULTS_PATH}'))
            USING SAMPLE reservoir(100000 ROWS) REPEATABLE ({seed})
        """).df()

        # ── Join with raw data (GPP/SVH + eco_id) ─────────────────────────
        # Batch scan: read 2M rows at a time and filter in Python.
        # Avoids DuckDB OOM from decompressing all 47 columns × 33M rows at once.
        pids_remaining = set(res_sample['pixel_id'].tolist())
        needed_cols = ['pixel_id'] + GPP_COLS + SVH_COLS + ['eco_id']
        cols_str    = ', '.join(f'"{c}"' for c in needed_cols)
        BATCH = 2_000_000
        offset = 0
        batches = []

        while pids_remaining:
            chunk = con.execute(f"""
                SELECT {cols_str} FROM '{RAW_PATH}'
                LIMIT {BATCH} OFFSET {offset}
            """).df()
            if chunk.empty:
                break
            matched = chunk[chunk['pixel_id'].isin(pids_remaining)]
            if not matched.empty:
                batches.append(matched)
                pids_remaining -= set(matched['pixel_id'].tolist())
            offset += BATCH

        raw_sample = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(columns=needed_cols)
        df = res_sample.merge(raw_sample, on='pixel_id').reset_index(drop=True)
        print(f"  Joined {len(df):,} pixels.")

        # ── Parallel catch22 extraction ────────────────────────────────────
        chunks = [df.iloc[i:i + CHUNK_SIZE].copy() for i in range(0, len(df), CHUNK_SIZE)]
        args   = [(chunk, stats_dict) for chunk in chunks]

        all_rows = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            for j, rows in enumerate(ex.map(_extract_chunk, args), start=1):
                all_rows.extend(rows)
                if j % 5 == 0 or j == len(chunks):
                    print(f"  {j}/{len(chunks)} chunks done ({len(all_rows):,} pixels)")

        feat_df = pd.DataFrame(all_rows)

        # Compact dtypes — lat/lon keep float32, features to float32
        float_cols = [c for c in feat_df.select_dtypes('float64').columns
                      if c not in ('latitude', 'longitude')]
        feat_df[float_cols]                = feat_df[float_cols].astype('float32')
        feat_df[['latitude', 'longitude']] = feat_df[['latitude', 'longitude']].astype('float32')

        feat_df.to_parquet(out_path, compression='zstd', compression_level=6, index=False)
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved {out_path}  ({size_mb:.1f} MB, {len(feat_df):,} rows, "
              f"{len(feat_df.columns)} columns)")
        merged_parts.append(out_path)

        del df, raw_sample, all_rows, feat_df, chunks
        gc.collect()

    # ── Merge all subsets ──────────────────────────────────────────────────
    merged_path = os.path.join(OUT_DIR, "all_subsets.parquet")
    print(f"\nMerging {len(merged_parts)} subsets into {merged_path} ...")
    con.execute(f"""
        COPY (SELECT * FROM read_parquet({merged_parts}))
        TO '{merged_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    size_mb = os.path.getsize(merged_path) / 1024 / 1024
    print(f"Done.  {merged_path}  ({size_mb:.1f} MB)")

    con.close()


if __name__ == "__main__":
    main()
