"""
Extract catch22 + coupling features for the full ~33M pixel dataset,
select representative samples via feature-space coverage sampling,
train UMAP + HDBSCAN, and apply to all pixels.

Stages (each resumable):
  1. Feature extraction  → data/full_features/chunk_*.parquet
  2. Feature selection    → data/full_features/selected_features.txt
  3. Coverage sampling    → data/full_features/coverage_sample_100k.parquet
  4. Model training       → models/trajectory_umap/*.pkl
  5. Full-dataset predict → data/trajectory_results_v2.parquet
"""

import argparse
import gc
import glob
import json
import os
import pickle
import time
import concurrent.futures
from datetime import datetime

import duckdb
import hdbscan
import numpy as np
import pandas as pd
import psutil
import pycatch22
import umap
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")
FULL_FEATURES_DIR = os.path.join(BASE_DIR, "data", "full_features")
MODEL_DIR = os.path.join(BASE_DIR, "models", "trajectory_umap")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "trajectory_results_v2.parquet")
PROGRESS_LOG = os.path.join(FULL_FEATURES_DIR, "progress.jsonl")

YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]

STAGE_NAMES = {
    1: "Feature Extraction",
    2: "Feature Selection",
    3: "Coverage Sampling",
    4: "UMAP + HDBSCAN Training",
    5: "Full-Dataset Prediction",
}


# ── Progress Logger ──────────────────────────────────────────────────────────
class ProgressLogger:
    def __init__(self, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._path = log_path
        self._t0 = time.time()
        # Clear previous log
        with open(self._path, "w") as f:
            pass

    def log(self, stage, step=0, total=0, msg="", extra=None):
        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        pct = (step / total * 100) if total > 0 else 0
        entry = {
            "stage": stage,
            "stage_name": STAGE_NAMES.get(stage, ""),
            "step": step,
            "total": total,
            "pct": round(pct, 1),
            "msg": msg,
            "mem_mb": round(mem_mb, 1),
            "elapsed_s": round(time.time() - self._t0, 1),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        if extra:
            entry.update(extra)
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[S{stage}] {msg}  ({pct:.1f}%  mem={mem_mb:.0f}MB)")


# Module-level logger, initialised in main()
plog: ProgressLogger = None  # type: ignore


# ── Utilities ────────────────────────────────────────────────────────────────
def log_mem(stage):
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"[MEM] {stage:<40} | {mem_mb:>8.1f} MB")


# ── Feature extraction worker (no breakpoint detection) ─────────────────────
def _extract_worker(args):
    """Extract catch22 + coupling for a sub-chunk. Writes parquet, returns path."""
    df_chunk, stats_dict, out_path = args

    gpp_raw = df_chunk[GPP_COLS].to_numpy(dtype=np.float32, na_value=0.0)
    svh_raw = df_chunk[SVH_COLS].to_numpy(dtype=np.float32, na_value=0.0)
    ecos = df_chunk["eco_id"].values

    # Per-ecoregion standardisation
    G_mean = np.array([stats_dict.get(e, {"GPP_mean": 0})["GPP_mean"] for e in ecos])[:, None]
    G_std  = np.array([stats_dict.get(e, {"GPP_std":  1})["GPP_std"]  for e in ecos])[:, None]
    S_mean = np.array([stats_dict.get(e, {"SVH_mean": 0})["SVH_mean"] for e in ecos])[:, None]
    S_std  = np.array([stats_dict.get(e, {"SVH_std":  1})["SVH_std"]  for e in ecos])[:, None]

    G = (gpp_raw - G_mean) / (G_std + 1e-9)
    S = (svh_raw - S_mean) / (S_std + 1e-9)

    pixel_ids = df_chunk["pixel_id"].values
    lats = df_chunk["latitude"].values
    lons = df_chunk["longitude"].values

    results = []
    for i in range(len(df_chunk)):
        row = {
            "pixel_id": pixel_ids[i],
            "latitude": lats[i],
            "longitude": lons[i],
        }

        for name, data in [("GPP", G[i]), ("SVH", S[i])]:
            try:
                c22 = pycatch22.catch22_all(data.tolist())
                for cn, cv in zip(c22["names"], c22["values"]):
                    row[f"{name}_c22_{cn}"] = cv
            except Exception:
                pass

        # Whole-series coupling correlation
        g, s = G[i], S[i]
        row["coupling_corr"] = float(np.corrcoef(g, s)[0, 1]) if len(g) > 1 else 0.0

        results.append(row)

    if results:
        df_out = pd.DataFrame(results)
        f32 = ["latitude", "longitude"]
        f16 = [c for c in df_out.select_dtypes("float64").columns if c not in f32]
        df_out[f32] = df_out[f32].astype("float32")
        if f16:
            df_out[f16] = df_out[f16].astype("float16")
        df_out.to_parquet(out_path, compression="zstd", compression_level=6, index=False)
        return out_path
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════
def stage_extract(con, args):
    plog.log(1, 0, 1, "Starting feature extraction")

    os.makedirs(FULL_FEATURES_DIR, exist_ok=True)
    tmp_dir = os.path.join(BASE_DIR, "data", "duckdb_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp_dir}'")

    # Eco-stats
    print("Computing per-ecoregion standardisation stats...")
    gpp_sum = " + ".join([f"COALESCE({c},0)" for c in GPP_COLS])
    svh_sum = " + ".join([f"COALESCE({c},0)" for c in SVH_COLS])
    n = len(YEARS)

    stats_df = con.execute(f"""
        SELECT eco_id,
               AVG(mean_gpp) AS GPP_mean, STDDEV(mean_gpp) AS GPP_std,
               AVG(mean_svh) AS SVH_mean, STDDEV(mean_svh) AS SVH_std
        FROM (
            SELECT eco_id,
                   ({gpp_sum}) / {n} AS mean_gpp,
                   ({svh_sum}) / {n} AS mean_svh
            FROM '{RAW_DATA_PATH}'
        )
        GROUP BY eco_id
    """).df().set_index("eco_id")
    stats_dict = stats_df.to_dict("index")
    plog.log(1, 0, 1, f"Stats for {len(stats_dict)} ecoregions")

    total = con.execute(f"SELECT count(*) FROM '{RAW_DATA_PATH}'").fetchone()[0]
    plog.log(1, 0, 1, f"Total pixels: {total:,}")

    chunk_size = args.chunk_size
    n_jobs = args.n_jobs
    feature_files = []
    n_batches = (total + chunk_size - 1) // chunk_size

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for batch_idx in tqdm(range(n_batches), desc="Extracting features"):
            offset = batch_idx * chunk_size
            limit = min(chunk_size, total - offset)

            # Resume check
            expected = [
                os.path.join(FULL_FEATURES_DIR, f"chunk_{offset}_{i}.parquet")
                for i in range(n_jobs)
            ]
            if all(os.path.exists(f) for f in expected):
                feature_files.extend(expected)
                continue

            df = con.execute(
                f"SELECT * FROM '{RAW_DATA_PATH}' LIMIT {limit} OFFSET {offset}"
            ).df()

            subs = np.array_split(df, n_jobs)
            worker_args = [
                (sub.copy(), stats_dict,
                 os.path.join(FULL_FEATURES_DIR, f"chunk_{offset}_{i}.parquet"))
                for i, sub in enumerate(subs)
            ]
            del df, subs
            gc.collect()

            futures = [executor.submit(_extract_worker, wa) for wa in worker_args]
            for fut in concurrent.futures.as_completed(futures):
                path = fut.result()
                if path:
                    feature_files.append(path)

            del worker_args
            gc.collect()

            if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
                plog.log(1, batch_idx + 1, n_batches,
                         f"Batch {batch_idx+1}/{n_batches}")

    plog.log(1, n_batches, n_batches,
             f"Done — {len(feature_files)} chunk files", extra={"status": "complete"})
    return sorted(feature_files)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Feature Selection
# ══════════════════════════════════════════════════════════════════════════════
def stage_select_features(con, feature_files, args):
    plog.log(2, 0, 1, "Starting feature selection")

    sel_path = os.path.join(FULL_FEATURES_DIR, "selected_features.txt")
    if os.path.exists(sel_path) and not args.force:
        cols = open(sel_path).read().strip().split("\n")
        plog.log(2, 1, 1, f"Loaded {len(cols)} features (cached)", extra={"status": "complete"})
        return cols

    exclude_meta = {"pixel_id", "latitude", "longitude"}

    # 5 batches with different seeds, average correlation matrices
    n_batches = 5
    seeds = [42 + i * 137 for i in range(n_batches)]
    corr_sum = None
    all_cols = None

    for seed in tqdm(seeds, desc="Computing correlations"):
        sample = con.execute(f"""
            SELECT * FROM read_parquet({feature_files})
            USING SAMPLE reservoir(100000 ROWS) REPEATABLE ({seed})
        """).df()
        cols = [c for c in sample.columns if c not in exclude_meta]
        if all_cols is None:
            all_cols = cols
        corr = sample[cols].corr().abs().values
        corr_sum = corr if corr_sum is None else corr_sum + corr
        del sample

    avg_corr = corr_sum / n_batches
    avg_corr_df = pd.DataFrame(avg_corr, index=all_cols, columns=all_cols)

    upper = avg_corr_df.where(np.triu(np.ones(avg_corr_df.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.75)]
    cols_to_keep = [c for c in all_cols if c not in to_drop]

    plog.log(2, 1, 1,
             f"Dropped {len(to_drop)}, keeping {len(cols_to_keep)} features",
             extra={"status": "complete", "kept": len(cols_to_keep), "dropped": len(to_drop)})

    with open(sel_path, "w") as f:
        f.write("\n".join(cols_to_keep))

    return cols_to_keep


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Coverage Sampling
# ══════════════════════════════════════════════════════════════════════════════
def stage_coverage_sampling(con, feature_files, cols_to_keep, args):
    plog.log(3, 0, 1, "Starting coverage sampling")

    sample_path = os.path.join(FULL_FEATURES_DIR, "coverage_sample_100k.parquet")
    if os.path.exists(sample_path) and not args.force:
        plog.log(3, 1, 1, "Coverage sample exists (cached)", extra={"status": "complete"})
        return sample_path

    K = args.n_clusters
    TARGET = args.n_sample

    # 1. Fit MiniBatchKMeans on 500k preliminary sample
    print(f"  Fitting MiniBatchKMeans (k={K}) on 500k preliminary sample...")
    prelim = con.execute(f"""
        SELECT * FROM read_parquet({feature_files})
        USING SAMPLE reservoir(500000 ROWS) REPEATABLE (42)
    """).df()
    scaler_tmp = StandardScaler()
    X_prelim = scaler_tmp.fit_transform(prelim[cols_to_keep].fillna(0).astype("float32"))
    kmeans = MiniBatchKMeans(n_clusters=K, random_state=42, batch_size=10000, n_init=3)
    kmeans.fit(X_prelim)
    del prelim, X_prelim
    gc.collect()
    plog.log(3, 0, len(feature_files), "KMeans fitted, assigning clusters")
    all_pids = []
    all_labels = []

    n_feat = len(feature_files)
    for fi, f in enumerate(tqdm(feature_files, desc="Cluster assignment")):
        chunk = pd.read_parquet(f)
        X = scaler_tmp.transform(chunk[cols_to_keep].fillna(0).astype("float32"))
        labels = kmeans.predict(X)
        all_pids.append(chunk["pixel_id"].values)
        all_labels.append(labels.astype(np.int16))
        del chunk, X, labels
        gc.collect()
        if (fi + 1) % 100 == 0 or fi == n_feat - 1:
            plog.log(3, fi + 1, n_feat, f"Cluster assign {fi+1}/{n_feat}")

    pids = np.concatenate(all_pids)
    labels = np.concatenate(all_labels)
    del all_pids, all_labels
    gc.collect()

    # 3. Stratified sampling — proportional allocation
    unique, counts = np.unique(labels, return_counts=True)
    total_n = counts.sum()
    alloc = np.maximum(1, np.round(TARGET * counts / total_n).astype(int))
    # Adjust to hit exactly TARGET
    diff = TARGET - alloc.sum()
    if diff > 0:
        # Add to largest clusters
        order = np.argsort(-counts)
        for idx in order[:diff]:
            alloc[idx] += 1
    elif diff < 0:
        # Remove from largest clusters (keep min 1)
        order = np.argsort(-alloc)
        for idx in order:
            reduce = min(alloc[idx] - 1, -diff)
            alloc[idx] -= reduce
            diff += reduce
            if diff >= 0:
                break

    print(f"  Sampling {TARGET:,} pixels across {len(unique)} clusters...")
    rng = np.random.RandomState(42)
    sampled_pids = set()
    for cl, n_alloc in tqdm(zip(unique, alloc), total=len(unique), desc="Sampling"):
        mask = labels == cl
        cl_pids = pids[mask]
        chosen = rng.choice(cl_pids, size=min(n_alloc, len(cl_pids)), replace=False)
        sampled_pids.update(chosen.tolist())

    del pids, labels
    gc.collect()
    print(f"  Selected {len(sampled_pids):,} unique pixels.")

    # 4. Collect full feature rows for sampled pixels
    print("  Collecting feature rows for sampled pixels...")
    parts = []
    remaining = set(sampled_pids)
    for f in tqdm(feature_files, desc="Collecting rows"):
        if not remaining:
            break
        chunk = pd.read_parquet(f)
        matched = chunk[chunk["pixel_id"].isin(remaining)]
        if not matched.empty:
            parts.append(matched)
            remaining -= set(matched["pixel_id"].tolist())
        del chunk
        gc.collect()

    sample_df = pd.concat(parts, ignore_index=True)
    sample_df.to_parquet(sample_path, compression="zstd", compression_level=6, index=False)
    plog.log(3, 1, 1, f"Saved {len(sample_df):,} coverage samples",
             extra={"status": "complete", "n_samples": len(sample_df)})
    del parts, sample_df
    gc.collect()
    return sample_path


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4: Train UMAP + HDBSCAN
# ══════════════════════════════════════════════════════════════════════════════
def stage_train_models(sample_path, cols_to_keep, args):
    plog.log(4, 0, 3, "Starting model training")

    os.makedirs(MODEL_DIR, exist_ok=True)

    umap_path = os.path.join(MODEL_DIR, "umap_model.pkl")
    if os.path.exists(umap_path) and not args.force:
        plog.log(4, 3, 3, "Models loaded (cached)", extra={"status": "complete"})
        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(umap_path, "rb") as f:
            umap_model = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "hdbscan_model.pkl"), "rb") as f:
            hdbscan_model = pickle.load(f)
        return scaler, umap_model, hdbscan_model

    sample = pd.read_parquet(sample_path)
    print(f"  Loaded coverage sample: {len(sample):,} rows, {len(cols_to_keep)} features.")

    # StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(sample[cols_to_keep].fillna(0).astype("float32"))

    # UMAP
    plog.log(4, 1, 3, "Fitting UMAP...")
    umap_model = umap.UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0, random_state=42
    )
    embedding = umap_model.fit_transform(X)
    log_mem("After UMAP fit")

    # HDBSCAN (fine-grained)
    min_cs = args.min_cluster_size
    min_s = args.min_samples
    plog.log(4, 2, 3, f"Fitting HDBSCAN (mcs={min_cs}, ms={min_s})")
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cs, min_samples=min_s, prediction_data=True
    )
    hdbscan_model.fit(embedding)
    n_clusters = len(set(hdbscan_model.labels_)) - (1 if -1 in hdbscan_model.labels_ else 0)
    n_noise = (hdbscan_model.labels_ == -1).sum()
    print(f"  Found {n_clusters} clusters, {n_noise} noise points.")

    # Save artifacts
    for name, obj in [("scaler", scaler), ("umap_model", umap_model), ("hdbscan_model", hdbscan_model)]:
        with open(os.path.join(MODEL_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(obj, f)
    with open(os.path.join(MODEL_DIR, "selected_features.pkl"), "wb") as f:
        pickle.dump(cols_to_keep, f)

    # Save training sample with embeddings and labels
    for dim in range(embedding.shape[1]):
        sample[f"umap_{dim+1}"] = embedding[:, dim].astype("float32")
    sample["cluster"] = hdbscan_model.labels_
    sample["cluster_prob"] = hdbscan_model.probabilities_.astype("float32")
    sample.to_parquet(
        os.path.join(MODEL_DIR, "training_sample_with_clusters.parquet"),
        compression="zstd", compression_level=6, index=False,
    )
    plog.log(4, 3, 3, f"Done — {n_clusters} clusters, {n_noise} noise",
             extra={"status": "complete", "n_clusters": n_clusters, "n_noise": int(n_noise)})
    del sample, X, embedding
    gc.collect()

    return scaler, umap_model, hdbscan_model


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5: Apply to Full Dataset
# ══════════════════════════════════════════════════════════════════════════════
def stage_apply_full(con, feature_files, cols_to_keep, scaler, umap_model, hdbscan_model):
    import pyarrow as pa
    import pyarrow.parquet as pq

    plog.log(5, 0, len(feature_files), "Starting full-dataset prediction")

    if os.path.exists(OUTPUT_PATH):
        plog.log(5, 1, 1, "Output exists (cached)", extra={"status": "complete"})
        return

    schema = pa.schema([
        ("pixel_id",    pa.int64()),
        ("latitude",    pa.float32()),
        ("longitude",   pa.float32()),
        ("cluster",     pa.int32()),
        ("cluster_prob",pa.float32()),
    ])

    # Resumable streaming write: write to .partial file, track progress via row groups.
    # On restart, count existing row groups to know how many chunks to skip.
    PARTIAL_PATH = OUTPUT_PATH + ".partial"
    resume_from = 0
    if os.path.exists(PARTIAL_PATH):
        try:
            pf = pq.ParquetFile(PARTIAL_PATH)
            resume_from = pf.metadata.num_row_groups
            del pf
            print(f"  Resuming Stage 5: {resume_from}/{len(feature_files)} chunks already written.")
        except Exception:
            os.remove(PARTIAL_PATH)

    n_feat = len(feature_files)
    with pq.ParquetWriter(PARTIAL_PATH, schema, compression="zstd") as writer:
        for fi, f in enumerate(tqdm(feature_files, desc="Predicting clusters")):
            if fi < resume_from:
                continue  # already written in a previous run

            chunk = pd.read_parquet(f)
            # Drop rows with NaN pixel_id (corrupt/incomplete rows)
            chunk = chunk.dropna(subset=["pixel_id"])
            if len(chunk) == 0:
                continue
            X = scaler.transform(chunk[cols_to_keep].fillna(0).astype("float32"))
            chunk_emb = umap_model.transform(X)
            labels, probs = hdbscan.approximate_predict(hdbscan_model, chunk_emb)

            table = pa.table({
                "pixel_id":    pa.array(chunk["pixel_id"].values.astype(np.int64), type=pa.int64()),
                "latitude":    pa.array(chunk["latitude"].values,  type=pa.float32()),
                "longitude":   pa.array(chunk["longitude"].values, type=pa.float32()),
                "cluster":     pa.array(labels,                    type=pa.int32()),
                "cluster_prob":pa.array(probs.astype("float32"),   type=pa.float32()),
            })
            writer.write_table(table)
            del chunk, X, chunk_emb, labels, probs, table
            gc.collect()

            if (fi + 1) % 50 == 0 or fi == n_feat - 1:
                plog.log(5, fi + 1, n_feat, f"Predicted {fi+1}/{n_feat}")

    os.rename(PARTIAL_PATH, OUTPUT_PATH)

    size_mb = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
    print(f"  Saved {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Summary
    summary = con.execute(f"""
        SELECT cluster, COUNT(*) as count
        FROM '{OUTPUT_PATH}' GROUP BY cluster ORDER BY cluster
    """).df()
    print("\n  Cluster distribution:")
    for _, r in summary.iterrows():
        print(f"    Cluster {int(r['cluster']):>3}: {int(r['count']):>10,} pixels")

    plog.log(5, n_feat, n_feat, f"Done — saved {size_mb:.0f} MB",
             extra={"status": "complete", "size_mb": round(size_mb, 1)})


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Extract full features + coverage-sampled UMAP/HDBSCAN"
    )
    parser.add_argument("--chunk_size", type=int, default=50000)
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--n_clusters", type=int, default=500,
                        help="MiniBatchKMeans clusters for coverage sampling")
    parser.add_argument("--n_sample", type=int, default=100000,
                        help="Number of coverage samples")
    parser.add_argument("--min_cluster_size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (small = more clusters)")
    parser.add_argument("--min_samples", type=int, default=5,
                        help="HDBSCAN min_samples")
    parser.add_argument("--force", action="store_true",
                        help="Re-run stages even if outputs exist")
    parser.add_argument("--skip-extract", action="store_true",
                        help="Skip stage 1, use existing full_features/")
    args = parser.parse_args()

    global plog
    os.makedirs(FULL_FEATURES_DIR, exist_ok=True)
    plog = ProgressLogger(PROGRESS_LOG)

    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=2")

    # Stage 1
    if args.skip_extract:
        feature_files = sorted(glob.glob(os.path.join(FULL_FEATURES_DIR, "chunk_*.parquet")))
        print(f"Skipping extraction. Found {len(feature_files)} existing chunk files.")
    else:
        feature_files = stage_extract(con, args)

    if not feature_files:
        print("ERROR: No feature files found. Run without --skip-extract.")
        return

    # Stage 2
    cols_to_keep = stage_select_features(con, feature_files, args)

    # Stage 3
    sample_path = stage_coverage_sampling(con, feature_files, cols_to_keep, args)

    # Stage 4
    scaler, umap_model, hdbscan_model = stage_train_models(sample_path, cols_to_keep, args)

    # Stage 5
    stage_apply_full(con, feature_files, cols_to_keep, scaler, umap_model, hdbscan_model)

    con.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
