"""
Scale recovery analysis to all SA ecoregions.

Per-ecoregion pipeline:
  Stage 1a (identify):  MK+TS trend test on abandoned-ag → recovering pixels
  Stage 1b (extract):   AlphaEarth 64D embeddings from GEE for recovering pixels
  Stage 2  (score):     Metrics A, B, C_eco, C_local → recovery_scores

Skips eco_id=81 (already complete).

Usage:
    # Single ecoregion:
    python scripts/extraction/extract_all_ecoregions.py --eco_id 41

    # All ecoregions (skips 81):
    python scripts/extraction/extract_all_ecoregions.py --all

    # Stage control:
    python scripts/extraction/extract_all_ecoregions.py --eco_id 41 --stage extract
    python scripts/extraction/extract_all_ecoregions.py --eco_id 41 --stage score

    # Test mode (small sample):
    python scripts/extraction/extract_all_ecoregions.py --eco_id 41 --test_mode
"""
import sys
import os
import ee
import time
import json
import argparse
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from scipy import stats
from scipy.stats import percentileofscore
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm.auto import tqdm

sys.stdout.reconfigure(line_buffering=True)

# ── paths & constants ─────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
ABANDONED_AG = BASE_DIR / "data" / "abandoned_ag_gpp_2000_2022_SA.parquet"
INDICES_GPP = BASE_DIR / "data" / "indices_gpp_svh_2000_2022.parquet"
NAT_SUBSET = BASE_DIR / "data" / "dfsubsetNatural.parquet"

SKIP_ECO = 81  # already complete
YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
LATE_YEARS = list(range(2018, 2023))
LATE_GPP = [f"GPP_{y}" for y in LATE_YEARS]
LATE_SVH = [f"SVH_{y}" for y in LATE_YEARS]
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
KNN_K = 10
KNN_RADIUS_KM = 50.0
EARTH_R_KM = 6371.0
MIN_NATURAL_POINTS = 30  # skip ecoregion if fewer natural ref


def eco_paths(eco_id):
    """Return output paths for a given ecoregion."""
    d = BASE_DIR / "data"
    return {
        "recovering": d / f"recovering_eco{eco_id}.parquet",
        "embeddings": d / f"recovering_eco{eco_id}_embeddings.parquet",
        "scores": d / f"recovery_scores_eco{eco_id}.parquet",
    }


# ── utilities ─────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.processed = self._load()
        self.lock = Lock()

    def _load(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except (json.JSONDecodeError, ValueError):
                return set()
        return set()

    def mark(self, batch_id):
        with self.lock:
            self.processed.add(batch_id)
            with open(self.checkpoint_file, 'w') as f:
                json.dump(list(self.processed), f)

    def is_done(self, batch_id):
        return batch_id in self.processed


def retry_gee(func, max_retries=3, backoff=2):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(backoff ** (attempt + 1))
            else:
                raise


def mk_sen_batch(y_arr):
    """Vectorised Mann-Kendall + Theil-Sen for (N, T) array."""
    B, T = y_arr.shape
    i, j = np.triu_indices(T, k=1)
    dx = (j - i).astype(np.float32)
    diffs = y_arr[:, j] - y_arr[:, i]
    slopes = diffs / dx[None, :]
    sen_slope = np.median(slopes, axis=1).astype(np.float32)
    S = np.sum(np.sign(diffs), axis=1).astype(np.float64)
    var_s = (T * (T - 1) * (2 * T + 5)) / 18.0
    sigma = np.sqrt(var_s)
    Z = np.zeros(B)
    pos = S > 0; neg = S < 0
    Z[pos] = (S[pos] - 1) / sigma
    Z[neg] = (S[neg] + 1) / sigma
    p = 2 * stats.norm.sf(np.abs(Z))
    return sen_slope, Z, p


# ── inventory ─────────────────────────────────────────────────────────

def get_ecoregion_inventory():
    """Return DataFrame of eco_ids with abandoned-ag and natural ref counts."""
    con = duckdb.connect()
    ag = con.execute(f"""
        SELECT eco_id, count(*) as n_ag
        FROM '{ABANDONED_AG}'
        WHERE eco_id IS NOT NULL
        GROUP BY eco_id ORDER BY n_ag DESC
    """).df()

    nat = con.execute(f"""
        SELECT g.eco_id, count(*) as n_natural
        FROM '{NAT_SUBSET}' n
        JOIN '{INDICES_GPP}' g
          ON n.id = split_part(g.pixel_id, '_', 1)
        GROUP BY g.eco_id
    """).df()
    con.close()

    inv = ag.merge(nat, on="eco_id", how="left")
    inv["n_natural"] = inv["n_natural"].fillna(0).astype(int)
    inv = inv.sort_values("n_ag", ascending=False).reset_index(drop=True)
    return inv


# ======================================================================
# STAGE 1a: IDENTIFY RECOVERING PIXELS
# ======================================================================

def identify_recovering_pixels(eco_id, chunk_size=50000, test_mode=False):
    paths = eco_paths(eco_id)
    out_path = paths["recovering"]

    print("=" * 70)
    print(f"STAGE 1a: Identifying recovering pixels (eco_id={eco_id})")
    print("=" * 70)

    # Check for existing output — skip already-identified pixels
    existing_pids = set()
    if out_path.exists():
        existing = pd.read_parquet(str(out_path))
        existing_pids = set(existing["pixel_id"].values)
        print(f"  Found {len(existing_pids):,} already-identified pixels — will skip")

    con = duckdb.connect()
    total = con.execute(f"""
        SELECT count(*) FROM '{ABANDONED_AG}' WHERE eco_id = {eco_id}
    """).fetchone()[0]
    print(f"Total abandoned-ag pixels (eco_id={eco_id}): {total:,}")

    if test_mode:
        total = min(total, 100000)
        print(f"*** TEST MODE: processing {total:,} pixels ***")

    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)
    n_chunks = (total + chunk_size - 1) // chunk_size
    all_recovering = []
    total_recovering = 0

    for ci in tqdm(range(n_chunks), desc="Scanning chunks"):
        offset = ci * chunk_size
        limit = min(chunk_size, total - offset)

        chunk = con.execute(f"""
            SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
                   {gpp_sel}, {svh_sel}
            FROM '{ABANDONED_AG}'
            WHERE eco_id = {eco_id}
            LIMIT {limit} OFFSET {offset}
        """).df()
        if chunk.empty:
            break

        # Skip already-processed pixels
        if existing_pids:
            mask_new = ~chunk["pixel_id"].isin(existing_pids)
            chunk = chunk[mask_new].reset_index(drop=True)
            if chunk.empty:
                continue

        gpp = chunk[GPP_COLS].values.astype(np.float32)
        svh = chunk[SVH_COLS].values.astype(np.float32)

        sub_size = 5000
        gpp_slopes = np.empty(len(chunk), dtype=np.float32)
        svh_slopes = np.empty(len(chunk), dtype=np.float32)
        gpp_ps = np.empty(len(chunk), dtype=np.float64)
        svh_ps = np.empty(len(chunk), dtype=np.float64)

        for si in range(0, len(chunk), sub_size):
            se = min(si + sub_size, len(chunk))
            gs, _, gp = mk_sen_batch(gpp[si:se])
            ss, _, sp = mk_sen_batch(svh[si:se])
            gpp_slopes[si:se] = gs
            svh_slopes[si:se] = ss
            gpp_ps[si:se] = gp
            svh_ps[si:se] = sp

        mask = (gpp_ps < 0.05) & (gpp_slopes > 0) & (svh_ps < 0.05) & (svh_slopes > 0)
        n_rec = mask.sum()
        total_recovering += n_rec

        if n_rec > 0:
            rec = chunk[mask][["pixel_id", "latitude", "longitude", "sanlc_2022"] +
                              GPP_COLS + SVH_COLS].copy()
            rec["gpp_slope"] = gpp_slopes[mask]
            rec["svh_slope"] = svh_slopes[mask]
            all_recovering.append(rec)

        del chunk, gpp, svh

    con.close()

    if not all_recovering and not existing_pids:
        print(f"  [WARN] No recovering pixels found for eco_id={eco_id}")
        return None

    if all_recovering:
        df_new = pd.concat(all_recovering, ignore_index=True)
        if existing_pids:
            # Append new to existing
            existing_df = pd.read_parquet(str(out_path))
            df_rec = pd.concat([existing_df, df_new], ignore_index=True)
            df_rec = df_rec.drop_duplicates(subset="pixel_id", keep="first")
        else:
            df_rec = df_new
    else:
        df_rec = pd.read_parquet(str(out_path))

    print(f"\nRecovering pixels: {len(df_rec):,} / {total:,} "
          f"({100*len(df_rec)/max(total,1):.1f}%)")

    df_rec.to_parquet(str(out_path), compression="zstd")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")
    return df_rec


# ======================================================================
# STAGE 1b: EXTRACT AlphaEarth EMBEDDINGS
# ======================================================================

def extract_embeddings(eco_id, df_rec=None, batch_size=2000, max_workers=20,
                       project='ee-gsingh', test_mode=False):
    """Batch-extract AlphaEarth 64D embeddings from GEE.

    Benchmark lessons applied:
      - ee.Geometry() instead of ee.Feature() for grid cell construction
      - ic.reduce(ee.Reducer.first()) instead of .mosaic() to preserve footprint
      - max_workers capped at n_batches to avoid idle threads
      - Larger batch_size (2000) to reduce round-trips
    """
    paths = eco_paths(eco_id)
    out_path = paths["embeddings"]

    print("\n" + "=" * 70)
    print(f"STAGE 1b: Extracting AlphaEarth embeddings (eco_id={eco_id})")
    print("=" * 70)

    if df_rec is None:
        rec_path = paths["recovering"]
        if not rec_path.exists():
            print(f"  [ERROR] {rec_path} not found — run Stage 1a first")
            return
        df_rec = pd.read_parquet(str(rec_path))
        print(f"Loaded {len(df_rec):,} recovering pixels")

    # Skip already-extracted pixel_ids
    if out_path.exists():
        existing_embeds = pd.read_parquet(str(out_path))
        existing_pids = set(existing_embeds["pixel_id"].values)
        before = len(df_rec)
        df_rec = df_rec[~df_rec["pixel_id"].isin(existing_pids)].reset_index(drop=True)
        print(f"  Skipping {before - len(df_rec):,} already-extracted pixels, "
              f"{len(df_rec):,} remaining")
        if df_rec.empty:
            print("  All embeddings already extracted.")
            return

    # Initialize GEE
    try:
        ee.Initialize(project=project,
                      opt_url='https://earthengine-highvolume.googleapis.com')
        print(f"[OK] GEE High Volume Endpoint (project={project})")
    except Exception:
        ee.Initialize(project=project)
        print(f"[OK] GEE Standard Endpoint")

    # AlphaEarth 2022 — use reduce(first) instead of mosaic() to preserve footprint
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    aef_2022 = aef.filterDate("2022-01-01", "2023-01-01").reduce(ee.Reducer.first())

    n_total = len(df_rec)
    n_batches = (n_total + batch_size - 1) // batch_size
    # Cap workers at batch count (benchmark lesson: idle workers are wasted)
    effective_workers = min(max_workers, n_batches)

    if test_mode:
        n_batches = min(n_batches, 3)
        print(f"*** TEST MODE: {n_batches} batches ***")

    print(f"  {n_total:,} points -> {n_batches} batches of ~{batch_size}, "
          f"{effective_workers} workers")

    # DuckDB buffer + checkpoint
    db_path = str(out_path).replace('.parquet', '.duckdb')
    checkpoint_file = str(out_path) + '.checkpoint.json'

    if not out_path.exists() and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    ckpt = CheckpointManager(checkpoint_file)
    db_conn = duckdb.connect(db_path)

    try:
        db_conn.execute("SELECT 1 FROM embeddings LIMIT 1")
        existing_count = db_conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        print(f"  [OK] Resuming — {existing_count:,} rows in buffer")
    except Exception:
        if out_path.exists():
            db_conn.execute(f"CREATE TABLE embeddings AS SELECT * FROM '{out_path}'")
            existing_count = db_conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
            print(f"  [OK] Loaded existing parquet — {existing_count:,} rows")

    lock = Lock()
    successful = 0
    failed_batches = []

    def process_batch(batch_id, batch_df):
        if ckpt.is_done(batch_id):
            return "skipped"

        # Build FeatureCollection — use ee.Geometry.Point directly (benchmark lesson)
        features = []
        for _, row in batch_df.iterrows():
            feat = ee.Feature(
                ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])]),
                {"pid": int(row["pixel_id"])}
            )
            features.append(feat)
        fc = ee.FeatureCollection(features)

        sampled = aef_2022.sampleRegions(
            collection=fc,
            scale=10,
            projection='EPSG:4326',
            tileScale=4,
            geometries=False
        )

        def _compute():
            return ee.data.computeFeatures({
                'expression': sampled,
                'fileFormat': 'PANDAS_DATAFRAME'
            })

        result_df = retry_gee(_compute)

        if result_df is not None and not result_df.empty:
            keep_cols = ["pid"]
            # Handle band naming (reduce(first) appends _first suffix)
            embed_found = [c for c in result_df.columns if c in EMBED_COLS]
            if not embed_found:
                # Map sorted non-metadata columns to A00-A63
                band_cols = sorted([c for c in result_df.columns
                                    if c not in ("pid", "system:index", "geo")])
                if len(band_cols) >= 64:
                    rename_map = {band_cols[i]: EMBED_COLS[i] for i in range(64)}
                    result_df = result_df.rename(columns=rename_map)
                    embed_found = EMBED_COLS[:64]
            keep_cols += embed_found
            result_df = result_df[keep_cols]
            result_df = result_df.rename(columns={"pid": "pixel_id"})

            with lock:
                table_exists = True
                try:
                    db_conn.execute("SELECT 1 FROM embeddings LIMIT 0")
                except duckdb.CatalogException:
                    table_exists = False
                if table_exists:
                    db_conn.execute("INSERT INTO embeddings SELECT * FROM result_df")
                else:
                    db_conn.execute("CREATE TABLE embeddings AS SELECT * FROM result_df")

            ckpt.mark(batch_id)
            return len(result_df)
        return 0

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {}
        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, n_total)
            batch_df = df_rec.iloc[start:end]
            future = executor.submit(process_batch, bi, batch_df)
            futures[future] = bi

        with tqdm(total=n_batches, desc="Extracting embeddings") as pbar:
            for future in as_completed(futures):
                bi = futures[future]
                try:
                    result = future.result()
                    if result != "skipped" and isinstance(result, int):
                        successful += 1
                except Exception as e:
                    failed_batches.append(bi)
                    tqdm.write(f"  [ERROR] Batch {bi}: {str(e)[:80]}")
                pbar.update(1)

    print(f"\n  Successful: {successful}, Failed: {len(failed_batches)}")

    # ── Retry failed batches with smaller batch size + higher tileScale ──
    if failed_batches:
        print(f"\n  Retrying {len(failed_batches)} failed batches "
              f"(batch_size={batch_size // 2}, tileScale=8) ...")
        retry_size = max(batch_size // 2, 500)

        def process_batch_retry(batch_id, batch_df):
            """Retry with smaller sub-batches and higher tileScale."""
            if ckpt.is_done(batch_id):
                return "skipped"

            features = []
            for _, row in batch_df.iterrows():
                feat = ee.Feature(
                    ee.Geometry.Point([float(row["longitude"]), float(row["latitude"])]),
                    {"pid": int(row["pixel_id"])}
                )
                features.append(feat)
            fc = ee.FeatureCollection(features)

            sampled = aef_2022.sampleRegions(
                collection=fc,
                scale=10,
                projection='EPSG:4326',
                tileScale=8,   # higher tileScale for retry
                geometries=False
            )

            def _compute():
                return ee.data.computeFeatures({
                    'expression': sampled,
                    'fileFormat': 'PANDAS_DATAFRAME'
                })

            result_df = retry_gee(_compute)

            if result_df is not None and not result_df.empty:
                keep_cols = ["pid"]
                embed_found = [c for c in result_df.columns if c in EMBED_COLS]
                if not embed_found:
                    band_cols = sorted([c for c in result_df.columns
                                        if c not in ("pid", "system:index", "geo")])
                    if len(band_cols) >= 64:
                        rename_map = {band_cols[i]: EMBED_COLS[i] for i in range(64)}
                        result_df = result_df.rename(columns=rename_map)
                        embed_found = EMBED_COLS[:64]
                keep_cols += embed_found
                result_df = result_df[keep_cols]
                result_df = result_df.rename(columns={"pid": "pixel_id"})

                with lock:
                    try:
                        db_conn.execute("INSERT INTO embeddings SELECT * FROM result_df")
                    except (duckdb.CatalogException, duckdb.Error):
                        db_conn.execute("CREATE TABLE embeddings AS SELECT * FROM result_df")

                ckpt.mark(batch_id)
                return len(result_df)
            return 0

        still_failed = []
        retry_success = 0
        # Split failed batches into smaller sub-batches
        retry_items = []
        for bi in failed_batches:
            start = bi * batch_size
            end = min(start + batch_size, n_total)
            batch_df = df_rec.iloc[start:end]
            # Split into sub-batches of retry_size
            for sub_start in range(0, len(batch_df), retry_size):
                sub_df = batch_df.iloc[sub_start:sub_start + retry_size]
                retry_items.append((f"{bi}_{sub_start}", bi, sub_df))

        retry_workers = min(max_workers, len(retry_items))
        with ThreadPoolExecutor(max_workers=retry_workers) as executor:
            retry_futures = {}
            for sub_id, orig_bi, sub_df in retry_items:
                future = executor.submit(process_batch_retry, sub_id, sub_df)
                retry_futures[future] = (sub_id, orig_bi)

            with tqdm(total=len(retry_items), desc="Retrying failed") as pbar:
                for future in as_completed(retry_futures):
                    sub_id, orig_bi = retry_futures[future]
                    try:
                        result = future.result()
                        if result != "skipped" and isinstance(result, int):
                            retry_success += 1
                    except Exception as e:
                        still_failed.append(sub_id)
                        tqdm.write(f"  [RETRY ERROR] Batch {sub_id}: {str(e)[:80]}")
                    pbar.update(1)

        print(f"  Retry results: {retry_success} succeeded, "
              f"{len(still_failed)} still failed")

    try:
        row_count = db_conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        print(f"  Buffer: {row_count:,} rows")

        embed_cast = ", ".join(f'CAST("{c}" AS FLOAT) AS "{c}"' for c in EMBED_COLS)
        # Deduplicate on pixel_id
        db_conn.execute(f"""
            COPY (
                SELECT DISTINCT ON (pixel_id)
                    CAST(pixel_id AS BIGINT) AS pixel_id, {embed_cast}
                FROM embeddings
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """)
        final_count = duckdb.sql(f"SELECT count(*) FROM '{out_path}'").fetchone()[0]
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[OK] Saved: {out_path}")
        print(f"  Rows: {final_count:,} (deduped), Size: {size_mb:.1f} MB")
    except Exception as e:
        print(f"  [ERROR] Export failed: {e}")
    finally:
        db_conn.close()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass


# ======================================================================
# STAGE 2: COMPUTE RECOVERY SCORES
# ======================================================================

def load_natural_reference(eco_id):
    """Load natural ref with GPP/SVH + embeddings, deduplicated on pixel_id."""
    gpp_sel = ", ".join(f"g.{c}" for c in GPP_COLS)
    svh_sel = ", ".join(f"g.{c}" for c in SVH_COLS)
    embed_sel = ", ".join(f"n.{c}" for c in EMBED_COLS)

    sql = f"""
        SELECT DISTINCT ON (g.pixel_id)
               g.latitude, g.longitude,
               {gpp_sel}, {svh_sel}, {embed_sel}
        FROM '{NAT_SUBSET}' n
        JOIN '{INDICES_GPP}' g
          ON n.id = split_part(g.pixel_id, '_', 1)
        WHERE g.eco_id = {eco_id}
    """
    return duckdb.sql(sql).df()


def load_transformed_reference(eco_id):
    """Load non-natural reference pixels for comparison baseline."""
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)

    sql = f"""
        SELECT DISTINCT ON (pixel_id)
               latitude, longitude, {gpp_sel}, {svh_sel}
        FROM '{INDICES_GPP}'
        WHERE eco_id = {eco_id}
          AND split_part(pixel_id, '_', 1) NOT IN (
              SELECT id FROM '{NAT_SUBSET}'
          )
    """
    return duckdb.sql(sql).df()


def build_natural_baselines(nat_df, n_cosine_baseline=500):
    """Pre-compute baselines for percentile normalisation."""
    print("  Building natural baselines ...")
    nat_gpp = nat_df[LATE_GPP].values.astype(np.float32).mean(axis=1)
    nat_svh = nat_df[LATE_SVH].values.astype(np.float32).mean(axis=1)
    nat_embeds = nat_df[EMBED_COLS].values.astype(np.float32)
    nat_coords = np.radians(nat_df[["latitude", "longitude"]].values.astype(np.float64))
    tree = BallTree(nat_coords, metric="haversine")

    eco_gpp_mean = nat_gpp.mean()
    eco_svh_mean = nat_svh.mean()
    nat_gpp_ratios = nat_gpp / (eco_gpp_mean + 1e-9)
    nat_svh_ratios = nat_svh / (eco_svh_mean + 1e-9)

    rng = np.random.RandomState(42)
    n = len(nat_embeds)

    sample_n = min(n_cosine_baseline, n)
    sample_idx = rng.choice(n, sample_n, replace=False)
    eco_baseline_sims = cosine_similarity(nat_embeds[sample_idx], nat_embeds).mean(axis=1)

    local_n = min(300, n)
    local_idx = rng.choice(n, local_n, replace=False)
    local_baseline_sims = np.empty(local_n, dtype=np.float32)
    for i, idx in enumerate(local_idx):
        _, nn_idx = tree.query(nat_coords[idx:idx+1], k=KNN_K + 1)
        nn_idx = nn_idx[0]
        nn_idx = nn_idx[nn_idx != idx][:KNN_K]
        s = cosine_similarity(nat_embeds[idx:idx+1], nat_embeds[nn_idx])[0].mean()
        local_baseline_sims[i] = s

    return {
        "nat_gpp": nat_gpp,
        "nat_svh": nat_svh,
        "nat_embeds": nat_embeds,
        "nat_coords": nat_coords,
        "tree": tree,
        "eco_gpp_mean": eco_gpp_mean,
        "eco_svh_mean": eco_svh_mean,
        "nat_gpp_ratios": nat_gpp_ratios,
        "nat_svh_ratios": nat_svh_ratios,
        "eco_baseline_sims": eco_baseline_sims,
        "local_baseline_sims": local_baseline_sims,
    }


def score_batch(batch_gpp, batch_svh, batch_embeds, batch_coords_rad, baselines):
    """Score a batch of recovering pixels."""
    bl = baselines
    N = len(batch_gpp)

    test_gpp = batch_gpp.mean(axis=1)
    test_svh = batch_svh.mean(axis=1)

    # Metric A: percentile within ecoregion natural
    a_gpp = np.array([percentileofscore(bl["nat_gpp"], v) for v in test_gpp])
    a_svh = np.array([percentileofscore(bl["nat_svh"], v) for v in test_svh])

    # Metric B: KNN ratio
    dists, nn_idx = bl["tree"].query(batch_coords_rad, k=KNN_K)
    knn_gpp = bl["nat_gpp"][nn_idx].mean(axis=1)
    knn_svh = bl["nat_svh"][nn_idx].mean(axis=1)
    b_gpp_ratio = test_gpp / (knn_gpp + 1e-9)
    b_svh_ratio = test_svh / (knn_svh + 1e-9)
    b_gpp = np.array([percentileofscore(bl["nat_gpp_ratios"], v) for v in b_gpp_ratio])
    b_svh = np.array([percentileofscore(bl["nat_svh_ratios"], v) for v in b_svh_ratio])

    # Metric C_eco: cosine sim to all ecoregion natural
    cos_eco = cosine_similarity(batch_embeds, bl["nat_embeds"]).mean(axis=1)
    c_eco = np.array([percentileofscore(bl["eco_baseline_sims"], v) for v in cos_eco])

    # Metric C_local: cosine sim to own KNN
    cos_local = np.empty(N, dtype=np.float32)
    for i in range(N):
        knn_embeds = bl["nat_embeds"][nn_idx[i]]
        cos_local[i] = cosine_similarity(batch_embeds[i:i+1], knn_embeds)[0].mean()
    c_local = np.array([percentileofscore(bl["local_baseline_sims"], v) for v in cos_local])

    # Composite
    composite_gpp = (a_gpp + b_gpp + c_eco + c_local) / 4.0
    composite_svh = (a_svh + b_svh + c_eco + c_local) / 4.0
    composite = (composite_gpp + composite_svh) / 2.0

    return {
        "a_gpp_pctl": a_gpp.astype(np.float32),
        "a_svh_pctl": a_svh.astype(np.float32),
        "b_gpp_pctl": b_gpp.astype(np.float32),
        "b_svh_pctl": b_svh.astype(np.float32),
        "b_gpp_ratio": b_gpp_ratio.astype(np.float32),
        "b_svh_ratio": b_svh_ratio.astype(np.float32),
        "c_eco_sim": cos_eco.astype(np.float32),
        "c_eco_pctl": c_eco.astype(np.float32),
        "c_local_sim": cos_local.astype(np.float32),
        "c_local_pctl": c_local.astype(np.float32),
        "composite_gpp": composite_gpp.astype(np.float32),
        "composite_svh": composite_svh.astype(np.float32),
        "recovery_score": composite.astype(np.float32),
    }


def compute_scores(eco_id, test_mode=False):
    """Load recovering pixels + embeddings, compute all metrics, save results."""
    paths = eco_paths(eco_id)

    print("\n" + "=" * 70)
    print(f"STAGE 2: Computing recovery scores (eco_id={eco_id})")
    print("=" * 70)

    if not paths["recovering"].exists():
        print(f"  [ERROR] {paths['recovering']} not found — run Stage 1a first")
        return
    if not paths["embeddings"].exists():
        print(f"  [ERROR] {paths['embeddings']} not found — run Stage 1b first")
        return

    rec = pd.read_parquet(str(paths["recovering"]))
    print(f"Loaded {len(rec):,} recovering pixels")

    embeds = pd.read_parquet(str(paths["embeddings"]))
    print(f"Loaded {len(embeds):,} embeddings")

    rec["pixel_id"] = rec["pixel_id"].astype(np.int64)
    embeds["pixel_id"] = embeds["pixel_id"].astype(np.int64)
    merged = rec.merge(embeds, on="pixel_id", how="inner")
    merged = merged.drop_duplicates(subset="pixel_id", keep="first")
    print(f"Matched: {len(merged):,} pixels with embeddings")

    if len(merged) == 0:
        print("  [ERROR] No pixels matched — check pixel_id formats")
        return

    if test_mode:
        merged = merged.head(5000)
        print(f"*** TEST MODE: scoring {len(merged):,} pixels ***")

    # Load natural reference
    print("\nLoading natural reference ...")
    nat_df = load_natural_reference(eco_id)
    print(f"  Natural reference: {len(nat_df):,} pixels (eco_id={eco_id})")

    if len(nat_df) < MIN_NATURAL_POINTS:
        print(f"  [WARN] Only {len(nat_df)} natural ref pixels — "
              f"minimum is {MIN_NATURAL_POINTS}. Skipping.")
        return

    baselines = build_natural_baselines(nat_df)

    # Score in chunks
    chunk_size = 2000
    n_chunks = (len(merged) + chunk_size - 1) // chunk_size
    all_results = []

    print(f"\nScoring {len(merged):,} pixels in {n_chunks} chunks ...")
    for ci in tqdm(range(n_chunks), desc="Scoring"):
        start = ci * chunk_size
        end = min(start + chunk_size, len(merged))
        chunk = merged.iloc[start:end]

        batch_gpp = chunk[LATE_GPP].values.astype(np.float32)
        batch_svh = chunk[LATE_SVH].values.astype(np.float32)
        batch_embeds = chunk[EMBED_COLS].values.astype(np.float32)
        batch_coords = np.radians(
            chunk[["latitude", "longitude"]].values.astype(np.float64))

        scores = score_batch(batch_gpp, batch_svh, batch_embeds, batch_coords, baselines)

        chunk_result = pd.DataFrame({
            "pixel_id": chunk["pixel_id"].values,
            "latitude": chunk["latitude"].values.astype(np.float32),
            "longitude": chunk["longitude"].values.astype(np.float32),
            "sanlc_2022": chunk["sanlc_2022"].values,
            "gpp_slope": chunk["gpp_slope"].values,
            "svh_slope": chunk["svh_slope"].values,
            **scores,
        })
        all_results.append(chunk_result)

    df_out = pd.concat(all_results, ignore_index=True)
    df_out.to_parquet(str(paths["scores"]), compression="zstd")
    size_mb = os.path.getsize(paths["scores"]) / (1024 * 1024)
    print(f"\n[OK] Saved: {paths['scores']}")
    print(f"  Rows: {len(df_out):,}, Size: {size_mb:.1f} MB")

    # Summary
    print(f"\n  {'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print("  " + "-" * 50)
    for col in ["a_gpp_pctl", "a_svh_pctl", "b_gpp_pctl", "b_svh_pctl",
                "c_eco_pctl", "c_local_pctl", "recovery_score"]:
        v = df_out[col]
        print(f"  {col:<20} {v.mean():>8.1f} {v.median():>8.1f} {v.std():>8.1f}")


# ======================================================================
# MAIN
# ======================================================================

def run_ecoregion(eco_id, stage, test_mode, batch_size, max_workers, project):
    """Run pipeline for a single ecoregion."""
    print("\n" + "#" * 70)
    print(f"# ECOREGION {eco_id}")
    print("#" * 70)

    if stage in ("extract", "all"):
        df_rec = identify_recovering_pixels(eco_id, test_mode=test_mode)
        if df_rec is not None:
            extract_embeddings(eco_id, df_rec, batch_size=batch_size,
                             max_workers=max_workers, project=project,
                             test_mode=test_mode)

    if stage in ("score", "all"):
        compute_scores(eco_id, test_mode=test_mode)


def main():
    parser = argparse.ArgumentParser(
        description="Scale recovery analysis to all SA ecoregions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eco_id", type=int, help="Single ecoregion ID")
    group.add_argument("--all", action="store_true",
                       help="Run all ecoregions (skips eco_id=81)")
    parser.add_argument("--stage", choices=["extract", "score", "all"],
                        default="all", help="Pipeline stage")
    parser.add_argument("--test_mode", action="store_true",
                        help="Small sample for testing")
    parser.add_argument("--batch_size", type=int, default=2000,
                        help="GEE extraction batch size")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Max parallel GEE workers")
    parser.add_argument("--project", type=str, default="ee-gsingh",
                        help="GEE project ID")
    args = parser.parse_args()

    # Show inventory
    print("Ecoregion inventory:")
    inv = get_ecoregion_inventory()
    print(f"{'eco_id':>8} {'n_ag':>12} {'n_natural':>12} {'status':>20}")
    print("-" * 55)
    for _, row in inv.iterrows():
        eid = int(row["eco_id"])
        status = "SKIP (done)" if eid == SKIP_ECO else (
            "ready" if row["n_natural"] >= MIN_NATURAL_POINTS else
            f"need ref (n={int(row['n_natural'])})")
        print(f"{eid:>8} {int(row['n_ag']):>12,} {int(row['n_natural']):>12,} "
              f"{status:>20}")
    print()

    if args.eco_id:
        if args.eco_id == SKIP_ECO:
            print(f"eco_id={SKIP_ECO} is already complete. Exiting.")
            return
        run_ecoregion(args.eco_id, args.stage, args.test_mode,
                     args.batch_size, args.max_workers, args.project)
    else:
        # All ecoregions, sorted by n_ag descending (largest first)
        eco_ids = [int(row["eco_id"]) for _, row in inv.iterrows()
                   if int(row["eco_id"]) != SKIP_ECO]
        print(f"Processing {len(eco_ids)} ecoregions: {eco_ids}\n")
        for eco_id in eco_ids:
            run_ecoregion(eco_id, args.stage, args.test_mode,
                         args.batch_size, args.max_workers, args.project)


if __name__ == "__main__":
    main()
