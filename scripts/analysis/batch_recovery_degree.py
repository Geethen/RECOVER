"""
Batch recovery degree estimation for all recovering pixels in target ecoregion(s).

Two-stage pipeline:
  Stage 1 (extract): Identify recovering pixels via MK+TS, batch-extract
      AlphaEarth 64D embeddings from GEE using high-volume endpoint.
  Stage 2 (score): Compute Metrics A, B, C_eco, C_local for all recovering
      pixels against ecoregion-specific natural reference.

Usage:
    # Single ecoregion (full pipeline):
    python scripts/analysis/batch_recovery_degree.py --eco_id 40

    # All ecoregions with abandoned-ag data:
    python scripts/analysis/batch_recovery_degree.py --all

    # Stage 1 only (identify + extract):
    python scripts/analysis/batch_recovery_degree.py --eco_id 40 --stage extract

    # Stage 2 only (score, requires stage 1 output):
    python scripts/analysis/batch_recovery_degree.py --eco_id 40 --stage score

    # Test mode (small sample):
    python scripts/analysis/batch_recovery_degree.py --eco_id 40 --test_mode
"""
import sys
import os
import ee
import time
import json
import argparse
import numpy as np
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

# ── paths & constants ──────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"
ABANDONED_AG = DATA_DIR / "abandoned_ag_gpp_2000_2022_SA.parquet"
INDICES_GPP = DATA_DIR / "indices_gpp_svh_2000_2022.parquet"
NAT_SUBSET = DATA_DIR / "dfsubsetNatural.parquet"

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

# All SA ecoregions with abandoned-ag pixels
ALL_ECOS = [81, 41, 38, 40, 97, 48, 89, 90, 101, 88, 110,
            16, 102, 19, 94, 15, 116, 65]


def output_paths(eco_id):
    return {
        "recovering": DATA_DIR / f"recovering_eco{eco_id}.parquet",
        "embeddings": DATA_DIR / f"recovering_eco{eco_id}_embeddings.parquet",
        "scores": DATA_DIR / f"recovery_scores_eco{eco_id}.parquet",
    }


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


# ======================================================================
# STAGE 1: IDENTIFY RECOVERING PIXELS + EXTRACT EMBEDDINGS
# ======================================================================

def identify_recovering_pixels(eco_id, chunk_size=50000, test_mode=False):
    """Stream abandoned_ag, compute MK+TS, save recovering pixels."""
    paths = output_paths(eco_id)
    recovering_pq = paths["recovering"]

    print("=" * 70)
    print(f"STAGE 1a: Identifying recovering pixels (eco_id={eco_id})")
    print("=" * 70)

    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)

    total = duckdb.sql(f"""
        SELECT count(*) FROM '{ABANDONED_AG}' WHERE eco_id = {eco_id}
    """).fetchone()[0]
    print(f"Total abandoned-ag pixels (eco_id={eco_id}): {total:,}")

    if total == 0:
        print(f"  [SKIP] No abandoned-ag pixels for eco_id={eco_id}")
        return None

    if test_mode:
        total = min(total, 100000)
        print(f"*** TEST MODE: processing {total:,} pixels ***")

    n_chunks = (total + chunk_size - 1) // chunk_size
    all_recovering = []

    con = duckdb.connect()
    for ci in tqdm(range(n_chunks), desc="Scanning chunks"):
        offset = ci * chunk_size
        limit = min(chunk_size, total - offset)

        sql = f"""
            SELECT pixel_id, latitude, longitude, eco_id, sanlc_2022,
                   {gpp_sel}, {svh_sel}
            FROM '{ABANDONED_AG}'
            WHERE eco_id = {eco_id}
            LIMIT {limit} OFFSET {offset}
        """
        chunk = con.execute(sql).df()
        if chunk.empty:
            break

        gpp = chunk[GPP_COLS].values.astype(np.float32)
        svh = chunk[SVH_COLS].values.astype(np.float32)

        # Process in sub-batches for MK+TS memory
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

        # Both recovering: significant positive trend in GPP AND SVH
        mask = (gpp_ps < 0.05) & (gpp_slopes > 0) & (svh_ps < 0.05) & (svh_slopes > 0)

        if mask.sum() > 0:
            rec = chunk[mask][["pixel_id", "latitude", "longitude", "sanlc_2022"] +
                              GPP_COLS + SVH_COLS].copy()
            rec["gpp_slope"] = gpp_slopes[mask]
            rec["svh_slope"] = svh_slopes[mask]
            all_recovering.append(rec)

        del chunk, gpp, svh

    con.close()

    if not all_recovering:
        print(f"  [WARN] No recovering pixels found for eco_id={eco_id}")
        return None

    import pandas as pd
    df_rec = pd.concat(all_recovering, ignore_index=True)
    print(f"\nRecovering pixels: {len(df_rec):,} / {total:,} "
          f"({100*len(df_rec)/total:.1f}%)")

    df_rec.to_parquet(str(recovering_pq), compression="zstd")
    size_mb = os.path.getsize(recovering_pq) / (1024 * 1024)
    print(f"Saved: {recovering_pq} ({size_mb:.1f} MB)")
    return df_rec


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
    """Retry wrapper for GEE calls."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = backoff ** (attempt + 1)
                time.sleep(wait)
            else:
                raise


def extract_embeddings(eco_id, df_rec=None, batch_size=2000, max_workers=20,
                       project='ee-gsingh', test_mode=False):
    """Batch-extract AlphaEarth 64D embeddings from GEE for recovering pixels."""
    paths = output_paths(eco_id)
    recovering_pq = paths["recovering"]
    embeddings_pq = paths["embeddings"]

    print("\n" + "=" * 70)
    print(f"STAGE 1b: Extracting AlphaEarth embeddings (eco_id={eco_id})")
    print("=" * 70)

    if df_rec is None:
        import pandas as pd
        df_rec = pd.read_parquet(str(recovering_pq))
        print(f"Loaded {len(df_rec):,} recovering pixels from {recovering_pq}")

    # Initialize GEE
    try:
        ee.Initialize(project=project,
                      opt_url='https://earthengine-highvolume.googleapis.com')
        print(f"[OK] GEE High Volume Endpoint (project={project})")
    except Exception:
        ee.Initialize(project=project)
        print(f"[OK] GEE Standard Endpoint")

    # AlphaEarth 2022 — reduce(first) not mosaic() (benchmark gotcha #4)
    aef = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
    aef_2022 = aef.filterDate("2022-01-01", "2023-01-01").reduce(
        ee.Reducer.first()).regexpRename('_first$', '')

    n_total = len(df_rec)
    n_batches = (n_total + batch_size - 1) // batch_size

    if test_mode:
        n_batches = min(n_batches, 3)
        print(f"*** TEST MODE: {n_batches} batches ***")

    print(f"  {n_total:,} points -> {n_batches} batches of ~{batch_size}")

    # DuckDB buffer + checkpoint
    db_path = str(embeddings_pq).replace('.parquet', '.duckdb')
    checkpoint_file = str(embeddings_pq) + '.checkpoint.json'

    if not os.path.exists(str(embeddings_pq)) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    ckpt = CheckpointManager(checkpoint_file)
    db_conn = duckdb.connect(db_path)

    try:
        db_conn.execute("SELECT 1 FROM embeddings LIMIT 1")
        existing = db_conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        print(f"  [OK] Resuming — {existing:,} rows in buffer")
    except Exception:
        if os.path.exists(str(embeddings_pq)):
            db_conn.execute(f"CREATE TABLE embeddings AS SELECT * FROM '{embeddings_pq}'")
            existing = db_conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
            print(f"  [OK] Loaded existing parquet — {existing:,} rows")

    lock = Lock()
    successful = 0
    failed_batches = []

    def process_batch(batch_id, batch_df):
        """Extract embeddings for a batch of points."""
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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
    try:
        row_count = db_conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        print(f"  Buffer: {row_count:,} rows")

        embed_cast = ", ".join(f'CAST("{c}" AS FLOAT) AS "{c}"' for c in EMBED_COLS)
        db_conn.execute(f"""
            COPY (
                SELECT CAST(pixel_id AS BIGINT) AS pixel_id, {embed_cast}
                FROM embeddings
            ) TO '{embeddings_pq}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """)
        print(f"[OK] Saved: {embeddings_pq}")
        size_mb = os.path.getsize(embeddings_pq) / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB, Rows: {row_count:,}")
    except Exception as e:
        print(f"  [ERROR] Export failed: {e}")
    finally:
        db_conn.close()
        if os.path.exists(db_path):
            try: os.remove(db_path)
            except: pass


# ======================================================================
# STAGE 2: COMPUTE RECOVERY SCORES
# ======================================================================

def load_natural_reference(eco_id):
    """Load natural reference with GPP/SVH + embeddings for a given ecoregion.

    eco81: uses original dfsubsetNatural + indices_gpp_svh join.
    Other ecos: uses ref_samples_eco{id}.parquet (natural=1).
    """
    gpp_sel = ", ".join(GPP_COLS)
    svh_sel = ", ".join(SVH_COLS)
    embed_sel = ", ".join(EMBED_COLS)

    if eco_id == 81:
        # Original approach: join dfsubsetNatural (embeddings) with indices_gpp_svh (GPP/SVH)
        gpp_g = ", ".join(f"g.{c}" for c in GPP_COLS)
        svh_g = ", ".join(f"g.{c}" for c in SVH_COLS)
        embed_n = ", ".join(f"n.{c}" for c in EMBED_COLS)
        sql = f"""
            SELECT g.latitude, g.longitude,
                   {gpp_g}, {svh_g}, {embed_n}
            FROM '{NAT_SUBSET}' n
            JOIN '{INDICES_GPP}' g
              ON n.id = split_part(g.pixel_id, '_', 1)
            WHERE g.eco_id = {eco_id}
        """
    else:
        # New ecoregions: ref_samples_eco{id}.parquet has everything
        ref_path = DATA_DIR / f"ref_samples_eco{eco_id}.parquet"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"No reference data for eco_id={eco_id}. "
                f"Run sample_reference_points.py --eco_id {eco_id} first.")
        sql = f"""
            SELECT latitude, longitude,
                   {gpp_sel}, {svh_sel}, {embed_sel}
            FROM '{ref_path}'
            WHERE "natural" = 1
        """

    nat_df = duckdb.sql(sql).df()
    # Drop rows with NaN in critical columns
    nat_df = nat_df.dropna(subset=LATE_GPP + LATE_SVH + EMBED_COLS[:5])
    return nat_df


def build_natural_baselines(nat_df, n_cosine_baseline=500):
    """Pre-compute baselines for percentile normalisation (done once)."""
    print("  Building natural baselines ...")
    nat_gpp = nat_df[LATE_GPP].values.astype(np.float32).mean(axis=1)
    nat_svh = nat_df[LATE_SVH].values.astype(np.float32).mean(axis=1)
    nat_embeds = nat_df[EMBED_COLS].values.astype(np.float32)
    nat_coords = np.radians(nat_df[["latitude", "longitude"]].values.astype(np.float64))
    tree = BallTree(nat_coords, metric="haversine")

    # Metric B baseline: each natural pixel's ratio to eco mean
    eco_gpp_mean = nat_gpp.mean()
    eco_svh_mean = nat_svh.mean()
    nat_gpp_ratios = nat_gpp / (eco_gpp_mean + 1e-9)
    nat_svh_ratios = nat_svh / (eco_svh_mean + 1e-9)

    # Metric C_eco baseline: 500 natural pixels' mean cosine sim to all natural
    rng = np.random.RandomState(42)
    n = len(nat_embeds)
    sample_n = min(n_cosine_baseline, n)
    sample_idx = rng.choice(n, sample_n, replace=False)
    eco_baseline_sims = cosine_similarity(nat_embeds[sample_idx], nat_embeds).mean(axis=1)

    # Metric C_local baseline: 300 natural pixels' mean cosine sim to own KNN
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
    """Score a batch of recovering pixels. All arrays shape (N, ...)."""
    bl = baselines
    N = len(batch_gpp)

    # Late-period means
    test_gpp = batch_gpp.mean(axis=1)
    test_svh = batch_svh.mean(axis=1)

    # ── Metric A: percentile within ecoregion natural ──
    a_gpp = np.array([percentileofscore(bl["nat_gpp"], v) for v in test_gpp])
    a_svh = np.array([percentileofscore(bl["nat_svh"], v) for v in test_svh])

    # ── Metric B: KNN ratio ──
    dists, nn_idx = bl["tree"].query(batch_coords_rad, k=KNN_K)
    knn_gpp = bl["nat_gpp"][nn_idx].mean(axis=1)
    knn_svh = bl["nat_svh"][nn_idx].mean(axis=1)
    b_gpp_ratio = test_gpp / (knn_gpp + 1e-9)
    b_svh_ratio = test_svh / (knn_svh + 1e-9)
    b_gpp = np.array([percentileofscore(bl["nat_gpp_ratios"], v) for v in b_gpp_ratio])
    b_svh = np.array([percentileofscore(bl["nat_svh_ratios"], v) for v in b_svh_ratio])

    # ── Metric C_eco: cosine sim to all ecoregion natural ──
    cos_eco = cosine_similarity(batch_embeds, bl["nat_embeds"]).mean(axis=1)
    c_eco = np.array([percentileofscore(bl["eco_baseline_sims"], v) for v in cos_eco])

    # ── Metric C_local: cosine sim to own KNN ──
    cos_local = np.empty(N, dtype=np.float32)
    for i in range(N):
        knn_embeds = bl["nat_embeds"][nn_idx[i]]
        cos_local[i] = cosine_similarity(batch_embeds[i:i+1], knn_embeds)[0].mean()
    c_local = np.array([percentileofscore(bl["local_baseline_sims"], v) for v in cos_local])

    # ── Composite ──
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
    import pandas as pd

    paths = output_paths(eco_id)
    recovering_pq = paths["recovering"]
    embeddings_pq = paths["embeddings"]
    results_pq = paths["scores"]

    print("\n" + "=" * 70)
    print(f"STAGE 2: Computing recovery scores (eco_id={eco_id})")
    print("=" * 70)

    rec = pd.read_parquet(str(recovering_pq))
    print(f"Loaded {len(rec):,} recovering pixels")

    embeds = pd.read_parquet(str(embeddings_pq))
    print(f"Loaded {len(embeds):,} embeddings")

    rec["pixel_id"] = rec["pixel_id"].astype(np.int64)
    embeds["pixel_id"] = embeds["pixel_id"].astype(np.int64)
    merged = rec.merge(embeds, on="pixel_id", how="inner")
    print(f"Matched: {len(merged):,} pixels with embeddings")

    if len(merged) == 0:
        raise RuntimeError(
            "No pixels matched — check pixel_id formats in "
            "recovering + embeddings files")

    if test_mode:
        merged = merged.head(5000)
        print(f"*** TEST MODE: scoring {len(merged):,} pixels ***")

    # Load natural reference
    print(f"\nLoading natural reference for eco_id={eco_id} ...")
    nat_df = load_natural_reference(eco_id)
    print(f"  Natural reference: {len(nat_df):,} pixels")

    if len(nat_df) < KNN_K:
        print(f"  [WARN] Only {len(nat_df)} natural pixels — "
              f"insufficient for KNN_K={KNN_K}. Skipping.")
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

        scores = score_batch(
            batch_gpp, batch_svh, batch_embeds, batch_coords, baselines)

        chunk_result = pd.DataFrame({
            "pixel_id": chunk["pixel_id"].values,
            "latitude": chunk["latitude"].values.astype(np.float32),
            "longitude": chunk["longitude"].values.astype(np.float32),
            "eco_id": eco_id,
            "sanlc_2022": chunk["sanlc_2022"].values,
            "gpp_slope": chunk["gpp_slope"].values,
            "svh_slope": chunk["svh_slope"].values,
            **scores,
        })
        all_results.append(chunk_result)

    df_out = pd.concat(all_results, ignore_index=True)

    df_out.to_parquet(str(results_pq), compression="zstd")
    size_mb = os.path.getsize(results_pq) / (1024 * 1024)
    print(f"\n[OK] Saved: {results_pq}")
    print(f"  Rows: {len(df_out):,}, Size: {size_mb:.1f} MB")

    # Summary stats
    print(f"\n  {'Metric':<20} {'Mean':>8} {'Median':>8} {'Std':>8}")
    print("  " + "-" * 50)
    for col in ["a_gpp_pctl", "a_svh_pctl", "b_gpp_pctl", "b_svh_pctl",
                "c_eco_pctl", "c_local_pctl", "recovery_score"]:
        v = df_out[col]
        print(f"  {col:<20} {v.mean():>8.1f} {v.median():>8.1f} {v.std():>8.1f}")


def run_ecoregion(eco_id, stage="all", test_mode=False, batch_size=2000,
                  max_workers=20, project="ee-gsingh"):
    """Run the full pipeline for a single ecoregion."""
    paths = output_paths(eco_id)

    print(f"\n{'#' * 70}")
    print(f"# ECOREGION {eco_id}")
    print(f"{'#' * 70}")

    if stage in ("extract", "all"):
        # Skip if recovering pixels already exist
        if paths["recovering"].exists() and paths["embeddings"].exists():
            n_rec = duckdb.sql(
                f"SELECT count(*) FROM '{paths['recovering']}'").fetchone()[0]
            n_emb = duckdb.sql(
                f"SELECT count(*) FROM '{paths['embeddings']}'").fetchone()[0]
            print(f"  [SKIP] Stage 1 complete: {n_rec:,} recovering, "
                  f"{n_emb:,} embeddings")
        else:
            df_rec = identify_recovering_pixels(eco_id, test_mode=test_mode)
            if df_rec is not None:
                extract_embeddings(eco_id, df_rec, batch_size=batch_size,
                                   max_workers=max_workers, project=project,
                                   test_mode=test_mode)

    if stage in ("score", "all"):
        if paths["scores"].exists():
            n_scores = duckdb.sql(
                f"SELECT count(*) FROM '{paths['scores']}'").fetchone()[0]
            print(f"  [SKIP] Stage 2 complete: {n_scores:,} scores")
        elif not paths["recovering"].exists() or not paths["embeddings"].exists():
            print(f"  [SKIP] Stage 1 output missing — run extract first")
        else:
            compute_scores(eco_id, test_mode=test_mode)


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch recovery degree estimation")
    parser.add_argument("--eco_id", type=int, default=None,
                        help="Ecoregion ID to process")
    parser.add_argument("--all", action="store_true",
                        help="Process all ecoregions with abandoned-ag data")
    parser.add_argument("--stage", choices=["extract", "score", "all"],
                        default="all", help="Pipeline stage to run")
    parser.add_argument("--test_mode", action="store_true",
                        help="Small sample for testing")
    parser.add_argument("--batch_size", type=int, default=2000,
                        help="GEE extraction batch size")
    parser.add_argument("--max_workers", type=int, default=20,
                        help="Parallel GEE workers")
    parser.add_argument("--project", type=str, default="ee-gsingh",
                        help="GEE project ID")
    args = parser.parse_args()

    if args.all:
        eco_ids = ALL_ECOS
    elif args.eco_id is not None:
        eco_ids = [args.eco_id]
    else:
        parser.error("Specify --eco_id or --all")

    for eco_id in eco_ids:
        run_ecoregion(eco_id, stage=args.stage, test_mode=args.test_mode,
                      batch_size=args.batch_size, max_workers=args.max_workers,
                      project=args.project)


if __name__ == "__main__":
    main()
