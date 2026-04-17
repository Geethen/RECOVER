"""
Re-score recovery metrics stratified by bioregion instead of ecoregion.

Loads recovering pixels and natural reference samples per-bioregion (lazy)
to avoid memory issues, then recomputes metrics A, B, C_eco, C_local.

Prerequisite: run extract_bioregion_ids.py first to add bioregion_id.

Usage:
    python scripts/analysis/score_by_bioregion.py
    python scripts/analysis/score_by_bioregion.py --bio_id 10
    python scripts/analysis/score_by_bioregion.py --test_mode
"""
import sys
import os
import gc
import json
import argparse
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from scipy.stats import percentileofscore
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"

ALL_ECOS = [81, 41, 38, 40, 97, 48, 89, 90, 101, 88, 110,
            16, 102, 19, 94, 15, 116, 65]

YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]
LATE_YEARS = list(range(2018, 2023))
LATE_GPP = [f"GPP_{y}" for y in LATE_YEARS]
LATE_SVH = [f"SVH_{y}" for y in LATE_YEARS]
EMBED_COLS = [f"A{i:02d}" for i in range(64)]
KNN_K = 10
MIN_NATURAL_POINTS = 30

BIO_META_PATH = Path(r"c:\Users\coach\myfiles\postdoc2\data\bioregions_30m.json")


def load_bioregion_names():
    """Load bioregion ID -> name mapping from metadata JSON."""
    with open(BIO_META_PATH) as f:
        meta = json.load(f)
    names = {}
    for entry in meta["bioregions"]:
        bid = entry["id"]
        name = entry["name"]
        if bid not in names or name != "None":
            names[bid] = name
    return names


def discover_bioregions():
    """Scan score files to find which bioregion IDs exist and their counts.
    Returns dict: {bio_id: {"n_rec": count, "eco_files": [eco_ids]}}
    """
    con = duckdb.connect()
    bio_info = {}

    for eco_id in ALL_ECOS:
        score_path = DATA_DIR / f"recovery_scores_eco{eco_id}.parquet"
        if not score_path.exists():
            continue
        sp = str(score_path).replace("\\", "/")
        try:
            rows = con.sql(f"""
                SELECT bioregion_id, COUNT(*) as cnt
                FROM '{sp}'
                WHERE bioregion_id > 0
                GROUP BY bioregion_id
            """).fetchall()
        except Exception:
            continue

        for bio_id, cnt in rows:
            bio_id = int(bio_id)
            if bio_id not in bio_info:
                bio_info[bio_id] = {"n_rec": 0, "eco_files": []}
            bio_info[bio_id]["n_rec"] += cnt
            bio_info[bio_id]["eco_files"].append(eco_id)

    con.close()
    return bio_info


def discover_natural_bioregions():
    """Scan ref_samples files to find which bioregions have natural reference.
    Returns dict: {bio_id: count}
    """
    con = duckdb.connect()
    bio_nat = {}

    for eco_id in ALL_ECOS:
        ref_path = DATA_DIR / f"ref_samples_eco{eco_id}.parquet"
        if not ref_path.exists():
            continue
        rp = str(ref_path).replace("\\", "/")
        try:
            rows = con.sql(f"""
                SELECT bioregion_id, COUNT(*) as cnt
                FROM '{rp}'
                WHERE bioregion_id > 0 AND "natural" = 1
                GROUP BY bioregion_id
            """).fetchall()
        except Exception:
            continue

        for bio_id, cnt in rows:
            bio_id = int(bio_id)
            bio_nat[bio_id] = bio_nat.get(bio_id, 0) + cnt

    con.close()
    return bio_nat


def load_recovering_for_bioregion(bio_id):
    """Load recovering pixels for a specific bioregion across all eco files.
    Uses DuckDB to join rec + emb + scores per eco, filtered by bioregion_id.
    Returns a single DataFrame with pixel_id, lat, lon, sanlc_2022,
    gpp_slope, svh_slope, LATE_GPP, LATE_SVH, EMBED_COLS, bioregion_id, niaps.
    """
    frames = []
    late_gpp_sel = ", ".join(f"r.{c}" for c in LATE_GPP)
    late_svh_sel = ", ".join(f"r.{c}" for c in LATE_SVH)
    embed_sel = ", ".join(f"e.{c}" for c in EMBED_COLS)

    for eco_id in ALL_ECOS:
        rec_path = DATA_DIR / f"recovering_eco{eco_id}.parquet"
        emb_path = DATA_DIR / f"recovering_eco{eco_id}_embeddings.parquet"
        score_path = DATA_DIR / f"recovery_scores_eco{eco_id}.parquet"

        if not all(p.exists() for p in [rec_path, emb_path, score_path]):
            continue

        rp = str(rec_path).replace("\\", "/")
        ep = str(emb_path).replace("\\", "/")
        sp = str(score_path).replace("\\", "/")

        con = duckdb.connect()
        try:
            # Check if this eco file has any pixels for this bioregion
            cnt = con.sql(f"""
                SELECT COUNT(*) FROM '{sp}'
                WHERE bioregion_id = {bio_id}
            """).fetchone()[0]

            if cnt == 0:
                con.close()
                continue

            niaps_col = ""
            try:
                con.sql(f"SELECT niaps FROM '{sp}' LIMIT 1").fetchone()
                niaps_col = ", s.niaps"
            except Exception:
                pass

            sql = f"""
                SELECT DISTINCT ON (r.pixel_id)
                    CAST(r.pixel_id AS BIGINT) as pixel_id,
                    r.latitude, r.longitude,
                    r.sanlc_2022, r.gpp_slope, r.svh_slope,
                    {late_gpp_sel}, {late_svh_sel},
                    {embed_sel},
                    s.bioregion_id{niaps_col}
                FROM '{rp}' r
                JOIN '{ep}' e ON CAST(r.pixel_id AS BIGINT) = CAST(e.pixel_id AS BIGINT)
                JOIN '{sp}' s ON CAST(r.pixel_id AS BIGINT) = CAST(s.pixel_id AS BIGINT)
                WHERE s.bioregion_id = {bio_id}
            """
            df = con.sql(sql).df()
            if len(df) > 0:
                frames.append(df)
        except Exception as ex:
            print(f"    [WARN] eco{eco_id}: {ex}")
        finally:
            con.close()

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset="pixel_id", keep="first")
    return result


def load_natural_for_bioregion(bio_id):
    """Load natural reference samples for a specific bioregion across all eco files.
    Returns DataFrame with latitude, longitude, LATE_GPP, LATE_SVH, EMBED_COLS.
    """
    frames = []
    late_gpp_sel = ", ".join(LATE_GPP)
    late_svh_sel = ", ".join(LATE_SVH)
    embed_sel = ", ".join(EMBED_COLS)

    for eco_id in ALL_ECOS:
        ref_path = DATA_DIR / f"ref_samples_eco{eco_id}.parquet"
        if not ref_path.exists():
            continue

        rp = str(ref_path).replace("\\", "/")
        con = duckdb.connect()
        try:
            sql = f"""
                SELECT latitude, longitude,
                       {late_gpp_sel}, {late_svh_sel}, {embed_sel}
                FROM '{rp}'
                WHERE bioregion_id = {bio_id} AND "natural" = 1
            """
            df = con.sql(sql).df()
            if len(df) > 0:
                df = df.dropna(subset=LATE_GPP + LATE_SVH + EMBED_COLS[:1])
                if len(df) > 0:
                    frames.append(df)
        except Exception:
            pass
        finally:
            con.close()

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def build_natural_baselines(nat_df, n_cosine_baseline=500):
    """Pre-compute baselines for percentile normalisation."""
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
        _, nn_idx = tree.query(nat_coords[idx:idx + 1], k=KNN_K + 1)
        nn_idx = nn_idx[0]
        nn_idx = nn_idx[nn_idx != idx][:KNN_K]
        s = cosine_similarity(nat_embeds[idx:idx + 1], nat_embeds[nn_idx])[0].mean()
        local_baseline_sims[i] = s

    return {
        "nat_gpp": nat_gpp, "nat_svh": nat_svh,
        "nat_embeds": nat_embeds, "nat_coords": nat_coords,
        "tree": tree,
        "eco_gpp_mean": eco_gpp_mean, "eco_svh_mean": eco_svh_mean,
        "nat_gpp_ratios": nat_gpp_ratios, "nat_svh_ratios": nat_svh_ratios,
        "eco_baseline_sims": eco_baseline_sims,
        "local_baseline_sims": local_baseline_sims,
    }


def score_batch(batch_gpp, batch_svh, batch_embeds, batch_coords_rad, baselines):
    """Score a batch of recovering pixels against bioregion baselines."""
    bl = baselines
    N = len(batch_gpp)

    test_gpp = batch_gpp.mean(axis=1)
    test_svh = batch_svh.mean(axis=1)

    # Metric A: percentile within bioregion natural
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

    # Metric C_eco: cosine sim to all bioregion natural
    cos_eco = cosine_similarity(batch_embeds, bl["nat_embeds"]).mean(axis=1)
    c_eco = np.array([percentileofscore(bl["eco_baseline_sims"], v) for v in cos_eco])

    # Metric C_local: cosine sim to own KNN
    cos_local = np.empty(N, dtype=np.float32)
    for i in range(N):
        knn_embeds = bl["nat_embeds"][nn_idx[i]]
        cos_local[i] = cosine_similarity(batch_embeds[i:i + 1], knn_embeds)[0].mean()
    c_local = np.array([percentileofscore(bl["local_baseline_sims"], v) for v in cos_local])

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


def process_bioregion(bio_id, bio_names, test_mode=False):
    """Score all recovering pixels in a single bioregion (lazy loading)."""
    bio_name = bio_names.get(bio_id, f"Bioregion {bio_id}")

    # Load data for this bioregion only
    nat_df = load_natural_for_bioregion(bio_id)
    if len(nat_df) < MIN_NATURAL_POINTS:
        print(f"  [SKIP] bio{bio_id} ({bio_name}): only {len(nat_df)} "
              f"natural ref (need {MIN_NATURAL_POINTS})")
        return None

    rec_df = load_recovering_for_bioregion(bio_id)
    if rec_df.empty:
        print(f"  [SKIP] bio{bio_id} ({bio_name}): no recovering pixels")
        return None

    if test_mode:
        rec_df = rec_df.head(2000)

    print(f"\n  bio{bio_id} ({bio_name}): {len(rec_df):,} recovering, "
          f"{len(nat_df):,} natural ref")

    # Build baselines
    baselines = build_natural_baselines(nat_df)

    # Score in chunks
    chunk_size = 2000
    n_chunks = (len(rec_df) + chunk_size - 1) // chunk_size
    all_results = []

    for ci in tqdm(range(n_chunks), desc=f"  bio{bio_id}", leave=False):
        start = ci * chunk_size
        end = min(start + chunk_size, len(rec_df))
        chunk = rec_df.iloc[start:end]

        batch_gpp = chunk[LATE_GPP].values.astype(np.float32)
        batch_svh = chunk[LATE_SVH].values.astype(np.float32)
        batch_embeds = chunk[EMBED_COLS].values.astype(np.float32)
        batch_coords = np.radians(
            chunk[["latitude", "longitude"]].values.astype(np.float64))

        scores = score_batch(batch_gpp, batch_svh, batch_embeds,
                             batch_coords, baselines)

        chunk_result = pd.DataFrame({
            "pixel_id": chunk["pixel_id"].values,
            "latitude": chunk["latitude"].values.astype(np.float32),
            "longitude": chunk["longitude"].values.astype(np.float32),
            "bioregion_id": bio_id,
            "sanlc_2022": chunk["sanlc_2022"].values,
            "gpp_slope": chunk["gpp_slope"].values,
            "svh_slope": chunk["svh_slope"].values,
            **scores,
        })

        # Copy niaps from existing scores if available
        if "niaps" in chunk.columns:
            chunk_result["niaps"] = chunk["niaps"].values

        all_results.append(chunk_result)

    df_out = pd.concat(all_results, ignore_index=True)

    # Save
    out_path = DATA_DIR / f"recovery_scores_bio{bio_id}.parquet"
    df_out.to_parquet(str(out_path), compression="zstd")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)

    med = df_out["recovery_score"].median()
    n_inv = int((df_out.get("niaps", pd.Series()) == 1).sum())
    n_rec = len(df_out)
    n_nat = len(nat_df)
    print(f"  [OK] bio{bio_id}: {n_rec:,} scored, "
          f"median={med:.1f}, saved ({size_mb:.1f} MB)")

    # Free memory
    del rec_df, nat_df, baselines, df_out, all_results
    gc.collect()

    return {
        "bio_id": bio_id, "name": bio_name,
        "n_recovering": n_rec, "n_natural": n_nat,
        "median_score": med,
        "n_invasive": n_inv,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-score recovery metrics by bioregion")
    parser.add_argument("--bio_id", type=int, default=None,
                        help="Single bioregion ID to process")
    parser.add_argument("--test_mode", action="store_true",
                        help="Small sample for testing")
    args = parser.parse_args()

    bio_names = load_bioregion_names()
    print(f"Loaded {len(bio_names)} bioregion names")

    # Discover bioregions from existing files (lightweight scan)
    print("\nDiscovering bioregions in score files ...")
    bio_info = discover_bioregions()
    print(f"  Found {len(bio_info)} bioregions with recovering pixels")

    print("Discovering bioregions in ref_samples files ...")
    bio_nat = discover_natural_bioregions()
    print(f"  Found {len(bio_nat)} bioregions with natural reference")

    # Intersect: only process bioregions with both recovering and natural
    bio_ids = sorted(set(bio_info.keys()) & set(bio_nat.keys()))

    if args.bio_id:
        bio_ids = [args.bio_id]

    print(f"\nBioregions to process: {len(bio_ids)}")
    print(f"{'bio_id':>8} {'name':<40} {'n_rec':>10} {'n_nat':>8}")
    print("-" * 70)
    for bid in bio_ids:
        n_rec = bio_info.get(bid, {}).get("n_rec", 0)
        n_nat = bio_nat.get(bid, 0)
        name = bio_names.get(bid, "?")[:38]
        print(f"{bid:>8} {name:<40} {n_rec:>10,} {n_nat:>8,}")

    # Process each bioregion (one at a time to save memory)
    results = []
    for bid in bio_ids:
        try:
            r = process_bioregion(bid, bio_names, test_mode=args.test_mode)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [ERROR] bio{bid}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print(f"\n{'=' * 80}")
        print("BIOREGION RECOVERY SCORING SUMMARY")
        print(f"{'=' * 80}")
        print(f"{'bio_id':>8} {'Name':<35} {'Recovering':>12} "
              f"{'Median':>8} {'Invaded':>8}")
        print("-" * 80)
        total_rec = 0
        total_inv = 0
        for r in sorted(results, key=lambda x: x["n_recovering"], reverse=True):
            print(f"{r['bio_id']:>8} {r['name'][:33]:<35} "
                  f"{r['n_recovering']:>12,} "
                  f"{r['median_score']:>8.1f} {r['n_invasive']:>8,}")
            total_rec += r["n_recovering"]
            total_inv += r["n_invasive"]
        print("-" * 80)
        print(f"{'TOTAL':>8} {'':35} {total_rec:>12,} "
              f"{'':>8} {total_inv:>8,}")

    print("\n[DONE] Bioregion scoring complete.")


if __name__ == "__main__":
    main()
