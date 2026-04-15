"""
Extract Woody Plant Encroachment (WPE) data at abandoned-ag pixel locations
and calibrate ecoregion-specific thresholds using NIAPS as ground truth.

WPE dataset: Venter et al. 2018, "users/lukiejohn/WPE_Venter_etal_2018"
  Bands: trend (% cover change/yr), sig_mask (MK significance),
         mean_cover (fractional 0-1), residuals, QA, mask

Strategy:
  1. Sample WPE bands at a stratified subset of scored recovering pixels
  2. Compare distributions: NIAPS=1 (invaded) vs NIAPS=0
  3. Derive ecoregion-specific thresholds for woody encroachment flagging

Usage:
    python scripts/analysis/extract_wpe_calibration.py
    python scripts/analysis/extract_wpe_calibration.py --max_per_eco 20000
"""
import sys
import os
import ee
import time
import argparse
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"

ALL_ECOS = [81, 41, 38, 40, 97, 48, 89, 90, 101, 88, 110,
            16, 102, 19, 94, 15, 116, 65]

WPE_ASSET = "users/lukiejohn/WPE_Venter_etal_2018"
WPE_BANDS = ["trend", "sig_mask", "mean_cover", "QA"]
BATCH_SIZE = 4000
MAX_WORKERS = 10
MAX_RETRIES = 3
MAX_PER_ECO = 30000  # max pixels to sample per ecoregion


def init_gee(project="ee-gsingh"):
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    print(f"[OK] GEE initialized (project={project})")


def retry_gee(fn, retries=MAX_RETRIES, delay=5):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if attempt < retries - 1:
                wait = delay * (2 ** attempt)
                print(f"    Retry {attempt+1}/{retries} after {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


def extract_wpe_batch(lats, lons, pixel_ids):
    """Extract WPE bands for a batch of points via reduceRegions."""
    wpe = ee.Image(WPE_ASSET).select(WPE_BANDS)

    features = []
    for i in range(len(lats)):
        pt = ee.Geometry.Point([float(lons[i]), float(lats[i])])
        features.append(ee.Feature(pt, {"pid": int(pixel_ids[i])}))

    fc = ee.FeatureCollection(features)
    result = wpe.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=30,
    )

    info = retry_gee(lambda: result.getInfo())
    rows = []
    for feat in info["features"]:
        props = feat["properties"]
        rows.append({
            "pixel_id": props["pid"],
            "wpe_trend": props.get("trend"),
            "wpe_sig": props.get("sig_mask"),
            "wpe_mean_cover": props.get("mean_cover"),
            "wpe_qa": props.get("QA"),
        })
    return rows


def process_ecoregion(eco_id, max_per_eco):
    """Extract WPE for a sample of scored pixels in an ecoregion."""
    score_path = DATA_DIR / f"recovery_scores_eco{eco_id}.parquet"
    if not score_path.exists():
        print(f"  [SKIP] No scores for eco_id={eco_id}")
        return None

    conn = duckdb.connect()
    # Read with sampling — stratify to get both NIAPS=0 and NIAPS=1
    df = conn.sql(f"""
        SELECT pixel_id, latitude, longitude, niaps,
               recovery_score, svh_slope, gpp_slope, sanlc_2022
        FROM '{score_path}'
        USING SAMPLE reservoir({max_per_eco} ROWS) REPEATABLE(42)
    """).df()
    conn.close()

    n_total = len(df)
    n_inv = (df["niaps"] == 1).sum()
    print(f"\n  eco{eco_id}: Sampling {n_total:,} pixels "
          f"({n_inv:,} NIAPS-invaded) ...")

    lats = df["latitude"].values
    lons = df["longitude"].values
    pids = df["pixel_id"].values

    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    all_rows = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            e = min(s + BATCH_SIZE, n_total)
            future = executor.submit(
                extract_wpe_batch,
                lats[s:e], lons[s:e], pids[s:e])
            futures[future] = bi

        for future in tqdm(as_completed(futures), total=n_batches,
                           desc=f"  eco{eco_id} WPE"):
            try:
                batch_rows = future.result()
                all_rows.extend(batch_rows)
            except Exception as ex:
                bi = futures[future]
                print(f"    [ERROR] batch {bi}: {ex}")

    # Merge WPE data back to score data
    wpe_df = pd.DataFrame(all_rows)
    merged = df.merge(wpe_df, on="pixel_id", how="left")
    merged["eco_id"] = eco_id

    n_valid = merged["wpe_trend"].notna().sum()
    print(f"  [OK] eco{eco_id}: {n_valid:,}/{n_total:,} pixels with WPE data")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Extract WPE data for NIAPS calibration")
    parser.add_argument("--max_per_eco", type=int, default=MAX_PER_ECO,
                        help=f"Max pixels per ecoregion (default {MAX_PER_ECO})")
    parser.add_argument("--eco_id", type=int, default=None,
                        help="Single ecoregion to process")
    parser.add_argument("--project", type=str, default="ee-gsingh")
    args = parser.parse_args()

    init_gee(args.project)

    ecos = [args.eco_id] if args.eco_id else ALL_ECOS
    all_dfs = []

    for eco_id in ecos:
        try:
            df = process_ecoregion(eco_id, args.max_per_eco)
            if df is not None:
                all_dfs.append(df)
        except Exception as e:
            print(f"  [ERROR] eco{eco_id}: {e}")

    if not all_dfs:
        print("No data extracted.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = DATA_DIR / "wpe_niaps_calibration.parquet"
    combined.to_parquet(str(out_path), compression="zstd")
    print(f"\nSaved: {out_path}")
    print(f"  Total rows: {len(combined):,}")
    print(f"  Valid WPE: {combined['wpe_trend'].notna().sum():,}")
    print(f"  NIAPS=1: {(combined['niaps'] == 1).sum():,}")
    print(f"  NIAPS=0: {(combined['niaps'] == 0).sum():,}")

    # Quick summary per ecoregion
    print(f"\n{'eco':>5} {'n':>8} {'niaps1':>8} {'wpe_valid':>10} "
          f"{'trend_med':>10} {'trend_inv':>10} {'trend_nat':>10}")
    print("-" * 65)
    for eco_id in sorted(combined["eco_id"].unique()):
        sub = combined[combined["eco_id"] == eco_id]
        valid = sub["wpe_trend"].notna()
        inv = sub["niaps"] == 1
        nat = sub["niaps"] == 0
        t_med = sub.loc[valid, "wpe_trend"].median()
        t_inv = sub.loc[valid & inv, "wpe_trend"].median() if (valid & inv).sum() > 0 else float("nan")
        t_nat = sub.loc[valid & nat, "wpe_trend"].median() if (valid & nat).sum() > 0 else float("nan")
        print(f"{eco_id:>5} {len(sub):>8,} {inv.sum():>8,} {valid.sum():>10,} "
              f"{t_med:>10.4f} {t_inv:>10.4f} {t_nat:>10.4f}")


if __name__ == "__main__":
    main()
