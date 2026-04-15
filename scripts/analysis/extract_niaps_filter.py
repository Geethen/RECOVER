"""
Extract NIAPS (National Invasive Alien Plant Survey) binary values for all
scored recovering pixels and filter invasive pixels from recovery scores.

Reads recovery_scores_eco{id}.parquet for each ecoregion, extracts NIAPS
binary at 30m from GEE, adds 'niaps' column, and saves updated scores.
Pixels with niaps=1 are flagged for exclusion from recovery analysis.

Usage:
    python scripts/analysis/extract_niaps_filter.py
    python scripts/analysis/extract_niaps_filter.py --eco_id 81
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

NIAPS_ASSET = "projects/ee-gsingh/assets/RECOVER/niaps_binary_30m"
BATCH_SIZE = 5000  # pixels per GEE reduceRegions call
MAX_WORKERS = 10
MAX_RETRIES = 3


def init_gee(project="ee-gsingh"):
    try:
        ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")
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


def extract_niaps_batch(lats, lons, pixel_ids):
    """Extract NIAPS binary for a batch of points via reduceRegions."""
    niaps = ee.Image(NIAPS_ASSET)

    features = []
    for i in range(len(lats)):
        pt = ee.Geometry.Point([float(lons[i]), float(lats[i])])
        features.append(ee.Feature(pt, {"pid": int(pixel_ids[i])}))

    fc = ee.FeatureCollection(features)
    result = niaps.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=30,
    )

    info = retry_gee(lambda: result.getInfo())
    out = {}
    for feat in info["features"]:
        pid = feat["properties"]["pid"]
        val = feat["properties"].get("first", None)
        out[pid] = int(val) if val is not None else 0
    return out


def process_ecoregion(eco_id):
    """Extract NIAPS for all scored pixels in an ecoregion."""
    score_path = DATA_DIR / f"recovery_scores_eco{eco_id}.parquet"
    if not score_path.exists():
        print(f"  [SKIP] No scores for eco_id={eco_id}")
        return None

    df = pd.read_parquet(str(score_path))
    n_total = len(df)

    # Check if niaps column already exists
    if "niaps" in df.columns and df["niaps"].notna().sum() == n_total:
        n_inv = (df["niaps"] == 1).sum()
        pct = 100 * n_inv / n_total if n_total > 0 else 0
        print(f"  [SKIP] eco{eco_id}: niaps already extracted "
              f"({n_inv:,}/{n_total:,} = {pct:.1f}% invasive)")
        return {"eco_id": eco_id, "total": n_total, "invasive": n_inv, "pct": pct}

    print(f"\n  eco{eco_id}: Extracting NIAPS for {n_total:,} pixels ...")
    lats = df["latitude"].values
    lons = df["longitude"].values
    pids = df["pixel_id"].values

    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    niaps_map = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            e = min(s + BATCH_SIZE, n_total)
            future = executor.submit(
                extract_niaps_batch,
                lats[s:e], lons[s:e], pids[s:e])
            futures[future] = bi

        for future in tqdm(as_completed(futures), total=n_batches,
                           desc=f"  eco{eco_id} NIAPS"):
            batch_result = future.result()
            niaps_map.update(batch_result)

    # Map results back to dataframe
    df["niaps"] = df["pixel_id"].map(niaps_map).fillna(0).astype(int)
    n_inv = (df["niaps"] == 1).sum()
    pct = 100 * n_inv / n_total if n_total > 0 else 0

    # Save updated scores
    df.to_parquet(str(score_path), compression="zstd")
    print(f"  [OK] eco{eco_id}: {n_inv:,}/{n_total:,} = {pct:.1f}% invasive")
    print(f"       Saved with niaps column: {score_path}")

    return {"eco_id": eco_id, "total": n_total, "invasive": n_inv, "pct": pct}


def main():
    parser = argparse.ArgumentParser(
        description="Extract NIAPS invasive species values for recovery scores")
    parser.add_argument("--eco_id", type=int, default=None,
                        help="Single ecoregion to process")
    parser.add_argument("--project", type=str, default="ee-gsingh")
    args = parser.parse_args()

    init_gee(args.project)

    ecos = [args.eco_id] if args.eco_id else ALL_ECOS
    results = []

    for eco_id in ecos:
        try:
            r = process_ecoregion(eco_id)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  [ERROR] eco{eco_id}: {e}")

    # Summary table
    if results:
        print(f"\n{'='*60}")
        print(f"NIAPS INVASIVE SPECIES SUMMARY")
        print(f"{'='*60}")
        print(f"{'eco_id':>6} {'total':>10} {'invasive':>10} {'pct':>8}")
        print("-" * 36)
        total_all = 0
        inv_all = 0
        for r in results:
            print(f"{r['eco_id']:>6} {r['total']:>10,} {r['invasive']:>10,} "
                  f"{r['pct']:>7.1f}%")
            total_all += r["total"]
            inv_all += r["invasive"]
        print("-" * 36)
        pct_all = 100 * inv_all / total_all if total_all > 0 else 0
        print(f"{'TOTAL':>6} {total_all:>10,} {inv_all:>10,} "
              f"{pct_all:>7.1f}%")
        print(f"\nRecovering pixels after exclusion: {total_all - inv_all:,}")


if __name__ == "__main__":
    main()
