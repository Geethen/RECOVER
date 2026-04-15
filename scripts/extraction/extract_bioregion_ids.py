"""
Extract bioregion IDs from GEE raster for all recovery_scores and ref_samples.

Reads each recovery_scores_eco{id}.parquet and ref_samples_eco{id}.parquet,
extracts the bioregion integer value at each pixel's lat/lon from the
uploaded GEE asset, and adds a 'bioregion_id' column.

Usage:
    python scripts/extraction/extract_bioregion_ids.py
    python scripts/extraction/extract_bioregion_ids.py --scores_only
    python scripts/extraction/extract_bioregion_ids.py --refs_only
"""
import sys
import os
import ee
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
DATA_DIR = BASE_DIR / "data"

ALL_ECOS = [81, 41, 38, 40, 97, 48, 89, 90, 101, 88, 110,
            16, 102, 19, 94, 15, 116, 65]

BIOREGION_ASSET = "projects/ee-gsingh/assets/RECOVER/bioregions_30m"
BATCH_SIZE = 5000
MAX_WORKERS = 10
MAX_RETRIES = 3


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


def extract_bioregion_batch(lats, lons, row_indices):
    """Extract bioregion ID for a batch of points via reduceRegions."""
    bioregions = ee.Image(BIOREGION_ASSET)

    features = []
    for i in range(len(lats)):
        pt = ee.Geometry.Point([float(lons[i]), float(lats[i])])
        features.append(ee.Feature(pt, {"rid": int(row_indices[i])}))

    fc = ee.FeatureCollection(features)
    result = bioregions.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.first(),
        scale=30,
    )

    info = retry_gee(lambda: result.getInfo())
    out = {}
    for feat in info["features"]:
        rid = feat["properties"]["rid"]
        val = feat["properties"].get("first", None)
        out[rid] = int(val) if val is not None else 0
    return out


def extract_bioregion_for_df(df, lat_col="latitude", lon_col="longitude",
                              desc="bioregion"):
    """Extract bioregion IDs for all rows in a DataFrame. Returns array."""
    n_total = len(df)
    lats = df[lat_col].values
    lons = df[lon_col].values
    row_indices = np.arange(n_total)

    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE
    bio_map = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, n_batches)) as executor:
        futures = {}
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            e = min(s + BATCH_SIZE, n_total)
            future = executor.submit(
                extract_bioregion_batch,
                lats[s:e], lons[s:e], row_indices[s:e])
            futures[future] = bi

        for future in tqdm(as_completed(futures), total=n_batches,
                           desc=f"  {desc}"):
            batch_result = future.result()
            bio_map.update(batch_result)

    result = np.zeros(n_total, dtype=np.int32)
    for idx, val in bio_map.items():
        result[idx] = val
    return result


def process_score_files():
    """Add bioregion_id to all recovery_scores_eco{id}.parquet files."""
    print("\n" + "=" * 60)
    print("EXTRACTING BIOREGION IDs FOR RECOVERY SCORES")
    print("=" * 60)

    for eco_id in ALL_ECOS:
        score_path = DATA_DIR / f"recovery_scores_eco{eco_id}.parquet"
        if not score_path.exists():
            print(f"  [SKIP] No scores for eco_id={eco_id}")
            continue

        df = pd.read_parquet(str(score_path))

        if "bioregion_id" in df.columns and df["bioregion_id"].notna().sum() == len(df):
            n_unique = df["bioregion_id"].nunique()
            print(f"  [SKIP] eco{eco_id}: bioregion_id already extracted "
                  f"({len(df):,} pixels, {n_unique} bioregions)")
            continue

        print(f"\n  eco{eco_id}: {len(df):,} pixels ...")
        df["bioregion_id"] = extract_bioregion_for_df(
            df, desc=f"eco{eco_id} scores")

        n_zero = (df["bioregion_id"] == 0).sum()
        n_unique = df["bioregion_id"][df["bioregion_id"] > 0].nunique()
        print(f"  [OK] eco{eco_id}: {n_unique} bioregions, "
              f"{n_zero} unmatched (0)")

        df.to_parquet(str(score_path), compression="zstd")
        print(f"       Saved: {score_path}")


def process_ref_files():
    """Add bioregion_id to all ref_samples_eco{id}.parquet files."""
    print("\n" + "=" * 60)
    print("EXTRACTING BIOREGION IDs FOR REFERENCE SAMPLES")
    print("=" * 60)

    for eco_id in ALL_ECOS:
        ref_path = DATA_DIR / f"ref_samples_eco{eco_id}.parquet"
        if not ref_path.exists():
            print(f"  [SKIP] No ref_samples for eco_id={eco_id}")
            continue

        df = pd.read_parquet(str(ref_path))

        if "bioregion_id" in df.columns and df["bioregion_id"].notna().sum() == len(df):
            n_unique = df["bioregion_id"].nunique()
            print(f"  [SKIP] eco{eco_id}: bioregion_id already extracted "
                  f"({len(df):,} samples, {n_unique} bioregions)")
            continue

        print(f"\n  eco{eco_id}: {len(df):,} reference samples ...")
        df["bioregion_id"] = extract_bioregion_for_df(
            df, desc=f"eco{eco_id} refs")

        n_zero = (df["bioregion_id"] == 0).sum()
        n_natural = (df["natural"] == 1).sum() if "natural" in df.columns else "?"
        n_unique = df["bioregion_id"][df["bioregion_id"] > 0].nunique()
        print(f"  [OK] eco{eco_id}: {n_unique} bioregions, "
              f"{n_natural} natural, {n_zero} unmatched")

        df.to_parquet(str(ref_path), compression="zstd")
        print(f"       Saved: {ref_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract bioregion IDs from GEE for recovery scores and refs")
    parser.add_argument("--scores_only", action="store_true",
                        help="Only process recovery score files")
    parser.add_argument("--refs_only", action="store_true",
                        help="Only process reference sample files")
    parser.add_argument("--project", type=str, default="ee-gsingh")
    args = parser.parse_args()

    init_gee(args.project)

    if args.refs_only:
        process_ref_files()
    elif args.scores_only:
        process_score_files()
    else:
        process_score_files()
        process_ref_files()

    print("\n[DONE] Bioregion ID extraction complete.")


if __name__ == "__main__":
    main()
