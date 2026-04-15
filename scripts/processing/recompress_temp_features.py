"""
Recompress existing temp feature parquet files to match the new lean schema:
  - Drop: eco_id, has_break, breakpoint, and the 18 hand-crafted slope/mean/cv features
  - Cast: lat/lon -> float32, all other floats -> float16
  - Rewrite: ZSTD level 6 compression (vs default Snappy)

Run this once before resuming trajectory_classifier.py.
"""

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(BASE_DIR, "data", "temp_features")

COLS_TO_DROP = [
    'eco_id', 'has_break', 'breakpoint',
    'GPP_slope1', 'GPP_slope2', 'GPP_mean1', 'GPP_mean2',
    'GPP_cv1', 'GPP_cv2', 'GPP_delta_slope', 'GPP_delta_mean', 'GPP_delta_cv',
    'SVH_slope1', 'SVH_slope2', 'SVH_mean1', 'SVH_mean2',
    'SVH_cv1', 'SVH_cv2', 'SVH_delta_slope', 'SVH_delta_mean', 'SVH_delta_cv',
]

F32_COLS = ['latitude', 'longitude']  # need more precision than float16


def recompress(path):
    df = pd.read_parquet(path)
    df = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])
    f16 = [c for c in df.select_dtypes(include='float64').columns if c not in F32_COLS]
    df[F32_COLS] = df[[c for c in F32_COLS if c in df.columns]].astype('float32')
    df[f16] = df[f16].astype('float16')
    df.to_parquet(path, compression='zstd', compression_level=6)
    return path


def main():
    files = sorted(glob.glob(os.path.join(TEMP_DIR, "chunk_*.parquet")))
    if not files:
        print("No chunk files found in", TEMP_DIR)
        sys.exit(1)

    print(f"Found {len(files)} chunk files to recompress.")

    # Sample one file to show before/after schema
    sample = pd.read_parquet(files[0])
    print(f"  Before: {len(sample.columns)} columns, dtypes sample: {dict(sample.dtypes.value_counts())}")

    n_workers = min(8, os.cpu_count() or 4)
    done = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(recompress, f): f for f in files}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as e:
                errors += 1
                print(f"  ERROR on {futures[fut]}: {e}")
            if done % 200 == 0 and done > 0:
                print(f"  {done}/{len(files)} done ({errors} errors)...")

    print(f"\nDone. {done} recompressed, {errors} errors.")

    # Show after schema from same file
    sample2 = pd.read_parquet(files[0])
    print(f"  After:  {len(sample2.columns)} columns, dtypes sample: {dict(sample2.dtypes.value_counts())}")

    from pathlib import Path
    total_mb = sum(Path(f).stat().st_size for f in files) / 1024 / 1024
    print(f"  Total size of temp_features/: {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
