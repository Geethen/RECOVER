"""
Export trajectory classification (Recovering / Stable / Degrading) as a GeoTIFF.

Reads the abandoned_ag_gpp_2000_2022_SA.parquet dataset (~33M pixels),
classifies each pixel using Mann-Kendall + Theil-Sen on GPP and SVH time
series, removes NIAPS-invaded pixels, and rasterizes the result into a
Cloud-Optimized GeoTIFF suitable for upload to Google Earth Engine.

Classification scheme (uint8):
    0 = No data / NIAPS excluded
    1 = Recovering (both GPP and SVH significantly increasing)
    2 = Recovering GPP only (GPP↑, SVH stable or degrading)
    3 = Recovering SVH only (SVH↑, GPP stable or degrading)
    4 = Stable (neither GPP nor SVH significantly changing)
    5 = Degrading GPP only (GPP↓, SVH stable or recovering)
    6 = Degrading SVH only (SVH↓, GPP stable or recovering)
    7 = Degrading (both GPP and SVH significantly decreasing)

Memory-efficient: processes data in chunks via PyArrow, rasterizes via
windowed strip-writing (adapted from gpkg_to_binary.py pattern).

Usage:
    python scripts/processing/export_classification_geotiff.py
    python scripts/processing/export_classification_geotiff.py --resolution 30
    python scripts/processing/export_classification_geotiff.py --resolution 90 --block-mb 512
"""
import argparse
import gc
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rasterio
import rasterio.windows
from rasterio.transform import from_origin
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path(r"c:\Users\coach\myfiles\postdoc2\code")
RAW = BASE_DIR / "data" / "abandoned_ag_gpp_2000_2022_SA.parquet"

YEARS = list(range(2000, 2023))
GPP_COLS = [f"GPP_{y}" for y in YEARS]
SVH_COLS = [f"SVH_{y}" for y in YEARS]

# Ecoregions with NIAPS data in recovery_scores files
ALL_ECOS = [81, 41, 38, 40, 97, 48, 89, 90, 101, 88, 110,
            16, 102, 19, 94, 15, 116, 65]

# Classification codes
CLASS_NODATA = 0
CLASS_BOTH_RECOVERING = 1
CLASS_GPP_RECOVERING = 2
CLASS_SVH_RECOVERING = 3
CLASS_STABLE = 4
CLASS_GPP_DEGRADING = 5
CLASS_SVH_DEGRADING = 6
CLASS_BOTH_DEGRADING = 7

CLASS_LABELS = {
    0: "No data / NIAPS excluded",
    1: "Both recovering",
    2: "GPP recovering only",
    3: "SVH recovering only",
    4: "Stable",
    5: "GPP degrading only",
    6: "SVH degrading only",
    7: "Both degrading",
}

MK_ALPHA = 0.05


# ── Mann-Kendall + Theil-Sen (vectorised batch) ────────────────────────────

def mk_sen_batch(y_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mann-Kendall + Theil-Sen for a batch (B, T).
    Returns (sen_slope, p_values) arrays of shape (B,).
    """
    B, T = y_arr.shape
    i, j = np.triu_indices(T, k=1)
    dx = (j - i).astype(np.float32)
    diffs = y_arr[:, j] - y_arr[:, i]
    slopes = diffs / dx[None, :]
    sen_slope = np.median(slopes, axis=1).astype(np.float32)
    S = np.sum(np.sign(diffs), axis=1).astype(np.float64)
    del diffs, slopes
    var_s = (T * (T - 1) * (2 * T + 5)) / 18.0
    sigma = np.sqrt(var_s)
    Z = np.zeros(B)
    pos = S > 0
    neg = S < 0
    Z[pos] = (S[pos] - 1) / sigma
    Z[neg] = (S[neg] + 1) / sigma
    p = 2 * stats.norm.sf(np.abs(Z))
    return sen_slope, p


def classify_combined(gpp_slope, gpp_p, svh_slope, svh_p, alpha=MK_ALPHA):
    """
    Classify each pixel into one of 7 trajectory classes.
    Returns uint8 array of class codes.
    """
    n = len(gpp_slope)
    result = np.full(n, CLASS_STABLE, dtype=np.uint8)

    gpp_rec = (gpp_p < alpha) & (gpp_slope > 0)
    gpp_deg = (gpp_p < alpha) & (gpp_slope < 0)
    svh_rec = (svh_p < alpha) & (svh_slope > 0)
    svh_deg = (svh_p < alpha) & (svh_slope < 0)

    # Both recovering
    result[gpp_rec & svh_rec] = CLASS_BOTH_RECOVERING
    # GPP recovering only (SVH not recovering)
    result[gpp_rec & ~svh_rec] = CLASS_GPP_RECOVERING
    # SVH recovering only (GPP not recovering)
    result[~gpp_rec & svh_rec] = CLASS_SVH_RECOVERING
    # Both degrading
    result[gpp_deg & svh_deg] = CLASS_BOTH_DEGRADING
    # GPP degrading only (SVH not degrading)
    result[gpp_deg & ~svh_deg] = CLASS_GPP_DEGRADING
    # SVH degrading only (GPP not degrading)
    result[~gpp_deg & svh_deg] = CLASS_SVH_DEGRADING

    return result


# ── NIAPS lookup ────────────────────────────────────────────────────────────

def load_niaps_set() -> set[tuple[float, float]]:
    """
    Load all NIAPS-invaded pixel locations from recovery_scores files.
    Returns set of (round(lat,4), round(lon,4)) for niaps==1.
    """
    import duckdb
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'")
    niaps_set = set()

    for eco_id in ALL_ECOS:
        path = BASE_DIR / "data" / f"recovery_scores_eco{eco_id}.parquet"
        if not path.exists():
            continue
        df = con.sql(f"""
            SELECT latitude, longitude
            FROM '{path}'
            WHERE niaps = 1
        """).df()
        for lat, lon in zip(df["latitude"].values, df["longitude"].values):
            niaps_set.add((round(float(lat), 4), round(float(lon), 4)))

    con.close()
    print(f"  Loaded {len(niaps_set):,} NIAPS-invaded pixel locations")
    return niaps_set


# ── Step 1: Classify all pixels and collect (lat, lon, class) ──────────────

def classify_all_pixels(niaps_set: set, natural_only: bool = True):
    """
    Stream through abandoned_ag parquet, classify each pixel,
    and yield (latitude, longitude, class_code) arrays per chunk.
    """
    pf = pq.ParquetFile(str(RAW))
    needed_cols = ["latitude", "longitude", "sanlc_2022", "eco_id"] + GPP_COLS + SVH_COLS
    processed = 0
    counts = {v: 0 for v in CLASS_LABELS}
    niaps_excluded = 0

    for batch in pf.iter_batches(batch_size=50_000, columns=needed_cols):
        chunk = batch.to_pandas()

        # Filter to natural LC only (sanlc 1, 2) if requested
        if natural_only:
            mask = chunk["sanlc_2022"].isin([1, 2])
            chunk = chunk[mask].reset_index(drop=True)
        if len(chunk) == 0:
            continue

        lats = chunk["latitude"].values
        lons = chunk["longitude"].values
        ecos = chunk["eco_id"].values
        gpp = chunk[GPP_COLS].values.astype(np.float32)
        svh = chunk[SVH_COLS].values.astype(np.float32)

        # Per-ecoregion Z-score standardization
        for eco in np.unique(ecos):
            m = ecos == eco
            gpp_sub = gpp[m]
            svh_sub = svh[m]
            gpp[m] = (gpp_sub - np.nanmean(gpp_sub)) / (np.nanstd(gpp_sub) + 1e-9)
            svh[m] = (svh_sub - np.nanmean(svh_sub)) / (np.nanstd(svh_sub) + 1e-9)

        # MK + Theil-Sen in sub-batches
        B = len(gpp)
        SB = 500
        gpp_slope = np.empty(B, dtype=np.float32)
        gpp_p = np.empty(B, dtype=np.float64)
        svh_slope = np.empty(B, dtype=np.float32)
        svh_p = np.empty(B, dtype=np.float64)

        for s in range(0, B, SB):
            e = min(s + SB, B)
            gs, gp = mk_sen_batch(gpp[s:e])
            ss, sp = mk_sen_batch(svh[s:e])
            gpp_slope[s:e] = gs
            gpp_p[s:e] = gp
            svh_slope[s:e] = ss
            svh_p[s:e] = sp

        del gpp, svh
        classes = classify_combined(gpp_slope, gpp_p, svh_slope, svh_p)

        # Apply NIAPS exclusion
        for i in range(B):
            key = (round(float(lats[i]), 4), round(float(lons[i]), 4))
            if key in niaps_set:
                classes[i] = CLASS_NODATA
                niaps_excluded += 1

        # Count classes
        for cls_val in np.unique(classes):
            counts[int(cls_val)] += int(np.sum(classes == cls_val))

        processed += B
        if processed % 500_000 < 50_000:
            print(f"  Classified {processed:,} pixels...")

        yield lats.astype(np.float64), lons.astype(np.float64), classes

        del chunk, lats, lons, classes, gpp_slope, svh_slope, gpp_p, svh_p
        gc.collect()

    print(f"\n  Total classified: {processed:,}")
    print(f"  NIAPS excluded: {niaps_excluded:,}")
    print("\n  Class distribution:")
    for cls_val, label in CLASS_LABELS.items():
        print(f"    {cls_val} ({label}): {counts[cls_val]:,}")


# ── Step 2: Rasterize to GeoTIFF using windowed strip-writing ──────────────

def export_geotiff(
    output_path: str,
    resolution_m: float = 30.0,
    block_mb: int = 256,
    natural_only: bool = True,
):
    """
    Two-pass approach:
      Pass 1: Stream classify → collect all (lat, lon, class) into a temporary
              numpy memmap to avoid holding 33M+ rows in RAM.
      Pass 2: Write raster strips, burning pixels from the memmap.
    """
    import tempfile

    print("=" * 60)
    print("Export Trajectory Classification GeoTIFF")
    print("=" * 60)

    # Load NIAPS invaded locations
    print("\nStep 1: Loading NIAPS exclusion set...")
    niaps_set = load_niaps_set()

    # ── Pass 1: Classify and collect coordinates + classes ──
    print("\nStep 2: Classifying all pixels (MK + Theil-Sen)...")

    # Pre-allocate memmap for coordinates + class
    # Estimate max rows from file metadata
    pf = pq.ParquetFile(str(RAW))
    max_rows = pf.metadata.num_rows
    del pf

    tmpdir = tempfile.mkdtemp(prefix="traj_export_")
    lat_mm = np.memmap(
        Path(tmpdir) / "lats.bin", dtype=np.float64, mode="w+", shape=(max_rows,)
    )
    lon_mm = np.memmap(
        Path(tmpdir) / "lons.bin", dtype=np.float64, mode="w+", shape=(max_rows,)
    )
    cls_mm = np.memmap(
        Path(tmpdir) / "classes.bin", dtype=np.uint8, mode="w+", shape=(max_rows,)
    )

    offset = 0
    for lats, lons, classes in classify_all_pixels(niaps_set, natural_only):
        n = len(lats)
        lat_mm[offset:offset + n] = lats
        lon_mm[offset:offset + n] = lons
        cls_mm[offset:offset + n] = classes
        offset += n

    total_pixels = offset
    print(f"\n  Total pixels to rasterize: {total_pixels:,}")

    # Trim memmaps
    lat_mm = np.memmap(
        Path(tmpdir) / "lats.bin", dtype=np.float64, mode="r", shape=(total_pixels,)
    )
    lon_mm = np.memmap(
        Path(tmpdir) / "lons.bin", dtype=np.float64, mode="r", shape=(total_pixels,)
    )
    cls_mm = np.memmap(
        Path(tmpdir) / "classes.bin", dtype=np.uint8, mode="r", shape=(total_pixels,)
    )

    # ── Compute raster grid ──
    lat_min, lat_max = float(np.min(lat_mm)), float(np.max(lat_mm))
    lon_min, lon_max = float(np.min(lon_mm)), float(np.max(lon_mm))

    # Add small padding
    pad = 0.001
    lat_min -= pad
    lat_max += pad
    lon_min -= pad
    lon_max += pad

    # Pixel size in degrees (approximate)
    pixel_deg = resolution_m / 111_320.0
    ncols = math.ceil((lon_max - lon_min) / pixel_deg)
    nrows = math.ceil((lat_max - lat_min) / pixel_deg)

    # Origin is top-left (north-west corner)
    transform = from_origin(lon_min, lat_max, pixel_deg, pixel_deg)

    print(f"\nStep 3: Writing GeoTIFF...")
    print(f"  Bounds: lat [{lat_min:.4f}, {lat_max:.4f}], lon [{lon_min:.4f}, {lon_max:.4f}]")
    print(f"  Grid: {ncols} x {nrows} pixels ({pixel_deg:.7f}° per pixel)")
    uncompressed_mb = ncols * nrows / (1024 * 1024)
    print(f"  Uncompressed size: {uncompressed_mb:.1f} MB (uint8, LZW compressed)")

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "width": ncols,
        "height": nrows,
        "compress": "LZW",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }

    # Target strip height to fit within block_mb RAM
    strip_rows = max(256, block_mb * 1024 * 1024 // ncols)
    # Round to tile boundary
    strip_rows = (strip_rows // 256) * 256
    n_strips = math.ceil(nrows / strip_rows)

    print(f"  Writing {n_strips} strips of {strip_rows} rows each")

    # Pre-compute pixel row/col for all points
    print("  Computing pixel coordinates...")
    col_idx = np.floor((lon_mm - lon_min) / pixel_deg).astype(np.int32)
    row_idx = np.floor((lat_max - lat_mm) / pixel_deg).astype(np.int32)

    # Clamp to valid range
    np.clip(col_idx, 0, ncols - 1, out=col_idx)
    np.clip(row_idx, 0, nrows - 1, out=row_idx)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        for strip_i in range(n_strips):
            row_start = strip_i * strip_rows
            row_end = min(row_start + strip_rows, nrows)
            strip_h = row_end - row_start

            # Find pixels in this strip
            mask = (row_idx >= row_start) & (row_idx < row_end)
            if np.any(mask):
                strip_buf = np.zeros((strip_h, ncols), dtype=np.uint8)
                local_rows = row_idx[mask] - row_start
                local_cols = col_idx[mask]
                local_cls = cls_mm[mask]

                # Burn pixels — last-writer-wins for overlapping coords
                strip_buf[local_rows, local_cols] = local_cls
            else:
                strip_buf = np.zeros((strip_h, ncols), dtype=np.uint8)

            window = rasterio.windows.Window(
                col_off=0, row_off=row_start, width=ncols, height=strip_h
            )
            dst.write(strip_buf[np.newaxis, ...], window=window)

            pct = 100 * (strip_i + 1) / n_strips
            print(f"  Strip {strip_i + 1}/{n_strips} ({pct:.0f}%)", end="\r", flush=True)

    print(f"\n\n  GeoTIFF written: {output_path}")
    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    # ── Write metadata JSON ──
    meta_path = str(Path(output_path).with_suffix(".json"))
    meta = {
        "created": date.today().isoformat(),
        "source": "abandoned_ag_gpp_2000_2022_SA.parquet",
        "method": "Mann-Kendall trend test + Theil-Sen slope",
        "time_series": f"GPP and SVH, {YEARS[0]}-{YEARS[-1]} ({len(YEARS)} years)",
        "standardization": "Per-ecoregion Z-score (within-chunk)",
        "significance": f"alpha = {MK_ALPHA}",
        "niaps_excluded": True,
        "natural_lc_only": natural_only,
        "crs": "EPSG:4326",
        "resolution_m": resolution_m,
        "resolution_deg": round(pixel_deg, 9),
        "width_px": ncols,
        "height_px": nrows,
        "total_classified_pixels": total_pixels,
        "class_values": {str(k): v for k, v in CLASS_LABELS.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata: {meta_path}")

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Export trajectory classification as GeoTIFF for GEE"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(BASE_DIR / "data" / "trajectory_classification_30m.tif"),
        help="Output GeoTIFF path",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=float, default=30.0,
        help="Pixel size in metres (default: 30)",
    )
    parser.add_argument(
        "--block-mb",
        type=int, default=256,
        help="Target RAM per strip in MB (default: 256)",
    )
    parser.add_argument(
        "--include-transformed",
        action="store_true",
        help="Include transformed LC pixels (default: natural LC only, SANLC 1-2)",
    )
    args = parser.parse_args()

    export_geotiff(
        output_path=args.output,
        resolution_m=args.resolution,
        block_mb=args.block_mb,
        natural_only=not args.include_transformed,
    )


if __name__ == "__main__":
    main()
