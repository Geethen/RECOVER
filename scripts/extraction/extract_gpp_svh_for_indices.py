"""
Extract GPP and SVH time series (2000-2022) for all points in extracted_indices.parquet.

Reads point locations from the GEE asset (fscs_aef_samples), extracts GPP, SVH,
BII, SANLC, and ecoregion data at each point, and outputs a parquet file matching
the schema of abandoned_ag_gpp_2000_2022_SA.parquet.

Usage:
    python scripts/extraction/extract_gpp_svh_for_indices.py
    python scripts/extraction/extract_gpp_svh_for_indices.py --test_mode
"""
import ee
import pandas as pd
import numpy as np
import os
import time
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm.auto import tqdm
import duckdb


# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

def retry(max_retries=3, backoff_factor=2):
    """Decorator for retrying GEE calls with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        raise e
                    sleep_time = backoff_factor ** retries
                    print(f"  Retry {retries}/{max_retries} after {sleep_time}s: {str(e)[:100]}")
                    time.sleep(sleep_time)
            return None
        return wrapper
    return decorator


class CheckpointManager:
    """Manages checkpoints to enable resumable extraction."""
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.processed_batches = self._load_checkpoints()
        self.lock = Lock()

    def _load_checkpoints(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                return set()
        return set()

    def mark_processed(self, batch_id):
        with self.lock:
            self.processed_batches.add(batch_id)
            self._save_checkpoints()

    def is_processed(self, batch_id):
        return batch_id in self.processed_batches

    def _save_checkpoints(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed_batches), f)


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

@retry(max_retries=3, backoff_factor=2)
def extract_batch(batch_fc, image_stack, scale, output_path, lock, db_conn):
    """Extract image values at a batch of points."""
    # Sample the image stack at point locations
    sampled = image_stack.sampleRegions(
        collection=batch_fc,
        scale=scale,
        projection='EPSG:4326',
        tileScale=4,
        geometries=True
    )

    # Add pixel_id from original feature
    sampled = sampled.map(
        lambda ft: ft.set('pixel_id', ft.get('system:index'))
    )

    df_result = ee.data.computeFeatures({
        'expression': sampled,
        'fileFormat': 'PANDAS_DATAFRAME'
    })

    if not df_result.empty:
        # Parse geo column to extract lat/lon if not present
        if 'latitude' not in df_result.columns and 'geo' in df_result.columns:
            def parse_geo(g):
                try:
                    d = json.loads(str(g).replace("'", '"'))
                    return d['coordinates']
                except Exception:
                    return [np.nan, np.nan]
            coords = df_result['geo'].apply(parse_geo)
            df_result['longitude'] = coords.apply(lambda c: c[0])
            df_result['latitude'] = coords.apply(lambda c: c[1])

        with lock:
            table_exists = True
            try:
                db_conn.execute("SELECT 1 FROM data LIMIT 0")
            except duckdb.CatalogException:
                table_exists = False

            if table_exists:
                # Match columns to handle schema differences
                existing_cols = [r[0] for r in db_conn.execute(
                    "SELECT column_name FROM information_schema.columns WHERE table_name='data'"
                ).fetchall()]
                common_cols = [c for c in existing_cols if c in df_result.columns]
                cols_str = ", ".join(f'"{c}"' for c in common_cols)
                db_conn.execute(f"INSERT INTO data ({cols_str}) SELECT {cols_str} FROM df_result")
            else:
                db_conn.execute("CREATE TABLE data AS SELECT * FROM df_result")

        return len(df_result)
    return 0


def main():
    # Navigate from scripts/extraction/ up two levels to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_output = os.path.join(base_dir, "data", "indices_gpp_svh_2000_2022.parquet")

    parser = argparse.ArgumentParser(
        description='Extract GPP/SVH time series for extracted_indices points.')
    parser.add_argument('--project', type=str, default='ee-gsingh',
                        help='GEE Project ID')
    parser.add_argument('--output', type=str, default=default_output,
                        help='Output path (.parquet)')
    parser.add_argument('--start_year', type=int, default=2000)
    parser.add_argument('--end_year', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='Points per batch')
    parser.add_argument('--max_workers', type=int, default=20,
                        help='Max parallel workers')
    parser.add_argument('--scale', type=int, default=30,
                        help='Extraction scale in meters')
    parser.add_argument('--test_mode', action='store_true',
                        help='Run on first 2 batches only')
    args = parser.parse_args()

    # ── Initialize GEE ─────────────────────────────────────────────────
    try:
        ee.Initialize(project=args.project,
                      opt_url='https://earthengine-highvolume.googleapis.com')
        print(f"[OK] Initialized GEE High Volume Endpoint (Project: {args.project})")
    except Exception as e:
        print(f"High volume failed: {e}")
        ee.Initialize(project=args.project)
        print(f"[OK] Initialized with standard endpoint")

    # ── Load point collection ──────────────────────────────────────────
    print("\nSTEP 1: Loading point collection")
    ASSET_ID = 'projects/ee-gsingh/assets/RECOVER/fscs_aef_samples'
    points = ee.FeatureCollection(ASSET_ID)
    total_points = points.size().getInfo()
    print(f"[OK] Loaded {total_points:,} points from {ASSET_ID}")

    # ── Build image stack ──────────────────────────────────────────────
    print("\nSTEP 2: Building image stack")
    south_africa = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
        ee.Filter.eq('ADM0_NAME', 'South Africa'))
    sa_geometry = south_africa.geometry()
    years = list(range(args.start_year, args.end_year + 1))

    # GPP (annual sum)
    print("  Loading GPP...")
    gpp_collection = ee.ImageCollection(
        "projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m")
    gpp_bands = []
    for y in years:
        annual = gpp_collection.filter(
            ee.Filter.calendarRange(y, y, 'year')).sum().rename(f'GPP_{y}')
        gpp_bands.append(annual)
    gpp_stack = ee.Image.cat(gpp_bands)

    # SVH (annual mosaic)
    print("  Loading SVH...")
    gsvh_col = ee.ImageCollection(
        "projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m")
    svh_bands = []
    for y in years:
        annual = gsvh_col.filter(
            ee.Filter.calendarRange(y, y, 'year')).mosaic().multiply(0.1).rename(f'SVH_{y}')
        svh_bands.append(annual)
    svh_stack = ee.Image.cat(svh_bands)

    # BII
    print("  Loading BII...")
    bii_col = ee.ImageCollection(
        "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_1km")
    bii_bands = ['Land Use', 'Land Use Intensity', 'BII All',
                 'BII Amphibians', 'BII Birds', 'BII Forbs', 'BII Graminoids',
                 'BII Mammals', 'BII All Plants', 'BII Reptiles', 'BII Trees',
                 'BII All Vertebrates']
    bii_img = bii_col.toBands().rename(bii_bands)
    bii_main = bii_img.select('^BII.*').selfMask()
    lc_mask = bii_img.select('Land Use').neq(2).And(
        bii_img.select('Land Use').neq(5))
    lui = bii_img.select('Land Use Intensity').updateMask(lc_mask)
    bii_mask = ee.Image(
        "projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_Mask")
    bii_processed = bii_main.addBands(
        [bii_img.select('Land Use'), lui]).updateMask(bii_mask)

    # SANLC 2022
    print("  Loading SANLC 2022...")
    sanlc = ee.Image(
        "projects/ee-gsingh/assets/RECOVER/sanlc2022_7class").rename('sanlc_2022')

    # Ecoregions
    print("  Loading Ecoregions...")
    ecoregions = ee.FeatureCollection(
        "RESOLVE/ECOREGIONS/2017").filterBounds(sa_geometry)
    ecoregions_img = ecoregions.reduceToImage(
        properties=['ECO_ID'], reducer=ee.Reducer.first()).rename('eco_id')

    # Combine
    final_stack = ee.Image.cat([
        gpp_stack,
        svh_stack,
        bii_processed,
        sanlc,
        ecoregions_img,
        ee.Image.pixelLonLat()
    ])
    print(f"[OK] Image stack ready: GPP + SVH ({args.start_year}-{args.end_year}), "
          f"BII, SANLC, Ecoregions")

    # ── Batch extraction ───────────────────────────────────────────────
    print("\nSTEP 3: Batch extraction")

    # Split points into batches using randomColumn + filter ranges
    n_batches = max(1, (total_points + args.batch_size - 1) // args.batch_size)
    print(f"  {total_points:,} points -> {n_batches} batches of ~{args.batch_size}")

    # Use randomColumn for server-side batching (avoids toList)
    points_sharded = points.randomColumn('batch_rand', seed=42)

    batches = []
    step = 1.0 / n_batches
    for k in range(n_batches):
        lower = k * step
        upper = (k + 1) * step
        batch_fc = points_sharded.filter(ee.Filter.And(
            ee.Filter.gte('batch_rand', lower),
            ee.Filter.lt('batch_rand', upper)
        ))
        batches.append((k, batch_fc))

    if args.test_mode:
        print("*** TEST MODE: processing first 2 batches only ***")
        batches = batches[:2]

    # Initialize DuckDB buffer
    db_path = args.output.replace('.parquet', '.duckdb')
    checkpoint_file = args.output + '.checkpoint.json'

    # Reset checkpoint if output doesn't exist
    if not os.path.exists(args.output) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    checkpoint_mgr = CheckpointManager(checkpoint_file)
    db_conn = duckdb.connect(db_path)

    # Load existing data into buffer if resuming
    try:
        db_conn.execute("SELECT 1 FROM data LIMIT 1")
        print("  [OK] Found existing buffer table.")
    except Exception:
        if os.path.exists(args.output):
            print(f"  [INFO] Loading existing parquet into buffer...")
            start_load = time.time()
            db_conn.execute(f"CREATE TABLE data AS SELECT * FROM '{args.output}'")
            print(f"  [OK] Loaded in {time.time()-start_load:.1f}s")
        else:
            print("  [INFO] Starting fresh dataset.")

    shared_lock = Lock()
    successful = 0
    skipped = 0
    failed = 0
    total_pixels = 0

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_batch = {}

        for batch_id, batch_fc in batches:
            if checkpoint_mgr.is_processed(batch_id):
                skipped += 1
                continue

            future = executor.submit(
                extract_batch,
                batch_fc=batch_fc,
                image_stack=final_stack,
                scale=args.scale,
                output_path=args.output,
                lock=shared_lock,
                db_conn=db_conn
            )
            future_to_batch[future] = batch_id

        with tqdm(total=len(batches), desc="Extracting batches", ncols=100) as pbar:
            pbar.update(skipped)
            for future in as_completed(future_to_batch):
                batch_id = future_to_batch[future]
                try:
                    n_pixels = future.result()
                    if isinstance(n_pixels, int):
                        checkpoint_mgr.mark_processed(batch_id)
                        successful += 1
                        total_pixels += n_pixels
                    else:
                        failed += 1
                except Exception as e:
                    print(f"  [ERROR] Batch {batch_id}: {str(e)[:100]}")
                    failed += 1
                pbar.update(1)

    # ── Export ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print(f"  Successful batches: {successful}")
    print(f"  Skipped batches:    {skipped}")
    print(f"  Failed batches:     {failed}")
    print(f"  Total pixels:       {total_pixels:,}")
    print("=" * 70)

    print("\nSTEP 4: Exporting to Parquet (optimized dtypes, ZSTD compression)")
    try:
        row_count = db_conn.execute("SELECT count(*) FROM data").fetchone()[0]
        print(f"  Buffer contains {row_count:,} rows")

        # Drop extra columns and cast to efficient types
        years = list(range(args.start_year, args.end_year + 1))
        gpp_cols = ", ".join(f"CAST(GPP_{y} AS INTEGER) AS GPP_{y}" for y in years)
        svh_cols = ", ".join(f"CAST(SVH_{y} AS SMALLINT) AS SVH_{y}" for y in years)
        bii_cols = """
            CAST("BII All" AS FLOAT) AS "BII All",
            CAST("BII All Plants" AS FLOAT) AS "BII All Plants",
            CAST("BII All Vertebrates" AS FLOAT) AS "BII All Vertebrates",
            CAST("BII Amphibians" AS FLOAT) AS "BII Amphibians",
            CAST("BII Birds" AS FLOAT) AS "BII Birds",
            CAST("BII Forbs" AS FLOAT) AS "BII Forbs",
            CAST("BII Graminoids" AS FLOAT) AS "BII Graminoids",
            CAST("BII Mammals" AS FLOAT) AS "BII Mammals",
            CAST("BII Reptiles" AS FLOAT) AS "BII Reptiles",
            CAST("BII Trees" AS FLOAT) AS "BII Trees",
            CAST("Land Use" AS SMALLINT) AS "Land Use",
            CAST("Land Use Intensity" AS FLOAT) AS "Land Use Intensity"
        """

        export_sql = f"""
            COPY (
                SELECT
                    geo,
                    {gpp_cols},
                    {svh_cols},
                    {bii_cols},
                    CAST(eco_id AS SMALLINT) AS eco_id,
                    CAST(latitude AS FLOAT) AS latitude,
                    CAST(longitude AS FLOAT) AS longitude,
                    pixel_id,
                    CAST(sanlc_2022 AS TINYINT) AS sanlc_2022
                FROM data
            ) TO '{args.output}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 50000)
        """
        db_conn.execute(export_sql)
        db_conn.close()
        print(f"[OK] Saved to: {args.output}")

        # Cleanup DuckDB buffer
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
    except Exception as e:
        print(f"  [ERROR] Export failed: {e}")
        db_conn.close()

    # Quick schema check
    if os.path.exists(args.output):
        con = duckdb.connect()
        cols = con.execute(
            f"SELECT * FROM '{args.output}' LIMIT 0").df().columns.tolist()
        n = con.execute(
            f"SELECT count(*) FROM '{args.output}'").fetchone()[0]
        dtypes = con.execute(
            f"DESCRIBE SELECT * FROM '{args.output}'").fetchall()
        con.close()
        print(f"\n  Output columns ({len(cols)}): {cols}")
        print(f"  Total rows: {n:,}")
        print(f"  Column types:")
        for name, dtype, *_ in dtypes:
            print(f"    {name:<25} {dtype}")
        size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
