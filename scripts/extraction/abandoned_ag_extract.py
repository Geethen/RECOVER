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
        self.processed_cells = self._load_checkpoints()
        self.lock = Lock()

    def _load_checkpoints(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                return set()
        return set()

    def mark_processed(self, cell_index):
        with self.lock:
            self.processed_cells.add(cell_index)
            self._save_checkpoints()

    def is_processed(self, cell_index):
        return cell_index in self.processed_cells

    def _save_checkpoints(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed_cells), f)

# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================

@retry(max_retries=3, backoff_factor=2)
def process_pixel_batch(batch_fc, output_path, lock, db_conn=None):
    """
    Process a batch of sampled pixels: convert to DataFrame and append to destination.
    Uses DuckDB for Parquet/DB output or standard CSV append.
    """
    try:
        # Use computeFeatures to convert to pandas DataFrame (high-volume endpoint)
        df_result = ee.data.computeFeatures({
            'expression': batch_fc,
            'fileFormat': 'PANDAS_DATAFRAME'
        })
        
        if not df_result.empty:
            with lock:
                if output_path.endswith('.parquet') and db_conn:
                    # Performance Optimization: Attempt insert directly. 
                    # Only create table if it's the very first batch of the session.
                    try:
                        db_conn.execute("INSERT INTO data SELECT * FROM df_result")
                    except (duckdb.CatalogException, duckdb.Error):
                        db_conn.execute("CREATE TABLE data AS SELECT * FROM df_result")
                else:
                    # Legacy CSV logic
                    write_header = not os.path.exists(output_path) or os.path.getsize(output_path) == 0
                    df_result.to_csv(output_path, mode='a', header=write_header, index=False)
            
            return df_result['pixel_id'].tolist() if 'pixel_id' in df_result.columns else list(range(len(df_result)))
        return []
        
    except Exception as e:
        return e

def extract_gpp_for_grid_cell(cell_index, cell_feature, gpp_image, 
                                output_path, shared_lock, db_conn=None, scale=30, batch_size=3000, max_workers=20):
    """
    Extract GPP values for all abandoned ag pixels in one grid cell.
    """
    try:
        cell_geom = cell_feature.geometry()
        
        # Sample pixels in this grid cell
        
        # FAST PRE-CHECK: Check if any pixels exist in this cell at 1km scale
        # This avoids sending 20 requests for empty cells.
        pre_check = gpp_image.select(0).reduceRegion(
            reducer=ee.Reducer.max(),
            geometry=cell_geom,
            scale=100,
            maxPixels=1e9,
            tileScale=4
        )
        # Result is a dictionary. If empty or 0, skip.
        # We need getInfo() here, but it's a tiny request (1 number).
        # Optimization: We check "filled" mask which is band 0 in our logic?
        # Actually gpp_image is masked. So max() will be null if empty.
        pre_val = pre_check.values().get(0).getInfo()
        
        if pre_val is None:
             return True # Empty cell, skip
             
        pixels = gpp_image.sample(
            region=cell_geom,
            scale=scale,
            projection='EPSG:4326',
            geometries=True,
            tileScale=4,
            dropNulls=True
        )
        
        # Add unique IDs
        pixels_with_id = pixels.map(
            lambda ft: ft.set('pixel_id', ft.get('system:index'))
        )

        # OPTIMIZATION: Use random column for efficient batching instead of toList()
        # This avoids the O(N^2) complexity of iterating through large collections
        pixels_sharded = pixels_with_id.randomColumn('batch_rand', seed=42)
        
        # We don't know the exact count, but we can estimate or just use a fixed number of splits
        # For a 50km cell, 100 splits is usually safe (handling up to ~300k-500k pixels)
        # If a split is too large (>5MB payload), fetchFeatures will fail, but we retry.
        n_splits = 20 # Start with 20 splits per cell to keep batches reasonable
        
        batches = []
        step = 1.0 / n_splits
        for k in range(n_splits):
            lower = k * step
            upper = (k + 1) * step
            # Create a filter for this shard
            # We use a custom function to get the batch as a collection
            batch_fc = pixels_sharded.filter(ee.Filter.And(
                ee.Filter.gte('batch_rand', lower),
                ee.Filter.lt('batch_rand', upper)
            ))
            batches.append(batch_fc)
        
        # Process batches with threading and retry logic
        pending_batches = batches
        max_batch_retries = 3
        
        for attempt in range(max_batch_retries + 1):
            if not pending_batches:
                break
            
            if attempt > 0:
                print(f"  [Retrying] Cell {cell_index}: {len(pending_batches)} failed batches (Attempt {attempt}/{max_batch_retries})")
                time.sleep(2 ** attempt)
            
            failed_batches = []
            processed_pixels = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {
                    executor.submit(process_pixel_batch, batch, output_path, shared_lock, db_conn): batch 
                    for batch in pending_batches
                }
                
                # Progress bar per cell
                pbar_desc = f"Cell {cell_index} Batches"
                with tqdm(total=len(pending_batches), desc=pbar_desc, 
                         leave=False, ncols=100) as pbar:
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            result = future.result()
                            if isinstance(result, Exception):
                                # If payload too large, we might need to split this batch further?
                                # For now, we just mark as failed and retry.
                                failed_batches.append(batch)
                            elif isinstance(result, list):
                                processed_pixels += len(result)
                            else:
                                failed_batches.append(batch)
                        except Exception:
                            failed_batches.append(batch)
                        pbar.update(1)
            
            pending_batches = failed_batches
        
        if pending_batches:
            print(f"  [WARNING] Cell {cell_index}: Completed with {len(pending_batches)} failed batches")
            return False
        else:
            return True
            
    except Exception as e:
        print(f"  [ERROR] Error processing cell {cell_index}: {str(e)}")
        return False

def main():
    # Get the project base directory (one level up from scripts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_output = os.path.join(base_dir, "data", "abandoned_ag_gpp_2000_2022_SA.parquet")

    parser = argparse.ArgumentParser(description='Extract GPP data for abandoned agriculture areas.')
    parser.add_argument('--project', type=str, default='ee-gsingh', help='GEE Project ID')
    parser.add_argument('--output', type=str, default=default_output, help='Output path (.parquet or .csv)')
    parser.add_argument('--start_year', type=int, default=2000, help='Start year')
    parser.add_argument('--end_year', type=int, default=2022, help='End year')
    parser.add_argument('--grid_size', type=int, default=50000, help='Grid size in meters')
    parser.add_argument('--batch_size', type=int, default=3000, help='Batch size for extraction')
    parser.add_argument('--max_workers', type=int, default=20, help='Max parallel workers per cell')
    parser.add_argument('--max_cell_workers', type=int, default=5, help='Max parallel grid cells')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode (limit to 2 grid cells)')
    
    args = parser.parse_args()

    # Initialize Earth Engine with High Volume Endpoint
    try:
        ee.Initialize(project=args.project, opt_url='https://earthengine-highvolume.googleapis.com')
        print(f"[OK] Initialized GEE with High Volume Endpoint (Project: {args.project})")
    except Exception as e:
        print(f"Failed to initialize GEE High Volume: {e}")
        ee.Initialize(project=args.project)
        print(f"[OK] Initialized with standard endpoint (Project: {args.project})")

    # Define South Africa boundary early for spatial filtering
    south_africa = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'South Africa'))
    sa_geometry = south_africa.geometry()

    # 1. PREPARE THE ABANDONED AGRICULTURE MASK
    print("\nSTEP 1: Preparing Abandoned Agriculture Mask")
    # Asset location for abandoned ag
    aa_asset = "projects/ee-gsingh/assets/RECOVER/abandoned_ag"
    img = ee.Image(aa_asset).eq(10).selfMask()

    # Connected component analysis to clean up mask
    cc = img.connectedPixelCount(maxSize=50, eightConnected=True)
    min_pixels = 40
    cleaned = img.updateMask(cc.gte(min_pixels))
    cleaned = cleaned.unmask(0).rename('cleaned')

    # Fill small holes
    filled = (cleaned
        .focal_max(radius=1, units='pixels')
        .focal_min(radius=1, units='pixels'))
    print("[OK] Abandoned agriculture mask prepared")

    # 2. LOAD GPP DATA
    print("\nSTEP 2: Loading GPP Data")
    gpp_collection = ee.ImageCollection("projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m")
    years = list(range(args.start_year, args.end_year + 1))
    
    # Create annual GPP composites and stack them
    annual_list = []
    for y in years:
        annual_gpp = gpp_collection.filter(ee.Filter.calendarRange(y, y, 'year')).sum().rename(f'GPP_{y}')
        annual_list.append(annual_gpp)
    
    gpp_stack = ee.Image.cat(annual_list)

    # 2.1 ADD SHORT VEGETATION HEIGHT (GSVH)
    print(f"  Adding Short Vegetation Height ({args.start_year}-{args.end_year})...")
    gsvh_col = ee.ImageCollection("projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m")
    svh_list = []
    for y in years:
        # Get annual mosaic for SVH
        svh_annual = gsvh_col.filter(ee.Filter.calendarRange(y, y, 'year')).mosaic().multiply(0.1).rename(f'SVH_{y}')
        svh_list.append(svh_annual)
    
    svh_stack = ee.Image.cat(svh_list)

    # 2.2 ADD BII (Biodiversity Intactness Index)
    print("  Adding Biodiversity Intactness Index (BII 1km)...")
    bii_col = ee.ImageCollection("projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_1km")
    bii_mask = ee.Image("projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_Mask")
    
    bii_bands = ['Land Use', 'Land Use Intensity', 'BII All',
                 'BII Amphibians', 'BII Birds', 'BII Forbs', 'BII Graminoids',
                 'BII Mammals', 'BII All Plants', 'BII Reptiles', 'BII Trees',
                 'BII All Vertebrates']
    
    bii_img = bii_col.toBands().rename(bii_bands)
    bii_main = bii_img.select('^BII.*').selfMask()
    lc_mask = bii_img.select('Land Use').neq(2).And(bii_img.select('Land Use').neq(5))
    lui = bii_img.select('Land Use Intensity').updateMask(lc_mask)
    bii_processed = bii_main.addBands([bii_img.select('Land Use'), lui]).updateMask(bii_mask)

    # 2.3 ADD SANLC 2022
    print("  Adding SANLC 2022 Land Cover...")
    sanlc = ee.Image("projects/ee-gsingh/assets/RECOVER/sanlc2022_7class").rename('sanlc_2022')

    # 2.4 ADD ECOREGIONS
    print("  Adding Resolve Ecoregions...")
    ecoregions = ee.FeatureCollection("RESOLVE/ECOREGIONS/2017").filterBounds(sa_geometry)
    ecoregions_img = ecoregions.reduceToImage(properties=['ECO_ID'], reducer=ee.Reducer.first()).rename('eco_id')

    # Combine everything into a final multi-band stack
    final_stack = ee.Image.cat([
        gpp_stack,
        svh_stack,
        bii_processed,
        sanlc,
        ecoregions_img,
        ee.Image.pixelLonLat()
    ])
    
    # Mask to only abandoned agriculture pixels
    gpp_masked = final_stack.updateMask(filled.eq(1))
    print(f"[OK] Created final stack with GPP, SVH, BII, SANLC, and Ecoregions")

    # 3. CREATE COVERING GRID for South Africa
    print("\nSTEP 3: Creating Covering Grid (This may take a moment)...")
    start_grid = time.time()
    grid = sa_geometry.bounds().coveringGrid(
        proj=sa_geometry.projection(),
        scale=args.grid_size
    )

    total_cells = int(grid.size().getInfo())
    grid_list = grid.toList(total_cells)
    print(f"[OK] Generated {total_cells} grid cells in {time.time()-start_grid:.1f}s")

    # 4. BATCH EXTRACTION
    print("\nSTEP 4: Extracting GPP Data")
    checkpoint_file = f"{args.output}.checkpoint.json"
    
    # Reset checkpoint if output doesn't exist
    if not os.path.exists(args.output) and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    checkpoint_mgr = CheckpointManager(checkpoint_file)
    
    successful_cells = 0
    skipped_cells = 0
    failed_cells = 0
    shared_lock = Lock()
    
    # Initialize DuckDB if output is Parquet
    db_conn = None
    if args.output.endswith('.parquet'):
        db_path = args.output.replace('.parquet', '.duckdb')
        print(f"\nSTEP 4.1: Syncing DuckDB Buffer ({db_path})")
        db_conn = duckdb.connect(db_path)
        
        # Fast check: does table exist?
        try:
            db_conn.execute("SELECT 1 FROM data LIMIT 1")
            print("  [OK] Found existing buffer table.")
        except:
            if os.path.exists(args.output):
                print(f"  [INFO] Buffer empty. Loading existing Parquet data ({os.path.basename(args.output)})...")
                start_load = time.time()
                db_conn.execute(f"CREATE TABLE data AS SELECT * FROM '{args.output}'")
                print(f"  [OK] Loaded in {time.time()-start_load:.1f}s")
            else:
                print("  [INFO] Starting fresh dataset.")

    # Determine cells to process
    if args.test_mode:
        print("\n*** RUNNING IN TEST MODE: Finding first 2 cells with data... ***")
        cells_with_data = filled.reduceRegions(
            collection=grid,
            reducer=ee.Reducer.anyNonZero(),
            scale=1000
        ).filter(ee.Filter.eq('any', 1)).limit(2)
        
        data_ids = cells_with_data.aggregate_array('system:index').getInfo()
        grid_indices = grid.aggregate_array('system:index').getInfo()
        cells_to_process = [grid_indices.index(did) for did in data_ids if did in grid_indices]
        
        if not cells_to_process:
            print("  [WARNING] No abandoned agriculture pixels found in first search. Using first 2 cells anyway.")
            cells_to_process = [0, 1]
        
        total_to_do = len(cells_to_process)
    else:
        cells_to_process = range(total_cells)
        total_to_do = total_cells

    print(f"Processing in parallel: {args.max_cell_workers} cells, {args.max_workers} batches/cell")
    
    # Process cells in parallel
    with ThreadPoolExecutor(max_workers=args.max_cell_workers) as cell_executor:
        future_to_cell = {}
        
        for i in cells_to_process:
            if checkpoint_mgr.is_processed(i):
                skipped_cells += 1
                continue
            
            # Get cell feature
            cell = ee.Feature(grid_list.get(i))
            
            future = cell_executor.submit(
                extract_gpp_for_grid_cell,
                cell_index=i,
                cell_feature=cell,
                gpp_image=gpp_masked,
                output_path=args.output,
                shared_lock=shared_lock,
                db_conn=db_conn,
                scale=30,
                batch_size=args.batch_size,
                max_workers=args.max_workers
            )
            future_to_cell[future] = i

        # Results as they complete
        with tqdm(total=total_to_do, desc="Total Progress (Cells)", ncols=100) as pbar:
            pbar.update(skipped_cells)
            for future in as_completed(future_to_cell):
                cell_idx = future_to_cell[future]
                try:
                    success = future.result()
                    if success:
                        checkpoint_mgr.mark_processed(cell_idx)
                        successful_cells += 1
                    else:
                        failed_cells += 1
                except Exception as e:
                    print(f"  [ERROR] Unhandled error in cell {cell_idx}: {e}")
                    failed_cells += 1
                pbar.update(1)

    print("\n" + "="*80)
    print("EXTRACTION SUMMARY")
    print(f"Successful cells: {successful_cells}")
    print(f"Skipped cells:    {skipped_cells}")
    print(f"Total cells:      {total_cells}")
    print("="*80)

    # Final Export if using DuckDB
    if db_conn:
        print("\nSTEP 4.2: Exporting Buffer to Final Parquet")
        temp_parquet = args.output + ".tmp"
        db_conn.execute(f"COPY data TO '{temp_parquet}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        db_conn.close()
        
        # Replace old parquet (if any) with new one
        if os.path.exists(args.output):
            os.remove(args.output)
        os.rename(temp_parquet, args.output)
        print(f"[OK] Final Parquet saved to: {args.output}")
        
        # Cleanup temporary DuckDB file
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"[OK] Removed temporary buffer file: {db_path}")
            except Exception as e:
                print(f"[WARNING] Could not remove temporary buffer: {e}")

    # 5. INSPECT AND SAVE SUMMARY
    if os.path.exists(args.output):
        print("\nSTEP 5: Processing Results")
        try:
            if args.output.endswith('.parquet'):
                # Use standard connection for querying external files
                con = duckdb.connect()
                total_pixels = con.execute(f"SELECT count(*) FROM '{args.output}'").fetchone()[0]
                print(f"[OK] Total pixels: {total_pixels:,}")
                
                # Column check
                cols = con.execute(f"DESCRIBE SELECT * FROM '{args.output}' LIMIT 0").fetchall()
                col_names = [c[0] for c in cols]
                gpp_cols = [c for c in col_names if c.startswith('GPP_')]
                
                if gpp_cols:
                    # Summary stats via DuckDB (much faster than pandas)
                    summary_file = args.output.replace('.parquet', '_summary_stats.csv')
                    # Building a dynamic query for statistics
                    stats_queries = [f"mean({c}) as {c}_mean, stddev({c}) as {c}_std" for c in gpp_cols]
                    stats_df = con.execute(f"SELECT {', '.join(stats_queries)} FROM '{args.output}'").df()
                    stats_df.to_csv(summary_file)
                    print(f"[OK] Summary statistics saved to: {summary_file}")
                
                con.close()
            else:
                # Legacy pandas logic for CSV
                df = pd.read_csv(args.output)
                print(f"[OK] Total pixels: {len(df):,}")
                # ... (rest of summary logic simplified for brevity in this block)

            # Check for geometry columns
            if 'longitude' in df.columns and 'latitude' in df.columns:
                print(f"[OK] Found geometry columns (longitude, latitude)")
                print(f"     Extent: {df['longitude'].min():.4f}, {df['latitude'].min():.4f} to {df['longitude'].max():.4f}, {df['latitude'].max():.4f}")
        except Exception as e:
            print(f"  [ERROR] Error during result processing: {e}")

if __name__ == "__main__":
    main()