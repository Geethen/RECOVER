import ee
import pandas as pd
import numpy as np
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm.auto import tqdm

# Initialize Earth Engine with High Volume Endpoint
try:
    ee.Initialize(project='ee-gsingh', opt_url='https://earthengine-highvolume.googleapis.com')
except Exception as e:
    print(f"Failed to initialize GEE High Volume: {e}")
    ee.Initialize(project='ee-gsingh')

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
                    time.sleep(sleep_time)
            return None
        return wrapper
    return decorator

class CheckpointManager:
    def __init__(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        self.processed_cells = self._load_checkpoints()

    def _load_checkpoints(self):
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    return set(json.load(f))
            except json.JSONDecodeError:
                return set()
        return set()

    def mark_processed(self, cell_index):
        self.processed_cells.add(cell_index)
        self._save_checkpoints()

    def is_processed(self, cell_index):
        return cell_index in self.processed_cells

    def _save_checkpoints(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(self.processed_cells), f)

def get_s2_composite(roi, start_date='2022-01-01', end_date='2023-01-01', cs_threshold=0.6):
    """Creates a Sentinel-2 median composite with Cloud Score+ masking and indices."""
    
    # Sentinel-2 Harmonized Collection
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date))
          
    # Cloud Score+ Collection
    cs_plus = (ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start_date, end_date))
    
    # Link collections
    linked = s2.linkCollection(cs_plus, ['cs'])
    
    def mask_clouds_and_add_indices(img):
        # Mask clouds using Cloud Score+ 'cs' band
        is_clear = img.select('cs').gte(cs_threshold)
        
        # Scale
        scaled = img.divide(10000).updateMask(is_clear)
        
        # Indices
        nbr = scaled.normalizedDifference(['B8', 'B12']).rename('NBR')
        ndmi = scaled.normalizedDifference(['B8', 'B11']).rename('NDMI')
        ndwi = scaled.normalizedDifference(['B3', 'B8']).rename('NDWI')
        
        return nbr.addBands([ndmi, ndwi])

    composite = (linked.map(mask_clouds_and_add_indices)
                 .median())
    
    return composite

@retry()
def process_batch(batch_col, composite, output_csv, lock):
    """
    Helper function to process a single batch of points.
    Extracts indices values to points, convert to df, write to csv in append mode.
    """
    try:
        # reduceRegions on batch
        eedf_batch = composite.reduceRegions(
            collection=batch_col,
            reducer=ee.Reducer.first(),
            scale=10
        )
        
        # computeFeatures -> Pandas
        df_result = ee.data.computeFeatures({
            'expression': eedf_batch,
            'fileFormat': 'PANDAS_DATAFRAME'
        })
        
        if not df_result.empty:
            with lock:
                # Append to CSV safely
                # Check header only on first write (if file doesn't exist or is empty)
                write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
                df_result.to_csv(output_csv, mode='a', header=write_header, index=False)
            
            # Return list of processed IDs (unused in grid mode but kept for consistency)
            if 'id' in df_result.columns:
                return df_result['id'].tolist()
            return []
        return []
    except Exception as e:
        return e # Failure

def extract_efficiently(asset_id, output_csv, start_date='2022-01-01', end_date='2023-01-01', grid_scale_meters=50000):
    """
    Extracts indices using a grid-based approach:
    1. Calculate bounds of the asset
    2. Create a covering grid of 50x50km cells
    3. Iterate over cells -> filter points in cell -> reduceRegions -> Export
    """
    
    # 1. Define Asset and Grid
    print(f"Initializing grid-based extraction for {asset_id}...")
    asset = ee.FeatureCollection(asset_id)
    
    # Calculate bounds
    print("Calculating asset bounds...")
    try:
        asset_bounds = asset.geometry().bounds()
    except Exception as e:
        print(f"Error calculating asset bounds: {e}")
        return

    # Generate covering grid (EPSG:3857 for meter-based scale)
    print(f"Generating {(grid_scale_meters/1000):.0f}km grid...")
    sa = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', 'South Africa'))
    grid = asset_bounds.coveringGrid(sa.geometry().projection(), scale=grid_scale_meters)
    
    # Get total grid cells
    total_cells = grid.size().getInfo()
    print(f"Generated {total_cells} grid cells.")
    
    # Convert grid to list to iterate
    grid_list = grid.toList(total_cells)

    # Initialize Checkpoint Manager
    checkpoint_file = f"{output_csv}.checkpoint.json"
    if not os.path.exists(output_csv) and os.path.exists(checkpoint_file):
        print(f"Output CSV {output_csv} not found. Resetting checkpoint {checkpoint_file}...")
        os.remove(checkpoint_file)
        
    checkpoint_mgr = CheckpointManager(checkpoint_file)
    
    # 2. Iterate over Grid Cells
    for i in range(total_cells):
        if checkpoint_mgr.is_processed(i):
            print(f"Cell {i+1}/{total_cells}: Already processed. Skipping.")
            continue

        try:
            cell = ee.Feature(grid_list.get(i))
            cell_geom = cell.geometry()
            
            # Filter points within this cell
            points_in_cell = asset.filterBounds(cell_geom)
            
            # Check size
            points_count = points_in_cell.size().getInfo()
            
            if points_count == 0:
                print(f"Cell {i+1}/{total_cells}: Empty. Skipping.")
                checkpoint_mgr.mark_processed(i)
                continue
            
            print(f"Cell {i+1}/{total_cells}: Found {points_count} points. Processing...")
            
            # 3. Create Composite for this Cell
            composite = get_s2_composite(cell_geom, start_date, end_date)
            
            # 4. Batch Processing
            pts_wid = points_in_cell.map(lambda ft: ft.set('id', ft.get('system:index')))
            
            batch_size = 15
            batches = []
            for j in range(0, points_count, batch_size):
                batch_list = pts_wid.toList(batch_size, j)
                batch_col = ee.FeatureCollection(batch_list)
                batches.append(batch_col)
                
            lock = Lock()
            max_workers = 20
            
            # Retry loop for batches
            pending_batches = batches
            max_batch_retries = 3
            
            for attempt in range(max_batch_retries + 1):
                if not pending_batches:
                    break
                
                if attempt > 0:
                    print(f"Retrying {len(pending_batches)} failed batches (Attempt {attempt}/{max_batch_retries})...")
                    time.sleep(2 ** attempt) # Exponential backoff for retries

                failed_batches = []
                new_ids_count = 0
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Map future to the actual batch object (collection)
                    future_to_batch = {executor.submit(process_batch, batch, composite, output_csv, lock): batch for batch in pending_batches}
                    
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            res = future.result()
                            if isinstance(res, Exception):
                                failed_batches.append(batch)
                                print(f"Batch failed: {res}")
                            elif isinstance(res, list):
                                new_ids_count += len(res)
                            else:
                                # Should not happen based on process_batch return
                                failed_batches.append(batch)
                        except Exception as e:
                            failed_batches.append(batch)
                            print(f"Batch execution exception: {e}")
                
                pending_batches = failed_batches
            
            if pending_batches:
                print(f"Cell {i+1}/{total_cells}: Completed with {len(pending_batches)} failed batches.")
                # Decision: Mark processed even if some batches failed? 
                # Or keep it unprocessed?
                # User asked to resume/retry. If we mark it processed, we lose the failed ones forever unless we have logic to partial resume.
                # For now, let's NO mark processed if there are failures, so it retries the whole cell next time.
                print(f"Cell {i+1}/{total_cells}: NOT marking as processed due to failures.")
            else:
                print(f"Cell {i+1}/{total_cells}: Successfully processed all batches.")
                checkpoint_mgr.mark_processed(i)
            
            print("-" * 30)

        except Exception as e:
            print(f"Error processing Cell {i+1}: {e}")
    
    print("Grid extraction complete.")

if __name__ == "__main__":
    ASSET_ID = 'projects/ee-gsingh/assets/RECOVER/fscs_aef_samples'
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_CSV = os.path.join(base_dir, "data", "extracted_indices.csv")
    
    print(f"Starting extraction for asset: {ASSET_ID}")
    print(f"Output will be saved to: {OUTPUT_CSV}")
    
    extract_efficiently(ASSET_ID, OUTPUT_CSV)