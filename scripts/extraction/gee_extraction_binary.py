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

def get_natural_mask(roi):
    """Creates a binary mask of natural vs. not-natural areas."""

    # SBTN Natural Lands
    sbtn = ee.Image("WRI/SBTN/naturalLands/v1_1/2020")
    sbtn_nat = sbtn.select('natural')

    # GHM v3 (Human Modification)
    ghm_col = ee.ImageCollection("projects/sat-io/open-datasets/GHM/HM_2022_90M")
    ghm = ghm_col.select(0).filterBounds(roi).first()
    ghm_nat = ghm.lte(0.1).rename('natural')

    # BII (Biodiversity Intactness Index)
    bii1km_col = ee.ImageCollection("projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_1km")
    bii_mask = ee.Image("projects/earthengine-legacy/assets/projects/sat-io/open-datasets/BII/BII_Mask")

    bands1km = ee.List(['Land Use', 'Land Use Intensity', 'BII All',
    'BII Amphibians', 'BII Birds', 'BII Forbs', 'BII Graminoids',
    'BII Mammals', 'BII All Plants', 'BII Reptiles', 'BII Trees',
    'BII All Vertebrates'])

    bii1km = bii1km_col.toBands().rename(bands1km)
    biionekm = bii1km.select('^BII.*').selfMask()
    lcMask1km = bii1km.select('Land Use').neq(2).And(bii1km.select('Land Use').neq(5))
    LUI1km = bii1km.select('Land Use Intensity').updateMask(lcMask1km)
    bii1km_processed = biionekm.addBands([bii1km.select('Land Use'), LUI1km]).updateMask(bii_mask)

    bii_nat = bii1km_processed.select('BII All').gte(0.7).rename('natural')

    # Natural Forests
    probabilities = ee.ImageCollection(
        'projects/nature-trace/assets/forest_typology/natural_forest_2020_v1_0_collection'
    ).mosaic().select('B0')
    nf_nat = probabilities.gte(0.52).rename('natural')

    # Combine masks
    natural_mask = sbtn_nat.Or(nf_nat).And(ghm_nat).And(bii_nat).rename('natural')

    return natural_mask

@retry()
def process_batch(batch_col, natural_mask, output_csv, lock):
    """
    Helper function to process a single batch of points.
    Extracts binary natural/not-natural label to points, converts to df, and writes to csv.
    """
    try:
        # reduceRegions on batch
        eedf_batch = natural_mask.unmask(0).reduceRegions(
            collection=batch_col,
            reducer=ee.Reducer.first(),
            scale=natural_mask.projection().nominalScale()
        )

        # computeFeatures -> Pandas
        df_result = ee.data.computeFeatures({
            'expression': eedf_batch,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        if not df_result.empty:
            # Rename the 'first' column to 'natural'
            if 'first' in df_result.columns:
                df_result = df_result.rename(columns={'first': 'natural'})

            with lock:
                # Append to CSV safely
                write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
                df_result.to_csv(output_csv, mode='a', header=write_header, index=False)

            if 'id' in df_result.columns:
                return df_result['id'].tolist()
            return []
        return []
    except Exception as e:
        return e # Failure

def extract_efficiently(asset_id, output_csv, grid_scale_meters=50000):
    """
    Extracts binary natural labels using a grid-based approach.
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

            # 3. Create Natural Mask for this Cell
            natural_mask = get_natural_mask(cell_geom)

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
                    future_to_batch = {executor.submit(process_batch, batch, natural_mask, output_csv, lock): batch for batch in pending_batches}

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
    # NOTE: Extraction appends to a staging CSV, then convert_csv_to_parquet.py
    # converts it to extracted_natural_labels.parquet with optimised dtypes.
    OUTPUT_CSV = os.path.join(base_dir, "data", "extracted_natural_labels.csv")

    print(f"Starting extraction for asset: {ASSET_ID}")
    print(f"Output will be saved to: {OUTPUT_CSV}")

    extract_efficiently(ASSET_ID, OUTPUT_CSV)