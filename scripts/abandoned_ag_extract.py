import ee
import pandas as pd
import numpy as np
import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm.auto import tqdm

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize Earth Engine with High Volume Endpoint
try:
    ee.Initialize(project='ee-gsingh', opt_url='https://earthengine-highvolume.googleapis.com')
    print("✓ Initialized with High Volume Endpoint")
except Exception as e:
    print(f"Failed to initialize GEE High Volume: {e}")
    ee.Initialize(project='ee-gsingh')
    print("✓ Initialized with standard endpoint")

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
# 1. PREPARE THE ABANDONED AGRICULTURE MASK
# ============================================================================

print("="*80)
print("STEP 1: Preparing Abandoned Agriculture Mask")
print("="*80)

# Create the filled mask
img = ee.Image("projects/ee-gsingh/assets/RECOVER/abandoned_ag").eq(10).selfMask()

# Connected component analysis
cc = img.connectedPixelCount(maxSize=50, eightConnected=True)

# Remove small patches
min_pixels = 40
cleaned = img.updateMask(cc.gte(min_pixels))
cleaned = cleaned.unmask(0).rename('cleaned')

# Fill small holes
filled = (cleaned
    .focal_max(radius=1, units='pixels')
    .focal_min(radius=1, units='pixels'))

print("✓ Abandoned agriculture mask prepared")

# ============================================================================
# 2. LOAD GPP IMAGE COLLECTION (2000-2022)
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Loading GPP Data (2000-2022)")
print("="*80)

# Load the GPP collection
gpp_collection = ee.ImageCollection("projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m")

# Filter for years 2000-2022 and get annual composites
years = list(range(2000, 2023))

# Stack all years into a single multi-band image
gpp_stack = gpp_images.toBands()
print(f"\n✓ Stacked {len(years)} years into multi-band image")

# Mask to only abandoned agriculture pixels
gpp_masked = gpp_stack.updateMask(filled.eq(1))
print("✓ Masked to abandoned agriculture pixels")

# ============================================================================
# 3. CREATE COVERING GRID FOR SOUTH AFRICA
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Creating Covering Grid")
print("="*80)

# Define South Africa boundary
south_africa = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(
    ee.Filter.eq('ADM0_NAME', 'South Africa')
)
sa_geometry = south_africa.geometry()

# Get bounds of abandoned agriculture areas
print("Calculating bounds...")
# Fallback to SA bounds if above fails
aa_bounds = sa_geometry.bounds()

# Create covering grid using EPSG:3857 for meter-based scale
grid_scale_meters = 50000  # 50km grid cells
print(f"Generating {grid_scale_meters/1000:.0f}km grid...")

grid = aa_bounds.coveringGrid(
    proj=sa_geometry.projection(),
    scale=grid_scale_meters
)

# Get total grid cells
total_cells = grid.size().getInfo()
print(f"✓ Generated {total_cells} grid cells")

# Convert grid to list for iteration
grid_list = grid.toList(total_cells)

# ============================================================================
# 4. BATCH EXTRACTION WITH HIGH-VOLUME ENDPOINT
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Extracting GPP Data with High-Volume Endpoint")
print("="*80)

output_csv = "abandoned_ag_gpp_2000_2022_SA.csv"
checkpoint_file = f"{output_csv}.checkpoint.json"

# Reset checkpoint if output CSV doesn't exist
if not os.path.exists(output_csv) and os.path.exists(checkpoint_file):
    print(f"Output CSV not found. Resetting checkpoint...")
    os.remove(checkpoint_file)

# Initialize checkpoint manager
checkpoint_mgr = CheckpointManager(checkpoint_file)

@retry(max_retries=3, backoff_factor=2)
def process_pixel_batch(batch_fc, gpp_image, output_csv, lock):
    """
    Process a batch of pixels: extract GPP values, convert to DataFrame, append to CSV.
    
    Args:
        batch_fc: ee.FeatureCollection of point samples
        gpp_image: Multi-band GPP image
        output_csv: Path to output CSV
        lock: Threading lock for file writing
        
    Returns:
        List of processed IDs on success, Exception on failure
    """
    try:
        # Sample GPP values at point locations
        sampled = gpp_image.sampleRegions(
            collection=batch_fc,
            scale=30,
            projection='EPSG:4326',
            tileScale=4
        )
        
        # Use computeFeatures to convert to pandas DataFrame (high-volume endpoint)
        df_result = ee.data.computeFeatures({
            'expression': sampled,
            'fileFormat': 'PANDAS_DATAFRAME'
        })
        
        if not df_result.empty:
            # Write to CSV with thread safety
            with lock:
                write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
                df_result.to_csv(output_csv, mode='a', header=write_header, index=False)
            
            # Return IDs of processed features
            if 'system:index' in df_result.columns:
                return df_result['system:index'].tolist()
            return list(range(len(df_result)))
        return []
        
    except Exception as e:
        return e  # Return exception to signal failure

def extract_gpp_for_grid_cell(cell_index, cell_feature, gpp_image, filled_mask, 
                                output_csv, scale=30, batch_size=1000, max_workers=20):
    """
    Extract GPP values for all abandoned ag pixels in one grid cell.
    
    Args:
        cell_index: Index of the grid cell
        cell_feature: ee.Feature representing the grid cell
        gpp_image: Multi-band GPP image
        filled_mask: Binary mask of abandoned agriculture
        output_csv: Path to output CSV
        scale: Sampling resolution in meters
        batch_size: Number of pixels per batch
        max_workers: Number of parallel threads
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cell_geom = cell_feature.geometry()
        
        # Sample pixels in this grid cell
        print(f"  Sampling pixels in cell...")
        pixels = gpp_image.sample(
            region=cell_geom,
            scale=scale,
            projection='EPSG:4326',
            geometries=True,
            tileScale=4
        )
        
        # Add unique IDs
        pixels_with_id = pixels.map(
            lambda ft: ft.set('pixel_id', ft.get('system:index'))
        )
        
        # Check size
        pixels_count = pixels_with_id.size().getInfo()
        
        if pixels_count == 0:
            print(f"  No abandoned agriculture pixels in this cell.")
            return True
        
        print(f"  Found {pixels_count:,} pixels. Creating batches...")
        
        # Create batches
        batches = []
        for j in range(0, pixels_count, batch_size):
            batch_list = pixels_with_id.toList(batch_size, j)
            batch_fc = ee.FeatureCollection(batch_list)
            batches.append(batch_fc)
        
        print(f"  Created {len(batches)} batches of ~{batch_size} pixels each")
        
        # Process batches with threading and retry logic
        lock = Lock()
        pending_batches = batches
        max_batch_retries = 3
        
        for attempt in range(max_batch_retries + 1):
            if not pending_batches:
                break
            
            if attempt > 0:
                print(f"  Retrying {len(pending_batches)} failed batches (Attempt {attempt}/{max_batch_retries})...")
                time.sleep(2 ** attempt)
            
            failed_batches = []
            processed_pixels = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                future_to_batch = {
                    executor.submit(process_pixel_batch, batch, gpp_image, output_csv, lock): batch 
                    for batch in pending_batches
                }
                
                # Collect results with progress bar
                with tqdm(total=len(pending_batches), desc=f"  Processing batches", 
                         leave=False, ncols=100) as pbar:
                    for future in as_completed(future_to_batch):
                        batch = future_to_batch[future]
                        try:
                            result = future.result()
                            if isinstance(result, Exception):
                                failed_batches.append(batch)
                                print(f"  Batch failed: {str(result)[:80]}")
                            elif isinstance(result, list):
                                processed_pixels += len(result)
                            else:
                                failed_batches.append(batch)
                        except Exception as e:
                            failed_batches.append(batch)
                            print(f"  Batch execution error: {str(e)[:80]}")
                        pbar.update(1)
            
            pending_batches = failed_batches
            print(f"  Processed {processed_pixels:,} pixels in this attempt")
        
        if pending_batches:
            print(f"  ⚠ Completed with {len(pending_batches)} failed batches")
            return False  # Don't mark as processed if there are failures
        else:
            print(f"  ✓ Successfully processed all {pixels_count:,} pixels")
            return True
            
    except Exception as e:
        print(f"  ✗ Error processing cell: {str(e)}")
        return False

# ============================================================================
# 5. ITERATE OVER GRID CELLS
# ============================================================================

print(f"\nProcessing {total_cells} grid cells...")
print(f"Output: {output_csv}")
print(f"Checkpoint: {checkpoint_file}")
print("-"*80)

successful_cells = 0
skipped_cells = 0
failed_cells = 0

for i in range(total_cells):
    print(f"\nCell {i+1}/{total_cells}:")
    
    # Check if already processed
    if checkpoint_mgr.is_processed(i):
        print(f"  ✓ Already processed. Skipping.")
        skipped_cells += 1
        continue
    
    try:
        # Get cell geometry
        cell = ee.Feature(grid_list.get(i))
        
        # Extract GPP for this cell
        success = extract_gpp_for_grid_cell(
            cell_index=i,
            cell_feature=cell,
            gpp_image=gpp_masked,
            filled_mask=filled,
            output_csv=output_csv,
            scale=30,
            batch_size=1000,  # Adjust based on memory
            max_workers=20     # Adjust based on rate limits
        )
        
        if success:
            checkpoint_mgr.mark_processed(i)
            successful_cells += 1
            print(f"  ✓ Cell marked as complete")
        else:
            failed_cells += 1
            print(f"  ⚠ Cell NOT marked as complete due to failures")
        
    except Exception as e:
        failed_cells += 1
        print(f"  ✗ Cell processing error: {str(e)}")
    
    print("-"*80)

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
print(f"Successful cells: {successful_cells}")
print(f"Skipped cells:    {skipped_cells}")
print(f"Failed cells:     {failed_cells}")
print(f"Total cells:      {total_cells}")
print("="*80)

# ============================================================================
# 6. LOAD AND INSPECT RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Loading and Inspecting Results")
print("="*80)

if os.path.exists(output_csv):
    # Load the complete dataset
    print("Loading CSV...")
    df = pd.read_csv(output_csv)
    
    print("\nDATA SUMMARY")
    print("-"*80)
    print(f"Total pixels extracted: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")
    
    # Show column names
    gpp_cols = [col for col in df.columns if col.startswith('GPP_')]
    other_cols = [col for col in df.columns if not col.startswith('GPP_')]
    
    print(f"\nGPP columns ({len(gpp_cols)}): {gpp_cols[:5]}...{gpp_cols[-2:]}")
    print(f"Other columns ({len(other_cols)}): {other_cols}")
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing values
    print(f"\nMissing values per GPP column:")
    missing = df[gpp_cols].isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values in GPP columns!")
    
    # Basic statistics for GPP columns
    if len(gpp_cols) > 0:
        print(f"\nGPP Statistics - {gpp_cols[0]} (first year):")
        print(df[gpp_cols[0]].describe())
        
        print(f"\nGPP Statistics - {gpp_cols[-1]} (last year):")
        print(df[gpp_cols[-1]].describe())
    
    # Check spatial distribution
    if 'longitude' in df.columns and 'latitude' in df.columns:
        print(f"\nSpatial extent:")
        print(f"  Longitude: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
        print(f"  Latitude: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    elif '.geo' in df.columns:
        print(f"\nGeometry column found: .geo")
    
    # ========================================================================
    # 7. SAVE SUMMARY STATISTICS
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 7: Saving Summary Statistics")
    print("="*80)
    
    # Calculate and save summary stats
    if len(gpp_cols) > 0:
        summary_stats = df[gpp_cols].describe()
        summary_file = output_csv.replace('.csv', '_summary_stats.csv')
        summary_stats.to_csv(summary_file)
        print(f"✓ Summary statistics saved to: {summary_file}")
        
        # Calculate temporal trends
        gpp_means = df[gpp_cols].mean()
        trends_df = pd.DataFrame({
            'year': [int(col.split('_')[1]) for col in gpp_cols],
            'mean_gpp': gpp_means.values
        })
        trends_file