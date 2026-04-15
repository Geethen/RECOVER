import ee
import time
import os
import argparse
import pandas as pd
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Initialize Earth Engine with High Volume Endpoint
try:
    ee.Initialize(project='ee-gsingh', opt_url='https://earthengine-highvolume.googleapis.com')
    print("[OK] Initialized GEE with High Volume Endpoint")
except Exception as e:
    print(f"Failed to initialize GEE High Volume: {e}")
    ee.Initialize()
    print("[OK] Initialized with standard endpoint")

# A small region for fetching pixels
test_geometry = ee.Geometry.Polygon(
    [[[18.0, -34.0],
      [18.5, -34.0],
      [18.5, -34.5],
      [18.0, -34.5]]]
)

def get_test_image():
    """Create a sample image to extract, e.g. elevation and slope."""
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    img = ee.Image.cat([dem, slope]).rename(['elevation', 'slope'])
    return img

def create_grid(region, scale=10000):
    """Create a grid of cells for parallel extraction."""
    grid = region.coveringGrid(proj='EPSG:4326', scale=scale)
    return grid

def method_xee(image_proxy, cell_geom):
    """
    Method 1: xee
    """
    start_time = time.time()
    try:
        ds = xr.open_dataset(
            image_proxy,
            engine='ee',
            crs='EPSG:4326',
            scale=30,
            geometry=cell_geom
        )
        df = ds.to_dataframe().dropna().reset_index()
        elapsed = time.time() - start_time
        return len(df), elapsed
    except Exception as e:
        print(f"Error in xee: {e}")
        return 0, time.time() - start_time

def method_compute_features(image_proxy, cell_geom):
    """
    Method 2: ee.data.computeFeatures (Pandas Converter)
    """
    start_time = time.time()
    try:
        # Sample pixels in this grid cell
        pixels = image_proxy.sample(
            region=cell_geom,
            scale=30,
            projection='EPSG:4326',
            geometries=True,
            dropNulls=True
        )
        # Use computeFeatures to convert to pandas DataFrame
        df = ee.data.computeFeatures({
            'expression': pixels,
            'fileFormat': 'PANDAS_DATAFRAME'
        })
        elapsed = time.time() - start_time
        return len(df) if df is not None else 0, elapsed
    except Exception as e:
        # computeFeatures sometimes fails if payload is too large.
        print(f"Error in computeFeatures: {e}")
        return 0, time.time() - start_time

def method_compute_pixels(image_proxy, cell_geom):
    """
    Method 3: ee.data.computePixels (NumPy structured array converter)
    """
    start_time = time.time()
    try:
        image_clipped = image_proxy.clipToBoundsAndScale(
            geometry=cell_geom,
            scale=30
        )
        npy_data = ee.data.computePixels({
            'expression': image_clipped,
            'fileFormat': 'NUMPY_NDARRAY'
        })
        df = pd.DataFrame(npy_data)
        elapsed = time.time() - start_time
        return len(df), elapsed
    except Exception as e:
        print(f"Error in computePixels: {e}")
        return 0, time.time() - start_time

def run_benchmark(method_func, image_proxy, grid_cells, max_workers=5):
    """
    Run the chosen method in parallel across grid cells.
    """
    total_pixels = 0
    total_time = 0.0
    
    start_wall_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cell = {
            executor.submit(method_func, image_proxy, cell.geometry()): cell 
            for cell in grid_cells
        }
        
        for future in tqdm(as_completed(future_to_cell), total=len(grid_cells), desc=method_func.__name__):
            try:
                pixels, elapsed = future.result()
                total_pixels += pixels
                total_time += elapsed
            except Exception as e:
                print(f"Batch failed: {e}")
    
    wall_time = time.time() - start_wall_time
    return total_pixels, wall_time

def main():
    parser = argparse.ArgumentParser(description='Benchmark GEE Extraction Methods')
    parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
    args = parser.parse_args()

    print(f"Starting test with {args.workers} workers...")
    
    img = get_test_image()
    # Create 4 relatively small grid cells for the test to keep it quick
    grid = create_grid(test_geometry, scale=20000)
    grid_cells = [ee.Feature(f) for f in grid.toList(4).getInfo()]

    print(f"Created {len(grid_cells)} grid cells for testing.")

    results = []

    # 1. Benchmark computeFeatures
    print("\n--- Benchmarking method_compute_features ---")
    px1, t1 = run_benchmark(method_compute_features, img, grid_cells, args.workers)
    results.append({'Method': 'computeFeatures', 'Pixels': px1, 'Wall Time (s)': t1})

    # 2. Benchmark computePixels
    print("\n--- Benchmarking method_compute_pixels ---")
    px2, t2 = run_benchmark(method_compute_pixels, img, grid_cells, args.workers)
    results.append({'Method': 'computePixels', 'Pixels': px2, 'Wall Time (s)': t2})

    # 3. Benchmark xee
    print("\n--- Benchmarking method_xee ---")
    px3, t3 = run_benchmark(method_xee, img, grid_cells, args.workers)
    results.append({'Method': 'xee', 'Pixels': px3, 'Wall Time (s)': t3})

    print("\n==============================")
    print("BENCHMARK RESULTS")
    print("==============================")
    df_results = pd.DataFrame(results)
    print(df_results.to_markdown(index=False))
    
    # Save results to a markdown file
    results_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_results.md')
    with open(results_file, 'w') as f:
        f.write("# Benchmark Results\n\n")
        f.write("Comparison of Earth Engine data extraction methods using 4 parallel workers on a simple raster.\n\n")
        f.write(df_results.to_markdown(index=False))
        f.write("\n")
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    main()
