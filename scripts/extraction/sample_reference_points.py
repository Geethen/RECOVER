"""
Sample natural + transformed reference points for ecoregions missing from
the original fscs_aef_samples asset.

Pipeline (replicates ecosystem_condition_2022.ipynb, optimised per benchmark):
  1. Clip ecoregion to SA boundary → fewer grid cells
  2. FSCS (Feature Space Coverage Sampling) on AlphaEarth 2022 within
     10 km grid cells — KMeans clustering, pick pixel closest to centroid
  3. Extract at sampled points: embeddings (A00–A63) + natural label +
     GPP/SVH (2000–2022) + BII + SANLC + eco_id — single reduceRegions call
  4. Parallel cell processing with ThreadPoolExecutor

Benchmark optimisations applied (from benchmark_results.md):
  - reduce(ee.Reducer.first()) instead of mosaic() for AlphaEarth
  - ee.Geometry() instead of ee.Feature() for grid cells
  - 10 km grid cells (matches original eco81 density ~100 pts/cell)
  - max_workers capped at n_cells
  - High-volume endpoint

Usage:
  python scripts/extraction/sample_reference_points.py --eco_id 40
  python scripts/extraction/sample_reference_points.py --all
  python scripts/extraction/sample_reference_points.py --eco_id 40 --n_clusters 200
"""

import argparse
import ee
import json
import os
import time
import duckdb
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm.auto import tqdm

# ── Constants ───────────────────────────────────────────────────────
GRID_SCALE = 10_000          # 10 km cells (matches original eco81 extraction density)
N_CLUSTERS = 100             # FSCS clusters per cell (same as original)
N_INIT_POINTS = 500          # random init points for KMeans
FSCS_BANDS = 10              # use first N embedding bands for clustering (speed)
SCALE = 10                   # metres (AlphaEarth native)
SEED = 42
MAX_WORKERS = 20
START_YEAR = 2000
END_YEAR = 2022

# All SA ecoregions needing reference data
ALL_ECOS = [40, 89, 90, 110, 16, 102, 19, 94, 15, 116, 65,
            41, 38, 97, 48, 101, 88]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ── GEE Helpers ─────────────────────────────────────────────────────
def init_gee(project="ee-gsingh"):
    try:
        ee.Initialize(project=project,
                      opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception:
        ee.Initialize(project=project)
    print(f"[OK] GEE initialized (project={project})")


def retry_gee(func, max_retries=3, backoff=2):
    """Call func(), retry on failure with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = backoff ** (attempt + 1)
            print(f"    Retry {attempt+1}/{max_retries} after {wait}s: {e}")
            time.sleep(wait)


# ── FSCS (from ecosystem_condition_2022.ipynb) ──────────────────────
def fscs(covariates, n_clusters, n_init_points, geometry, scale=10, seed=42):
    """
    Feature Space Coverage Sampling on AlphaEarth embeddings.
    Returns (clusters_image, sample_points_fc).

    KMeans → per-cluster centroids → find closest pixel → sample 1 per cluster.
    """
    clipped = covariates.clip(geometry)

    # Random init points → sample image at those locations
    init_points = ee.FeatureCollection.randomPoints(
        region=geometry, points=n_init_points, seed=seed, maxError=scale)
    init_data = clipped.sampleRegions(collection=init_points, scale=scale)

    # Train KMeans
    clusterer = ee.Clusterer.wekaKMeans(
        nClusters=n_clusters, init=1, seed=seed)
    clusterer = clusterer.train(init_data, clipped.bandNames())
    clusters = clipped.cluster(clusterer)

    # Per-cluster centroids via grouped mean
    cluster_seq = ee.List.sequence(0, n_clusters - 1)
    centroids_result = init_data.cluster(clusterer).reduceColumns(
        selectors=clipped.bandNames().add('cluster'),
        reducer=ee.Reducer.mean().repeat(
            clipped.bandNames().size()).group(
            groupField=clipped.bandNames().size(), groupName='cluster'))

    # Convert grouped list to ee.Dictionary keyed by cluster id for direct
    # lookup — ee.List.filter() fails with "should refer to 'item'" on dicts
    groups_list = ee.List(centroids_result.get('groups'))

    centroid_dict = ee.Dictionary.fromLists(
        groups_list.map(lambda g: ee.Number(
            ee.Dictionary(g).get('cluster')).format('%d')),
        groups_list.map(lambda g: ee.Dictionary(g).get('mean')))

    # Euclidean distance from centroid per cluster — band named 'diff'
    # so toBands() produces predictable keys: '0_diff', '1_diff', ...
    def cluster_diff(cluster_id):
        cluster_id = ee.Number(cluster_id).toInt()
        means = ee.List(centroid_dict.get(cluster_id.format('%d')))
        centroid_img = ee.Image.constant(means).rename(clipped.bandNames())
        diff = clipped.subtract(centroid_img).pow(2).reduce(
            ee.Reducer.sum()).sqrt()
        return diff.updateMask(clusters.eq(cluster_id)).rename('diff')

    diffs = ee.ImageCollection(cluster_seq.map(cluster_diff))
    diffs_banded = diffs.toBands()

    # Find global min distance per cluster — tileScale=4 keeps memory
    # within GEE limits for dense cells (millions of pixels × 100 clusters)
    sample_mins = diffs_banded.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=geometry, scale=scale, maxPixels=1e13, tileScale=4)

    # Mask: only pixels closest to their cluster centroid
    def extract_samples(cluster_id):
        cluster_id = ee.Number(cluster_id).toInt()
        band_key = cluster_id.format('%d').cat('_diff')
        closest_val = sample_mins.getNumber(band_key)
        points = clusters.updateMask(
            diffs_banded.select([cluster_id]).eq(
                ee.Image.constant(closest_val))).rename('points')
        return points

    to_sample = ee.Image(ee.ImageCollection(
        cluster_seq.map(extract_samples)).max()).rename('toSample')
    samples = to_sample.stratifiedSample(
        numPoints=1, classBand='toSample', region=geometry,
        scale=scale, seed=seed, geometries=True)

    return clusters, samples


# ── Natural Mask (from gee_extraction_binary.py) ────────────────────
def get_natural_mask(roi):
    """
    Multi-criteria natural/transformed mask.
    Returns ee.Image band 'natural' (1=natural, 0=transformed).
    Logic: (SBTN OR NaturalForest) AND GHM≤0.1 AND BII≥0.7
    """
    # SBTN Natural Lands
    sbtn_nat = ee.Image("WRI/SBTN/naturalLands/v1_1/2020").select('natural')

    # GHM v3 (Human Modification)
    ghm = ee.ImageCollection(
        "projects/sat-io/open-datasets/GHM/HM_2022_90M"
    ).select(0).filterBounds(roi).first()
    ghm_nat = ghm.lte(0.1).rename('natural')

    # BII
    bii1km_col = ee.ImageCollection(
        "projects/earthengine-legacy/assets/projects/"
        "sat-io/open-datasets/BII/BII_1km")
    bii_mask = ee.Image(
        "projects/earthengine-legacy/assets/projects/"
        "sat-io/open-datasets/BII/BII_Mask")
    bands1km = ee.List([
        'Land Use', 'Land Use Intensity', 'BII All',
        'BII Amphibians', 'BII Birds', 'BII Forbs', 'BII Graminoids',
        'BII Mammals', 'BII All Plants', 'BII Reptiles', 'BII Trees',
        'BII All Vertebrates'])
    bii1km = bii1km_col.toBands().rename(bands1km)
    bii_nat = bii1km.select('BII All').updateMask(bii_mask).gte(0.7) \
        .rename('natural')

    # Natural Forests
    nf = ee.ImageCollection(
        "projects/nature-trace/assets/forest_typology/"
        "natural_forest_2020_v1_0_collection").mosaic().select('B0')
    nf_nat = nf.gte(0.52).rename('natural')

    return sbtn_nat.Or(nf_nat).And(ghm_nat).And(bii_nat).rename('natural')


# ── Image Stacks ────────────────────────────────────────────────────
def build_extraction_stack(roi):
    """
    Single combined image stack for one-shot extraction:
    AlphaEarth (64 bands) + natural mask + GPP (23) + SVH (23) + BII (12)
    + SANLC + eco_id + lat/lon
    """
    sa = ee.FeatureCollection(
        "USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.eq('country_na', 'South Africa'))
    sa_geometry = sa.geometry()

    # AlphaEarth 2022 — reduce(first) not mosaic() (benchmark gotcha #4)
    aef22 = ee.ImageCollection(
        "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    ).filterDate("2022-01-01", "2023-01-01").reduce(
        ee.Reducer.first()
    ).regexpRename('_first$', '')

    # Natural mask
    natural_mask = get_natural_mask(roi)

    # GPP (2000–2022) — Global Pasture Watch uGPP 30m, annual sum
    gpp_col = ee.ImageCollection(
        "projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m")
    gpp_bands = []
    for yr in range(START_YEAR, END_YEAR + 1):
        annual = gpp_col.filter(
            ee.Filter.calendarRange(yr, yr, 'year')
        ).sum().rename(f"GPP_{yr}")
        gpp_bands.append(annual)
    gpp_stack = ee.Image.cat(gpp_bands)

    # SVH (2000–2022) — Global Pasture Watch Short Vegetation Height 30m
    svh_col = ee.ImageCollection(
        "projects/global-pasture-watch/assets/gsvh-30m/v1/"
        "short-veg-height_m")
    svh_bands = []
    for yr in range(START_YEAR, END_YEAR + 1):
        annual = svh_col.filter(
            ee.Filter.calendarRange(yr, yr, 'year')
        ).median().multiply(0.1).rename(f"SVH_{yr}")
        svh_bands.append(annual)
    svh_stack = ee.Image.cat(svh_bands)

    # BII
    bii1km_col = ee.ImageCollection(
        "projects/earthengine-legacy/assets/projects/"
        "sat-io/open-datasets/BII/BII_1km")
    bii_mask = ee.Image(
        "projects/earthengine-legacy/assets/projects/"
        "sat-io/open-datasets/BII/BII_Mask")
    bands1km = ee.List([
        'Land Use', 'Land Use Intensity', 'BII All',
        'BII Amphibians', 'BII Birds', 'BII Forbs', 'BII Graminoids',
        'BII Mammals', 'BII All Plants', 'BII Reptiles', 'BII Trees',
        'BII All Vertebrates'])
    bii1km = bii1km_col.toBands().rename(bands1km)
    biionekm = bii1km.select('^BII.*').selfMask()
    lc_mask = bii1km.select('Land Use').neq(2).And(
        bii1km.select('Land Use').neq(5))
    lui = bii1km.select('Land Use Intensity').updateMask(lc_mask)
    bii_processed = biionekm.addBands(
        [bii1km.select('Land Use'), lui]).updateMask(bii_mask)

    # SANLC
    sanlc = ee.Image(
        "projects/ee-gsingh/assets/RECOVER/sanlc2022_7class"
    ).rename("sanlc_2022")

    # Ecoregions
    ecoregions = ee.FeatureCollection(
        "RESOLVE/ECOREGIONS/2017").filterBounds(sa_geometry)
    eco_img = ecoregions.reduceToImage(
        properties=['ECO_ID'], reducer=ee.Reducer.first()).rename('eco_id')

    return ee.Image.cat([
        aef22, natural_mask.unmask(0), gpp_stack, svh_stack,
        bii_processed, sanlc, eco_img, ee.Image.pixelLonLat()
    ])


# ── Main Pipeline per Ecoregion ─────────────────────────────────────
def run_ecoregion(eco_id, n_clusters=N_CLUSTERS, max_workers=MAX_WORKERS,
                  grid_scale=GRID_SCALE):
    """
    For one ecoregion:
      1. FSCS sampling within 10km grid cells (clipped to ecoregion)
      2. Single-shot extraction: embeddings + natural + GPP/SVH/BII/SANLC
      3. Parallel cell processing
      4. Save to data/ref_samples_eco{eco_id}.parquet
    """
    output_path = os.path.join(DATA_DIR, f"ref_samples_eco{eco_id}.parquet")
    checkpoint_file = os.path.join(
        DATA_DIR, f"ref_samples_eco{eco_id}.checkpoint.json")

    if os.path.exists(output_path) and not os.path.exists(checkpoint_file):
        existing = duckdb.sql(
            f"SELECT count(*) FROM '{output_path}'").fetchone()[0]
        n_nat = duckdb.sql(
            f'SELECT count(*) FROM \'{output_path}\' '
            f'WHERE "natural" = 1').fetchone()[0]
        print(f"  [SKIP] {output_path} exists "
              f"({existing:,} rows, {n_nat:,} natural)")
        return

    print(f"\n{'#' * 70}")
    print(f"# ECOREGION {eco_id}")
    print(f"{'#' * 70}")

    # ── Ecoregion geometry clipped to SA ──
    sa = ee.FeatureCollection(
        "USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.eq('country_na', 'South Africa'))
    ecoregions = ee.FeatureCollection(
        "RESOLVE/ECOREGIONS/2017").filterBounds(sa.geometry())
    ecoregion = ecoregions.filter(ee.Filter.eq('ECO_ID', eco_id))
    eco_name = retry_gee(
        lambda: ecoregion.first().get('ECO_NAME').getInfo())
    print(f"  Name: {eco_name}")

    # Clip to SA boundary to avoid processing outside SA
    eco_geom = ecoregion.geometry().intersection(sa.geometry(), ee.ErrorMargin(1))

    # ── AlphaEarth covariates for FSCS ──
    # reduce(first) not mosaic() — benchmark gotcha #4
    aef22 = ee.ImageCollection(
        "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    ).filterDate("2022-01-01", "2023-01-01").reduce(
        ee.Reducer.first()
    ).regexpRename('_first$', '')

    # Use first N bands for FSCS clustering (64 bands × 100 clusters
    # exceeds GEE computation limits on dense cells)
    aef22_fscs = aef22.select(aef22.bandNames().slice(0, FSCS_BANDS))

    # ── Full extraction stack (single reduceRegions per cell) ──
    print("  Building extraction stack (AEF + natural + GPP/SVH/BII)...")
    extraction_stack = build_extraction_stack(eco_geom)

    # ── Grid cells — ee.Geometry not ee.Feature (benchmark lesson #5) ──
    # Use .bounds() for grid generation to avoid memory overflow on complex
    # ecoregion geometries, then filterBounds(SA) to trim outside cells
    eco_raw_geom = ecoregion.geometry()
    grid = eco_raw_geom.bounds().coveringGrid(
        sa.geometry().projection(), grid_scale)
    grid_in_eco = grid.filterBounds(sa.geometry())
    total_cells = retry_gee(lambda: grid_in_eco.size().getInfo())
    print(f"  Grid: {total_cells} cells ({grid_scale/1000:.0f}km), "
          f"{n_clusters} FSCS clusters/cell")

    # Get grid geometries as list of dicts (ee.Geometry construction)
    # Paginate for large grids (>5000 cells) to avoid memory limits
    if total_cells > 5000:
        print(f"  Paginating grid download ({total_cells} cells)...")
        # Use unfiltered grid with randomColumn — filterBounds on large
        # collections hits GEE memory limits. Empty cells are skipped
        # by the pixel_count check in process_cell.
        grid_unfiltered = eco_raw_geom.bounds().coveringGrid(
            sa.geometry().projection(), grid_scale)
        total_cells = retry_gee(lambda: grid_unfiltered.size().getInfo())
        grid_tagged = grid_unfiltered.randomColumn('_rnd', seed=42)
        grid_features = []
        n_pages = max(1, (total_cells + 1999) // 2000)
        for pi in range(n_pages):
            lo = pi / n_pages
            hi = (pi + 1) / n_pages
            page = retry_gee(
                lambda l=lo, h=hi: grid_tagged.filter(
                    ee.Filter.And(
                        ee.Filter.gte('_rnd', l),
                        ee.Filter.lt('_rnd', h)
                    )).getInfo()['features'])
            grid_features.extend(page)
            print(f"    Downloaded {len(grid_features)}/{total_cells} cells")
        print(f"  Updated grid: {total_cells} cells (unfiltered bounds)")
    else:
        grid_features = retry_gee(
            lambda: grid_in_eco.getInfo()['features'])

    # ── Checkpoint ──
    processed_cells = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed_cells = set(json.load(f))
        print(f"  Resuming: {len(processed_cells)}/{total_cells} cells done")

    lock = Lock()

    def save_checkpoint():
        with lock:
            with open(checkpoint_file, 'w') as f:
                json.dump(sorted(processed_cells), f)

    # ── DuckDB buffer ──
    db_conn = duckdb.connect()

    if os.path.exists(output_path):
        print(f"  Loading existing parquet into buffer...")
        db_conn.execute(
            f"CREATE TABLE data AS SELECT * FROM '{output_path}'")
        existing = db_conn.execute(
            "SELECT count(*) FROM data").fetchone()[0]
        print(f"  [OK] Loaded {existing:,} existing rows")

    # ── Process each grid cell (parallel) ──
    successful = 0
    failed_cells = []

    def process_cell(cell_idx):
        """FSCS + single-shot extraction for one grid cell."""
        if cell_idx in processed_cells:
            return 0

        # ee.Geometry not ee.Feature — benchmark lesson #5
        cell_geom = ee.Geometry(grid_features[cell_idx]['geometry'])

        # Clip cell to ecoregion — some edge cells produce degenerate
        # geometries that fail intersection even with error margin
        try:
            cell_eco = cell_geom.intersection(eco_geom, ee.ErrorMargin(1))

            # Check pixel count first — skip near-empty cells, adapt n_clusters
            pixel_count = aef22.select(0).reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=cell_eco,
                scale=SCALE,
                maxPixels=1e8
            ).getNumber(aef22.bandNames().get(0)).getInfo()
        except Exception as e:
            tqdm.write(f"    Cell {cell_idx}: geom error — {e}")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return 0

        if pixel_count is None or pixel_count < 10:
            tqdm.write(f"    Cell {cell_idx}: skip ({pixel_count} pixels)")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return 0

        # Adaptive cluster count: can't have more clusters than pixels
        cell_n_clusters = min(n_clusters, max(2, int(pixel_count // 5)))
        cell_n_init = min(N_INIT_POINTS, int(pixel_count))

        # FSCS on reduced bands (10 of 64) — full 64 bands cause
        # computation timeouts on dense cells (100 clusters × 64 bands)
        try:
            clusters, samples = fscs(
                covariates=aef22_fscs,
                n_clusters=cell_n_clusters,
                n_init_points=cell_n_init,
                geometry=cell_eco,
                scale=SCALE,
                seed=SEED
            )
        except Exception as e:
            tqdm.write(f"    Cell {cell_idx}: FSCS failed — {e}")
            processed_cells.add(cell_idx)
            save_checkpoint()
            return 0

        # Single-shot extraction: embeddings + natural + GPP/SVH/BII
        sampled = extraction_stack.reduceRegions(
            collection=samples,
            reducer=ee.Reducer.first(),
            scale=SCALE,
            tileScale=4
        )

        df = ee.data.computeFeatures({
            'expression': sampled,
            'fileFormat': 'PANDAS_DATAFRAME'
        })

        n_rows = 0 if df is None else len(df)
        tqdm.write(
            f"    Cell {cell_idx}: {n_rows} pts "
            f"(k={cell_n_clusters}, px={pixel_count:.0f})")

        if df is not None and not df.empty:
            df['cell_index'] = cell_idx
            with lock:
                table_exists = True
                try:
                    db_conn.execute("SELECT 1 FROM data LIMIT 0")
                except duckdb.CatalogException:
                    table_exists = False

                if table_exists:
                    db_conn.execute(
                        "INSERT INTO data SELECT * FROM df")
                else:
                    db_conn.execute(
                        "CREATE TABLE data AS SELECT * FROM df")

            processed_cells.add(cell_idx)
            save_checkpoint()
            return len(df)

        processed_cells.add(cell_idx)
        save_checkpoint()
        return 0

    # Parallel execution — workers capped at n_cells (benchmark lesson #6)
    effective_workers = min(max_workers, total_cells)
    print(f"  Processing with {effective_workers} workers...\n")

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(
            lambda idx=i: retry_gee(lambda: process_cell(idx))
        ): i for i in range(total_cells)}

        for future in tqdm(as_completed(futures),
                           total=total_cells, desc="FSCS cells"):
            cell_idx = futures[future]
            try:
                n = future.result()
                if n and n > 0:
                    successful += 1
            except Exception as e:
                failed_cells.append(cell_idx)
                tqdm.write(f"  [ERROR] Cell {cell_idx}: {e}")

    if failed_cells:
        print(f"\n  {len(failed_cells)} cells failed after retries: "
              f"{failed_cells}")

    # ── Export ──
    print(f"\n  Cells: {successful} with data, "
          f"{len(failed_cells)} failed, "
          f"{len(processed_cells)} total processed")

    try:
        buf_rows = db_conn.execute(
            "SELECT count(*) FROM data").fetchone()[0]
    except Exception:
        buf_rows = 0

    if buf_rows == 0:
        print(f"  [WARN] No data extracted for eco_id={eco_id}")
        db_conn.close()
        return

    # Deduplicate and export
    db_conn.execute(f"""
        COPY (
            SELECT DISTINCT ON (geo) *
            FROM data
        ) TO '{output_path}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    final_rows = db_conn.execute(
        f"SELECT count(*) FROM '{output_path}'").fetchone()[0]
    n_natural = db_conn.execute(f"""
        SELECT count(*) FROM '{output_path}'
        WHERE "natural" = 1
    """).fetchone()[0]
    file_mb = os.path.getsize(output_path) / 1e6

    print(f"\n[OK] Saved: {output_path}")
    print(f"  Total: {final_rows:,} points "
          f"({n_natural:,} natural, "
          f"{final_rows - n_natural:,} transformed)")
    print(f"  Size: {file_mb:.1f} MB")

    db_conn.close()

    # Clean up checkpoint on full success
    if not failed_cells and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FSCS reference sampling for ecoregions missing "
                    "from fscs_aef_samples")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eco_id", type=int,
                       help="Single ecoregion ECO_ID")
    group.add_argument("--all", action="store_true",
                       help="All 11 missing ecoregions")
    parser.add_argument("--n_clusters", type=int, default=N_CLUSTERS,
                        help=f"FSCS clusters per grid cell "
                             f"(default {N_CLUSTERS})")
    parser.add_argument("--grid_scale", type=int, default=GRID_SCALE,
                        help=f"Grid cell size in metres "
                             f"(default {GRID_SCALE})")
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--project", default="ee-gsingh")
    args = parser.parse_args()

    init_gee(args.project)

    if args.eco_id:
        run_ecoregion(args.eco_id, args.n_clusters, args.max_workers,
                      args.grid_scale)
    else:
        for eco_id in ALL_ECOS:
            run_ecoregion(eco_id, args.n_clusters, args.max_workers,
                          args.grid_scale)


if __name__ == "__main__":
    main()
