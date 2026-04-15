# Abandoned Agriculture GPP Extraction Methodology

This document outlines the systematic steps performed by `abandoned_ag_extract.py` to extract Gross Primary Productivity (GPP) data for abandoned agricultural areas in South Africa.

## Workflow Overview

The extraction follows a spatial-grid-based approach to handle large-scale data while utilizing the Google Earth Engine (GEE) High-Volume endpoint for efficiency.

---

### Step 1: Abandoned Agriculture Mask Preparation
*   **Asset**: `projects/ee-gsingh/assets/RECOVER/abandoned_ag`
*   **Filtering**: Selects pixels where the value equals `10` (Abandoned Agriculture).
*   **Cleaning (Denoising)**: Performs Connected Component Analysis (CCA). Any isolated patch smaller than **40 pixels** is removed to eliminate noise and small artifacts.
*   **Morphological Refinement**: Applies a `focal_max` followed by a `focal_min` (Closing operation) to fill small internal holes within the remaining patches.

### Step 2: GPP Data Integration (2000–2022)
*   **Asset Collection**: `projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m`
*   **Temporal Aggregation**: Iterates through each year from 2000 to 2022.
*   **Compositing**: For each year, it calculates the **Sum** of GPP (Annual GPP).
*   **Stacking**: All 23 years of annual composites are stacked into a single multi-band image (`GPP_2000`, `GPP_2001`, ... `GPP_2022`).
*   **Spatial Masking**: The multi-band GPP image is masked using the refined Abandoned Ag mask from Step 1. Only pixels confirmed as abandoned agriculture are retained.

### Step 3: Spatial Partitioning (Grid Generation)
*   **Area of Interest**: South Africa boundary (`FAO/GAUL/2015/level0`).
*   **Grid Size**: A **50km x 50km** covering grid is generated over the boundary.
*   **Rational**: Partitioning the country into cells prevents memory overflows and allows for a resumable workflow. If the script stops, it can skip already-processed cells.

### Step 4: Distributed Batch Extraction
For each grid cell, the following sub-steps occur:
1.  **Pixel Sampling**: Identifies all valid (non-masked) pixels within the cell at a **30m resolution**.
2.  **Batching**: Pixels are grouped into batches (standard size: 3000 pixels).
3.  **Parallel Execution**: Uses a `ThreadPoolExecutor` with 20 parallel workers.
4.  **High-Volume API**: Calls `ee.data.computeFeatures` to fetch data directly into a Pandas DataFrame using GEE’s High-Volume endpoint.
5.  **Thread-Safe Saving**: Appends results to the main CSV file using a threading lock to prevent file corruption.

### Step 5: Checkpointing and Error Handling
*   **Resumability**: A `.checkpoint.json` file tracks successfully completed grid cells.
*   **Retry Logic**: Includes an exponential backoff decorator for GEE API calls to handle transient network errors or rate limits.

### Step 6: Post-Extraction Analytics
*   **Summary Statistics**: Generates a file (`..._summary_stats.csv`) containing count, mean, std, min, and max for every year.
*   **Temporal Trends**: Generates a file (`..._trends.csv`) that averages GPP across all pixels for each year, useful for plotting vegetation recovery/degradation over time.

---

## Sanity Check Parameters
*   **Spatial Resolution**: 30 meters.
*   **Temporal Range**: Jan 1, 2000 – Dec 31, 2022.
*   **Coordinate System**: WGS84 (EPSG:4326) for output CSV.
*   **Masking**: Strict abandoned ag (Class 10) only.
