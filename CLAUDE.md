# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RECOVER** — Ecosystem Condition and Recovery Modeling. Extracts ecological indices from Google Earth Engine (GEE) and trains deep learning regression models to predict ecosystem reference conditions and classify ecological trajectories of abandoned agricultural land in Southern Africa.

## Python Environment

- Environment: `C:\Users\coach\.conda\envs\erthy\python.exe`
- Run scripts: `& "C:\Users\coach\.conda\envs\erthy\python.exe" scripts/path/to/script.py`
- Install packages: `& "C:\Users\coach\.conda\envs\erthy\python.exe" -m uv pip install <package>`

## Pipeline Stages (in order)

### 1. Data Extraction (GEE)
```bash
python scripts/extraction/gee_extraction.py
```
Requires authenticated GEE project (`ee-gsingh`). Outputs parquet/CSV files with remote sensing indices (NBR, NDMI, NDWI) + 64-dimensional embeddings.

### 2. Model Training
```bash
python scripts/ml/train_regression_model.py
```
- Input: `data/dfsubsetNatural.parquet` (natural reference sites only)
- Output: `models/best_model.pth`, training plots, W&B experiment logs
- Architecture: `MultiHeadRegressionModel` — EmbeddingAttention gate → 3 ResidualBlocks (256D) → 3 per-target heads (NBR, NDMI, NDWI)
- Loss: `MultiTaskUncertaintyLoss` (homoscedastic uncertainty weighting across tasks)
- Key hyperparams: `hidden_dim=256`, `lr=0.001`, `epochs=150`, `batch_size=64`, `patience=25`

### 3. Predict Reference Conditions
```bash
python scripts/ml/predict_reference_conditions.py
```
Applies trained model to all sites; outputs `data/reference_departure_with_intervals.parquet`.

### 4. Trajectory Classification
```bash
python scripts/ml/trajectory_classifier.py
```
Multi-stage pipeline on `abandoned_ag_gpp_2000_2022_SA.parquet`:
1. Biome-stratified Z-score normalization
2. Vectorized PCA1 (analytic 2×2 eigen-solution for speed)
3. Piecewise linear regression with breakpoint detection via precomputed pseudo-inverses (300× speedup)
4. Catch22 feature extraction (44 time-series statistics)
5. UMAP (→3D) + HDBSCAN clustering
- Output: `data/trajectory_results.parquet`

### 5. Benchmarking (HCAS v3.1)
```bash
python scripts/analysis/select_benchmark_sites.py
```
Selects natural benchmark sites per transformed site using BallTree spatial indexing (Haversine), Manhattan distance in reference-condition space, geographic penalty, and PCHIP spline calibration. Key params: 200 km radius, K=70 reference-space filter, K=10 final benchmarks, geographic penalty factor=30, Half-Cauchy λ=2.0.
- Output: `data/benchmarked_condition.parquet`

### 6. Validation
```bash
python scripts/validation/validate_internal.py      # Silhouette, Davies-Bouldin, Calinski-Harabasz
python scripts/validation/validate_spatial.py       # Moran's I spatial autocorrelation
python scripts/validation/validate_ecological.py    # Domain-specific metrics
python scripts/validation/generate_summary_report.py
```

### 7. Visualization
```bash
python scripts/visualization/plot_benchmarking_results.py
python scripts/visualization/visualize_clusters.py
python scripts/visualization/generate_interactive_map.py
```

## Diagnostics / Utilities
```bash
python scripts/utils/check_auth.py      # Verify GEE authentication
python scripts/utils/check_schema.py   # Validate parquet schema
python scripts/utils/check_counts.py   # Check record counts
python scripts/utils/test_extraction.py
```

## Architecture Notes

- **Data format**: All intermediate datasets are parquet. DuckDB is used for columnar SQL queries (see `DUCKDB_WORKFLOW.md`).
- **Embeddings**: 64-dimensional embeddings (columns `A00`–`A63`) are the primary model input, derived from GEE extraction.
- **Feature scaling**: `StandardScaler` on inputs; `MinMaxScaler(-0.95, 0.95)` on targets — scalers are fit on training data and must be saved/reloaded for inference.
- **Geospatial**: `geopandas` + `BallTree` with Haversine for spatial operations. Ecoregion boundaries in `data/ecoregion_sds.geojson`.
- **Experiment tracking**: Weights & Biases (`wandb`, project `"expected_Ref_Model"`). Runs logged per training session.
- **GEE extraction**: Uses high-volume endpoint, `CheckpointManager` for resumable downloads, and `ThreadPoolExecutor` for parallel batch processing.

## Key Data Files

| File | Description |
|------|-------------|
| `data/dfsubsetNatural.parquet` | Training data (natural sites) |
| `data/extracted_indices.parquet` | All sites with indices + embeddings |
| `data/abandoned_ag_gpp_2000_2022_SA.parquet` | GPP time series 2000–2022 |
| `data/reference_departure_with_intervals.parquet` | Model predictions + intervals |
| `data/trajectory_results.parquet` | Cluster assignments |
| `data/benchmarked_condition.parquet` | Final HCAS condition scores |
| `models/best_model.pth` | Best trained model checkpoint |

## Documentation

- `BENCHMARKING_STEPS_v2.md` — Detailed HCAS v3.1 methodology
- `DOC_TRAJECTORY_ANALYSIS.md` — Trajectory classification technical detail
- `DUCKDB_WORKFLOW.md` — DuckDB query patterns
- `scripts/EXTRACTION_METHODOLOGY.md` — GEE extraction details
