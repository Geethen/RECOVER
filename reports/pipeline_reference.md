# RECOVER Pipeline Reference

Quick-reference for onboarding a new agent or collaborator. Covers data flow, scripts, key files, and current state.

---

## Environment

```bash
PYTHON="C:\Users\coach\.conda\envs\erthy\python.exe"
# Run:    & "$PYTHON" scripts/path/to/script.py
# Install: & "$PYTHON" -m uv pip install <package>
```

GEE project: `ee-gsingh` | High-volume endpoint: `https://earthengine-highvolume.googleapis.com`

---

## Pipeline Overview

```
 ┌─────────────────────────────────────────────────────────────────┐
 │  1. EXTRACTION (GEE → parquet)                                 │
 │     abandoned_ag_extract.py → abandoned_ag_gpp_2000_2022_SA    │
 │     gee_extraction.py       → extracted_indices                │
 │     sample_reference_points → ref_samples_eco{id}              │
 │     extract_gpp_svh_for_indices → indices_gpp_svh_2000_2022    │
 ├─────────────────────────────────────────────────────────────────┤
 │  2. MODEL TRAINING                                             │
 │     train_regression_model.py                                  │
 │       dfsubsetNatural.parquet → best_model.pth + scalers       │
 ├─────────────────────────────────────────────────────────────────┤
 │  3. REFERENCE PREDICTION                                       │
 │     predict_reference_conditions.py                            │
 │       extracted_indices + model → reference_departure_with_*   │
 ├─────────────────────────────────────────────────────────────────┤
 │  4a. RECOVERY SCORING (per ecoregion, 18 total)                │
 │     extract_all_ecoregions.py                                  │
 │       abandoned_ag + ref_samples → recovering_eco{id}          │
 │       + AlphaEarth embeddings   → recovering_eco{id}_embeddings│
 │       + Metrics A/B/C           → recovery_scores_eco{id}      │
 │     extract_niaps_filter.py     → adds niaps column to scores  │
 ├─────────────────────────────────────────────────────────────────┤
 │  4b. TRAJECTORY CLASSIFICATION                                 │
 │     trajectory_classifier.py (or extract_full_features.py)     │
 │       abandoned_ag → Z-norm → PCA1 → breakpoints → Catch22    │
 │                    → UMAP 3D → HDBSCAN → trajectory_results    │
 ├─────────────────────────────────────────────────────────────────┤
 │  5. BENCHMARKING                                               │
 │     select_benchmark_sites.py                                  │
 │       reference_departure + ecoregion_sds → benchmarked_cond.  │
 ├─────────────────────────────────────────────────────────────────┤
 │  6. VALIDATION                                                 │
 │     validate_{internal,spatial,ecological,external,stability}.py│
 │     generate_summary_report.py                                 │
 ├─────────────────────────────────────────────────────────────────┤
 │  7. VISUALIZATION                                              │
 │     plot_benchmarking_results, visualize_clusters,             │
 │     plot_example_with_satellite, generate_interactive_map      │
 └─────────────────────────────────────────────────────────────────┘
```

---

## Key Data Files

### Core (read-only, authoritative)

| File | Rows | Description |
|------|------|-------------|
| `data/abandoned_ag_gpp_2000_2022_SA.parquet` | 33.4M | All abandoned-ag pixels. Cols: pixel_id, lat, lon, eco_id, sanlc_2022, GPP_2000–GPP_2022, SVH_2000–SVH_2022, BII. **Warning**: ~16 GB in pandas; use DuckDB. pixel_id is **not globally unique** (resets per grid cell); unique key is (lat, lon, eco_id). |
| `data/extracted_indices.parquet` | ~330K | Natural + transformed sites: pixel_id, lat/lon, eco_id, 64D embeddings (A00–A63), NBR, NDMI, NDWI. |
| `data/dfsubsetNatural.parquet` | ~97K | Natural-only subset of extracted_indices. Training data for regression model. |
| `data/indices_gpp_svh_2000_2022.parquet` | ~330K | GPP/SVH time series aligned to extracted_indices. |
| `data/ecoregion_sds.geojson` | 18 | Ecoregion polygon boundaries for SA. |

### Per-Ecoregion (18 ecoregions)

IDs: 81, 41, 38, 40, 97, 48, 89, 90, 101, 88, 110, 16, 102, 19, 94, 15, 116, 65

| Pattern | Description |
|---------|-------------|
| `ref_samples_eco{id}.parquet` | FSCS-sampled reference points (natural + transformed). Cols: lat, lon, `"natural"` (0/1), GPP/SVH, 64D embeddings, BII, sanlc. Quote `"natural"` in DuckDB — reserved word. |
| `ref_samples_eco{id}.checkpoint.json` | List of processed grid cell indices. Eco19 and eco116 have no checkpoint (single-run). |
| `recovering_eco{id}.parquet` | Abandoned-ag pixels with significant recovery trend (MK p<0.05). Has duplicate pixel_ids (from abandoned_ag duplication). |
| `recovering_eco{id}_embeddings.parquet` | 64D AlphaEarth embeddings for recovering pixels. 1,639 missing across all ecos (0.13%) — structural gaps at tile boundaries. |
| `recovery_scores_eco{id}.parquet` | Final scores. Cols: pixel_id, lat, lon, eco_id, sanlc_2022, gpp_slope, svh_slope, a_gpp_pctl, a_svh_pctl, b_gpp_pctl, b_svh_pctl, b_gpp_ratio, b_svh_ratio, c_eco_sim, c_eco_pctl, c_local_sim, c_local_pctl, composite_gpp, composite_svh, recovery_score, niaps. |

### Derived

| File | Description |
|------|-------------|
| `data/reference_departure_with_intervals.parquet` | Model predictions + 95% CI for all sites. |
| `data/trajectory_results.parquet` | Cluster assignments from UMAP+HDBSCAN. ~6 GB. |
| `data/benchmarked_condition.parquet` | HCAS condition scores. |

### Models

| File | Description |
|------|-------------|
| `models/best_model.pth` | PyTorch checkpoint: MultiHeadRegressionModel (EmbeddingAttention → 3×ResidualBlock 256D → 3 heads). |
| `models/trajectory_umap/*.pkl` | UMAP reducer, HDBSCAN clusterer, scaler, selected features. |
| `models/locart/locart_{NBR,NDMI,NDWI}.pkl` | LO-CART calibrated uncertainty per target. |

---

## Recovery Scoring Metrics

All percentile-normalised (0–100, 50 = natural median):

| Metric | What it measures | Column(s) |
|--------|-----------------|-----------|
| **A** | Recent GPP/SVH (2018–2022 mean) vs ecoregion natural distribution | `a_gpp_pctl`, `a_svh_pctl` |
| **B** | GPP/SVH ratio to 10 nearest natural neighbours within 50 km | `b_gpp_pctl`, `b_svh_pctl`, `b_gpp_ratio`, `b_svh_ratio` |
| **C_eco** | Cosine similarity of 64D AlphaEarth embedding to ecoregion natural pixels | `c_eco_sim`, `c_eco_pctl` |
| **C_local** | Cosine similarity to 10 nearest natural neighbours | `c_local_sim`, `c_local_pctl` |
| **Composite** | Mean of all metric percentiles, per indicator | `composite_gpp`, `composite_svh` |
| **Recovery score** | Mean of composite_gpp and composite_svh | `recovery_score` |
| **NIAPS** | Invasive alien plant flag (1 = invaded, exclude) | `niaps` |

---

## Key Numbers (current state, 2026-03-25)

| Item | Count |
|------|-------|
| Ecoregions scored | 18 |
| Total recovering pixels | 2,881,285 |
| Invasive (NIAPS=1) | 661,217 (22.9%) |
| Post-exclusion recovering | 2,220,068 |
| Missing embeddings | 1,639 (0.13%) — structural, at AlphaEarth tile boundaries |
| Failed ref_sample cells | 139 (0.3%) — structural, ecoregion edge cells |
| Abandoned-ag extraction gap | ~5.6M pixels (14.5%) vs GEE mask; cause under investigation (likely dropNulls filtering with BII layer) |

---

## Script Quick-Reference

### Extraction (`scripts/extraction/`)

| Script | Run | Notes |
|--------|-----|-------|
| `abandoned_ag_extract.py` | `python scripts/extraction/abandoned_ag_extract.py` | Checkpointed. 50 km grid, 20 workers/cell. `--start_year 2000 --end_year 2022` |
| `sample_reference_points.py` | `python scripts/extraction/sample_reference_points.py --eco_id 81` | FSCS on AlphaEarth. 10 km grid, 100 clusters/cell. Checkpointed. |
| `extract_all_ecoregions.py` | `python scripts/extraction/extract_all_ecoregions.py --eco_id 81` | Full per-ecoregion pipeline: MK → embeddings → scoring. |
| `extract_gpp_svh_for_indices.py` | `python scripts/extraction/extract_gpp_svh_for_indices.py` | Add GPP/SVH time series to extracted_indices points. |

### Analysis (`scripts/analysis/`)

| Script | Purpose |
|--------|---------|
| `batch_recovery_degree.py` | Alternative to extract_all_ecoregions for recovery scoring. |
| `extract_niaps_filter.py` | Add NIAPS invasive column to score files. Batch 5000, 10 workers. |
| `generate_narrative_plots.py` | High-quality Seaborn charts for the presentation narrative (e.g., recovery scores by ecoregion, invasion stats, adjusted scatter plots). |
| `generate_preprocessing_plots.py` | Context variables comparing Active Area vs Abandoned Area and data excluded during initial extraction. |
| `generate_nested_context_plots.py` | Consistent scale nested bubble plots showing the size cascade from SA land to true native recovery. |
| `generate_new_narrative_plots.py` | Radar chart (metric decomposition), violin distributions, filtering funnel, and Metric A vs C scatter. |
| `retry_failed_extractions.py` | Retry missing embeddings (batch 500, tileScale 8). |
| `retry_ref_samples.py` | Retry failed ref_sample cells. |
| `trajectory_recovery_report.py` | Generate markdown report with cross-tabulated recovery stats. |
| `test_recovery_degree.py` | Single-pixel diagnostic plots. |

### ML (`scripts/ml/`)

| Script | Purpose |
|--------|---------|
| `train_regression_model.py` | Train MultiHeadRegressionModel. W&B project: `expected_Ref_Model`. |
| `predict_reference_conditions.py` | Inference: model + scalers → reference departures. |
| `trajectory_classifier.py` | Z-norm → PCA → breakpoints → Catch22 → UMAP → HDBSCAN. |

### Visualization (`scripts/visualization/`)

| Script | Purpose |
|--------|---------|
| `plot_example_with_satellite.py` | Trajectory + Sentinel-2 before/after for example pixels A–D. |
| `plot_trajectory_examples.py` | Per-ecoregion example trajectory panels. |
| `generate_interactive_map.py` | Leafmap HTML output. |

---

## GEE Assets

| Asset | Resolution | Used by |
|-------|-----------|---------|
| `projects/ee-gsingh/assets/RECOVER/abandoned_ag` | 30 m | abandoned_ag_extract (mask: eq(10), connected≥40, morphological close) |
| `projects/ee-gsingh/assets/RECOVER/niaps_binary_30m` | 30 m | extract_niaps_filter (1=invaded) |
| `projects/ee-gsingh/assets/RECOVER/sanlc2022_7class` | 30 m | Land cover classification |
| `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 10 m | AlphaEarth 64D embeddings (columns A00–A63) |
| `projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m` | 30 m | GPP (monthly → annual sum) |
| `projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m` | 30 m | SVH (monthly → annual mosaic) |
| `COPERNICUS/S2_SR_HARMONIZED` | 10 m | Sentinel-2 satellite imagery |
| `RESOLVE/ECOREGIONS/2017` | Vector | Ecoregion boundaries (ECO_ID) |
| BII 1km collection | 1 km | Biodiversity Intactness Index |

---

## DuckDB Patterns

```python
import duckdb
con = duckdb.connect()
con.execute("SET memory_limit='4GB'")

# Query large parquet without loading into memory
df = con.sql("""
    SELECT pixel_id, latitude, longitude, GPP_2022
    FROM 'data/abandoned_ag_gpp_2000_2022_SA.parquet'
    WHERE eco_id = 81 AND sanlc_2022 = 2
    LIMIT 1000
""").df()

# Quote "natural" — it's a reserved word
ref = con.sql("""
    SELECT * FROM 'data/ref_samples_eco38.parquet'
    WHERE "natural" = 1
""").df()
```

---

## Known Issues & Gotchas

1. **pixel_id is not globally unique** in abandoned_ag — it's GEE `system:index`, resets per grid cell. Use (lat, lon, eco_id) as the true key.
2. **DuckDB reserved word**: `WHERE natural = 1` fails; must use `WHERE "natural" = 1`.
3. **abandoned_ag is ~16 GB in pandas** — always use DuckDB for queries. Set `memory_limit='4GB'` to avoid OOM.
4. **9 NaN rows in ref_samples_eco38** — natural reference has 9 all-NaN GPP/SVH rows. Use `.dropna()` before computing means.
5. **1 null row in abandoned_ag** — single all-NULL row from empty batch insertion. Filtered out by any `WHERE eco_id = X`.
6. **Recovering files have duplicate pixel_ids** from abandoned_ag source — same pixel_id, different physical locations. Scoring deduplicates correctly.
7. **AlphaEarth tile boundaries** — 1,639 pixels at integer-degree lon/lat edges have no embedding data. Structural, not retryable.
8. **Ref_sample edge cells** — 139 cells at ecoregion boundaries fail deterministically (GEE Dictionary key errors).
9. **BII layer masking** — BII is 1 km resolution with selfMask + bii_mask. When `dropNulls=True` is used in extraction, pixels without BII data are silently dropped. This likely explains the ~5.6M pixel gap between the GEE mask (39M) and extracted data (33.4M).

---

## Reports & Outputs

| File | Content |
|------|---------|
| `reports/recovery_assessment_guide.md` | Full methodology guide with 4 worked examples (Pixels A–D), all-ecoregion results table, NIAPS invasive data. All referenced plots exist. |
| `reports/trajectory_recovery_report.md` | Trajectory classification results by ecoregion. |
| `reports/trajectory_and_recovery_methodology.md` | Combined technical methodology. |
| `plots/example_trajectory_{a,b,c,d}.png` | GPP/SVH time series with natural/transformed/KNN envelopes. |
| `plots/example_satellite_{a,b,c,d}.png` | Sentinel-2 before/after composites (2016–2017 vs 2022–2023). |
| `plots/recovery_degree_{comparison,diagnostic}.png` | Single-pixel detailed assessment plots. |

---

## Example Pixels (used in guide)

| Label | Eco | pixel_id | Lat | Lon | Score | Profile |
|-------|-----|----------|-----|-----|-------|---------|
| A | 81 | 8393 | -32.163 | 26.624 | 53.6 | Moderate: high A/B, low C |
| B | 81 | 34240 | -27.485 | 28.202 | 82.2 | Advanced: high across all |
| C | 81 | 79489 | -30.672 | 26.698 | 24.1 | Early: A≈50, B/C very low |
| D | 38 | 80892 | -25.283 | 26.408 | 94.1 | Full: c_eco=98.8, all >87 |
