# Trajectory Classification and Recovery Degree Estimation

## Overview

This document describes the two-stage methodology for assessing ecological recovery on abandoned agricultural land in South Africa:

1. **Trajectory Classification** — classifies 23-year GPP and SVH trends as Recovery, Stable, or Degradation using Mann-Kendall + Theil-Sen trend analysis
2. **Recovery Degree Estimation** — quantifies how far a recovering pixel has progressed relative to natural reference conditions using four complementary metrics

---

## 1. Data Sources

### Remote Sensing Time Series

| Variable | GEE Collection | Aggregation | Resolution | Period |
|----------|----------------|-------------|------------|--------|
| Gross Primary Productivity (GPP) | `projects/global-pasture-watch/assets/ggpp-30m/v1/ugpp_m` | Annual sum | 30 m | 2000–2022 |
| Short Vegetation Height (SVH) | `projects/global-pasture-watch/assets/gsvh-30m/v1/short-veg-height_m` | Annual mosaic | 30 m | 2000–2022 |

### Ancillary Layers

| Layer | GEE Asset | Purpose |
|-------|-----------|---------|
| Abandoned agriculture mask | `projects/ee-gsingh/assets/RECOVER/abandoned_ag` | Identifies abandoned agricultural pixels (class 10) |
| SANLC 2022 (7-class) | `projects/ee-gsingh/assets/RECOVER/sanlc2022_7class` | Land cover classification |
| Ecoregions | `RESOLVE/ECOREGIONS/2017` | Biogeographic stratification |
| AlphaEarth embeddings | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 64-dimensional foundation model embeddings (10 m, 2022) |

### AlphaEarth Satellite Embeddings

The AlphaEarth Foundations model (Google DeepMind) fuses a full year of multi-source satellite observations into a compact 64-dimensional unit-length vector per pixel at 10 m resolution. The model learns to compress these diverse inputs into a representation where similar landscapes map to nearby points in 64-dimensional space — effectively creating a rich "fingerprint" for each piece of land. Input data sources include:

- **Sentinel-2 multispectral imagery** — captures vegetation type, health, phenology, and soil properties via optical and near-infrared bands
- **Landsat 8 and 9** — multispectral, panchromatic, and thermal bands providing additional spectral coverage and a longer temporal archive
- **Sentinel-1 C-Band SAR radar** — sensitive to vegetation structure and moisture; operates through cloud cover
- **ALOS PALSAR-2 ScanSAR L-band radar** — penetrates deeper into vegetation canopies, sensitive to woody biomass and forest structure
- **GEDI-derived canopy height metrics** — lidar-based measurements of vertical vegetation structure
- **Copernicus GLO-30 digital elevation model** — terrain, slope, and aspect
- **ERA5-Land monthly climate aggregates** — temperature, precipitation, and seasonal climate patterns
- **GRACE monthly mass grids** — gravity-derived measurements reflecting groundwater and hydrological changes

The embeddings encode vegetation structure, phenology, moisture, elevation, and land cover simultaneously, providing a holistic characterisation of landscape condition that goes beyond any single index. Licensed under CC-BY 4.0.

### SANLC 2022 Classes

| Code | Label |
|------|-------|
| 1 | Natural / near-natural |
| 2 | Secondary natural |
| 3 | Artificial water |
| 4 | Built-up |
| 5 | Cropland |
| 6 | Mine |
| 7 | Plantation |

### Key Datasets

| File | Records | Description |
|------|---------|-------------|
| `abandoned_ag_gpp_2000_2022_SA.parquet` | ~33.4 M | All abandoned-ag pixels with GPP/SVH time series, SANLC, ecoregion |
| `indices_gpp_svh_2000_2022.parquet` | ~283 K | Stratified reference sample with GPP/SVH time series (matched to embedding points) |
| `extracted_indices.parquet` | ~331 K | Reference sample with AlphaEarth 64D embeddings (A00–A63) |
| `dfsubsetNatural.parquet` | ~97 K | Subset of extracted_indices identified as natural by composite mask |

---

## 2. Trajectory Classification

### 2.1 Preprocessing

**Abandoned agriculture mask**: Pixels where the land-use map value equals 10 (abandoned agriculture). The mask is cleaned using connected component analysis (patches < 40 pixels removed) and morphological closing (focal_max then focal_min) to fill small internal gaps.

**Land cover filter**: Only pixels with natural or near-natural land cover are retained for classification (SANLC 2022 classes 1 and 2), yielding ~28.3 M pixels (84.8% of total).

**Ecoregion Z-score normalisation**: Before trend analysis, GPP and SVH values are standardised per ecoregion to account for biogeographic differences in baseline productivity. For each ecoregion, the mean and standard deviation are computed across all pixels, and each pixel's value is transformed to:

> Z = (pixel value - ecoregion mean) / ecoregion standard deviation

A Z-score of +1.0 means the pixel is one standard deviation above its ecoregion average. This ensures that trend detection is not confounded by cross-ecoregion differences in absolute productivity.

### 2.2 Trend Analysis

For each pixel, two non-parametric tests (i.e., methods that make no assumptions about the statistical distribution of the data) are applied independently to the 23-year GPP and SVH time series:

**Theil-Sen slope estimator**: Computes the slope between every possible pair of years in the time series (253 pairs for 23 years), then takes the median of all those slopes. Because it uses the median rather than the mean, a single extreme value (e.g., a drought year) cannot skew the result. This provides a robust estimate of the linear rate of change.

**Mann-Kendall test**: Evaluates whether a trend is statistically significant. For every pair of years (t_i, t_j where j > i), it records whether the later value is higher (+1), lower (-1), or equal (0), then sums these signs into a statistic S. In a trendless series, roughly equal numbers of pairs would increase and decrease, so S would hover near zero. A large positive S indicates consistent increases across the time series. The test converts S into a standardised Z-score and computes a two-tailed p-value — the probability that a trend this strong or stronger would appear purely by chance in a random series. If p < 0.05, the trend is considered statistically significant.

### 2.3 Classification Rules

| Class | Criterion |
|-------|-----------|
| **Recovery** | p < 0.05 and slope > 0 |
| **Degradation** | p < 0.05 and slope < 0 |
| **Stable** | p >= 0.05 |

Classification is applied separately to GPP (functional trajectory) and SVH (structural trajectory), then cross-tabulated to produce a combined 3x3 matrix.

### 2.4 Summary Results

**Functional (GPP) trajectory**:
- Recovery: 3.63 M pixels (12.8%)
- Stable: 21.01 M pixels (74.2%)
- Degradation: 3.67 M pixels (13.0%)

**Structural (SVH) trajectory**:
- Recovery: 3.93 M pixels (13.9%)
- Stable: 21.56 M pixels (76.1%)
- Degradation: 2.83 M pixels (10.0%)

**Combined classification**:
- Both recovering: 2.06 M (7.3%)
- Any recovering (GPP or SVH): 5.50 M (19.4%)
- Both stable: 17.85 M (63.0%)
- Both degrading: 1.46 M (5.2%)

---

## 3. Recovery Degree Estimation

### 3.1 Objective

For pixels classified as recovering, we quantify the *degree* of recovery relative to natural reference conditions. Four metrics are computed — each capturing a different dimension of recovery — and percentile-normalised for aggregation into a composite score.

### 3.2 Defining Natural Reference

Natural reference pixels are established using **Feature Space Coverage Sampling (FSCS)**, a technique designed to ensure that sample points are highly representative of heterogeneous landscapes. For each ecoregion, a 10 km grid is laid over the region, and within each grid cell, AlphaEarth satellite embeddings are clustered into 100 groups using KMeans. The pixel closest to each cluster centroid is selected. This ensures comprehensive coverage across the full range of natural environmental conditions (topography, climate, soil types), preventing the over-sampling of common land covers while missing rarer habitat types.

A sampled pixel is definitively classified as "natural" if it satisfies a strict composite mask applied during GEE extraction:

- **(SBTN Natural Lands OR Natural Forests probability >= 0.52)** — must be classified as natural land by the WRI SBTN dataset (2020) or as natural forest by the Nature-Trace forest typology
- **AND Global Human Modification <= 0.1** — very low human footprint (GHM v3, 2022). On the 0–1 scale, 0.1 indicates essentially no roads, settlements, agriculture, or other human infrastructure
- **AND Biodiversity Intactness Index >= 0.7** — ecological community remains largely intact (BII 1 km). A value of 0.7 means at least 70% of original species abundance is estimated to persist

Across all 18 ecoregions, the FSCS design generated a total of **1,849,707 reference points**, of which **1,004,898** were verified as natural. These natural pixels form an independent reference dataset (completely separate from the abandoned-ag pixels) that provide baseline boundaries. For any given ecoregion's recovery assessment, only the natural reference pixels belonging specifically to that same ecoregion are used.

### 3.3 Test and Reference Data

| Role | Source | Selection |
|------|--------|-----------|
| **Test pixel** | `abandoned_ag_gpp_2000_2022_SA.parquet` | Recovering pixel (both GPP and SVH significant positive trend), target ecoregion |
| **Natural reference** | `dfsubsetNatural` x `indices_gpp_svh` | Same ecoregion, composite natural mask |
| **Transformed reference** | `indices_gpp_svh_2000_2022.parquet` | Same ecoregion, SANLC 2022 classes 3–7 |

### 3.4 Percentile Normalisation

Each metric below produces a raw score on its own scale (a percentile rank, a ratio, or a similarity value). To make these comparable and combinable, each is expressed as a **percentile within the natural reference distribution**.

**How this works**: For a given metric, we first compute the metric for every natural reference pixel. This creates a distribution of "natural" values. We then compute the same metric for the test pixel (recovering pixel) and determine what fraction of natural pixels it outperforms:

> Percentile = (number of natural pixels with a lower score / total natural pixels) x 100

This transforms all metrics onto a common 0–100 scale where:
- **0** = worse than all natural pixels
- **50** = at the median of natural conditions (typical natural performance)
- **100** = better than all natural pixels

This is important because the raw values of each metric are on different scales (a GPP value in gC/m2/yr, a dimensionless ratio, a cosine similarity score between 0 and 1). Percentile normalisation makes them directly comparable and allows meaningful averaging into a composite score.

### 3.5 Metric A: Percentile within Natural Distribution

**What it measures**: Where the test pixel's recent productivity falls within the distribution of natural reference pixels in the same ecoregion.

**Computation**: The test pixel's mean GPP (or SVH) over 2018–2022 is ranked within the distribution of late-period means across all ecoregion natural reference pixels.

**Interpretation**: 50th percentile = test pixel matches the median natural condition. Values above 50 indicate productivity equal to or exceeding natural; below 50 indicates the pixel has not yet reached typical natural levels. This metric uses the entire ecoregion as context and does not account for local spatial variation.

### 3.6 Metric B: K-Nearest Natural Neighbours Ratio

**What it measures**: How the test pixel compares to its geographically closest natural counterparts, capturing local environmental context such as topographic and microclimatic variation within an ecoregion.

**Computation**: The K=10 nearest natural reference pixels within 50 km are identified using geographic (great-circle) distance. The ratio of the test pixel's late-period mean to the KNN mean is computed, then percentile-normalised within the natural distribution. The natural baseline is constructed by computing each natural pixel's ratio to the ecoregion-wide mean.

**Interpretation**: A ratio of 1.0 means the test pixel matches its local natural neighbours. The percentile places this in context of natural variability — controlling for the fact that some natural areas are inherently more or less productive than the ecoregion average.

### 3.7 Metric C: Cosine Similarity (AlphaEarth Embeddings)

**What it measures**: How similar the test pixel's overall landscape "fingerprint" is to natural reference pixels, based on the full 64-dimensional AlphaEarth embedding. This captures characteristics beyond GPP and SVH — including vegetation composition, canopy structure, phenological timing, moisture regime, and terrain.

**How cosine similarity works**: Each pixel's embedding is a vector of 64 numbers. Cosine similarity measures how similar the *direction* of two vectors is, ignoring their length (all AlphaEarth embeddings are unit-length). It ranges from -1 (opposite) through 0 (unrelated) to +1 (identical). For landscape embeddings, values typically fall between 0.7 and 1.0. Two patches of intact grassland in the same ecoregion might score 0.95; a recovering field compared to natural grassland might score 0.85.

The test pixel's embedding is extracted directly from the AlphaEarth annual mosaic (2022) via GEE at the pixel's coordinates. Cosine similarity is computed at two spatial scales:

**Ecoregion scope (C_eco)**: Mean cosine similarity between the test embedding and all ecoregion natural reference embeddings. Percentile-normalised against a baseline of natural-to-natural similarities (500 randomly sampled natural pixels, each compared against all other natural pixels in the ecoregion). This asks: is the test pixel as similar to natural as natural pixels are to each other?

**Local scope (C_local)**: Mean cosine similarity between the test embedding and the same K=10 nearest natural neighbours used in Metric B. To percentile-normalise, a baseline is constructed by sampling 300 natural pixels across the ecoregion, computing each one's mean cosine similarity to its own K=10 nearest natural neighbours. This mirrors the KNN approach used in Metric B — each natural pixel is compared to its own local neighbours, producing a smooth baseline distribution rather than relying on the 10 test-specific neighbours alone.

**Interpretation**: A percentile near 50 indicates the test pixel's multi-source landscape fingerprint is indistinguishable from natural. Lower values suggest that despite productivity recovery (captured by Metrics A and B), differences in vegetation composition, structure, phenology, or other landscape properties persist. This metric is particularly valuable for detecting situations where productivity has recovered but the ecosystem has not yet returned to a natural-like state in other ecologically important dimensions.

### 3.8 Composite Score and Gradient Position

**Composite recovery score**: The mean of all percentile scores across metrics A, B, C (ecoregion), and C (local), computed separately for GPP and SVH, then averaged:

> Score_overall = mean(Score_GPP, Score_SVH)

where each sub-score is the mean of the four metric percentiles for that variable. The overall score ranges from 0 to 100, where 50 represents the median natural condition.

**Gradient position**: The test pixel can be positioned on a transformed-to-natural continuum (0 = equivalent to transformed conditions, 1 = equivalent to median natural), using the percentile scores of both the test pixel and the average transformed pixel within the natural distribution.

---

## 4. Batch Results: Highveld Grasslands (Ecoregion 81)

### 4.1 Processing Scale

The full recovery degree analysis was run for all recovering pixels in the Highveld Grasslands ecoregion — the dominant grassland biome across South Africa's interior plateau.

| Item | Count |
|------|-------|
| Recovering pixels identified (Stage 1) | 317,811 |
| AlphaEarth embeddings extracted via GEE | 317,811 |
| Total pixel-records scored (Stage 2) | 826,973 |
| Natural reference pixels (eco_id=81) | 59,195 |

Stage 1 identified recovering pixels using vectorised Mann-Kendall + Theil-Sen in 5,000-pixel sub-batches, then extracted AlphaEarth embeddings via the GEE high-volume endpoint (20 parallel workers, 2,000 points per batch, DuckDB buffer with JSON checkpoint resumability).

Stage 2 pre-computed all baselines once (natural GPP/SVH distributions, KNN ratios, cosine similarity baselines), then scored in 2,000-pixel chunks.

### 4.2 Summary Statistics

| Metric | Mean | Median | Std Dev |
|--------|------|--------|---------|
| A (GPP percentile) | 90.9 | 94.7 | 11.6 |
| A (SVH percentile) | 86.3 | 88.7 | 12.0 |
| B (KNN GPP ratio percentile) | 72.5 | 77.0 | 19.3 |
| B (KNN SVH ratio percentile) | 74.8 | 80.1 | 23.4 |
| C eco (embedding similarity percentile) | 28.5 | 25.0 | 18.5 |
| C local (embedding similarity percentile) | 17.6 | 15.0 | 16.7 |
| **Composite recovery score** | **52.1** | **51.9** | **9.5** |

Score range: 0.1 to 91.2 (out of 100).

### 4.3 Ecological Interpretation

The results reveal a clear divergence between productivity-based and embedding-based metrics:

**Metrics A and B** (well above 50): Recovering pixels have regained or exceeded the GPP and SVH of natural grasslands. The high percentiles indicate that these pixels are more productive than the majority of natural reference sites. This is consistent with the significant positive trends that flagged them as recovering.

**Metric C** (well below 50): Despite strong productivity, the multi-source satellite fingerprint of these pixels still differs from intact natural grassland. Possible ecological explanations include:
- Dominance of pioneer or secondary grass species rather than climax grassland assemblages
- Uniform canopy structure rather than the spatial heterogeneity of mature grassland
- Different phenological timing (green-up, peak, senescence) relative to natural areas
- Residual soil compaction or nutrient profiles from prior agricultural use
- Altered moisture dynamics or microhabitat diversity

**Composite score of ~52**: On average, recovering Highveld pixels sit near the natural median when all dimensions are considered. However, this masks the divergence between productivity recovery (strong, metrics A and B) and compositional/structural recovery (lagging, metric C). This pattern — functional recovery outpacing compositional recovery — is well documented in restoration ecology and is consistent with secondary succession dynamics where productivity recovers before species composition and community structure.

---

## 5. Outputs

### Trajectory Classification
- `data/trajectory_results.parquet` — Pixel-level classification results
- `reports/trajectory_recovery_report.md` — Full results including per-ecoregion breakdowns, cross-tabulation of GPP x SVH classes, and mean slope statistics

### Recovery Degree Estimation
- `data/recovering_eco81.parquet` — Recovering pixels identified for ecoregion 81
- `data/recovering_eco81_embeddings.parquet` — AlphaEarth embeddings for recovering pixels
- `data/recovery_scores_eco81.parquet` — Full recovery scores (826,973 rows, 19 columns)
- `plots/recovery_degree_comparison.png` — 2x2 panel: GPP trajectory, SVH trajectory, percentile bar chart, summary table
- `plots/recovery_degree_diagnostic.png` — 2x3 diagnostic: late-period histograms, spatial map with KNN, spaghetti time series, GPP-vs-SVH scatter

---

## 6. Key Parameters

| Parameter | Value | Stage |
|-----------|-------|-------|
| Significance level | 0.05 | Trajectory classification |
| Time series length | 23 years (2000–2022) | Both |
| Normalisation | Per-ecoregion Z-score | Trajectory classification |
| Late period | 2018–2022 | Recovery degree |
| KNN neighbours (K) | 10 | Recovery degree (Metrics B, C local) |
| KNN search radius | 50 km | Recovery degree (Metrics B, C local) |
| Ecoregion cosine baseline sample | 500 natural pixels | Recovery degree (Metric C eco) |
| Local cosine baseline sample | 300 natural pixels | Recovery degree (Metric C local) |
| Embedding dimension | 64 | Recovery degree (Metric C) |
| Embedding resolution | 10 m | Recovery degree (Metric C) |
| Embedding source year | 2022 | Recovery degree (Metric C) |
| Batch chunk size (scoring) | 2,000 pixels | Recovery degree (batch) |
| GEE extraction workers | 20 parallel | Recovery degree (batch) |

---

## 7. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/extraction/abandoned_ag_extract.py` | Extract GPP/SVH time series for abandoned-ag pixels from GEE |
| `scripts/extraction/extract_gpp_svh_for_indices.py` | Extract GPP/SVH time series for reference sample points from GEE |
| `scripts/extraction/gee_extraction.py` | Extract remote sensing indices + AlphaEarth embeddings for reference sample |
| `scripts/extraction/gee_extraction_binary.py` | Extract composite natural mask labels for reference sample |
| `scripts/analysis/trajectory_recovery_report.py` | Trajectory classification (MK + Theil-Sen) and report generation |
| `scripts/analysis/test_recovery_degree.py` | Recovery degree estimation (single pixel, Metrics A, B, C) with visualisation |
| `scripts/analysis/batch_recovery_degree.py` | Batch recovery degree estimation for all recovering pixels in an ecoregion |
