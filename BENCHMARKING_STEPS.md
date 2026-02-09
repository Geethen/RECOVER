# Ecological Condition Benchmarking Methodology (HCAS v3.1)

This document outlines the workflow and parameter settings implemented in `scripts/select_benchmark_sites.py` for calculating calibrated habitat condition scores.

## Overview
The benchmarking methodology follows the **Habitat Condition Assessment System (HCAS) v3.1** protocol. It aims to determine the condition of "transformed" (test) sites by comparing their spectral signatures against a set of ecologically similar "natural" (reference) sites.

## Steps and Rationale

### 1. Spatial Filtering (200km Inclusion Zone)
**Action:** For each test site, identify all eligible reference sites within a 200km radius.
**Rationale:** Restricting benchmarks to the local geographic context ensures that the environmental processes and potential "reference states" are biologically and climatically relevant to the site being assessed (Tobler's First Law of Geography).

### 2. Spectral Distance Calculation (Manhattan Distance)
**Action:** Calculate the Manhattan distance between the test site and all reference sites in the inclusion zone using **Predicted Reference Condition** values (`NBR_ref`, `NDMI_ref`, `NDWI_ref`).
**Rationale:** Selection is based on environmental potential. We want to find benchmark sites that *should naturally look* like how the test site would naturally look, regardless of their current state. Manhattan distance effectively captures this multi-index similarity.

### 3. Background Distribution & Interpolation (Step 3)
**Action:** Construct a histogram of reference-to-reference distances (400 bins, 0.05 width, up to 20.0 max distance) and apply linear interpolation to create a continuous Probability Density Function (PDF).
**Rationale:** This distribution serves as the **Natural Variability Baseline**. It defines the statistical range of "normal" spectral differences within undisturbed ecosystems.
**Link to Step 6:** The continuous PDF generated here is used in Step 6 to evaluate the quality of candidate benchmarks. By mapping the distance of a benchmark candidate onto this PDF, we can determine its "typicality." This allows us to select benchmarks that represent the core mode (highest frequency) of natural behavior, rather than focusing on the absolute closest or geographically nearest sites which might be outliers.

### 4. Benchmark Filtering (Top 70 RS Filter)
**Action:** Filter the candidate reference sites to the top 70 closest in spectral feature space.
**Rationale:** This narrows the pool to the most ecologically similar potential benchmarks before applying penalties.

### 5. Geographic Distance Penalty (Factor = 30)
**Action:** Penalize the spectral distance based on geographic distance: `Adj_Dist = RS_Dist * (1 + Geo_Dist_km / 30.0)`.
**Rationale:** This encourages the selection of benchmarks that are geographically closer, acknowledging that spatial proximity often correlates with unmeasured environmental factors (e.g., local soil types, specific fire histories).

### 6. Mode-Based Selection (Highest Frequency)
**Action:** Of the penalized top 70, select the 10 sites that fall into distance bins with the **highest histogram frequency** (highest probability density).
**Rationale:** Selecting benchmarks that represent the "most common" natural comparisons (the mode of the reference distribution) reduces bias from extreme/unusual reference sites and ensures the condition score reflects typical ecosystem integrity.

### 7. Weighted Averaging & Multivariate Departure (Half-Cauchy λ=2.0)
**Action:** Calculate the weighted average "Reference Departure" of the 10 benchmarks across all three indices. Then, calculate the **Composite Departure** as the Manhattan distance between the test site's observed departure vector (`_diff`) and this Benchmark Departure vector.
**Rationale:** Using the multivariate departure ensures the condition score considers how far the site has strayed from its specific benchmark expectations across multiple dimensions simultaneously.

### 8. Monotonic H-Spline Calibration
**Action:** Transform the Composite Departure into a 0.0–1.0 condition index and calibrate using a monotonic PCHIP spline anchored at **0.101** (Highly Modified) and **0.944** (Reference). A total standardized departure of 2.0 is currently used as the threshold for high modification.
**Rationale:** Calibration ensures that scores are comparable across different regions. Using a monotonic spline (PCHIP) guarantees that greater spectral similarity to benchmarks always results in a higher condition score, avoiding non-physical artifacts.

## Outputs
- **`benchmarked_condition.csv`**: Contains the composite departure value (`comp_dep`) and calibrated `HCAS_score` for all test sites.
- **`transformed_hcas.shp`**: A spatial dataset for mapping and GIS analysis of habitat integrity across transformed areas, featuring the single multivariate HCAS score.
