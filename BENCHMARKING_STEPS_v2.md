# Ecological Condition Benchmarking Methodology (HCAS v3.1)

> **Script:** `scripts/select_benchmark_sites.py`
> **Version:** v2 — February 2026
> **Purpose:** This document describes every step, hyperparameter, and design rationale implemented in the benchmarking script, intended for peer review.

---

## Overview

The script calculates a calibrated **Habitat Condition Assessment System (HCAS)** score for each "transformed" (test) site. It does this by comparing each test site against a set of ecologically similar "natural" (reference) sites, quantifying how far the test site's current spectral condition departs from the expected natural baseline.

### Input

- **`reference_departure_with_intervals.csv`** — one row per site, containing:
  - `natural` flag (1 = reference, 0 = transformed)
  - `geo_x` — GeoJSON-formatted coordinate string
  - Predicted reference values from expected reference condition model: `NBR_ref`, `NDMI_ref`, `NDWI_ref`
  - Observed departure values (expected-observed): `NBR_diff`, `NDMI_diff`, `NDWI_diff`

### Output

- **`benchmarked_condition.csv`** — all transformed-site columns plus `comp_dep` and `HCAS_score`
- **`transformed_hcas.shp`** — point shapefile (EPSG:4326) with `id`, `comp_dep`, `HCAS_score`

---

## Feature Sets (Step 1)

Two distinct feature sets are defined, each with its own `StandardScaler`:

| Purpose | Columns | Scaler | Used in |
|---|---|---|---|
| **Benchmark selection** (environmental similarity) | `NBR_ref`, `NDMI_ref`, `NDWI_ref` | `scaler_dist` | Steps 2, 4, 5, 6 |
| **Condition scoring** (departure from expected) | `NBR_diff`, `NDMI_diff`, `NDWI_diff` | `scaler_score` | Step 7 |

> [!IMPORTANT]
> Two separate `StandardScaler` instances are used. `scaler_dist` is fit on the **reference-site** predicted values and then applied to both reference and transformed sites. `scaler_score` is fit on the **reference-site** departure values and then applied to both. Missing values are filled with 0 before scaling.

**Rationale:** Benchmark selection should be based on *environmental potential* (what a site should naturally look like), not its current degraded state. Scoring, on the other hand, uses the *departure* between observed and predicted values to measure actual condition. Keeping separate scalers prevents cross-contamination of units between these two conceptually different measurements.

---

## Data Split (Step 2)

Sites are split by the `natural` column:
- `natural == 1` → Reference pool
- `natural == 0` → Test (transformed) sites

> [!NOTE]
> Fallback: If no transformed sites exist (e.g., during development), 500 natural sites are randomly sampled as test sites (`random_state=42`).

---

## Standardization (Step 3)

Both feature sets are independently standardized (zero mean, unit variance) using `sklearn.preprocessing.StandardScaler`.

| Scaler | Fit on | Transform applied to |
|---|---|---|
| `scaler_dist` | Reference predicted values (`NBR_ref`, `NDMI_ref`, `NDWI_ref`) | Reference + Transformed predicted values |
| `scaler_score` | Reference departure values (`NBR_diff`, `NDMI_diff`, `NDWI_diff`) | Reference + Transformed departure values |

**Rationale:** Standardization ensures that each index contributes equally to multi-dimensional distance calculations, preventing any single index from dominating due to scale differences.

---

## Spatial Index (Step 4)

A `BallTree` is built from reference-site coordinates using the **Haversine** metric (coordinates converted to radians).

**Rationale:** Enables efficient radius-based and k-nearest-neighbor queries in geographic space, required for Steps 5 and the background distribution.

---

## Background Density Distribution (Step 5 — Pre-computation)

> [!IMPORTANT]
> This step runs **once** before the per-site loop. It constructs the natural-variability baseline used later in Step 9.

### Procedure

1. Randomly sample **2,000** reference sites (without replacement).
2. For each of the first **200** pivot sites in that sample:
   a. Find all reference neighbors within a **1,000 km** radius.
   b. Sub-sample up to **100** of those neighbors (without replacement).
   c. Compute **Manhattan distance** in standardized reference-condition space between the pivot and each neighbor.
3. Aggregate all pairwise distances into a histogram.

### Histogram Parameters

| Parameter | Value | Notes |
|---|---|---|
| Number of bins | 400 | |
| Range | (0, 20.0) | Standardized Manhattan distance units |
| Bin width | 0.05 | = 20.0 / 400 |
| `density` flag | `True` | Histogram integrates to 1.0 (PDF) |

4. Convert the histogram to a continuous function via `scipy.interpolate.interp1d` (linear interpolation) with a floor of `1e-9` to avoid zero-density artifacts.

**Rationale:** This distribution defines the range of "normal" spectral differences between pairs of undisturbed natural sites. It is used in Step 9 to select benchmark sites that represent the *most typical* natural comparisons — i.e., the mode of the distribution — rather than extreme outliers.

> [!NOTE]
> The 1,000 km sampling radius for the background distribution is intentionally much larger than the 200 km inclusion zone (Step 6). This ensures the density baseline captures the full natural variability of reference-to-reference distances at a regional scale, providing a stable estimate of "typicality."

---

## Per-Site Benchmarking Loop (Steps 6–11)

The following steps are repeated for **each** transformed (test) site.

---

### Step 6 — Spatial Filtering (200 km Inclusion Zone)

| Parameter | Value |
|---|---|
| `BENCHMARK_INCLUSION_KM` | 200.0 |
| Earth radius | 6,371.0 km |

**Action:** Query the BallTree for all reference sites within 200 km of the test site.

**Fallback:** If fewer than `FINAL_K` (10) reference sites are found within 200 km, the script falls back to the **100 nearest** reference sites regardless of distance.

**Rationale:** Restricting benchmarks to the local geographic context ensures climate, soil, and biome relevance (Tobler's First Law of Geography). The fallback prevents failure for geographically isolated test sites.

---

### Step 7 — Spectral Distance Calculation (Manhattan Distance)

**Action:** Compute the **Manhattan distance** (L1 norm) in standardized reference-condition space between the test site and each candidate reference site from Step 6.

**Formula:** `d = Σ |x_test,j − x_ref,j|` over `j ∈ {NBR_ref, NDMI_ref, NDWI_ref}` (standardized)

**Rationale:** Manhattan distance on predicted reference values finds reference sites with the most similar *environmental potential*, independent of each site's current condition.

---

### Step 8 — Top-K Reference-Space Filter

| Parameter | Value |
|---|---|
| `TOP_K_RS` | 70 |

**Action:** Rank candidates by ascending Manhattan distance and retain the closest **70**.

**Rationale:** Reduces the candidate pool to the most ecologically similar sites before applying the computationally heavier geographic penalty and density-based ranking.

---

### Step 9 — Geographic Distance Penalty

| Parameter | Value |
|---|---|
| `PENALTY_FACTOR` | 30.0 (km) |

**Action:** Adjust each candidate's spectral distance using its geographic distance from the test site:

```
Adj_Distance = RS_Distance × (1 + Geo_Distance_km / 30.0)
```

Geographic distance is computed using the **Haversine formula** (inline, not via BallTree).

**Rationale:** Geographically closer reference sites are more likely to share unmeasured environmental drivers (local soil type, fire history, specific hydrology). The penalty inflates the effective distance of geographically distant candidates, favoring nearby ones when spectral similarity is comparable. A `PENALTY_FACTOR` of 30.0 means a site 30 km away has its spectral distance doubled; a site 300 km away has it multiplied by 11×.

---

### Step 10 — Mode-Based Selection (Density Ranking)

| Parameter | Value |
|---|---|
| `FINAL_K` | 10 |

**Action:** Using the background density function from Step 5, look up the probability density for each of the 70 penalized distances. Select the **10 candidates with the highest density** (i.e., whose adjusted distance falls closest to the mode of the natural reference distribution).

**Rationale:** Rather than simply selecting the 10 closest sites (which may be statistical outliers — unusually similar by chance), this step selects sites whose similarity to the test site is *most representative of typical natural variation*. This yields more robust, less noisy benchmarks.

---

### Step 11 — Weighted Scoring (Half-Cauchy, Per-Pair Departure)

| Parameter | Value |
|---|---|
| `CAUCHY_LAMBDA` (λ) | 2.0 |

#### 11a. Weighting

Each of the 10 selected benchmarks is weighted using a **Half-Cauchy** kernel on its penalized distance:

```
w = 1 / (1 + (distance / λ)²)
```

Weights are then normalized to sum to 1.0.

**Rationale:** The Half-Cauchy kernel has heavier tails than a Gaussian, gracefully down-weighting distant benchmarks without completely discarding them. λ = 2.0 means a benchmark at distance 2.0 receives half the weight of one at distance 0.

#### 11b. Composite Departure Calculation

**Actual procedure:**

1. For each of the 10 benchmarks, compute the **Manhattan distance** between the test site's standardized departure vector (`NBR_diff`, `NDMI_diff`, `NDWI_diff`) and the benchmark's standardized departure vector:
   ```
   pair_departure_i = |T_NBR_diff − B_i_NBR_diff| + |T_NDMI_diff − B_i_NDMI_diff| + |T_NDWI_diff − B_i_NDWI_diff|
   ```
2. Compute the **weighted average** of these 10 individual departures:
   ```
   Composite_Departure = Σ (w_i × pair_departure_i)
   ```

**Rationale (from code comments):** Computing per-pair distances and then averaging prevents "averaging out" the departures. If benchmark departure vectors point in different directions, a centroid approach could produce a near-zero vector even when individual departures are large. The per-pair approach yields a more realistic (and typically higher) composite departure.

---

### Step 12 — Calibration to HCAS Score [0, 1]

| Parameter | Value |
|---|---|
| Departure threshold | 2.0 (standardized units) |
| Calibration anchor: "Highly Modified" | (0.0, 0.101) |
| Calibration anchor: "Reference" | (1.0, 0.944) |
| Interpolation | PCHIP (monotonic cubic Hermite) |

**Action:**

1. Map the composite departure to an uncalibrated score: `uncalibrated = clip(1 − comp_dep / 2.0, 0, 1)`
2. Apply the PCHIP spline to produce the final calibrated HCAS score.

**Interpretation:**
- A composite departure of 0.0 → uncalibrated = 1.0 → calibrated ≈ **0.944** (near-reference condition)
- A composite departure of ≥ 2.0 → uncalibrated = 0.0 → calibrated ≈ **0.101** (highly modified)

**Rationale:** The threshold of 2.0 standardized Manhattan-distance units corresponds conceptually to a "2 standard deviations" departure across 3 indices. The PCHIP spline guarantees monotonicity — greater similarity to benchmarks always produces a higher condition score, avoiding non-physical inversions. The calibration anchors (0.101 and 0.944 rather than 0.0 and 1.0) compress the score range to avoid overconfident extremes.

---

## Hyperparameter Summary Table

| Parameter | Symbol | Value | Step | Purpose |
|---|---|---|---|---|
| Spatial inclusion radius | `BENCHMARK_INCLUSION_KM` | 200 km | 6 | Geographic relevance filter |
| Top spectral candidates | `TOP_K_RS` | 70 | 8 | Pre-penalty shortlist |
| Geographic penalty factor | `PENALTY_FACTOR` | 30.0 km | 9 | Scales geographic-distance penalty |
| Final benchmark count | `FINAL_K` | 10 | 10 | Benchmarks per test site |
| Cauchy weighting scale | `CAUCHY_LAMBDA` | 2.0 | 11 | Controls weight falloff |
| Departure threshold | — | 2.0 | 12 | Maps departure to [0,1] |
| Calibration anchors | `CALIB_POINTS` | (0→0.101, 1→0.944) | 12 | PCHIP spline endpoints |
| Background sample pool | — | 2,000 sites | 5 | Density estimation |
| Background pivot count | — | 200 sites | 5 | Actual density computation |
| Background neighbor radius | — | 1,000 km | 5 | Regional density scope |
| Background neighbor cap | — | 100 per pivot | 5 | Limits pairwise comparisons |
| Histogram bins | — | 400 (width 0.05) | 5 | Density resolution |
| Spatial fallback k | — | 100 | 6 | If < 10 sites in 200 km |

---

## Questions for Reviewer

Please evaluate whether the following choices are appropriate, or flag any misunderstandings:

1. **Dual scaler design** — Is it correct that benchmark selection uses predicted reference values while scoring uses departure values, each with an independently fit `StandardScaler`?
2. **Per-pair vs. centroid departure** (Step 11b) — The code computes Manhattan distance from the test site to each benchmark individually and then averages. Is this the intended behaviour, or should a centroid approach be used?
3. **Background density scope** (1,000 km) vs. inclusion zone (200 km) — The background density is sampled at a much wider geographic scale than the per-site inclusion zone. Is this intentional?
4. **Departure threshold of 2.0** — Is this an appropriate mapping threshold for standardized Manhattan distance across 3 indices?
5. **Calibration anchors (0.101, 0.944)** — Where do these specific values originate? Are they derived from field validation or expert judgement?
6. **Missing values filled with 0** — Before standardization, `fillna(0)` is applied to both feature sets. Is zero-fill appropriate, or should these rows be excluded?
7. **Spatial fallback (k=100)** — When fewer than 10 reference sites exist within 200 km, the code retrieves the 100 nearest reference sites globally. Is this fallback acceptable, or should it use a different strategy (e.g., expanding the radius incrementally)?
