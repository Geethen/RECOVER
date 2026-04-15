# Trajectory Recovery Analysis Report

**Date**: 2026-03-10  
**Dataset**: `abandoned_ag_gpp_2000_2022_SA.parquet`  
**Time series**: GPP and SVH, 2000-2022 (23 years)  
**Method**: Mann-Kendall trend test + Theil-Sen slope (per-ecoregion Z-score standardized)  
**Significance level**: alpha = 0.05

## 1. Data Overview

- **Total pixels**: 33,403,934
- **Natural-LC pixels (SANLC 1-2)**: 28,312,040 (84.8%)
- **Transformed-LC pixels dropped**: 5,091,894 (15.2%)

### Overall land cover distribution

| SANLC Code | Land Cover | n | % |
|---:|---|---:|---:|
| 1 | Natural/near-natural | 451,073 | 1.4% |
| 2 | Secondary natural | 27,860,967 | 83.4% |
| 3 | Artificial water | 63,157 | 0.2% |
| 4 | Built-up | 926,040 | 2.8% |
| 5 | Cropland | 3,592,319 | 10.8% |
| 6 | Mine | 92,679 | 0.3% |
| 7 | Plantation | 417,698 | 1.3% |

### Land cover composition by ecoregion (%)

| Ecoregion | Artificial water | Built-up | Cropland | Mine | Natural/near-natural | Plantation | Secondary natural | Total pixels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Knysna-Amatole montane forests | 0.1% | 14.4% | 11.8% | 0.1% | 0.7% | 13.2% | 59.8% | 71,613 |
| Kwazulu Natal-Cape coastal forests | 0.0% | 5.5% | 4.1% | 0.2% | 0.9% | 0.6% | 88.6% | 621,049 |
| Maputaland coastal forests and woodlands | 0.1% | 2.0% | 13.6% | 0.0% | 2.4% | 2.6% | 79.5% | 258,039 |
| Central bushveld | 0.1% | 5.2% | 10.7% | 0.6% | 1.0% | 0.6% | 81.7% | 4,116,741 |
| Drakensberg Escarpment savanna and thicket | 0.1% | 1.9% | 4.2% | 0.0% | 1.4% | 0.8% | 91.5% | 2,787,561 |
| Drakensberg grasslands | 0.3% | 1.9% | 10.4% | 0.1% | 1.4% | 2.1% | 83.7% | 5,253,413 |
| Limpopo lowveld | 0.1% | 4.8% | 12.0% | 0.1% | 2.1% | 0.4% | 80.6% | 1,879,597 |
| Zambezian mopane woodlands | 0.1% | 0.0% | 3.0% | 0.0% | 1.0% | 0.3% | 95.6% | 1,536 |
| Highveld grasslands | 0.2% | 2.9% | 12.5% | 0.5% | 1.5% | 2.0% | 80.3% | 9,950,898 |
| Albany thickets | 0.0% | 0.9% | 3.5% | 0.0% | 1.5% | 0.3% | 93.6% | 697,759 |
| Fynbos shrubland | 0.1% | 2.8% | 16.3% | 0.1% | 1.1% | 0.6% | 79.0% | 1,855,927 |
| Renosterveld shrubland | 0.1% | 3.0% | 22.2% | 0.3% | 0.9% | 0.4% | 73.1% | 1,106,327 |
| Gariep Karoo | 1.6% | 0.3% | 1.8% | 0.0% | 1.3% | 0.0% | 95.0% | 234,553 |
| Kalahari xeric savanna | 0.1% | 0.9% | 11.9% | 0.1% | 1.0% | 0.3% | 85.7% | 2,129,672 |
| Nama Karoo shrublands | 0.4% | 0.2% | 5.8% | 0.0% | 1.4% | 0.2% | 91.9% | 1,101,617 |
| Namaqualand-Richtersveld steppe | 0.0% | 0.1% | 2.4% | 0.0% | 1.1% | 0.0% | 96.3% | 606,273 |
| Succulent Karoo xeric shrublands | 0.1% | 0.7% | 4.7% | 0.0% | 1.3% | 0.1% | 93.1% | 688,144 |
| Southern Africa mangroves | 0.0% | 17.1% | 1.3% | 0.2% | 1.1% | 4.7% | 75.7% | 43,214 |

## 2. Functional Trajectory (GPP)

Classification based on Mann-Kendall significance (p < 0.05) and Theil-Sen slope direction.

| Class | n | % |
|---|---:|---:|
| Recovery | 3,627,800 | 12.8% |
| Stable | 21,014,891 | 74.2% |
| Degradation | 3,669,349 | 13.0% |

## 3. Structural Trajectory (SVH)

| Class | n | % |
|---|---:|---:|
| Recovery | 3,927,356 | 13.9% |
| Stable | 21,559,613 | 76.1% |
| Degradation | 2,825,071 | 10.0% |

## 4. Combined Functional x Structural Classification

Cross-tabulation of GPP and SVH trajectory classes.

### Counts

| GPP \ SVH | Recovery | Stable | Degradation | Total |
|---|---:|---:|---:|---:|
| **Recovery** | 2,059,775 | 1,540,761 | 27,264 | 3,627,800 |
| **Stable** | 1,830,171 | 17,850,415 | 1,334,305 | 21,014,891 |
| **Degradation** | 37,410 | 2,168,437 | 1,463,502 | 3,669,349 |
| **Total** | 3,927,356 | 21,559,613 | 2,825,071 | 28,312,040 |

### Percentages

| GPP \ SVH | Recovery | Stable | Degradation | Total |
|---|---:|---:|---:|---:|
| **Recovery** | 7.28% | 5.44% | 0.10% | 12.81% |
| **Stable** | 6.46% | 63.05% | 4.71% | 74.23% |
| **Degradation** | 0.13% | 7.66% | 5.17% | 12.96% |
| **Total** | 13.87% | 76.15% | 9.98% | 100.00% |

### Combined class distribution

| Combined Class | n | % |
|---|---:|---:|
| Degradation + Degradation | 1,463,502 | 5.2% |
| Degradation + Recovery | 37,410 | 0.1% |
| Degradation + Stable | 2,168,437 | 7.7% |
| Recovery + Degradation | 27,264 | 0.1% |
| Recovery + Recovery | 2,059,775 | 7.3% |
| Recovery + Stable | 1,540,761 | 5.4% |
| Stable + Degradation | 1,334,305 | 4.7% |
| Stable + Recovery | 1,830,171 | 6.5% |
| Stable + Stable | 17,850,415 | 63.0% |

### Key recovery metrics

- **Both recovering** (GPP + SVH): 2,059,775 (7.3%)
- **Functional recovery only** (GPP recovering, SVH stable): 1,540,761 (5.4%)
- **Structural recovery only** (SVH recovering, GPP stable): 1,830,171 (6.5%)
- **Any recovery** (at least one recovering): 5,495,381 (19.4%)
- **Both stable**: 17,850,415 (63.0%)
- **Both degrading**: 1,463,502 (5.2%)

## 5. Ecoregion Breakdown

### GPP trajectory by ecoregion (%)

| Ecoregion | n | Recovery | Stable | Degradation |
|---|---:|---:|---:|---:|
| Knysna-Amatole montane forests | 43,321 | 34.5% | 56.2% | 9.3% |
| Kwazulu Natal-Cape coastal forests | 555,848 | 42.5% | 52.2% | 5.3% |
| Maputaland coastal forests and woodlands | 211,120 | 23.5% | 68.2% | 8.3% |
| Central bushveld | 3,405,923 | 9.8% | 79.6% | 10.6% |
| Drakensberg Escarpment savanna and thicket | 2,590,966 | 21.4% | 64.2% | 14.4% |
| Drakensberg grasslands | 4,473,419 | 14.1% | 67.9% | 18.0% |
| Limpopo lowveld | 1,553,313 | 17.2% | 69.2% | 13.6% |
| Zambezian mopane woodlands | 1,484 | 2.8% | 97.1% | 0.1% |
| Highveld grasslands | 8,133,594 | 6.3% | 81.5% | 12.2% |
| Albany thickets | 664,078 | 18.2% | 70.6% | 11.3% |
| Fynbos shrubland | 1,485,903 | 24.4% | 58.3% | 17.3% |
| Renosterveld shrubland | 818,745 | 25.8% | 61.1% | 13.1% |
| Gariep Karoo | 225,941 | 10.5% | 65.5% | 24.0% |
| Kalahari xeric savanna | 1,847,115 | 3.4% | 94.2% | 2.4% |
| Nama Karoo shrublands | 1,028,055 | 3.4% | 81.2% | 15.4% |
| Namaqualand-Richtersveld steppe | 590,821 | 18.6% | 74.3% | 7.1% |
| Succulent Karoo xeric shrublands | 649,222 | 13.8% | 65.2% | 21.1% |
| Southern Africa mangroves | 33,172 | 48.0% | 46.3% | 5.6% |

### SVH trajectory by ecoregion (%)

| Ecoregion | n | Recovery | Stable | Degradation |
|---|---:|---:|---:|---:|
| Knysna-Amatole montane forests | 43,321 | 31.5% | 56.3% | 12.1% |
| Kwazulu Natal-Cape coastal forests | 555,848 | 32.5% | 58.7% | 8.8% |
| Maputaland coastal forests and woodlands | 211,120 | 38.5% | 53.2% | 8.4% |
| Central bushveld | 3,405,923 | 15.1% | 72.6% | 12.3% |
| Drakensberg Escarpment savanna and thicket | 2,590,966 | 27.1% | 61.0% | 12.0% |
| Drakensberg grasslands | 4,473,419 | 13.1% | 75.4% | 11.5% |
| Limpopo lowveld | 1,553,313 | 27.1% | 58.0% | 15.0% |
| Zambezian mopane woodlands | 1,484 | 1.5% | 97.2% | 1.2% |
| Highveld grasslands | 8,133,594 | 8.5% | 84.9% | 6.6% |
| Albany thickets | 664,078 | 28.4% | 64.5% | 7.1% |
| Fynbos shrubland | 1,485,903 | 15.0% | 65.4% | 19.6% |
| Renosterveld shrubland | 818,745 | 13.1% | 73.1% | 13.9% |
| Gariep Karoo | 225,941 | 14.2% | 69.5% | 16.4% |
| Kalahari xeric savanna | 1,847,115 | 5.0% | 89.2% | 5.8% |
| Nama Karoo shrublands | 1,028,055 | 4.2% | 91.6% | 4.2% |
| Namaqualand-Richtersveld steppe | 590,821 | 1.9% | 93.7% | 4.4% |
| Succulent Karoo xeric shrublands | 649,222 | 4.3% | 84.6% | 11.1% |
| Southern Africa mangroves | 33,172 | 33.7% | 58.5% | 7.8% |

### Combined recovery by ecoregion

| Ecoregion | n | Both recovering (%) | Any recovering (%) | Both degrading (%) |
|---|---:|---:|---:|---:|
| Knysna-Amatole montane forests | 43,321 | 25.0% | 41.0% | 7.1% |
| Kwazulu Natal-Cape coastal forests | 555,848 | 26.3% | 48.7% | 2.9% |
| Maputaland coastal forests and woodlands | 211,120 | 19.5% | 42.5% | 4.9% |
| Central bushveld | 3,405,923 | 6.8% | 18.2% | 5.8% |
| Drakensberg Escarpment savanna and thicket | 2,590,966 | 15.5% | 33.0% | 7.0% |
| Drakensberg grasslands | 4,473,419 | 7.4% | 19.8% | 6.0% |
| Limpopo lowveld | 1,553,313 | 13.9% | 30.3% | 8.5% |
| Zambezian mopane woodlands | 1,484 | 0.7% | 3.7% | 0.1% |
| Highveld grasslands | 8,133,594 | 2.8% | 12.0% | 3.4% |
| Albany thickets | 664,078 | 14.3% | 32.3% | 4.3% |
| Fynbos shrubland | 1,485,903 | 12.3% | 27.0% | 11.3% |
| Renosterveld shrubland | 818,745 | 10.6% | 28.3% | 6.8% |
| Gariep Karoo | 225,941 | 6.1% | 18.6% | 11.0% |
| Kalahari xeric savanna | 1,847,115 | 1.5% | 6.9% | 0.8% |
| Nama Karoo shrublands | 1,028,055 | 1.2% | 6.3% | 2.5% |
| Namaqualand-Richtersveld steppe | 590,821 | 1.5% | 19.0% | 1.4% |
| Succulent Karoo xeric shrublands | 649,222 | 3.3% | 14.8% | 7.8% |
| Southern Africa mangroves | 33,172 | 29.4% | 52.3% | 3.6% |

## 6. Slope Statistics

Mean Theil-Sen slopes by GPP trajectory class.

| GPP Class | GPP slope (mean) | SVH slope (mean) |
|---|---:|---:|
| Recovery | 0.03673 | 0.03477 |
| Stable | -0.00094 | 0.00159 |
| Degradation | -0.03461 | -0.01793 |

---

*Analysis limited to pixels with natural/near-natural or secondary natural land cover (SANLC 2022 classes 1-2). Trends computed on per-ecoregion Z-score standardized values. Classification: significant positive slope = Recovery, significant negative slope = Degradation, non-significant = Stable (alpha = 0.05).*
