# Assessing Ecological Condition and Recovery of Abandoned Agricultural Lands: A Scalable Framework for the 30x30 Target

## Abstract
Agricultural abandonment offers significant opportunities for ecological restoration and carbon sequestration. However, assessing whether abandoned lands are genuinely recovering into diverse, resilient ecosystems remains a major challenge. We present a scalable framework to operationalise criteria for assessing ecosystem condition and recovery degrees across 33 million pixels of abandoned agriculture in South Africa (2000–2022). By integrating functional indicators (Gross Primary Productivity, GPP) and structural indicators (Short Vegetation Height, SVH) with multi-modal AlphaEarth embeddings, we differentiate basic greening from true convergence towards natural reference states. Results reveal that while functional recovery (productivity) is common, landscape and compositional convergence significantly lags behind. In the Highveld Grasslands, recovering pixels outperform 90% of natural references in productivity, but their landscape similarity (AlphaEarth embeddings) remains in the lowest quartile. This framework directly addresses the Kunming-Montreal Global Biodiversity Framework’s 30x30 targets by providing empirical criteria to evaluate restoration progress (Target 2) and identifying ecologically recovered abandoned lands as candidates for formal protection (Target 3).

## 1. Introduction
Agricultural land abandonment is occurring at unprecedented scales globally, driven by socio-economic shifts, climatic variability, and soil degradation. These abandoned lands present a vast, untapped opportunity for large-scale ecological recovery and nature-based climate solutions.

However, monitoring the progress of these unmanaged landscapes poses a critical challenge for ecologists and policymakers. The definition of "recovery" is multidimensional—involving functional, structural, and compositional pathways. A recovering landscape might rapidly regain its capacity to fix carbon (functional recovery) without returning to the complex, diverse species composition typical of its natural climax state (compositional recovery).

### Criteria for Assessing Condition and Recovery
Historically, large-scale remote sensing approaches have relied heavily on productivity metrics (e.g., NDVI, GPP) to assess landscape condition. While useful for quantifying broad trends, such singular metrics are insufficient for capturing the true "ecological condition" of a site. A field overrun by highly productive, invasive pioneer species may appear fully recovered from a functional standpoint, yet possess poor biodiversity, simplified physical structure, and low resilience. Comprehensive criteria for assessing ecosystem condition must instead evaluate landscapes against pristine natural reference states, incorporating both structural components (like canopy height) and multifaceted landscape signatures.

### Alignment with the 30x30 Target
The Kunming-Montreal Global Biodiversity Framework (GBF) established ambitious "30x30" targets to halt biodiversity loss. Target 2 mandates that 30% of degraded terrestrial areas are under effective restoration by 2030, while Target 3 calls for the protection of 30% of critical land areas. Realising these goals requires operational, empirically grounded methods to track restoration success at scale. Without robust multidimensional criteria for assessing condition, countries risk classifying highly degraded, monoculture-like abandoned fields as "effectively restored," undermining the integrity of the 30x30 target.

To address these challenges, we evaluated the trajectory and condition of abandoned agricultural land across South Africa spanning 23 years (2000–2022). By triangulating productivity, vegetation structure, and integrative AI-derived landscape embeddings against heavily constrained natural baselines, we present a quantitative framework for measuring true ecological convergence.

## 2. Methods

The methodology integrates time-series anomaly detection with a multidimensional similarity assessment to answer two fundamental questions: (1) Is the abandoned landscape recovering? and (2) How closely does its present condition match a natural baseline?

### 2.1 Study Area and Trajectory Classification
The study covered ~33 million (30 m resolution) pixels of abandoned agricultural land across South Africa. Time series data spanning 2000 to 2022 were sourced for Gross Primary Productivity (GPP) and Short Vegetation Height (SVH) from the Global Pasture Watch dataset.

To ensure regional comparability, we applied Z-score normalisation within spatial blocks defined by RESOLVE 2017 ecoregions. Each pixel’s trajectory was classified using non-parametric methods: 
*   **Theil-Sen slope**: to estimate the rate of change over time while resisting extreme outliers.
*   **Mann-Kendall test**: to establish the statistical significance of the derived trend (p < 0.05).
Pixels were broadly categorised as *Recovering*, *Degrading*, or *Stable* based on these trends.

### 2.2 Establishing Natural Reference Baselines
Evaluating ecosystem "condition" requires well-defined, intact reference states. We developed a composite reference mask using independent global datasets mapping natural lands (WRI SBTN), low human footprint (Global Human Modification <= 0.1), and high biodiversity intactness (BII >= 0.7). A stratified sample in South Africa yielded ~97,000 pristine reference points. Within any given ecoregion, test pixels were compared strictly to the distribution of natural pixels from that same ecoregion.

### 2.3 Criteria for Assessing Recovery Degree
To measure how far along a landscape is on the path to full recovery, we computed three complementary percentile metrics based on the average properties from the most recent five years (2018–2022). All metrics are scaled from 0-100, where 50 reflects the median typical natural condition.

1.  **Metric A (Global Ecoregion Percentile)**: Ranks the test pixel’s absolute GPP and SVH within the ecoregion’s natural distribution.
2.  **Metric B (Local Neighbours Comparison)**: Assesses structural and productivity convergence against the test pixel’s 10 closest natural neighbours (within 50 km).
3.  **Metric C (Landscape Similarity via AlphaEarth Embeddings)**: Evaluates compositional condition using AlphaEarth 64D embeddings. This model distils multi-modal Earth observation data (Sentinel-1/2, Landsat, GEDI, ERA-5, GRACE) into a comprehensive landscape signature. Similarity is computed via cosine distance at both the ecoregion scale (C_eco) and local scale (C_local).

The composite **Recovery Score** is the average of these percentile metrics.

## 3. Results

### 3.1 Regional Trajectory Analysis
Out of the ~28 million abandoned natural and secondary-natural pixels analysed:
*   **19.4%** exhibited recovery in at least one indicator (GPP or SVH).
*   **7.3%** showed simultaneous recovery in both functional and structural dimensions.
*   **63.0%** remained stable, while **5.2%** degraded in both indicators.

### 3.2 Highveld Grasslands Ecoregion: The Divergence of Metrics
A batch assessment of 317,811 recovering pixels in the Highveld Grasslands ecoregion highlighted stark divergences in recovery pathways.
*   **Functional Recovery (Productivity)**: The average GPP percentile was 90.9 (median 94.7), indicating that recovering pixels are significantly more productive than the typical natural reference landscape (median of 914 gC/m2/yr).
*   **Structural Recovery**: Vegetation height (SVH) also exceeded natural baselines, with an average percentile of 86.3.
*   **Compositional Condition (Landscape Similarity)**: Despite surpassing productivity baselines, AlphaEarth similarity scores averaged deeply in the bottom quartile (C_eco = 28.5, C_local = 17.6). 

The composite Recovery Score for the ecoregion averaged 52.1, perfectly balancing between hyper-productivity and low compositional convergence. 

### 3.3 Case Studies of Recovery Pathways
The multi-metric approach distinguishes between stages of conditional recovery that traditional indices merge:
*   **Active Recovery (Productivity Overshoot)**: Characterised by immense productivity (GPP > 40% above natural median) but low AlphaEarth similarity (30th percentile). Likely driven by fast-growing pioneer grasses lacking multi-layered structural integrity.
*   **Advanced Recovery / Encroachment**: Sites where both GPP and SVH heavily exceed local natural neighbours (SVH reaching 3.9 m, far above the ~1.2 m grassland average) and AlphaEarth embeddings align well with natural forest or thicket signatures. In grass-dominated ecoregions, this flag indicates woody encroachment or exotic plantation establishment rather than true grassland recovery.
*   **Stalled Recovery**: Pixels with GPP matching local natural averages (50th percentile) but deeply stunted structural height (2nd percentile) and anomalous landscape signatures (1st percentile), pointing to heavily altered phenology or persistent land-use legacy constraints.

## 4. Discussion

### 4.1 Refining the Criteria for Ecological Condition
Our results demonstrate that defining "condition" and "recovery" solely through the lens of carbon productivity (GPP or NDVI) will generate dangerous false positives. Within the Highveld grasslands, relying on functional metrics alone would suggest that the vast majority of abandoned fields have "recovered" to natural levels.

However, the inclusion of AlphaEarth embeddings—representing a holistic fusion of canopy texture, moisture regimes, and spectral diversity—reveals a significant condition gap. The observed pattern of functional recovery strongly outpacing compositional convergence is established in restoration ecology, yet rarely mapped accurately at scale. A landscape might quickly green-up with pioneer or invasive species, but the restoration of complex interaction networks, typical seasonal timing, and species diversity (the true marks of ecological condition) requires decades, or active intervention.

### 4.2 Contribution to the 30x30 Kunming-Montreal Target
The methodology introduced here serves as a direct operational mechanism for reporting and strategising around the Global Biodiversity Framework's 30x30 goals.

*   **Target 2 (30% Restoration)**: The metrics form an empirical checklist for what qualifies as "effectively restored." Stakeholders tracking restoration can filter out sites with mere productivity increases, focusing instead on regions with steadily improving composite Recovery Scores and advancing AlphaEarth landscape similarities. Our multidimensional anomaly scoring enables governments to track the actual *quality* and *maturity* of restoration occurring passively across vast tracts of territory.
*   **Target 3 (30% Protection)**: Expanding protected areas to 30% of the terrestrial surface requires prioritising land that supports high biodiversity and ecological intactness. By identifying regions of abandoned agricultural land that show advanced recovery (composite scores > 75 with high embedding convergence), policymakers can flag these exact parcels as highly suitable, cost-effective candidates for incorporation into formal protected area networks.

## 5. Conclusion
Scaling global restoration goals requires moving beyond one-dimensional remote sensing metrics. By assessing abandoned agriculture in South Africa using a comprehensive framework of functional, structural, and AI-embedded landscape condition criteria, we show that true ecosystem recovery is slower, and much more compositionally nuanced than productivity metrics imply. This methodology offers conservation planners an accessible, rigorous pathway to verify environmental condition and target interventions, directly enabling the successful deployment of the Kunming-Montreal 30x30 targets.
