# Assessing Ecological Recovery on Abandoned Agricultural Land: An Executive Summary

## Overview

When agricultural land is abandoned, natural vegetation often begins to regrow. However, true ecological recovery goes beyond just becoming "green" again—it requires the ecosystem to regain its natural structure, productivity, and composition. 

To determine the extent of this recovery across South Africa, we analyzed **~33 million abandoned agricultural pixels** (30 m resolution) using a 23-year archive of satellite data (2000–2022). For pixels showing signs of a positive trajectory, we assessed their recovery degree across 18 distinct ecoregions by comparing them to known, intact natural areas. To ensure these natural benchmarks truly represented the complex, heterogeneous landscapes of each ecoregion, we employed **Feature Space Coverage Sampling (FSCS)**. This approach generated exactly 1,849,707 sample points, of which **1,004,898** were definitively classified as intact natural reference pixels against which recovering lands could be scored.

---

## 1. Where is Recovery Happening?

Across the country, we identified nearly **2.88 million pixels** that show statistically significant signs of recovery. However, the distribution of this recovery is highly concentrated.

The **Highveld Grasslands** alone account for nearly a third of all recovering land, followed by the Drakensberg grasslands and the KwaZulu-Cape coastal forest mosaic.

![Recovering Pixels by Ecoregion](../plots/narrative/pixels_by_ecoregion.png)

---

## 2. How Complete is the Recovery?

We scored the degree of recovery on a scale from **0 to 100**, where **50 represents the typical (median) natural condition**. The score combines metrics for functional productivity (Gross Primary Productivity), structural height (Short Vegetation Height), and landscape fingerprint similarity (AlphaEarth embeddings).

### The "Green but Different" Phenomenon
A major finding across the country is that **functional recovery outpaces compositional recovery**. In grassland and savanna ecoregions, abandoned lands often become highly productive, regaining or even exceeding natural biomass levels. However, their physical structure and multi-sensor satellite "fingerprint" remain distinct from pristine natural landscapes. 

They are effectively "green," but they do not yet look or act completely like mature native ecosystems, often dominated by pioneer or secondary plant species. In contrast, forest-mosaic and bushveld ecoregions show a more balanced recovery, inching closer to their natural benchmarks.

![Recovery Scores by Ecoregion](../plots/narrative/recovery_scores_by_ecoregion.png)

*Ecoregions like Namaqualand-Richtersveld and the Maputaland-Pondoland bushland show the highest composite recovery scores, surpassing the natural median, while Fynbos and Karoo environments struggle to recover passively.*

---

## 3. The Threat of Invasive Alien Plants

A critical barrier to genuine native recovery is the spread of alien vegetation. Our analysis reveals that **nearly 1 in 4 recovering pixels (22.9%) are invaded by alien plant species**. 

In some ecoregions, what appears to be rapid "recovery" of vegetation biomass is actually the aggressive expansion of invasives. For example, over 50% of recovering pixels in the Eastern Zimbabwe montane forest-grassland mosaic, and nearly half in the Lowland fynbos, are heavily invaded.

![Invasive Plants by Ecoregion](../plots/narrative/invasive_plants_by_ecoregion.png)

### Does Invasion Impact Recovery Quality?

When we map median recovery scores against the percentage of invaded pixels, a concerning trend emerges. Ecoregions with the lowest overall recovery scores (such as Montane fynbos and Nama Karoo) tend to face unique ecological barriers. Invasions complicate the picture: highly invaded regions often exhibit altered growth patterns, artificially inflating biomass while displacing native biodiversity. 

![Recovery vs. Invasion](../plots/narrative/recovery_vs_invasive.png)

---

## Conclusion

The data tells a clear story: **abandoned agricultural land in South Africa retains significant potential to recover its carbon-fixing capacity.** However, passive recovery (simply leaving the land alone) is often insufficient to restore full ecological composition. 

- **Grasslands and Savannas** recover productivity rapidly but may settle into alternate, structurally distinct secondary states.
- **Fynbos and Karoo** ecosystems face substantial barriers, requiring active restoration rather than passive abandonment.
- **Invasive alien plants** represent a massive roadblock, hijacking the recovery process on nearly a quarter of all abandoned lands.

Targeted interventions—such as invasive clearing and active native seeding—will be vital to push these lands past merely being "green" and toward full ecosystem restoration.
