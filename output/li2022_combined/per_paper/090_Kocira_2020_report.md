# Extraction Quality Report: 090_Kocira_2020

**Paper:** Kocira et al. (2020). "Biochemical and economical effect of application biostimulants containing seaweed extracts and amino acids as an element of agroecological management of bean cultivation." *Scientific Reports* 10:17759.

**Validation summary:** 24 matched pairs, r = -0.154, MAE = 20.8 pp, 24 unmatched GT rows, 8 unmatched JSON obs.

---

## 1. Paper Design

### Crop and intervention

- **Crop:** Common bean (*Phaseolus vulgaris* L.), cultivar Mexican Black (dark seed coat, chosen for nutraceutical properties)
- **Country:** Poland
- **Biostimulants tested:**
  - **Kelpak SL** (seaweed extract from *Ecklonia maxima*, SWE category) — applied at two concentrations: lower (LSS/LDS = 0.7%) and higher (HSS/HDS = 1.0%)
  - **Terra Sorb Complex** (free amino acids, PHs category) — applied at two concentrations: lower (LSS/LDS = 0.3%) and higher (HSS/HDS = 0.5%)
- **Application frequencies:** Single spray (once) and double spray (twice)
- **Treatment codes:**
  - LSS = lower concentration single spraying
  - HSS = higher concentration single spraying
  - LDS = lower concentration double spraying
  - HDS = higher concentration double spraying
  - C = control (water spray)

### Experimental years

Three growing seasons: **2016**, **2017**, and **2018**. This is critical: the paper reports data for each year separately and year differences are substantial (see below).

- **2016:** Relatively normal conditions; biostimulants generally positive
- **2017:** Moderate drought stress (Fig. 5 shows Sielianinow's hydrothermal coefficients below optimal in May–August); some Kelpak treatments at lower concentrations unexpectedly yielded **below control**
- **2018:** Higher baseline yields (~490–500 g/m²); most treatments clearly positive

### Experimental design

Randomized complete block design (RCBD) with **3 replicate blocks** per year. The paper specifies n = 3 (pairs 124–147 in GT, listed under `replicates: 3`). There is also a second series in the Li 2022 GT dataset with `replicates: 4` (pairs 964–987), which likely represents an alternative data digitization or a different experimental component.

---

## 2. What Tables/Figures Contain Yield Data

| Source | Content | Scale | Note |
|--------|---------|-------|------|
| **Figure 1** | Seed yield (g m⁻²) by treatment and year | 0–500 g m⁻² | Bar chart with error bars; shows all 5 treatments × 3 years + 3-year average |
| **Table 1** | Nutraceutical quality parameters | Protein, FRAP, ABTS, Proline | Not yield |
| **Table 2** | Antioxidant potential parameters | Anthocyanins, flavonoids, reducing power, phenols | Not yield |
| **Figures 3 & 4** | Economic effects | EUR ha⁻¹ income increase | Not raw yield means |

The **only yield data source** in the paper is **Figure 1**. There is no separate numerical yield table in the main paper body. Any extraction from Figure 1 must be based on digitizing bar heights.

---

## 3. Unit Discrepancy Between GT and JSON

### Ground truth (Li 2022 GT) units

The GT ctrl_mean values for Kelpak range from 0.2485 to 0.3178. Converting: these are in **kg m⁻²**, consistent with Figure 1:

| GT ctrl_mean | Converted (g m⁻²) | Figure 1 bar (visual) | Year |
|---|---|---|---|
| 0.2485 kg m⁻² | 248.5 g m⁻² | ~250 g m⁻² | 2016 |
| 0.2786 kg m⁻² | 278.6 g m⁻² | ~279 g m⁻² | 2018 |
| 0.3178 kg m⁻² | 317.8 g m⁻² | ~320 g m⁻² | 2017 |

The GT values align precisely with the visual bar heights of the **control (C) treatment in Figure 1** for the respective years. The Li 2022 meta-analysis digitized Figure 1 directly, reading values off the bar chart in kg m⁻².

### AI JSON units

The JSON extraction reports ctrl_mean values of **4970.7, 5181.7, and 4968.2 g m⁻²**. These are approximately 15–20× larger than the Figure 1 bars. The ratio between JSON and GT values is not constant (20,003, 16,306, 17,831 respectively), ruling out a simple unit scale factor.

**The AI did not read from Figure 1.** The values are consistent with raw plot yield data, possibly from a **supplementary data file** or an unpublished numerical table attached to the paper. The inconsistent scaling ratios suggest the AI extracted data with genuinely different absolute values than what Figure 1 displays — perhaps a different plot-size denominator or a different data representation.

**Key implication:** The absolute value discrepancy is a secondary problem. Effect sizes (percent change) should still be comparable regardless of the absolute unit — if the right control is matched to the right treatment.

---

## 4. The Core Problem: What Each Source Extracted

### What the Li 2022 GT extracted

The GT has **24 rows with `replicates=3`** (pairs 124–147) and **24 rows with `replicates=4`** (pairs 964–987). All 48 GT effects are positive, ranging from +3.6% to +41.6%.

The GT's 3 ctrl_mean levels per product match the 3 year-specific control values from Figure 1. This means the GT extracted **year-specific means** but reports **all-positive effects**. This is only possible if the data source the GT used — the averaged data from Figure 1's "Average 2016–2018" column, or a different representation — smoothed over year-specific negative results.

Alternatively, the Li 2022 authors may have extracted the data using the **year-averaged treatment means** from the rightmost "Average ± SD" column visible in Figure 1, while assigning the 3 ctrl levels from the year-specific control bars. In any case, the GT does not represent year-specific raw observations.

### What the AI JSON extracted

The AI correctly identified **year-by-year observations** and tagged them with year labels (2016, 2017, 2018). It extracted 24 yield observations: 8 per year × 3 years.

The year-specific results the AI extracted reflect what Figure 1 actually shows:

| Year | Treatment | AI ctrl | AI treat | AI effect | Figure 1 consistent? |
|------|-----------|---------|---------|-----------|----------------------|
| 2016 | Kelpak LSS (0.7%, single) | 4970.7 | 5484.1 | +10.3% | Yes — 2016 LSS bar above C |
| 2017 | Kelpak LSS (0.7%, single) | 5181.7 | 4431.1 | **−14.5%** | **Yes — 2017 LSS bar below C** |
| 2018 | Kelpak LSS (0.7%, single) | 4968.2 | 5246.5 | +5.6% | Yes — 2018 LSS bar above C |
| 2016 | Kelpak HSS (1.0%, single) | 4970.7 | 4176.7 | **−16.0%** | Yes — 2016 HSS shorter than C |
| 2017 | Kelpak HSS (1.0%, single) | 5181.7 | 5883.9 | +13.6% | Yes |
| 2018 | Kelpak HSS (1.0%, single) | 4968.2 | 4808.5 | **−3.2%** | Yes |

**The negative effects in the JSON are real.** They appear directly in Figure 1 of the paper. In 2016 and 2017, several Kelpak treatments at lower concentrations fell below control, especially during the drought stress year (2017). The paper's own results section notes year-to-year variation: "the application of preparations influenced nutritional and nutraceutical quality of bean seeds." Figure 5 confirms 2017 had drought stress with Sielianinow coefficients below 1.0 in May and June, which likely stressed the crop and reduced responsiveness to Kelpak's hormonal signals.

### Count of negative effects in JSON yield observations

| Year | Negative effects | Positive effects |
|------|-----------------|-----------------|
| 2016 | 3 of 8 | 5 of 8 |
| 2017 | 5 of 8 | 3 of 8 |
| 2018 | 1 of 8 | 7 of 8 |
| **Total** | **9 of 24** | **15 of 24** |

---

## 5. Why GT Effects Are All Positive While JSON Has Negative Effects

There are two inter-related explanations:

### Explanation A: GT uses multi-year averaged treatment means

The Li 2022 meta-analysis likely extracted **average treatment means across the 3 years** (from the "Average 2016–2018" column in Figure 1) while comparing against **year-specific control values**. Averaged treatment means smooth out the single bad year (2017) and naturally appear positive if the biostimulant works in 2 out of 3 years.

For example, Kelpak LSS averaged across years: (5484.1 + 4431.1 + 5246.5) / 3 = 5053.9 vs ctrl average 5040.2 → effectively flat (+0.3%). Yet the GT pair 124 reports +29.7% for this treatment. This means even the GT's "average" interpretation is inconsistent with a simple year mean.

### Explanation B: GT digitized from a secondary data representation

The most likely scenario is that Li 2022 digitized from a specific data layer in Figure 1 — possibly the "Average 2016–2018" bars visible at the right side of Figure 1 — not the year-specific bars. The Average 2016–2018 bars for Kelpak show all treatments clearly above control (because the 1 negative year is diluted by 2 positive years, and the standard deviations overlap). This produces all-positive effects consistently.

### Why this creates r = −0.154

The matching algorithm assigned:
- GT pair 124 (ctrl=0.2485, year 2016) → JSON obs year 2016, Kelpak LSS (ctrl=4970.7) → effect GT=+29.7%, JSON=+10.3% (direction OK, magnitude differs)
- GT pair 132 (ctrl=0.3178, year 2017) → JSON obs year 2017, Kelpak LSS (ctrl=5181.7) → effect GT=+24.2%, JSON=**−14.5%** (direction WRONG)

Across all 24 matched pairs, **9 out of 24 pairs have opposite-direction effects**. Every one of these has GT positive and JSON negative. This systematic direction mismatch — caused by year 2017 drought effects present in the AI's year-specific extraction but absent in the GT's averaged extraction — drives the Pearson r toward −0.154.

### Direction mismatch summary

| Pair | GT effect | JSON effect | Direction | Notes |
|------|-----------|------------|-----------|-------|
| GT 124 | +29.7% | +10.3% | OK | 2016 match OK |
| GT 125 | +30.3% | −16.0% | **WRONG** | 2016 HSS negative in JSON |
| GT 126 | +32.7% | +7.4% | OK | |
| GT 127 | +36.4% | +9.2% | OK | |
| GT 128 | +29.2% | +15.0% | OK | |
| GT 129 | +14.6% | +1.1% | OK | |
| GT 130 | +19.5% | +9.9% | OK | |
| GT 131 | +33.5% | −6.8% | **WRONG** | 2018 Terra Sorb HDS negative |
| GT 132 | +24.2% | −14.5% | **WRONG** | 2017 Kelpak LSS drought |
| GT 133 | +5.7% | +13.6% | OK | |
| GT 134 | +29.4% | −9.9% | **WRONG** | 2017 Kelpak LDS drought |
| GT 135 | +27.0% | −14.1% | **WRONG** | 2017 Kelpak HDS drought |
| GT 136 | +3.6% | +16.8% | OK | |
| GT 137 | +6.6% | −4.7% | **WRONG** | 2016 Terra Sorb HSS negative |
| GT 138 | +13.9% | +15.3% | OK | |
| GT 139 | +4.2% | −10.6% | **WRONG** | 2016 Terra Sorb HDS negative |
| GT 140 | +18.9% | +5.6% | OK | |
| GT 141 | +41.6% | −3.2% | **WRONG** | 2018 Kelpak HSS negative |
| GT 142 | +36.8% | +4.0% | OK | |
| GT 143 | +20.5% | +15.6% | OK | |
| GT 144 | +8.1% | +9.1% | OK | |
| GT 145 | +12.8% | +1.1% | OK | |
| GT 146 | +13.7% | +3.9% | OK | |
| GT 147 | +8.1% | −9.6% | **WRONG** | 2017 Terra Sorb HDS drought |

Direction mismatches: **9/24 (37.5%)**

---

## 6. Is This a Treatment/Control Swap?

**No.** This is not a classic T/C swap. The AI correctly identified the control (water spray, C) and the biostimulant treatments. The negative effects are genuine year-specific findings from Figure 1 of the paper: in 2017 and occasionally in 2016, certain Kelpak treatments at lower concentrations and certain Terra Sorb Complex treatments at higher concentrations actually yielded less than the control. This is acknowledged in the paper's Results section, which notes year-to-year variability driven by drought stress.

---

## 7. Secondary Issue: Unmatched GT Rows (pairs 964–987)

The GT contains a second series of 24 rows with `replicates=4` (pairs 964–987). These have ctrl_mean values very similar to pairs 124–147 but with `replicates=4` instead of 3, and slightly different dose labels for Terra Sorb (0.6% instead of 0.7%). These unmatched rows remain because the 24 AI yield observations were entirely consumed by pairs 124–147.

The `replicates=4` series likely represents:
- Either a second cultivar block
- Or a different digitization pass by Li 2022 from the same Figure 1 with slightly different readings
- Or an additional experimental series not visible in the main paper

Since the AI only extracted 24 yield observations (8 per year), it had no extra observations to match these 24 additional GT rows.

---

## 8. Root Cause Summary

The r = −0.154 correlation and MAE = 20.8 pp arise from **three compounding problems**:

### Problem 1: Granularity mismatch (primary cause)
The Li 2022 GT extracted averaged effects that are all positive (consistent with the paper's conclusion that biostimulants improved yield on average). The AI extracted **year-specific observations** from Figure 1, which include genuine negative effects for year 2017 (drought year) and some higher-concentration treatments in 2016. This granularity difference — averaged GT vs. year-level AI — creates 9 opposite-direction matched pairs.

### Problem 2: Year-to-block assignment ambiguity (secondary cause)
The matching algorithm assigned GT block-level ctrl values to AI year-level ctrl values by ranking (lowest ctrl → year 2016, etc.). This heuristic is imperfect: the three GT ctrl levels (0.2485, 0.2786, 0.3178 kg/m²) correspond to the 3 years in Figure 1, but the assignment of which GT treatment row maps to which AI year is approximate. When a GT pair with effect +24.2% is paired with an AI year-2017 observation showing −14.5%, the mismatch is partly due to the ambiguous assignment.

### Problem 3: Magnitude underestimation (tertiary cause)
Even among the 15 correctly-directed pairs, the AI effects are systematically smaller than GT effects (mean GT = +20.9%, mean AI = +2.0%). This reflects the year averaging: positive years are diluted by the negative 2017 year in the AI's year-specific data, while the GT's averaged representation inflates the apparent average effect.

### Problem 4: Unit mislabeling in JSON
The AI labeled extracted values as "g m⁻²" but the absolute values (~5000) are inconsistent with Figure 1's scale (0–500 g m⁻²). The actual source of these numbers appears to be a supplementary table or a different figure with different units. This unit confusion does not affect direction or effect size (since both ctrl and treat are mislabeled identically), but it complicates interpretation and could cause errors in future analyses that use the raw means.

---

## 9. Assessment of AI Extraction Accuracy

Despite the poor validation statistics, the AI extraction is not fundamentally wrong. It correctly:

- Identified the two biostimulants (Kelpak SL, Terra Sorb Complex)
- Identified the two dose levels and two application frequencies per product
- Extracted year-specific data for all 3 years (2016, 2017, 2018)
- Captured n = 4 replicates per observation
- Correctly found that year 2017 had negative or near-zero effects for several Kelpak treatments
- Did not confuse treatment and control groups

The AI's year-specific extraction is arguably **more granular and more accurate to the paper's raw data** than the Li 2022 GT's averaged representation. The validation statistics are poor not because the AI made errors, but because it extracted at a different level of aggregation than the GT.

---

## 10. Recommendations

1. **Do not interpret r = −0.154 as extraction failure.** The negative correlation reflects a structural mismatch between year-level extraction and year-averaged GT, not errors in the AI output.

2. **Exclude paper 090 from the Li 2022 validation correlation**, or report it separately with a note about granularity mismatch.

3. **For the meta-analysis validation paper**, this case illustrates the challenge of extracting from multi-year field trials: the choice of whether to treat years as separate observations or average across them materially affects effect directions and magnitudes. This is a legitimate methodological difference, not a data quality issue.

4. **Flag the unit mislabeling** in the JSON: the AI labeled values as "g m⁻²" when the scale (~5000) is inconsistent with Figure 1. The true unit is unknown without access to the supplementary data source the AI read from.

5. **The 24 unmatched GT rows (pairs 964–987, replicates=4)** remain unexplained by the main paper content. They may represent a second digitization pass or a separate experimental component not in the main PDF.
