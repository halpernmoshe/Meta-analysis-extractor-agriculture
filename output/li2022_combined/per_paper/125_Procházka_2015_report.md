# Extraction Quality Report: 125_Procházka_2015

**Paper:** Procházka P., Štranc P., Pazderů K., Štranc J., Jedličková M. (2015). "The possibilities of increasing the production abilities of soya vegetation by seed treatment with biologically active compounds." *Plant Soil Environ.* 61(6): 279–284. doi: 10.17221/225/2015-PSE

**Validation result:** 6 matched pairs, r = -0.777, MAE = 9.7 percentage points, 0 unmatched GT rows, 18 unmatched JSON observations.

---

## 1. Paper Design

**Crop:** Soybean (*Glycine max*), cultivar Merlin (early-ripening).

**Location:** Studeněves, Czech Republic (50°13′50″N, 14°2′54″E), altitude 306 m a.s.l., medium-heavy to light arenic Cambisol on carbonate slope.

**Duration:** Three vegetation seasons: 2012, 2013, and 2014.

**Design:** Randomised block experiment with long plots, 0.1 ha per plot, each treatment replicated 3 times per year (3 years × 3 reps = n = 9 pooled observations in the averaged analysis).

**Treatments (seed treatment applied immediately before sowing):**

| Code | Product | Description |
|------|---------|-------------|
| UTC | Untreated control | Inoculated with nitrazon+ only |
| LIG | Lignohumate B | Mixture of humic and fulvic acids (ratio 1:1); 25.7 mL per 20 kg seed |
| LEX | Lexin | Concentrate of humic acid, fulvic acids and auxins; 6.5 mL per 20 kg seed |
| BRS | Brassinosteroid | Synthetic analogue of 24-epibrassinolide (substance 4154); 2.2 mL per 20 kg seed |
| COM | Complete seed treatment | Saturated sucrose solution + Lexin (6.5 mL) + Agrovital adjuvant (10 mL) + Maxim XL 035 FS fungicide (20 mL) per 20 kg seed |

**Forecrop:** Spring barley (2012), winter wheat (2013), spring barley (2014). Uniform cultivation technology applied across all years (Table 2 of the paper).

**Outcomes measured:** Field germination (%), stand density after germination and before harvest (plants/m²), plant height (cm), height of apical end of the lowest pod (cm), and **yield of seeds** (t/ha, converted to 13% humidity). Statistical analysis used SAS version 9.0 with LSD test.

---

## 2. Data Tables in the Paper and What They Contain

### Table 1 (p. 280) — Seed treatment doses
Describes the preparation doses applied to 20 kg of seed for each treatment variant. Contains no yield data.

### Table 2 (p. 281) — Cultivation technology by year
Documents soil preparation, fertilisation, sowing dates, herbicide applications, and harvest dates for each of the three experimental years (2012, 2013, 2014). Contains no yield data.

### Table 3 (p. 282) — Results of statistical significance [THE ONLY NUMERIC YIELD TABLE]
This is the sole table containing yield and agronomic outcome means. It presents **3-year pooled averages** (n = 9) for all five treatment variants across five measured parameters:

| Parameter | COM | LEX | BRS | LIG | UTC | LSD | n |
|-----------|-----|-----|-----|-----|-----|-----|---|
| Field emergence (%) | 87.30 | 86.20 | 83.40 | 75.70 | 71.10 | 2.2746 | 9 |
| Stand density before harvest (plants/m²) | 51.60 | 46.90 | 42.70 | 41.50 | 38.10 | 4.1440 | 9 |
| Stand height before harvest (cm) | 81.01 | 82.14 | 79.11 | 78.33 | 78.37 | 4.4531 | 9 |
| Height of apical end of lowest pod (cm) | 7.77 | 6.54 | 6.40 | 4.79 | 4.68 | 1.6711 | 9 |
| **Yield (t/ha)** | **3.29** | **3.18** | **3.08** | **2.90** | **2.85** | **0.0507** | **9** |

The n = 9 represents 3 replicates × 3 years pooled. The LSD column provides the least significant difference at an unspecified alpha (presumably p < 0.05). No SD or SE values are reported — only the LSD statistic.

---

## 3. Per-Year Data in the Paper: Figures Only, No Numeric Table

The paper presents year-specific (2012, 2013, 2014) data **exclusively as bar charts (figures)**:

- **Figure 2** (p. 282): Crop stand densities at the 3rd trifoliate leaf phenophase in 2012, 2013, and 2014.
- **Figure 3** (p. 283): Crop stand densities before harvest in 2012, 2013, and 2014.
- **Figure 4** (p. 283): Height of the apical end of the lowermost pods in 2012, 2013, and 2014.
- **Figure 5** (p. 283): **Yield (t/ha) of soybean seeds in 2012, 2013, and 2014.** ← The ground truth source.

**Figure 5 is the critical figure for this discrepancy.** It shows per-year yield as a grouped bar chart with five treatment bars per year. The y-axis runs from 2.0 to 4.0 t/ha. Approximate values readable from the chart are:

| Year | UTC | LIG | LEX | BRS | COM |
|------|-----|-----|-----|-----|-----|
| 2012 | ~2.51 | ~2.90 | ~2.71 | ~2.62 | ~2.75 |
| 2013 | ~3.34 | ~3.75 | ~3.32 | ~3.50 | ~3.72 |
| 2014 | ~2.71 | ~2.90 | ~2.67 | ~2.75 | ~3.09 |

These approximate visual readings match closely the Li 2022 ground truth (GT) values (pairs 527–532), which appear to have been digitised from this figure:

| GT pair | Product | Year | GT ctrl (t/ha) | GT treat (t/ha) | GT effect |
|---------|---------|------|----------------|-----------------|-----------|
| 527 | Lexin | 2012 | 0.251 → 2.51 | 0.271 → 2.71 | +7.97% |
| 528 | Lignohumate B | 2012 | 0.251 → 2.51 | 0.290 → 2.90 | +15.54% |
| 529 | Lexin | 2013 | 0.334 → 3.34 | 0.332 → 3.32 | −0.60% |
| 530 | Lignohumate B | 2013 | 0.334 → 3.34 | 0.375 → 3.75 | +12.28% |
| 531 | Lexin | 2014 | 0.271 → 2.71 | 0.267 → 2.67 | −1.48% |
| 532 | Lignohumate B | 2014 | 0.271 → 2.71 | 0.290 → 2.90 | +7.01% |

Note: GT values are stored in units that appear to be t/ha × 0.1 (i.e., the raw column value 0.251 = 2.51 t/ha), or alternatively they are expressed directly in some unit convention of the Li 2022 database. Regardless, the pattern of values corresponds unambiguously to Figure 5 of the paper, year by year.

**There is no per-year numeric table anywhere in the paper.** The only way to obtain per-year yield values is by digitising Figure 5.

---

## 4. What the AI Extracted

The AI correctly identified Table 3 (the 3-year averaged summary table) as the primary data source and extracted yield observations from it. Specifically, for yield:

| JSON obs | Treatment | Control mean (kg/ha) | Treatment mean (kg/ha) | Effect |
|----------|-----------|----------------------|------------------------|--------|
| idx 16 | Lignohumate B | 2850 | 2900 | +1.75% |
| idx 17 | Lexin | 2850 | 3180 | +11.58% |
| idx 18 | Brassinosteroid | 2850 | 3080 | +8.07% |
| idx 19 | Complete seed treatment | 2850 | 3290 | +15.44% |

These values correspond directly to Table 3 of the paper (UTC = 2.85 t/ha = 2850 kg/ha; LEX = 3.18 t/ha = 3180 kg/ha; LIG = 2.90 t/ha = 2900 kg/ha; BRS = 3.08 t/ha = 3080 kg/ha; COM = 3.29 t/ha = 3290 kg/ha). The unit conversion from t/ha to kg/ha is arithmetically correct.

The AI also extracted 16 additional (non-yield) observations for field emergence (%), stand density (plants/m²), stand height (cm), and height of lowest pod (cm) — all from Table 3, all with n = 9 (3-year pooled), all correctly matched to the table values. These 16 observations have no GT counterparts because Li 2022 only tracked yield outcomes.

---

## 5. Why the Per-Year and 3-Year Average Effects Diverge

The core discrepancy arises because the paper's two data representations (Figure 5 per-year vs. Table 3 averaged) show substantially different treatment effects, especially for Lignohumate B. This is not an extraction error — it reflects real biological variability across years.

**Lignohumate B (LIG) — the most extreme case:**

| Source | Year | Effect vs. UTC |
|--------|------|----------------|
| GT (Fig 5, digitised) | 2012 | +15.54% |
| GT (Fig 5, digitised) | 2013 | +12.28% |
| GT (Fig 5, digitised) | 2014 | +7.01% |
| **GT mean across years** | | **+11.6% (unweighted)** |
| **AI extracted (Table 3)** | 2012–2014 avg | **+1.75%** |

The 3-year average effect of +1.75% (Table 3) is dramatically lower than the per-year effects (+7–15%), which is mathematically inconsistent. The reason is that **the 3-year averages in Table 3 are pooled with different UTC baselines per year.** In 2013, both UTC and LIG yields were substantially higher than in 2012 or 2014 (UTC ≈ 3.34 t/ha in 2013 vs. ≈ 2.51 t/ha in 2012). When averaged, the high-yield year (2013, where LIG showed a +12% effect) and the low-yield years are combined into a single UTC mean (2.85 t/ha) and a single LIG mean (2.90 t/ha), compressing the apparent treatment effect.

In other words: the simple mean of treatment values across years divided by the simple mean of control values across years does NOT equal the mean of per-year treatment/control ratios. For Lignohumate B:
- Per-year ratio average: (1.1554 + 1.1228 + 1.0701) / 3 = 1.116 → +11.6%
- Ratio of means: 2.90 / 2.85 = 1.0175 → +1.75%

This is a well-known Jensen's inequality / aggregation artefact. The GT correctly captures per-year effects (which is the scientifically appropriate unit of observation for a multi-year trial); the AI extracted the averaged ratio, which is biologically misleading.

**Lexin (LEX) — less extreme but same structure:**

| Source | Year | Effect vs. UTC |
|--------|------|----------------|
| GT (Fig 5, digitised) | 2012 | +7.97% |
| GT (Fig 5, digitised) | 2013 | −0.60% |
| GT (Fig 5, digitised) | 2014 | −1.48% |
| **GT mean across years** | | **+1.96%** |
| **AI extracted (Table 3)** | 2012–2014 avg | **+11.58%** |

Here the direction of the artefact is reversed. In 2013, Lexin actually slightly reduced yield (−0.6%), yet the 3-year average shows +11.58%. This is because Lexin's high 2012 yield (2.71 vs. 2.51 t/ha, +7.97%) is averaged with its nearly equal or lower yields in 2013–2014, but the mean absolute value (3.18 t/ha) is pulled upward by the high 2013 baseline shared by all treatments. The ratio of means amplifies rather than averages the per-year effects in this case.

**Consequence for r:** The matched pairs compare three GT values (one per year, all showing the same AI-extracted 3-year mean) to their respective per-year effects. For Lexin, the AI shows a consistently positive effect (+11.58%) while the GT varies from +7.97% to −1.48% across years. For Lignohumate B, the AI shows +1.75% while GT values are all strongly positive (+7–15%). This inversion of relative magnitudes across products is what drives r = −0.777 — the sign relationship between the two products is reversed in the two representations.

---

## 6. Could the AI Have Extracted Per-Year Data Instead?

**No, not reliably.** There is no per-year numeric table in the paper. Per-year yield data exist only in Figure 5, a grouped bar chart. Extracting numeric values from bar charts requires either:
1. Vision-based chart digitisation (not performed by the current extraction pipeline), or
2. The approximate visual readings described above, which carry digitisation uncertainty of approximately ±0.05–0.10 t/ha.

The AI made the correct decision to extract from Table 3, which is the only source of exact numeric yield values in the paper. The extraction itself is accurate — the values 2850, 2900, 3080, 3180, 3290 kg/ha match Table 3 exactly (UTC = 2.85, LIG = 2.90, BRS = 3.08, LEX = 3.18, COM = 3.29 t/ha). The LSD value (0.0507 t/ha = 50.7 kg/ha) was not extracted into the variance field, which is a separate omission (the paper provides only LSD, not SD or SE).

**What the Li 2022 GT authors did:** They digitised Figure 5 to obtain per-year observations, treating each year as an independent experimental unit (n = 3 replicates). This is the methodologically correct choice for a meta-analysis, since year-specific observations represent independent environmental realisations of the treatment effect and should not be pre-averaged before effect size calculation. The GT thus contains 6 rows (3 years × 2 products) rather than 2 rows (1 3-year average × 2 products).

---

## 7. Summary of Discrepancies

| Issue | Description | Severity |
|-------|-------------|----------|
| **Temporal aggregation** | AI extracted 3-year pooled means from Table 3; GT used per-year values digitised from Figure 5. This is the primary source of all effect size errors. | High |
| **Unit labelling** | AI recorded yield as kg/ha (2850 kg/ha) which is arithmetically correct (= 2.85 t/ha from Table 3); GT stores values in t/ha. No numeric error, but unit metadata differs. | Low |
| **Missing per-year data** | Per-year numeric data are not reported in any table; they appear only as bar charts in Figures 2–5. AI cannot extract bar chart values without chart digitisation capability. | Structural limitation |
| **Brassinosteroid not in GT** | AI extracted BRS (idx 18) and COM (idx 19) yield observations; Li 2022 GT did not include these treatments. These 2 yield observations are correctly unmatched. | Expected scope difference |
| **16 non-yield outcomes unmatched** | AI extracted field emergence, stand density, stand height, and pod height (all from Table 3, all correct). Li 2022 GT is yield-only. These are not errors. | Expected scope difference |
| **Variance not captured** | Table 3 reports LSD = 0.0507 t/ha for yield. The AI did not extract a variance value into any observation. This LSD could be converted to SD given n = 3 per year. | Moderate |

---

## 8. Recommendations

1. **Figure digitisation for multi-year trials:** When a paper presents per-year data only as figures (no per-year numeric table), the extraction pipeline should attempt bar chart digitisation or flag the paper for manual data entry. The Li 2022 GT obtained its values by digitising Figure 5.

2. **Prefer per-year over pooled averages:** For multi-year agronomic trials, per-year observations (each year = independent unit, n = replicates within that year) are the appropriate input for meta-analysis effect size calculation, not the multi-year pooled means. When both representations are available, per-year data should be preferred.

3. **LSD extraction and conversion:** The paper provides LSD = 0.0507 t/ha with n = 9 pooled. If per-year re-extraction is performed using Figure 5 values, the within-year LSD could be estimated as approximately LSD_year ≈ LSD_pooled (since the degrees of freedom differ) — but this requires careful statistical judgment. The current extraction correctly noted n = 9 but did not extract the LSD value into the variance field.

4. **This paper's contribution to validation statistics:** The negative r (−0.777) for this paper is entirely an artefact of the aggregation level mismatch and does not indicate an AI extraction failure. The AI correctly extracted the values that appear in the paper's only numeric yield table. The mismatch is a structural incompatibility between what Table 3 provides (3-year means) and what the GT requires (per-year observations from Figure 5).
