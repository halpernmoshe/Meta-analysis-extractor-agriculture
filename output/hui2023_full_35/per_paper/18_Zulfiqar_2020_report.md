# Extraction Quality Report: 18_Zulfiqar_2020

**Paper:** Zulfiqar, U., Hussain, S., Ishfaq, M., Matloob, A., Ali, N., Ahmad, M., Alyemeni, M.N., Ahmad, P. (2020). Zinc-induced effects on productivity, zinc use efficiency, and grain biofortification of bread wheat under different tillage permutations. *Agronomy-Basel*, 10, 1566.

**Match summary:** 8/8 GT matched | r = 0.972 | MAE = 2.0%

---

## 1. Paper Design

**Species:** Bread wheat (*Triticum aestivum*)
**Country:** Pakistan
**Study type:** Field experiment (split-plot design)
**Duration:** Two cropping seasons (2017-2018, 2018-2019)
**Experimental design:** Split-plot with tillage as main-plot factor and Zn application method as sub-plot factor. Replicated three times (n = 3).

**Factorial structure:**
- Tillage: 2 levels — Plough Tillage (PT) and Zero Tillage (ZT)
- Zn application method: 7 levels — No application (control), Zn-coating, Hydro-priming, Zn-priming, Soil application, Hydro-foliar, Zn-foliar
- Year: 2 growing seasons

**Soil characteristics (Pakistan, all observations):**
- Available Zn: 0.67 mg kg-1
- pH: 7.4 (alkaline)
- Organic matter: 6.6 g kg-1
- N rate: 100 kg N ha-1; P rate: 85.6 kg P ha-1; K rate: 64.7 kg K ha-1

**Zn fertilizer:** ZnSO4 (soil: 10 kg Zn ha-1 basal; foliar: 0.5 g Zn L-1 at booting stage)

**GT study IDs in MOESM5:**
- Data 2 Soil application sheet: study_id = 18 (4 observations, obs IDs 842-845)
- Data 3 Foliar application sheet: study_id = 71 (4 observations, obs IDs 762-765)

**PDF status:** Scanned PDF with OCR. Estimated difficulty: HARD. Primary data in Figure 2 (bar charts), not numerical tables.

---

## 2. Grain Zn Data

Grain Zn concentration (mg kg-1) is reported in Figure 2 (panels a and b for the two years, c and d for straw). The paper also reports straw Zn concentration, yield, and Zn efficiency indices in Tables 3-7, but the meta-analysis extraction targets grain Zn concentration only.

**Data structure in Figure 2:**
- Figure 2a: Year 1 (2017-2018) grain Zn concentration by tillage × Zn method
- Figure 2b: Year 2 (2018-2019) grain Zn concentration by tillage × Zn method
- Each panel shows 7 grouped bars (treatment methods) with 2 sub-groups per bar (PT vs ZT)

The "No application" bar in each tillage-year combination is the control. The GT in MOESM5 includes 4 control-treatment pairs per application route (soil or foliar), spanning both years and both tillage types, likely selecting specific Zn treatments (soil = 10 kg Zn ha-1; foliar = 0.5% ZnSO4).

**Variance reporting:** Tukey's HSD (honestly significant difference, p ≤ 0.05) is used throughout. Variance is reported as letter notation (a, b, c) over bars in Figure 2, with no numeric LSD/HSD values extractable from the figure. No numeric SE or SD reported per observation.

---

## 3. AI Extraction

**Models used:** Claude only (Kimi extracted 0 observations; tiebreaker applied single-model fallback)

**Total observations extracted:** 80 consensus observations (all claude_only)
- 16 grain Zn observations (Figure 2a/2b)
- 16 straw Zn observations (Figure 2c/2d)
- Additional: seed/coating/priming and other Zn treatment types

The AI correctly identified:
- Figure 2 as the primary data source for grain Zn concentration
- Control as "No application" treatment
- n = 3 replicates (from Methods: "All the treatments were replicated thrice")
- Split-plot factorial structure (tillage × Zn method × year)
- Application route distinction (soil vs foliar vs seed coating vs priming)

The AI extracted all 4 Soil application and 4 Foliar application grain Zn control-treatment pairs that match the GT. Matching was possible because the extracted values lie within tolerance (combined ctrl + treat relative error ≤ 30%).

**Variance:** All extracted observations have `variance_type = null` and `treatment_variance = null`. The figure shows only letter-notation significance groups, making numeric variance extraction impossible. This is correct — the AI did not hallucinate variance values.

**Confidence:** Medium for all observations (figure-based reading, OCR-challenged scanned PDF).

---

## 4. GT Data

Ground truth from MOESM5_dataset.xlsx. All GT observations are grain Zn concentration (mg kg-1) with n = 3, ZnSO4 fertilizer, Pakistan.

**Sheet: Data 2 Soil application (study_id = 18)**

| Obs ID | GT ctrl (mg/kg) | GT treat (mg/kg) | GT effect (%) |
|--------|----------------|-----------------|---------------|
| 842    | 33.1984        | 42.1053         | +26.83%       |
| 843    | 34.0081        | 43.1174         | +26.79%       |
| 844    | 36.0729        | 44.0891         | +22.22%       |
| 845    | 37.4089        | 47.2065         | +26.19%       |

**Sheet: Data 3 Foliar application (study_id = 71)**

| Obs ID | GT ctrl (mg/kg) | GT treat (mg/kg) | GT effect (%) |
|--------|----------------|-----------------|---------------|
| 762    | 34.2105        | 45.1417         | +31.95%       |
| 763    | 35.0202        | 40.0810         | +14.45%       |
| 764    | 36.2955        | 45.8704         | +26.38%       |
| 765    | 38.0769        | 44.9798         | +18.13%       |

The 4 GT control values per sheet span four different tillage-by-year combinations:
- ~33-34 mg/kg = Year 1, lower-performing tillage type
- ~34-35 mg/kg = Year 1, other tillage type
- ~36 mg/kg = Year 2, one tillage type
- ~37-38 mg/kg = Year 2, other tillage type

GT values carry 4-decimal precision because they appear to be computed averages derived from raw data in the MOESM5 spreadsheet, not direct text transcription from the paper.

---

## 5. Root Cause: What Causes the 2% MAE?

The 2% MAE arises from a single, structural source: **the AI reads grain Zn values by visual estimation from bar charts in a scanned figure, while the GT contains computed numeric averages with 4-decimal precision.** Both ctrl and treat values are subject to this bar-reading imprecision.

### Per-Observation Match Table

| App type | AI ctrl | AI treat | GT ctrl  | GT treat  | AI effect% | GT effect% | Abs error |
|----------|---------|---------|---------|---------|-----------|-----------|-----------|
| Soil     | 34.0    | 43.0    | 34.0081 | 43.1174 | +26.47    | +26.79    | 0.32 pp   |
| Soil     | 36.0    | 44.0    | 36.0729 | 44.0891 | +22.22    | +22.22    | 0.00 pp   |
| Soil     | 33.5    | 42.0    | 33.1984 | 42.1053 | +25.37    | +26.83    | 1.46 pp   |
| Soil     | 36.5    | 47.5    | 37.4089 | 47.2065 | +30.14    | +26.19    | 3.95 pp   |
| Foliar   | 36.0    | 46.5    | 36.2955 | 45.8704 | +29.17    | +26.38    | 2.79 pp   |
| Foliar   | 33.5    | 45.5    | 34.2105 | 45.1417 | +35.82    | +31.95    | 3.87 pp   |
| Foliar   | 36.0    | 40.0    | 35.0202 | 40.0810 | +11.11    | +14.45    | 3.34 pp   |
| Foliar   | 36.5    | 43.0    | 38.0769 | 44.9798 | +17.81    | +18.13    | 0.32 pp   |

**MAE = 2.01 pp | r = 0.972 | All 8 within 5 pp | All 8 within 10 pp**

### Mechanism of the MAE

**1. Bar chart reading precision (primary cause)**
The AI reads values from Figure 2 bar charts in a scanned PDF. The y-axis likely spans roughly 25-55 mg/kg, meaning each pixel corresponds to several tenths of a mg/kg. The AI rounds extracted values to the nearest 0.5 mg/kg. Observed rounding errors:

- Control values: deviations of 0.01 to 1.58 mg/kg from GT (mean ~0.56 mg/kg)
- Treatment values: deviations of 0.09 to 1.98 mg/kg from GT (mean ~0.63 mg/kg)

**2. Error cancellation and amplification in effect size calculation**
The effect size (% change) is: `(treat - ctrl) / ctrl * 100`. When both ctrl and treat are over-estimated by similar amounts, errors partially cancel (good case: 0.00 pp error for the Soil obs 2). When ctrl is over-estimated but treat is close, or vice versa, errors accumulate (worst case: 3.95 pp for Soil obs 4, where AI ctrl = 36.5 vs GT = 37.41, a 0.91 under-estimate of the control denominator, inflating the apparent % increase).

**3. Control value ambiguity across tillage types within a figure panel**
Each figure panel (year) shows multiple bars including a "No application" bar. In a split-plot design with two tillage types, there are actually two different control means per year (one for PT, one for ZT). The AI correctly extracts both (evidenced by 8 distinct extracted observations), but reading two closely spaced bars in a compressed bar chart introduces cross-bar confusion. The two Year-2 control values are GT 36.07 and 37.41 (1.34 mg/kg apart) — within the visual resolution of a scanned bar chart.

**4. No systematic bias**
The errors are mixed in sign: the AI over-estimates the effect in 4 cases and under-estimates in 4 cases. There is no consistent direction (bias), only random visual estimation noise.

### Why the MAE Is Not Lower

The GT values (e.g., 33.1984, 42.1053) are computed to 4 decimal places from data that does not appear directly in the paper's figure. They likely represent means calculated from raw replication data in the authors' dataset, which may carry slightly more precision than what the published bar chart can convey. Even a human reader digitizing Figure 2 would expect ~1-2 mg/kg absolute reading error, translating to approximately 2-4 percentage-point effect-size error — consistent with the observed MAE.

---

## 6. Assessment: Good Quality

### Summary

| Metric            | Value            |
|-------------------|------------------|
| GT observations   | 8 (Soil: 4, Foliar: 4) |
| Matched           | 8 / 8 (100%)     |
| MAE               | 2.01 pp          |
| Pearson r         | 0.972            |
| Within 5 pp       | 8 / 8 (100%)     |
| Within 10 pp      | 8 / 8 (100%)     |
| Systematic bias   | None (mixed sign errors) |
| Variance captured | 0 / 8 (letter notation only, not extractable) |

### Positive aspects

- **Perfect match rate (8/8):** The AI identified the correct control-treatment pairing for all 8 GT observations across two application routes (soil and foliar) and two year-tillage combinations. This is non-trivial given the paper's complex factorial design.
- **Correct control identification:** "No application" correctly selected as control; hydro-priming (water-only) not confused with Zn treatments.
- **Correct n:** n = 3 extracted correctly from Methods text.
- **No hallucinated variance:** The AI recognized that figure-based bar charts provide no numeric variance and correctly returned null, rather than fabricating values.
- **Correct application route tagging:** Soil and foliar treatments correctly distinguished, enabling match to the correct MOESM5 sheets.

### Limitations

- **No variance:** HSD significance letters in Figure 2 cannot be converted to numeric SE/SD. This observation set contributes 0 variance-equipped rows to any inverse-variance weighted analysis.
- **Figure-reading imprecision:** ~2 pp MAE is the inherent floor for bar-chart digitization from a scanned PDF. This is not an AI failure — it is the physical limit of the data representation in the source document.
- **Single model:** Kimi extracted 0 observations (likely unable to read the scanned figure), so no cross-model consensus was available. All values come from Claude alone, reducing confidence slightly.

### Overall rating: GOOD

The extraction correctly recovers all 8 GT-comparable observations, achieves r = 0.972, and maintains MAE within the expected precision ceiling for scanned bar-chart data. The 2% MAE is attributable entirely to visual estimation imprecision in figure reading, not to conceptual or structural extraction errors. This paper makes a valid contribution to the Hui 2023 validation dataset.
