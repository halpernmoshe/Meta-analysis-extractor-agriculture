# Extraction Quality Report: 21_Wang_2012

**Paper:** Wang, J.W., 2012. Effects of selenium and zinc on yield and mineral nutrition of main crops on dryland. *Thesis for Doctor's Degree*. Northwest A&F University, China.

**Match summary:** 14/15 GT matched, r = 0.406, MAE = 13.78%

**Status: POOR — Structurally broken matching despite high pair count**

---

## 1. Paper Design

Wang (2012) is a doctoral thesis reporting two factorial field experiments on dryland in China's Loess Plateau. The study examined the effects of soil and foliar zinc application on two crops grown in rotation:

- **Crop 1 (maize):** Zhendan 958 cultivar, two growing seasons (2008, 2009)
- **Crop 2 (wheat):** Jinmai 47 cultivar, two growing seasons (2008–2009, 2009–2010)
- **Treatments (wheat/foliar):** 5 Zn rates applied as foliar spray (0.23, 0.45, 0.68, 0.91, 1.14 kg Zn ha−1 as ZnSO4)
- **Treatments (wheat/soil):** 4 Zn rates applied as soil broadcast (3.405, 6.810, 10.215, 13.620 kg Zn ha−1 as ZnSO4·7H2O; also labelled as 0, 1×, 2×, 3× multiples)
- **Design:** Randomized complete block, n = 4 replicates (wheat), n = 6 replicates (maize; recon incorrectly used n=6 for all)
- **Key outcome:** Grain Zn concentration (mg/kg DW) in **wheat only** — the Hui 2023 meta-analysis covers wheat and rice exclusively, so maize data is out-of-scope for the GT

**Site characteristics (from MOESM5):**
| Experiment | Available soil Zn (mg/kg) | pH | Country |
|---|---|---|---|
| Soil app (wheat) | 0.73 | 8.12 | China |
| Foliar app (wheat) | 0.73–0.76 | 8.12–8.18 | China |

**MOESM5 encoding:** The paper appears in two sheets with separate study IDs:
- "Data 2 Soil application" → study_id = **21** (4 wheat observations)
- "Data 3 Foliar application" → study_id = **11** (11 wheat observations)
- Total GT observations: **15**

---

## 2. Grain Zn Data in PDF

The paper's Table 3 presents Zn concentrations for grain, ear leaf, and shoot for both crops. Based on the AI extraction and the MOESM5 values, the PDF contains the following grain Zn data structure:

**Wheat grain Zn — Soil application experiment:**
| Zn rate (kg Zn ha−1) | Control (mg/kg) | Treatment (mg/kg) | Effect (%) |
|---|---|---|---|
| 3.405 (low) | 21.2 | 21.4 | +0.9% |
| 6.810 (medium) | 21.2 | 23.2 | +9.4% |
| 10.215 (high) | 21.2 | 22.3 | +5.2% |
| 13.620 (very high) | 21.2 | 21.8 | +2.8% |

Note: These are modest effects (all < 10%), consistent with a Zn-deficient but high-pH calcareous soil where soil-applied Zn has low bioavailability.

**Wheat grain Zn — Foliar application experiment (two field sites/seasons):**
| Zn rate (kg Zn ha−1) | Control (mg/kg) | Treatment (mg/kg) | Effect (%) |
|---|---|---|---|
| 0.23 | 20.0 | 22.0 | +10.0% |
| 0.45 | 20.0 | 27.0 | +35.0% |
| 0.68 | 20.0 | 28.0 | +40.0% |
| 0.91 | 20.0 | 31.0 | +55.0% |
| 1.14 | 20.0 | 30.0 | +50.0% |
| 0.28 | 19.8 | 22.7 | +14.6% |
| 0.57 | 19.8 | 23.3 | +17.7% |
| 0.85 | 19.8 | 24.8 | +25.3% |
| 0.28 | 21.2 | 22.0 | +3.8% |
| 0.57 | 21.2 | 23.8 | +12.3% |
| 0.85 | 21.2 | 27.4 | +29.2% |

Note: Three groups of foliar observations arise because the paper reports results from two different experimental locations/seasons (leading to different baseline control values: 20.0, 19.8, 21.2 mg/kg).

The PDF also contains equivalent data for **maize** (grain Zn, ear leaf Zn, shoot Zn) and **wheat shoot/leaf Zn** — these are additional tissues/crops that the Hui 2023 meta-analysis does not include in its grain Zn GT dataset.

---

## 3. AI Consensus Extraction Results

The consensus pipeline (Claude + Kimi; Gemini produced 0 observations) extracted **24 Zn observations** from Table 3, covering all crops and tissues. The extraction was technically accurate — values match the PDF — but the scope was far broader than the 15 GT rows.

**Extracted observations by category:**

| Tissue | Crop | Count | Treatment types |
|---|---|---|---|
| Ear leaf | Maize | 6 | S50, F4, F4+S50 × 2 seasons |
| Shoot | Wheat | 6 | S50, F4, F4+S50 × 2 seasons |
| Grain | Maize | 6 | S50, F4, F4+S50 × 2 seasons |
| Grain | Wheat | 6 | S50, F4, F4+S50 × 2 seasons |

**Wheat grain Zn extraction (the only GT-relevant subset):**

| Season | Treatment | Control (mg/kg) | Treatment (mg/kg) | Effect (%) |
|---|---|---|---|---|
| First (2008–09) | S50 (soil) | 18.79 | 19.48 | +3.7% |
| First (2008–09) | F4 (foliar) | 18.79 | 24.40 | +29.9% |
| First (2008–09) | F4+S50 (combined) | 18.79 | 24.03 | +27.9% |
| Second (2009–10) | S50 (soil) | 23.11 | 29.11 | +26.0% |
| Second (2009–10) | F4 (foliar) | 23.11 | 35.59 | +54.0% |
| Second (2009–10) | F4+S50 (combined) | 23.11 | 43.61 | +88.7% |

**Critical observations about the extraction:**

1. The extraction extracted only **6 wheat grain observations** (3 treatments × 2 seasons), whereas the GT requires **15** (4 soil-rate obs + 11 foliar-rate obs).

2. The AI simplified the Zn-rate dose-response design into a 3-level treatment structure (S50, F4, F4+S50) rather than extracting each Zn dose level as a separate observation.

3. The control values extracted (18.79 and 23.11 mg/kg) do not match the MOESM5 GT control values (20.0, 19.8, 21.2, 21.2 mg/kg). This indicates the AI extracted the Zn-0 control from a different data column, possibly averaging across locations/seasons or reading the "initial" value from a different table.

4. The variance type is LSD (correctly identified), but all variance values are null — LSD values were present in the PDF but not captured numerically.

5. The recon correctly warned about "Combined treatment F4+S50 is not just soil application" and about the dose-response structure, but the extraction did not act on this guidance to extract each dose level separately.

---

## 4. Ground Truth (MOESM5) Data

### 4a. Data 2 — Soil Application (study_id = 21)

| Obs ID | Zn rate (kg/ha) | n | Grain Zn ctrl (mg/kg) | Grain Zn treat (mg/kg) | Effect (lnRR) | Effect (%) |
|---|---|---|---|---|---|---|
| 121 | 3.405 | 4 | 21.2 | 21.4 | 0.0094 | +0.9% |
| 122 | 6.810 | 4 | 21.2 | 23.2 | 0.0902 | +9.4% |
| 123 | 10.215 | 4 | 21.2 | 22.3 | 0.0506 | +5.2% |
| 124 | 13.620 | 4 | 21.2 | 21.8 | 0.0279 | +2.8% |

**Soil metadata:** Available Zn = 0.73 mg/kg, pH = 8.12, N = 160 kg/ha, P = 35.2 kg/ha, K = 0 kg/ha

The GT text file (`gt_21_Wang_2012.txt`) showed `Grain Zn concentration = 21.2` for all four soil rows — this was the CONTROL value only (column 33 in MOESM5). The treatment values in column 34 (21.4, 23.2, 22.3, 21.8) were present in the spreadsheet but not displayed in the text summary. This did not affect validation (the script reads MOESM5 directly), but it could mislead anyone relying on the text file alone.

### 4b. Data 3 — Foliar Application (study_id = 11)

| Obs ID | Zn rate (kg/ha) | n | Grain Zn ctrl (mg/kg) | Grain Zn treat (mg/kg) | Effect (%) |
|---|---|---|---|---|---|
| 77 | 0.23 | 4 | 20.0 | 22.0 | +10.0% |
| 78 | 0.45 | 4 | 20.0 | 27.0 | +35.0% |
| 79 | 0.68 | 4 | 20.0 | 28.0 | +40.0% |
| 80 | 0.91 | 4 | 20.0 | 31.0 | +55.0% |
| 81 | 1.14 | 4 | 20.0 | 30.0 | +50.0% |
| 82 | 0.28 | 4 | 19.8 | 22.7 | +14.6% |
| 83 | 0.57 | 4 | 19.8 | 23.3 | +17.7% |
| 84 | 0.85 | 4 | 19.8 | 24.8 | +25.3% |
| 85 | 0.28 | 4 | 21.2 | 22.0 | +3.8% |
| 86 | 0.57 | 4 | 21.2 | 23.8 | +12.3% |
| 87 | 0.85 | 4 | 21.2 | 27.4 | +29.2% |

**Foliar metadata:** Available Zn = 0.73–0.76 mg/kg, pH = 8.12–8.18; three subgroups reflect different experimental conditions or seasons (baseline ctrl varies across 20.0, 19.8, 21.2 mg/kg). Straw Zn concentrations are only present for obs 82–87 (not 77–81), suggesting site-level differences in measurement scope.

---

## 5. Root Cause Analysis

### 5.1 Why 14/15 Matched Despite Fundamental Mismatch

The 14/15 match rate is **numerically misleading**. The matching algorithm (`match_observations`) pairs extracted and GT observations by minimizing the combined relative error in control and treatment values, with a generous tolerance of `combined_error ≤ 0.30`. Because the AI extracted 24 observations with diverse control/treatment value combinations (ranging from 14.36 to 43.61 mg/kg), and the GT has 15 observations with values in a similar range (19.8–31.0 mg/kg), the algorithm finds approximate numerical coincidences across the two datasets — even though the observations being paired do not correspond to the same experimental conditions.

Reconstructing the matching reveals the following 14 pairs (sorted by combined error, ascending):

| Extracted observation | GT observation | Combined err | Effect error |
|---|---|---|---|
| Shoot Zn / wheat / F4+S50 / s1 (ctrl=19.97, treat=25.04, +25.4%) | Foliar 0.85 kg (ctrl=19.8, treat=24.8, +25.3%) | 0.018 | **0.1 pp** |
| Shoot Zn / wheat / S50 / s1 (ctrl=19.97, treat=23.53, +17.8%) | Foliar 0.57 kg (ctrl=19.8, treat=23.3, +17.7%) | 0.018 | **0.1 pp** |
| Ear leaf / maize / F4 / s1 (ctrl=22.98, treat=22.5, −2.1%) | Soil 10.215 kg (ctrl=21.2, treat=22.3, +5.2%) | 0.093 | 7.3 pp |
| Shoot Zn / wheat / F4 / s2 (ctrl=22.98, treat=23.53, +2.4%) | Foliar 0.57 kg (ctrl=21.2, treat=23.8, +12.3%) | 0.095 | 9.9 pp |
| Shoot Zn / wheat / F4 / s1 (ctrl=19.97, treat=19.73, −1.2%) | Foliar 0.23 kg (ctrl=20.0, treat=22.0, +10.0%) | 0.105 | 11.2 pp |
| Grain Zn / wheat / F4+S50 / s1 (ctrl=18.79, treat=24.03, +27.9%) | Foliar 0.28 kg (ctrl=19.8, treat=22.7, +14.6%) | 0.110 | 13.3 pp |
| Ear leaf / maize / S50 / s1 (ctrl=22.98, treat=26.42, +15.0%) | Foliar 0.85 kg (ctrl=21.2, treat=27.4, +29.2%) | 0.120 | 14.2 pp |
| Grain Zn / wheat / F4 / s1 (ctrl=18.79, treat=24.4, +29.9%) | Foliar 0.45 kg (ctrl=20.0, treat=27.0, +35.0%) | 0.157 | 5.1 pp |
| Shoot Zn / wheat / F4+S50 / s2 (ctrl=22.98, treat=31.68, +37.9%) | Foliar 0.91 kg (ctrl=20.0, treat=31.0, +55.0%) | 0.171 | 17.1 pp |
| Grain Zn / wheat / S50 / s2 (ctrl=23.11, treat=29.11, +26.0%) | Foliar 1.14 kg (ctrl=20.0, treat=30.0, +50.0%) | 0.185 | 24.0 pp |
| Grain Zn / wheat / S50 / s1 (ctrl=18.79, treat=19.48, +3.7%) | Soil 3.405 kg (ctrl=21.2, treat=21.4, +0.9%) | 0.203 | 2.8 pp |
| Ear leaf / maize / F4+S50 / s1 (ctrl=22.98, treat=26.24, +14.2%) | Foliar 0.68 kg (ctrl=20.0, treat=28.0, +40.0%) | 0.212 | 25.8 pp |
| Grain Zn / maize / F4 / s2 (ctrl=16.47, treat=21.69, +31.7%) | Soil 13.62 kg (ctrl=21.2, treat=21.8, +2.8%) | 0.228 | **28.9 pp** |
| Grain Zn / maize / F4+S50 / s2 (ctrl=16.47, treat=22.54, +36.9%) | Foliar 0.28 kg (ctrl=21.2, treat=22.0, +3.8%) | 0.248 | **33.1 pp** |

**Unmatched GT row:** Soil Zn 6.810 kg/ha (ctrl=21.2, treat=23.2, +9.4%) — this is the one GT row the algorithm could not pair.

### 5.2 Primary Root Cause: Wrong Extraction Schema for Dose-Response Design

The paper reports a **Zn-rate dose-response study**, not a simple treatment/control comparison. The Hui 2023 meta-analysis encodes this as one row per Zn dose level (each vs. the zero-Zn control). The AI consensus pipeline instead collapsed the dose-response into three aggregate treatment categories (S50, F4, F4+S50), losing the within-treatment dose variation.

The wheat section in MOESM5 has:
- 4 soil Zn dose levels → 4 GT observations
- 11 foliar Zn dose levels (across 2–3 experimental conditions) → 11 GT observations

The AI extracted only 6 wheat grain observations (3 treatments × 2 seasons), missing the dose-rate dimension entirely.

### 5.3 Wrong Source Values: Control Mean Discrepancy

The extracted wheat grain control values (18.79 and 23.11 mg/kg) do not match any GT control value. The MOESM5 GT shows controls of 20.0, 19.8, and 21.2 mg/kg for foliar, and 21.2 mg/kg for soil. This suggests the AI read a different data column — possibly an aggregate or "initial concentration" value from a different table row — rather than the per-experiment zero-Zn control means.

This control value error means that even when the AI paired a wheat grain observation with the correct GT row type (foliar wheat), the extracted effect size was computed from a wrong baseline, systematically distorting the ratio.

### 5.4 Wrong Tissue / Crop Matched to GT

Of the 14 matched pairs, only 2 pairs link an observation that is semantically correct (both are wheat grain Zn):
- Grain Zn / wheat / F4 / s1 → Foliar 0.45 kg (+29.9% vs. GT +35.0%, err = 5.1 pp)
- Grain Zn / wheat / S50 / s1 → Soil 3.405 kg (+3.7% vs. GT +0.9%, err = 2.8 pp)

The remaining 12 matches pair GT grain Zn rows with extracted **shoot Zn, ear leaf Zn, or maize grain Zn** values. These are numerically close enough to pass the tolerance filter but are biologically entirely different observations:

- **8 of 14 matches** involve extracted shoot or leaf observations matched to GT grain observations
- **2 of 14 matches** involve extracted maize grain matched to GT wheat grain

### 5.5 The Correlation Collapse: r = 0.406

With 14 pairs, the Pearson r between extracted effects and GT effects is 0.406 — near-zero correlation. This happens because:

1. The correct pairs (semantically matching grain Zn) have small effect errors but are few.
2. The wrong-tissue pairs produce arbitrary effect combinations:
   - GT effect of +55% paired with extracted effect of +37.9% (shoot data)
   - GT effect of +2.8% paired with extracted effect of +31.7% (maize grain data)
   - GT effect of +3.8% paired with extracted effect of +36.9% (maize grain data)
3. There is no systematic direction to the error — some extracted effects are too high, some too low — so the correlation is near-zero rather than negative (which would indicate a pure T/C swap).

The MAE of 13.78 pp reflects the average of these mismatched pairs.

### 5.6 The Maize Contamination Problem

The Hui 2023 meta-analysis covers wheat and rice only. The AI extracted maize data (6 grain Zn + 6 ear leaf Zn observations), which should never enter the validation. However, because the matching is purely numerical (no crop filter), maize observations compete with — and win over — wheat observations when their control/treatment values happen to be numerically proximate to GT wheat values.

This is a **scope contamination error**: out-of-scope data being matched against in-scope GT creates spurious pairings.

### 5.7 n = 6 vs. n = 4

The recon identified n = 6 from the maize experiment description ("randomized block designs with six replicates"). The wheat experiment used n = 4, as shown in the MOESM5. This incorrect n (6 vs. 4) would propagate to any LSD→SD conversion, but since LSD values were null anyway, the impact is limited to metadata accuracy.

---

## 6. Overall Assessment

### What the AI Got Right

- Correctly identified Table 3 as the primary data source
- Correctly labeled LSD as the variance type (matching the PDF footnote)
- Extracted accurate absolute values from the PDF (the 24 obs match the PDF data)
- Correctly identified two crops, two growing seasons, and three treatment categories
- Zero disagreement between Claude and Kimi (all 24 obs matched perfectly between models)

### What Went Wrong

| Failure Mode | Severity | Impact |
|---|---|---|
| Collapsed dose-response into treatment categories (S50/F4/F4+S50 instead of per-Zn-rate rows) | Critical | Extracted 6 wheat grain obs instead of 15 |
| Wrong control values (18.79 / 23.11 vs. GT 20.0 / 19.8 / 21.2 / 21.2) | High | Effect sizes systematically off even for correct pairs |
| Extracted out-of-scope data (maize, shoot, leaf) that pollutes matching | High | 12/14 matched pairs are tissue/crop mismatches |
| LSD variance values not captured (all null) | Medium | No effect-size precision for inverse-variance weighting |
| n = 6 applied to wheat observations (correct n = 4) | Low | Metadata error only; no effect if LSD values absent |

### Interpretation of r = 0.406, MAE = 13.78%, 14/15 Matched

This combination is characteristic of a **structural schema mismatch**, not extraction noise. The high pair count (14/15) reflects the numerical promiscuity of the matching algorithm at tolerance = 0.30 combined with a large pool of extracted observations (24 total). The near-zero r reflects that the pairings are semantically invalid — the algorithm is comparing arbitrary combinations of extracted and GT effects.

A paper-level r closer to 0.90 would require:
1. Extracting each soil Zn dose rate (3.405, 6.810, 10.215, 13.620 kg/ha) as a separate observation
2. Extracting each foliar Zn dose rate (0.23, 0.45, 0.68, 0.91, 1.14 kg/ha) as a separate observation, per experimental condition
3. Using the correct per-experiment control values (20.0, 19.8, 21.2 mg/kg)
4. Restricting extraction to wheat grain Zn only (or clearly tagging crop/tissue for filtering)

### Recommended Fix

The extraction prompt needs explicit guidance for dose-response papers:
> "If a paper reports multiple application rates of the same treatment type (e.g., 1×, 2×, 3× Zn), extract each rate as a separate observation with the zero-application treatment as the control. Do not average or collapse dose levels."

The validation pipeline should also add a crop filter, accepting only observations tagged as wheat (or rice) when validating against the Hui 2023 wheat/rice meta-analysis GT, to prevent maize/shoot/leaf observations from polluting the matching.
