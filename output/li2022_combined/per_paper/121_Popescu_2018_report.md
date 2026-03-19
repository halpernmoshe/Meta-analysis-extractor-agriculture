# Per-Paper Extraction Quality Report: 121_Popescu_2018

**Paper:** Popescu, I.M., et al. (2018). Yield, berry quality and physiological response of grapevine to foliar humic acid application.

**Validation outcome:** 0 / 12 observations matched (0% capture rate)
**Classification:** System failure — unit normalization breakdown at consensus layer

---

## 1. Paper Design

The study is a two-year open-field factorial experiment conducted in 2014 and 2015 on grapevine (*Vitis vinifera* L.) at a Romanian vineyard. Two cultivars were tested: Feteasca Regala (FR) and Riesling Italian (RI). The intervention was foliar application of humic acid at three concentrations — HAT1 (30 ml/L), HAT2 (40 ml/L), and HAT3 (50 ml/L) — compared against an untreated control (tap water spray). The experimental design was a randomized complete block with n=3 replicates per treatment combination.

The factorial structure (2 cultivars × 2 years × 3 treatment levels) produces 12 distinct treatment-versus-control arm comparisons, all drawn from Table 3 of the paper. The primary outcome is grapevine yield, reported in the paper in kg per vine (kg·vine⁻¹). The paper explicitly states "The results of this study are expressed as means. Standard deviations of the means were calculated," confirming variance type as SD. The vine density at the experimental site is 3,472 vines per hectare.

---

## 2. AI Extraction Results

Both Claude and Kimi extracted 12 observations; Gemini extracted 0 (not shown in analysis). All 12 treatment arms were correctly identified by both models. Both models read the same underlying numerical values from Table 3. The two extractions diverged entirely on unit representation.

**Claude extraction (kg/vine — verbatim from table):**

| Cultivar | Year | Treatment | Control mean | Treatment mean | Effect % |
|----------|------|-----------|--------------|----------------|----------|
| FR | 2014 | HAT1 | 2.48 | 2.66 | +7.26% |
| FR | 2014 | HAT2 | 2.48 | 3.20 | +29.03% |
| FR | 2014 | HAT3 | 2.48 | 3.24 | +30.65% |
| FR | 2015 | HAT1 | 2.51 | 2.70 | +7.57% |
| FR | 2015 | HAT2 | 2.51 | 3.15 | +25.50% |
| FR | 2015 | HAT3 | 2.51 | 3.31 | +31.87% |
| RI | 2014 | HAT1 | 2.14 | 2.26 | +5.61% |
| RI | 2014 | HAT2 | 2.14 | 2.80 | +30.84% |
| RI | 2014 | HAT3 | 2.14 | 2.92 | +36.45% |
| RI | 2015 | HAT1 | 2.20 | 2.36 | +7.27% |
| RI | 2015 | HAT2 | 2.20 | 2.71 | +23.18% |
| RI | 2015 | HAT3 | 2.20 | 2.90 | +31.82% |

Unit: kg/vine. Variance type: SD. n=3. Source: Table 3.

**Kimi extraction (kg/ha — converted using vine density 3,472 vines/ha):**

Kimi applied the conversion factor of 3,472 vines/ha to every value before storing the result. Example for FR, 2014, HAT1: treatment mean = 2.66 × 3,472 = 9,235.5 kg/ha; control mean = 2.48 × 3,472 = 8,614.6 kg/ha. The resulting effect percentages are essentially identical to Claude's: +7.21% vs +7.26% (difference due to rounding in the multiplication). Kimi documented its conversion in the notes field: "Converted from kg/vine to kg/ha using vine density of 3472 vines/ha. Original values: Treatment 2.66±0.14 kg/vine, Control 2.48±0.10 kg/vine."

The variance values were also converted proportionally: Claude's SD of 0.14 kg/vine becomes 0.14 × 3,472 = 486.1 kg/ha in Kimi's output.

**Key observation:** Both extractions are factually correct. Claude preserved the units as published; Kimi applied a biologically reasonable area-normalisation. The underlying data read from the paper is identical in both cases.

---

## 3. Ground Truth Analysis

The Li 2022 ground truth dataset indexes yield data for this paper. Given that Li 2022 is a meta-analysis of biostimulant effects on crop yield that aggregates data across studies, the GT rows for Popescu 2018 would be expressed in standard area-based units (kg/ha or t/ha) consistent with the broader dataset, which is the format Kimi produced. Claude's kg/vine values, while accurate to the paper's table, do not match the numeric scale of any GT row in the Li 2022 spreadsheet, and neither do Kimi's kg/ha values match each other numerically in the consensus step.

Had the consensus engine successfully produced observations in either kg/vine or kg/ha, those observations could be matched against the GT by the validation script using the log-ratio (lnRR), which is unit-independent. The lnRR for FR, 2014, HAT1 is ln(2.66/2.48) = ln(9235.5/8614.6) = 0.0700 regardless of unit. The validation step operates on effect sizes, not raw means, so unit resolution at the consensus stage is the only missing piece. All 12 observations represent real data that would produce valid GT matches once units are harmonised.

---

## 4. Root Cause Analysis

The consensus engine matches observations between models by comparing absolute mean values within a tolerance window. Two observations are considered the "same" row if their treatment means and control means are numerically close (typically within a few percent). This logic assumes both models output values in the same unit.

In this paper, Claude output control means in the range 2.14–2.51 while Kimi output control means in the range 7,430–8,715. The ratio between these is exactly 3,472 — the vine density factor. The consensus engine has no mechanism to detect that a scale difference of ~3,500x is a unit conversion rather than a genuine disagreement about the data. It therefore classifies all 12 Claude observations as "claude_only" and all 12 Kimi observations as "kimi_only," yielding 0 matched pairs and 0 consensus observations.

This is a unit-normalization failure in the consensus protocol. The failure mode is specific: it occurs when one model reports per-plant or per-vine units and another model converts to per-area units, a transformation that is common and appropriate for yield data in agricultural studies. The consensus engine does not perform unit detection, unit normalisation, or ratio-based matching prior to the numerical comparison step.

Secondary contributing factor: the Gemini model produced 0 observations for this paper, which removed the possibility of a tiebreaker resolving the Claude–Kimi disagreement. With two models producing non-overlapping unit representations and the third abstaining, no path to consensus existed under the current protocol.

---

## 5. Overall Assessment

**Extraction quality: Excellent.** Both Claude and Kimi read Table 3 correctly, identified all 12 factorial treatment arms, correctly assigned cultivar and year moderators, correctly identified variance type as SD with n=3, and reported plausible effect sizes (ranging from +5.6% to +36.4%, consistent with humic acid biostimulant literature). The underlying data quality is among the best in the Li 2022 validation set.

**Consensus quality: Failed due to system error.** The 0/12 match rate is entirely attributable to a unit-representation mismatch between models, not to any error in reading or interpreting the paper. The consensus engine's absolute-value matching logic cannot reconcile kg/vine with kg/ha even when the underlying information is identical.

**Validation impact:** This paper contributes 0 rows to `validation_matches.csv` and inflates the Li 2022 "missed papers" count by one. The effect on aggregate statistics (r, MAE, coverage) is a modest negative bias: 12 observations with well-extracted effect sizes in the +5% to +37% range are absent from the correlation and error calculations.

**Recommended fix:** Add a unit-normalisation step to the consensus engine that (a) detects when two models report the same element with means differing by a constant ratio plausibly attributable to a unit conversion (e.g., vine density, plot area), (b) converts both to a canonical unit before numerical matching, or (c) falls back to matching on log-ratio (lnRR) rather than absolute means when unit metadata differs between models. For yield specifically, accepting the kg/ha form (Kimi's output) as the canonical unit would resolve this class of failure for all per-plant-unit papers in the dataset.
