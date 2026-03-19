# Per-Paper Extraction Quality Report: 127_Radkowski_2018

**Paper:** Radkowski, A. (2018). Influence of foliar fertilization with amino acid preparations on morphological traits and seed yield of timothy (*Phleum pratense* L.)
**Paper ID:** `127_Radkowski_2018_Influence of foliar fertilization with a`
**Dataset:** Li 2022 validation set
**Report date:** 2026-02-18

---

## 1. Paper Design

This Polish field trial examined the effect of foliar application of Microfert, a commercial amino acid (protein hydrolysate) biostimulant, on seed yield and morphological traits of timothy grass (*Phleum pratense* cv. Owacja). The experiment ran across three consecutive growing seasons (2015, 2016, 2017) using a randomized complete block design with four replications (plot size 10 m²).

The intervention applied the biostimulant at three dose levels: 1.8, 3.0, and 4.5 L/ha, compared against an unfertilized water-only control. The primary reportable outcome for the Li 2022 meta-analysis is seed yield (kg/ha), presented by year and dose in Table 3. The paper also reports germination capacity (%), shoot height, flag leaf dimensions, inflorescence dimensions, and SPAD chlorophyll index values across multiple stages — none of which appear in the meta-analysis ground truth. The PDF is a scanned document, introducing potential OCR noise. Variance is presented as mean ± value throughout tables, but the variance type (SD vs SE) is not explicitly declared in the Methods section.

---

## 2. AI Consensus Extraction Results

The consensus pipeline ran two models: Claude (24 observations extracted) and Kimi (45 observations extracted). Gemini was not used for this paper. No tiebreaker was required. The combined consensus output contains 29 matched observations.

The 9 per-year, per-dose seed yield observations (Table 3, json_idx 0-8) were extracted in full by both models with consistent values:

| Year | Dose (L/ha) | Control (kg/ha) | Treatment (kg/ha) | Effect (%) |
|------|------------|-----------------|-------------------|-----------|
| 2015 | 1.8 | 994 | 1003 | +0.91 |
| 2015 | 3.0 | 994 | 1071 | +7.75 |
| 2015 | 4.5 | 994 | 1132 | +13.88 |
| 2016 | 1.8 | 820 | 875 | +6.71 |
| 2016 | 3.0 | 820 | 942 | +14.88 |
| 2016 | 4.5 | 820 | 948 | +15.61 |
| 2017 | 1.8 | 809 | 838 | +3.58 |
| 2017 | 3.0 | 809 | 888 | +9.77 |
| 2017 | 4.5 | 809 | 915 | +13.10 |

Sample size (n=4 replicate plots) was correctly identified from the Methods section. Variance type was flagged as SD with medium confidence, based on the "mean ± value" cell format, with a warning that the type was not explicitly declared. Beyond the 9 seed yield rows, the consensus also captured 3-year mean seed yields (3 dose arms), 3-year mean germination capacity (3 dose arms from Claude), and a full set of morphological and SPAD observations from Kimi (shoot height, flag leaf length, flag leaf width, inflorescence dimensions) — all correctly sourced from Tables 1 and 2 but outside the scope of the Li 2022 yield meta-analysis.

---

## 3. Ground Truth Comparison (9 Matched Pairs)

All 9 ground truth rows (GT pair IDs 426-434) were matched with perfect confidence.

| GT pair | Year | Dose (normalized) | GT effect (%) | Extracted effect (%) | Absolute error |
|---------|------|-------------------|--------------|----------------------|----------------|
| 426 | 2015 | 0.40 | 0.9055 | 0.9055 | 0.00% |
| 427 | 2015 | 0.67 | 7.7465 | 7.7465 | 0.00% |
| 428 | 2015 | 1.00 | 13.8832 | 13.8832 | 0.00% |
| 429 | 2016 | 0.40 | 6.7073 | 6.7073 | 0.00% |
| 430 | 2016 | 0.67 | 14.8780 | 14.8780 | 0.00% |
| 431 | 2016 | 1.00 | 15.6098 | 15.6098 | 0.00% |
| 432 | 2017 | 0.40 | 3.5847 | 3.5847 | 0.00% |
| 433 | 2017 | 0.67 | 9.7650 | 9.7650 | 0.00% |
| 434 | 2017 | 1.00 | 13.1000 | 13.1000 | 0.00% |

**Summary statistics:** N=9, MAE=0.00%, direction agreement=100/100%. No unmatched ground truth rows. The ground truth normalizes doses as fractions of the maximum (1.8/4.5=0.40, 3.0/4.5=0.667, 4.5/4.5=1.00), which differs from the JSON's absolute L/ha values, but this is a metadata encoding difference only — effect sizes are numerically identical. The ground truth also stores control means in t-equivalent units (e.g., 0.0994 t/ha = 99.4 kg/ha × 10 = 994 kg/ha), a 10,000x unit scaling artifact that does not affect the computed percent effects.

---

## 4. Root Cause Analysis

**Why this paper achieves perfect accuracy:**

1. **Clean tabular structure.** Table 3 presents seed yield in a simple rows-by-treatment layout with one outcome variable (seed yield), three dose arms, and three year columns. There is no factorial interaction complexity requiring inference about which cell belongs to which comparison.

2. **Unambiguous control.** The control is labeled "Control" in every table row — no synonym matching, treatment-control swap risk, or implicit baseline was required.

3. **Consistent units.** Yield values are in kg/ha throughout, with no within-paper unit conversion needed. The unit discrepancy between the JSON (kg/ha) and the Li 2022 ground truth (t-equivalent scaled values) is a meta-analysis encoding convention, not an extraction error.

4. **No variance ambiguity for the matched outcome.** While the variance type was flagged as uncertain (SD vs SE, medium confidence), this ambiguity was irrelevant to effect size computation, which depends only on means. The percent effect calculation is fully determined by control and treatment means alone.

5. **Strong model agreement.** Both Claude and Kimi independently extracted the same 9 per-year seed yield rows with matching values, providing mutual validation without requiring a tiebreaker pass.

6. **No OCR-corrupted numbers.** Despite the scanned PDF warning, the key numeric values in Table 3 were read correctly by both models.

---

## 5. Overall Assessment

**Quality rating: PERFECT (5/5)**

This paper represents best-case extraction performance. All 9 ground truth observations were captured with zero numeric error and 100% directional accuracy. The AI consensus pipeline handled the multi-year, multi-dose structure correctly, correctly identified per-year annual observations as the unit of analysis (rather than the 3-year means), and correctly sourced data from Table 3.

The paper's breadth of additional outcomes — morphological traits and SPAD values extracted by Kimi, averaged yield rows from Claude — represents appropriate over-extraction rather than a quality failure: the pipeline correctly captured everything in the paper and the matching step then correctly identified which subset maps to the Li 2022 inclusion criteria.

The one minor limitation is that no variance values were reported for the 3-year mean rows, and the variance type for the annual data was flagged as uncertain. This does not affect the validity of the matched effect sizes but would limit downstream random-effects meta-analysis weighting for this paper's observations unless variance is imputed.
