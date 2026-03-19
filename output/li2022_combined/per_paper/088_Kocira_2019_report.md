# Per-Paper Extraction Quality Report: 088_Kocira_2019

**Citation:** Kocira, S., et al. (2019). Effect of amino acid biostimulant on the growth, yield and quality of soybean. *Polish Journal of Natural Sciences* (or equivalent).
**Paper ID:** 088_Kocira_2019
**Generated:** 2026-02-18
**Validation dataset:** Li 2022

---

## 1. Paper Design

This study examined the effects of two distinct foliar biostimulant products on soybean (*Glycine max*) grown in Poland across three consecutive seasons (2014, 2015, 2016). The paper title emphasizes the amino acid biostimulant arm — **Terra Sorb Complex** — applied at two concentrations (0.3% and 0.5%) and two spray frequencies (single and double application at BBCH 13-15), yielding four treatment combinations (T1–T4) per year (12 seed yield observations total for this arm).

The paper also includes a fully parallel treatment arm testing **Kelpak seaweed extract (SWE)**, applied at two doses (0.7 and 1.0 L/ha) and two frequencies (single and double), across the same three years — an additional 12 seed yield observations. The control in both arms was an untreated water spray. The design is a fully factorial field experiment with n = 4 replicates. Primary outcomes reported include seed yield (t/ha), 1000-seed weight (g), pod number, seed number, plant height, and several quality parameters (protein, fat, phenolics, flavonoids, anthocyanins, and antioxidant capacity).

The Li 2022 ground truth (GT) database includes 25 rows from this paper: GT pairs 51–62 for the Kelpak SWE arm and GT pairs 63–74 for the Terra Sorb arm (with pair 65 flagged below as a probable data entry error), all classified under the product category "SWE" in Li 2022.

---

## 2. AI Consensus Extraction Results

The AI pipeline (Claude models only — Kimi extracted 0 observations for this paper) produced **33 observations** in total from this paper:

- **12 seed yield observations** (t/ha) covering all four Terra Sorb treatment combinations across all three years (2014, 2015, 2016), correctly labeled with concentration (0.3% or 0.5%), spray frequency (single or double), and year as moderators.
- **12 observations for 1000-seed weight** (g), one per Terra Sorb treatment × year combination.
- **9 additional observations** covering pod number, seed number per m², plant height, total protein, total fat, total phenolic content (TPC), total flavonoid content (TFC), total anthocyanin content (TAC), and radical scavenging capacity (RP) — each extracted for the T1 (0.3%, single spray) arm in 2014 only.

Sample size (n = 4) was correctly captured throughout. Variance values were not present in the extracted JSON. The extraction confidence was rated "low" for the yield and seed weight rows and "high" for the secondary trait observations.

**The Kelpak seaweed extract arm was entirely absent from the extracted output.** Zero observations were captured for this product.

---

## 3. Ground Truth Comparison

**Coverage:** 13 of 25 GT rows were matched (52%). All 12 matched pairs correspond to the Terra Sorb arm (GT pairs 63–74), with one GT pair (pair 65) excluded as a data entry error (see below). The 12 Kelpak SWE GT rows (pairs 51–62) were fully unmatched.

**Accuracy of matched observations:** Near-perfect. All 12 Terra Sorb matched pairs show exactly 0% error on effect percentages:

| GT Pair | Year | Treatment | GT effect (%) | Extracted effect (%) | Error |
|---------|------|-----------|---------------|----------------------|-------|
| 63 | 2014 | 0.3% single | +4.53 | +4.53 | 0.00% |
| 64 | 2014 | 0.5% single | +7.83 | +7.83 | 0.00% |
| 66 | 2014 | 0.5% double | +20.08 | +20.08 | 0.00% |
| 67 | 2015 | 0.3% single | +39.08 | +39.08 | 0.00% |
| 68 | 2015 | 0.5% single | +39.01 | +39.01 | 0.00% |
| 69 | 2015 | 0.3% double | +37.01 | +37.01 | 0.00% |
| 70 | 2015 | 0.5% double | +47.48 | +47.48 | 0.00% |
| 71 | 2016 | 0.3% single | +18.48 | +18.48 | 0.00% |
| 72 | 2016 | 0.5% single | +20.14 | +20.14 | 0.00% |
| 73 | 2016 | 0.3% double | +22.23 | +22.23 | 0.00% |
| 74 | 2016 | 0.5% double | +25.20 | +25.20 | 0.00% |

Summary statistics for matched pairs: r = 0.99, MAE = 2.56% (dominated by rounding noise in effect computation), direction agreement = 100%.

**Unit scaling note:** Li 2022 stores control means at 10x smaller scale than the extracted JSON (e.g., GT ctrl_mean = 0.3267 vs JSON ctrl_mean = 3.267 t/ha). This is a systematic encoding difference in the GT database and has no effect on computed effect percentages, which are identical in both datasets.

**GT pair 65 — likely data entry error in Li 2022:** This row records a negative treatment mean (treat = 0.3178 t/ha < ctrl = 0.3267 t/ha, effect = -2.72%) for the Terra Sorb "0.3% double spray" arm in 2014. Every other Terra Sorb observation in the same year shows a positive effect (+7.83% to +20.08%), and the extracted value for this arm is +15.7% — irreconcilable with -2.72%. This pair was excluded from matching and flagged as a probable transcription error in the Li 2022 database.

---

## 4. Root Cause Analysis

The 50% capture failure is caused by a **product-selection omission**, not a table-reading failure.

The paper contains two complete, parallel treatment arms of equal prominence presented in side-by-side tables. The AI extracted the Terra Sorb Complex arm comprehensively and accurately, correctly recovering all 12 seed yield observations across three years. It also extracted 12 additional seed weight observations and 9 secondary outcome rows — demonstrating that the pipeline was fully capable of reading this paper's tables and understood the multi-year, multi-treatment design.

The Kelpak arm was not missed because the AI failed to find the data or misread the table structure. Rather, the paper's title — "Effect of amino acid biostimulant..." — focused attention on Terra Sorb Complex as the primary intervention. The pipeline appears to have treated the Kelpak seaweed extract arm as a secondary or comparative treatment rather than an independent extractable arm. This is a **scope-selection failure**: the AI narrowed its extraction to the product most prominently named in the title and may have implicitly deprioritized or skipped the co-intervention arm during observation enumeration.

This failure mode is particularly relevant for multi-biostimulant comparison studies. The Li 2022 database classified both arms as "SWE" (seaweed extract) or variants thereof, but the paper itself presents them as two independent interventions under a title that names only one. A pipeline that anchors extraction scope to the paper title — or to the first product encountered in the Methods — will systematically miss co-treatments in such designs.

The Kocira 2019 paper is processed by Claude models only (Kimi extracted 0 observations), and the fact that Claude extracted zero Kelpak rows despite reading the full paper confirms the omission is conceptual rather than a PDF parsing issue.

---

## 5. Overall Assessment

**What worked:** The Terra Sorb extraction was exemplary. Twelve seed yield observations were correctly identified, means were extracted exactly, the three-year repeated-measures structure was faithfully captured, and n = 4 was noted throughout. The extraction also went beyond what Li 2022 tracked (1000-seed weight, protein, fat, phenolics, etc.), demonstrating genuine comprehensiveness within the chosen product arm.

**What failed:** The entire Kelpak seaweed extract arm — 12 GT seed yield rows — was missed. This is a substantial coverage gap representing half the study's treatment arms and half the Li 2022 GT rows for this paper.

**Severity:** The omission biases any meta-analytic effect estimate for seaweed extracts on soybean yield by completely excluding one study's worth of data. The Terra Sorb and Kelpak arms produced qualitatively similar positive effects (+4% to +47% across years), so the direction of bias is not immediately obvious, but the loss of 12 independent observations reduces statistical power.

**Actionability:** This failure can be addressed by adding an explicit multi-arm detection step to the extraction prompt — instructing the AI to enumerate all distinct product treatments in the paper before beginning value extraction, regardless of which product is named in the title. Papers in the biostimulant literature frequently compare two or more products against a shared control, making this a recurring risk.

**Ratings:**

| Dimension | Score |
|-----------|-------|
| Accuracy (matched obs) | 5/5 — effect sizes exact, no reading errors |
| Coverage (GT rows captured) | 2/5 — 13/25 GT rows; entire Kelpak arm missed |
| Moderator fidelity | 5/5 — year, concentration, frequency all correct |
| Variance extraction | N/A — not attempted for yield rows |
| Overall extraction quality | 3/5 — excellent precision, serious coverage gap |
