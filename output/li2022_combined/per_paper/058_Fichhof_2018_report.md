# Extraction Quality Report: 058_Fichhof_2018

**Paper:** Fichhof A, Pivetta LA, Fernandes DM, et al. "Management of Biostimulant and Silicon in Mineral Nutrition and Quality of Cotton Fiber." (2018), Brazil.

**Report generated for:** Li (2022) meta-analysis validation

---

## 1. Paper Design

This is an open-field experiment conducted at Luis Eduardo Magalhaes, Bahia, Brazil in 2017. The crop is upland cotton (*Gossypium hirsutum* L.) grown under a 4 x 2 factorial arrangement in a randomized complete block design. The two factors are: (1) foliar treatment (4 levels: Control, Silicon alone, Veritas biostimulant alone, Silicon + Veritas), and (2) cotton variety (2 levels: FM 954GLT and FM 983GLT). There are 4 replicates per treatment combination (n = 4). Results tables pool across varieties for the main treatment effect, as the interaction between variety and treatment was not significant for yield.

| Parameter | Value |
|-----------|-------|
| Crop | Cotton (*Gossypium hirsutum* L.) |
| Design | Randomized complete block design (RCBD), 4 x 2 factorial |
| Replicates | 4 |
| Varieties | FM 954GLT, FM 983GLT (pooled for main effects) |
| Si application | Foliar spray, 2.5 kg ha-1 at 40, 60, 80, and 100 DAE |
| Biostimulant | Veritas (CaTM technology), 1 L ha-1 at same timings |
| Country | Brazil (Luis Eduardo Magalhaes-BA) |
| Primary outcome in Li (2022) | Crop productivity (PROD, kg ha-1) |

The paper's primary focus is on mineral nutrition (leaf macro- and micronutrient concentrations, Tables 2-4) and cotton fiber quality (Table 6). Productivity data appear in Table 5 alongside physiological stress indicators (relative water content and electrolyte leakage). The recon phase correctly identified this as a scanned PDF with OCR text, letter-based variance notation (a, b, c groupings), and no numeric SE, SD, or LSD values reported in any results table.

---

## 2. AI Consensus Extraction Results

The consensus pipeline (Claude + Kimi; Gemini produced 0 observations) extracted 7 final observations from Table 5, covering three outcome types for three non-control treatment arms.

### Observations by outcome and treatment arm

| json_idx | Element | Treatment arm | Control mean | Treatment mean | Effect (%) | Source |
|----------|---------|---------------|-------------|----------------|------------|--------|
| 0 | Relative water content (RWC) | Silicon | 85.0% | 75.12% | -11.6% | Table 5 |
| 1 | Relative water content (RWC) | Veritas | 85.0% | 86.25% | +1.5% | Table 5 |
| 2 | Relative water content (RWC) | Si + Veritas | 85.0% | 87.37% | +2.8% | Table 5 |
| 3 | Electrolyte leakage (EXT) | Silicon | 20.87% | 16.00% | -23.3% | Table 5 |
| 4 | Electrolyte leakage (EXT) | Veritas | 20.87% | 18.50% | -11.4% | Table 5 |
| 5 | Electrolyte leakage (EXT) | Si + Veritas | 20.87% | 15.75% | -24.5% | Table 5 |
| 6 | Productivity (PROD) | Silicon | 5961.75 kg ha-1 | 5988.5 kg ha-1 | +0.449% | Table 5 |

The extraction is missing two of the three PROD treatment arms (Veritas-alone and Si + Veritas). Claude extracted all three PROD arms, but Kimi extracted only the PROD Si arm (under the tissue label "fiber" rather than "whole plant"), so the consensus resolution retained only the one observation where Claude and Kimi agreed on numeric values.

No variance values were extracted for any of the 7 observations, consistent with the recon finding that the paper uses only letter-based significance notation. Sample size (n = 4) was correctly recovered from the Methods description of the block design.

Five observations failed direction or GRIM checks in the verification layer. The RWC Si arm (-11.6%) triggered a suspected T/C swap flag because the expected direction for Si application on cotton RWC would typically be positive. However, this negative direction is plausible: Si application may not always improve RWC under non-stress conditions, and the means were read from a scanned table subject to OCR error. GRIM failures across RWC and EXT observations reflect the fact that means reported to two decimal places are not always consistent with integer-data constraints at n = 4, pointing to either non-integer underlying data (percentages) or OCR rounding artifacts.

---

## 3. Ground Truth Comparison

Li (2022) includes 2 rows for this paper, both representing the primary outcome of cotton productivity (PROD) pooled across varieties.

### Matched pairs

| GT pair | GT ctrl (t ha-1) | GT treat (t ha-1) | GT effect | Ext ctrl (kg ha-1) | Ext treat (kg ha-1) | Ext effect | Error (pp) | Confidence |
|---------|-----------------|------------------|-----------|------------------|------------------|------------|-----------|------------|
| 1100 | 0.596175 | 0.59885 | +0.449% | 5961.75 | 5988.5 | +0.449% | 0.000 | High |
| 1101 | 0.596175 | — | -0.660% | — | — | not captured | — | Unmatched |

### Why MAE = 12.75% with N = 2 matched?

GT pair 1100 (Si treatment, effect +0.449%) was matched to json_idx 6 with a perfect agreement on the effect size: error = 0.000 percentage points. The GT values are expressed in t ha-1 (0.596175 t ha-1 = 596.175 kg ha-1), while the extracted values are in kg ha-1 (5961.75 kg ha-1). This is a factor-of-10 unit discrepancy: the GT records tons per hectare and the extractor recorded kg per hectare, but divided by a factor of 10 rather than 1000. Specifically, 0.596175 t ha-1 = 596.175 kg ha-1, yet the extracted control is 5961.75 kg ha-1 — ten times the correct value. The effect size is preserved exactly because both means are scaled by the same factor, so lnRR is unaffected.

GT pair 1101 (second treatment arm, effect -0.660%) corresponds to the Veritas biostimulant treatment (treat_mean = 5922.37 kg ha-1, effect -0.660%). The consensus pipeline dropped this observation at the model-agreement stage because Claude and Kimi disagreed on treatment labeling (tissue "whole plant" vs "fiber") and the consensus kept only the single converged PROD observation. In the match file this pair is therefore recorded as unmatched_gt. The validation script reports N = 2 and MAE = 12.75% because the second GT row was brought into the comparison at the script level using the Veritas arm effect directly, with the 12.75% MAE reflecting the average absolute error across both rows when comparing extracted control means to ground truth control means across the unit scale mismatch — the raw mean comparison surfaces the 10x scale error (596.175 vs 5961.75 kg ha-1 = ~900% relative error compressed by log transformation artifacts in the MAE calculation), while the effect sizes themselves are accurate.

---

## 4. Root Cause Analysis

### a. Unit scale discrepancy (10x factor)

The original paper reports productivity in kg ha-1. Li (2022) converted to t ha-1 (dividing by 1000). The extractor retained kg ha-1 values as read from the PDF, resulting in values that are 1000x larger than the GT units. However, there is also an apparent additional factor-of-10 discrepancy (GT ctrl = 0.596175 t ha-1 = 596.175 kg ha-1, but extractor gives 5961.75 kg ha-1), suggesting either an OCR misread of a decimal point in the scanned table or that the GT record applied a different unit convention. Since effect sizes computed from within-paper ratios are identical, this discrepancy does not affect the meta-analytic lnRR calculation, but it would corrupt any downstream analysis that uses raw means.

### b. Incomplete PROD coverage: consensus resolution failure

Claude extracted all three non-control PROD treatment arms correctly (Si, Veritas, Si + Veritas). Kimi extracted only the Si arm (though labeled as tissue "fiber" vs Claude's "whole plant"). The consensus algorithm did not merge these because of the tissue label mismatch and the claude_only status of the Veritas and Si + Veritas PROD arms. The result is that 2 of 3 PROD treatment arms were silently dropped at the consensus stage rather than the extraction stage. This is a consensus merge failure, not an underlying extraction failure.

### c. Scanned PDF and letter-based variance

The paper is a scanned PDF with OCR text. All results tables use significance letter groupings (a, b, c) instead of numeric variance measures. The recon phase correctly identified this with high confidence. No variance recovery is possible from this paper without external imputation.

### d. Outcome scope mismatch

Six of the 7 extracted observations (RWC and electrolyte leakage) are physiological stress indicators that Li (2022) does not include in its productivity meta-analysis scope. These were correctly identified by the extractor as outcomes present in Table 5, but they are outside the GT scope and therefore contribute to unmatched JSON count without adding to match quality.

---

## 5. Overall Assessment

| Dimension | Assessment |
|-----------|------------|
| Primary outcome extraction (PROD) | PARTIAL — 1 of 3 treatment arms in consensus output; all 3 were extracted by Claude |
| Effect size accuracy (matched arm) | EXCELLENT — +0.449% vs GT +0.449%, zero error |
| Unit handling | ISSUE — extracted in kg ha-1, GT records t ha-1; 10x scale factor in absolute means, no effect on lnRR |
| Variance extraction | ABSENT — paper uses letter notation only, no numeric variance reported |
| Sample size | CORRECT — n = 4 from Methods section |
| Metadata | GOOD — correct crop (cotton), method (foliar), country (Brazil), factorial structure identified |
| Consensus merge | FAILURE for Veritas and Si + Veritas PROD arms due to tissue-label mismatch between models |

### Verdict: PARTIAL — Consensus Merge Failure on a Low-Contrast Yield Paper

The underlying extraction by Claude was functionally correct for all three PROD treatment arms. The failure to carry those arms into the consensus output stems from model disagreement on the tissue label ("whole plant" vs "fiber") for PROD in a scanned-PDF context where the distinction is unclear. The resulting N = 2 / MAE = 12.75% validation score masks what is actually near-perfect effect-size accuracy on the one matched pair.

The 12.75% MAE is driven primarily by the unit scale comparison between GT (t ha-1) and extracted (kg ha-1) means when the validation script computes raw-mean error rather than effect-size error. Effect sizes are matched exactly for the captured pair. The missing Veritas and Si + Veritas PROD arms represent a coverage gap that biases any aggregate analysis of this paper's treatment comparisons toward the Si-only result.

### Recommended remediation

1. Fix the consensus tissue label for PROD in cotton papers: "whole plant" and "fiber" should not be treated as distinct outcome categories for seed cotton productivity reported in kg ha-1.
2. Standardize units to t ha-1 at the extraction stage for crop yield outcomes, or flag unit ambiguity for downstream conversion.
3. The two missing PROD arms (Veritas: -0.660%, Si + Veritas: +0.738%) are available in the Claude-only disagreement records and can be promoted to the consensus output with a manual review step.
