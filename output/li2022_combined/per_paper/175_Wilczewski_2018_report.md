# Per-Paper Extraction Quality Report: 175_Wilczewski_2018

**Paper:** Wilczewski, E., et al. (2018). Response of sugar beet to humic substances and foliar fertilization with potassium.
**Dataset:** Li 2022 validation
**Report generated:** 2026-02-18

---

## 1. Paper Design

This is a three-year (2006–2008) open-field experiment conducted at Chrząstowo, Poland, on sugar beet (*Beta vulgaris* L., cultivar Lubelska). The design is a randomized block with 4 replications (29.7 m² plots) and 4 treatment arms:

- **Control** — no amendments applied
- **Humistar** — leonardite-derived humic substance (12% humic acids, 3% fulvic acids) applied to soil at 40 dm³/ha before sowing
- **Drakar** — foliar potassium fertilizer (31% K₂O, 3% N) applied twice at 3 dm³/ha during the 6–12 leaf stage
- **Humistar + Drakar** — combined application of both

For biostimulant meta-analysis, the critical distinction is that **only Humistar qualifies as a biostimulant**. Drakar is a conventional potassium fertilizer; its sole-treatment arm is not relevant to the meta-analysis. Accordingly, the two extractable comparisons are: (1) Humistar vs. Control and (2) Humistar+Drakar vs. Control.

The paper reports four outcome variables across two tables. Table 3 contains field yield of storage roots (Mg·ha⁻¹) and leaf yield (Mg·ha⁻¹), averaged across the three growing seasons. Table 7 contains sugar content in fresh roots (%) and biological sugar yield (Mg·ha⁻¹). Variance is reported as ± values; the table footnotes reference letter-based significance grouping at P<0.05 (Tukey-type), which the recon module correctly flagged as ambiguous — the ± values most likely represent SE, but the paper does not state this explicitly. Variance confidence is rated "medium."

---

## 2. AI Extraction Results

Both Claude and Kimi extracted exactly 8 observations — 4 outcome variables × 2 treatment comparisons — with no observations from Gemini (Gemini returned 0 observations, likely excluded during the recon or extraction phase).

**Claude extracted (all 8 classified as `claude_only`):**

| Outcome | Table | Treatment | Trt mean | Ctrl mean | Trt SE | Ctrl SE | Effect % |
|---------|-------|-----------|----------|-----------|--------|---------|----------|
| Yield of storage roots [Mg·ha-1] | 3 | Humistar | 77.7 | 75.3 | 1.35 | 1.02 | +3.19% |
| Yield of storage roots [Mg·ha-1] | 3 | Humistar+Drakar | 78.2 | 75.3 | 1.42 | 1.02 | +3.85% |
| Yield of leaves [Mg·ha-1] | 3 | Humistar | 40.4 | 39.4 | 1.65 | 1.62 | +2.54% |
| Yield of leaves [Mg·ha-1] | 3 | Humistar+Drakar | 41.3 | 39.4 | 1.58 | 1.62 | +4.82% |
| Content of sugar in fresh roots [%] | 7 | Humistar | 17.2 | 17.1 | 0.48 | 0.46 | +0.58% |
| Content of sugar in fresh roots [%] | 7 | Humistar+Drakar | 17.1 | 17.1 | 0.39 | 0.46 | 0.00% |
| Biological yield of sugar [Mg·ha-1] | 7 | Humistar | 13.4 | 12.8 | 0.36 | 0.27 | +4.69% |
| Biological yield of sugar [Mg·ha-1] | 7 | Humistar+Drakar | 13.3 | 12.8 | 0.36 | 0.27 | +3.91% |

Claude labeled element names with full Polish-journal-style notation including units in brackets (e.g., `"Yield of storage roots [Mg·ha-1]"`, `"Content of sugar in fresh roots [%]"`). All observations assigned `n=4`, `variance_type="SE"`, moderators: site=Chrząstowo, years=2006–2008, soil_type="Mesic Typic Hapludalfs". Confidence rated "high" by Claude.

**Kimi extracted (all 8 classified as `kimi_only`):**

Kimi extracted the same 8 observations with numerically identical means, variances, n, and units. The values are byte-for-byte equal to Claude's. However, Kimi used truncated element name strings without the bracketed unit suffixes:

- `"Yield of storage roots"` (vs. Claude's `"Yield of storage roots [Mg·ha-1]"`)
- `"Yield of leaves"` (vs. `"Yield of leaves [Mg·ha-1]"`)
- `"Content of sugar in fresh roots"` (vs. `"Content of sugar in fresh roots [%]"`)
- `"Biological yield of sugar"` (vs. `"Biological yield of sugar [Mg·ha-1]"`)

Kimi also added a `cultivar` moderator field ("Lubelska") and a `location` field ("Chrząstowo, Poland"), whereas Claude stored site and years as separate moderator keys. Kimi confidence was rated "medium."

**The disagreement is purely in string representation, not in data content.**

---

## 3. Ground Truth Analysis

The Li 2022 meta-analysis focuses on biostimulant effects on crop yield. Wilczewski 2018 fits squarely within scope: Humistar is a humic acid biostimulant, sugar beet is a major European arable crop, and the study reports three-year replicated field data. The most likely ground truth rows in the Li 2022 dataset would be derived from:

- Storage root yield, Humistar vs. Control: +3.19% effect (77.7 vs. 75.3 Mg·ha⁻¹)
- Biological sugar yield, Humistar vs. Control: +4.69% effect (13.4 vs. 12.8 Mg·ha⁻¹)

The Humistar+Drakar combination arms may or may not be included in the Li 2022 ground truth, depending on whether the meta-analysts chose to include combined-treatment arms or only pure-biostimulant arms. Leaf yield (Table 3) and sugar content (Table 7, %) are secondary outcomes and may also appear. The validation match file (`validation_matches.csv`) shows zero matched observations for this paper, so no direct comparison against Li 2022 ground truth values was possible.

---

## 4. Root Cause Analysis

The consensus engine matched on the combination of `element` name string, `tissue`, and numerical means. All 8 Claude observations were flagged `claude_only` and all 8 Kimi observations were flagged `kimi_only`, producing 0 matched pairs and triggering the tiebreaker with the message: *"Low consensus: 0/8 (0%) — Claude=8, Kimi=8."* Because neither model reached a majority share of matched observations, the tiebreaker was not applied and 0 consensus observations were emitted.

The matching failure is attributable entirely to element name normalization. The consensus engine performs exact or near-exact string matching on `element`. Claude used the full column header from the paper including units (e.g., `"Yield of storage roots [Mg·ha-1]"`), while Kimi stripped the bracketed unit suffix and returned only the bare outcome label (e.g., `"Yield of storage roots"`). These strings do not match under exact comparison.

Crucially, if the unit suffix had been stripped before matching — or if a normalized form such as the first 20 characters or a lowercased tokenized comparison were used — all 8 pairs would have matched with zero numerical difference and yielded 8 consensus observations rated at 100% agreement. The extraction engines themselves performed correctly and in full agreement.

A secondary contributing factor is the different moderator key structure (Claude used `"years"` and `"site"` keys; Kimi used `"year"` and `"location"`), though moderator fields are typically not part of the primary matching key, so this alone would not have caused the failure.

---

## 5. Overall Assessment

**Extraction quality: Correct.** Both Claude and Kimi independently read the paper, identified the correct target tables (Tables 3 and 7), correctly excluded the Drakar-only arm as a non-biostimulant treatment, extracted the correct 4 outcome variables for both biostimulant comparison arms, and produced numerically identical means, variances, n, and units.

**Consensus quality: Failed.** The consensus layer produced 0 output observations from 8 correct and mutually consistent inputs. This is a false negative generated by a string-matching mismatch at the element name level — specifically, inclusion vs. exclusion of bracketed unit suffixes in the element label string. No data was lost at the extraction stage; the loss occurred entirely in the aggregation step.

**Recommended fix:** Implement element name normalization in the consensus matching function. Stripping unit suffixes enclosed in parentheses or brackets (e.g., `[Mg·ha-1]`, `(%)`) before comparison, or using a token-overlap similarity metric (e.g., Jaccard similarity on whitespace-tokenized strings with threshold ≥ 0.7), would have matched all 8 pairs correctly in this case and would likely recover similar failures across other papers in the dataset.

**Validation outcome:** 0 observations contributed to the Li 2022 validation from this paper, not because extraction failed, but because the consensus aggregation layer discarded all correctly extracted data due to label format inconsistency between models.
