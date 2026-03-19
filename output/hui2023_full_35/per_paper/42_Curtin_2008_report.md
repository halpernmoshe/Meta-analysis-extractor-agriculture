# Extraction Quality Report: 42_Curtin_2008
**Match summary:** no_gt (0 grain Zn concentration rows in MOESM5 "Data 2 Soil application" sheet)

---

## 1. Paper Design (what Zn outcomes does it measure?)

**Citation:** Curtin, D., Martin, R.J., Scott, C.L. (2008). Wheat (*Triticum aestivum*) response to micronutrients (Mn, Cu, Zn, B) in Canterbury, New Zealand. *New Zealand Journal of Crop and Horticultural Science*, 36(3), 169–181.

**Design:** Multi-site field experiment across 22 Canterbury sites in Year 1 (2005–06) and 3 sites in Year 2 (2006–07). Randomised block (Year 1) and row-column (Year 2) designs, n=4 replicates. Micronutrient treatments applied as soil applications: Mn, Cu, Zn (as ZnSO4·7H2O at 4.4 kg Zn/ha), B, and combined (Cu+Mn+Zn). A lime factorial (0 vs 10 t/ha) was superimposed in Year 2.

**Outcomes reported:**
- Year 1 (22 sites): Grain yield only — no significant yield response to any micronutrient. No grain micronutrient concentrations reported.
- Year 2 (3 sites — Barrhill, Rakaia, Leeston): **Table 8** reports grain Cu, Mn, and Zn concentrations (mg/kg) for Control vs Cu+Mn+Zn combined treatment under both lime conditions.

**Key limitation:** Grain Zn concentration data exist only for Year 2 (3 sites) and only for the **combined Cu+Mn+Zn treatment** — not for a Zn-only treatment arm. The paper is scanned (OCR quality variable).

---

## 2. AI Extraction

The AI consensus pipeline (Claude + Kimi; Gemini produced 0 observations) extracted **18 observations** from Table 8, covering all three elements (Cu, Mn, Zn) across 3 sites × 2 lime conditions:

| Element | Sites | Lime levels | Obs | Effect range |
|---------|-------|-------------|-----|--------------|
| grain Cu concentration (mg/kg) | Barrhill, Rakaia, Leeston | No lime / Limed | 6 | +3% to +16% |
| grain Mn concentration (mg/kg) | Barrhill, Rakaia, Leeston | No lime / Limed | 6 | +1% to +17% |
| grain Zn concentration (mg/kg) | Barrhill, Rakaia, Leeston | No lime / Limed | 6 | +8% to +25% |

**Grain Zn observations extracted (6 total):**

| Site | Lime | Control (mg/kg) | Treatment (mg/kg) | Effect |
|------|------|-----------------|-------------------|--------|
| Barrhill | No lime | 29.8 | 34.0 | +14.1% |
| Barrhill | Limed | 28.3 | 31.3 | +10.6% |
| Rakaia | No lime | 14.8 | 17.0 | +14.9% |
| Rakaia | Limed | 13.0 | 16.3 | +25.4% |
| Leeston | No lime | 23.0 | 24.8 | +7.8% |
| Leeston | Limed | 22.0 | 24.0 | +9.1% |

**Treatment definition used by AI:** "Combined micronutrients (Cu+Mn+Zn)" vs "Control (micronutrients not applied)." The AI correctly identified this as the only available contrast for grain Zn in Table 8 and flagged (in recon) that a Zn-only treatment arm is not separately reported for grain concentrations.

**Variance:** LSD values extracted from Table 8 for most observations (treatment_variance reported for 4/6 Zn rows; 2 rows at Rakaia have null treatment_variance, likely OCR failure). Control variance consistently null — LSD is a single pooled value per site/lime combination, not per arm.

**Verification flags:** All 18 observations fail GRIM test (1-decimal means with n=4 are often mathematically impossible at 1 decimal place; likely a continuous measurement rounding issue, not an error). All flagged for variance_type mismatch (reported=LSD but CV heuristic suggests SD/SE) — this is expected for LSD-only papers and is a known limitation of the CV heuristic.

**Recon quality:** Excellent. The AI correctly identified Table 8 as the sole source of grain micronutrient data, noted Year 1 had no grain concentration data, flagged the scanned PDF risk, and warned that Cu+Mn+Zn is a combined treatment (not Zn-only). Confidence rating: HARD difficulty, hybrid extraction method.

---

## 3. Why No GT?

The MOESM5 "Data 2 Soil application" sheet records study_id=42 with **23 rows** but these rows contain only:
- Soil moderator columns: Available Zn (mg/kg), pH, CaCO3, OM, N/P/K rates
- Treatment: Zn fertilizer type (ZnSO4) and Zn rate (4.4 kg Zn/ha)
- Outcome: **Grain yield (kg/ha)** — present for 18/23 rows; absent for 5 rows

There is **no grain Zn concentration column** in the "Data 2 Soil application" sheet. The sheet documents the study's treatment structure and grain yield response for meta-regression purposes (moderator encoding), but Hui 2023 did not include grain Zn concentration from this paper in their dataset.

**Most likely reason:** The Hui 2023 meta-analysis required a **Zn-only soil application** as the treatment arm to define the standardized effect size (Zn treatment vs control). Curtin 2008 Table 8 reports grain Zn only for the **combined Cu+Mn+Zn treatment** — it does not isolate the effect of Zn alone on grain Zn concentration. Hui et al. therefore included the study in their dataset (23 rows for moderator structure and grain yield) but excluded it from the grain Zn concentration meta-analysis because the confounded treatment design does not permit a clean Zn-only effect estimate.

An alternative explanation (less likely) is that grain Zn concentration data exist in a different MOESM5 sheet (Data 3 Foliar or Data 4 Soil+Foliar) under a different study_id assignment, but the GT file confirms study_id=42 maps only to the Soil application sheet with 23 rows containing no grain Zn concentration variable.

---

## 4. Assessment

**Extraction quality: GOOD — correctly identifies available data; no_gt is expected and appropriate.**

The AI pipeline performed correctly on this paper:
- It correctly identified Table 8 as the only source of grain Zn data.
- It correctly extracted 6 grain Zn observations with plausible values (14.8–34.0 mg/kg, effects +8% to +25%) consistent with the paper's reported finding of ~13.5% average grain Zn increase.
- It correctly flagged the treatment confounding issue (combined Cu+Mn+Zn, not Zn-only).
- The 0-match to GT is not an extraction failure — it reflects a deliberate inclusion/exclusion decision by Hui et al. to omit confounded treatment arms.

**Implication for validation:** This paper should be classified as `no_gt_expected` (GT excludes it by design) rather than a missed extraction. The 6 grain Zn observations extracted by the AI are factually correct but represent a different contrast (combined treatment) than the Hui 2023 protocol required (Zn-only). Including this paper's data in a supplementary sensitivity analysis could be considered, noting the treatment confounding.

**Variance note:** LSD values are available from Table 8 for most observations. Conversion to SD requires: SD = LSD × sqrt(n) / (t_crit × sqrt(2)), with n=4 and t_crit at P=0.05 for df appropriate to the row-column design. The 2 Rakaia Zn rows with null variance likely reflect OCR failure on the scanned table and would require manual verification from the PDF.
