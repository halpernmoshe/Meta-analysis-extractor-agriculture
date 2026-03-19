# Extraction Quality Report: 63_Mosavian_2021
**Match summary:** no_gt

---

## 1. Paper Design

**Citation:** Mosavian, S.N., Eisvand, H.R., Akbari, N., Moshatati, A., Ismaili, A., 2021. Do nitrogen and zinc application alleviate the adverse effect of heat stress on wheat (*Triticum aestivum* L.)? *Notulae Botanicae Horti Agrobotanici Cluj-Napoca*, 49(2), 12252.

**Study type:** Field experiment, Iran.

**Primary research question:** Whether combined nitrogen and zinc fertilization can mitigate heat-stress damage in wheat. Heat stress is induced by delayed planting dates (not CO2 manipulation).

**Experimental design:** Split-split plot, randomized complete block, 4 replications.

**Factor structure (3-way factorial):**
- Planting date: 3 levels (induces heat stress)
- Nitrogen rate: 4 levels (0, 75, 150, 225 kg N ha-1)
- Zinc rate: 3 levels (0, 10, 20 kg Zn ha-1 as ZnSO4)

**Intervention relevant to Hui 2023 meta-analysis:** Soil zinc application (ZnSO4 at 10 or 20 kg Zn ha-1 versus 0 kg Zn ha-1 control).

**Soil characteristics (from MOESM5):**
- Available Zn: 0.013 mg kg-1 (severely Zn-deficient; well below the 0.5 mg kg-1 deficiency threshold)
- pH: 6.895
- Organic matter: 16.03 g kg-1

**PDF note:** Scanned PDF with OCR; variance reported only as Duncan letter groupings (no numeric LSD or SE values published).

**Outcome variables in paper (Table 8):** Grain protein (%), grain zinc (mg kg-1), hectoliter weight (g). Grain yield (kg ha-1) is reported separately in Table 5/6.

---

## 2. AI Extraction

**Consensus pipeline:** Claude extracted 24 observations, Gemini extracted 28, Kimi extracted 0. Tiebreaker applied (Kimi = 0, Claude accepted). Final consensus: **12 observations** after matching.

**Elements extracted in consensus set:**

| Element | Tissue | N obs | Zn rates compared | Effect range |
|---|---|---|---|---|
| Zinc (mg kg-1) | grain | 8 | 10 vs 0, 20 vs 0 (4 N-level combinations) | +10.9% to +25.0% |
| Protein (%) | grain | 2 | 10 vs 0, 20 vs 0 (N=0 only) | +8.6%, +33.5% |
| Hectoliter weight (g) | grain | 2 | 10 vs 0, 20 vs 0 (N=0 only) | +1.8%, +2.1% |

**Extracted grain Zn values (consensus, optimal planting date):**

| Treatment | Control Zn (mg/kg) | Treatment Zn (mg/kg) | Effect (%) |
|---|---|---|---|
| 10 kg Zn, N=0 | 1.11 | 1.42 | +27.9% |
| 20 kg Zn, N=0 | 1.11 | 1.32 | +18.9% |
| 10 kg Zn, N=75 | 1.29 | 1.43 | +10.9% |
| 20 kg Zn, N=75 | 1.29 | 1.52 | +17.8% |
| 10 kg Zn, N=150 | 1.39 | 1.66 | +19.4% |
| 20 kg Zn, N=150 | 1.39 | 1.68 | +20.9% |
| 10 kg Zn, N=225 | 1.52 | 1.80 | +18.4% |
| 20 kg Zn, N=225 | 1.52 | 1.90 | +25.0% |

All 8 grain Zn observations show positive effects (Zn fertilization increases grain Zn), consistent with biofortification expectations. All effects were also independently agreed upon by both Claude and Gemini (diff = 0.0%), indicating high extraction reliability.

**Variance:** All observations have `variance_type = "LSD"` but `treatment_variance = null` and `control_variance = null`. This is correct: the paper uses Duncan's multiple range test letters (a, b, c notation) exclusively, and no numeric LSD or SE values are published. No variance values can be recovered.

**Sample size:** n = 4 (replications), correctly extracted from the Methods section.

**Verification flags:** All 12 consensus observations failed the GRIM test. This is expected and not a sign of error: the values are means of continuous physiological measurements (mg kg-1, %, g), not means of integer-valued counts. GRIM applies only to data that are sums of integers, so these failures should be disregarded.

**Claude-only observations (12 additional, not in consensus):** Claude extracted Protein (%) and Hectoliter weight (g) for the N=75, 150, and 225 levels, which Gemini did not match. These were excluded from the consensus set by the voting rule. The Protein and Hectoliter data are real (Table 8 contains them), so these exclusions represent a conservative voting artifact rather than a true extraction error.

---

## 3. Why No GT?

**The MOESM5 Data 2 (Soil application) sheet contains 24 rows for study_id = 63** (observation IDs 582-605), covering all 3 planting dates x 4 N levels x 2 Zn treatment rates = 24 treatment observations.

**The grain Zn concentration column (col 33: "Grain Zn concentration (mg kg-1)") is NULL for all 24 rows.** Hui et al. included this study in the meta-analysis for the grain yield outcome only; grain Zn concentration was apparently not available or not extracted by Hui for this study.

The GT file (`gt_63_Mosavian_2021.txt`) was successfully generated and confirms this: all 24 rows are present, but the only non-null outcome columns are grain yield (col 23: "Grain yield (kg ha-1)") and its paired treatment column (col 24). No grain Zn, straw Zn, or shoot Zn data appear in the GT record for this study.

The `validate_hui2023_full.py` script correctly maps `63_Mosavian_2021` to `Data 2 Soil application: [63]` and annotates it with the comment `# 0 Grain Zn data in GT`, which is the source of the `no_gt` status assigned to this paper.

**Why did Hui et al. omit grain Zn from this study's GT record?**

Two explanations are plausible:

1. **Extremely low reported grain Zn values.** The AI extraction produced grain Zn values of 1.11-1.90 mg kg-1, which are roughly 10-20x lower than typical wheat grain Zn concentrations (20-40 mg kg-1). The recon module flagged this: "grain zinc content data appears in Table 8 but values are very low (1.1-1.9 mg kg-1) - verify units." This strongly suggests a units problem: the paper likely reports zinc content in a unit that the OCR or authors expressed in a non-standard way (e.g., mg 100g-1 instead of mg kg-1, or the scanned table introduced OCR errors that dropped a digit). Hui et al. may have detected the same anomaly and excluded the grain Zn data as implausible.

2. **Paper included for grain yield only.** The Hui 2023 meta-analysis used multiple outcome variables. The MOESM5 soil sheet records grain yield for all 24 observations (ranging 1839-5563 kg ha-1, biologically plausible). It is possible that Hui et al. extracted the paper solely for grain yield and did not process grain Zn from this study, either because the values were suspected of unit errors or because they were not needed to reach the grain Zn sample size for the meta-analysis.

---

## 4. Assessment

**Extraction quality: PLAUSIBLE but LIKELY UNIT ERROR in grain Zn values.**

The AI pipeline extracted grain Zn data from Table 8 correctly in the sense that it faithfully read the numbers from a difficult scanned PDF and the two independent models (Claude and Gemini) agreed exactly on all 8 grain Zn values. The extraction process itself appears sound.

However, the extracted grain Zn concentrations (1.11-1.90 mg kg-1) are implausibly low for wheat grain. Normal wheat grain Zn ranges from approximately 15 to 60 mg kg-1. The extracted values are approximately 20-fold too small. This discrepancy almost certainly reflects one of the following:

- The PDF table reports zinc in units of mg 100g-1 (= 0.01 mg g-1 = 10 mg kg-1), and the AI accepted the numerical values but applied the wrong unit denominator.
- The scanned OCR dropped a leading digit (e.g., reading "1.42" instead of "14.2" or "21.42").
- The paper's Table 8 reports a different Zn metric (e.g., Zn content per plant or per tiller rather than per unit grain mass).

**The `no_gt` status is correct and appropriate.** There is no grain Zn ground truth in MOESM5 for this paper, so validation is impossible. This paper cannot contribute to the accuracy statistics for the Hui 2023 validation set.

**Recommendation:** If grain Zn data from Mosavian 2021 are to be used in any downstream analysis, the original PDF Table 8 should be inspected manually to determine the correct units. If the true values are ~15-30 mg kg-1 (approximately 10-15x the extracted values), the effects of Zn fertilization (approximately +10% to +28%) would remain consistent with the extracted percentage changes, since both control and treatment values would be scaled equally. The direction and relative magnitude of the AI-extracted effects are therefore likely reliable even if the absolute concentrations require a unit correction.
