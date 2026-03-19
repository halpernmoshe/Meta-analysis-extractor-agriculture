# Extraction Quality Report: Li_2013

**Paper:** Li, M.H., Wang, Z.H., Wang, J.W., Mao, H., 2013. Effect of Zn application methods on wheat grain yield and Zn utilization in Zn-deficient soils of dryland. *Journal of Plant Nutrition and Fertilizer*, 19(6), 1346–1355. [Chinese journal, written in Chinese]

**Match summary:** 6/12 GT matched, r = 0.643, MAE = 16.38%

---

## 1. Paper Design

**What the GT expects:** A Chinese-language wheat fertilization study examining the effect of ZnSO4 application method (soil vs. foliar vs. soil+foliar) on grain Zn concentration in Zn-deficient dryland soils. The study crossed two site types (two available Zn levels: 0.45 and 0.74 mg kg⁻¹) with two N rates (180 and 300 kg N ha⁻¹), yielding 4 observations per application type (12 GT rows total). Key design features:

- **Species:** Wheat (*Triticum aestivum*)
- **Country:** China
- **Treatment variable:** Zn fertilizer application method (soil, foliar, soil+foliar), all using ZnSO4
- **Soil Zn rate:** ~11.35 kg Zn ha⁻¹ (soil); ~1.09 kg Zn ha⁻¹ (foliar)
- **N rates:** 180 and 300 kg N ha⁻¹
- **Site soil pH:** 8.2 (Zn-deficient site) and 7.9 (second site)
- **n = 4** replicates per observation
- **Outcome:** Grain Zn concentration (mg kg⁻¹), grain yield (kg ha⁻¹), Zn accumulation

**What the PDF actually contains:** The file `Li_2013.pdf` is a completely different paper:

> Impa, S.M., Morete, M.J., Ismail, A.M., Schulin, R., Johnson-Beebout, S.E. (2013). Zn uptake, translocation and grain Zn loading in rice (*Oryza sativa* L.) genotypes selected for Zn deficiency tolerance and high grain Zn. *Journal of Experimental Botany*, 64(10), 2739–2751. doi:10.1093/jxb/ert118

This is an IRRI greenhouse study growing multiple rice genotypes to maturity in agar nutrient solution (ANS) under Zn-deficient (0.005 µM ZnSO4) vs. Zn-sufficient (1.5 µM ZnSO4) conditions. It has three experiments, 10–12 rice genotypes, and Table 4 as the primary data source. It is entirely unrelated to the Hui 2023 meta-analysis scope (wheat, field Zn fertilization).

---

## 2. Grain Zn Data in PDF (What Tables Exist)

The PDF (Impa et al. 2013) contains the following data tables:

| Table | Content | Relevant to Hui 2023? |
|-------|---------|----------------------|
| Table 1 | Genotype descriptions, seed Zn, days to flowering/maturity | No |
| Table 2 | Biomass accumulation and Zn translocation in IR74 (Experiment 1, hydroponic) | No |
| Table 3 | Zn efficiency and leaf symptom scores (Experiments 2 and 3) | No |
| **Table 4** | **Panicle Zn concentration (50% flowering) and brown rice Zn concentration (maturity) for 8–12 rice genotypes × Zn-sufficient/deficient** | No — rice, hydroponic, not wheat field fertilizer |
| Table 5 | Correlation coefficients for brown rice Zn vs. tissue Zn | No |
| Table 6 | Average tissue Zn concentrations (Experiments 2 and 3, maturity) | No |
| Table 7 | Mass balance of Zn movement between tissues | No |

**There are no tables in this PDF showing wheat grain Zn from soil/foliar/soil+foliar ZnSO4 field applications.** The document does not contain any data matching the scope of Hui 2023 or the ground truth rows.

---

## 3. AI Consensus Extraction Results

The AI pipeline (Claude + Kimi, no Gemini output) correctly identified Table 4 as the primary data source and extracted brown rice Zn concentration as the outcome variable. The recon phase flagged this as a wrong-scope paper with multiple warnings.

**Recon warnings (correctly issued):**
- "This is a rice study, not wheat — may not be relevant to wheat Zn meta-analysis"
- "Uses agar nutrient solution (ANS) system, not traditional soil/foliar fertilizer application"
- "Zn treatments are solution concentrations (0.005 µM vs 1.5 µM), not fertilizer applications"
- "WARNING: SCANNED PDF — Text may have OCR errors"
- "WARNING: LETTER-VAR — Variance shown as letters (a,b,c), not numbers"

Despite these warnings, the paper was included in the extraction run and produced 48 consensus observations (claude_obs=52, kimi_obs=48, matched_obs=48, 1 T/C swap corrected, 0 duplicates removed).

**Consensus observations summary:**

| Experiment | Genotypes | Outcomes extracted | n per obs |
|-----------|-----------|-------------------|-----------|
| Experiment 2 | IR64, IR68144, IR74, IR82247, IR69428, IR75862, Joryoongbyeo, RIL-46, SWHOO | Panicle Zn, Brown rice Zn, Panicle weight, Grain weight × Zn-sufficient vs. Zn-deficient | 5 |
| Experiment 3 | A69-1, IR55179, IR69428 | Same four outcomes | 5 |
| Disagreements (claude-only) | KP (did not flower under Zn-deficient) | 4 claude-only records dropped | — |

**Representative extracted values (Brown rice Zn, mg kg⁻¹):**

| Genotype | Experiment | Extracted Treatment | Extracted Control | Effect % |
|----------|-----------|---------------------|-------------------|----------|
| IR64 | Exp 2 | 26 | 17 | +52.9% |
| IR68144 | Exp 2 | 24 | 12 | +100.0% |
| IR74 | Exp 2 | 21 | 16 | +31.3% |
| SWHOO | Exp 2 | 38 | 32 | +18.8% |
| IR69428 | Exp 2 | 35 | 23 | +52.2% |
| A69-1 | Exp 3 | 32 | 10 | +220.0% |
| IR55179 | Exp 3 | 20 | 7 | +185.7% |

**All 48 observations are from a rice hydroponic experiment and have no relevance to the Hui 2023 wheat fertilizer meta-analysis.**

---

## 4. Ground Truth (MOESM5) Data — All 3 Sheets

The ground truth for "Li_2013" spans three sheets of MOESM5_dataset.xlsx, covering all three Zn application methods. All 12 rows refer to the same Chinese wheat field study (Li, M.H. et al. 2013, *Journal of Plant Nutrition and Fertilizer*) and share the same experimental structure: 2 sites (soil Zn 0.45 vs. 0.74 mg kg⁻¹) × 2 N rates (180 vs. 300 kg N ha⁻¹).

### Sheet: Data 2 — Soil Application (study_id = 19, obs IDs 113–116)

| Obs ID | Site Zn (mg kg⁻¹) | N rate (kg ha⁻¹) | Grain Zn (mg kg⁻¹) | Grain yield (kg ha⁻¹) |
|--------|-----------------|-----------------|--------------------|-----------------------|
| 113 | 0.45 | 180 | **27.9** | 5356 |
| 114 | 0.45 | 300 | **28.6** | 5764 |
| 115 | 0.74 | 180 | **28.6** | 6874 |
| 116 | 0.74 | 300 | **28.7** | 6986 |

Zn rate: 11.35 kg Zn ha⁻¹ as ZnSO4 (soil applied). n = 4. pH = 8.2 / 7.9.

### Sheet: Data 3 — Foliar Application (study_id = 10, obs IDs 73–76)

| Obs ID | Site Zn (mg kg⁻¹) | N rate (kg ha⁻¹) | Grain Zn (mg kg⁻¹) | Grain yield (kg ha⁻¹) |
|--------|-----------------|-----------------|--------------------|-----------------------|
| 73 | 0.45 | 180 | **27.9** | 5356 |
| 74 | 0.45 | 300 | **28.6** | 5764 |
| 75 | 0.74 | 180 | **28.6** | 6874 |
| 76 | 0.74 | 300 | **28.7** | 6986 |

Zn rate: 1.09 kg Zn ha⁻¹ as ZnSO4 (foliar, 2 sprays, 0.0908 g Zn L⁻¹). n = 4.

### Sheet: Data 4 — Soil+Foliar Application (study_id = 3, obs IDs 7–10)

| Obs ID | Grain yield (kg ha⁻¹) | Grain Zn (mg kg⁻¹) | Straw biomass (kg ha⁻¹) |
|--------|----------------------|--------------------|------------------------|
| 7 | 5356 | **27.9** | 5717 |
| 8 | 5764 | **28.6** | 5995 |
| 9 | 6874 | **28.6** | 7752 |
| 10 | 6986 | **28.7** | 7148 |

Note: The Data 4 sheet provides fewer moderator columns (no separate N rate or site Zn fields visible), but grain Zn concentrations are identical to the corresponding soil and foliar treatments — suggesting that across all three application methods in this study, the grain Zn concentrations at the treatment level were essentially equal (27.9–28.7 mg kg⁻¹).

**GT grain Zn range across all 12 rows: 27.9–28.7 mg kg⁻¹ — a narrow 0.8 mg kg⁻¹ spread.**

---

## 5. Root Cause Analysis

### Primary cause: Mislabeled PDF — Complete paper identity mismatch

The fundamental problem is that **the file named `Li_2013.pdf` contains a completely different paper** (Impa et al. 2013, rice hydroponic). The correct paper — Li, M.H. et al. 2013, *Journal of Plant Nutrition and Fertilizer* — is a Chinese-language article and was never downloaded to the PDF folder.

**Consequence:** The AI extracted 48 observations from the wrong paper (rice hydroponic) that have no relationship to the 12 GT rows (wheat field fertilization). No genuine match is possible.

### Why 6/12 appear "matched" at r = 0.643, MAE = 16.38%

The validation algorithm matched 6 rows by finding numerical coincidences between the two completely unrelated datasets:

- GT grain Zn values cluster tightly at **27.9–28.7 mg kg⁻¹**
- The extracted Impa et al. rice data contains brown rice Zn values ranging from 7 to 38 mg kg⁻¹, including several genotypes in the 20–35 mg kg⁻¹ range
- The matching algorithm found 6 AI extraction rows whose treatment or control means were numerically close to the GT wheat Zn values (~27–29 mg kg⁻¹)

These are **spurious numerical coincidences**. The r = 0.643 and MAE = 16.38% reflect how well randomly overlapping numbers from an unrelated rice study happen to align with wheat GT values — not genuine extraction quality.

The 6 "matched" rows are presumably rice genotype observations where brown rice Zn under Zn-sufficient conditions (e.g., SWHOO: 38, Joryoongbyeo: 33, IR64: 26, IR74: 21) or Zn-deficient conditions happened to fall numerically near the wheat GT values of 27.9–28.7 mg kg⁻¹. The 6 unmatched rows are GT observations for which no AI-extracted value fell within the matching tolerance.

### Why the 12 GT rows could never be matched from this PDF

The GT data structure requires matching on:
1. Grain Zn concentration of wheat (27.9–28.7 mg kg⁻¹)
2. Zn application method (soil / foliar / soil+foliar)
3. Site (soil Zn 0.45 vs. 0.74 mg kg⁻¹)
4. N rate (180 vs. 300 kg N ha⁻¹)

None of these dimensions exist in Impa et al. 2013. The PDF has no wheat data, no field fertilizer applications, no site soil Zn treatments, and no N rate treatments. Six of the 12 GT rows (50%) happened to fall within the numerical range of rice Zn values extracted; the other 6 did not.

### Secondary issue: Recon warnings were ignored

The recon stage correctly issued 7 warnings identifying this as out-of-scope (wrong species, wrong experimental system, hydroponic not field). The pipeline should have **excluded** this paper from the extraction run based on these warnings. Instead, it proceeded to extraction and generated 48 useless observations.

### Tertiary issue: No variance extracted

All 48 consensus observations have `variance_type: null` and `treatment_variance: null`. The paper uses letter-based significance notation (a, b, c groupings per LSD) rather than numeric SE/SD values in the tables, consistent with the recon warning "LETTER-VAR." Even if this had been the correct paper, variance would have been unrecoverable.

---

## 6. Overall Assessment

**Extraction quality: INVALID — Wrong paper (complete PDF mislabeling)**

| Dimension | Status |
|-----------|--------|
| Paper identity | WRONG — file is Impa et al. 2013 (rice/hydroponic), not Li M.H. et al. 2013 (wheat/field) |
| Capture rate | 6/12 (50%) — spurious numerical coincidences only |
| r = 0.643 | Meaningless — artefact of chance numerical overlap |
| MAE = 16.38% | Meaningless — artefact of chance numerical overlap |
| Variance extracted | None (0/48) — letter notation in source |
| AI recon warning quality | Good — correctly flagged as out-of-scope, but warnings not acted upon |
| Consensus agreement | High (claude+kimi agree on 48/48 matched obs) — but both models read the same wrong paper |

**What needs to happen:**

1. **Locate the correct PDF**: Li, M.H., Wang, Z.H., Wang, J.W., Mao, H. (2013). *Journal of Plant Nutrition and Fertilizer*, 19(6), 1346–1355. This is a Chinese-language article. DOI or source unknown from available files. It may require access to CNKI (China National Knowledge Infrastructure) or a Chinese agricultural database.

2. **Replace the PDF** in the source folder with the correct paper, or exclude Li_2013 from the dataset and mark the 12 GT rows as unrecoverable.

3. **Implement scope exclusion**: Papers where recon issues 2+ out-of-scope warnings (wrong species, wrong experimental system) should be auto-excluded from extraction to avoid wasting API calls and polluting results.

4. **Treat the 6 "matched" observations as invalid** in the validation statistics. The r and MAE figures for this paper are not informative and should be excluded from aggregate accuracy metrics.

**Impact on overall Hui 2023 validation metrics**: This paper contributes 12 GT rows and 6 apparent matches to the aggregate statistics. Since all 6 matches are coincidental, both the numerator (matches) and denominator (GT rows) should ideally be removed from the r and MAE calculations for an accurate picture of true system accuracy.
