# Extraction Quality Report: 62_Morshedi_2012
**Match summary:** no_gt

---

## 1. Paper Design (wheat grain Zn study? what outcomes?)

**Full citation:** Morshedi, A. & Farahbakhsh, H. (2012). The role of potassium and zinc in reducing salinity and alkalinity stress conditions in two wheat genotypes. *Archives of Agronomy and Soil Science*, 58(4), 371-384.

**Study description:** Greenhouse (2006) and field (2007) factorial experiment in saline/calcareous soils near Kerman, Iran. Two wheat genotypes (Baccrosroshan and Line No. 4) crossed with 4 K rates (K0=0, K1=72, K2=144, K3=216 kg K2O ha-1) and 3 Zn rates (Zn0=0, Zn1=20, Zn2=40 kg Zn ha-1 as ZnSO4). Randomized complete block design, n=3 replications.

**Outcome variables reported (Tables 2-5):**
- Ear length (cm)
- Number of grains per ear
- 1000-grain weight (g)
- Grain weight per ear (g)
- Grain yield (g pot-1 greenhouse; t ha-1 field)
- Grain protein content (%)

**Grain Zn concentration reported?** No. The paper does not measure or report grain Zn concentration (mg kg-1 or mg Zn grain-1) anywhere. The intervention uses ZnSO4 as a soil amendment under salinity stress, but no tissue mineral analysis is performed. The study focus is agronomic performance (yield and protein), not Zn biofortification.

**Soil Zn context:** Table 1 reports DTPA-extractable Zn of 0.6-0.7 mg kg-1 (Zn-deficient threshold), confirming Zn deficiency as the agronomic rationale, but no grain Zn measurements follow.

---

## 2. AI Extraction

**Recon result:** The recon model correctly identified that this paper does not measure grain Zn concentration and flagged it for exclusion:
> "This paper should be EXCLUDED from the Zn biofortification meta-analysis as it does not measure grain Zn concentration."

**tables_with_target_data:** [] (empty - correctly identified no usable tables)

**Extraction result:** Claude extracted 0 observations. Kimi extracted 46 observations, all of which are agronomic yield/weight variables (1000-grain weight, grain yield, grain weight per ear, grain protein content) from Tables 3 and 5 (the Zn x K factorial interaction tables).

**Consensus:** Kimi's 46 observations were accepted (tiebreaker: Claude=0, Kimi=46, Gemini=0). All 46 are yield-component variables, none is grain Zn concentration. Variance is null for all observations (DMR letter notation only, no numeric SE/SD/LSD reported).

**Sample of Kimi-extracted observations:**
- 1000-grain weight (g): treatment 19.9 vs control 18.1 (Baccrosroshan, Zn1K0, greenhouse, Table 3)
- 1000-grain weight (g): treatment 24.5 vs control 18.1 (Baccrosroshan, Zn2K0, greenhouse, Table 3)
- 1000-grain weight (g): treatment 31.3 vs control 25.9 (Line No. 4, Zn1K0, greenhouse, Table 3)

---

## 3. Why No GT?

The MOESM5 spreadsheet (Hui 2023 meta-analysis source data) places study_id=62 in the **Data 2 Soil application** sheet only. The 16 GT rows (Obs IDs 566-581) for this study contain only grain yield (kg ha-1) as the outcome variable. There are no grain Zn concentration rows for study_id=62 in any MOESM5 sheet (Soil, Foliar, or Soil+Foliar).

This is consistent with the paper itself: Hui et al. (2023) included Morshedi 2012 in their meta-analysis for the **grain yield response** to soil Zn application (which is what Data 2 tracks), not for grain Zn concentration. The Hui meta-analysis also tracked grain Zn as a separate outcome (Data 3 Foliar / Data 4), but Morshedi 2012 contributes no data to those sheets because the paper never measured grain Zn.

**Root cause of no_gt status:** The validation pipeline searched MOESM5 Data 2 Soil sheet for grain Zn concentration rows under study_id=62 and found none - correctly, because none exist. The paper is legitimately outside the grain Zn concentration sub-analysis.

---

## 4. Assessment

**Extraction correctness:** The recon correctly flagged this paper as lacking grain Zn data and recommended exclusion from the Zn biofortification analysis. Claude correctly extracted 0 grain Zn observations. Kimi's 46 extracted observations are real data from the paper but represent the wrong outcome variable (yield components, not grain Zn concentration); they should not be used in a grain Zn meta-analysis.

**no_gt status explanation:** Expected and correct. Hui et al. used this paper for its grain yield response data (Data 2 Soil sheet, 16 rows), not for grain Zn concentration. No grain Zn is reported in the paper.

**Action required:** None. This paper should remain excluded from grain Zn concentration analysis. The no_gt status is not a validation failure - it correctly reflects that the paper contributes no grain Zn data. If the validation pipeline needs to categorise such cases, this paper belongs in the class "paper in Hui Soil sheet for yield outcome only, no grain Zn measured."
