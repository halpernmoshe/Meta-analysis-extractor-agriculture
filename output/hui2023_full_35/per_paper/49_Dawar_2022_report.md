# Extraction Quality Report: 49_Dawar_2022
**Match summary:** no_gt (0 Grain Zn rows in MOESM5 for this study; GT contains only Soil application metadata)

---

## 1. Paper Design

**Full citation:** Dawar, K., Khan, N., Fahad, S., Alam, S.S., Khan, S., Mian, I.A., Akbar, W.A., 2022. Effect of sulfur and zinc nutrition on yield and uptake by wheat. *Journal of Plant Growth Regulation*, 41, 2338–2346.

- **Country:** Pakistan
- **Crop:** Wheat (*Triticum aestivum* L.), cv. Shahkar-2013
- **Site:** Swabi Agricultural Research Station (ARS), season 2015–2016
- **Design:** Field experiment, randomized complete block (RCB), n=3 replicates
- **Treatments:** Factorial: zinc rates (0, 2.5, 5.0, 10.0 kg Zn ha-1 as ZnO) × compost rates (0, 5, 10 t ha-1), 12 treatment combinations (T1–T12)
- **Control:** T1 = 0 compost + 0 Zn
- **Primary focus:** Agronomic yield, yield components, soil nutrient status, and shoot/soil Zn after Zn + compost co-application
- **Variance reporting:** Fisher's LSD test; data presented in figures (bar charts), no numeric variance values tabulated

**Outcome variables reported in paper:**
- Plant height, tillers/m2, spikes/m2, spike length (Figure 1)
- 1000-grain weight, biological yield, grain yield (Figure 2)
- Soil EC, soil organic matter (Figure 3)
- Soil mineral N, total soil N, soil extractable P (Figure 4)
- Shoot Zn concentration and accumulation, soil extractable Zn (implied figures/tables)

**Critically absent:** Grain Zn concentration (mg kg-1) is NOT reported anywhere in this paper. The study measures shoot Zn concentration (foliar) and soil extractable Zn, but not grain Zn biofortification.

---

## 2. AI Extraction

The AI recon phase correctly identified that this paper contains no grain Zn concentration data and issued an explicit do-not-extract warning:

> "This paper does NOT measure grain Zn concentration - only morphological traits and yield components."
> "DO NOT EXTRACT - This paper contains no grain zinc concentration data and is not relevant for zinc biofortification meta-analysis."

Despite the recon warning, the consensus pipeline extracted **13 observations** from figures via vision (all models extracted 0 tabular observations; matched_obs=13 implies vision-based figure reading was applied). The extracted variables were:

| Element / Variable | Tissue | Data Source | Effect (%) |
|--------------------|--------|-------------|------------|
| plant height | whole plant | Figure 1A | +14.9% |
| tillers/m2 | whole plant | Figure 1B | +55.3% |
| spikes/m2 | whole plant | Figure 1C | +58.1% |
| spike length | whole plant | Figure 1D | +68.3% |
| 1000-grain weight | grain | Figure 2A | +33.3% |
| biological yield | whole plant | Figure 2B | +17.8% |
| grain yield | grain | Figure 2C | +38.3% |
| soil EC | soil | Figure 3A | +52.2% |
| soil organic matter | soil | Figure 3B | +112.9% |
| soil mineral nitrogen | soil | Figure 4A | +121.7% |
| total soil nitrogen | soil | Figure 4B | +209.5% |
| soil extractable P | soil | Figure 4C | +100.0% |
| soil extractable Zn | soil | Figure 7B | +189.5% |

**Extraction notes:**
- All observations compare the best treatment (10 kg Zn ha-1 + 10 t compost ha-1) vs. control (0 Zn + 0 compost); intermediate Zn rates are not represented.
- No variance values were extracted (all null); variance_type recorded as LSD.
- 9 of 13 observations failed the GRIM test, consistent with figure-estimated (rounded) values from bar charts with n=3.
- 5 observations flagged for extreme effect magnitudes (>100%), which are plausible for soil nutrient variables when compost is co-applied but warrant caution as figure reads.
- Zero grain Zn concentration observations were extracted, consistent with the paper not reporting this outcome.

---

## 3. Why No GT?

The MOESM5 ground truth spreadsheet (Hui 2023) lists this paper under **"Data 2 Soil application"** (study_id=49) with 3 rows (Observation IDs 334, 335, 336). Examining those rows reveals:

- The GT rows contain **soil and agronomic moderator metadata** (soil pH, available Zn, organic matter, N/P/K rates, grain yield, straw biomass, shoot biomass, shoot Zn concentration, shoot Zn accumulation) alongside Zn rate and fertilizer type.
- The unnamed numeric column (values: 0.991, 1.346, 1.754) in the GT rows appears to be log response ratios or effect size values computed by Hui et al. from **shoot Zn** or **grain yield**, not grain Zn concentration.
- **There are zero Grain Zn concentration (mg kg-1) rows** for study_id=49 in the "Data 2 Soil application" sheet, and the paper is absent from the "Data 3 Foliar application" and "Data 4 Soil+Foliar application" sheets.

**Root cause of no_gt status:** Hui et al. 2023 included this paper for its **soil Zn availability and shoot Zn uptake data** (as contextual/moderator information), not because it reported grain Zn biofortification. The meta-analysis outcome variable for the Hui 2023 dataset is grain Zn concentration; this paper does not measure it. Therefore no matchable GT rows exist for our validation pipeline.

---

## 4. Assessment

**Extraction correctness:** The AI correctly identified this paper as out-of-scope for grain Zn biofortification at the recon stage. The 13 extracted observations (agronomic/soil variables) are technically coherent as figure reads but are irrelevant to the meta-analysis outcome. The extraction pipeline should have suppressed output for this paper given the explicit recon guidance.

**GT absence explanation:** Legitimate. Dawar et al. 2022 does not report grain Zn concentration. Hui et al. used it only as a data source for soil Zn availability context (the "Data 2 Soil application" sheet records the soil zinc status and agronomic covariates, with the 3 GT rows corresponding to the 3 Zn treatment levels—not to distinct grain Zn measurements).

**Validation impact:** This paper correctly contributes 0 matched observations to the Hui 2023 validation. The no_gt classification is accurate and expected; it does not reflect an extraction error.

**Recommendation:** No re-extraction warranted. If the pipeline is updated to enforce recon-level exclusion (do-not-extract flag), this paper should be excluded from the output JSON entirely to avoid cluttering results with irrelevant agronomic variables.
