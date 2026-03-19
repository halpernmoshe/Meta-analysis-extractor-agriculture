# Extraction Quality Report: 52_Forster_2018

**Match summary:** 6/6 GT matched | r = 1.0 | MAE = 0.0% — PERFECT

---

## 1. Paper Design

**Full citation:** Forster, S.M., Rickertsen, J.R., Mehring, G.H., Ransom, J.K., 2018. Type and placement of zinc fertilizer impacts cadmium content of harvested durum wheat grain. *Journal of Plant Nutrition*, 41(11), 1471–1481.

**Country:** USA (North Dakota)
**Crop:** Durum wheat (*Triticum turgidum* L.); cultivars Carpio, Joppa, AC Strongfield
**Study system:** Multi-site field experiment; sites at Crosby, Hettinger, and Minot ND; 2014 and 2015 growing seasons
**Experimental design:** Randomised complete block, n = 4 replicates

**Two separate trials reported in the paper:**

- **Trial 1 (fertilizer type and placement):** Compares ZnSO4 applied as soil broadcast (12 kg Zn ha-1) versus Zn-EDTA applied as foliar at Feekes 10 (1.1 kg Zn ha-1), versus untreated control. Data reported in Table 3. Means combined across 3 sites × 2 years.
- **Trial 2 (foliar application timing):** Compares Zn-EDTA at 1.1 kg Zn ha-1 applied at four growth stages (Feekes 4, 10, 10.54, 11.1) versus untreated control. Data reported in Table 4. Means combined across 2 sites × 2 years.

**Primary research question:** Does zinc fertilization reduce grain cadmium (Cd) accumulation in durum wheat? Grain Zn concentration is a secondary biofortification outcome.

**Outcomes in target tables:** Grain test weight (kg m-3), yield (kg ha-1), protein (g kg-1), Cd (mg kg-1), Fe (mg kg-1), Zn (mg kg-1), and element uptake in mg ha-1.

**Variance:** LSD(0.05) reported in all result tables. Paper states "Mean comparisons using F-protected least significant differences (LSD) were made where F-tests indicated significant differences existed (p < .05)." Recon correctly identified LSD as the variance type (confidence: medium — PDF is scanned with potential OCR errors).

**Scanned PDF:** Yes. Recon flagged OCR risk and set extraction method to "hybrid" (vision supplementing text). Estimated difficulty: HARD.

**GT study IDs:** Soil application sheet, study_id = 52 (1 row, Obs ID 393); Foliar application sheet, study_id = 48 (5 rows, Obs IDs 471–475).

---

## 2. AI Extraction and GT Match

### Extraction pipeline summary

| Model | Raw observations extracted |
|-------|--------------------------|
| Claude | 54 |
| Gemini | 11 |
| Kimi | 0 |
| Consensus (post-vote) | 11 |

Kimi extracted 0 observations (likely a parsing failure on the scanned PDF), so the tiebreaker rule fell to Claude vs Gemini. The 11 consensus observations are those where Claude and Gemini agreed within tolerance.

### Matched observations (6/6)

The 6 GT rows span both MOESM5 sheets. The extractable GT values are control grain Zn concentration (mg kg-1) and implied treatment grain Zn concentration, recoverable as: `treat = ctrl × (1 + Zn_biofortification_index / 100)`. The pipeline extracted these directly from Table 3 (ZnSO4 and Zn-EDTA Feekes 10) and Table 4 (Zn-EDTA at four timings) and matched all 6 with zero error:

| GT Obs ID | Sheet | App type | Control (mg/kg) | Treatment (mg/kg) | GT effect (%) | Ext effect (%) | Error |
|-----------|-------|----------|-----------------|-------------------|---------------|----------------|-------|
| 393 | Data 2 Soil | ZnSO4 broadcast, 12 kg Zn ha-1 | 25.4 | 27.8 | +9.45 | +9.45 | 0.0% |
| 471 | Data 3 Foliar | Zn-EDTA foliar, Feekes 10 (Trial 1) | 25.4 | 29.1 | +14.57 | +14.57 | 0.0% |
| 472 | Data 3 Foliar | Zn-EDTA foliar, Feekes 4 (Trial 2) | 32.3 | 33.0 | +2.17 | +2.17 | 0.0% |
| 473 | Data 3 Foliar | Zn-EDTA foliar, Feekes 10 (Trial 2) | 32.3 | 37.9 | +17.34 | +17.34 | 0.0% |
| 474 | Data 3 Foliar | Zn-EDTA foliar, Feekes 11.1 (Trial 2) | 32.3 | 38.5 | +19.20 | +19.20 | 0.0% |
| 475 | Data 3 Foliar | Zn-EDTA foliar, Feekes 10.54 (Trial 2) | 32.3 | 37.0 | +14.55 | +14.55 | 0.0% |

All six values were extracted verbatim from the paper. Effect sizes range from +2.2% (early foliar, Feekes 4) to +19.2% (late foliar, Feekes 11.1), capturing the biologically meaningful timing gradient.

### Variance extraction

LSD values were recovered for the Zn concentration rows in Table 4: LSD = 3.0 mg kg-1 for all four timing treatments. Table 3 Zn rows have null variance (likely OCR failure on the scanned table). Variance is recorded as variance_type = "LSD" throughout. The post-processing CV heuristic flagged several observations for "variance_type mismatch" (reported LSD, heuristic estimates SD), which is a known false-positive artifact for LSD-only papers — the LSD values are numerically correct.

### Verification flags

The automated checker flagged 10 of 11 consensus observations for one or more of:

- **GRIM failure** (all Zn concentration observations): Values like 27.8, 29.1, 32.3, 33.0, 37.0, 37.9, 38.5 are continuous measurements expressed to 1 decimal place with n = 4. GRIM is not applicable to continuous data and these failures are expected and uninformative.
- **Variance type mismatch** (7 obs): CV heuristic suggests SD, paper reports LSD. This is a known limitation of the heuristic when variance is LSD rather than SE/SD.
- **Direction flag on Cd** (1 obs, not a Zn row): The Cd observation from Table 4 (Feekes 4 timing) shows Cd decreasing from 0.10 to 0.09 mg/kg. The checker expected Cd to increase with Zn treatment (since Zn is the "treatment"), but this paper's primary finding is that Zn application *reduces* grain Cd — so the negative Cd effect is scientifically correct, not a T/C swap.

No verification flags apply to the 6 matched Zn observations. All pass the direction check (positive Zn response to Zn fertilization), GRIM is not meaningful for continuous data, and LSD variance values are plausible.

---

## 3. Assessment: Perfect

This paper is one of the cleanest extractions in the Hui 2023 validation set. Several factors contributed to the perfect match:

**Why extraction succeeded despite a scanned PDF:**

1. **Unambiguous control definition.** The paper uses "Untreated control" (no fertilizer) throughout, with no baseline confusion across trials.
2. **Clear table structure.** Table 3 and Table 4 present treatment means in separate columns with a single shared control column — a layout that LLM extraction handles well even on scanned PDFs.
3. **No multi-cultivar disaggregation required.** Hui 2023 used trial-level means averaged across cultivars (Carpio, Joppa, AC Strongfield), which is what both tables report in their combined-analysis rows. The AI correctly averaged across cultivar sub-rows where needed.
4. **Distinct treatment arms across two trials.** The two trials have non-overlapping treatment definitions (soil vs. foliar type; foliar timing), making each GT observation uniquely identifiable.
5. **Exact numeric agreement.** Control values (25.4, 32.3 mg/kg) and treatment values (27.8, 29.1, 33.0, 37.9, 38.5, 37.0 mg/kg) were read from the tables without OCR corruption, confirming good PDF text layer quality for the numerical cells despite the scanned status.

**Recon quality:** The recon phase produced accurate, actionable guidance — it correctly identified Tables 3 and 4 as the target tables, flagged the scanned OCR risk, correctly detected LSD as the variance type with the verbatim quote from the statistical methods section, identified n = 4 from the Methods, and warned about the Cd-reduction framing (Zn as a secondary outcome). The extraction guidance note correctly instructed extraction of ZnSO4 and Zn-EDTA arms from Table 3 and all four timing arms from Table 4 against the untreated control. This level of recon accuracy is a direct driver of extraction success.

**Biological interpretation note:** The range of foliar timing effects (+2.2% at Feekes 4 through +19.2% at Feekes 11.1) reflects the paper's finding that late-season foliar Zn application is more effective for grain Zn biofortification than early-season application. The soil ZnSO4 effect (+9.5%) falls between early and mid-season foliar applications. These gradients are scientifically coherent and correctly captured.

**Remaining limitation:** Variance values for Table 3 Zn rows (ZnSO4 and Zn-EDTA Feekes 10 from Trial 1) are null in the extracted output — likely an OCR failure on the scanned Table 3. Table 4 LSD values were recovered (LSD = 3.0 mg/kg). This does not affect the match score (GT does not include variance), but would require manual PDF verification before these two observations can be used in a weighted meta-analysis.
