# Extraction Quality Report: 3_Zhang_2012

**Paper:** Xue et al. (2012) "Grain and shoot zinc accumulation in winter wheat affected by nitrogen management." *Plant and Soil* 361:153–163. DOI 10.1007/s11104-012-1510-2
**Match summary:** `no_gt` — MOESM5 "Data 3 Foliar application" has study_id=3 mapped to this paper, but those 6 rows contain **zero grain Zn control/treatment concentration values** (only moderator/design metadata and a "Zn biofortification index" derived metric). The validation script correctly finds 0 matchable GT rows.

---

## 1. Paper Design

- **Intervention:** Nitrogen (N) fertilization rate — four levels: N0 (0), N1 (~99/82.5 kg N ha⁻¹), N2 (~198/195 kg N ha⁻¹, optimal), N3 (~297/292.5 kg N ha⁻¹) — across two field seasons (2007–2008 and 2008–2009) at Quzhou, Hebei, China.
- **Zinc treatment:** All plots received an identical basal application of 30 kg ha⁻¹ ZnSO₄·7H₂O before sowing in **every** season. There is **no Zn fertilizer vs. no-Zn control** contrast anywhere in the paper.
- **Crop:** Winter wheat (*Triticum aestivum* L., var. Kenong 9204).
- **Design:** Randomized complete block, 4 replicates, 300 m² plots.
- **Primary questions:** (1) Does N rate affect grain Zn concentration and content? (2) What fraction of grain Zn originates from pre-anthesis remobilization vs. post-anthesis root uptake?
- **Variance:** LSD test at P<0.05 (confirmed in Methods); SE bars plotted in figures (caption: "bars represent the standard error of the mean, n=4"). Numeric LSD or SE values are **not tabulated** — only letter-group significance notation is used in Table 2.

---

## 2. Grain Zn Data in PDF?

Yes. Table 2 (p. 158) reports grain Zn concentration (mg kg⁻¹, DM basis) for all eight treatment × year combinations:

| Year | Treatment | N rate (kg ha⁻¹) | Grain Zn conc. (mg kg⁻¹) | Grain Zn content (g ha⁻¹) |
|------|-----------|-----------------|--------------------------|---------------------------|
| 2007–2008 | N0 | 0   | 21.5c | 88.6c  |
| 2007–2008 | N1 | 99  | 25.1bc | 130.5b |
| 2007–2008 | N2 | 198 | 30.9ab | 175.8a |
| 2007–2008 | N3 | 297 | 37.0a  | 208.7a |
| 2008–2009 | N0 | 0   | 24.7b  | 54.8c  |
| 2008–2009 | N1 | 82.5| 25.0b  | 91.8b  |
| 2008–2009 | N2 | 195 | 29.1ab | 146.4a |
| 2008–2009 | N3 | 292.5| 32.0a | 156.8a |

The abstract also states these key values explicitly (21.5 → 30.9 mg kg⁻¹ in 2007–2008; 24.7 → 29.1 mg kg⁻¹ in 2008–2009 from N0 to optimal N2). Grain Zn data are **unambiguously present and clearly readable** from Table 2.

**However, this is not a Zn biofortification study.** The paper compares N rates, not Zn application rates. Every treatment plot received the same Zn fertilizer dose. The Zn "treatment" vs. "control" contrast required by the Hui 2023 meta-analysis — i.e., Zn-fertilized vs. unfertilized — does not exist in this paper.

---

## 3. AI Extraction Results

The consensus JSON (`3_Zhang_2012_consensus.json`) contains **5 extracted observations**, all from Table 2 (season 2007–2008, N0 vs. N1 contrast only):

| Element | Tissue | Control mean | Treatment mean | Unit | Effect (%) | Notes |
|---------|--------|-------------|----------------|------|-----------|-------|
| Grain Zn concentration | grain | 21.5 | 25.1 | mg/kg | +16.7% | GRIM fail on treatment |
| Grain yield | grain | 4.2 | 5.2 | Mg/ha | +23.8% | GRIM fail both |
| Straw yield | shoot | 4.4 | 5.2 | Mg/ha | +18.2% | GRIM fail both |
| Grain Zn content | grain | 88.6 | 130.5 | g/ha | +47.3% | GRIM fail on control |
| ZnHI | whole plant | 80.0 | 86.0 | % | +7.5% | — |

**AI interpretation of the study:** The recon correctly identified this as a nitrogen experiment, not a Zn fertilizer experiment. The recon JSON carries an explicit extraction_guidance field: "DO NOT EXTRACT — This paper does not meet meta-analysis inclusion criteria. It studies nitrogen fertilizer effects on Zn accumulation, not zinc fertilizer application effects." Despite this warning, extraction proceeded (likely because the paper passed the PICO recon step for inclusion) and extracted 5 observations comparing N0 (control) vs. N1 (treatment), which is a valid N-rate contrast but **not** a Zn application contrast.

**Variance:** All 5 observations have `variance_type = "LSD"` but `treatment_variance = null` and `control_variance = null`, consistent with the paper's use of letter notation only (no numeric LSD values in Table 2).

**GRIM failures:** 4 of 5 observations fail the GRIM test. The means in Table 2 are reported to one decimal place with n=4. GRIM expects means of integer-valued measurements to satisfy mean × n ≈ integer. For n=4, valid one-decimal means are multiples of 0.25. Values like 25.1, 5.2, 4.4 are not multiples of 0.25, indicating these are continuous (not count) measurements — GRIM does not apply here. The GRIM failures are false positives and do not indicate extraction error.

---

## 4. Why No GT? (What does the MOESM5 Foliar sheet row look like for study_id=3?)

The MOESM5 "Data 3 Foliar application" sheet contains 6 rows for study_id=3 (Observation IDs 10–15). Inspection of the GT text file reveals the structure:

**What IS present** in these 6 rows:
- Moderator variables: soil available Zn (0.3 mg kg⁻¹), pH (7.3), CaCO3 (6.46%), organic matter (10.3 g kg⁻¹), N rate (225 kg N ha⁻¹), P rate (varying: 0, 25.19, 50.38, 100.76, 201.52, 403.04 kg P ha⁻¹), K rate (49.8 kg K ha⁻¹)
- Zn fertilizer type: ZnSO4; Zn rate: 1.0896 kg Zn ha⁻¹
- Spraying concentration: 0.0908 g Zn L⁻¹; spraying frequency: 2 times; spraying timing: 5
- Numbers of replicates (n): 4
- **Zn biofortification index**: values ~7.4–11.3 (a dimensionless effect-size metric, not a raw concentration)

**What IS NOT present** in these rows:
- Grain Zn concentration in the control plot (mg kg⁻¹)
- Grain Zn concentration in the treatment plot (mg kg⁻¹)
- Any raw means that the validation script can match against extracted ctrl/treat values

The validation script searches for a column header containing "Grain Zn concentration" in row 2 of the sheet, then expects numeric ctrl and treat values in that column and the next. The 6 rows for study_id=3 have `None` (empty cells) in those columns. The `load_gt()` function's guard `if ctrl is None or treat is None or ctrl <= 0: continue` therefore skips all 6 rows, yielding 0 usable GT observations.

**Why does the Hui 2023 database include this paper at all if it has no raw grain Zn means?**

The paper that actually corresponds to the Hui 2023 Foliar study_id=3 rows appears to be a **different Zhang 2012 paper**: Zhang et al. (2012) "Zinc biofortification of wheat through fertilizer applications in different locations of China" (*Field Crops Research* 125:1–7), which is a proper foliar Zn biofortification study with a Zn-fertilized vs. unfertilized contrast. That paper is cited in the reference list of the PDF being reviewed (last line of Discussion, p. 162: "Zhang et al. 2012") and appears as one of three "Zhang 2012" entries in the Hui dataset. The PDF named `3_Zhang_2012.pdf` is Xue et al. 2012 (Plant and Soil), which the Hui database may have catalogued under a different study ID or a different sheet entry. The 6 Foliar rows for study_id=3 in MOESM5 describe a foliar Zn experiment (spraying concentration, spraying frequency, spraying timing are all populated), which does not match Xue et al. 2012 at all — that paper has no foliar Zn spraying. This is consistent with the PDF being mislabeled or the MOESM5 study_id=3 in the Foliar sheet referring to the *Field Crops Research* Zhang 2012 paper rather than the *Plant and Soil* paper in the PDF.

---

## 5. Assessment

| Dimension | Finding |
|-----------|---------|
| **Is grain Zn data present in the PDF?** | Yes — Table 2 has clear grain Zn concentration values for 8 treatment combinations |
| **Does the paper qualify for the Hui 2023 Zn biofortification meta-analysis?** | No. It is a nitrogen management study; all plots received identical Zn fertilization. There is no Zn-fertilized vs. Zn-unfertilized contrast. |
| **AI recon quality** | Excellent — correctly identified the paper as outside the Zn biofortification PICO and flagged it with "DO NOT EXTRACT" in `extraction_guidance`. |
| **AI extraction quality** | Partially correct — extracted valid N-rate contrasts from Table 2 with correct values, but should not have proceeded to extraction given the recon guidance. Only 1 of 8 possible contrasts was extracted (N0 vs. N1, 2007–2008 only). |
| **GRIM failures** | False positives — continuous measurements with 1 decimal place; GRIM is not applicable. |
| **Variance extraction** | Correct: LSD type identified; numeric values not available (paper uses letter notation only). |
| **Root cause of `no_gt` status** | Twofold: (1) The 6 MOESM5 Foliar rows for study_id=3 appear to belong to a *different* Zhang 2012 paper (*Field Crops Research*), not the PDF in the dataset. (2) Even if the mapping were correct, those rows contain only the derived "Zn biofortification index," not the raw grain Zn control/treatment concentrations needed for validation matching. |
| **Action needed** | None for the meta-analysis — this paper is correctly excluded from effect-size computation. The `no_gt` status is appropriate. No re-extraction is warranted. Consider flagging this PDF as a potential mislabeling (the file `3_Zhang_2012.pdf` contains the *Plant and Soil* nitrogen paper, while MOESM5 study_id=3 in the Foliar sheet describes a foliar Zn application experiment). |
