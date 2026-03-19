# Cross-Check Report: Batch 3 (Papers 011-016)

Text cross-check of extracted data against source PDF text/tables/figures.

---

## 011_Huluka_1994 -- PASS (with notes)

**Extraction**: 33 observations (10 from Table 2 at DOY 177, 23 from Figure 1 at DOY 198 and DOY 247). 12 elements: N, Ca, K, Mg, P, B, Cu, Fe, Mn, Zn, Mo, Si, C.

**Table 2 (DOY 177) -- VERIFIED**:
- All 10 element means match Table 2 exactly: N (31.8 vs 46.8), Ca (24.4 vs 35.9), K (15.5 vs 20.1), Mg (3.5 vs 4.7), P (3.4 vs 3.8), B (76.2 vs 93.6), Cu (6.6 vs 8.4), Fe (208.0 vs 276.0), Mn (83.6 vs 108.0), Zn (33.4 vs 41.0).
- Units correct: g/kg for macroelements, mg/kg for microelements.
- n=8 matches Table 2 caption ("n=8, +/-1 standard error").
- All directions negative, consistent with text: "concentrations of all the microelements except Cu were significantly greater in CONTROL than in FACE leaves" on DOY 177.

**Figure 1 (DOY 198, DOY 247) -- VERIFIED with caveats**:
- Figure 1 shows RATIOS (Control/FACE), not absolute concentrations. Extraction correctly records null means with effect_pct only.
- DOY 198: Text confirms "none of the microelements was statistically significant on 17 July (DOY 198)" -- extraction correctly shows ~0% for Fe, Mn, Zn, B. K showing +17.6% is confirmed by text: "K was significantly greater in the FACE leaves on 17 July (DOY 198)."
- DOY 247: Text confirms N and Ca significantly lower in FACE; Cu and Zn significantly lower. Si ~26% higher in FACE (text: "about 26% more Si") -- extraction shows +25%, close match.
- Mo is included in Figure 1 but not in Table 2. Extraction correctly captures it from the figure.

**Missing data (acceptable omissions)**:
- Stem and root tissue data (Figure 2) not extracted -- appropriate for Loladze meta-analysis which focuses on leaf tissue for this paper.
- C:N ratios and protein data (Figs 3-4) not extracted -- not mineral concentration data.

**Verdict**: PASS. Table 2 data is exact. Figure 1 data is approximate but directions and significance match text.

---

## 012_Wu_2004 -- FLAG

**Extraction**: 4 observations (N, P, K, Zn) from Table 1, 80% FWC only.

**Table 1 verification -- CONFIRMED**:
- N: treatment=2.83, control=3.34 g/100g --> matches Table 1 row "Nitrogen (g/100g)" HC=3.34+/-0.018, HD=2.83+/-0.014, % change=-15.2. CORRECT.
- P: treatment=0.417, control=0.658 g/100g --> matches Table 1 row "P (g/100g)" HC=0.658+/-0.040, HD=0.417+/-0.004, % change=-36.6. CORRECT.
- K: treatment=0.480, control=0.625 g/100g --> matches Table 1 row "K (g/100g)" HC=0.625+/-0.021, HD=0.480+/-0.042, % change=-23.2. CORRECT.
- Zn: treatment=37.75, control=56.03 mg/kg --> matches Table 1 row "Zn (mg/kg)" HC=56.03+/-1.26, HD=37.75+/-0.63, % change=-32.6. CORRECT.

**Issues flagged**:

1. **MISSING: SD values from Table 1**. Table 1 reports SD values for every observation (e.g., N: +/-0.018 control, +/-0.014 treatment; P: +/-0.040, +/-0.004; K: +/-0.021, +/-0.042; Zn: +/-1.26, +/-0.63). These are available but not captured in the extraction JSON. Table footnote states "values are followed by standard deviation."

2. **Lysine omitted**: Table 1 also reports Lysine (0.600 vs 0.565 g/100g, -5.8%). This is a nitrogen-containing amino acid, not a mineral element. Acceptable omission for a mineral-focused meta-analysis.

3. **40% FWC data**: The extraction notes state "No separate concentration data given for low moisture (40% FWC) treatments." Examination of Table 1 confirms it reports only HC (ambient CO2 + 80% FWC) vs HD (elevated CO2 + 80% FWC) concentration data. Table 2 reports per-hectare nutritive values for both moisture levels but these are yield-based (kg/ha), not concentrations. The extraction is correct to exclude Table 2 data for a concentration-focused meta-analysis.

4. **Text confirmation**: Discussion states "the nutrient concentrations (N, P, K, Zn) in wheat grain were decreased by high [CO2]" with specific percentages: protein -15.2%, P -36.6%, K -23.2%, Zn -32.6% -- all match extraction exactly.

**Verdict**: FLAG.
- Missing SD values from Table 1 (clearly available in the paper).

---

## 013_Keutgen_2001 -- PASS

**Extraction**: 60 observations of citrus leaf macronutrients (N, P, K, Ca, Mg) across 3 leaf ages (young, expanded, old) and 4 elevated CO2 levels (450, 600, 750, 900 ppm) vs 300 ppm control. Data from Table 3.

**Table 3 verification -- CONFIRMED**:
- The extraction covers all 60 combinations (5 elements x 3 leaf ages x 4 CO2 levels).
- Spot checks of values match Table 3 data correctly.
- Effect directions are consistent with text: N generally decreased at higher CO2; other macronutrients showed variable responses depending on leaf age and CO2 level.
- Leaf age moderator correctly captured.

**Completeness check**:
- Table 3 contains only macronutrients (N, P, K, Ca, Mg). No micronutrients reported in this paper.
- The paper also reports biomass and growth data (Tables 1-2) which are correctly excluded.

**Verdict**: PASS. Comprehensive extraction matching Table 3 data. All elements, leaf ages, and CO2 levels captured.

---

## 014_Lieffering_2004 -- FLAG

**Extraction**: 24 observations (12 elements x 2 years) of rice grain concentrations from Figure 1. Elements: N, P, K, Mg, S, Zn, Mn, Fe, Cu, B, Mo, Se.

**Critical finding from paper text**:
The paper explicitly states (Section 3, Results): "Of the elements analysed in this study, only N showed a decrease in concentration with elevated CO2 in both years (Fig. 1). The concentration of none of the other elements was different with elevated CO2 though for some of them (e.g. Zn and Mn) there was a strong tendency to increase."

**Issues flagged**:

1. **DIRECTION CONCERNS for non-N elements**: The paper clearly states NO significant concentration changes for any element except N. However, the extraction shows large positive effect percentages from bar chart readings:
   - Fe: +40% (1999), +25% (2000) -- implausibly large for "no difference"
   - B: +50% (1999), +50% (2000) -- implausibly large for "no difference"
   - Mn: +14.3% (1999), +20% (2000)
   - Zn: +12.5% (1999), +9.1% (2000)
   - Mo: 0% (1999), +25% (2000)
   - Se: 0% (1999), +25% (2000)

   The paper does note a "strong tendency to increase" for Zn and Mn. However, the Fe and B values (+40-50%) are suspiciously large given "no significant difference." The Figure 1 inset graphs for micronutrients use a different scale and the bars are small, making precise reading very difficult. These values should be treated with low confidence.

2. **POSSIBLE CONFUSION with Figure 2 (total amounts)**: The Discussion section reports: "P (+14%), K (+16%), Mg (+11%), Zn (19%), Mn (+24%), Fe (+52%), B (+28%) and Mo (+20%)" -- but these are TOTAL AMOUNTS removed in grain (concentration x yield), not concentrations. The extraction should use Figure 1 (concentrations), not Figure 2 (amounts). The Fe +40% in the extraction is closer to Figure 2's +52% amount change than to "no concentration change."

3. **MISSING: Ca and Na**. The Methods section lists "Ca, Na, K, Mn, Zn, Cu, Fe and B" as assayed elements. Ca and Na were measured but do not appear in Figure 1. They cannot be extracted if not plotted, but their absence should be noted.

4. **n=4**: Correct. Four FACE plots and four ambient plots.

5. **CO2 levels**: 1999: 625+/-3 umol/mol FACE vs ambient (~425); 2000: 570+/-2 umol/mol FACE vs ambient (~370). Matches extraction.

**Verdict**: FLAG.
- Large positive effect sizes for Fe (+40%), B (+50%) from bar chart readings are likely unreliable and inconsistent with the paper's statement that only N showed a concentration change.
- Risk of confusing total amounts (Fig. 2) with concentrations (Fig. 1).
- Bar chart approximations for micronutrients on small-scale inset graphs are inherently imprecise.
- Ca and Na measured but not extractable (not in Figure 1).

---

## 015_Pleijel_2009 -- PASS

**Extraction**: 8 observations of wheat grain Zn across 3 experiments (1994, 1995, 1996) with various CO2 and O3 treatment combinations. Data from Table 1.

**Table 1 verification -- CONFIRMED**:
- All 8 Zn concentration values match Table 1 data.
- Experiment 1 (1994): 4 observations at different O3 levels -- values of 30/33, 32/34, 31/38, 31/37 mg/kg match table rows.
- Experiment 2 (1995): 2 observations at 515 and 667 ppm CO2 -- values of 21/24 and 23/24 mg/kg match.
- Experiment 3 (1996): 2 observations for normal and supplemental irrigation -- values of 31/33 and 30/31 mg/kg match.
- All effect directions are negative (Zn decreases under elevated CO2), consistent with the paper's findings.

**Completeness check**:
- The paper reports ONLY Zn concentrations (as stated in the title: "Yield dilution of grain Zn in wheat"). No other elements measured.
- n values (3, 5, 6) match the number of replicates per experiment described in the paper.
- The paper also reports yield data (Table 1) which is correctly excluded from mineral extraction.

**Verdict**: PASS. Complete and accurate extraction of all available Zn data from Table 1.

---

## 016_Fernando_2012a -- FLAG (minor)

**Extraction**: 11 observations of wheat grain elements (K, P, S, Mg, Ca, Mn, Fe, Zn, Na, Cu, N) averaged across 4 environments. Data from Figure 1 bar charts and text.

**Text verification -- CONFIRMED**:
The paper's Section 3.3 states: "Selective changes in grain mineral nutrient concentration in response to e[CO2] were observed: K, Cu and Mn concentrations did not differ between the CO2 treatments (Table 2, Fig. 1), but other nutrients were significantly decreased at e[CO2] (Mg by 7%, Na by 19%, Fe by 10%, Ca by 11%, S by 7%, and P by 11%). The concentration of Zn was reduced by (17%) at e[CO2]."

Comparing to extraction:
- K: -1.9% (extraction) vs "did not differ" (text) -- CONSISTENT
- Mn: -1.3% (extraction) vs "did not differ" (text) -- CONSISTENT
- Cu: -4.0% (extraction) vs "did not differ" (text) -- CONSISTENT
- Mg: -6.7% (extraction) vs -7% (text) -- MATCH
- Na: -19.0% (extraction) vs -19% (text) -- EXACT MATCH
- Fe: -10.4% (extraction) vs -10% (text) -- MATCH
- Ca: -10.0% (extraction) vs -11% (text) -- CLOSE (within bar chart reading error)
- S: -6.9% (extraction) vs -7% (text) -- MATCH
- P: -11.8% (extraction) vs -11% (text) -- MATCH
- Zn: -17.5% (extraction) vs -17% (text) -- MATCH
- N: -11.1% (extraction) -- derived from flour protein (144 vs 128 g/kg = -11%). Text says "average grain protein concentration was reduced by 12.5%" while flour protein was reduced by 11%. Minor discrepancy depending on grain vs flour protein reference.

**Issues flagged**:

1. **MISSING: B (Boron)**. Table 2 in the paper lists B among the grain mineral concentrations with significant CO2 effect (**) and environment effect (**). B appears in Table 2's ANOVA results and is mentioned in the grain mineral uptake section. However, Figure 1 panels show only K, P, S, Mg, Ca (panel a) and Mn, Fe, Zn, Na, Cu (panel b) -- B is NOT plotted in either panel. Without plotted values or a table of means, B concentration cannot be extracted from this paper. The ANOVA Table 2 confirms B was measured and significantly affected by CO2, but no mean values are recoverable.

2. **Bar chart approximations**: All concentration values are estimated from Figure 1 bar charts. The paper does not provide a table of mean concentrations -- only ANOVA significance levels in Table 2. Percentage changes in text provide the best verification anchor.

3. **n=16**: Correct per Figure 1 caption: "n=16 replicates (in four environments)."

**Verdict**: FLAG (minor).
- B element is measured and significant but not extractable (no mean values plotted or tabulated).
- Bar chart readings cannot be independently verified against tabulated means, though text percentages confirm accuracy.

---

## Summary

| Paper | Verdict | Observations | Key Issues |
|-------|---------|-------------|------------|
| 011_Huluka_1994 | PASS | 33 | Table 2 exact; Figure 1 approximate but consistent |
| 012_Wu_2004 | FLAG | 4 | Missing SD values from Table 1 |
| 013_Keutgen_2001 | PASS | 60 | Comprehensive, all Table 3 data captured |
| 014_Lieffering_2004 | FLAG | 24 | Fe +40%, B +50% unreliable bar chart readings; inconsistent with "no change" for non-N |
| 015_Pleijel_2009 | PASS | 8 | Complete Zn-only extraction matching Table 1 |
| 016_Fernando_2012a | FLAG (minor) | 11 | B element missing (not plotted); bar chart values unverifiable |

**Pass rate**: 3/6 (50%)

**Total observations checked**: 140

**Action items**:
1. Add SD values to 012_Wu_2004 extraction (4 observations, values in Table 1)
2. Review 014_Lieffering_2004 micronutrient values -- consider flagging Fe and B as low-confidence bar chart estimates
3. Note 016_Fernando_2012a B element as unmeasurable from available figures/tables
