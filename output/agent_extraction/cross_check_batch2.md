# Cross-Check Report: Batch 2 (Papers 006-010)

## Summary

| Paper | Verdict | Elements | Obs | Key Finding |
|-------|---------|----------|-----|-------------|
| 006_Azam_2013 | **PASS** | 11 (C,N,H,S,Ca,Mg,K,Zn,Mn,Fe,Cu) | 29 | All values exact match to Table 3 |
| 007_Woodin_1992 | **FLAG** | 5 (N,P,K,Ca,Mg) | 20 | Figure-read approximations; no exact values verifiable |
| 008_Campbell_2002 | **FLAG** | 1 (P) | 6 | Figure-read values; sub-ambient comparisons not extracted |
| 009_Barnes_1992 | **PASS** | 6 (N,S,P,K,Mg,Ca) | 24 | All values exact match to Tables 4-5 |
| 010_Li_2010 (Hogy 2009) | **FLAG** | 21 elements | 21 | All treatment/control means null; N derivation discrepancy |

**Overall: 2 PASS, 3 FLAG. No outright extraction errors or direction contradictions found across all 5 papers.**

---

## 006_Azam_2013 -- PASS

**Paper**: Azam et al. 2013, elemental composition of carrot, radish, and turnip under elevated CO2
**Data source**: Table 3
**Extraction**: 29 observations across C, N, H, S, Ca, Mg, K, Zn, Mn, Fe, Cu for 3 vegetable species

### Checks Performed

1. **Values vs Table 3**: All 29 extracted treatment_mean and control_mean values match Table 3 exactly. No discrepancies found.

2. **Effect % calculations**: All effect_pct values are arithmetically correct from the extracted means.

3. **Direction consistency**: The paper discusses CO2-induced dilution of mineral nutrients, consistent with the predominantly negative effect percentages in the extraction.

4. **Missing elements**: Table 3 also reports Pb, Ni, Cr, and Cd for all three species. These 4 elements x 3 species = **12 additional observations are present in the paper but not extracted**. This is a coverage gap, though the omitted elements are trace metals/heavy metals that may be outside the meta-analysis scope.

5. **No additional tissues**: Only aboveground tissue reported in the paper, matching the extraction.

### Issues
- **Minor coverage gap**: 12 observations for Pb, Ni, Cr, Cd available in Table 3 but not extracted.

---

## 007_Woodin_1992 -- FLAG

**Paper**: Woodin et al. 1992, nutrient limitation of heather response to CO2 enrichment
**Data source**: Figure 3a (bar charts only -- no numerical table in paper)
**Extraction**: 20 observations across N, P, K, Ca, Mg in leaf and stem at 470 and 570 ppm vs 370 ppm ambient

### Checks Performed

1. **Direction consistency with text**: The paper states elevated CO2 caused reductions in N, P, and K concentrations in leaf tissue, consistent with the extracted negative effect percentages. Ca and Mg show mixed effects matching the paper's discussion.

2. **Magnitude plausibility**: The extracted leaf N decline of -28.6% (470 ppm) and -42.9% (570 ppm) are large but consistent with the paper's emphasis on severe nutrient limitation.

3. **No missing elements**: The paper only reports N, P, K, Ca, Mg -- all extracted.

4. **No tabulated values exist**: All data from Figure 3a bar charts. Exact numerical verification is impossible.

### Flagged Issues
- **Inherent figure-reading uncertainty**: All 20 observations are read from bar charts with no tabulated values to verify. Extracted means are approximations.
- **No error bars or variance data**: Figure 3a shows bars but variance information is limited.

---

## 008_Campbell_2002 -- FLAG

**Paper**: Campbell & Sage 2002, CO2 and phosphorus interactions on proteoid root formation in white lupin
**Data source**: Figures 4a (low P) and 4b (high P)
**Extraction**: 6 observations for P concentration across leaf, normal root, and proteoid root

### Checks Performed

1. **Direction consistency**: The paper states CO2 enrichment had "no effect on the P concentration of leaves, normal roots, or proteoid roots at either P treatment." The extracted effect percentages are small (0%, 11%, -12%, 0%, 0%, -8%), consistent with non-significant effects.

2. **P-only extraction**: Appropriate -- the paper focuses exclusively on phosphorus nutrition. No other mineral concentrations are reported.

3. **Figure-read values**: All values from Figure 4 bar charts. No exact tabulated P concentrations to verify.

### Flagged Issues
- **Sub-ambient comparisons not extracted**: The paper includes a 200 ppm (Pleistocene) CO2 treatment. Comparing 200 vs 410 ppm would yield 6 additional observations. While elevated-vs-ambient is the standard meta-analysis comparison, the sub-ambient data represents a missed opportunity.
- **Figure-read uncertainty**: All 6 observations are approximations from bar charts.

---

## 009_Barnes_1992 -- PASS

**Paper**: Barnes & Pfirrmann 1992, CO2 and O3 effects on gas exchange, growth and nutrient status of radish
**Data source**: Table 4 (shoot nutrients) and Table 5 (root nutrients)
**Extraction**: 24 observations for N, S, P, K, Mg, Ca across shoot and root at 2 O3 levels

### Checks Performed

1. **Exact value verification (Table 4 - shoot)**:
   - N at low O3: extracted 3.61 vs 4.27 -- matches Table 4 exactly
   - N at high O3: extracted 4.12 vs 4.13 -- matches Table 4 exactly
   - S at low O3: extracted 0.99 vs 1.35 -- matches Table 4 exactly
   - P at low O3: extracted 0.259 vs 0.329 -- matches Table 4 exactly
   - K at low O3: extracted 3.24 vs 3.19 -- matches Table 4 exactly
   - Mg, Ca values all match Table 4 exactly

2. **Exact value verification (Table 5 - root)**:
   - N at low O3: extracted 1.27 vs 1.46 -- matches Table 5 exactly
   - S at low O3: extracted 0.31 vs 0.46 -- matches Table 5 exactly
   - All 12 root observations match Table 5 exactly

3. **Effect % calculations**: All 24 computed effect percentages are arithmetically correct.

4. **Direction consistency**: The paper's Discussion states "a general decrease in the concentrations of N, S, P and Mg" under elevated CO2, matching the extracted negative effects. K and Ca show smaller/mixed effects, also consistent.

### Minor Notes
- CO2 levels labeled as 350/750 in the extraction vs actual ~385/765 in the paper (minor labeling issue, does not affect data).
- C concentration and C/N ratio are present in Tables 4-5 but not extracted. Defensible omission since C is not typically a mineral nutrient of interest.

---

## 010_Li_2010 (actually Hogy et al. 2009) -- FLAG

**Paper**: Hogy et al. 2009, "Effects of elevated CO2 on grain yield and quality of wheat" (3-year FACE study)
**Data source**: Figure 2 (relative % change bar chart), Results text, Figure 5 (protein)
**Extraction**: 21 observations for grain concentrations of N, C, S, P, K, Ca, Mg, Na, Fe, Zn, Cu, Mn, Se, Mo, Cr, Ni, Si, B, Al, Cd, Pb

### Checks Performed

1. **Text-stated values match exactly**:
   - B: -26.3% stated in Results text, extracted as -26.3%
   - Al: -11.7% stated in Results text, extracted as -11.7%
   - C: -0.3% (not significant) stated in text, extracted correctly

2. **N derivation**: Extraction derives N from protein: ambient 15.5%, elevated 14.2%, divided by 5.7. Calculated effect: (2.491-2.719)/2.719 = -8.4%. Paper reports protein decrease of -7.4%. The -8.4% is mathematically correct from the stated protein means; the paper's -7.4% may reflect a different calculation method (e.g., response ratio from individual replicates rather than ratio of means).

3. **Figure 2 element coverage**: All macro- and micro-elements visible in Figure 2 are captured in the extraction.

4. **Significance flags consistent**: Paper reports Fe, Cd, Pb as significant; K as significant increase; Mg and Mo as trends. All match the extraction notes.

5. **Direction check**: All 21 extracted directions are consistent with Figure 2 bar chart orientations.

### Flagged Issues
- **All treatment_mean and control_mean values are null** for 20 of 21 elements. Only effect_pct is available. This is because Figure 2 only reports relative % changes, not absolute concentrations. While this is accurate to the paper, it limits the utility of these observations for meta-analytic calculations that require means.
- **N derivation discrepancy**: -8.4% computed vs paper's -7.4% reported for protein effect. The extraction's calculation is mathematically correct from stated means, but the mismatch may warrant a note.
- **Figure-estimated values for most elements**: The ~3%, ~18%, ~-10% etc. are read from bar charts and carry inherent uncertainty of approximately +/-1-2 percentage points.

---

## Cross-Check Summary

### Verdict Table

| Paper | Verdict | Reason |
|-------|---------|--------|
| 006_Azam_2013 | **PASS** | All values exact match; minor coverage gap (4 trace elements) |
| 007_Woodin_1992 | **FLAG** | All data from figure approximations; no table values to verify |
| 008_Campbell_2002 | **FLAG** | Figure-read values; sub-ambient CO2 comparisons omitted |
| 009_Barnes_1992 | **PASS** | All 24 values exact match to Tables 4-5 |
| 010_Hogy_2009 | **FLAG** | Null means for 20/21 obs; N effect discrepancy; figure-estimated |

### Data Quality Tiers
- **High confidence**: 009_Barnes_1992 (exact table values), 006_Azam_2013 (exact table values)
- **Moderate confidence**: 010_Hogy_2009 (figure-estimated %, but text-stated values for B, Al match exactly)
- **Lower confidence**: 007_Woodin_1992, 008_Campbell_2002 (all figure-read, no tabulated values)

### Coverage Gaps Identified
1. **006_Azam_2013**: 12 obs for Pb, Ni, Cr, Cd in Table 3 not extracted
2. **008_Campbell_2002**: 6 obs for sub-ambient (200 ppm) vs ambient comparisons not extracted
3. **009_Barnes_1992**: C concentration in Tables 4-5 not extracted (defensible)
4. **010_Hogy_2009**: Absolute concentration means not recoverable from paper (only % changes in Figure 2)
