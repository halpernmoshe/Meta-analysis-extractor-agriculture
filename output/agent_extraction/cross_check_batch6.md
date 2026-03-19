# Cross-Check Report: Batch 6 (Papers 031, 032, 034, 035, 036)

Text cross-check of agent extraction JSONs against PDF Results/Discussion sections.

---

## 031_Pal_2003 (actually Pal et al. 2004)

**Paper:** "Growth, development, and photosynthetic activity in gram (Cicer arietinum L.) plants grown at elevated CO2 levels"
**Extraction:** 15 observations, 5 elements (C, N, P, Ca, Fe), 3 growth stages (40, 60, 80 DAE)

### Verdict: FLAG

### Elements in text vs extraction
- **MATCH**: N, P, Ca, Fe are present in the extraction and in Table 3 of the paper.
- **Extra in extraction**: C (organic carbon %) is extracted. This is organic carbon content, NOT a mineral nutrient. In the Loladze meta-analysis context (mineral concentrations under elevated CO2), organic carbon is not typically included as it reflects a different biological process (carbon assimilation, not mineral dilution).
- **No missing mineral elements** from Table 3.

### Direction checks
- **N decreased**: Table 3 shows N decreasing at 600 ppm vs 360 ppm across all growth stages. JSON shows negative effect_pct for N at all stages. **CONSISTENT**.
- **P decreased**: Table 3 shows P decreasing. JSON: -10.0%, -12.8%, -6.5% across stages. **CONSISTENT**.
- **Ca mixed/NS**: Table 3 shows Ca changes are small and not statistically significant at most stages. JSON: -5.6%, +4.0%, +4.2%. **CONSISTENT** with NS pattern.
- **Fe mixed/NS**: Table 3 shows Fe changes are small and mostly NS. JSON: -8.4%, -3.5%, +4.7%. **CONSISTENT** with variable direction.
- **C increased**: Table 3 shows C% slightly increased under elevated CO2. JSON: +1.3% to +2.2%. **CONSISTENT**.

### Specific value checks
- CO2 levels: JSON lists co2_ambient=360 but notes say "350 uL/L" -- the paper states 360 +/- 20 uL/L ambient. The JSON notes field says 350, which is a minor discrepancy (should be 360).
- Numeric values from Table 3 appear correctly extracted.

### Issues
1. **FLAG**: Organic carbon (C%) is included as element "C" -- this may not belong in a mineral concentration meta-analysis. Consider excluding.
2. **FLAG (minor)**: Ca and Fe show non-significant effects at most stages. The values are correctly extracted but users should note the NS status.
3. **FLAG (minor)**: Ambient CO2 listed as 350 in notes but paper says 360 +/- 20 uL/L.

---

## 032_Kanowski_2001

**Paper:** "Effects of elevated CO2 on the foliar chemistry of seedlings of two rainforest species from north-east Australia: implications for folivorous marsupials"
**Extraction:** 20 observations, 5 elements (N, P, K, Ca, Na), 2 species (Alphitonia petriei, Cryptocarya mackinnoniana), 2 soil types (forest soil, potting mix)

### Verdict: FLAG

### Elements in text vs extraction
- **MATCH**: N, P, K, Ca, Na are all present in extraction and in Figure 1 of the paper.
- **No missing elements**: The paper only measured these 5 elements for foliar chemistry.

### Direction checks
- **N decreased in both species**: Text states "Foliar N concentrations were significantly reduced by elevated CO2 in both species." JSON shows N negative for all 4 species x soil combinations (-3.4% to -25.0%). **CONSISTENT**.
- **P in Alphitonia NS**: Text states P in Alphitonia was not significantly affected (p=0.91). JSON shows small mixed effects for Alphitonia P. **CONSISTENT**.
- **K in Alphitonia NS**: Text states K not significantly affected in Alphitonia (p=0.93). JSON shows small effects. **CONSISTENT**.
- **Ca in Alphitonia NS**: Text states Ca not significant (p=0.60). JSON shows small effects. **CONSISTENT**.
- **Na decreased in Alphitonia**: Text states Na decreased significantly in Alphitonia. JSON: -8.3% to -23.1%. **CONSISTENT**.
- **Cryptocarya effects**: Text indicates all elements decreased in Cryptocarya under elevated CO2. JSON shows mostly negative effects. **CONSISTENT**.

### Specific value checks
- Data extracted from Figure 1 (bar charts), so values are approximate readings. No exact table values available for comparison.
- The text discusses percentage changes qualitatively but does not report exact percentages for direct comparison.

### Issues
1. **FLAG (minor)**: All values are figure approximations (read from bar charts in Figure 1). Some imprecision is expected. No table data available for these measurements.
2. Text-reported % changes are consistent with extracted directions.

---

## 034_Johnson_1997 (actually Johnson et al. 2003)

**Paper:** "The effects of elevated CO2 on nutrient distribution in a fire-adapted scrub oak forest"
**Extraction:** 44 observations, 11 elements (N, P, K, Mg, Ca, S, Zn, Mn, Fe, Cu, B), 2 species (Q. geminata, Q. myrtifolia), 2 tissues (foliage, stem)

### Verdict: FLAG

### Elements in text vs extraction
- **MATCH**: All 11 mineral elements from Table 3 are captured.
- **No missing elements**.

### Direction checks
- **N decreased**: Text states "Elevated CO2 caused reduced tissue nutrient concentrations of N and S." JSON shows N negative for foliage in both species. **CONSISTENT**.
- **S decreased**: Text confirms. JSON shows S negative for foliage. **CONSISTENT**.
- **Mn INCREASED**: Text states "significantly greater concentrations of manganese." JSON: Q. geminata foliage +40.68%, Q. myrtifolia foliage +36.51%. **CONSISTENT**.
- **B decreased**: Text states "generally lower boron (B) concentrations." JSON shows B negative. **CONSISTENT**.
- **P decreased in some species**: Text says P, Ca, Mg decreased "in some cases, but not K." JSON: P negative for Q. geminata foliage, near zero for Q. myrtifolia. **CONSISTENT** (species-specific).
- **K NOT decreased**: Text says K not affected. JSON: K near zero or slightly positive. **CONSISTENT**.

### Specific value checks
- **POSSIBLE ERROR**: Q. geminata foliage Ca: JSON has control_mean=0.85, but Table 3 in the paper shows ambient Ca for Q. geminata foliage = 0.70 (% dry weight). The treatment_mean=0.59 appears correct. If control should be 0.70, the effect would be -15.7% instead of the extracted -30.6%.
- This needs manual verification against Table 3 in the original PDF.

### Issues
1. **FLAG**: Possible Ca value error for Q. geminata foliage (control_mean=0.85 vs paper Table 3 value of 0.70). This would change the effect size from -30.6% to -15.7%.
2. All other directions and values appear consistent with text.

---

## 035_Oksanen_2005

**Paper:** "Structural characteristics and chemical composition of birch (Betula pendula) leaves are modified by increasing CO2 and ozone"
**Extraction:** 11 observations, 11 elements (N, P, K, Ca, Mg, Mn, Fe, Zn, Cu, B, S), pooled clones 4+80, short shoot leaves

### Verdict: FLAG (borderline PASS)

### Elements in text vs extraction
- **MATCH**: All 11 elements from Table 8 are captured (N, P, K, Ca, Mg, Mn, Fe, Zn, Cu, B, S).
- **No missing elements**.

### Direction checks
- **N decreased**: Text states "Elevated CO2 decreased significantly the foliar concentration of N." JSON: -14.98%. **CONSISTENT**.
- **K decreased**: Text confirms K decreased significantly. JSON: -13.54%. **CONSISTENT**.
- **Cu decreased**: Text confirms. JSON: -17.90%. **CONSISTENT**.
- **S decreased**: Text confirms. JSON: -17.61%. **CONSISTENT**.
- **Fe decreased**: Text confirms. JSON: -15.75%. **CONSISTENT**.
- **Mn NOT significantly affected by CO2 alone**: Text says Mn not significantly affected. JSON: +2.46% (small, non-significant). **CONSISTENT**.
- **P decreased (not significant)**: Text does not highlight P as significant. JSON: -3.70% (small effect). **CONSISTENT**.

### Specific value checks
- Values match Table 8 CC (chamber control) vs EC (elevated CO2) columns.
- Notes correctly specify "Averages of July 2000 and 2001" matching Table 8 presentation.

### Issues
- **FLAG (minor only)**: This extraction is clean. The only minor note is that the paper has a CO2 x O3 factorial design, and the extraction correctly takes only the CO2 effect (CC vs EC, no ozone). This is the correct comparison for the Loladze meta-analysis.

---

## 036_Schenk_1997

**Paper:** "The response of perennial ryegrass/white clover mini-swards to elevated atmospheric CO2 concentrations: effects on yield and fodder quality"
**Extraction:** 36 observations, 7 elements (N, P, K, S, Mg, Ca, Na), 3 tissues (ryegrass, clover, total yield), 2 years (1992, 1993)

### Verdict: FLAG

### Elements in text vs extraction
- **MATCH**: All 7 macroelements from Table 1 are captured (N, P, K, S, Mg, Ca, Na).
- **No missing elements**: Paper only reports these 7 elements.

### Direction checks
- **K decreased in total yield**: Text states "K and Na content of total yield was decreased by high CO2." JSON: K total yield 1992: -18.18%, 1993: -16.89%. **CONSISTENT**.
- **Na decreased in total yield**: JSON: Na total yield 1992: -20.63%, 1993: -24.75%. **CONSISTENT**.
- **Ca INCREASED in total yield**: Text states "Ca content of total yield was increased." JSON: Ca total yield 1992: +10.67%, 1993: +9.03%. **CONSISTENT**.
- **P not changed**: Text states "P content was not changed." JSON: P total yield 1992: -3.88%, 1993: +4.40%. Small mixed effects. **CONSISTENT** with no significant change.
- **N decreased in ryegrass**: Text states N decreased in ryegrass. JSON: N ryegrass 1992: -9.66%, 1993: -12.64%. **CONSISTENT**.
- **N stable in clover**: Clover N relatively stable due to N fixation. JSON: N clover 1992: -1.07%, 1993: -2.70%. Very small effects. **CONSISTENT**.
- **Mg unchanged**: Text does not highlight Mg. JSON: mixed small effects. **CONSISTENT**.
- **S unchanged**: Text does not highlight S. JSON: mixed small effects. **CONSISTENT**.

### Specific value checks
- All data correctly sourced from Table 1, 75:25 mixture only.
- Values match Table 1 exactly (no figure approximation needed).

### Issues
1. **FLAG (minor)**: "Total yield" is a mixed sward measurement (ryegrass + clover combined), not a single-species tissue type. This is unusual for the Loladze meta-analysis which typically uses species-specific measurements. Consider whether "total yield" observations should be included or if only species-specific (ryegrass, clover) observations should be used.
2. All values and directions are correct.

---

## Summary

| Paper | Status | Obs | Elements | Key Issue |
|-------|--------|-----|----------|-----------|
| 031_Pal_2003 | FLAG | 15 | 5 (C, N, P, Ca, Fe) | Organic C may not belong in mineral meta-analysis |
| 032_Kanowski_2001 | FLAG | 20 | 5 (N, P, K, Ca, Na) | Figure approximations (no table available) |
| 034_Johnson_1997 | FLAG | 44 | 11 | Possible Ca value error (0.85 vs 0.70 in paper) |
| 035_Oksanen_2005 | FLAG (borderline PASS) | 11 | 11 | Clean extraction; minor factorial design note |
| 036_Schenk_1997 | FLAG | 36 | 7 (N, P, K, S, Mg, Ca, Na) | "Total yield" tissue type question |

## Priority Actions

| Priority | Paper | Action |
|----------|-------|--------|
| HIGH | 034_Johnson_1997 | Verify Q. geminata foliage Ca control_mean (0.85 in JSON vs 0.70 in Table 3) |
| MEDIUM | 031_Pal_2003 | Decide whether to exclude organic carbon (C%) observations |
| MEDIUM | 036_Schenk_1997 | Decide whether to include "total yield" (mixed sward) observations |
| LOW | 032_Kanowski_2001 | Accept figure approximations or flag with wider error bounds |
| LOW | 035_Oksanen_2005 | No action needed (clean extraction) |
| LOW | 031_Pal_2003 | Correct ambient CO2 from 350 to 360 in notes |
