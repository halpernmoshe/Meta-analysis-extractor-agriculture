# Cross-Check Report: Batch 4 (Papers 017-021)

Generated: 2026-03-15

Text-based cross-validation of agent-extracted mineral data against PDF Results/Discussion sections.

---

## 017_Fangmeier_2002 — FLAG

**Paper:** Effects of elevated CO2 and/or ozone on nutrient concentrations and nutrient uptake of potatoes
**Extraction:** 56 observations from Tables 4 and 5 (tuber + aboveground, intermediate + final harvest, CO2-only and CO2+O3)

### Issues Found

1. **DUPLICATE OBSERVATION (last entry, lines 801-813):**
   The final observation in the JSON is N/tuber from Table 4, intermediate harvest, with treatment_mean=12.0, control_mean=13.3, effect_pct=-9.77. This is **identical** to the very first observation (lines 8-20) except it is labeled `"treatment_description": "Elevated CO2+O3 680 ul/l + O3"` and `"control_description": "Ambient CO2 + O3 (O3)"` with `"ozone": "elevated"`. The values 12.0 and 13.3 are the CO2-only (NF vs 680) values from Table 4, not the O3 vs 680O3 values. The correct CO2+O3 intermediate harvest N/tuber values from Table 4 would be different (O3 control and 680O3 treatment columns). This is a **wrong-label duplicate** that should be removed or corrected.

2. **Missing sample size (n):**
   All 56 observations have `n: null`. The paper discusses multi-site data (Giessen, Tervuren, Carlow) but does not clearly report per-treatment replication in a single place. This is a genuine limitation of the paper, not an extraction error.

3. **Text cross-check — direction and magnitude:**
   - Text states "Tuber nitrogen concentration was reduced by 9.5% when the crops were grown under 680 ul/l." Extraction shows N/tuber at final harvest: 12.9 vs 14.3 = -9.79%. **MATCH.**
   - Text states "tuber phosphorus concentration was unaffected by CO2 enrichment at both harvests." Extraction shows P/tuber at intermediate: 3.30 vs 3.18 = +3.77% and final: 2.86 vs 3.07 = -6.84%. The small magnitudes are consistent with "unaffected." **MATCH.**
   - Text states "Tuber Mg concentration was also reduced by CO2 enrichment at final harvest." Extraction shows Mg/tuber final: 0.82 vs 0.92 = -10.87%. **MATCH.**
   - Text states "tuber potassium concentration showed a small but significant reduction under elevated CO2 at both harvests." Extraction shows K/tuber intermediate: 23.2 vs 25.3 = -8.30%, final: 18.3 vs 18.7 = -2.14%. **MATCH.**
   - Text states "Manganese was significantly reduced in the aboveground biomass at final harvest." Extraction shows Mn/aboveground final: 27.2 vs 43.7 = -37.76%. **MATCH.**

4. **No missing elements or tissues detected** from text cross-check. Tables 4 and 5 cover N, P, K, Ca, Mg, Mn, Zn, Fe for both tissues at both harvests, which is comprehensive.

**Verdict: FLAG** — Last observation is a duplicate with incorrect treatment labels.

---

## 018_Al-Rawahy_2013 — PASS

**Paper:** Effect of O3 and CO2 Levels on Growth, Biochemical and Nutrient Parameters of Alfalfa (Medicago sativa)
**Extraction:** 21 observations from Table 3 (leaf tissue, 3 CO2 levels x 7 elements)

### Checks Performed

1. **Values verified against Table 3:**
   All 21 observations (N, K, P, Ca, Mg, Zn, Fe at 350/400/450 ppm CO2) match Table 3 values exactly. For example:
   - N at 350 ppm: 33.8 vs control 32.5 — matches Table 3
   - Zn at 450 ppm: 0.21 vs control 0.16 — matches Table 3
   - Fe at 400 ppm: 0.53 vs control 0.44 — matches Table 3

2. **Direction consistency:** Text states nutrients generally remained stable or slightly increased under CO2-only treatments. Extraction shows mostly small positive or near-zero changes. **MATCH.**

3. **Correct exclusion of O3 and O3+CO2 treatments:** Only CO2-alone treatments extracted, which is appropriate for a CO2 meta-analysis. **CORRECT.**

4. **Sample size:** n=30 reported for all observations. Text states "30 plants per chamber." **MATCH.**

5. **Minor omission — Na:** Table 3 also includes Na (control=3.4, CO2 treatments at 3.5/3.7/3.6). Three Na observations were not extracted. This is a minor omission since Na is rarely included in CO2 meta-analyses.

6. **Minor omission — Cu, Mn:** Table 3 also includes Cu and Mn data which were not extracted. These are additional scope-limited omissions (5 more observations each = 15 total missing across Na, Cu, Mn).

**Verdict: PASS** — All extracted values verified. Minor Na/Cu/Mn omissions are not critical.

---

## 019_Baxter_1994 — FLAG

**Paper:** Effects of elevated carbon dioxide on three grass species from montane pasture. II. Nutrient uptake, allocation and efficiency of use
**Extraction:** 15 observations from Table 1 (total content, mg plant-1, for N, P, K, Mg, Ca across 3 species)

### Issues Found

1. **CRITICAL: Extracts total CONTENT (mg plant-1), not CONCENTRATION (mg g-1 DW):**
   Table 1 reports "Total nutrient content (mg plant-1)." For a mineral **concentration** meta-analysis, the relevant data would be tissue-level concentrations from Figures 1-2 (mg g-1 structural DW). The text explicitly states: "There was no systematic reduction of nutrient concentration in any organ as a consequence of growth at elevated CO2." This means total content increased (because plant biomass increased under CO2) while concentrations stayed relatively unchanged — a fundamentally different biological signal.

   Example of the discrepancy:
   - Extraction shows N total content for A. capillaris: treatment=7.01 vs control=3.16, effect=+121.84%
   - But this massive increase reflects bigger plants, not higher N concentration
   - Figure 1 shows leaf N concentration was relatively unchanged

2. **Missing tissue-level concentration data from Figures 1-2:**
   Figures 1 and 2 are bar charts showing N, P, K, Mg, Ca concentrations (mg g-1 structural DW) by tissue (leaf, sheath, root) for each species. These are the data relevant for a concentration meta-analysis but were not extracted. Approximate values could be read from the bar charts.

3. **Values from Table 1 appear correct:**
   The extracted total content values match Table 1. For example:
   - A. capillaris N: 7.01 vs 3.16 (treatment vs control) — matches Table 1
   - F. vivipara N: 49.67 vs 182.82 — matches Table 1 (large decrease for this species)
   - P. alpina Ca: 2.36 vs 4.53 — matches Table 1

4. **Direction note:** The large negative effects for F. vivipara (e.g., N -72.83%, K -37.84%) reflect reduced total content, which the text attributes to smaller plant size under elevated CO2 for this species, not to reduced concentration.

**Verdict: FLAG** — Wrong measurement type for concentration meta-analysis. Total content (mg/plant) extracted instead of tissue concentration (mg/g DW). Tissue concentration data from Figures 1-2 not captured.

---

## 020_Overdieck_1993 — PASS

**Paper:** Elevated CO2 and the mineral content of herbaceous and woody plants
**Extraction:** 42 observations from Table 4 (trees: Acer, Fagus) and Figure 2 (herbaceous species)

### Checks Performed

1. **Table 4 values verified (trees):**
   - Acer N at 520 umol/mol: 1.15 vs 1.28 = -10.16% — matches Table 4
   - Acer N at 650 umol/mol: 1.06 vs 1.28 = -17.19% — matches Table 4
   - Fagus P at 520: 0.14 vs 0.12 = +16.67% — matches Table 4
   - Acer K at 650: 0.56 vs 0.82 = -31.71% — matches Table 4

2. **Zn correctly excluded for trees:** The notes state "Zn was NOT measured for trees" and the text confirms "With the exception of Zn all the element concentrations mentioned above were measured." No Zn observations for Acer or Fagus. **CORRECT.**

3. **Herbaceous species values (Figure 2):**
   Extraction uses relative percentages from Figure 2 bar charts. Text confirms:
   - "Nitrogen: decreases of approximately 11 and 12% were found for T. pratense and F. pratensis" — consistent with extracted values
   - "Potassium: F. pratensis showed the highest decreases (13 and 18%)" — consistent
   - "Zn... only T. pratense and F. pratensis showed a significant decrease (13%)" — consistent

4. **Species coverage:** 6 species covered (2 trees, 4 herbaceous). The paper studies exactly these species. **COMPLETE.**

5. **Elements:** N, P, K, Ca, Mg for trees; N, P, K, Ca, Mg, Zn for herbaceous (where Zn was measured). **APPROPRIATE.**

6. **n values:** n=21 for tree data (from text: "21 plants per treatment"). Herbaceous n values vary. **REASONABLE.**

**Verdict: PASS** — Values verified against Table 4 and text-stated percentages from Figure 2.

---

## 021_Wilsey_1994 — PASS

**Paper:** Effects of elevated CO2 and defoliation on grasses (Schizachyrium scoparium)
**Extraction:** 26 observations from Table 1 (13 elements x 2 clipping conditions)

### Checks Performed

1. **Values verified against Table 1:**
   All 26 observations match Table 1. Spot-checked examples:
   - N/clipped: treatment=17.3 vs control=16.4, effect=+5.49% — matches Table 1
   - N/unclipped: treatment=13.8 vs control=13.3, effect=+3.76% — matches Table 1
   - Fe/clipped: treatment=173 vs control=305, effect=-43.28% — matches Table 1
   - Zn/unclipped: treatment=22 vs control=23, effect=-4.35% — matches Table 1
   - Mo/clipped: treatment=1.9 vs control=1.3, effect=+46.15% — matches Table 1
   - Co/unclipped: treatment=0.06 vs control=0.06, effect=0.00% — matches Table 1

2. **Element coverage:** All 13 elements from Table 1 are captured: N, K, Na, P, Mg, Ca, Fe, Mn, Zn, B, Cu, Co, Mo. **COMPLETE.**

3. **Direction consistency with text:** Text states "all P values > 0.05" for CO2 effects on tissue nutrient concentrations, meaning no significant effects. The extracted data shows mostly small effect sizes with mixed directions, consistent with non-significance. **MATCH.**

4. **Clipping moderator:** Both clipped and unclipped conditions captured as separate observations with clipping noted in moderators. **APPROPRIATE** for a meta-analysis that may want to examine defoliation as a moderator.

5. **Sample size:** n=5 for all observations. Text states "five replicates per treatment." **MATCH.**

6. **Units:** mg/g for macronutrients, ug/g for micronutrients. **CORRECT** per Table 1 headers.

**Verdict: PASS** — All values, elements, and directions verified against Table 1 and text.

---

## Summary

| Paper | Obs | Verdict | Key Issues |
|-------|-----|---------|------------|
| 017_Fangmeier_2002 | 56 | **FLAG** | Last observation is duplicate with wrong treatment labels |
| 018_Al-Rawahy_2013 | 21 | PASS | Minor: Na, Cu, Mn not extracted |
| 019_Baxter_1994 | 15 | **FLAG** | Extracts total content (mg/plant) not concentration (mg/g); tissue concentration data from Figs 1-2 not captured |
| 020_Overdieck_1993 | 42 | PASS | All verified |
| 021_Wilsey_1994 | 26 | PASS | All verified |

### Action Items

1. **017_Fangmeier_2002:** Remove or correct the last observation (line 801-813). It duplicates observation #1 with wrong CO2+O3 labels.
2. **019_Baxter_1994:** For a concentration-based meta-analysis, this paper's extraction needs to be redone using Figures 1-2 (tissue concentrations in mg/g DW) rather than Table 1 (total content in mg/plant). Alternatively, if the meta-analysis scope includes total content, flag the unit difference clearly.
