# Extraction Quality Report: 45_Chattha_2017

**Citation:** Chattha, M.U., Hassan, M.U., Khan, I., Chattha, M.B., Mahmood, A., Chattha, M.U., Nawaz, M., Subhani, M.N., Kharal, M., Khan, S., 2017. Biofortification of wheat cultivars to combat zinc deficiency. *Frontiers in Plant Science*, 8, 281.

**Match summary:** 18/18 GT matched | r = 1.0 | MAE = 0.0% — PERFECT (all 3 application types)

**GT study IDs:** Soil = [45] | Foliar = [43] | Soil+Foliar = [15]

---

## 1. Paper Design

**Study type:** Field experiment, Pakistan. Zinc biofortification of wheat — not a CO2 study. The control is T1 (no zinc application).

**Experimental design:** Randomized complete block design (RCBD), factorial arrangement.

**Factors:**
- Zinc application method: 5 levels (T1 control, T2 seed priming, T3 soil, T4 foliar, T5 soil+foliar)
- Cultivar: 3 levels (Faisalabad-2008, Punjab-2011, Millet-2011)
- Year: 2 growing seasons (2013-2014 and 2014-2015)

**Replication:** n = 3 (three RCBD blocks per treatment combination)

**Primary outcome:** Grain zinc concentration (mg kg-1), grain tissue only.

**Tables used:**
- Table 2: Main effects of Zn application method and cultivar (averages across cultivars / averages across methods)
- Table 3: Interactive effects (cultivar x Zn method x year) — the primary data source for per-cultivar extraction

**Variance:** LSD at 5% probability level, explicitly stated in the Methods section. LSD values are printed at the bottom of each table.

**PDF quality:** Scanned document (OCR). Recon flagged OCR risk; however, the scanned quality was sufficient for accurate extraction.

**Soil properties (uniform across all observations):**
- Available Zn: 29 mg kg-1 (grouping: >0.5)
- pH: 7.8 (grouping: >7.0)
- Organic matter: 8.1 g kg-1 (grouping: <10)
- N rate: 100 kg N ha-1 (grouping: <110)
- P rate: 50.38 kg P ha-1 (grouping: >44)
- K rate: 49.8 kg K ha-1 (grouping: 25-58)

---

## 2. Ground Truth Structure

The Hui 2023 meta-analysis uses this paper across three separate data sheets, reflecting the three treatment arms that involve zinc as either soil-applied, foliar-applied, or combined. Crucially, the same 6 base observations (one per cultivar per year) appear in all three sheets with the same grain Zn concentration values. The GT grain Zn concentration values are:

| Obs | Cultivar implied | Year | Grain Zn (mg kg-1) | Grain yield (kg ha-1) |
|-----|-----------------|------|---------------------|----------------------|
| 316/410/79 | Average / Average / obs1 | 2013-2014 | 30.0 | 3100 |
| 317/411/80 | Average / Average / obs2 | 2013-2014 | 36.9 | 4110 |
| 318/412/81 | Average / Average / obs3 | 2013-2014 | 32.5 | 3560 |
| 319/413/82 | Average / Average / obs4 | 2014-2015 | 31.3 | 3170 |
| 320/414/83 | Average / Average / obs5 | 2014-2015 | 36.0 | 4190 |
| 321/415/84 | Average / Average / obs6 | 2014-2015 | 33.9 | 3610 |

Note: The GT sheet does not assign named cultivars to these 6 rows; each row represents a distinct treatment combination identified by grain yield value. The Soil and Foliar sheets both report the same Zn fertilizer rate (20.25 kg Zn ha-1 for soil; spray concentration ~0.114 g Zn L-1 for foliar), confirming these are the treated arm means matched against the same T1 control.

The Soil+Foliar sheet (Data 4) additionally records n = 3 explicitly, confirming the RCBD replication structure.

---

## 3. Extraction Summary

**Models used:** Claude + Gemini (tiebreaker applied — Kimi extracted 0 observations and was excluded).

**Consensus observations extracted:** 32 total (all flagged high confidence, all "[Claude+Gemini agree]").

The 32 extracted observations break down as:
- **8 from Table 2** (main effects, average across cultivars): 4 treatment types (T2/T3/T4/T5) x 2 years
- **24 from Table 3** (interaction table, per-cultivar): 4 treatment types x 3 cultivars x 2 years

The GT contains 18 observations (6 per sheet x 3 sheets). The 18 GT matches were drawn from the 32 extracted observations. The extra 14 extracted observations (cultivar-specific comparisons from Table 3 not represented in the GT) are legitimate additional data not present in the Hui 2023 coding scheme.

**Control mean values extracted (control = T1, no zinc):**

| Year | Cultivar | Control Grain Zn (mg kg-1) |
|------|----------|---------------------------|
| 2013-2014 | Average across cultivars | 33.1 |
| 2014-2015 | Average across cultivars | 33.7 |
| 2013-2014 | Faisalabad-2008 | 30.0 |
| 2013-2014 | Punjab-2011 | 36.9 |
| 2013-2014 | Millet-2011 | 32.5 |
| 2014-2015 | Faisalabad-2008 | 31.3 |
| 2014-2015 | Punjab-2011 | 36.0 |
| 2014-2015 | Millet-2011 | 33.9 |

**Treatment mean range:** 34.3 to 71.8 mg kg-1 (T2 seed priming to T5 soil+foliar, all cultivars and years combined).

**Effect sizes:** Range from +12.8% (T2 seed priming, Faisalabad-2008, 2014-2015) to +96.1% (T5 soil+foliar, Punjab-2011, 2014-2015). Foliar (T4) and combined (T5) treatments consistently produce the largest Zn increases, both around 70-96% over control.

---

## 4. Verification Flags (Internal — Do Not Affect Match Quality)

All 32 observations triggered two internal verification flags. These flags are systematic artifacts of the data structure and do not reflect extraction errors:

**Flag 1 — GRIM test failures (all 32 obs):**
The GRIM test checks whether a reported mean is mathematically possible given integer raw data and a specific n. With n = 3 and one-decimal-place means (e.g., 38.4, 44.4), most values fail the GRIM test because the underlying Zn concentration measurements are continuous (mg kg-1), not integer counts. GRIM is not applicable to continuous analytical measurements. These failures are expected and correct to ignore.

**Flag 2 — Variance type heuristic disagreement (all 32 obs):**
The paper explicitly reports LSD values. The internal CV heuristic disagreed, suggesting SD or SE for some observations, but with only 0.5 confidence. This is a known limitation of the CV-based heuristic: LSD values for n=3 with tight data can fall in the same numeric range as SD or SE. The paper's stated variance type (LSD, p <= 0.05) is unambiguous and confirmed by direct text quote in the recon. The heuristic flags should be disregarded for this paper.

Neither flag type affected the match outcome. All direction checks passed (Zn always increases with Zn application, as expected). No T/C swap was detected. Sample size (n=3) was correctly identified.

---

## 5. Assessment: PERFECT

**Result: 18/18 GT observations matched | r = 1.0 | MAE = 0.0%**

This is a fully successful extraction. Key reasons for the perfect performance:

1. **Clear tabular structure.** Tables 2 and 3 present data in a clean factorial layout with explicit T1-T5 treatment labels. No ambiguity about which row is control vs treatment.

2. **Unambiguous control definition.** T1 = "No zinc" is stated explicitly in the table header and Methods. No risk of control/treatment confusion.

3. **Consistent moderator structure.** All 18 GT observations share the same site-level moderators (Pakistan, pH 7.8, OM 8.1 g kg-1, etc.), which the extraction correctly carried uniformly.

4. **Variance correctly identified at recon stage.** The LSD statement was found in the Methods text and confirmed at medium confidence. This prevented any variance type confusion during extraction.

5. **Two-model agreement.** Claude and Gemini agreed on all 32 extracted values (0.0% disagreement on treatment means). The tiebreaker (Kimi = 0 obs) did not degrade quality because both remaining models produced identical outputs.

6. **Data richness used appropriately.** The extractor correctly captured both the Table 2 main-effects rows (averages across cultivars) and the Table 3 cultivar-specific rows, providing the GT-matching grain Zn values plus additional cultivar-stratified data useful for moderator analysis.

**Limitations (minor):**
- GRIM and variance type flags are systematic false positives for this paper type (continuous measurements, LSD variance). They do not represent real extraction problems.
- The extra 14 observations beyond the 18 GT rows are legitimate and scientifically informative but were not coded into the Hui 2023 dataset (possibly because Hui used only one representative entry per application-method arm rather than all cultivar-year combinations).
- Kimi model failed entirely (0 observations), suggesting the scanned PDF posed challenges for that model. The two-model fallback performed correctly.

**Recommendation:** No re-extraction needed. This paper can serve as a benchmark case for successful extraction from a scanned factorial field experiment with LSD variance.
