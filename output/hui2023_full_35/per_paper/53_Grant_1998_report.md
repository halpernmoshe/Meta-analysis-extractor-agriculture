# Extraction Quality Report: 53_Grant_1998
**Match summary:** no_gt

---

## 1. Paper Design

**Full citation:** Grant, C.A., Bailey, L.D., 1998. Nitrogen, phosphorus and zinc management effects on grain yield and cadmium concentration in two cultivars of durum wheat. *Canadian Journal of Plant Science*, 78(1), 63-70.

**Country:** Canada
**Crop:** Durum wheat (*Triticum turgidum*) - two cultivars: Medora and Sceptre
**System:** Multi-year field experiment (1991-1993), scanned/OCR PDF
**Design:** Randomized complete block with 4 replications; factorial N × P × Zn fertilizer treatments across two soil types (clay loam, silty clay)
**Primary outcome measured in the paper:** Grain yield (kg ha-1) and grain **cadmium (Cd)** concentration (mg kg-1)
**Zinc role:** Applied as a soil treatment (ZnSO4, 10 kg Zn ha-1) to test whether Zn fertilization reduces cadmium uptake into grain — **not** to biofortify grain with Zn
**Tables:** 7 tables — soil properties (T1), meteorological data (T2), ANOVA F-values for yield and Cd (T3), year-averaged Cd and yield (T4), yield on clay loam (T5), yield on silty clay (T6), grain Cd response (T7)

---

## 2. AI Extraction

| Model | Observations extracted | Element(s) |
|-------|------------------------|------------|
| Claude | 0 | — (correctly declined) |
| Kimi | 48 | grain yield (all 48) |
| Gemini | 0 | — |
| **Consensus** | **47** (matched) | **grain yield only** |

Kimi extracted 47-48 observations from Tables 5 and 6, covering grain yield comparisons between Zn-applied treatments (Treatments 2, 8, 11) and the unfertilized control (Treatment 1), broken out by cultivar, year, and soil type. Variance was correctly identified as SE (e.g., "SE 169.3" from Table 5), n=4. No grain Zn concentration values were extracted by any model because none are reported in the paper.

**Recon warnings generated (correct):**
- "This paper focuses on cadmium concentration, NOT zinc concentration in grain"
- "Zinc is applied as a treatment but grain Zn concentration is NOT measured or reported"
- "The study examines whether Zn application affects Cd concentration (potential antagonistic effect), not Zn biofortification"
- Extraction guidance: "This paper should be excluded from the zinc biofortification meta-analysis"

Claude correctly extracted 0 observations, recognizing the paper is out of scope. Kimi extracted yield observations, which are technically present in the tables, but these are not the target outcome for the Hui 2023 grain Zn meta-analysis.

---

## 3. Why No GT?

The MOESM5 spreadsheet assigns study_id 53 exclusively to **Data 2 (Soil application)** sheet. That sheet records soil moderator variables and grain yield, but its schema contains **no grain Zn concentration column**. The Hui 2023 meta-analysis uses this paper only as a moderator data record for soil context (available Zn: 0.62-1.32 mg kg-1, pH: 7.7-7.9, ZnSO4 application at 10 kg Zn ha-1, n=4, yield 515-5881 kg ha-1 across 30 records) — **not** as a source of grain Zn outcome data.

The underlying reason is the paper's scope: Grant & Bailey (1998) measured grain **cadmium** as a food safety concern, investigating whether Zn fertilization could competitively inhibit Cd uptake. Grain Zn concentration was never an endpoint because agronomic zinc biofortification was not the research question. Hui 2023 included this paper in the meta-analysis dataset solely as contextual information about soil Zn availability and agronomic conditions at sites where other grain Zn effects were studied — or it may have been retained as a study contributing yield data within the broader zinc-soil meta-analytic framework.

**Summary:** 0 grain Zn GT rows is structurally correct; the paper does not report grain Zn concentration. The MOESM5 Soil sheet provides 30 rows of moderator/yield metadata, none of which carry a grain Zn outcome value that the validation script would seek.

---

## 4. Assessment

**No_gt status verdict: Correct and expected.**

The no_gt status is fully explained by the paper's primary focus. Grant & Bailey (1998) is a cadmium safety study, not a grain Zn biofortification study. Grain Zn concentration is absent from all seven tables. The Hui 2023 meta-analysis apparently included this paper only as a source of soil-application context records (Data 2 Soil sheet), which carry no grain Zn outcome.

**AI performance: Excellent at recon, acceptable at extraction.**
- The recon phase correctly diagnosed the scope mismatch with high-confidence warnings across all three models, and issued an explicit "exclude from meta-analysis" guidance statement.
- Claude's decision to return 0 observations was the correct response to those warnings.
- Kimi's extraction of 47 grain yield observations is technically faithful to the paper but wrong for the target outcome; however, these observations are harmlessly discarded by the validation filter (no grain Zn column to match).
- No hallucinated grain Zn values were produced by any model.

**Action required:** None. This paper is correctly handled as out-of-scope for grain Zn extraction. No re-extraction is warranted.
