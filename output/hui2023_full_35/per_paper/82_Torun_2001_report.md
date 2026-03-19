# Extraction Quality Report: 82_Torun_2001
**Match summary:** no_gt

---

## 1. Paper Design

**Citation:** Torun, A., Gultekin, I., Kalayci, M., Yilmaz, A., Eker, S., Cakmak, I. (2001). Effects of zinc fertilization on grain yield and shoot concentrations of zinc, boron, and phosphorus of 25 wheat cultivars grown on a zinc-deficient and boron-toxic soil. *Journal of Plant Nutrition*, 24(11), 1817–1829.

**Study type:** Two-year field experiment (1993–1994 and 1994–1995) at the Bahri Dagdas International Winter Cereals Research Center, Konya, Central Anatolia, Turkey.

**Design:** Two-factor randomized complete block, strip-plot layout, n = 4 replications. Factors: Zn treatment (2 levels: -Zn = no Zn added; +Zn = 23 kg Zn ha⁻¹ as ZnSO₄·7H₂O) × cultivar (25: 21 bread wheat *T. aestivum* + 4 durum wheat *T. durum*).

**Soil:** Zn-deficient (DTPA-Zn = 0.10 mg kg⁻¹), pH 7.8, CaCO₃ 38%, organic matter 20 g kg⁻¹. Simultaneously boron-toxic (B = 11 mg kg⁻¹, far above the 0.5 mg B kg⁻¹ normal threshold).

**Outcomes measured and reported:**
- **Table 1:** Grain yield (t ha⁻¹) for all 25 cultivars × 2 years × 2 Zn treatments. Mean yields: -Zn = 2.57 t ha⁻¹, +Zn = 4.09 t ha⁻¹ (year 1); -Zn = 2.21, +Zn = 3.44 (year 2).
- **Table 2:** Zinc efficiency ratio (%) = (grain yield -Zn / grain yield +Zn) × 100 for all 25 cultivars × 2 years.
- **Table 3:** Shoot concentrations of Zn (mg kg⁻¹ DW), B (mg kg⁻¹ DW), and P (mg g⁻¹ DW) at the tillering stage, for all 25 cultivars, year 1 only. Mean shoot Zn: 9 (-Zn) vs. 19 (+Zn) mg kg⁻¹ DW.

**What the paper does NOT report:** Grain Zn concentration (mg kg⁻¹ grain DW). The paper explicitly states at the abstract level that it studies "grain yield, shoot concentrations of zinc, boron, and phosphorus." Shoot samples were collected "at the end of tillering" — a vegetative stage — not at grain maturity. No table or figure anywhere in the paper reports Zn content measured in harvested grain.

**Variance:** LSD at P ≤ 0.05. LSD values are given at the foot of Tables 1 and 3. No numeric variance values (SE, SD) are reported; only LSD thresholds per source of variation (cultivar, Zn treatment, interaction). Numeric means only; variance values are not extractable per-observation.

---

## 2. AI Extraction

**Recon phase (correct):** The recon module correctly identified that the paper lacks grain Zn concentration data. The recon JSON explicitly states:

> "This paper does NOT contain the primary outcome data (grain Zn concentration) needed for the meta-analysis. It only reports shoot Zn concentrations at tillering stage and grain yields. The study should be excluded from extraction as it lacks grain Zn concentration data at harvest."

`tables_with_target_data: []` — all three tables were correctly placed in `tables_without_target_data`.

**Extraction phase (incorrect outcome selection):** Despite the recon warning, Claude extracted 54 observations from Table 1, all for grain yield (t ha⁻¹), not grain Zn concentration. These are 25 cultivars × 2 years = 50 data rows, plus some averages. Kimi extracted 0 observations (correctly declined). Gemini also extracted 0. The tiebreaker logic awarded Claude's extraction by default ("Kimi extracted 0 obs, Claude extracted 54"), resulting in 54 grain-yield observations entering the consensus output.

**Extracted element:** `grain yield (t ha-1)`, tissue = `grain`. Example observation: BDME 9, year 1993–1994, control mean = 3.46 t ha⁻¹, treatment mean = 4.40 t ha⁻¹, effect +27.2%, n = 4, variance = null (LSD type flagged but no numeric value).

**No grain Zn observations were extracted**, because none exist in the paper.

---

## 3. Why No GT?

The MOESM5 spreadsheet for the Hui 2023 meta-analysis is structured by outcome type across three sheets:
- **Data 2 Soil application** — grain Zn concentration as a function of soil Zn application
- **Data 3 Foliar application** — grain Zn concentration as a function of foliar Zn spray
- **Data 4 Soil + Foliar** — grain Zn concentration with combined application

Study ID 82 (Torun 2001) appears in **Data 2 Soil application** with 50 rows (Observation IDs 777–826). However, inspection of the gt file confirms that every one of these 50 rows contains only:
- Soil moderator variables (available Zn, pH, CaCO₃, organic matter, N/P/K rates, Zn rate)
- `Grain yield (kg ha⁻¹)` values per cultivar
- n = 4

There are **no grain Zn concentration columns** (no control Zn, no treatment Zn, no effect size) in these rows. The Hui et al. meta-analysis used this paper solely as a source of moderator/contextual data (soil properties, yield response to Zn fertilization) for modelling the soil-application effect on grain Zn across the dataset, without extracting a grain Zn effect size from Torun 2001 directly.

**Root cause of no_gt:** Torun (2001) never measured grain Zn concentration. Hui et al. included this paper in their database only as a yield-response record, not as a grain-Zn record. Therefore the "Data 2 Soil" sheet rows for study_id = 82 contain no grain Zn control/treatment pair, and the validation script correctly found 0 matchable grain Zn rows — producing the `no_gt` status.

---

## 4. Assessment

**Extraction correctness:** The AI extraction is technically a failure relative to the meta-analysis target outcome. The 54 extracted observations are for grain yield, not grain Zn concentration, which is the only matchable outcome for the Hui 2023 validation. However, this is an unavoidable outcome: the paper genuinely contains no grain Zn concentration data.

**Recon quality: Excellent.** The recon phase correctly flagged the paper as lacking the primary outcome and explicitly recommended exclusion. The extraction phase should have respected this guidance; the tiebreaker logic that awarded Claude's grain-yield extraction over Kimi's zero-extraction was the proximate cause of the mismatch.

**GT status: Correct.** `no_gt` is the right classification. Torun (2001) is a paper about cultivar variation in Zn efficiency (yield-based) and shoot Zn/B/P at tillering. It is included in the Hui 2023 dataset as a moderator-context record for soil Zn application conditions, not as a grain Zn effect-size source.

**Impact on validation statistics:** Zero. Because no grain Zn GT rows exist for this paper, it contributes nothing to the r, MAE, or ICC calculations. The 54 spuriously extracted grain-yield observations are not matched against any GT rows and are silently dropped by the validation matcher.

**Recommendation:** This paper should be hard-excluded at the recon stage for the grain Zn extraction pipeline. The recon warning was correct; enforcement of the exclusion at the extraction stage should be tightened (e.g., if `tables_with_target_data` is empty and the recon provides an explicit exclusion recommendation, skip extraction automatically rather than defaulting to the tiebreaker).
