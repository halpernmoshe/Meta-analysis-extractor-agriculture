# Extraction Quality Report: 65_Oliver_1994

**Match summary:** no_gt

---

## 1. Paper Design

**Full citation:** Oliver, D.P., Hannam, R., Tiller, K.G., Wilhelm, N.S., Merry, R.H., Cozens, G.D., 1994. The effects of zinc fertilization on cadmium concentration in wheat-grain. *Journal of Environmental Quality*, 23(4), 705–711.

**Country:** Australia
**Crop:** Wheat (*Triticum aestivum* L.)
**Study system:** 12 field experiments across South Australia (Eyre Peninsula and Mallee regions)
**Experimental design:** Multi-site dose-response; zinc sulfate (ZnSO4) applied at 0, 5, 10, 15, 20, and 25 kg Zn ha-1 (fresh or residual). Replication: 4–6 plots per treatment depending on site.

**Primary research question:** Does soil zinc fertilization reduce cadmium (Cd) accumulation in wheat grain?

**Tables reported:**
- Table 1: Site locations, years, wheat variety, replication details
- Table 2: Soil characteristics (pH, texture, EDTA-extractable Zn and Cd)
- Table 3: Exponential equation parameters for grain Cd vs. Zn rate
- Table 4: Mean grain yield (t ha-1) at each Zn application rate, with SED

**Figures reported:** Figure 2, Figure 3 (grain Cd concentration vs. Zn rate curves)

---

## 2. AI Extraction

The consensus pipeline extracted **12 observations** (post-processing: 15 raw, 3 null-mean rows removed, 0 duplicates removed). All 12 observations came from Claude only; Kimi extracted 0 and Gemini extracted 0, so the tiebreaker fell to the single-model fallback with confidence marked **low**.

**What was extracted:** Grain yield (t ha-1) from Table 4 for 10 of the 12 experiments (Exps 1–7, 9–12), comparing 5 kg Zn ha-1 treatment vs. 0 kg Zn ha-1 control. Variance type reported as SED (standard error of difference) drawn from the SED column in Table 4. Sample sizes n = 4 or 6 per site, consistent with Table 1.

**Example extracted observations:**

| Exp | Location | Control (t/ha) | Treatment (t/ha) | Effect | n | Variance |
|-----|----------|---------------|-----------------|--------|---|----------|
| 1 | Tuckey, EP | 1.10 | 1.17 | +6.4% | 4 | SED=0.43 |
| 6 | Lameroo site2, MM | 1.20 | 2.38 | +98.3% | 4 | SED=0.21 |
| 9 | Ungarra, EP | 5.21 | 4.88 | -6.3% | 4 | SED=3.04 |

**Verification flags:** All 12 observations failed GRIM (continuous yield data expressed in t/ha with two decimal places, so GRIM failure is expected and not meaningful). All but one also failed the variance_type heuristic check (SED flagged as ambiguous vs. SD by the CV heuristic). One observation (Exp 9) additionally failed direction and triggered a T/C swap warning, though the negative effect for that site is genuinely non-significant in the paper.

**Model agreement:** Single-model output only (Claude). No cross-model consensus was possible.

---

## 3. Why No GT? (What does the paper measure — not grain Zn?)

**The paper reports cadmium concentration in wheat grain, not zinc concentration.**

The study's central finding is that applying zinc fertilizer to low-Cd soils reduces cadmium uptake into grain — a cadmium contamination remediation study, not a zinc biofortification study. The primary outcome reported in Tables and Figures is grain Cd concentration (mg Cd kg-1 grain), modeled as an exponential decay function of Zn application rate (Table 3, Figures 2 and 3).

**Grain Zn concentration is never reported.** The paper contains no table, figure, or in-text value for grain Zn content. Zinc is used solely as a soil amendment to displace cadmium from the food chain.

**Why study_id=65 appears in MOESM5 at all:** The Hui 2023 meta-analysis included this paper in the **"Data 2 Soil application"** sheet, but only to extract agronomic moderator variables (site soil available Zn, pH, CaCO3, organic matter, N/P/K rates, Zn rate applied, replication count, and initial grain yield in kg ha-1). The MOESM5 "Data 2 Soil" sheet for study_id=65 contains 27 rows (Observation IDs 610–636) spanning multiple Zn application rates (5, 10, 15, 20 kg Zn ha-1) and sites, but the only outcome column populated is `Grain yield (kg ha-1)` — a moderator covariate, not the meta-analysis target outcome (grain Zn concentration mg kg-1).

The MOESM5 "Data 3 Foliar application" and "Data 4 Soil+Foliar" sheets contain no rows for study_id=65.

**Conclusion:** The absence of grain Zn rows in the ground truth is correct and expected. Oliver 1994 is a Cd-remediation paper that cannot contribute any grain Zn effect size to the Hui 2023 meta-analysis.

---

## 4. Assessment

**Extraction verdict: Irrelevant paper — no_gt status is fully justified.**

The AI pipeline correctly identified this mismatch. The recon phase issued explicit, accurate warnings:

> "This paper studies cadmium (Cd) concentration, NOT zinc (Zn) concentration — it may not be relevant for the meta-analysis."
> "No direct grain Zn concentration data is reported in this paper."
> "Tables focus on soil characteristics and Cd concentrations only."
> "Extraction guidance: This paper is NOT suitable for the Zn biofortification meta-analysis."

Despite this, Claude's extraction pass nonetheless extracted 12 grain yield observations from Table 4, which is the only numerically structured table with treatment/control means. This is a reasonable fallback behavior when no target-outcome data is found, but the extracted variable (grain yield in t/ha) is not grain Zn and cannot be matched to ground truth.

**AI performance summary:**

| Criterion | Result |
|-----------|--------|
| Correct paper identification | Yes — cadmium study, not Zn biofortification |
| Correct recon warning | Yes — explicitly flagged as non-target paper |
| Extraction of correct variable | No — extracted grain yield, not grain Zn (none available) |
| no_gt classification | Correct — paper has no matchable GT rows |
| Action required | None — paper should remain excluded |

**Recommendation:** No re-extraction warranted. Oliver 1994 should be flagged as excluded from the grain Zn meta-analysis with the reason "primary outcome is grain cadmium concentration; grain zinc concentration not reported." It may be retained in the dataset as a soil-Zn application moderator-data source only, consistent with its role in MOESM5 Data 2.
