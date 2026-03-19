# Extraction Quality Report: 69_Ramzan_2020

**Paper:** Ramzan Y, Hafeez MB, Khan S, Nadeem M, Saleem-ur-R, Batool S, Ahmad J (2020). Biofortification with zinc and iron improves the grain quality and yield of wheat crop. *International Journal of Plant Production*, 14(3), 501–510. https://doi.org/10.1007/s42106-020-00100-w

**Match summary:** 0 extracted (no_extraction), 4 GT rows missed

**Validation outcome:** COMPLETE FAILURE for Hui 2023 target variable (grain Zn concentration)

---

## 1. Paper Design

Two-year field experiment (2016–17 and 2017–18) at Wheat Research Institute Farm, Faisalabad, Pakistan (73.87°E, 31.87°N; altitude 184 m).

**Design:** Randomized complete block design (RCBD), factorial, n = 3 replications.

**Treatments (7 total):**
| # | Treatment | Zn source |
|---|-----------|-----------|
| 1 | Control (no Zn or Fe) | — |
| 2 | Foliar 0.5% ZnSO4 | Foliar |
| 3 | Foliar 1% FeSO4 | — (Fe only) |
| 4 | Foliar 0.5% ZnSO4 + 1% FeSO4 | Foliar combined |
| 5 | Soil 10 kg Zn ha-1 | Soil |
| 6 | Soil 12 kg Fe ha-1 | — (Fe only) |
| 7 | Soil 10 kg Zn ha-1 + 12 kg Fe ha-1 | Soil combined |

Foliar sprays applied three times: tillering, booting, and heading stages. Soil application at sowing. Background fertilization: N 114 kg ha-1, P 120 kg ha-1, K 60 kg ha-1.

**Soil properties (Table 1):**
- Year 2016–17 (0–15 cm): pH 7.6, OM 0.084 g kg-1, DTPA Zn 0.52 mg kg-1
- Year 2017–18 (0–15 cm): pH 7.5, OM 0.077 g kg-1, DTPA Zn 0.62 mg kg-1

These soil properties are the moderator variables Hui 2023 uses to stratify observations into meta-analysis groups.

---

## 2. Grain Zn Data in PDF?

**Yes — but in figures only, not tables.**

All grain mineral concentration data (Zn, Fe, Ca, Mg, Cu, protein) are reported exclusively in bar charts (Figures 2 and 3). Tables 2–4 contain only agronomic variables (tillers, plant height, spike length, spikelets per spike, grains per spike, 1000-grain weight, biological yield, grain yield, harvest index).

**Figure 3b** shows grain Zn concentration (mg kg-1) for all 7 treatments × 2 years = 14 bars, displayed side-by-side as paired dark/light bars.

Key numeric values recoverable from text (Results section, p. 506):
- Control year 2016–17: **41.08 mg kg-1** (lowest value, confirmed in text)
- Foliar 0.5% ZnSO4 year 2017–18: **61.33 mg kg-1** (highest value, confirmed in text)
- All other values require reading directly from Figure 3b bars

**Variance:** No numeric variance values are reported for the grain mineral data in figures. The LSD values given at the bottom of Tables 2–4 apply only to the agronomic variables in those tables; separate LSD values for the figure data are not provided numerically. Statistical significance is indicated by letter codes on bars (A, B, C, etc.) and p-values shown in the figure panel annotation (T: 0.000, Y: 0.4714 for grain Zn).

**Conclusion:** The data exists in the PDF but is figure-embedded and figure-only, with no numeric variance recoverable for grain Zn.

---

## 3. AI Extraction — What Happened

### 3a. Recon phase (correctly diagnosed the problem)

The recon correctly identified:
- Primary outcome (grain Zn) is **in Figure 3b only**, not in tables
- Tables 2, 3, 4 contain agronomic data only (listed under `tables_without_target_data`)
- `is_fig_only: true`
- `estimated_difficulty: "HARD"`
- `extraction_method: "vision"`
- Warning: "WARNING: FIG-ONLY — Data is in figures, not tables"

The recon guidance was technically correct: "Extract Zn treatments vs control... Do NOT extract Fe-only treatments."

### 3b. Consensus result

```
claude_obs: 0   (for grain Zn in consensus — all claude grain Zn observations were DISAGREEMENTS)
kimi_obs:   0   (same)
gemini_obs: 0   (same)
matched_obs: 68 (these are from Tables 2–4, agronomic variables only)
```

The 68 consensus observations are all from tables (tillers, plant height, spike length, spikelets, grains per spike, 1000-grain weight, biological yield, grain yield, harvest index) where both models agreed. None of these are the target variable for Hui 2023.

### 3c. Grain Zn attempts — all in DISAGREEMENTS (not accepted into consensus)

Claude extracted **8 grain Zn observations** from Figure 3b, covering:
- Foliar 0.5% ZnSO4 × 2 years (treatment_mean: 60.2, 61.33)
- Foliar 0.5% ZnSO4 + 1% FeSO4 × 2 years (treatment_mean: 58.5, 60.1)
- Soil 10 kg Zn ha-1 × 2 years (treatment_mean: 52.3, 54.2)
- Soil 10 kg Zn ha-1 + 12 kg Fe ha-1 × 2 years (treatment_mean: 48.1, 50.3)

Kimi extracted **8 grain Zn observations** from Figure 3b covering the same treatment × year combinations but with different bar-chart readings:
- Foliar 0.5% ZnSO4 × 2 years (treatment_mean: 60.0, 61.33)
- Foliar 0.5% ZnSO4 + 1% FeSO4 × 2 years (treatment_mean: 50.0, 52.0)
- Soil 10 kg Zn ha-1 × 2 years (treatment_mean: 50.0, 52.0)
- Soil 10 kg Zn ha-1 + 12 kg Fe ha-1 × 2 years (treatment_mean: 55.0, 57.0)

**Why these were classified as disagreements and rejected:**

The two models diverged enough on the bar-chart readings that the consensus algorithm flagged them as `claude_only` or `kimi_only` disagreements rather than matches. Example discrepancies for the same observation:
- Foliar Zn + Fe, 2016–17: Claude = 58.5, Kimi = 50.0 (16% gap)
- Soil Zn + Fe, 2016–17: Claude = 48.1, Kimi = 55.0 (14% gap)

Additionally, all control mean readings differed between models:
- Claude used 41.08 (from text) for year 1 and ~43.5 (estimated from chart) for year 2
- Kimi used 41.08 (from text) for year 1 and ~45.0 (estimated from chart) for year 2

With no agreed consensus observations for grain Zn, the paper contributes **zero usable observations** to the validation set for Hui 2023.

### 3d. Gemini

Gemini extracted **0 grain Zn observations**. The consensus JSON shows `gemini_obs: 0` with no gemini entries in the disagreements block for grain Zn. Gemini appears to have abstained entirely from figure data extraction, possibly declining to read bar charts or failing silently on the figure-only outcome.

### 3e. Variance

No model extracted variance for grain Zn. All entries show `treatment_variance: null, control_variance: null, variance_type: null`. This is correct — numeric LSD/SE values are not reported in Figure 3, only letter codes.

---

## 4. GT Data

The ground truth in MOESM5 contains **4 rows** for study_id 69 across two sheets:

### Sheet: Data 3 — Foliar application (Obs IDs 646, 647)

| Field | Obs 646 | Obs 647 |
|-------|---------|---------|
| Zn rate (kg Zn ha-1) | 10 (as 0.5% ZnSO4 foliar) | 10 (as 0.5% ZnSO4 foliar) |
| Spraying conc (g Zn L-1) | 0.2025 | 0.2025 |
| Spraying frequency (times) | 3 | 3 |
| Spraying timing | 6 | 6 |
| n (replicates) | 3 | 3 |
| Grain yield (kg ha-1) | 3,200 | 3,310 |
| Straw biomass (kg ha-1) | 7,700 | 7,490 |
| Grain Zn concentration (mg kg-1) | **40.7168** | **42.1505** |
| Zn biofortification index | (not stored) | (not stored) |
| Soil pH | 7.6 | 7.6 |
| Available Zn (mg kg-1) | 0.52 | 0.52 |
| OM (g kg-1) | 0.084 | 0.084 |
| Country | Pakistan | Pakistan |

### Sheet: Data 2 — Soil application (Obs IDs 650, 651)

| Field | Obs 650 | Obs 651 |
|-------|---------|---------|
| Zn rate (kg Zn ha-1) | 10 | 10 |
| n (replicates) | 3 | 3 |
| Grain yield (kg ha-1) | 3,200 | 3,310 |
| Straw biomass (kg ha-1) | 7,700 | 7,490 |
| Grain Zn concentration (mg kg-1) | **40.7168** | **42.1505** |
| Zn biofortification index | 1.06094 | 0.83154 |
| Soil pH | 7.5 | 7.5 |
| Available Zn (mg kg-1) | 0.62 | 0.62 |
| OM (g kg-1) | 0.077 | 0.077 |
| Country | Pakistan | Pakistan |

**Critical observation about GT values:** The grain Zn concentrations in the GT (40.72 and 42.15 mg kg-1) are notably close to the **control** values from the paper (41.08 mg kg-1 confirmed in text), not the Zn-treatment values (~52–61 mg kg-1 range visible in Figure 3b). This strongly suggests that Hui 2023 recorded the **grain Zn concentration in the Zn-treated plots as the outcome variable** but the values themselves represent the observed concentration in treated plots, not the change from control.

The two GT rows per sheet correspond to the **two crop years** (2016–17 grain yield 3200 kg ha-1 = year 1, 2017–18 grain yield 3310 kg ha-1 = year 2), matching Table 4 grain yield values exactly (Control: 3.20 and 3.31 t ha-1).

Wait — re-reading: grain yield 3,200 kg ha-1 = 3.20 t ha-1 (year 2016–17 control) and 3,310 = 3.31 t ha-1 (year 2017–18 control). This matches Table 4 **control row** values exactly. The grain Zn values (40.72, 42.15) are also near the control text value (41.08). It appears Hui 2023 extracted these as two separate site-year observations — one per cropping season — with the grain Zn from the treatment arm (foliar or soil Zn application) being reported. The foliar obs (sheet 3) and soil obs (sheet 2) have different pH/OM values matching the 0–15 cm soil data for years 2016–17 and 2017–18 in Table 1. The two rows per sheet represent the **two experimental years treated as separate site-year replicate observations**, not two different treatments.

---

## 5. Root Cause

### Primary cause: Figure-only data with bar chart reading disagreement

The grain Zn data is embedded in Figure 3b as a bar chart. The two models that attempted extraction (Claude and Kimi) both read the bars but arrived at divergent values for several treatments, causing all grain Zn entries to fall into the "disagreement" bin rather than the consensus bin. The consensus algorithm correctly refused to accept estimates where models disagreed — but this left zero validated grain Zn observations.

### Secondary cause: Gemini abstention

Gemini produced zero grain Zn observations. If Gemini had corroborated either Claude or Kimi on any observations, those might have formed a consensus. Gemini's abstention on figure-based data eliminated the possibility of a 2-vs-1 tiebreaker.

### Contributing factor: Control value ambiguity

Claude used 41.08 (text-confirmed) for year 2016–17 control but estimated ~43.5 from the chart for year 2017–18. Kimi used 41.08 for year 1 and ~45.0 for year 2. Since the paper states "no significant difference between growing years" for grain Zn (p=0.4714 for Year effect), both years should have similar control values (~41 mg kg-1). The inconsistency in how models read the control bar introduced additional disagreement.

### Structural mismatch with GT extraction logic

The GT records only **one treatment per application method per year** (foliar Zn or soil Zn), not the full factorial. Hui 2023 selected what appear to be the simplest Zn-only treatment (foliar 0.5% ZnSO4 or soil 10 kg Zn ha-1) matched with soil property data specific to each growing year's 0–15 cm horizon (Table 1). Our pipeline extracted the full 8 treatment × year combinations, which is more complete but mismatches the GT's simpler structure.

### No numeric variance available

This is a hard limitation of the paper. Grain mineral data in figures have no numeric LSD or SE reported — only letter notation (A, B, C) and overall p-values. Even a correct extraction would yield null variance, making these observations unusable for weighted meta-analysis without imputation.

---

## 6. Assessment

**Extractability rating: LOW — Figure-only with bar chart estimation required**

| Criterion | Status |
|-----------|--------|
| Target data present in PDF | Yes (Figure 3b) |
| Data in table form | No |
| Numeric variance available | No |
| Bar chart readable by AI | Partially (models disagree on ~15–20% of bars) |
| Consensus achieved | No — all grain Zn in disagreement bin |
| GT rows missed | 4 of 4 (100% miss rate) |

**Could this paper be rescued?**

With a manual bar chart reading pass: yes, but only partially. The two anchor values (control 41.08 and max treatment 61.33) are confirmed in text; all intermediate bars require visual estimation. Manual reading of Figure 3b would allow extraction of all 14 bars (7 treatments × 2 years). However, variance would remain null, limiting the observation's weight in any meta-analysis.

**Matching GT's extraction logic:** The GT extracted only 4 rows (2 treatment types × 2 years), whereas the paper has 4 Zn-containing treatments × 2 years = 8 possible treatment comparisons. To match Hui 2023's approach, only the pure single-nutrient treatments (foliar 0.5% ZnSO4 and soil 10 kg Zn ha-1) appear to have been included, excluding combined Zn+Fe treatments.

**Recommended action for pipeline improvement:**
1. When `is_fig_only: true`, lower the consensus agreement threshold for figure-based observations or introduce a Gemini mandatory attempt via vision API.
2. Implement a fallback: if both Claude and Kimi read a figure but disagree by <25%, average their readings and flag as `figure_estimate`.
3. For bar charts with labeled extreme values (min/max noted in text), use text anchors to calibrate bar readings.
4. Gemini should not abstain from figure data when `extraction_method: "vision"` is set in recon guidance.

**Impact on validation statistics:** This paper contributes 0 observations to the Hui 2023 validation match set. The 4 missed GT rows represent a valid extraction target that could in principle be recovered with vision-based figure reading and a relaxed consensus threshold.
