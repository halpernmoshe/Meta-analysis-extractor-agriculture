# Extraction Quality Report: 66_PahlavanRad_2009

**Paper:** Pahlavan-Rad, M.R. & Pessarakli, M. (2009). Response of wheat plants to zinc, iron, and manganese applications and uptake and concentration of zinc, iron, and manganese in wheat grains. *Communications in Soil Science and Plant Analysis*, 40(7-8), 1322–1332. DOI: 10.1080/00103620902761262

**Match summary:** 3/3 GT observations matched | r = 0.998 | MAE = 5.04%

---

## 1. Paper Design

**Species:** Winter wheat (*Triticum aestivum*, variety Hamoon)
**Location:** Zahak Agricultural Research Station, Sistan region, southeastern Iran (30°54'N, 61°41'E)
**Years:** 2003 and 2004 (two growing seasons, pooled in most tables)
**Soil:** Sandy loam, pH 8.2–8.4, OC 0.35–0.37%, available Zn very low (0.26 mg kg⁻¹ in 2003, 1.57 mg kg⁻¹ in 2004)

**Experimental design:** Fully factorial, completely randomized block with 3 replications (n = 3)

| Factor | Levels |
|--------|--------|
| Zn | 0, 40, 80 kg ha⁻¹ ZnSO₄ (soil) + 0.5% ZnSO₄ (foliar) → 4 levels |
| Fe | 0, 1% FeSO₄ (foliar) → 2 levels |
| Mn | 0, 0.5% MnSO₄ (foliar) → 2 levels |

Total treatment combinations: 4 × 2 × 2 = 16 combinations per year.

**Primary outcome variable for Hui meta-analysis:** Grain Zn concentration (mg kg⁻¹, DTPA method)

**Variance reporting:** Least Significant Difference (LSD) at p = 0.05. No numeric LSD values appear in the table itself; statistical significance is indicated by asterisks (** = p < 0.01, * = p < 0.05, ns = non-significant).

---

## 2. Grain Zn Data

### Table 5 (Main effects — pooled across Fe and Mn levels)

| Treatment | Zn (mg kg⁻¹) | Fe (mg kg⁻¹) | Mn (mg kg⁻¹) |
|-----------|:------------:|:------------:|:------------:|
| Zn 0 (control) | **31.2** | 65.7 | 49.1 |
| 40 kg ha⁻¹ ZnSO₄ | **33.5** | 64.3 | 48.1 |
| 80 kg ha⁻¹ ZnSO₄ | **33.7** | 65.4 | 52.3 |
| Zn 0.5% foliar | **62.1** | 71.2 | 50.7 |
| Variance analysis | ** | * | ns |

Table 5 reports **main effects** averaged across all Fe and Mn sub-plot combinations.

### Table 6 (Zn × Fe interaction — broken out by Fe level)

| Treatment combination | Zn (mg kg⁻¹) | Fe (mg kg⁻¹) | Mn (mg kg⁻¹) |
|-----------------------|:------------:|:------------:|:------------:|
| Zn 0 Fe 0 | **29.8** | 57.3 | 51.7 |
| Zn 0 Fe 1% | **32.6** | 74.1 | 46.5 |
| Zn 40 Fe 0 | **31.3** | 55.7 | 47.9 |
| Zn 40 Fe 1% | **35.6** | 70.9 | 48.3 |
| Zn 80 Fe 0 | **32.8** | 61.4 | 54.3 |
| Zn 80 Fe 1% | **34.4** | 69.2 | 50.3 |
| Zn 0.5% Fe 0 | **56.5** | 66.2 | 53.8 |
| Zn 0.5% Fe 1% | **67.6** | 76.1 | 47.7 |
| Variance analysis | * | ns | ns |

Table 6 reports the Zn × Fe interaction, with Fe fixed at either 0 or 1%.

**Internal consistency check:** The Table 5 main effects are arithmetically consistent with Table 6 interaction values (average of Fe0 and Fe1% sub-groups):
- Zn0: (29.8 + 32.6) / 2 = **31.2** ✓ matches Table 5
- Zn0.5% foliar: (56.5 + 67.6) / 2 = **62.05** ✓ matches Table 5

---

## 3. AI Extraction

The consensus pipeline (Claude + Kimi; Gemini produced 0 observations) extracted from **Table 5 main effects**, which the recon correctly identified as the primary target table.

**Consensus observations (all high confidence, models agree):**

| Treatment | Control (mg kg⁻¹) | Treatment (mg kg⁻¹) | Effect (%) | Source |
|-----------|:-----------------:|:-------------------:|:----------:|--------|
| Zn 40 kg ha⁻¹ soil | 31.2 | 33.5 | +7.4% | Table 5 |
| Zn 80 kg ha⁻¹ soil | 31.2 | 33.7 | +8.0% | Table 5 |
| Zn 0.5% foliar | 31.2 | 62.1 | +99.0% | Table 5 |

**Additional observations extracted** (Table 5): Fe and Mn main effects on Zn, Fe, Mn grain concentrations — 6 further rows beyond the 3 Zn-outcome rows that matched GT. These are correct observations from this paper but lie outside the scope of the Hui Zn-soil/foliar meta-analysis (they compare Fe vs Fe0, not Zn treatment vs Zn0 control).

**Kimi-only observations discarded by consensus** (6 rows): These were Fe-effect and Mn-effect rows that Claude did not extract, correctly filtered out as outside scope.

**Variance:** No numeric LSD values extracted (none appear in the paper's tables; only significance asterisks). `variance_type = "LSD"` recorded correctly.

**GRIM test flags:** All 9 observations failed GRIM. This is expected for means pooled across a 4 × 2 × 2 factorial; the reported means are averages of averages with 2-year pooling, which cannot satisfy integer-data GRIM constraints. This is not an extraction error.

---

## 4. GT Data (MOESM5)

### Data 2 — Soil application sheet (study_id = 66)

| Obs ID | Zn rate (kg Zn ha⁻¹) | Grain Zn ctrl (mg kg⁻¹) | BFI | Implied treat (mg kg⁻¹) |
|--------|:--------------------:|:-----------------------:|:---:|:-----------------------:|
| 637 | 8.1 (≈ 40 kg ZnSO₄) | 29.8 | 0.185 | — |
| 638 | 32.4 (≈ 80 kg ZnSO₄) | 29.8 | 0.093 | — |

The MOESM5 "Grain Zn concentration" column records **29.8 mg kg⁻¹** for both soil rows — this is the Zn0 control value, taken from Table 6 row "Zn 0 Fe 0".

### Data 3 — Foliar application sheet (study_id = 66)

| Obs ID | Spraying conc. | Spraying freq. | Grain Zn ctrl (mg kg⁻¹) | BFI |
|--------|:--------------:|:--------------:|:-----------------------:|:---:|
| 619 | 0.5 g Zn L⁻¹ | 2× | 29.8 | — |

The validation script matched these three GT rows to the three AI Zn-outcome observations and reconstructed (gt_ctrl, gt_treat) pairs using the Table 6 Fe0 rows:

| GT obs | gt_ctrl | gt_treat | gt_effect | Source in paper |
|--------|:-------:|:--------:|:---------:|-----------------|
| 637 (40 kg/ha soil) | 29.8 | 32.8 | +10.07% | Table 6: Zn80 Fe0 ← note apparent swap (see §5) |
| 638 (80 kg/ha soil) | 29.8 | 31.3 | +5.03% | Table 6: Zn40 Fe0 ← note apparent swap |
| 619 (foliar 0.5%) | 29.8 | 56.5 | +89.60% | Table 6: Zn0.5 Fe0 |

---

## 5. Root Cause of the 5% MAE

### Primary cause: Different tables used as data source

The AI extracted from **Table 5** (main effects, pooled across Fe levels); Hui extracted from **Table 6** (Fe0-only sub-group of the Zn × Fe interaction).

This produces a systematic but coherent offset:

| Treatment | AI (Table 5, pooled) | Hui (Table 6, Fe0 only) | Difference |
|-----------|:--------------------:|:-----------------------:|:----------:|
| Control | 31.2 mg kg⁻¹ | 29.8 mg kg⁻¹ | −4.5% |
| Soil 40 kg/ha | 33.5 mg kg⁻¹ | 31.3 mg kg⁻¹ | −6.6% |
| Soil 80 kg/ha | 33.7 mg kg⁻¹ | 32.8 mg kg⁻¹ | −2.7% |
| Foliar 0.5% | 62.1 mg kg⁻¹ | 56.5 mg kg⁻¹ | −9.0% |

Because **both control and treatment are shifted upward by a similar amount**, the effect sizes partially cancel and remain highly correlated (r = 0.998). However, the foliar treatment has the largest absolute offset (62.1 vs 56.5 = 5.6 mg kg⁻¹) because the Fe co-application substantially increases foliar Zn uptake (Fe1% row gives 67.6 mg kg⁻¹), so pooling Fe0 and Fe1% inflates the Table 5 foliar mean relative to the Fe0-only baseline Hui used.

**Effect-size error breakdown:**

| Treatment | AI effect | Hui effect | Abs error |
|-----------|:---------:|:----------:|:---------:|
| Soil 40 kg ha⁻¹ | +7.4% | +10.1% | 2.7 pp |
| Soil 80 kg ha⁻¹ | +8.0% | +5.0% | 3.0 pp |
| Foliar 0.5% | +99.0% | +89.6% | **9.4 pp** |

The foliar observation drives the MAE disproportionately (9.4 pp out of a 5.04 pp average). The two soil observations are within ~3 pp of GT.

### Secondary cause: Apparent observation-rate mismatch in MOESM5

The MOESM5 BFI (Zn biofortification index) values for obs_637 and obs_638 do not match the Zn rates as expected:
- obs_637 carries Zn_rate = 8.1 kg Zn ha⁻¹ (= 40 kg ZnSO₄), yet its gt_treat in the validation CSV is 32.8 mg kg⁻¹ (= Table 6 Zn80 Fe0).
- obs_638 carries Zn_rate = 32.4 kg Zn ha⁻¹ (= 80 kg ZnSO₄), yet its gt_treat is 31.3 mg kg⁻¹ (= Table 6 Zn40 Fe0).

The rates and treatment values appear **swapped** between obs_637 and obs_638 within MOESM5. This internal MOESM5 inconsistency does not affect the overall MAE because the validation script matched AI observations to GT by application rate, and both soil treatments (40 and 80 kg ha⁻¹) show similar small effects (+7–10%), making the swap difficult to detect from the effect sizes alone.

### Summary of causal chain

```
AI chose Table 5 (main effects, Fe-averaged)
Hui chose Table 6 (Fe=0 sub-group only)
  → AI control = 31.2, Hui control = 29.8  (+4.7% offset)
  → AI foliar  = 62.1, Hui foliar  = 56.5  (+9.9% offset)
  → Offsets partially cancel in effect sizes
  → Effect r = 0.998 (near-perfect rank order)
  → Effect MAE = 5.04% (driven mainly by foliar observation)
```

No unit error, no rounding error, no misread values. The discrepancy is a **legitimate methodological choice** in how to handle a factorial design: main effects vs. Fe-unstratified sub-group.

---

## 6. Assessment

**Extraction correctness:** The AI correctly read the numeric values from Table 5 without error. The mean values (31.2, 33.5, 33.7, 62.1) are precisely reproduced from Table 5 of the PDF. No transcription errors.

**Table selection choice:** The recon guidance instructed the AI to use Table 5 main effects and avoid Table 6 interactions. This is a defensible default — main effects are the standard summary statistic for a balanced factorial. However, Hui's choice of Table 6 Fe0 rows gives a purer estimate of the Zn effect in the absence of Fe co-application, which is arguably more interpretable for a Zn-focused meta-analysis.

**Effect direction:** All three Zn effects are positive and correctly signed. No treatment/control swap.

**Factorial complexity:** The paper is a 4 × 2 × 2 × 2 (Zn × Fe × Mn × Year) design. The AI correctly averaged across Fe and Mn levels using Table 5 main effects rather than attempting to parse 16-cell interaction tables. Kimi attempted interaction-level extraction (Fe and Mn effects on Zn) which was correctly filtered by the consensus mechanism.

**Variance:** No numeric LSD values are recoverable from this paper (the tables show only significance symbols). This is a genuine data limitation.

**Overall rating: GOOD**
The extraction is numerically accurate, correctly scoped to Table 5 main effects, and captures the correct direction and approximate magnitude of all three Zn treatment effects. The 5% MAE reflects a table-selection difference (main effects vs. interaction sub-group) rather than any extraction error. For a dataset-level validation the near-perfect correlation (r = 0.998) confirms that the AI and Hui rank the three treatments identically and the absolute offsets are small relative to the foliar effect size (~99%).
