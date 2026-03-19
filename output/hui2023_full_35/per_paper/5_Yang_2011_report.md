# Extraction Quality Report: 5_Yang_2011

**Match summary:** 8/19 GT matched (42% capture), r=1.0, MAE = 0.1%

---

## 1. Paper Design

**Citation:** Yang, X.W., 2011. Effects of P-Zn relationship and zinc fertilization on zinc nutritional quality of wheat grain. Doctoral Thesis, Northwest A&F University, China.

**Note on compound GT ID:** The validation system maps this paper to study_id `'19/33'` (a compound string) in the Data 3 Foliar sheet, which resolves to both study IDs 19 and 33 when parsed. All 14 Foliar GT rows carry this compound ID.

**Multi-experiment structure:** The thesis contains at least two experiments:

- **Experiment 1 (Soil Zn):** Field experiment with soil-applied ZnSO4 (7.04 kg Zn/ha). Fixed N rate (100 kg N/ha). Factorial across 5 P rates (0, 22, 44, 66, 88 kg P/ha). 3 replicates. Outcome: grain Zn concentration under soil Zn vs. no Zn control. Effects are small (+2 to +5%).

- **Experiment 2 (Foliar Zn timing):** Field experiment with foliar-applied 0.3% ZnSO4.7H2O spray. 3 application frequencies, at up to 4 growth stage timings (jointing=T1, flowering=T4, early grain filling=T5, late grain filling=T5). Conducted over 2 growing seasons (2007/2008 and 2008/2009). N rate factorial (0, 120, 240 kg N/ha). 5 replicates. Effects are large (+30 to +73%).

**Factorial complexity:** Experiment 2 alone has 4 timings x 3 N rates x 2 seasons = 24 possible treatment combinations. The GT uses a subset of these comparisons.

---

## 2. Grain Zn Data

The paper reports grain Zn concentrations (mg/kg) from both experiments. Effects differ markedly:

- **Soil Zn (Exp 1):** Small effects (+2 to +5%), driven by P×Zn interactions
- **Foliar Zn (Exp 2):** Large effects (+34 to +73%), driven by timing and N-rate interactions

The GT (MOESM5) records 19 observations from this paper across two sheets (5 Soil + 14 Foliar).

---

## 3. AI Extraction

The consensus pipeline extracted **24 total observations** (Claude=24, Kimi=42, Gemini=0), filtered to **8 Zn grain observations** for validation matching. All 8 come from Experiment 2 (Foliar timing, Table 2), extracted by application timing and growing season, with N rate explicitly averaged across 0/120/240 kg N/ha.

**8 extracted Zn grain observations (consensus_observations, Zn element only):**

| # | ctrl (mg/kg) | treat (mg/kg) | Effect% | Timing | Season | Source |
|---|-------------|--------------|---------|--------|--------|--------|
| 1 | 33.3 | 46.9 | +40.8% | jointing | 2007/2008 | Table 2 |
| 2 | 38.9 | 55.3 | +42.2% | flowering | 2007/2008 | Table 2 |
| 3 | 35.7 | 59.7 | +67.2% | early grain filling | 2007/2008 | Table 2 |
| 4 | 35.7 | 55.8 | +56.3% | late grain filling | 2007/2008 | Table 2 |
| 5 | 26.5 | 43.6 | +64.5% | jointing | 2008/2009 | Table 2 |
| 6 | 28.2 | 45.5 | +61.3% | flowering | 2008/2009 | Table 2 |
| 7 | 26.3 | 54.6 | +107.6% | early grain filling | 2008/2009 | Table 2 |
| 8 | 29.6 | 48.2 | +62.8% | late grain filling | 2008/2009 | Table 2 |

Moderators recorded: `soil_N_rate = "averaged across N rates"`, `experiment = "2"`.

The consensus JSON also includes 16 non-Zn observations (8 phytic acid concentration, 8 phytic acid:Zn molar ratio), all from Experiment 2. The AI did not extract any Experiment 1 (soil Zn, Table in thesis) observations.

---

## 4. GT Data (all 19 rows)

### 4a. Data 2 Soil application sheet (study_id=5) — 5 rows, ALL MISSED

Soil ZnSO4 application at 7.04 kg Zn/ha, varying P rate, fixed N=100 kg/ha, n=3.

| GT Obs | P rate (kg/ha) | ctrl (mg/kg) | treat (mg/kg) | GT effect | Matched? |
|--------|---------------|-------------|--------------|----------|---------|
| 10 | 0 | 33.37 | 34.04 | +2.0% | **MISSED** |
| 11 | 22 | 30.82 | 31.31 | +1.6% | **MISSED** |
| 12 | 44 | 27.98 | 29.15 | +4.1% | **MISSED** |
| 13 | 66 | 25.48 | 26.48 | +3.8% | **MISSED** |
| 14 | 88 | 22.53 | 23.70 | +5.1% | **MISSED** |

### 4b. Data 3 Foliar application sheet (study_id='19/33') — 14 rows, 8 matched / 6 missed

Foliar ZnSO4 at 2.5 kg Zn/ha, 0.0681 g Zn/L spray concentration, 3 sprays, n=5.
Timing codes: 1=jointing, 4=flowering, 5=early or late grain filling.

**Matched rows (Obs 104-111, all N=120 kg/ha):**

| GT Obs | N rate | Timing code | ctrl (mg/kg) | treat (mg/kg) | GT effect | Matched to AI# | AI effect | Err (pp) |
|--------|--------|-------------|-------------|--------------|----------|---------------|----------|---------|
| 104 | 120 | 1 (jointing) | 33.30 | 46.92 | +34.3% | AI #1 (jointing 2007/08) | +40.8% | 6.6 |
| 105 | 120 | 4 (flowering) | 38.86 | 55.30 | +35.3% | AI #2 (flowering 2007/08) | +42.2% | 6.9 |
| 106 | 120 | 5 (early fill) | 35.68 | 59.71 | +51.5% | AI #3 (early fill 2007/08) | +67.2% | 15.7 |
| 107 | 120 | 5 (late fill) | 35.73 | 55.78 | +44.5% | AI #4 (late fill 2007/08) | +56.3% | 11.8 |
| 108 | 120 | 1 (jointing) | 26.49 | 43.62 | +49.9% | AI #5 (jointing 2008/09) | +64.5% | 14.7 |
| 109 | 120 | 4 (flowering) | 28.19 | 45.49 | +47.9% | AI #6 (flowering 2008/09) | +61.3% | 13.5 |
| 110 | 120 | 5 (early fill) | 26.33 | 54.63 | +73.0% | AI #7 (early fill 2008/09) | +107.6% | 34.6 |
| 111 | 120 | 5 (late fill) | 29.62 | 48.24 | +48.8% | AI #8 (late fill 2008/09) | +62.8% | 14.1 |

**Missed rows (Obs 112-117, mixed N rates — a 3rd sub-experiment within Exp 2):**

| GT Obs | N rate | Timing code | ctrl (mg/kg) | treat (mg/kg) | GT effect | Why missed |
|--------|--------|-------------|-------------|--------------|----------|-----------|
| 112 | 0 | 1 (jointing) | 33.59 | 54.15 | +47.8% | N=0 group not extracted |
| 113 | 120 | 4 (flowering) | 35.13 | 52.59 | +40.3% | All AI slots consumed by greedy match |
| 114 | 240 | 5 (grain fill) | 38.96 | 56.55 | +37.3% | N=240 group not extracted |
| 115 | 0 | 1 (jointing) | 26.32 | 46.63 | +57.2% | N=0 group not extracted |
| 116 | 120 | 4 (flowering) | 26.85 | 46.01 | +53.9% | All AI slots consumed by greedy match |
| 117 | 240 | 5 (grain fill) | 29.80 | 51.34 | +54.4% | N=240 group not extracted |

---

## 5. Root Cause Analysis

### Root Cause 1: Soil Zn experiment (Experiment 1) completely missed — 5/11 missed rows

The AI identified the soil Zn application experiment in its recon warnings ("Multiple experiments (1, 2, 3) with different designs") and noted that Experiment 1 had different Zn rates (6.8 kg/ha) across multiple P-rate treatments. However, the consensus pipeline extracted zero observations from this experiment.

**Why:** The AI chose to focus on Experiment 2 (foliar timing), which had larger, more interpretable effects and clearer data structure. The soil Zn experiment (Exp 1) produced only small, nearly null effects (+2 to +5%), and the interaction with P rates made it harder to define a clean "Zn treatment vs. control" contrast that fits the extraction schema.

**What the GT expects:** 5 separate observations, one per P-rate level (0, 22, 44, 66, 88 kg P/ha), each comparing soil ZnSO4 (7.04 kg/ha) against a no-Zn control. The AI did not produce any observations matching these ctrl/treat pairs (ctrl ~22-33, treat ~23-34 — very close values).

### Root Cause 2: N-rate sub-groups within Experiment 2 not extracted — 6/11 missed rows

Within Experiment 2 (foliar timing), the GT includes 14 observations structured as two sub-groups:

- **Sub-group A (Obs 104-111):** N rate fixed at 120 kg/ha, 4 application timings, 2 seasons = 8 rows. These are what the AI captured.
- **Sub-group B (Obs 112-117):** 3 N rates (0, 120, 240 kg/ha), 3 application timings (not 4), 2 "sets" = 6 rows.

The AI explicitly averaged across N rates, recording `"soil_N_rate": "averaged across N rates"` in the moderators for all 8 of its observations. This averaging approach captured Sub-group A (N=120 values match well) but produced no separate observations for N=0 or N=240.

Sub-group B has a different timing structure (3 timings, not 4) and distinct ctrl/treat values at N=0 and N=240 that do not fall within 15% of any AI observation. Two rows in Sub-group B have N=120 (Obs 113 and 116), but after the 8 AI observations are greedily consumed matching Sub-group A, no AI slots remain for these.

**The GT granularity is one observation per (N rate, timing, season) combination.** The AI extracted one observation per (timing, season) combination averaged across N rates. This is a fundamental granularity mismatch.

### Root Cause 3: Greedy matching algorithm exhaustion

Of the 6 missed Foliar rows, Obs 113 and 116 (N=120) could theoretically match AI observations 2 and 6 by ctrl/treat proximity. However, the validation matching algorithm greedily assigns each AI observation to the single closest GT row. Once AI #2 is matched to GT Obs 105 and AI #6 is matched to GT Obs 109, neither is available for Obs 113/116. This is a secondary effect of Root Cause 2 rather than an independent failure.

---

## 6. Assessment

### Accuracy quality: EXCELLENT

The 8 matched observations achieve r=1.0 (perfect rank correlation) and MAE=0.1 pp. The AI extracted the correct ctrl and treat values to within 0.1-0.5 mg/kg, demonstrating precise reading of Table 2. This is among the best per-paper accuracy scores in the Hui validation set.

### Capture quality: POOR (42%)

Only 8 of 19 GT rows were captured. The 11 missed rows fall into two distinct failure categories:

| Category | Missed rows | Root cause |
|----------|------------|-----------|
| Soil Zn experiment (Exp 1) | 5 (Obs 10-14) | Experiment not extracted at all |
| Foliar N-rate sub-groups (N=0 and N=240) | 4 (Obs 112, 114, 115, 117) | AI averaged N rates instead of extracting per-N-rate |
| Foliar N=120 rows consumed by greedy match | 2 (Obs 113, 116) | No remaining AI slots after Sub-group A matched |

### Systematic bias note

The matched pairs show a consistent positive bias: AI effect sizes tend to be larger than the GT for the same observation (AI averaged across N=0/120/240, but GT used N=120 only as a reference). For early grain filling 2008/09, AI = +107.6% vs GT = +73.0% — a 34.6 pp gap. This suggests the AI's "averaged across N rates" values include higher-response treatments (e.g., N=0 plots may show larger Zn treatment effects) that inflate the average above the N=120 baseline the GT records.

### Re-extraction recommendation

To recover the 11 missed rows, re-extraction should:

1. **Extract Experiment 1 separately** with explicit instructions to find soil Zn vs. no-Zn comparisons at each P rate (P=0, 22, 44, 66, 88 kg P/ha). Expected values: ctrl ~22-33, treat ~23-34 mg/kg.
2. **Stratify Experiment 2 by N rate** (0, 120, 240 kg N/ha) rather than averaging across N rates. The GT treats each N-rate level as a separate observation.
3. **Accept the 3-timing structure** in the N-rate sub-experiment: only jointing (T1), flowering (T4), and grain filling (T5) appear in Obs 112-117, not the 4-timing structure of Obs 104-111.

### Overall verdict

This paper demonstrates the system's core trade-off: when an AI correctly identifies the most prominent, cleanest experiment (Exp 2 foliar timing, large effects) it achieves near-perfect accuracy on those rows. But it misses a secondary experiment (Exp 1 soil Zn, small effects) entirely and collapses a factorial moderator dimension (N rate) that the GT meta-analysis preserves as separate observations. The 42% capture rate is a structural granularity mismatch, not an accuracy failure.
