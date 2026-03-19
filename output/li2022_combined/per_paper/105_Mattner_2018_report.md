# Extraction Quality Report: 105_Mattner_2018

**Paper**: Mattner SW, Milinkovic M, Arioli T (2018). "Increased growth response of strawberry roots to a commercial extract from *Durvillaea potatorum* and *Ascophyllum nodosum*." *Journal of Applied Phycology*, 30: 2943–2951.

**Report generated for:** Li (2022) meta-analysis validation

**Summary result**: N=3 matched pairs, MAE=0.00%, direction agreement=100% (3/3). PERFECT extraction.

---

## 1. Paper Design

### Study overview

This is an open-field trial conducted at two commercial strawberry sites in Victoria, Australia (Toolangi and Coldstream), across two consecutive growing seasons (2014/2015 nursery trial; 2015/2016 fruit trial). The study investigates the growth and yield response of strawberry (*Fragaria* × *ananassa*) to monthly drench applications of a commercial seaweed extract (SE) — Seasol®, derived from *Durvillaea potatorum* and *Ascophyllum nodosum* — applied at 1:400 dilution (10 L ha⁻¹ per application). The control received equivalent volumes of water.

### Experimental design

The study comprises two linked experiments with different structures:

**Nursery trial (Table 1):**

| Parameter | Value |
|-----------|-------|
| Crop | Strawberry cvs. Fortuna and Albion (tested separately) |
| Design | Randomized complete block design (RCBD) |
| Replicates | 16 blocks |
| Treatments | Seaweed extract drench vs. water control |
| Outcome | Commercial runner yield (runners m⁻¹) and reject rate (%) |
| Season | 2014/2015, Toolangi |
| Variance | Fisher's LSD (p ≤ 0.05) |

**Fruit trial (Table 6):**

| Parameter | Value |
|-----------|-------|
| Crop | Strawberry cv. Albion |
| Design | Randomized complete block design (RCBD) |
| Replicates | 3 blocks |
| Structure | 2×2 factorial: nursery-sector SE (yes/no) × fruit-sector SE (yes/no) |
| Outcomes | Total fruit yield (g plant⁻¹), revenue (AUS$ plant⁻¹), commercial-grade berries (%), fruit size (g berry⁻¹) |
| Season | 2015/2016, Coldstream |
| Variance | Fisher's LSD (p ≤ 0.05) |

The factorial design in the fruit trial creates four treatment combinations: (1) neither sector treated, (2) nursery sector only, (3) fruit sector only, (4) both sectors treated. Li (2022) included only the fruit-sector-only drench arm (treatment 3 vs. treatment 1) as its ground truth observation for total fruit yield.

---

## 2. AI Consensus Extraction Results

The consensus pipeline (Claude + Kimi, each returning 12 observations; Gemini not used) produced 12 matched observations with zero tiebreaker invocations and zero model disagreements. All 12 observations are rated high confidence. No post-processing corrections (duplicate removal, null-mean removal, or T/C swaps) were applied.

**Observations extracted (12 total):**

| json_idx | Element | Cultivar / Sector | Treatment mean | Control mean | Effect (%) |
|----------|---------|-------------------|----------------|--------------|------------|
| 0 | Runner yield (runners m⁻¹) | Fortuna / nursery | 103.4 | 86.7 | +19.26 |
| 1 | Rejects (%) | Fortuna / nursery | 35.7 | 43.7 | −18.31 |
| 2 | Runner yield (runners m⁻¹) | Albion / nursery | 156.7 | 144.5 | +8.44 |
| 3 | Rejects (%) | Albion / nursery | 10.1 | 11.4 | −11.40 |
| 4 | Total fruit yield (g plant⁻¹) | Albion / fruit (SE only) | 547.6 | 502.8 | +8.91 |
| 5 | Total revenue (AUS$ plant⁻¹) | Albion / fruit (SE only) | 4.10 | 3.79 | +8.18 |
| 6 | Commercial-grade berries (%) | Albion / fruit (SE only) | 40.45 | 41.88 | −3.41 |
| 7 | Fruit size (g berry⁻¹) | Albion / fruit (SE only) | 10.07 | 10.30 | −2.23 |
| 8 | Total fruit yield (g plant⁻¹) | Albion / fruit (SE + nursery SE) | 544.1 | 502.8 | +8.21 |
| 9 | Total revenue (AUS$ plant⁻¹) | Albion / fruit (SE + nursery SE) | 4.09 | 3.79 | +7.92 |
| 10 | Commercial-grade berries (%) | Albion / fruit (SE + nursery SE) | 40.36 | 41.88 | −3.63 |
| 11 | Fruit size (g berry⁻¹) | Albion / fruit (SE + nursery SE) | 10.59 | 10.30 | +2.82 |

The consensus pipeline correctly identified the factorial structure of the fruit trial, correctly attributed the nursery trial data to two separate cultivars (Fortuna and Albion), and correctly reported LSD as the variance type with numeric LSD values for every observation. Moderator metadata (cultivar, site, season, sector, number of applications) was fully and accurately captured.

---

## 3. Ground Truth Comparison

Li (2022) included three observations from this paper (GT pairs 441–443). All three were matched with perfect effect-size agreement (MAE = 0.00 percentage points across all three pairs, 100% directional accuracy).

| GT pair | Element | GT ctrl | GT treat | GT effect (%) | Ext ctrl | Ext treat | Ext effect (%) | Confidence |
|---------|---------|---------|---------|---------------|----------|----------|----------------|------------|
| 441 | Runner yield (runners m⁻¹) | 0.867* | 1.034* | +19.26 | 86.7 | 103.4 | +19.26 | High |
| 442 | Runner yield (runners m⁻¹) | 1.445* | 1.567* | +8.44 | 144.5 | 156.7 | +8.44 | High |
| 443 | Total fruit yield (g plant⁻¹) | 0.5028† | 0.5476† | +8.91 | 502.8 | 547.6 | +8.91 | High |

*GT stores runner yield in units 100× smaller than the paper's reported runners m⁻¹ (e.g., GT 0.867 = 86.7 runners m⁻¹). This is a unit-scaling convention in the Li 2022 database and does not affect computed effect sizes.

†GT stores total fruit yield in units 1000× smaller than the paper's g plant⁻¹ (e.g., GT 0.5028 = 502.8 g plant⁻¹). Again, effect percentages are identical.

Pairs 441 and 442 correspond to the Fortuna and Albion nursery runner yield observations respectively (json_idx 0 and 2), both from Table 1 of the paper. Pair 443 corresponds to the fruit-sector-only drench arm of the 2×2 factorial (json_idx 4), from Table 6. The AI correctly distinguished this arm from the dual-sector (nursery+fruit) treatment (json_idx 8), which has no corresponding GT row because Li (2022) did not include the combined treatment.

---

## 4. Root Cause Analysis

### Why extraction was perfect

Three structural features of this paper combined to make it an ideal extraction target:

**1. Clean tabular presentation with unambiguous treatment labels.** Both Table 1 (nursery runner yield) and Table 6 (fruit sector yields) present treatment and control means in adjacent columns labeled "Untreated" and "SE" respectively. There is no ambiguity in which column is which. The paper's own notation directly maps onto the meta-analysis comparison of interest.

**2. Explicit, paper-wide variance declaration.** The Methods section states Fisher's LSD was used throughout (p ≤ 0.05), and LSD values appear as the final row in each table. The AI correctly identified LSD as the variance type and extracted numeric LSD values for every observation. No inferential reasoning about ± notation was required.

**3. Straightforward factorial structure.** Although the fruit trial uses a 2×2 factorial, the individual treatment combinations are listed as discrete rows in Table 6. The AI correctly identified the "Untreated nursery / SE fruit" row as the fruit-sector-only arm relevant to Li (2022)'s comparison, and extracted its means without confusing it with the dual-sector "SE nursery / SE fruit" arm.

### Why 9 extracted observations are unmatched (correct exclusions)

Of the 12 consensus observations, 9 have no GT counterpart. These are correctly excluded from the Li (2022) ground truth for the following reasons:

- **Rejects (%) — json_idx 1, 3**: Quality/rejection-rate metrics; Li (2022) scoped to yield and production outcomes only.
- **Commercial-grade berries (%) — json_idx 6, 10**: Fruit quality/grading metric, outside Li (2022) scope.
- **Fruit size (g berry⁻¹) — json_idx 7, 11**: Per-berry size metric, outside Li (2022) scope.
- **Total revenue (AUS$ plant⁻¹) — json_idx 5, 9**: Economic metric, outside Li (2022) scope.
- **Total fruit yield, dual-sector arm — json_idx 8**: Li (2022) included only the fruit-sector-only arm (SE applied in fruit sector, not nursery). The dual-sector arm (SE in both nursery and fruit sectors) has no corresponding GT row.

All nine exclusions reflect deliberate scope decisions by the Li (2022) meta-analysis, not extraction errors. The AI correctly extracted these observations — they are genuinely present in the paper — but Li (2022) simply did not include them in its dataset.

### Verification flag notes

The automated verification system raised flags for GRIM test failures and variance type ambiguity across most observations. These are expected artefacts rather than genuine errors:

- **GRIM failures**: GRIM tests are designed for integer-valued data (e.g., counts of discrete items). Runner yield (runners m⁻¹) and gram-based yields are continuous decimal measurements. GRIM is not applicable to continuous data and should be disregarded here.
- **Variance type flag (LSD vs SD)**: The CV-heuristic correctly found that treating the LSD values as SD gives plausible CVs (7–18%), which triggered a flag suggesting the reported type might be SD rather than LSD. However, the paper is explicit that Fisher's LSD was used. The flag is a known limitation of the CV-heuristic when LSD values happen to fall in the SD-plausible range.
- **n=16 "too large" flag**: The system flagged n=16 for nursery observations as unusually large. This is incorrect — 16 randomized complete blocks is an entirely standard and well-powered design for an agricultural field trial.

---

## 5. Overall Assessment

| Dimension | Assessment |
|-----------|------------|
| Effect size accuracy (matched pairs) | PERFECT — 0.00 pp MAE across all 3 GT pairs |
| Directional accuracy | PERFECT — 3/3 correct |
| Coverage of GT pairs | COMPLETE — 3/3 GT pairs captured |
| Factorial structure handling | EXCELLENT — fruit-sector-only arm correctly isolated |
| Cultivar-level disaggregation | EXCELLENT — Fortuna and Albion extracted separately |
| Variance type identification | CORRECT — LSD identified from Methods section |
| Numeric LSD values | EXTRACTED — present for all 12 observations |
| Moderator metadata | COMPLETE — cultivar, site, season, sector, application count |
| Unmatched JSON obs (9) | CORRECT EXCLUSIONS — outside Li 2022 scope, not errors |

**Overall verdict: PERFECT (Grade A)**

This is an exemplary extraction. The AI pipeline produced zero effect-size error across all three ground truth pairs. Both models (Claude and Kimi) reached identical results for all 12 observations, with no tiebreaker needed and no disagreements. The complex factorial design in the fruit trial was handled correctly, with the fruit-sector-only arm correctly distinguished from the dual-sector arm. The nine unmatched observations represent genuine, accurately extracted data that simply falls outside the Li (2022) meta-analysis scope. This paper illustrates that when a study presents clean tabular data with explicit variance declarations and clearly labeled treatment arms, the consensus extraction pipeline operates at ceiling performance.
