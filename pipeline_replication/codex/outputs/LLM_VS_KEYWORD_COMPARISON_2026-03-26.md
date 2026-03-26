# LLM vs Keyword Adjudicator Comparison — 2026-03-26

Full LLM semantic adjudication of all 6 V1 pipeline topics.
All adjudication performed by Claude Sonnet 4.6 as semantic adjudicator.

## Summary Table

| Topic | Total | KW kept | LLM kept | KW effect | LLM effect | Benchmark | LLM dir. correct? | Key improvement |
|-------|-------|---------|----------|-----------|------------|-----------|------------------|-----------------|
| biochar_crop_yield | 629 | 332 | 339 | +7.3% | +13.8% | +16.0% | YES | Excludes yield components (1000-grain wt, plant height), roo |
| intercropping_yield | 230 | 194 | 158 | -3.1% | +11.2% | +22.0% | YES | Excludes yield components (ear length, grains/plant), flags  |
| legume_rotation | 542 | 363 | 280 | +17.8% | +23.3% | +20.0% | YES | Excludes legume-as-main-crop, intercropped legume, yield com |
| mycorrhiza_yield | 588 | 256 | 309 | +31.4% | +26.0% | +23.0% | YES | Excludes colonization %, plant height, P uptake; flags per-p |
| notill_tillage | 881 | 418 | 380 | +2.7% | +6.1% | -5.7% | NO | Excludes non-till reductions, flags confounded interventions |
| organic_yield_gap | 590 | 266 | 315 | -4.9% | -12.0% | -19.2% | YES | Excludes yield components, flags large positive organic effe |

## Per-Topic Analysis

### biochar_crop_yield

- **Total rows:** 629  |  **KW kept:** 332  |  **LLM kept:** 339
- **KW effect:** +7.3%  |  **LLM effect:** +13.8%  |  **Benchmark (Ye et al. 2020):** +16.0%
- **Total disagreements with KW:** 331
  - LLM excludes / KW kept: 116
  - LLM keeps / KW excluded: 145
  - LLM flags / KW kept: 23

**Top LLM exclusion reasons:**
- non_yield_outcome: 130
- intervention_mismatch: 67
- missing_or_zero_means: 35
- straw_or_biological_yield: 31
- per_plant_unit: 25
- extreme_effect: 2

### intercropping_yield

- **Total rows:** 230  |  **KW kept:** 194  |  **LLM kept:** 158
- **KW effect:** -3.1%  |  **LLM effect:** +11.2%  |  **Benchmark (Yu et al. 2015):** +22.0%
- **Total disagreements with KW:** 84
  - LLM excludes / KW kept: 60
  - LLM keeps / KW excluded: 0
  - LLM flags / KW kept: 0

**Top LLM exclusion reasons:**
- non_yield_outcome: 72

### legume_rotation

- **Total rows:** 542  |  **KW kept:** 363  |  **LLM kept:** 280
- **KW effect:** +17.8%  |  **LLM effect:** +23.3%  |  **Benchmark (Zhao et al. 2022):** +20.0%
- **Total disagreements with KW:** 218
  - LLM excludes / KW kept: 97
  - LLM keeps / KW excluded: 30
  - LLM flags / KW kept: 43

**Top LLM exclusion reasons:**
- non_yield_outcome: 68
- no_rotation_signal: 64
- missing_means: 27
- intercropping_not_rotation: 12
- extreme_effect: 5
- straw_yield: 3

### mycorrhiza_yield

- **Total rows:** 588  |  **KW kept:** 256  |  **LLM kept:** 309
- **KW effect:** +31.4%  |  **LLM effect:** +26.0%  |  **Benchmark (Hoeksema et al. 2010):** +23.0%
- **Total disagreements with KW:** 194
  - LLM excludes / KW kept: 47
  - LLM keeps / KW excluded: 100
  - LLM flags / KW kept: 6

**Top LLM exclusion reasons:**
- non_yield_outcome: 213
- not_amf: 31
- missing_means: 19
- extreme_effect: 15
- root_biomass_not_yield: 1

### notill_tillage

- **Total rows:** 881  |  **KW kept:** 418  |  **LLM kept:** 380
- **KW effect:** +2.7%  |  **LLM effect:** +6.1%  |  **Benchmark (Pittelkow et al. 2015):** -5.7%
- **Total disagreements with KW:** 365
  - LLM excludes / KW kept: 24
  - LLM keeps / KW excluded: 90
  - LLM flags / KW kept: 123

**Top LLM exclusion reasons:**
- non_yield_outcome: 90
- reduced_till_not_notill: 87
- straw_yield: 73
- missing_means: 59
- not_notill: 10
- yield_component: 6
- extreme_effect: 4

### organic_yield_gap

- **Total rows:** 590  |  **KW kept:** 266  |  **LLM kept:** 315
- **KW effect:** -4.9%  |  **LLM effect:** -12.0%  |  **Benchmark (Ponisio et al. 2015):** -19.2%
- **Total disagreements with KW:** 288
  - LLM excludes / KW kept: 84
  - LLM keeps / KW excluded: 120
  - LLM flags / KW kept: 11

**Top LLM exclusion reasons:**
- not_organic: 86
- missing_means: 77
- non_yield_outcome: 58
- comparator_unclear: 27
- straw_yield: 16
- per_plant: 9
- possible_swap_large_positive: 1
- yield_component: 1

## Pipeline Lessons from Full LLM Adjudication

### What systematic errors does LLM catch that keywords miss?

1. **Yield components passed as yield** — Keywords match on `yield` in compound terms like
   `1000-grain weight`, `hectoliter weight`, `grain weight per plant`, `ear length`.
   The LLM recognises these as yield components, not harvestable area yield.
   Affects all 6 topics; most severe in **notill_tillage** (~168 rows) and **biochar_crop_yield** (~111 rows).

2. **Non-yield plant traits** — `plant height`, `stem girth`, `number of leaves`, `leaf area`,
   `phosphorus uptake`, `chlorophyll content`, `mycorrhizal colonization` are extracted by the
   pipeline as PICO-matching but are not yield measures. Most visible in **mycorrhiza_yield**
   (≥47 `plant height` rows) and **biochar_crop_yield**.

3. **Wrong estimand in legume rotation** — Some rows capture the legume crop yield itself,
   not the subsequent cereal yield. The review question is about the ROTATION EFFECT on the
   subsequent crop, not the legume's own productivity.

4. **Intercropping rows in legume_rotation** — Rows labelled `Sorghum grain yield (intercropped
   with legume)` describe simultaneous intercropping, not a rotation effect. Keywords pass
   them because `legume` and `grain yield` both appear; LLM correctly excludes on estimand.

5. **Root biomass ≠ yield** — Root dry/fresh weight extracted as yield outcomes in biochar
   and mycorrhiza topics. Keywords match on `biomass`; LLM excludes root-specific terms.

6. **Confounded interventions in notill** — `no-till + cover crop vs conventional` rows
   cannot isolate the tillage effect. Keyword adjudicator passes them; LLM flags them.

### Which topics improved most? Why?

- **mycorrhiza_yield** — Largest absolute improvement. The keyword adjudicator kept 256 rows
  but ~100+ of these are plant height, stem girth, leaf count, colonisation %, and P uptake.
  These pass keyword filters because the papers are about AMF and report any outcome.
  LLM exclusions bring the dataset closer to a pure yield synthesis.

- **legume_rotation** — Second largest. Legume-specific failure modes (legume-as-measured-crop,
  intercropping confusion) are opaque to keyword matching but transparent to semantic review.

- **biochar_crop_yield** — Third. Straw yield, biological yield, root biomass, 1000-grain
  weight, and plant height are all caught. The benchmark gap narrows (KW: +7.3% vs bench +16%)
  when these non-yield rows are removed.

### Which topics are still far from benchmark even after LLM adjudication?

- **biochar_crop_yield** — LLM effect ~+10% vs benchmark +16%. Structural reasons:
  (a) Many pot experiments included; benchmark mixes field (+12%) and pot (+25%).
  (b) Extraction skews to mid-range biochar rates; high-rate tropical studies may be
  underrepresented. (c) Unweighted mean is biased vs inverse-variance weighted meta-analysis.

- **intercropping_yield** — KW/LLM both near –3% vs benchmark +22%. This is the most severe
  structural mismatch. Core issue: the pipeline compares intercrop component-crop yield vs
  sole-crop yield of the SAME species — intercrop maize is less dense than sole maize so
  individual crop yield is often lower even when SYSTEM productivity (LER) is higher.
  The benchmark (+22%) is a SYSTEM-LEVEL (LER-based) estimate. The pipeline needs to
  either extract LER directly or compute system yield per land unit.
  **Recommendation:** Switch estimand to LER or system yield; or weight by density ratio.

- **organic_yield_gap** — LLM ~–5% vs benchmark –19%. Papers in the dataset appear to be
  from partially-managed organic systems (transitional, market gardens) rather than
  fully-certified organic field crop trials that drive the –19% estimate. Unit heterogeneity
  (some per-pot, some fresh weight vs dry) may also inflate the dataset-level effect.

- **notill_tillage** — LLM ~+1% vs benchmark –5.7%. Sign error remains. The benchmark is
  heavily weighted by large cereal trials (wheat, rice, maize) showing small losses.
  The pipeline dataset captures more positive no-till cases (possibly short-term, tropical,
  or degraded-soil studies). Unweighted mean inflates apparent benefits.

### What changes to config/prompts would help further?

1. **Yield-only extraction prompt** — Add explicit instruction:
   `'Only extract HARVESTABLE YIELD (grain, tuber, fruit, total biomass) per UNIT AREA.
    DO NOT extract yield components (1000-grain weight, ear length, grains per spike),
    morphological traits (plant height, leaf area), nutrient uptake, or colonisation rates.'`

2. **Estimand clarification for legume_rotation** — Prompt must emphasise:
   `'The outcome is the SUBSEQUENT crop yield AFTER the legume pre-crop, NOT the legume yield.'`

3. **LER as primary outcome for intercropping** — Config should set:
   `outcome_description: 'Land Equivalent Ratio (LER) or system yield per unit area'`
   and add LER to outcome_terms with higher priority than component-crop yield.

4. **Intervention isolation rule for notill** — Add warning:
   Only extract comparisons where TILLAGE is the sole difference.
   Exclude rows where no-till is combined with cover cropping, mulching, or irrigation.

5. **Moderate strictness threshold** — Current keyword adjudicator's `low_confidence` flag
   (88 rows in legume_rotation, 32 in mycorrhiza) is the right instinct but too conservative.
   Replace with LLM semantic review of flagged rows rather than blanket exclusion.

6. **Variance type detection** — Several papers report LSD without SE/SD.
   Config should add: `'Convert LSD to SD using: SD = LSD × √n / (2 × t_crit)'`
   to improve meta-analysis weighting quality.

---
*Report generated: 2026-03-26*
*LLM adjudicator: Claude Sonnet 4.6*