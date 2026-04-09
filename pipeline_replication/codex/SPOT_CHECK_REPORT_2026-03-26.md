# Spot-Check Report: Keyword Adjudication Quality Assessment
**Date**: 2026-03-26
**Scope**: Four pipeline replication topics (legume_rotation, mycorrhiza_yield, organic_yield_gap, notill_tillage)
**Purpose**: Assess whether keyword-based adjudication decisions are defensible, identify failure modes, and determine where LLM adjudication is necessary.

---

## Overview

Keyword adjudication applies rule-based filters (numeric thresholds, string matching, hard-coded exclusion lists) to decide which extracted rows enter the meta-analytic pool. This report audits the decisions across four topics by comparing retained effects against published benchmarks and examining the reasons for exclusion.

---

## Topic-by-Topic Assessment

### 1. legume_rotation

**Summary statistics:**
- Total rows: 542 | Kept: 363 (67%) | Excluded: 179 (33%)
- Pipeline effect: +17.66% | Benchmark: +20.0%
- Direction: CORRECT | CI overlap: YES

**Quality rating: GOOD**

The 67% retention rate is reasonable and the pipeline estimate falls within plausible range of the +20.0% benchmark. The confidence intervals overlap, and directional agreement is confirmed. This is the best-performing topic in this audit.

**Key issues identified:**

1. **82 low_confidence exclusions are opaque.** The "low_confidence" flag is applied by a rule that presumably checks some extraction quality score, but no threshold or rationale is documented in the adjudication log. It is unclear whether these 82 rows are genuinely unreliable or whether the threshold is too conservative.

2. **59 universal hard excludes.** These are rows matched to the universal outcome exclusion list (non-yield outcomes such as nodulation counts, root biomass, soil nitrogen). The list appears to be working as intended for legume rotation, but spot-checking a sample of these 59 would confirm no yield-adjacent outcomes were caught in the net.

3. **No verification of legume species scope.** The benchmark (+20.0%) is based on a defined set of legume rotation types. The pipeline does not verify whether the rotation interval, legume species, or cropping system matches the benchmark population. A 17.66% vs 20.0% gap is plausible given this scope variation, but it cannot be confirmed without LLM-level semantic checks.

**What keyword approach got right:**
- Hard-outcome exclusions (nodulation, chlorophyll, root length) correctly removed non-yield rows.
- Numeric plausibility filters caught extreme outliers.
- Retention rate of 67% is not so aggressive as to introduce strong selection bias.

**What LLM adjudication would fix:**
- Clarify the 82 low_confidence exclusions: an LLM reading the source row and context could give a reasoned keep/exclude with citation.
- Verify that "control" comparators are truly untreated or conventional rotation, not a different legume system.
- Confirm that the crop receiving the legume rotation benefit is the target crop (i.e., subsequent cereal, not the legume itself).

---

### 2. mycorrhiza_yield

**Summary statistics:**
- Total rows: 588 | Kept: 256 (43.5%) | Excluded: 332 (56.5%)
- Pipeline effect: +29.26% | Benchmark: +23.0%
- Direction: CORRECT | CI overlap: YES

**Quality rating: ADEQUATE**

The direction is correct and CI overlap holds, but the 56.5% exclusion rate is high and the top exclusion reason ("outcome_mismatch", 112 rows) warrants scrutiny. The pipeline effect (+29.26%) overshoots the benchmark (+23.0%) by 6.3pp — a gap consistent with over-filtering of low-effect studies.

**Key issues identified:**

1. **"outcome_mismatch" accounts for 112 exclusions (34% of all excluded rows).** This is the single largest exclusion category. The keyword matcher is presumably flagging rows where the outcome variable string does not match an approved yield keyword. However, mycorrhizal inoculation studies frequently report yield in non-standard language: "marketable fruit weight," "tuber dry matter," "grain filling rate," "aboveground biomass." Some of these are legitimate yield proxies that keyword matching rejects.

2. **Over-filtering hypothesis supported by effect size overshoot.** If the excluded 112 "outcome_mismatch" rows contain disproportionately many near-zero or negative effects (i.e., studies where mycorrhiza did not boost yield, often reported in weight/area units that look unfamiliar to the keyword list), their removal would upward-bias the retained pool. The 6.3pp overshoot is consistent with this.

3. **No check for mycorrhiza type or inoculation dose.** The benchmark is likely based on AM fungi inoculation of field or greenhouse crops. The pipeline may be including ectomycorrhizal studies, pot experiments with artificial substrate, or dual inoculation (mycorrhiza + rhizobium) without flagging these as scope mismatches.

4. **51% exclusion rate is the second highest in this audit.** While some exclusion is necessary, a rate above 50% on a 588-row dataset suggests the topic configuration may be too narrow or the outcome keyword list too restrictive.

**What keyword approach got right:**
- Correctly excluded colonization rate, spore density, root colonization percentage (non-yield universal excludes).
- Numeric plausibility filters likely caught some extreme pot-experiment values.
- Topic routing to mycorrhiza_yield appears correct (no evidence of off-topic papers).

**What LLM adjudication would fix:**
- Re-examine the 112 "outcome_mismatch" rows: an LLM can read the full row context and determine whether "marketable fruit weight per plant" or "total dry matter per pot" is an acceptable yield proxy for this benchmark.
- Flag ectomycorrhizal vs AM fungi distinction, which keywords cannot detect.
- Identify dual-inoculation confounds (mycorrhiza + PGPR, mycorrhiza + biostimulant) that inflate apparent mycorrhiza effect.

---

### 3. organic_yield_gap

**Summary statistics:**
- Total rows: 590 | Kept: 266 (45.1%) | Excluded: 324 (54.9%)
- Pipeline effect: -4.89% | Benchmark: -19.2%
- Direction: CORRECT | CI overlap: UNKNOWN (gap is 14.3pp, likely no overlap)

**Quality rating: POOR**

Although the direction is correct, the 14.3pp gap between the pipeline estimate (-4.89%) and the benchmark (-19.2%) is too large to attribute to sampling variation alone. Two structural problems are evident: 77 rows missing numeric means, and 197 rows excluded as "topic_exclude_outcome." Together these suggest either the extraction failed to capture the most informative observations or the adjudication filters are systematically removing the high-magnitude yield-gap observations.

**Key issues identified:**

1. **77 rows missing numeric means.** These rows were extracted but could not be entered into effect-size computation. In yield-gap studies, tables often present organic and conventional yield side by side with the gap stated as a ratio or percentage rather than as two separate absolute values. If the extraction only captured one arm (e.g., organic yield only) and could not find the conventional comparator, the row is lost. This is a systematic extraction gap, not a filtering failure.

2. **197 "topic_exclude_outcome" exclusions are the dominant driver.** This is 61% of all excluded rows and 33% of the entire dataset. The topic_exclude_outcome filter is presumably removing rows where the outcome is not farm/field yield — but organic farming studies commonly report both system-level outcomes (yield, profitability) and component outcomes (soil organic matter, biodiversity indices, pest pressure). If the keyword list is removing rows that represent the primary crop yield outcome because the outcome label looks like "system productivity" or "total output," this is a false exclusion.

3. **The benchmark (-19.2%) is based on paired organic-conventional comparisons at field scale.** If the pipeline is predominantly retaining greenhouse, small-plot, or individual component crop rows while excluding field-scale system comparisons, the retained pool will structurally underestimate the yield gap.

4. **No verification of the conventional comparator.** "Conventional" in organic vs. conventional studies has a wide range of meanings (high-input intensive vs. low-input conventional vs. integrated). Keyword matching cannot distinguish these comparator types, which drives heterogeneity.

**What keyword approach got right:**
- Correctly excluded soil chemistry, biodiversity, and water-use outcomes via universal hard excludes.
- Topic routing appears to have correctly identified organic vs. conventional farming papers.

**What LLM adjudication would fix:**
- Recover the 77 rows with missing means: an LLM can re-read the source context and attempt to impute or locate the missing comparator arm.
- Re-examine the 197 "topic_exclude_outcome" rows for false positives — yield-adjacent terms that the keyword list wrongly rejects.
- Verify that "conventional" comparators are genuinely high-input rather than another organic or low-input system.
- Confirm field-scale vs. pot-scale scope alignment with the benchmark.

---

### 4. notill_tillage

**Summary statistics:**
- Total rows: 881 | Kept: 418 (47.4%) | Excluded: 463 (52.6%)
- Pipeline effect: +1.2% | Benchmark: -5.7%
- Direction: WRONG | Magnitude gap: 6.9pp

**Quality rating: POOR**

This is the most problematic topic in the audit. The direction of effect is wrong (pipeline shows a yield benefit from no-till; the benchmark shows a yield penalty), and there is a known data quality failure (AbdulsattarAlrijabo 2014 with implausible 144–609% effects). Two compounding problems are identified: a unit error in at least one study, and a fundamental intervention definition mismatch between the pipeline's inclusion criteria and the benchmark's scope.

**Key issues identified:**

1. **AbdulsattarAlrijabo 2014: 144–609% effect sizes are almost certainly unit errors.** Values of this magnitude for a tillage comparison are physically implausible (a 600% yield increase from switching tillage method has no agronomic basis). The pipeline's numeric plausibility filter did not catch these, which means either the filter threshold is too permissive (e.g., it only flags >1000%) or the values passed some format check while still being wrong. These rows, if retained, would strongly upward-bias the pooled estimate and could alone reverse the sign.

2. **Intervention definition mismatch: "no-till" vs. "conservation agriculture."** The benchmark (-5.7%) is likely based on a narrow definition of no-till (zero tillage, permanent soil cover, crop rotation). The pipeline may be including "reduced tillage," "minimum tillage," "strip-till," and "conservation agriculture" under the same umbrella because keyword matching on "no-till" or "conservation" is ambiguous. These practices have heterogeneous yield effects — strip-till often shows smaller yield penalties or neutral effects compared to full no-till — so including them inflates the estimated benefit.

3. **Wrong-direction result is a red flag for the entire row pool.** A direction error is more serious than a magnitude error because it cannot be explained by scope or measurement variation alone. The combination of implausible outliers and definition mismatch is a plausible mechanism for direction reversal, but this must be verified by manually reviewing the highest-weight retained rows.

4. **52.6% exclusion rate on the largest dataset (881 rows).** Despite heavy filtering, the retained pool still produces a wrong-direction result, indicating that the filtering did not successfully remove the structurally problematic rows.

**What keyword approach got right:**
- Excluded non-yield outcomes (soil structure, aggregate stability, infiltration rate) via universal hard excludes.
- Correctly routed tillage papers to this topic.

**What LLM adjudication would fix:**
- Flag AbdulsattarAlrijabo 2014 and any other rows with effects outside a domain-plausible range (e.g., |effect| > 100% for tillage).
- Classify "reduced tillage," "strip-till," and "conservation agriculture" vs. "zero tillage" — keyword matching cannot make this distinction reliably.
- Verify that the control arm is "conventional tillage" (inversion plowing), not another reduced-tillage variant.
- Detect T/C arm swaps in tables, which could convert a yield penalty into a yield gain.

---

## Cross-Topic Patterns

### What Keywords Reliably Do Well

1. **Universal outcome exclusion.** Removing colonization rates, leaf area, root length, chlorophyll content, and soil chemistry outcomes via keyword list works consistently across all topics. False positive rates for these categories appear low.

2. **Hard numeric filters.** Extreme implausible values (e.g., negative yields, >10,000% effects) are caught when thresholds are set appropriately. The notill_tillage failure shows what happens when the threshold is too permissive.

3. **Topic routing.** Papers are correctly assigned to their topic category. No evidence of systematic cross-topic contamination was found in this audit.

4. **Basic comparator structure.** Two-arm comparisons (treatment vs. control) are generally correctly parsed from tabular data.

### Where Keywords Systematically Fail

1. **Outcome label heterogeneity.** Yield is reported as "grain weight," "marketable fruit weight," "aboveground dry matter," "total shoot biomass per area," etc. A fixed keyword list inevitably has false positives (non-yield caught as yield) and false negatives (yield excluded because label is non-standard).

2. **Intervention granularity.** No-till vs. conservation agriculture, AM fungi vs. ectomycorrhiza, organic vs. low-input conventional — these distinctions require reading the Methods section, not matching a title keyword.

3. **Comparator identity.** What counts as "control" varies by study. Keyword matching cannot determine whether "untreated" means no fertilizer, conventional fertilizer, or a different biostimulant product.

4. **Aggregation level.** Pot vs. field, per-plant vs. per-area, seasonal vs. annual — LLM is needed to apply benchmark-consistent scope criteria.

5. **T/C swap detection.** Requires understanding table orientation and treatment labels, which is beyond keyword capability.

---

## Summary Table

| Topic | Quality | LLM Priority | Key Issue |
|-------|---------|--------------|-----------|
| legume_rotation | GOOD | LOW | 82 opaque low_confidence exclusions; comparator identity unverified |
| mycorrhiza_yield | ADEQUATE | MEDIUM | 112 outcome_mismatch exclusions likely over-filter legitimate yield proxies; effect overshoots benchmark by 6.3pp |
| organic_yield_gap | POOR | HIGH | 77 rows missing means + 197 topic_exclude_outcome exclusions = 14.3pp gap vs. benchmark; field-scale scope unverified |
| notill_tillage | POOR | CRITICAL | Wrong direction; AbdulsattarAlrijabo unit error not caught; intervention definition mismatch (no-till vs. conservation ag) |

---

## Recommendations

1. **Immediate**: Manually inspect and remove AbdulsattarAlrijabo 2014 from notill_tillage before any further analysis. Rerun the pooled estimate without it.

2. **Short-term**: Expand the numeric plausibility filter for tillage studies to reject any single-study effect > 100% (agronomically implausible for tillage).

3. **Medium-term**: Implement LLM-based semantic adjudication for the three failure categories that keywords cannot handle: outcome label normalization, intervention scope verification, and comparator identity check.

4. **Ongoing**: Log the reason and source evidence for every low_confidence exclusion. Anonymous flags without justification make spot-checks impossible.

5. **Design change**: For organic_yield_gap, audit the 197 "topic_exclude_outcome" rows before accepting a -4.89% estimate. A 14.3pp gap from a widely-cited benchmark (-19.2%) is too large to ignore.
