# Benchmark-Aligned Subset Analysis
**Date**: 2026-03-26
**Scope**: Six pipeline replication topics
**Purpose**: Evaluate whether restricting analysis to benchmark-aligned subsets (rows with matching study design, crop, and outcome characteristics) improves agreement with published benchmark estimates.

---

## Background

The benchmark-aligned filtering approach starts from the hypothesis that pipeline estimates diverge from published benchmarks because the pipeline retains rows from studies that are outside the benchmark's implicit scope (different crops, different experimental scales, different outcome definitions). By filtering to a subset that more closely mirrors the benchmark's inclusion criteria, the pipeline estimate should move toward the benchmark value.

This analysis tests that hypothesis across six topics and finds that it holds in only one case (legume_rotation), partially in one (mycorrhiza_yield), and fails in four.

---

## Topic-by-Topic Results

### 1. organic_yield_gap

| Metric | Value |
|--------|-------|
| Full pool effect | -4.89% |
| Benchmark-aligned subset effect | -5.1% |
| Benchmark | -19.2% |
| Subset improvement | No |
| Gap remaining | 14.1pp |

**Interpretation:**

The aligned subset produces essentially the same estimate as the full pool (-5.1% vs. -4.89%), a difference of 0.21pp. Filtering did not bring the estimate closer to the benchmark. This means the rows removed by benchmark-alignment were not the source of the divergence — the divergence is structural.

The -19.2% benchmark is drawn from a literature that compares organic and conventional farming systems at scale, typically using multi-year paired field experiments. The pipeline's retained rows appear to systematically underrepresent high-magnitude yield gaps, likely because: (a) 77 rows with missing means were never entered into effect-size computation, and these may disproportionately come from studies with large gaps; (b) the "topic_exclude_outcome" filter removed 197 rows, some of which may be field-scale system comparisons that would show larger gaps; (c) the pipeline may over-represent short-duration or greenhouse studies where organic yield deficits are smaller.

Benchmark-aligned filtering cannot fix these problems because it operates on the rows that are already present and computable. It cannot recover the 77 missing-mean rows or reverse the 197 incorrect exclusions.

---

### 2. notill_tillage

| Metric | Value |
|--------|-------|
| Full pool effect | +1.2% |
| Benchmark-aligned subset effect | +6.7% |
| Benchmark | -5.7% |
| Subset improvement | No (worsens) |
| Direction | WRONG in both |

**Interpretation:**

Benchmark-aligned filtering makes the result worse, moving the estimate from +1.2% to +6.7% while the benchmark is -5.7%. This is the clearest case of subset filtering backfiring.

The explanation is that the alignment criteria selected the wrong rows. If "benchmark-aligned" is operationalized as rows from cereal crops in temperate field experiments — which is a reasonable approximation of what the benchmark covers — this filter may have preferentially retained the implausible high-effect rows from AbdulsattarAlrijabo 2014 (144–609% effects attributed to a unit error) while removing true low-effect or negative-effect cereal rows.

More fundamentally, the direction error in the full pool cannot be corrected by subset selection. A direction error means the pipeline is measuring the opposite of what the benchmark measures. This is either a T/C arm swap at scale, a systematic intervention definition mismatch (no-till being compared to conservation agriculture rather than conventional tillage), or a combination of both. No subset filter can fix a systematic misclassification of which arm is treatment and which is control.

---

### 3. mycorrhiza_yield

| Metric | Value |
|--------|-------|
| Full pool effect | +29.26% |
| Benchmark-aligned subset effect | +74.7% |
| Benchmark | +23.0% |
| Subset improvement | Partial (CI overlap, but overshoots badly) |
| Direction | Correct in both |

**Interpretation:**

This is a case of partial improvement that is actually degradation in a different dimension. The direction is correct in both the full pool and the aligned subset, and technically the aligned subset's CI may still overlap with +23.0% given the high variance of mycorrhiza effects. However, the point estimate nearly triples from +29.26% to +74.7%, moving further from the benchmark, not closer.

The aligned subset appears to have selected rows from highly controlled studies (single-crop, single-inoculation, pot experiments with favorable soil conditions) where mycorrhiza effects are strongest. These studies are statistically "aligned" to the benchmark's design criteria but are not representative of the mean effect across realistic field conditions, which is what the benchmark captures.

This illustrates a fundamental paradox in benchmark-alignment: the rows most closely matching the benchmark's design criteria are often the rows with the most extreme effects (because tight experimental control amplifies the treatment effect). Aligning on design does not align on effect magnitude if the benchmark aggregates across heterogeneous conditions.

---

### 4. legume_rotation

| Metric | Value |
|--------|-------|
| Full pool effect | +17.66% |
| Benchmark-aligned subset effect | +17.5% |
| Benchmark | +20.0% |
| Subset improvement | Minimal |
| Direction | Correct in both |

**Interpretation:**

This is the one topic where benchmark-alignment neither helps nor hurts: the full pool and aligned subset give almost identical estimates (+17.66% vs. +17.5%), and both are reasonably close to the benchmark (+20.0%). The 2.5pp gap is within plausible range of scope variation (different legume species, different subsequent crops, different climate zones).

The near-identical full-pool and subset estimates indicate that the pipeline's retained rows are already fairly homogeneous with respect to the benchmark's scope. The rows removed by alignment filtering did not carry a systematic effect-size signal in either direction.

The residual 2.5pp gap is most likely explained by scope heterogeneity (the pipeline includes some tropical cropping systems and annual vegetable rotations that the temperate-cereal benchmark does not cover), which is not fixable by further filtering without reducing n to an unreliable level.

---

### 5. biochar_crop_yield

| Metric | Value |
|--------|-------|
| Full pool effect | +6.66% |
| Benchmark-aligned subset effect | +23.8% |
| Benchmark | +16.0% |
| Subset improvement | Better direction, but overshoots |
| Direction | Correct in both |

**Interpretation:**

This is a moderate success case: alignment moves the estimate from +6.66% toward the benchmark's +16.0%, and the aligned subset at +23.8% overshoots but is in the right vicinity. The CI of the aligned subset likely overlaps with +16.0%.

The large movement from full pool to subset (+17pp) suggests the full pool contains many low-effect or near-zero rows from applications where biochar is not agronomically effective (e.g., high-fertility soils, neutral pH soils, crops that don't respond to soil amendment). The benchmark was likely compiled from studies in lower-fertility, slightly acidic soils where biochar's liming and water-retention effects are most pronounced. Filtering to those conditions moves the estimate upward.

The overshoot to +23.8% indicates the subset went too far — it may have removed some legitimate low-response rows along with the off-scope rows. A more refined alignment criterion (e.g., sandy or degraded soil only, not all acid soils) would likely land closer to +16.0%.

This is the best case for benchmark-alignment in this audit: it demonstrates that when the benchmark's implicit scope can be operationalized in row-level metadata (soil type, pH, crop type), filtering genuinely improves estimate accuracy.

---

### 6. intercropping_yield

| Metric | Value |
|--------|-------|
| Full pool effect | -3.09% |
| Benchmark-aligned subset effect | +107.8% (n=9) |
| Benchmark | +22.0% |
| Subset improvement | No |
| Note | n=9 makes estimate unreliable |

**Interpretation:**

The aligned subset is meaningless at n=9. With 9 observations, the confidence interval is wide enough to be consistent with virtually any benchmark, and the point estimate (+107.8%) is driven by a handful of extreme rows that survived aggressive filtering. This is not an estimate; it is noise.

The deeper problem is that intercropping yield is a fundamentally ambiguous estimand. The benchmark (+22.0%) is presumably based on land equivalent ratio (LER) or total system yield relative to sole cropping. The pipeline appears to be computing per-component-crop yield changes, which is a different quantity. When component crop A decreases in a cereal-legume intercrop (because it is competing with component crop B), a naive per-crop comparison produces a negative effect. This is not a measurement of intercropping benefit; it is a measurement of within-system resource competition.

No amount of subset filtering can resolve this estimand mismatch. The pipeline needs to be configured to compute system-level yield (LER or total biomass per area), not component-crop yield, before any comparison to the benchmark is meaningful.

---

## Why Benchmark-Aligned Filtering Fails for Hard Topics

### 1. It cannot recover missing data

For organic_yield_gap, 77 rows have no computable effect size because one arm's mean is missing. Subset filtering operates only on rows with complete data. If the missing-mean rows are not a random sample of the full pool — and they are likely not, since complex multi-arm tables are more likely to lose one arm during extraction — then the computable rows are already a biased sample before any filtering is applied.

### 2. It amplifies selection effects in the wrong direction

For mycorrhiza_yield and notill_tillage, filtering to "aligned" rows selected the rows with the most extreme effects. This happens because experimental designs that closely match a benchmark's criteria are often highly controlled studies where treatment effects are amplified by optimal conditions. Field-scale, multisite, multi-year studies — which the benchmark aggregates — are harder to align precisely and are more likely to be filtered out as "not matching" the benchmark's narrow design criteria.

### 3. It cannot fix systematic misclassification

For notill_tillage, the direction of effect is wrong. No subset of wrongly-classified rows is correctly classified. The pipeline needs to fix its intervention taxonomy (no-till vs. conservation agriculture vs. reduced tillage) before any estimate — full pool or aligned subset — will be correct.

### 4. It collapses sample size dangerously

For intercropping_yield, successive alignment filters reduce n from 590 to 9. Meta-analytic pooling requires adequate n (typically ≥ 20–30 independent studies) to produce a stable estimate. At n=9, the estimate is dominated by individual study characteristics and is not interpretable as a pooled effect.

### 5. It is a post-hoc search for agreement, not a validation

Benchmark-aligned filtering is inherently circular when used as a validation tool. By definition, the subset that most closely resembles the benchmark's inclusion criteria will produce an estimate closest to the benchmark — not because the pipeline is accurate, but because the same selection criteria were applied to both. True validation requires an independent, prospectively defined sample, not a post-hoc filter derived from knowledge of the benchmark's scope.

---

## What Would Actually Help

### For organic_yield_gap

1. **Recover missing-mean rows.** Re-run LLM extraction on the 77 rows with a targeted prompt for the missing comparator arm. A field-scale organic farming paper almost always reports both organic and conventional yield in the same table.

2. **Audit the 197 "topic_exclude_outcome" exclusions.** Present each row to an LLM with the question: "Is this a crop yield outcome comparable to a field-scale organic vs. conventional trial?" Recover false positives.

3. **Add field-scale as an extraction variable.** Many of the low-effect rows are likely pot or microplot experiments. Stratify the analysis by experimental scale and report field-scale separately.

### For notill_tillage

1. **Remove AbdulsattarAlrijabo 2014 immediately.** Rerun the pooled estimate without it.

2. **Add intervention taxonomy as an extraction variable.** Extract the specific tillage practice (zero-till, strip-till, reduced-till, conservation agriculture) and run separate analyses by category. Do not pool these into a single estimate.

3. **Verify T/C arm orientation.** For the 20 highest-weight retained rows, manually check whether the "treatment" arm is no-till and the "control" arm is conventional tillage, or vice versa.

### For mycorrhiza_yield

1. **Expand the outcome keyword list.** Add "marketable yield," "fruit weight per area," "tuber dry weight per plant scaled to area," and similar synonyms. Recover the 112 outcome_mismatch rows that are yield proxies.

2. **Add mycorrhiza type as an extraction variable.** Stratify AM vs. ectomycorrhizal fungi. The benchmark is likely AM-only.

3. **Flag pot vs. field experiments.** The full pool estimate (+29.26%) may be an average of +15% field and +50% pot. Report these separately.

### For intercropping_yield

1. **Redefine the estimand.** Configure extraction to compute land equivalent ratio (LER) or total system yield relative to the best-performing sole crop. Do not use per-component-crop yield changes.

2. **Do not use benchmark-aligned filtering until estimand is fixed.** Any estimate derived from per-component yield is not comparable to a system-yield benchmark, regardless of how the rows are filtered.

### Cross-topic

1. **Implement LLM-based semantic adjudication for the five failure categories identified in the spot-check report.** Keywords handle hard exclusions well but cannot handle outcome label normalization, intervention scope verification, comparator identity checks, T/C swap detection, or aggregation level judgment.

2. **Track effect-size plausibility as a filter.** A domain-specific plausibility range (e.g., tillage effects are unlikely to exceed ±50% in single-study field experiments) should be encoded as a filter prior to pooling. This would have caught the AbdulsattarAlrijabo values.

---

## Summary Recommendations Table

| Topic | Aligned Subset Helps? | Primary Failure Mode | Recommended Fix | Priority |
|-------|----------------------|---------------------|-----------------|----------|
| organic_yield_gap | No (0.21pp gain) | Missing means + over-exclusion | Recover 77 missing-mean rows; audit 197 topic_exclude exclusions | HIGH |
| notill_tillage | No (worsens by 5.5pp) | Estimand mismatch + unit error | Remove outlier; add intervention taxonomy variable; verify T/C orientation | CRITICAL |
| mycorrhiza_yield | Partial (wrong direction, CI overlaps) | Over-filtering of yield proxies + design amplification | Expand outcome keyword list; stratify pot vs. field | MEDIUM |
| legume_rotation | Minimal (0.16pp) | Scope heterogeneity (minor) | None urgent; add LLM comparator verification for precision | LOW |
| biochar_crop_yield | Yes (best case, overshoots slightly) | Low-fertility scope not captured in full pool | Refine alignment criteria to soil type + pH; good benchmark for the approach | MEDIUM |
| intercropping_yield | No (n collapses to 9) | Estimand mismatch (component vs. system yield) | Redefine estimand to LER before any further analysis | CRITICAL |

---

## Conclusion

Benchmark-aligned subset filtering is a useful diagnostic tool but a poor correction mechanism. It works best when: (a) the benchmark's scope can be operationalized in row-level metadata; (b) the full pool contains a large fraction of genuinely off-scope rows; (c) n remains adequate after filtering.

It fails when: (a) the divergence is caused by missing data that cannot be recovered by filtering; (b) systematic misclassification affects the entire pool, not just off-scope rows; (c) the estimand is wrong at the extraction level.

For four of the six topics in this audit, the root cause of benchmark divergence is upstream of filtering — it lies in the extraction configuration, intervention taxonomy, or estimand definition. Fixing those upstream problems will produce larger improvements than any post-hoc subset selection strategy.
