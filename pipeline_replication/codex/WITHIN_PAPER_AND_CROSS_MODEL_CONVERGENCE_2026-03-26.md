## Purpose

This memo proposes a new validation direction for hallucination-risk diagnostics in the extraction pipeline.

The key shift is:

- Do not rely only on row-to-row matching across LLMs.
- Also measure whether an extracted claim is supported by multiple evidence channels within the same paper.

This matters because cross-LLM alignment is difficult, brittle, and expensive. Different models often extract:

- different row granularities
- different subgroup decompositions
- different outcome labels
- different units or effect formulations
- different subsets of the same table or figure

That makes exact matching hard even when both models are substantively correct.

By contrast, a paper itself contains multiple partially independent evidence streams:

- abstract
- results text
- tables
- figure captions
- figures
- discussion / conclusions
- significance statements
- intervention and outcome terminology repeated across sections

If an extracted row or paper-level conclusion is correct, it should usually cohere with more than one of these streams. Hallucinations should often break that coherence.

## Core Idea

We should build two related but distinct validation layers.

### 1. External convergence

Does an extracted result converge with:

- another LLM
- another run of the same LLM
- a benchmark dataset
- a consensus extraction

This is already partially supported by existing artifacts in:

- `meta_analysis_extractor/output/hui2023_full_35`
- `meta_analysis_extractor/output/li2022_combined`
- `meta_analysis_extractor/output/loladze_v3_combined`
- `meta_analysis_extractor/output/hui2023_agent_replication`
- `meta_analysis_extractor/output/li2022_agent_replication`
- `meta_analysis_extractor/output/loladze_agent_replication`
- `meta_analysis_extractor/output/inter_model_agreement`
- `meta_analysis_extractor/output/claude_kimi_full_comparison`
- `meta_analysis_extractor/output/gemini_claude_validation`

### 2. Internal convergence

Does an extracted result converge with multiple parts of the same paper?

This can be assessed even when row-to-row matching across LLMs is messy.

Instead of forcing exact row identity, we can measure convergence at coarser levels:

- direction of effect
- outcome class
- intervention identity
- comparator identity
- whether a benefit / harm / null result is reported
- approximate effect magnitude band
- presence of statistical significance

## Why This Is Better Than Matching Alone

Cross-model matching has been one of the hardest engineering problems in the project.

The main reasons:

- one model may split a paper into many subgroup rows while another aggregates
- one model may use paper-specific labels while another normalizes
- one model may read a figure and another a table
- one model may output component yields and another total yield
- one model may omit rows that another includes because of ambiguity

This means disagreement is informative, but exact row matching is often too strict as the only basis for trust.

Within-paper convergence avoids some of that brittleness.

For example, an extracted claim is more plausible if:

- the abstract says yield increased
- the results text describes a significant treatment advantage
- a table shows treatment means above control
- the conclusion says the intervention improved yield

Even if no second model extracts the exact same rows, the paper-internal evidence is convergent.

## Statistical Framing

The right target is not a single hallucination test. It is a composite risk model.

Each extracted row or paper-level claim should have a hallucination-risk profile based on:

- external convergence
- internal convergence
- arithmetic consistency
- provenance specificity
- extremeness / anomaly
- missingness / uncertainty

## Candidate Feature Families

### A. External convergence features

- `cross_model_n_support`
  Number of models producing a compatible claim.

- `cross_model_direction_agreement`
  Fraction of models agreeing on sign.

- `cross_model_magnitude_dispersion`
  Spread in effect magnitude across compatible extractions.

- `cross_model_outcome_agreement`
  Whether models agree on outcome class.

- `cross_model_intervention_agreement`
  Whether models agree on intervention identity.

- `rerun_direction_agreement`
  Same model, multiple runs, direction consistency.

- `rerun_magnitude_stability`
  Same model, multiple runs, absolute difference in effect estimate.

- `benchmark_match_status`
  Exact / approximate / no match against human dataset where available.

### B. Internal convergence features

- `abstract_support`
  Abstract supports positive / negative / null direction.

- `results_text_support`
  Results section supports the extracted direction.

- `table_support`
  Structured table evidence supports the extracted direction or comparison.

- `figure_support`
  Figure / caption support exists.

- `conclusion_support`
  Discussion/conclusion supports the extracted interpretation.

- `section_conflict_count`
  Number of sections that explicitly contradict the extracted claim.

- `source_count_supporting`
  Number of paper sections supporting the claim.

- `source_count_conflicting`
  Number of paper sections conflicting with the claim.

- `within_paper_direction_consensus`
  Strength of directional convergence across sections.

- `within_paper_outcome_consensus`
  Whether sections refer to the same outcome class.

- `within_paper_intervention_consensus`
  Whether sections refer to the same intervention/comparator concept.

### C. Provenance and extraction quality features

- `source_specificity`
  Named table/cell > figure panel > vague prose.

- `has_numeric_trace`
  Whether the extraction cites concrete values visible elsewhere in the paper.

- `has_variance`
  Whether SD/SE/CI/N exist.

- `unit_consistency`
  Whether units are consistent across the paper and extraction.

- `row_granularity`
  Simple arm-level / subgroup-level / multi-factor / unclear.

- `paper_complexity`
  Proxy such as number of tables, factors, outcomes, or extracted rows.

### D. Statistical anomaly features

- `effect_extremeness_z`
  Effect magnitude relative to topic-specific distribution.

- `variance_extremeness_z`
  Variance relative to topic- and outcome-specific distribution.

- `digit_pattern_flag`
  Suspiciously rounded or templated values.

- `incoherent_significance_flag`
  Numeric effect inconsistent with textual significance claims.

- `count_inconsistency_flag`
  Extracted number of comparisons not plausible given paper structure.

## Levels Of Analysis

We should not force everything to row level immediately.

There should be at least three levels:

### 1. Paper-level

Question:
Does this paper’s extracted evidence cohere internally and externally?

Useful when row matching is too noisy.

### 2. Claim-level

Question:
Does this coarse claim hold?

Examples:

- treatment increased yield
- no significant effect on grain yield
- FACE reduced mineral concentration

This may be the most robust intermediate level.

### 3. Row-level

Question:
Is this exact extracted row likely real and correctly grounded?

This is the hardest but ultimately most useful level.

## Suggested First Statistical Design

Start with a simpler paper-level or claim-level model before attempting full row-level hallucination detection.

### Phase 1: Paper-level risk model

Outcome:

- `paper_has_major_mismatch`
- `paper_ground_truth_mae_high`
- `paper_cross_model_disagreement_high`
- `paper_rerun_instability_high`

Predictors:

- counts of supporting sections
- counts of conflicting sections
- disagreement among models
- instability across reruns
- figure-only dependence
- missing variance rate
- outcome diversity
- number of extracted rows

Reason:
Paper-level labels already exist in many prior outputs and are easier to align.

### Phase 2: Claim-level support model

Outcome:

- claim supported / unsupported / conflicted

A claim can be represented as:

- paper
- intervention
- comparator
- outcome class
- direction
- approximate effect band

This reduces dependence on exact numeric row alignment.

### Phase 3: Row-level hallucination-risk model

Outcome:

- exact ground-truth mismatch
- severe disagreement across models
- unstable rerun
- manually flagged suspicious row

This phase should use only the subset with high-quality aligned labels.

## Existing Data Sources That Make This Feasible

### For ground-truth labels

- `artificial ground truth/combined_ground_truth_final.csv`
- `ground truth dryad/Effect+size+data.csv`
- `meta_analysis_extractor/output/hui2023_full_35`
- `meta_analysis_extractor/output/li2022_combined`
- `meta_analysis_extractor/output/loladze_v3_combined`

### For rerun stability

- `meta_analysis_extractor/output/hui2023_agent_replication`
- `meta_analysis_extractor/output/li2022_agent_replication`
- `meta_analysis_extractor/output/loladze_agent_replication`
- `meta_analysis_extractor/output/replication_agreement.json`

### For cross-model convergence

- `meta_analysis_extractor/output/inter_model_agreement/pairwise_comparison.csv`
- `meta_analysis_extractor/output/inter_model_agreement/agreement_stats.json`
- `meta_analysis_extractor/output/claude_kimi_full_comparison/consensus_results.csv`
- `meta_analysis_extractor/output/gemini_claude_validation/results.json`
- `meta_analysis_extractor/output/model_comparison`

### For difficult-case subsets

- `meta_analysis_extractor/output/fig_only_validation`
- `meta_analysis_extractor/output/rerun_worst_gemini3`
- `meta_analysis_extractor/output/validation_experiment_improved`
- `pipeline_replication/codex/outputs/row_audit`

## Practical Recommendation

Do not begin by solving full exact row matching across all models and all papers.

Instead:

1. Build paper-level and claim-level convergence scores first.
2. Use existing matched datasets to calibrate those scores against known mismatch outcomes.
3. Only then move to exact row-level hallucination-risk prediction on the aligned subsets.

This is likely to be more robust and much less brittle than forcing universal row matching from the start.

## Initial Hypotheses To Test

1. Low rerun stability predicts ground-truth mismatch.
2. Low cross-model agreement predicts ground-truth mismatch.
3. Low within-paper convergence predicts ground-truth mismatch.
4. Figure-only dependence raises hallucination risk.
5. Claims supported by abstract + results text + table are much less error-prone than claims supported by only one source.
6. Direction agreement is more stable than exact magnitude agreement.
7. Paper-level convergence can identify high-risk papers even when row matching is poor.

## Bottom Line

The project should stop treating row alignment as the only path to validation.

The better framework is:

- external convergence across models and reruns
- internal convergence across paper sections
- arithmetic and provenance checks
- calibrated risk scoring against existing ground-truth datasets

That framework is more realistic, more scientifically interpretable, and probably much easier to scale.
