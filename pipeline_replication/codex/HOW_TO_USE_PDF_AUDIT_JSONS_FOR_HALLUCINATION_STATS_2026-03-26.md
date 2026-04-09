## Purpose

This note explains how to use the existing PDF-based audit artifacts for convergence and hallucination-risk statistics.

The key point is that these files are not just model-to-model comparisons. They contain a second-layer PDF reading pass that often reconstructs:

- what the paper is actually about
- where the relevant data are in the paper
- how treatment and control should be defined
- what design features create extraction or matching risk
- which disagreements are likely extraction errors versus alignment artifacts

## Main Audit Families

### 1. Per-paper consensus JSONs

Primary folders:

- `meta_analysis_extractor/output/claude_kimi_full_comparison`
- `meta_analysis_extractor/output/claude_kimi_comparison_full`

Representative files:

- `002_Ziska_1997_consensus.json`
- `003_Baslam_2012_consensus.json`
- `020_Overdieck_1993_consensus.json`

These are especially useful because they combine:

- a `recon` section
- matched / unmatched observation counts
- structured disagreements
- consensus observations
- verification flags

### 2. Concordant error audits

Primary files:

- `meta_analysis_extractor/output/concordant_error_audit.md`
- `meta_analysis_extractor/output/concordant_error_audit_v2.md`

These are highly valuable because they already classify hard cases into root-cause categories.

### 3. Smaller review artifact

- `meta_analysis_extractor/output/model_comparison/claude_review.json`

This appears to be a smaller proof-of-concept review rather than the full archive-wide audit, but it confirms the same idea: the audit LLM read the PDF and judged whether extracted values were genuinely present.

## What The Consensus JSONs Contain

The consensus JSONs expose a useful schema:

- `paper_id`
- `recon`
- `claude_obs`
- `kimi_obs`
- `matched_obs`
- `disagreements`
- `consensus_observations`
- `verification_flags`

### A. `recon`

This is one of the most important sections.

It can include:

- warnings
- variance type/source/confidence
- control definition
- treatment definition
- tables with relevant data
- potential treatment/control confusion
- experimental design
- sample size source
- factorial structure
- extraction guidance
- raw response from the auditing pass

This gives paper-level risk features directly from the PDF audit.

### B. `disagreements`

This section can be used to quantify:

- number of disagreements
- disagreement type (`claude_only`, `kimi_only`, etc.)
- missing control or treatment
- tissue mismatch
- element mismatch
- variance mismatch
- unit mismatch
- moderator mismatch

### C. `consensus_observations`

This is a useful set of “high-support” extracted observations.

It often includes:

- means
- variance info
- `data_source`
- moderators
- confidence
- notes
- effect_pct

These are good candidates for lower-risk observations, but they should not be treated as guaranteed truth.

## What The Concordant Error Audits Add

The concordant error audits are especially important because they expose a crucial lesson:

Some large apparent errors are not hallucinations at all.

Instead, they are often:

- alignment artifacts
- wrong temporal point
- factorial averaging artifact
- year-level averaging mismatch
- moderator-level averaging mismatch
- extraction coverage limitations

This means the system should not use a single “wrong” label for all discrepancies.

It should distinguish at least:

1. `extraction_error`
2. `alignment_error`
3. `paper_structure_ambiguity`
4. `coverage_limitation`

## Best Use Cases For These Audit Files

### 1. Weak labels for error type

The concordant audits provide a starting taxonomy:

- alignment artifact
- extraction limitation
- wrong temporal point
- tissue/condition mismatch
- averaging mismatch

These can be converted into weak supervision labels.

### 2. Paper-level risk features

From `recon`, derive:

- `n_recon_warnings`
- `has_factorial_structure`
- `has_tc_confusion`
- `variance_confidence`
- `tables_with_target_data_n`
- `sample_size_found`
- `experimental_design`

From paper summaries and disagreements:

- `claude_obs`
- `kimi_obs`
- `matched_obs`
- `agreement_fraction`
- `n_disagreements`

### 3. Claim-level risk features

For each disagreement or consensus claim, derive:

- `disagreement_type`
- `element`
- `tissue`
- `confidence`
- `notes_length`
- `has_variance`
- `has_data_source`
- `has_moderators`
- `treatment_control_clarity`

### 4. Dangerous paper-type priors

The audits show some paper structures are intrinsically risky:

- multiple valid CO2 baselines
- multiple timepoints with GT selecting only one
- cross-year averages in GT but year-specific paper data
- factorial designs where GT selects one condition-specific slice
- papers where correct values exist but matching averages them away

These should become explicit paper-level prior flags.

## Recommended Derived Variables

### Paper-level

- `paper_warning_count`
- `paper_has_factorial_design`
- `paper_has_tc_confusion`
- `paper_has_multi_baseline`
- `paper_has_timepoint_specific_gt_risk`
- `paper_has_averaging_risk`
- `paper_has_alignment_artifact_history`
- `paper_has_extraction_limitation_history`

### Claim-level

- `claim_in_consensus`
- `claim_in_disagreement`
- `claim_disagreement_type`
- `claim_has_exact_data_source`
- `claim_has_variance`
- `claim_confidence`
- `claim_has_complete_tc_means`
- `claim_has_structured_moderators`

### Observation-level

- `obs_consensus_support`
- `obs_cross_model_support_n`
- `obs_missingness_level`
- `obs_source_specificity`
- `obs_condition_specificity`
- `obs_temporal_specificity`
- `obs_alignment_risk`

## How These Files Help The Convergence Framework

They help in three distinct ways.

### A. They provide paper-internal structure

The `recon` block is effectively a second-LLM reconstruction of the paper’s evidentiary structure.

### B. They provide external convergence structure

Matched observations and disagreement lists encode cross-model convergence.

### C. They help separate coherent wrongness from hallucination

This is the biggest conceptual contribution.

The concordant error audits show that even when all models agree, the cause of error may be:

- GT aggregation mismatch
- wrong observation selected by the matcher
- missing access to the timepoint the GT used

So these audit files help prevent the false conclusion that:

- `agreement = truth`
- or `disagreement = hallucination`

## Practical Recommendation

These files should be used as:

1. a source of weak labels
2. a source of paper-level risk features
3. a source of error taxonomy
4. a bridge between convergence and document-grounded verification

They should not be treated as a final ground-truth oracle.

## Most Important Lesson

The PDF-audit artifacts strongly suggest that a large fraction of difficult cases are not pure hallucination problems.

They are often:

- matching problems
- condition-selection problems
- averaging problems
- partial extraction coverage problems

That means any future statistical framework should model:

- extraction risk
- alignment risk
- structural ambiguity risk

separately whenever possible.
