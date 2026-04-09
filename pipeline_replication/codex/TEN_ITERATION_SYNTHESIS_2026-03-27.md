# Ten-Iteration Synthesis

This note summarizes the work completed in the current iterative loop after the earlier 4-way policy work.

## What Changed

### 1. Explicit construct-drift features were added

The dataset now carries claim-level construct-drift flags:
- `concentration_vs_content`
- `tissue_mismatch`
- `arm_mismatch`
- `timepoint_mismatch`
- `pooled_vs_subgroup_mismatch`
- `figure_only_target`

These are produced in:
- [build_audit_subset_convergence.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/build_audit_subset_convergence.py)

### 2. A real weak-label bug was found and fixed

The old report parser treated strings like `10/10` as zero-match because it looked for the substring `0/`.
That was contaminating report-derived labels, especially in the second held-out batch.

This was fixed in:
- [build_within_paper_report_features.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/build_within_paper_report_features.py)
- [build_audit_subset_convergence.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/build_audit_subset_convergence.py)

### 3. The report-derived layer became more structured

The parser now captures stronger report signals such as:
- skip / zero-match
- wrong tissue
- treatment-arm confusion
- no concentration data
- figure-digitization limitation
- overall partial rating

It also now computes weighted channel support rather than only simple channel counts.

### 4. A consilience-profile representation was added

Each claim can now be represented by:
- numeric grounding
- cross-model concordance
- within-paper support
- construct drift
- benchmark comparability
- structural risk

Implemented in:
- [build_consilience_profiles.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/build_consilience_profiles.py)

### 5. Counterfactual rescue analysis was added

This asks whether a claim would become usable if one restricted to the correct:
- tissue
- arm
- timepoint
- scale
- subgroup
or added figure digitization.

Implemented in:
- [analyze_counterfactual_rescue.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/analyze_counterfactual_rescue.py)

### 6. A corpus-level extraction-bias dashboard was added

Implemented in:
- [build_extraction_bias_dashboard.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/build_extraction_bias_dashboard.py)

## What Generalized

### Audit subset

The original audited set remains cleanly separated:
- clean -> clean
- alignment -> alignment
- coverage -> coverage

This is the strongest evidence that the newer feature layer did not break the core benchmark subset.

### Held-out batch 1

Main pattern:
- clean claims mostly stay clean
- the remainder fall into `low_support_uncertain`
- alignment claims still generalize well

The persistent ambiguity is mostly in the papers previously labeled as coverage, which often look partly like construct/alignment problems instead.

### Held-out batch 2

This batch looked much worse before the parsing fixes. After the fixes:
- the label structure became much more coherent
- the remaining failures are mostly explainable by construct drift
- the batch is no longer evidence that the 4-way policy collapsed

## What Did Not Generalize Cleanly

### A naive profile policy did not outperform the current rule set

The consilience profile is useful diagnostically, but a simple profile-based classifier did not beat the existing hand-tuned rule policy on the held-out batches.

Reason:
- coverage cases can have high numeric grounding and high within-paper support
- therefore they cannot be separated by a single “overall score”
- the dimensions must be used structurally, not collapsed too early

This is an important result, not a failure:
- the profile works as a representation
- it does not yet work as a better classifier

## Strongest New Empirical Conclusions

### 1. Construct drift is real and useful

The new construct-drift features are more informative than generic `risk_flag_count`, especially in held-out batch 2.

### 2. Alignment/structure remains the dominant failure mode

Across batches, the main failure class is still:
- wrong arm
- wrong tissue
- pooled vs subgroup
- timepoint mismatch
- forced matching of structurally non-equivalent claims

### 3. Coverage limitation is narrower than it first appeared

True coverage cases exist, but they are a smaller and more specific set than the earlier broader “coverage” interpretation suggested.

### 4. Counterfactual rescue is promising

Held-out batch 2 rescue summary:
- `restrict_to_correct_arm`: 23 claims
- `restrict_to_correct_subgroup`: 11 claims
- `restrict_to_correct_scale`: 5 claims
- `add_figure_digitization`: 7 claims
- `restrict_to_correct_tissue`: 2 claims
- `restrict_to_correct_timepoint`: 2 claims

This supports a deeper point:
- many “bad” claims are not unrecoverable extraction failures
- they are recoverable under the right construct restriction

## Taxonomy Decision

At this stage, a fifth top-level bucket is **not yet justified**.

Best current structure:
- keep the 4-way policy
  - `clean_support`
  - `alignment_or_structure_problem`
  - `extraction_coverage_problem`
  - `low_support_uncertain`
- treat construct drift as an orthogonal diagnostic layer, not a replacement taxonomy

Why:
- construct-drift features cut across both alignment and coverage
- they explain *why* a claim is risky
- they do not yet form a stable standalone class

## Literature-Informed Reframing

The broader literature search supports this direction:
- collaborative LLM work supports agreement as evidence, but not as proof
- responsible-AI guidance supports explicit auditability and methodological accountability
- philosophy of science supports triangulation and robustness rather than a single gold standard
- psychometrics suggests that many observed failures are failures of measurement invariance / construct validity

The best high-level framing now is:

This is not only a hallucination-detection problem. It is a construct-validation problem in autonomous evidence synthesis.

## Strongest Publishable Claim Now Supported

The strongest claim currently supported is something like:

Autonomous evidence-synthesis outputs can be screened more effectively by combining cross-model agreement with explicit construct-drift diagnostics than by using generic extraction-risk heuristics alone.

More ambitious but still defensible extension:

In benchmark replication, the dominant failure mode is often construct mismatch rather than pure extraction hallucination, and this can be partially diagnosed using claim-level drift flags and counterfactual rescue analysis.

## Best Next Experiment

The best next experiment is:

1. Implement claim-level construct-drift parsing directly from a larger set of per-paper reports.
2. Expand the rescue analysis to a broader held-out paper family.
3. Test whether a two-stage policy works better:
   - Stage A: clean / uncertain / drifted
   - Stage B: for drifted claims, classify drift mode
4. Use that to decide whether a fifth category becomes justified.
