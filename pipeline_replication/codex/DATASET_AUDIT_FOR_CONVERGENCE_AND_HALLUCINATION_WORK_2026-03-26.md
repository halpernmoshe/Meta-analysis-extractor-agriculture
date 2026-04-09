## Purpose

This is a practical audit of the datasets currently available in:

- `C:\Users\moshe\Dropbox\Testing metaanalyis program`
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor`
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\pipeline_replication`

The goal is to identify which datasets are most useful for:

- convergence statistics
- hallucination-risk modeling
- weak labels for error type
- paper-level and row-level risk features

## Bottom Line

Yes, you have enough data to begin serious work on this.

The strongest resources fall into six groups:

1. ground-truth / matched validation corpora
2. PDF-based audit and consensus corpora
3. cross-model agreement corpora
4. rerun stability corpora
5. row-level semantic audit / adjudication corpora
6. paper-level extraction stashes and figure-stress subsets

## Group 1: Ground-Truth / Matched Validation Corpora

These are the most important sources for labeled mismatch outcomes.

### A. Hui 2023

Folder:

- `meta_analysis_extractor/output/hui2023_full_35`

Key files:

- `validation_matches.csv`
- `validation_report.json`
- `validation_results.json`
- `per_paper/*`

Why useful:

- strong numeric agreement
- paper-level stats
- row-level matched observations
- relatively clean benchmark set

Representative metrics from `validation_report.json`:

- 34 papers processed
- 18 papers matched
- 310 matched observations
- Pearson `r = 0.993`
- MAE `= 1.73 pp`
- direction accuracy `= 0.994`

Use:

- low-noise validation set
- sanity check for risk features
- positive control for “good extraction”

### B. Li 2022

Folder:

- `meta_analysis_extractor/output/li2022_combined`

Key files:

- `validation_matches.csv`
- `validation_report.json`
- `validation_results*.json`
- `per_paper/*`

External folder also important:

- `Li 2022/final_consolidated.csv`
- `Li 2022/comparison_data.csv`
- `Li 2022/comparison_data_improved.csv`
- `Li 2022/paper_stats.csv`
- `Li 2022/failed_matches_log.csv`

Why useful:

- harder, noisier cross-domain set
- direct paper-level mismatch cases
- useful for error taxonomy and paper-level predictors

Representative metrics from `validation_report.json`:

- 49 papers processed
- 28 papers matched
- 166 matched observations
- Pearson `r = 0.74`
- MAE `= 10.19 pp`
- direction accuracy `= 0.873`

Use:

- hard-set benchmark
- paper-level mismatch labels
- provenance/alignment difficulty analysis

### C. Loladze

Folder:

- `meta_analysis_extractor/output/loladze_v3_combined`

Key files:

- `validation_matches.csv`
- `validation_report_full.json`
- `formal_stats/*`

Related folders:

- `meta_analysis_extractor/output/loladze_v3_fixed`
- `meta_analysis_extractor/output/loladze_v3_dual_vision`
- `meta_analysis_extractor/output/loladze_validation`
- `meta_analysis_extractor/output/loladze_validation_pooled`
- `Loladze/validated`
- `Loladze/validation_input`
- `Loladze/mineral_validation_input`

Why useful:

- richest source of cross-model and matching difficulties
- many papers with exact PDFs available
- high value for alignment-risk and structure-risk modeling

Representative metrics from `validation_report_full.json`:

- 50 papers processed
- 46 papers with GT
- 646 matched of 763 GT rows
- Pearson `r = 0.812`
- MAE `= 6.2 pp`
- direction agreement `= 86%`

Use:

- best source for studying matching/alignment versus extraction
- best source for the sandbox paper set

### D. Other ground-truth sources

- `artificial ground truth/combined_ground_truth_final.csv`
- `ground truth dryad/Effect+size+data.csv`
- `meta_analysis_extractor/data/ground_truth`
- `meta_analysis_extractor/references/validation_datasets`

Use:

- supplemental labeling
- cross-checking benchmark assumptions

## Group 2: PDF-Based Audit And Consensus Corpora

These are critical for the current convergence project.

### A. Per-paper consensus JSONs

Folders:

- `meta_analysis_extractor/output/claude_kimi_full_comparison`
- `meta_analysis_extractor/output/claude_kimi_comparison_full`

Why useful:

- one JSON per paper
- includes `recon`, `disagreements`, `consensus_observations`, `verification_flags`
- effectively a PDF-grounded audit layer

Use:

- paper-level risk features
- claim-level support features
- disagreement-type features
- document-structure warnings

### B. Concordant error audits

Files:

- `meta_analysis_extractor/output/concordant_error_audit.md`
- `meta_analysis_extractor/output/concordant_error_audit_v2.md`

Why useful:

- already classify some large-error cases into root-cause categories
- shows many “errors” are actually alignment or GT-aggregation problems

Use:

- weak labels for:
  - alignment artifact
  - extraction limitation
  - temporal-point mismatch
  - averaging mismatch
  - condition mismatch

### C. Small review artifact

- `meta_analysis_extractor/output/model_comparison/claude_review.json`

Use:

- proof that PDF-based value confirmation exists in the archive
- less important than the per-paper consensus set

## Group 3: Cross-Model Agreement Corpora

These are the best source for external convergence features.

### A. Inter-model agreement

Folder:

- `meta_analysis_extractor/output/inter_model_agreement`

Key files:

- `pairwise_comparison.csv`
- `agreement_stats.json`

Why useful:

- direct Claude/Gemini/Kimi comparisons
- row-level direction and effect comparisons
- explicit agreement statistics

Representative metrics:

- 3-model agreement on direction = `96.1%`
- Fleiss kappa = `0.942`
- pairwise agreement high but not perfect

Use:

- `cross_model_support_n`
- `direction_agreement_rate`
- `magnitude_dispersion`
- model-specific disagreement patterns

### B. Gemini/Claude validation

Folder:

- `meta_analysis_extractor/output/gemini_claude_validation`

Key file:

- `results.json`

Use:

- detailed paired model outputs
- observation-level comparison

### C. Other comparison sets

- `meta_analysis_extractor/output/comparison`
- `meta_analysis_extractor/output/comparison_v2`
- `meta_analysis_extractor/output/model_comparison`
- `meta_analysis_extractor/output/merged_gemini3`
- `meta_analysis_extractor/output/baseline_claude_loladze`
- `meta_analysis_extractor/output/baseline_gemini_loladze`
- `meta_analysis_extractor/output/baseline_kimi_loladze`

Use:

- stress tests
- ablations
- alternate comparison baselines

## Group 4: Rerun Stability Corpora

These support reproducibility features.

Primary files/folders:

- `meta_analysis_extractor/output/replication_agreement.json`
- `meta_analysis_extractor/output/reproducibility`
- `meta_analysis_extractor/output/reproducibility_test`
- `meta_analysis_extractor/output/hui2023_agent_replication`
- `meta_analysis_extractor/output/li2022_agent_replication`
- `meta_analysis_extractor/output/loladze_agent_replication`

Why useful:

- same model rerun differences
- run-to-run instability as a risk signal

Representative metrics from `replication_agreement.json`:

- Loladze run-to-run `r = 0.816`
- Hui run-to-run `r = 0.946`
- Li run-to-run `r = 0.836`

Use:

- `rerun_direction_agreement`
- `rerun_magnitude_stability`
- `paper_instability_score`

## Group 5: Row-Level Semantic Audit / Adjudication Corpora

These are newer and highly useful for row-risk labeling.

### A. Row audit

Folder:

- `pipeline_replication/codex/outputs/row_audit`

Key files:

- `row_audit_summary.json`
- topic-specific `flagged_rows_top50.json`
- topic-specific `row_audit.jsonl`
- topic-specific `summary.json`

Why useful:

- explicit severity labels:
  - `clean`
  - `needs_review`
  - `likely_off_target`
- explicit reason tags
- explanatory notes

Use:

- weak supervision labels for row risk
- topic-specific error taxonomies

### B. Codex decisions

Folder:

- `pipeline_replication/codex/outputs/codex_decisions`

Key files:

- `universal_adjudication_summary.json`
- topic-specific `decisions.jsonl`
- topic-specific `summary.json`

Why useful:

- structured keep/exclude/flag decisions
- normalized outcome/intervention/estimand fields
- swap flags
- exclusion reasons

Use:

- row-level semantic labels
- estimand mismatch features
- intervention/comparator mismatch features

### C. LLM decisions

Folder:

- `pipeline_replication/codex/outputs/llm_decisions`

Use:

- topic-level summaries of LLM adjudication versus keyword systems
- additional evidence on systematic row filtering behavior

## Group 6: Paper-Level Extraction Stashes And Stress-Test Sets

These are not ideal labels by themselves, but they are useful for reconstruction and targeted experiments.

### A. Paper-level extraction stashes

- `Li 2022/downloaded_papers/extraction_*`
- `Loladze/mineral_extraction`
- `Loladze/validated/extraction_*`
- `My pdfs/Included_Papers/extraction_output`
- `My pdfs/extraction_*`

Use:

- raw output reconstruction
- prompt/version history
- per-paper source evidence

### B. Figure and vision stress tests

- `meta_analysis_extractor/output/fig_only_validation`
- `meta_analysis_extractor/output/figure_extraction_comparison`
- `meta_analysis_extractor/output/kimi_figure_test`
- `meta_analysis_extractor/output/kimi_vision_rescue`
- `meta_analysis_extractor/output/vision_test_2026-02-03`
- `meta_analysis_extractor/output/rerun_worst_gemini3`

Use:

- figure-only dependence
- extraction coverage limitations
- paper-type specific risk priors

## Best Datasets By Task

### Best for direct mismatch labels

1. `hui2023_full_35`
2. `li2022_combined`
3. `loladze_v3_combined`

### Best for PDF-grounded audit labels

1. `claude_kimi_full_comparison`
2. `claude_kimi_comparison_full`
3. `concordant_error_audit*.md`

### Best for cross-model convergence features

1. `inter_model_agreement`
2. `gemini_claude_validation`
3. `model_comparison`

### Best for rerun stability features

1. `replication_agreement.json`
2. `*_agent_replication`
3. `reproducibility*`

### Best for row-level semantic risk labels

1. `pipeline_replication/codex/outputs/row_audit`
2. `pipeline_replication/codex/outputs/codex_decisions`
3. `pipeline_replication/codex/outputs/llm_decisions`

## Most Important Gaps

You do not yet have:

- one unified training table
- one clean binary hallucination label
- one universal row matcher across all models and datasets

But these are integration gaps, not data-availability gaps.

You already have enough to build:

- weak labels
- paper-level risk models
- claim-level convergence models
- row-level pilot models on aligned subsets

## Recommended First Integration Order

1. `claude_kimi_*consensus.json`
   Use for paper-level warnings, disagreement counts, and document-structure features.

2. `validation_matches.csv` plus validation reports
   Use for mismatch outcomes.

3. `inter_model_agreement/pairwise_comparison.csv`
   Use for cross-model support and direction agreement.

4. `replication_agreement.json` and `*_agent_replication`
   Use for stability features.

5. `row_audit` and `codex_decisions`
   Use for semantic row-risk labels and outcome-type labels.

## Bottom Line

The archive is stronger than it may feel from memory.

You do not have one perfect dataset, but you do have a very good ecosystem of partially overlapping datasets that support:

- labeled mismatch analysis
- PDF-based audit reconstruction
- cross-model convergence
- rerun stability
- row-level semantic audit

That is enough to start building a serious convergence / hallucination-risk framework now.
