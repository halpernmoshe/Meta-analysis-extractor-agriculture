## Purpose

This note defines the first unified analysis schema for convergence and hallucination-risk work.

The goal is not to merge every dataset into one perfect row table immediately.

Instead, the design uses multiple linked tables:

- source index
- paper-level features
- claim-level features
- row-level features
- label table

This avoids forcing exact row matching where it is not yet reliable.

## Design Principles

1. Separate extraction risk from alignment risk.
2. Separate paper-level, claim-level, and row-level evidence.
3. Preserve provenance back to the source file.
4. Allow weak labels and partial labels.
5. Support multiple datasets with different schemas.

## Table 1: `dataset_index`

Purpose:

- inventory of all usable datasets
- one row per dataset bundle

Primary key:

- `dataset_name`

Key fields:

- `dataset_group`
- `dataset_name`
- `path`
- `role`
- `granularity`
- `label_type`
- `priority`
- `key_artifacts`
- `notes`

Source:

- `DATASET_INDEX_FOR_CONVERGENCE_WORK_2026-03-26.csv`

## Table 2: `paper_features`

Purpose:

- one row per paper per dataset context
- store document-structure risk and agreement features

Primary key:

- `paper_key`

Suggested key:

- `dataset_name + paper_id`

Core identity fields:

- `dataset_name`
- `paper_id`
- `paper_title`
- `paper_year`
- `topic_family`
- `pdf_path`

PDF-audit / recon fields:

- `paper_warning_count`
- `paper_warnings_json`
- `has_factorial_structure`
- `factorial_structure_raw`
- `has_tc_confusion`
- `variance_confidence`
- `variance_type_detected`
- `tables_with_target_data_n`
- `sample_size_found`
- `experimental_design`

Cross-model fields:

- `n_models_compared`
- `claude_obs`
- `kimi_obs`
- `gemini_obs`
- `matched_obs`
- `agreement_fraction`
- `n_disagreements`

Rerun fields:

- `rerun_available`
- `rerun_stability_score`
- `rerun_direction_agreement`
- `rerun_magnitude_dispersion`

Paper-risk flags:

- `paper_has_multi_baseline`
- `paper_has_timepoint_specific_risk`
- `paper_has_averaging_risk`
- `paper_has_alignment_artifact_history`
- `paper_has_extraction_limitation_history`
- `paper_has_figure_dependence`

Outputs / labels:

- `paper_risk_level_initial`
- `paper_error_type_primary`
- `paper_notes`

## Table 3: `claim_features`

Purpose:

- intermediate representation when exact row matching is too brittle

Primary key:

- `claim_key`

Suggested key:

- `dataset_name + paper_id + outcome_class + tissue_class + direction + source_channel`

Identity fields:

- `dataset_name`
- `paper_id`
- `claim_id`
- `model`
- `source_channel`

Normalized semantics:

- `outcome_class`
- `tissue_class`
- `intervention_class`
- `comparator_class`
- `estimand_class`
- `study_setting`

Effect fields:

- `direction`
- `magnitude_band`
- `effect_pct`
- `effect_scale`

Internal convergence fields:

- `abstract_support`
- `results_support`
- `table_support`
- `caption_support`
- `figure_support`
- `conclusion_support`
- `supporting_sections_n`
- `conflicting_sections_n`
- `missing_sections_n`
- `internal_convergence_score`

External convergence fields:

- `cross_model_support_n`
- `cross_model_conflict_n`
- `direction_agreement_rate`
- `magnitude_dispersion`
- `rerun_support`
- `benchmark_support`
- `external_convergence_score`

Risk flags:

- `swap_risk`
- `granularity_risk`
- `averaging_risk`
- `timepoint_risk`
- `figure_only_dependence`
- `missing_variance_flag`
- `missing_control_flag`
- `missing_treatment_flag`

Labels:

- `claim_label_initial`
- `claim_error_type`
- `claim_confidence_label`

## Table 4: `row_features`

Purpose:

- exact extracted row representation
- used only where row alignment is available or acceptable

Primary key:

- `row_key`

Suggested key:

- use existing `row_id` when available
- otherwise derive `dataset_name + paper_id + source_row_number`

Identity fields:

- `dataset_name`
- `topic`
- `paper_id`
- `row_id`
- `row_number`
- `model`

Raw extraction fields:

- `outcome_raw`
- `tissue_raw`
- `element_raw`
- `treatment_description`
- `control_description`
- `treatment_mean`
- `control_mean`
- `treatment_variance`
- `control_variance`
- `variance_type`
- `n_treatment`
- `n_control`
- `unit`
- `data_source`
- `confidence_raw`
- `notes`

Normalized fields:

- `normalized_outcome_class`
- `normalized_tissue_class`
- `normalized_estimand_class`
- `normalized_study_setting`

Cross-model fields:

- `cross_model_support_n`
- `cross_model_direction_agreement`
- `cross_model_effect_range`

Audit fields:

- `severity`
- `reason_tags_json`
- `decision`
- `intervention_match`
- `comparator_match`
- `outcome_match`
- `estimand_match`
- `needs_tc_swap`
- `exclusion_reason`

Ground-truth alignment fields:

- `gt_available`
- `gt_effect`
- `gt_match_status`
- `effect_abs_error`
- `direction_match`
- `alignment_candidate_count`

Derived risk fields:

- `row_internal_support_score`
- `row_external_support_score`
- `row_numeric_grounding_score`
- `row_alignment_risk_score`
- `row_hallucination_risk_score`

## Table 5: `labels`

Purpose:

- central place for all labels, even if weak or partial

Primary key:

- `label_key`

Suggested key:

- `entity_type + entity_id + label_source + label_name`

Identity fields:

- `entity_type`
  - `paper`
  - `claim`
  - `row`
- `entity_id`
- `label_source`

Label fields:

- `label_name`
- `label_value`
- `label_confidence`
- `label_type`
  - `ground_truth`
  - `weak_supervision`
  - `audit`
  - `derived`

Examples:

- `match_status = matched`
- `error_type = alignment_artifact`
- `severity = needs_review`
- `decision = exclude`
- `paper_risk = high`

## Minimum Viable Derived Labels

These can be built now from existing data.

### Paper-level

- `paper_has_major_mismatch`
- `paper_has_alignment_artifact_history`
- `paper_has_extraction_limitation_history`
- `paper_high_disagreement`
- `paper_high_instability`

### Claim-level

- `claim_supported`
- `claim_conflicted`
- `claim_unresolved`

### Row-level

- `row_clean`
- `row_needs_review`
- `row_likely_off_target`
- `row_keep`
- `row_exclude`
- `row_flag`

## Schema Crosswalk From Existing Files

### `hui2023_full_35/validation_matches.csv`

Maps to:

- `paper_id`
- `tissue`
- `ext_effect`
- `gt_effect`
- `abs_error`
- `app_type`

### `li2022_combined/validation_matches.csv`

Maps to:

- `paper_id`
- `crop`
- `category`
- `product`
- `scale`
- `direction_match`
- `effect_diff_pp`

### `loladze_v3_combined/validation_matches.csv`

Maps to:

- `paper`
- `actual_paper`
- `el`
- `gt_tissue`
- `gt_eco2`
- `info`
- `err`
- `n_candidates`

### `inter_model_agreement/pairwise_comparison.csv`

Maps to:

- `paper_id`
- `element`
- `tissue`
- `claude_eff`
- `kimi_eff`
- `gemini_eff`
- `claude_dir`
- `kimi_dir`
- `gemini_dir`
- `all_agree`

### `*_consensus.json`

Maps to:

- `paper_id`
- `recon.*`
- `claude_obs`
- `kimi_obs`
- `matched_obs`
- `disagreements[]`
- `consensus_observations[]`
- `verification_flags`

### `row_audit/*.jsonl`

Maps to:

- `row_number`
- `paper_id`
- `outcome`
- `effect_pct`
- `confidence`
- `severity`
- `reasons`
- `treatment_description`
- `control_description`
- `notes`

### `codex_decisions/*/decisions.jsonl`

Maps to:

- `row_id`
- `decision`
- `intervention_match`
- `comparator_match`
- `outcome_match`
- `estimand_match`
- `needs_tc_swap`
- `normalized_outcome_class`
- `normalized_study_setting`
- `normalized_estimand_class`
- `exclusion_reason`

## What To Build First

### Step 1

Create `paper_features` for the sandbox papers and the core validation corpora.

### Step 2

Create `claim_features` for the sandbox papers using the consensus JSONs and inter-model files.

### Step 3

Create `row_features` only for:

- rows with validation matches
- rows with row_audit labels
- rows with codex decision labels

This avoids overcommitting to bad row matching.

## Bottom Line

The first unified schema should be relational, not monolithic.

That will let you:

- use the data you already have
- avoid forcing exact row matching too early
- combine convergence, audit, ground-truth, and adjudication evidence cleanly
