# Benchmark Spec Template

Use this template to convert a benchmark meta-analysis paper into a structured input for the pipeline before running a topic.

The purpose is not to copy the benchmark result. The purpose is to capture the benchmark paper's explicit operational definitions so the pipeline estimates the same construct.

## 1. Benchmark Identity

- benchmark_citation:
- benchmark_title:
- benchmark_year:
- benchmark_journal:
- benchmark_doi:

## 2. Core Research Question

- benchmark_question:
- target_estimand:
  - examples:
    - direct harvest yield
    - system productivity
    - land equivalent ratio
    - biomass response

## 3. Intervention Definition

- intervention_label:
- intervention_required_features:
- intervention_excluded_variants:
- intervention_ambiguity_notes:

Examples:
- strict no-till only
- exclude reduced tillage
- AMF inoculation versus non-AMF control
- certified organic or explicit organic-principles system

## 4. Comparator Definition

- comparator_label:
- comparator_required_features:
- comparator_excluded_variants:
- comparator_ambiguity_notes:

## 5. Outcome Definition

- primary_outcome_label:
- acceptable_primary_outcomes:
- excluded_outcomes:
- acceptable_units:
- outcome_hierarchy:

Examples:
- include grain yield, fruit yield, tuber yield
- exclude quality traits, concentration traits, straw yield
- include LER only for primary replication

## 6. Study Setting

- allowed_settings:
  - field
  - greenhouse
  - pot
  - mixed
- excluded_settings:
- setting_notes:

## 7. Study Design / Eligibility

- included_study_types:
- excluded_study_types:
- special_design_rules:

Examples:
- field trials only
- exclude reviews and meta-analyses
- require true treatment-control comparison

## 8. Critical Moderators

- moderator_1:
- moderator_2:
- moderator_3:

For each moderator:
- name:
- why_it_matters:
- expected_levels:
- whether_required_for_alignment:

Examples:
- crop class
- climate zone
- residue retention
- crop rotation
- biochar rate
- N fertilizer level
- study setting

## 9. Benchmark Subgroup Logic

- subgroup_1:
  - definition:
  - reported_effect:
  - notes:
- subgroup_2:
  - definition:
  - reported_effect:
  - notes:

This section is for pre-specifying benchmark-aligned secondary analyses.

## 10. Known Estimand Traps

- trap_1:
- trap_2:
- trap_3:

Examples:
- LER vs component crop yield
- direct yield vs biomass proxy
- strict no-till vs conservation agriculture
- organic system vs amendment trial

## 11. Prompt Consequences

How should this benchmark spec change extraction prompts?

- extraction_priority_1:
- extraction_priority_2:
- extraction_priority_3:

## 12. Post-Processing Consequences

How should this benchmark spec change post-processing?

- keep_rules:
- exclude_rules:
- flag_rules:
- benchmark_alignment_labels:

## 13. Provenance

- who_created_spec:
- date_created:
- created_before_results_seen:
- notes:
