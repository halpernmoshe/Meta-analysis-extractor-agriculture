## Purpose

This note defines a small, high-signal sandbox for developing convergence-based hallucination-risk statistics.

The sandbox is intentionally small. The goal is not broad coverage at first. The goal is to choose papers and model outputs that already contain:

- meaningful cross-model disagreement
- at least some clean agreement controls
- enough internal paper structure to score abstract/text/table/figure/conclusion convergence
- existing comparison artifacts in the repository

## Selected LLMs

Use these three models first:

- Claude
- Gemini
- Kimi

Rationale:

- they already appear in multiple comparison and agreement outputs
- pairwise and 3-model agreement statistics already exist
- they provide enough heterogeneity to create useful disagreement signals

Main supporting folders:

- `meta_analysis_extractor/output/inter_model_agreement`
- `meta_analysis_extractor/output/claude_kimi_full_comparison`
- `meta_analysis_extractor/output/gemini_claude_validation`
- `meta_analysis_extractor/output/model_comparison`

## Selected Papers

### Clean / control papers

These are useful because they should have high internal and external convergence.

1. `020_Overdieck_1993`
2. `031_Pal_2003`

Rationale:

- high Claude/Kimi agreement in `claude_kimi_full_comparison/summary.json`
- good candidates for low-risk baseline papers

### Hard disagreement papers

These are useful because they expose the exact pathologies we care about.

3. `002_Ziska_1997`
4. `003_Baslam_2012`
5. `004_Finzi_2001`

Rationale:

- `002_Ziska_1997` shows element mismatch and direction conflict
- `003_Baslam_2012` shows high disagreement and treatment/control swap risk
- `004_Finzi_2001` shows tissue-label mismatch and swap risk

Supporting evidence:

- `meta_analysis_extractor/output/model_comparison/disagreement_analysis.json`
- `meta_analysis_extractor/output/inter_model_agreement/pairwise_comparison.csv`

### Figure / structure-stress paper

6. `007_Woodin_1992`

Rationale:

- good candidate for figure/caption versus text/table convergence checks
- appears in comparison artifacts with clear tissue-level variation

Optional alternates:

- papers from `meta_analysis_extractor/output/fig_only_validation`
- `040_Pfirrmann_1996`
- `051_Niu_2013`

## Evidence Channels To Score

For each paper, score these channels independently:

1. Abstract
2. Results text
3. Tables
4. Figure captions
5. Figures
6. Discussion / conclusion

Optional extra channels:

- significance statements
- methods labels for intervention/comparator identity
- table titles / headers

## Unit Of Analysis

We should not start with exact row matching.

Use three nested units:

### A. Paper-level

Question:
Does the paper as a whole show coherent support for the extracted interpretation?

### B. Claim-level

Represent a claim as:

- paper_id
- outcome class
- intervention identity
- comparator identity
- direction
- approximate magnitude band

Magnitude band can initially be:

- strong decrease
- decrease
- no-change
- increase
- strong increase

This is much easier to align than exact numeric rows.

### C. Row-level

Only use later, on the subset with clean matching and/or ground truth.

## Proposed Labels

### Paper-level labels

- `paper_high_confidence`
- `paper_mixed`
- `paper_high_risk`

These can initially be assigned from the existing comparison artifacts and then refined.

### Claim-level labels

- `claim_supported`
- `claim_conflicted`
- `claim_unresolved`

### Row-level labels

- `row_likely_real`
- `row_possible_hallucination`
- `row_known_mismatch`

## Feature Sets

### 1. External convergence features

- `n_models_supporting`
- `n_models_conflicting`
- `direction_agreement_rate`
- `magnitude_range_across_models`
- `outcome_name_entropy`
- `tissue_name_entropy`
- `rerun_stability_available`
- `benchmark_match_available`

### 2. Internal convergence features

- `abstract_direction`
- `results_direction`
- `table_direction`
- `caption_direction`
- `figure_direction`
- `conclusion_direction`
- `n_sections_supporting`
- `n_sections_conflicting`
- `n_sections_missing`
- `table_specificity_score`
- `figure_specificity_score`
- `conclusion_strength_score`

### 3. Structural / provenance features

- `has_named_table`
- `has_named_figure`
- `has_exact_numeric_trace`
- `has_variance`
- `has_sample_size`
- `intervention_clear`
- `comparator_clear`
- `outcome_class_clear`
- `paper_complexity_proxy`

### 4. Known pathology flags

- `swap_risk`
- `element_coverage_mismatch`
- `tissue_granularity_mismatch`
- `figure_only_dependence`
- `derived_metric_risk`
- `factorial_averaging_risk`

## First Statistics To Compute

These should be simple and interpretable.

### Paper-level

- mean internal support count by paper
- mean conflict count by paper
- correlation between support count and model agreement
- correlation between conflict count and disagreement
- rank papers by composite risk score

### Claim-level

- proportion of supported vs conflicted claims
- agreement between internal and external convergence
- confusion matrix:
  internal support high/low vs cross-model support high/low

### Row-level, only on aligned subset

- logistic model:
  `known_mismatch ~ low_model_support + low_section_support + swap_risk + figure_only_dependence + variance_missing`

- optional regularized model if enough rows exist

## Composite Scores

### Internal convergence score

Simple first version:

`ICS = supporting_sections - conflicting_sections`

Weighted version later:

`ICS_w = 1*abstract + 2*results + 3*tables + 2*captions + 2*figures + 1*conclusion - conflicts`

### External convergence score

Simple first version:

`ECS = n_models_supporting - n_models_conflicting`

Weighted version later:

- add rerun stability
- add benchmark support when available
- add consensus confidence

### Hallucination risk score

Simple first version:

`HRS = high if ICS low AND ECS low`

Medium if only one is low.

Low if both are high and no pathology flags are present.

## Concrete Prototype Matrix

### Phase 1 sandbox

6 papers x 3 models x 5 to 6 evidence channels

This is large enough to be informative and small enough to score carefully.

Suggested matrix:

- `020_Overdieck_1993` : clean control
- `031_Pal_2003` : clean control
- `002_Ziska_1997` : label and direction conflict
- `003_Baslam_2012` : swap and granularity conflict
- `004_Finzi_2001` : tissue and swap conflict
- `007_Woodin_1992` : structure / figure stress-test

## What To Extract From Each Paper

For each paper and each model, record:

- extracted claims
- extracted rows or coarse outcomes
- outcome class
- tissue / subgroup level
- direction
- approximate effect band
- whether table / figure / text was cited
- any confidence / warning metadata already present

For the paper itself, record:

- abstract direction for each main claim
- results direction
- table direction
- caption direction
- conclusion direction

## Why This Sandbox Is Good

It contains:

- high-agreement papers
- high-disagreement papers
- known swap-risk papers
- known granularity mismatch papers
- likely figure-sensitive papers

This means it can test whether convergence-based statistics separate easy and hard cases before we scale up.

## Immediate Next Implementation Step

Build a single sandbox table with columns:

- `paper_id`
- `claim_id`
- `model`
- `outcome_class`
- `tissue_class`
- `direction`
- `magnitude_band`
- `source_channel`
- `section_support`
- `section_conflict`
- `cross_model_support_n`
- `swap_risk`
- `granularity_risk`
- `benchmark_match`
- `risk_label_initial`

This table can then be expanded once the first pass works.

## Concrete File Mapping

These are the primary files to use for the first sandbox pass.

### 020_Overdieck_1993

- PDF:
  `meta_analysis_extractor/input/020_Overdieck_1993.pdf`
- backup PDFs:
  `Loladze/validated/020_Overdieck_1993.pdf`
  `Loladze/mineral_validation_input/020_Overdieck_1993.pdf`
  `Loladze/validation_input/020_Overdieck_1993.pdf`
- consensus:
  `meta_analysis_extractor/output/claude_kimi_full_comparison/020_Overdieck_1993_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_comparison_full/020_Overdieck_1993_consensus.json`

### 031_Pal_2003

- PDF:
  `Loladze/validated/031_Pal_2003.pdf`
  `Loladze/validation_input/031_Pal_2003.pdf`
- consensus:
  `meta_analysis_extractor/output/claude_kimi_full_comparison/031_Pal_2003_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_comparison_full/031_Pal_2003_consensus.json`

### 002_Ziska_1997

- PDF:
  `meta_analysis_extractor/input/002_Ziska_1997.pdf`
  `Loladze/validated/002_Ziska_1997.pdf`
- comparison:
  `meta_analysis_extractor/output/002_Ziska_1997_comparison.json`
- consensus:
  `meta_analysis_extractor/output/claude_kimi_full_comparison/002_Ziska_1997_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_comparison_full/002_Ziska_1997_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_test_quick/002_Ziska_1997_consensus.json`
- figure-specific:
  `meta_analysis_extractor/face_wheat_mineral/bar_graph_extractor/output/*Ziska*`

### 003_Baslam_2012

- PDF:
  `meta_analysis_extractor/input/003_Baslam_2012.pdf`
  `Loladze/validated/003_Baslam_2012.pdf`
  `Loladze/mineral_validation_input/003_Baslam_2012.pdf`
  `Loladze/validation_input/003_Baslam_2012.pdf`
- comparison:
  `meta_analysis_extractor/output/003_Baslam_2012_comparison.json`
- consensus:
  `meta_analysis_extractor/output/claude_kimi_full_comparison/003_Baslam_2012_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_comparison_full/003_Baslam_2012_consensus.json`
- raw response:
  `Loladze/mineral_extraction/debug/003_Baslam_2012_raw_response.txt`

### 004_Finzi_2001

- PDF:
  `meta_analysis_extractor/input/004_Finzi_2001.pdf`
  `Loladze/validated/004_Finzi_2001.pdf`
- consensus:
  `meta_analysis_extractor/output/claude_kimi_full_comparison/004_Finzi_2001_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_comparison_full/004_Finzi_2001_consensus.json`

### 007_Woodin_1992

- PDF:
  `meta_analysis_extractor/input/007_Woodin_1992.pdf`
  `Loladze/validated/007_Woodin_1992.pdf`
- consensus:
  `meta_analysis_extractor/output/claude_kimi_full_comparison/007_Woodin_1992_consensus.json`
  `meta_analysis_extractor/output/claude_kimi_comparison_full/007_Woodin_1992_consensus.json`
- figure / vision stress tests:
  `meta_analysis_extractor/output/kimi_figure_test/007_Woodin_1992_*`
  `meta_analysis_extractor/output/kimi_vision_rescue/007_Woodin_1992_*`
  `meta_analysis_extractor/output/vision_test_2026-02-03/007_Woodin_1992_*`
  `meta_analysis_extractor/output/rerun_worst_gemini3/007_Woodin_1992_*`

## Bottom Line

The first prototype should not try to solve full universal matching.

It should test a simpler hypothesis:

Claims and papers with strong within-paper convergence and strong cross-model convergence should be much less likely to be hallucinated or badly mis-extracted.

This sandbox is the right place to test that.
