# Multi-Role Cross-Paper Comparison (2026-03-27)

This note summarizes the differentiated pilot cases in the full-context multi-role prototype.

## Papers

- clean support: `015_Pleijel_2009`
- construct drift / modality mismatch: `019_Baxter_1994`
- arm mismatch + candidate corruption: `035_Oksanen_2005`
- tissue mismatch + figure-only missingness: `026_Seneweera_1997`
- wrong temporal point + ratio-only benchmark target: `011_Huluka_1994`

## 1. Clean support: Pleijel 2009

Files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/015_Pleijel_2009/merged_claims.csv)

Pattern:

- `target::grain_zn_pure_co2` has support from 4 roles:
  - `benchmark_agent`
  - `design_agent`
  - `narrative_agent`
  - `table_agent`
- no contradiction flags on the main benchmark claim
- nearby extractions exist (`target::grain_zn_factorial_extensions`) but they are scope extensions, not contradictions

Interpretation:

- this is the prototype clean-support anchor
- one element, one tissue, one benchmark construct
- extra structure exists, but it does not undermine the benchmark row family

## 2. Construct drift / modality mismatch: Baxter 1994

Files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/019_Baxter_1994/merged_claims.csv)
- [LIVE_MULTI_ROLE_PILOT_019_BAXTER_1994_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/LIVE_MULTI_ROLE_PILOT_019_BAXTER_1994_2026-03-27.md)

Pattern:

- `target::foliar_concentration` is supported by figure-role evidence
- `target::total_content` is supported by table-role evidence
- explicit contradiction set:
  - `construct_drift`
  - `benchmark_comparability_conflict`
  - `unit_conflict`

Interpretation:

- the benchmark construct exists in the paper
- the explicit numeric table supports a different construct
- this is not a generic bad extraction; it is a paper where different evidence channels support different constructs

## 3. Arm mismatch + candidate corruption: Oksanen 2005

Files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/035_Oksanen_2005/merged_claims.csv)

Pattern:

- `target::ec_only_leaf` is the benchmark target
- `target::ec_eo_leaf` is a nearby factorial arm
- `target::vision_corrupted_leaf` is a contaminated candidate family
- explicit contradiction set:
  - `arm_mismatch`
  - `candidate_corruption`

Interpretation:

- unlike Baxter, this is not primarily construct drift
- the overall construct family is correct
- the errors arise from mixing:
  - the wrong biological arm
  - corrupted candidate rows from the vision pass

## 4. Tissue mismatch + figure-only missingness: Seneweera 1997

Files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/026_Seneweera_1997/merged_claims.csv)

Pattern:

- `target::blade_table_average` is a valid blade-side benchmark family
- `target::grain_np_table` is a valid grain-table benchmark family
- `target::grain_figures` is a separate figure-only grain benchmark family
- `target::leaf_substitute_for_grain` is the wrong-tissue fallback artifact
- explicit contradiction set:
  - `tissue_mismatch`
  - `figure_only_missingness`

Interpretation:

- this paper is partly successful on tables and explicitly incomplete on figures
- the main problem is not generic extraction error
- the main problem is letting missing figure-side grain targets collapse into wrong-tissue table substitutes

## 5. Wrong temporal point + ratio-only benchmark target: Huluka 1994

Files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/011_Huluka_1994/merged_claims.csv)

Pattern:

- `target::doy177_leaf_table` is the only numerically explicit table family
- `target::doy247_leaf_benchmark` is the final-harvest benchmark family
- `target::variance_se_table2` is strong variance support, but only for the wrong date
- explicit contradiction set:
  - `timepoint_mismatch`
  - `ratio_only_target`

Interpretation:

- unlike Baxter, the main construct is not wrong
- unlike Oksanen, the main arm is not wrong
- unlike Seneweera, the missing target is not a different tissue but a later timepoint
- the key problem is that the benchmark target exists only as figure ratios at DOY 247, while the easy numeric table is DOY 177

## What the prototype is already good at

- distinguishing clean support from contradiction-bearing papers
- separating different mismatch modes instead of collapsing everything into "hallucination"
- representing papers as bundles of evidence channels rather than single extracted row sets
- showing that contradictions can be:
  - cross-construct
  - cross-arm
  - cross-modality
  - candidate-quality related
  - tissue/timepoint related
  - figure-only missingness related
  - ratio-only target related

## What it still misses

- stronger automatic role synthesis for papers where many claims remain `single_role_only`
- better cross-claim linking for scope-extension claims that are not true contradictions
- paper-native section parsing rather than report-driven role filling
- richer support from narrative and consistency roles across all pilot papers
- live role execution exists in `run_multi_role_paper.py`, but successful live role filling is still blocked by Claude quota in this workspace

## Best next extension

Move from report-seeded / report-guided live role filling to one paper-level runner that produces all six role outputs from the actual PDF in one pass, then merges them immediately.

The runner bridge improved in this round:

- [run_multi_role_paper.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/run_multi_role_paper.py) can emit ready-to-run per-role prompt files
- it can also attempt live per-role execution safely
- quota failures are logged without corrupting role files

The next concrete target should be:

1. generate direct role outputs for one clean paper and one failure-mode paper from the actual PDF
2. compare those direct outputs against the current report-guided pilot outputs
3. decide whether the prototype survives outside the curated pilot stage
