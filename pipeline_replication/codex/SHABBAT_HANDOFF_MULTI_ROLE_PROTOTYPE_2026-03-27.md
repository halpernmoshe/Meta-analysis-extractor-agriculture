# Multi-Role Prototype Handoff

This note summarizes the multi-role full-context synthesis prototype as it stands now.

## 1. Big Picture

The project moved from a single-extractor framing toward a multi-role, full-context synthesis prototype.

The current idea is:

- every role sees the full paper
- each role has a different epistemic job
- the system merges the role outputs
- contradictions and consilience become first-class outputs

Current roles:

- `design_agent`
- `narrative_agent`
- `table_agent`
- `figure_agent`
- `benchmark_agent`
- `consistency_agent`

The point is not just to extract numbers. It is to distinguish:

- clean support
- construct drift
- arm mismatch
- candidate corruption
- tissue mismatch
- figure-only missingness
- wrong temporal point
- ratio-only benchmark target

## 2. Core Files

Conceptual notes:

- [MULTI_ROLE_FULL_CONTEXT_PROTOTYPE_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/MULTI_ROLE_FULL_CONTEXT_PROTOTYPE_2026-03-27.md)
- [LLM_SUITED_SYNTHESIS_PRACTICAL_AND_PHILOSOPHICAL_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/LLM_SUITED_SYNTHESIS_PRACTICAL_AND_PHILOSOPHICAL_2026-03-27.md)
- [MULTI_ROLE_CROSS_PAPER_COMPARISON_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/MULTI_ROLE_CROSS_PAPER_COMPARISON_2026-03-27.md)
- [LIVE_ROLE_RUNNER_BRIDGE_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/LIVE_ROLE_RUNNER_BRIDGE_2026-03-27.md)

Build / orchestration scripts:

- [build_multi_role_pilot_inputs.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/build_multi_role_pilot_inputs.py)
- [seed_multi_role_pilot_from_reports.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/seed_multi_role_pilot_from_reports.py)
- [merge_multi_role_pilot_outputs.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/merge_multi_role_pilot_outputs.py)
- [run_multi_role_paper.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/run_multi_role_paper.py)

Pilot package root:

- [outputs/multi_role_pilot](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot)

Indexes:

- [index.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/index.json)
- [merged_index.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/merged_index.json)

## 3. Important Engineering Changes

### 3.1 Built the pilot package

The pilot package currently covers:

- `011_Huluka_1994`
- `015_Pleijel_2009`
- `019_Baxter_1994`
- `026_Seneweera_1997`
- `035_Oksanen_2005`

Each paper has six role files plus merged outputs.

### 3.2 Built the merger

`merge_multi_role_pilot_outputs.py` now:

- reads the six role files
- merges claims by `claim_key`
- writes:
  - `merged_claims.csv`
  - `merged_summary.json`
  - `merged_summary.md`
- produces `merged_index.json`

### 3.3 Fixed merger and scaffolding bugs

Fixed issues:

- merger no longer ingests unrelated JSONs as if they were role files
- single-paper merges no longer wipe `merged_index.json`
- pilot scaffolding no longer wipes `index.json`
- pilot scaffolding no longer overwrites existing role files when rerun

### 3.4 Upgraded the merger to use explicit contradiction links

The merger no longer relies only on simple set conflicts like:

- direction mismatch
- unit mismatch
- benchmark-comparable mismatch

It now also propagates explicit contradiction relations from role outputs, including:

- `construct_drift`
- `arm_mismatch`
- `candidate_corruption`
- `benchmark_comparability_conflict`
- `unit_conflict`
- `tissue_mismatch`
- `figure_only_missingness`
- `timepoint_mismatch`
- `ratio_only_target`

### 3.5 Added a paper-level runner

[run_multi_role_paper.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/run_multi_role_paper.py) now:

- validates that all six role files exist
- counts claims / constraints / contradictions by role
- can emit ready-to-run role prompt files under each paper's `live_role_prompts/`
- can invoke live Claude Code role runs with `--run-roles`
- records per-role live attempt artifacts under `live_role_attempts/`
- reruns the merger for one paper
- writes `run_status.json`

It now has a real live-execution bridge, but successful live filling is currently blocked by Claude quota exhaustion in this workspace.

## 4. Paper-by-Paper Results

### 4.1 `011_Huluka_1994`

Key files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/011_Huluka_1994/merged_claims.csv)
- [run_status.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/011_Huluka_1994/run_status.json)

This is the wrong-timepoint plus ratio-only-target case.

Main pattern:

- `target::doy177_leaf_table` is the only table-side numeric family
- `target::doy247_leaf_benchmark` is the final-harvest benchmark family
- `target::variance_se_table2` gives strong variance support, but only for DOY 177

Main contradiction pattern:

- `timepoint_mismatch`
- `ratio_only_target`

### 4.2 `015_Pleijel_2009`

Key files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/015_Pleijel_2009/merged_claims.csv)
- [run_status.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/015_Pleijel_2009/run_status.json)

This is the clean-support anchor.

### 4.3 `019_Baxter_1994`

Key files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/019_Baxter_1994/merged_claims.csv)
- [run_status.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/019_Baxter_1994/run_status.json)

This is the construct-drift / modality-mismatch case.

### 4.4 `035_Oksanen_2005`

Key files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/035_Oksanen_2005/merged_claims.csv)
- [run_status.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/035_Oksanen_2005/run_status.json)

This is the arm-mismatch plus candidate-corruption case.

### 4.5 `026_Seneweera_1997`

Key files:

- [merged_claims.csv](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/026_Seneweera_1997/merged_claims.csv)
- [run_status.json](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/multi_role_pilot/026_Seneweera_1997/run_status.json)

This is the tissue-mismatch plus figure-only-missingness case.

## 5. Current Taxonomy

The current pilot taxonomy is now:

- Huluka = wrong temporal point + ratio-only benchmark target
- Pleijel = clean support
- Baxter = construct drift / modality mismatch
- Oksanen = arm mismatch + candidate corruption
- Seneweera = tissue mismatch + figure-only missingness

That is already better than a generic hallucination bucket.

## 6. What Is Solid Now

These points are supported by the current outputs:

- a multi-role, full-context representation is workable in the repo
- explicit contradiction types are more informative than a single generic error label
- different papers fail in qualitatively different ways
- the merger preserves cross-claim contradiction structure
- the prototype separates at least five distinct paper regimes
- the runner can emit prompt bundles and also attempt live per-role execution safely

## 7. What Is Still Weak

These limitations remain:

- role outputs are still hand-upgraded or report-guided, not produced end-to-end directly from PDF by one runner
- some papers still have many `single_role_only` rows
- successful live role execution is still blocked by Claude quota in this workspace
- the prototype is not yet a production pipeline stage

## 8. Most Important Conceptual Takeaway

This supports the broader idea:

- papers are not just numeric row sources
- papers are bundles of evidence channels
- LLM-suited synthesis may be better framed as contradiction-aware, construct-aware consilience synthesis

That means:

- not just narrative review
- not just meta-analysis
- but structured synthesis across:
  - design constraints
  - narrative claims
  - table values
  - figure-only targets
  - benchmark comparability
  - contradiction analysis

## 9. Best Next Step

The next practical step is:

1. generate direct role outputs for one clean paper and one failure-mode paper from the actual PDF
2. compare those direct outputs against the current report-guided pilot outputs
3. decide whether the prototype survives outside the curated pilot stage

## 10. Minimal Restart Path

If restarting quickly, open these first:

- [SHABBAT_HANDOFF_MULTI_ROLE_PROTOTYPE_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/SHABBAT_HANDOFF_MULTI_ROLE_PROTOTYPE_2026-03-27.md)
- [MULTI_ROLE_CROSS_PAPER_COMPARISON_2026-03-27.md](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/MULTI_ROLE_CROSS_PAPER_COMPARISON_2026-03-27.md)
- [run_multi_role_paper.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/run_multi_role_paper.py)
- [merge_multi_role_pilot_outputs.py](C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/merge_multi_role_pilot_outputs.py)
