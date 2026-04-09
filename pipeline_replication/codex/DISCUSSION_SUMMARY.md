# Discussion Summary

## Project State

This repository contains an autonomous meta-analysis replication workflow built around topic-specific configs and staged outputs:

- `1_search`
- `2_screen`
- `3_download`
- `4_extract`
- `5_synthesize` / `6_synthesis`

The submitted manuscript in `SUBMISSION_v23/SUBMISSION_CLEAN` is the extraction-validation paper. The `pipeline_replication` folder is the later end-to-end replication project.

## What We Established

1. The project is not back to square one.
2. The pipeline is producing real search, extraction, validation, and synthesis artifacts.
3. Several topics already have usable post-extraction validation and pooled results.
4. The major weaknesses are mostly not row-level hallucinated numbers.
5. The larger failure mode is semantic over-inclusion:
   - wrong outcome
   - wrong intervention contrast
   - wrong study setting
   - wrong estimand relative to the benchmark paper

## Current Topic-Level Read

- `legume_rotation`: strongest success
- `mycorrhiza_yield`: qualified success
- `organic_yield_gap`: primary miss, but likely crop-composition mismatch
- `notill_tillage`: primary miss, but likely management/climate composition mismatch
- `biochar_crop_yield`: primary miss, but likely field-vs-pot / rate / study-design mismatch
- `intercropping_yield`: failure driven mainly by estimand mismatch (`LER` vs component crop yield)

## Key Diagnostic Conclusions

### 1. Post-extraction validation is doing important work

The pipeline improves materially after `pico_validate.py`. This means the extraction stage is broad and the validator is already rescuing the dataset.

### 2. Many remaining bad rows are not fake numbers

They are often valid extracted values for the wrong target:

- quality traits instead of yield
- straw / biological yield instead of grain yield
- reduced tillage instead of strict no-till
- pot/greenhouse biomass studies mixed into field-yield benchmarks
- individual crop yield mixed into system-productivity benchmarks

### 3. Intercropping is a special case

The benchmark is system-level `LER` / land-use efficiency. The extracted corpus is mostly individual component crop yield. That makes the primary synthesis not benchmark-comparable.

### 4. Moderator-aligned diagnostics matter

Benchmark-aligned subgroup checks suggested:

- `organic_yield_gap`: cereal subset gets very close to benchmark
- `notill_tillage`: residue + rotation subset gets much closer to benchmark
- `biochar_crop_yield`: field-only and lower-rate subsets get much closer to benchmark

These help explain the failures, but they remain secondary diagnostics rather than replacements for the preregistered primary analyses.

## Strategic Conclusion

The highest-leverage next step is to tighten post-extraction processing rather than redesign the whole pipeline.

The extractor can stay broad.
The post-extraction layer should become stricter and more benchmark-aware.

## Universal-Pipeline Constraint

The pipeline is supposed to be universal.

That means the final solution should **not** rely on hand-written rules for each topic after looking at the results. Topic-specific diagnostics are still useful for understanding failures, but the production fix should be:

- generic
- config-driven
- based only on the original topic prompt / config and extracted row fields
- portable to the final production model (`Claude Opus 4.6`)

For that reason, the topic-specific strict rules in this folder should be treated as exploratory diagnostics, not the intended final architecture.

## Important Link To The Submitted Paper

The user clarified that LLM-based post-processing was the breakthrough that made the submitted extraction-validation paper work.

That strongly supports the current direction:

- use deterministic rules only for lightweight hard checks and triage
- use an LLM as the main semantic adjudicator in post-processing
- keep the prompt universal and config-driven
- target the final production implementation at `Claude Opus 4.6`

## Constraint For This Work

Per user instruction, all new work in this phase is kept under the `codex` subfolder and the main pipeline files are left unchanged.
