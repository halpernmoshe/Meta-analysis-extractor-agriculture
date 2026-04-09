## Latest figure extractor note

### Bottom line

The strongest candidate for your latest serious figure extractor is the `figure_benchmark` stack under:

`C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark`

This is the path that best matches your description:

- LLM or vision model used for semantic labeling / calibration
- programmatic CV measurement used for the actual numeric readout
- benchmarked across multiple datasets

It is more advanced and more relevant to the current multi-role prototype than the older figure modules in `meta_analysis_extractor\modules`.

### What I found

There appear to be three distinct figure-extraction lines in the repo:

1. Older generic module:
`C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\modules\figure_extract.py`

- Generic vision-based figure extraction module
- Designed as a fallback when tables are incomplete
- Focuses on deciding whether figure extraction is needed and then using vision extraction
- Does not appear to be the latest hybrid measurement stack

2. Kimi direct-vision module:
`C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\modules\kimi_figure_extract.py`

- Uses Kimi vision directly for multimodal figure reading
- Strong for direct API-based figure extraction
- Still primarily a model-reading-the-figure approach
- Not the best match to your description of "LLM labels, then programmatic calculates"

3. Newer hybrid benchmarked stack:
`C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark`

Important files:

- [HANDOFF_PIPELINE_INTEGRATION.md](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\HANDOFF_PIPELINE_INTEGRATION.md)
- [RESULTS_SUMMARY.md](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\RESULTS_SUMMARY.md)
- [SESSION_SUMMARY_20260322.md](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\SESSION_SUMMARY_20260322.md)
- [cv_hybrid_extractor.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\cv_hybrid_extractor.py)
- [cv_precise_measure.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\cv_precise_measure.py)
- [cv_panel_extractor.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\cv_panel_extractor.py)
- [cv_webplotdigitizer.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\cv_webplotdigitizer.py)
- [auto_calibrate.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\auto_calibrate.py)
- [multi_dataset_benchmark.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\multi_dataset_benchmark.py)

### Why `figure_benchmark` looks like the latest serious extractor

The strongest evidence is in:

- [HANDOFF_PIPELINE_INTEGRATION.md](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\HANDOFF_PIPELINE_INTEGRATION.md)
- [cv_hybrid_extractor.py](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\cv_hybrid_extractor.py)

The handoff doc explicitly describes the intended pipeline as:

1. render the PDF page
2. use Claude vision to identify panels, axes, groups, control/treatment encoding, and error-bar meaning
3. refine positions with CV
4. use `cv_precise_measure.py` for actual numeric measurement
5. fall back to LLM vision only on edge cases

That is exactly the "semantic labeling by LLM, numeric calculation by programmatic code" architecture.

The `cv_hybrid_extractor.py` file says the same thing in code:

- OpenCV pipeline measures exact bars
- a cheap LLM call performs semantic interpretation
- the prompt explicitly tells the LLM to use exact numeric values provided by CV rather than eyeballing

So this is not just another vision prompt. It is a hybrid measurement system.

### Benchmarking evidence

The benchmark notes make this look like a real development branch, not a toy:

- [SESSION_SUMMARY_20260322.md](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\SESSION_SUMMARY_20260322.md)
- [RESULTS_SUMMARY.md](C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\figure_benchmark\RESULTS_SUMMARY.md)

What stands out:

- benchmark expanded beyond one dataset
- multi-dataset support across Loladze, Biochar, Boldorini, Hui, Li 2022
- explicit panel-mode extraction
- text expectations from the paper used to constrain figure reading
- text-based swap detection
- multiple strategy comparisons
- warnings about circularity in GT-based swap correction

This is exactly the kind of evidence that suggests this was your current serious figure-reading effort.

### Best current interpretation

If the question is "what should count as the latest figure extractor in the repo?", the answer is:

`figure_benchmark`

If the question is "what specific components matter most?", the answer is:

- semantic side:
  - `auto_calibrate.py`
  - `cv_hybrid_extractor.py`
  - the prompting logic described in `HANDOFF_PIPELINE_INTEGRATION.md`
- numeric side:
  - `cv_precise_measure.py`
  - `cv_panel_extractor.py`
  - `cv_webplotdigitizer.py`
- evaluation side:
  - `multi_dataset_benchmark.py`
  - `RESULTS_SUMMARY.md`
  - `SESSION_SUMMARY_20260322.md`

### Relevance to the current multi-role prototype

The current multi-role synthesis prototype under:

`C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\pipeline_replication\codex`

is not yet using this figure stack. The current `figure_agent` outputs are still mostly report-guided or manually upgraded.

That means the prototype is currently missing your best figure-reading backend.

This matters most for papers already in the pilot set with figure-sensitive failure modes:

- `019_Baxter_1994`
  - construct drift / modality mismatch
- `026_Seneweera_1997`
  - tissue mismatch + figure-only missingness
- `011_Huluka_1994`
  - timepoint mismatch + ratio-only target

### Recommended integration plan

Use `figure_benchmark` as the real backend for `figure_agent`.

Practical integration shape:

1. `figure_agent` should still read the full paper and output:
- target figure number
- panel
- target outcome
- tissue
- arm/comparator
- timepoint
- expected unit
- whether the target is figure-only

2. That structured target should then call the hybrid figure extractor:
- identify the correct page and panel
- calibrate semantically
- measure programmatically

3. The merged prototype should compare figure-derived claims against:
- `design_agent`
- `table_agent`
- `benchmark_agent`
- `consistency_agent`

4. Contradiction types should explicitly include:
- `figure_only_target`
- `timepoint_mismatch`
- `tissue_mismatch`
- `unit_conflict`
- `construct_drift`
- `benchmark_comparability_conflict`

### Practical implication

If we want the multi-role prototype to become a real workflow instead of a curated demo, the next high-value integration is not another generic vision call. It is plugging the `figure_benchmark` hybrid extractor into the `figure_agent` role.

That is the most likely path that uses your strongest existing figure work rather than rebuilding a weaker version inside the prototype.
