# Change log

## Restructured release (November 2026)

This release reorganises the previous release (commit `f0f6e30`) into five workflow stages, adds
source documentation and the reference-side conversion, and removes six files that no analysis
reads. No retained data file, key table, extraction record or analysis calculation was changed.
Every analysis script prints the same text as before, apart from file paths (listed below), and the
generated figures and coverage tables are identical to those of the previous release on the tested
environment.

### Files moved

| Previous location | New location |
|---|---|
| `source_records/<dataset>/` | `1_extraction/records/<dataset>/` |
| `source_records/SHA256SUMS.txt` | `1_extraction/records/SHA256SUMS.txt` (regenerated; see below) |
| `runs/<dataset>/keys/ai/`, `runs/<dataset>/keys/gt/` | `2_conversion/keys/<dataset>/ai/`, `2_conversion/keys/<dataset>/gt/` |
| `decoders/decode_*.py`, `verify_source_record_release.py` | `2_conversion/ai_side/` |
| `decoders/*_DECODER_LEDGER.md` | `docs/decoder_ledgers/` (except `boldorini_DECODER_LEDGER.md`, removed; see below) |
| `line_by_line_scope_aware.py`, `scope_aware_paired_tost.py`, `scope_aware_aggregate_tost.py`, `biochar_native_control_tost.py`, `reconciliation_analysis.py`, `make_bland_altman.py` | `3_evaluation/` |
| `EXPECTED_OUTPUT_*.txt` | `3_evaluation/expected_outputs/` |
| `round2_additional_analysis/` | `3_evaluation/round2_additional_analysis/` |
| `make_fig1_fidelity.py`, `make_fig2_equivalence.py`, `make_fig3_reconciliation.py` | `4_figures/` |
| `figures/` | `4_figures/figures/` |
| `corpus_mislabels_D2.csv` | `0_sources/` |
| `TESTED_ENVIRONMENT.md` | `docs/` |
| `REPOSITORY_AUDIT_MANIFEST.md` | replaced by `docs/FILE_MAP.md` (file map for the new layout) |
| `INCLUDED_SOURCE_PAPERS.md`, `PUBLIC_SOURCE_DATA.md` | replaced by `0_sources/source_papers.csv`, `0_sources/comparator_datasets.csv` and `0_sources/README.md` |
| `run.ps1`, `run.sh` | replaced by `run_all.ps1`, `run_all.sh` |

Script file names are unchanged.

### Files removed

Six files in the biochar record folder were removed: `alignment.json`, `llm_alignment_v2.json`,
`llm_alignment_v3.json`, `validation_moderator_matched.json`, `variance_imputation_report.json` and
`variance_recovered.json`. They are not extraction records: none has both a top-level `paper_id` and
an `observations` list, so the biochar decoder never used them, and no script in the repository
opens them. They are not inputs to any analysis, and the editor asked for only the files needed to
understand and reproduce the analyses. They remain in the history of the previous release.

`decoders/boldorini_DECODER_LEDGER.md` was removed: it describes a superseded record set that is not
deposited, and a decoder (`decode_boldorini.py`) that is not in the repository. The ledger for the
deposited August 2026 Boldorini records, `boldorini_march_style_DECODER_LEDGER.md`, is kept. Each
remaining ledger now starts with one line saying what it is.

### Record manifest

`1_extraction/records/SHA256SUMS.txt` was regenerated with paths relative to the repository root
(`1_extraction/records/...`) for the 178 retained records. The SHA-256 digest and byte size of every
one of the 178 records are unchanged from the previous manifest; the only entries no longer listed
are the six removed files.

### Files added

- `verify.py`: one command that checks the records, the AI-side decoders, the reference-side
  conversion, every analysis output and every figure.
- `0_sources/`: `comparator_datasets.csv`, `source_papers.csv`, and four published data files in
  `raw/` (Loladze 2014, Li J et al. 2022, Li X et al. 2024, Boldorini et al. 2024), each with an
  `ATTRIBUTION.txt`; `raw/hui/DOWNLOAD.txt` for the Hui et al. 2025 workbook, which is not
  redistributed.
- `1_extraction/protocol/extract-meta-analysis/`: the Claude Code extraction skill, unedited.
- `1_extraction/protocol/boldorini_2026-08/EXTRACTION_PROMPT.md`: the prompt used for the August 2026 re-extraction of the Boldorini corpus, with two local folder paths replaced by placeholders.
- `2_conversion/reference_side/`: four reference-side builders and `row_trace/` (one trace per
  dataset).
- `3_evaluation/expected_outputs/EXPECTED_OUTPUT_COVERAGE.txt`: printed output of the coverage script.
- `4_figures/expected_outputs/EXPECTED_OUTPUT_FIG2_EQUIVALENCE.txt` and
  `EXPECTED_OUTPUT_FIG3_RECONCILIATION.txt`: values printed by the two figure scripts.
- A README for each stage folder; this change log.

### Changes inside scripts

Only path constants, printed path strings and comments changed:

- Analysis and figure scripts locate the repository root from their own file location and read key
  tables from `2_conversion/keys/<dataset>/{ai,gt}/`. The dataset path strings lost their `/keys`
  segment (for example `"biochar_v2/keys"` became `"biochar_v2"`).
- Figure scripts write to `4_figures/figures/`. `make_bland_altman.py` now prints
  `Wrote 4_figures/figures/figS1_bland_altman.png` instead of `Wrote figures/figS1_bland_altman.png`.
- Decoders read records from `1_extraction/records/<dataset>/`; `decode_li_j.py` and
  `decode_boldorini_march_v11.py` read reference keys from `2_conversion/keys/<dataset>/gt/`.
- In the decoders' comments and module docstrings, internal folder paths were removed or rephrased;
  the code is unchanged (checked by comparing the parsed code without docstrings before and after).
  The decoder ledgers in `docs/decoder_ledgers/` received the same edits.
- The docstrings of `decode_boldorini_march_v11.py` now state exactly what the decoder reads from the
  reference key tables: the `paper_id` and `treatment_level` columns, used to choose the form in which
  `treatment_level` is written per paper; the reference means are not used. The previous text said
  the reference tables were never read and called them values-stripped.
- The docstring of `make_bland_altman.py` no longer refers to a reviewer.
- `verify_source_record_release.py` uses the new folders and expects 178 records.
- `coverage_structural_complexity.py` locates the repository root two levels up.

Printed text that differs from the previous release: the `wrote <absolute path>` line of the three
`make_fig*.py` scripts; the `SRC=` and `OUT=` lines of `decode_loladze.py`; the `source_dir` field of
the Boldorini decoder's statistics file; the `make_bland_altman.py` line given above; and the count
line of `decode_biochar.py` (and its audit file), which now reports 28 JSON files and no non-record
files instead of 34 and 6; and the first line of `verify_source_record_release.py`, which now reports
178 records and 2,494,170 bytes instead of 184 and 2,825,558. The generated key tables are unchanged.

`.gitignore` and `.gitattributes` were updated for the new folders. `requirements.txt` lists
matplotlib (figures), openpyxl and pandas (reference-side builders).

### Expected-output files that were out of date

In the previous release, five `EXPECTED_OUTPUT_*.txt` files still showed Li J et al. 2022 values from
before the final Li J crosswalk correction (commit `f0f6e30`, "Fix Li J crosswalk provenance
collisions"). That correction leaves two same-author, same-year reference studies (`gt_study08`,
`gt_study146`) unmatched instead of merging them with an AI paper. The scripts and the manuscript
already used the corrected values; only the expected files had not been regenerated. They are now
regenerated from the current script output. Old and new Li J lines:

| File | Previous expected value | Current output |
|---|---|---|
| `EXPECTED_OUTPUT_PAIRED_TOST.txt` | `+0.28 [-1.42,+2.00]` | `+0.26 [-1.44,+1.99]` |
| `EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt` (Li J row, same as the paired test) | `+0.28 [-1.42,+2.00]` | `+0.26 [-1.44,+1.99]` |
| `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` | `+0.28 [-5.53,+6.43]` | `+0.26 [-5.54,+6.42]` |
| `EXPECTED_OUTPUT_LINEBYLINE.txt`, Table 3 row | MAE `4.83pp` | MAE `4.82pp` |
| `EXPECTED_OUTPUT_LINEBYLINE.txt`, pooled-effect line | `diff=+0.68pp` | `diff=+0.67pp` |
| `EXPECTED_OUTPUT_BLAND_ALTMAN.txt` | bias `+0.74`, LoA `-15.77` to `+17.24`, median `+0.60` | bias `+0.72`, LoA `-15.79` to `+17.23`, median `+0.00` |

The Li J equivalence results (pass at 15% and 20%) and every other dataset's values are unchanged.
`EXPECTED_OUTPUT_RECONCILIATION.txt` already matched the script output.

The previous README said the crosswalk correction changed only the paired difference, from +0.28% to
+0.26%. It also changed the Li J values in the other four files listed above.

### Other branches

The public repository also has a historical branch, `dev-archive-pre-curation`, with older
development material (earlier JSON outputs and prompts). It is not an input to this release and
should not be used to reproduce the reported analyses. Its history was not rewritten.
