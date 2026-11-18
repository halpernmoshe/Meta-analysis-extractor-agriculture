# Agreement Between AI-Assisted and Human Data Extraction Across Five Agricultural and Ecological Datasets: Reading Fidelity versus Analytical Choice

This repository holds the code and data needed to reproduce the analyses in the paper above. The
paper compares data extracted by an AI workflow (Claude Code agents) with five published reference
datasets:

| Paper label | Topic | Key-table folder | Record and raw-data folder | AI-side decoder |
|---|---|---|---|---|
| Boldorini et al. 2024 | predator biocontrol | `boldorini` | `boldorini` | `decode_boldorini_march_v11.py` |
| Li X et al. 2024 | biochar | `biochar_v2` | `biochar` | `decode_biochar.py` |
| Hui et al. 2025 | zinc biofortification | `hui_v4` | `hui` | `decode_hui.py` |
| Li J et al. 2022 | biostimulants | `li2022_v2` | `li_j` | `decode_li_j.py` |
| Loladze 2014 | elevated CO2 and mineral nutrition | `loladze_v2` | `loladze` | `decode_loladze.py` |

The folder names are internal identifiers kept from the analysis. Inside the scripts the same
datasets are called `Boldorini`, `Biochar`, `Hui`, `Li2022` and `Loladze`. In file and column names,
`ai` is the AI workflow, and the `gt` folders hold the published-reference tables. `gt` is a
historical file label; it does not mean the reference data were independently verified.

## Quick start

```bash
pip install -r requirements.txt     # matplotlib, openpyxl, pandas
python verify.py                    # checks every stage below; prints a PASS/FAIL table
```

`verify.py` takes under a minute on the tested machine and does not modify the repository. To
regenerate the analysis outputs and figures in place, run `run_all.ps1` (Windows PowerShell) or
`bash run_all.sh`. Every script finds its inputs from its own location, so it can be run from any
working directory.

On Windows, keep the full path of the repository folder under about 150 characters. The longest
file path inside the repository is 104 characters, and Windows rejects paths longer than 260
characters unless long paths are enabled.

## The workflow, by stage

**`0_sources/`: where the inputs come from.** The five published reference datasets, with their
locations, licences and SHA-256 digests (`comparator_datasets.csv`), and the 150 primary papers with
DOIs (`source_papers.csv`). Four of the published data files are included unmodified in `raw/`. The
Hui et al. 2025 workbook is not redistributed; `raw/hui/DOWNLOAD.txt` says where to get it. Source
PDFs are not redistributed (publisher copyright).

**`1_extraction/`: AI extraction.** Claude Code agents (model Claude Opus 4.6) read each source PDF
and wrote one JSON record per paper. The 178 records are the extraction outputs analysed in the
paper; `records/SHA256SUMS.txt` lists the SHA-256 digest and size of each. The extraction protocol is
deposited as a Claude Code skill, together with the written prompt used for the August 2026
re-extraction of the Boldorini corpus. A hosted language model does not return identical output when
re-run, so this stage is documented, not re-executed.

**`2_conversion/`: conversion into canonical key tables.** Each side is converted into a shared set
of categorical key fields, one CSV per paper. `ai_side/` holds the five decoders that build the
AI-side tables from the records. `reference_side/` holds four scripts that build the reference
tables from the published files, and a row trace for each dataset. `keys/<dataset>/ai/` and
`keys/<dataset>/gt/` hold the key tables that every analysis reads.

**`3_evaluation/`: evaluation metrics.** Matching, reading-fidelity statistics, paired and unpaired
equivalence tests, reconciliation analyses, Bland-Altman limits of agreement, and the coverage
analysis. Each script prints its results; `expected_outputs/` holds the expected text.

**`4_figures/`: figures.** Scripts for Figures 1 to 3 and the generated PNG files. Figure S1 is
written here by `3_evaluation/make_bland_altman.py`.

`docs/` holds the decoder ledgers (the decoding decisions, per dataset), the tested software
environment, a file map and the change log.

## What can be re-run and what cannot

- Stage 2 (AI side), stage 3 and stage 4 re-run deterministically from the deposited records and key
  tables.
- Stage 2 (reference side): the reference key tables are regenerated from the published files for
  Loladze 2014 and Li J et al. 2022 (exactly), Hui et al. 2025 (exactly, after downloading the
  workbook; the first step is a November 2026 reconstruction of a lost script) and Boldorini et al.
  2024 (two rows differ in `crop` and `evidence` because of a documented hand correction). The Li X
  et al. 2024 (biochar) reference tables were prepared with a language-model agent and cannot be
  re-run as a script; `2_conversion/reference_side/row_trace/` gives the published row behind each
  of their rows.
- Stage 1 is documented by the protocol and by its output records. Re-running the extraction would
  need the source PDFs and the hosted model, and would not return identical records.
- Stage 0 documents where the inputs come from.

## What `verify.py` checks

In the printed table these checks are labelled 1, 2, 2r, 3 and 4.

1. The 178 extraction records match `SHA256SUMS.txt` (digest and byte size) and parse as JSON.
2. The five AI-side decoders, run into a temporary folder, produce CSV files with the same names as
   the deposited `keys/<dataset>/ai/` tables and the same content after line endings are normalised
   (170 files, 3,151 rows).
3. The reference-side builders, run on the published files into a temporary folder, reproduce the
   deposited `keys/<dataset>/gt/` tables after line-ending normalisation: Loladze and Li J in full;
   Boldorini except the two documented rows, which may differ only in `crop` and `evidence`. The Hui
   check is reported as SKIPPED unless the workbook has been downloaded. A trace check confirms that
   every reference key row of all five datasets appears once in its row trace with a match flag and
   the same means.
4. Every analysis script runs, and its printed output matches its expected-output file when compared
   as text after line-ending normalisation. The coverage tables are regenerated and compared with the
   deposited copies.
5. Every figure is regenerated. Printed figure values are compared with expected text. Each PNG is
   compared with the deposited file: identical file, else identical decoded pixels. On the tested
   environment (`docs/TESTED_ENVIRONMENT.md`) all four PNG files are reproduced identically. Other
   matplotlib versions may render the same numbers slightly differently. A different image counts
   as a failure unless `python verify.py --allow-image-differences` is used; that option reports
   image differences as warnings and relies on the printed-value checks, which cover Figures 2, 3
   and S1 but not Figure 1.

## Manuscript items and the scripts that produce them

| Manuscript item | Script | Output |
|---|---|---|
| Table 3 (reading fidelity on matched cells), including the Loladze no-selection subset quoted in the text | `3_evaluation/line_by_line_scope_aware.py`; subset in `3_evaluation/reconciliation_analysis.py` block (B) | `EXPECTED_OUTPUT_LINEBYLINE.txt`, `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| Figure 1 | `4_figures/make_fig1_fidelity.py` | `4_figures/figures/fig1_fidelity.png` |
| Table 4 (paired equivalence) | `3_evaluation/scope_aware_paired_tost.py` | `EXPECTED_OUTPUT_PAIRED_TOST.txt` |
| Figure 2 | `4_figures/make_fig2_equivalence.py` | `4_figures/figures/fig2_equivalence.png` |
| Biochar native-control comparison reported with Table 4 | `3_evaluation/biochar_native_control_tost.py` | `EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt` |
| Secondary unpaired comparison | `3_evaluation/scope_aware_aggregate_tost.py` | `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` |
| Table 5, biochar reconciliation, Loladze per-element comparison, power decomposition, Li J units example; Supplementary S1, S3 and the biochar and Li J parts of S2 | `3_evaluation/reconciliation_analysis.py` | `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| Figure 3 | `4_figures/make_fig3_reconciliation.py` | `4_figures/figures/fig3_reconciliation.png` |
| Supplementary S5: Table S1 and Figure S1 | `3_evaluation/make_bland_altman.py` | `EXPECTED_OUTPUT_BLAND_ALTMAN.txt`, `4_figures/figures/figS1_bland_altman.png` |
| Supplementary S6, Table S2 (flow and attrition counts) | counts printed by `line_by_line_scope_aware.py`, `scope_aware_aggregate_tost.py` and `scope_aware_paired_tost.py` | as above |
| Coverage results (paper-level burden, figure versus table sources) | `3_evaluation/round2_additional_analysis/coverage_structural_complexity.py` | `EXPECTED_OUTPUT_COVERAGE.txt` and the CSV tables beside the script |
| Supplementary S7 (variance provenance) | not reproduced by a script here; the two published files it uses are in `0_sources/raw/li_j/` and `0_sources/raw/biochar/` | |
| Supplementary S8 (source papers) | | `0_sources/source_papers.csv` |

All expected-output files are in `3_evaluation/expected_outputs/` unless a path is given.

## How effects and equivalence are computed

Effects are log response ratios, ln(treatment mean / control mean). Rows are paired by categorical
key fields only; outcome values are never used to choose a match. Where several rows share a key,
the cell value is their mean.

In the paired test (`scope_aware_paired_tost.py`), each matched cell gives an AI-minus-reference
difference in mean log response ratio. These differences are averaged within each study, and the
per-study values are averaged across studies. The standard error is the sample standard deviation
of the per-study values divided by the square root of the number of studies. The 90% interval is
the mean plus or minus 1.645 standard errors. A dataset passes at a margin of 5, 10, 15 or 20% when
the interval lies inside plus or minus that fraction of the absolute mean reference log response
ratio. Differences and intervals are computed on the log scale and reported as relative
response-ratio differences, 100 × (exp(d) − 1), where d is the log-scale value. They are not
percentage-point differences between the displayed effects.

## Changes from the previous release

Files were reorganised into the stages above and path constants were updated. Added: `verify.py`,
`run_all.ps1`, `run_all.sh`, the extraction protocol, the stage-0 source documentation with four
published data files, and the reference-side conversion scripts and row traces. Removed: six
auxiliary files from the biochar records, which no analysis reads. No retained data file, key table,
record or analysis calculation was changed. Five expected-output files from the previous release were
out of date for the Li J row and have been refreshed from the current output, which matches the
manuscript. Details are in `docs/CHANGELOG.md`.

## Licence

The code and documentation are under the MIT licence; see `LICENSE`. The published data files in
`0_sources/raw/` keep their own terms, stated in the `ATTRIBUTION.txt` beside each file. The source
papers and the published reference datasets remain under their publishers' terms.
