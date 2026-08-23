# Reproducibility repository

**Agreement Between AI-Assisted Extraction and Published Reference Data Across Five Agricultural and Ecological Datasets**

This repository reproduces every reported mean- and effect-based analysis number and figure in the manuscript from the deposited
key tables. It also deposits the 184 finalized JSON extraction artifacts (178 paper records plus six
biochar auxiliaries) and verifies that the five
AI-side decoders rebuild the frozen AI key tables byte-for-byte. The comparison is outcome-blind:
each side's row metadata is decoded into a canonical
categorical key, pairing is a deterministic categorical operation that never consults outcome
values, multi-row cells are summarised by their mean, and effects are compared as the log response
ratio. Pooled effects are back-transformed to percentage change for presentation, while aggregate
differences, confidence intervals, and margins are reported on the 100 × lnRR scale. For the aggregate biochar comparison, the AI effect is harmonised to
the reference dataset's absolute-control definition using control means the workflow had already
extracted. This common-definition analysis is comparator-informed; the prospective biochar
holdout result is the raw-mean comparison. Li J paper identifiers are crosswalked by author and
year where unique; two same-author/year collisions are excluded after source-title checks. All scripts are
deterministic and require only the Python standard library, except the figure scripts, which also
require `matplotlib`.

## What changed in this revision

Three corrections are reflected throughout.

**The AI-side inputs.** Several comparisons had been run on earlier extraction attempts rather than
on the extraction the Methods describe. Every key table under `runs/*/keys/ai/` is now rebuilt from
that extraction. The reference side, `runs/*/keys/gt/`, is unchanged from the original deposit: only
the AI side moved.

**Boldorini et al. 2024.** That corpus had never been extracted by the evaluated workflow; its
previously reported values were produced by hand-written scripts. It was re-extracted in August 2026
and the comparison rebuilt. The resulting frozen AI-side key tables are deposited in
`runs/boldorini/keys/ai/`; the corrected August 2026 JSON source records are deposited under
`source_records/boldorini/`.

**Li J et al. 2022 crosswalk.** A source-title check found two distinct reference studies that
shared an author and year with an AI paper. The analysis now leaves those ambiguous studies
unmatched rather than merging them. This changes the Li J paired relative response-ratio difference
from +0.28% to +0.26% and does not change any equivalence conclusion.

`decoders/` holds the script that builds each dataset's AI-side key table, one per dataset, with a
ledger recording every decision it makes and its vocabulary comparison against the reference side.
These are the deposited AI-side decoding scripts. The finalized inputs they read are under
`source_records/`; `source_records/SHA256SUMS.txt` authenticates every JSON file. The
published-reference side is supplied as frozen canonical key tables, rather than as rerunnable
raw-label-to-key construction scripts. The final cross-dataset mappings can therefore be inspected
row by row in the ledgers and key tables. `python verify_source_record_release.py` runs all five
decoders in a temporary directory and requires their outputs to be content-identical to the frozen AI
key tables used by every reported analysis.

Two key fields were restored for Boldorini specifically, both documented in its ledger:
`unit_canonical` in the raw-mean comparison, because its reference stores percentages, gram weights
and plant counts under one structural key, and `outcome_canonical` in the paired equivalence test,
because the AI side also records pest abundance and crop damage, which the reference does not carry.

### Boldorini analysis levels

The two Boldorini results intentionally have different comparison units. Strict raw-treatment-mean
fidelity, including the Bland--Altman result, uses paper, outcome, crop, treatment level,
co-amendment, co-amendment level, time point, and unit, yielding 9 matched cells. The paired-effect
TOST uses paper, outcome, crop, and treatment level, yielding 16 matched effect cells by pooling
across the further structural descriptors. Both keys are outcome-blind. The 9-cell raw-mean result
does not establish the source of the 16-cell effect difference.

## How to run

Verify the 184 source records and reproduce the frozen AI keys first:

```bash
python verify_source_record_release.py
```

Then run the reported analyses:

```bash
python line_by_line_scope_aware.py     # Part 1: conditional agreement   -> EXPECTED_OUTPUT_LINEBYLINE.txt
python scope_aware_paired_tost.py      # Part 2: paired equivalence TOST -> EXPECTED_OUTPUT_PAIRED_TOST.txt
python scope_aware_aggregate_tost.py   # Part 2: unpaired aggregate      -> EXPECTED_OUTPUT_AGGREGATE_TOST.txt
python biochar_native_control_tost.py  # Biochar native-control comparison -> EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt
python reconciliation_analysis.py      # Table 5 / Fig 3 / power / Biostimulant  -> EXPECTED_OUTPUT_RECONCILIATION.txt
python make_bland_altman.py            # Supplement S5 LoA + Figure S1   -> EXPECTED_OUTPUT_BLAND_ALTMAN.txt
python make_fig1_fidelity.py           # Figure 1  -> figures/fig1_fidelity.png
python make_fig2_equivalence.py        # Figure 2  -> figures/fig2_equivalence.png
python make_fig3_reconciliation.py     # Figure 3  -> figures/fig3_reconciliation.png
python round2_additional_analysis/coverage_structural_complexity.py
```

Each analysis script prints to stdout; compare against the locked `EXPECTED_OUTPUT_*.txt`. On Windows:

```powershell
python .\line_by_line_scope_aware.py > out.txt ; fc .\EXPECTED_OUTPUT_LINEBYLINE.txt .\out.txt
```

`run.sh` / `run.ps1` run the full set of deposited core analyses and the
coverage/source-format check. Both wrappers stop if any command fails.

## What each script produces

| Script | Manuscript element | Locked output |
|---|---|---|
| `line_by_line_scope_aware.py` | Table 3 (conditional numerical agreement); Figure 1 | `EXPECTED_OUTPUT_LINEBYLINE.txt` |
| `scope_aware_paired_tost.py` | Table 4 (paired equivalence); Figure 2 | `EXPECTED_OUTPUT_PAIRED_TOST.txt` |
| `scope_aware_aggregate_tost.py` | Part 2 unpaired aggregate (power setup) | `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` |
| `biochar_native_control_tost.py` | Biochar native-control comparison reported beside Table 4 | `EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt` |
| `reconciliation_analysis.py` | Table 5; Figure 3; power paragraph; Elevated-CO₂ mineral nutrition per-element; Biostimulant | `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| `make_bland_altman.py` | Supplement S5 (Table S1, Figure S1) | `EXPECTED_OUTPUT_BLAND_ALTMAN.txt` |
| `make_fig1_fidelity.py` | Figure 1 | `figures/fig1_fidelity.png` |
| `make_fig2_equivalence.py` | Figure 2 | `figures/fig2_equivalence.png` |
| `make_fig3_reconciliation.py` | Figure 3 | `figures/fig3_reconciliation.png` |

The deposited coverage/source-format analysis is in `round2_additional_analysis/`. It uses only the
frozen key tables and reproduces the descriptive coverage results reported in the manuscript.

## Datasets and roles

| Directory (`runs/`) | Manuscript label | Role |
|---|---|---|
| `boldorini` | Boldorini et al. 2024 | Validation |
| `biochar_v2` | Li X et al. 2024 | Prospective holdout for raw means; aggregate control harmonisation is post hoc |
| `hui_v4` | Hui et al. 2025 | Validation (clean corpus after excluding 8 mislabelled PDFs) |
| `li2022_v2` | Li J et al. 2022 | Validation |
| `loladze_v2` | Loladze 2014 | Development-adjacent |

Each `runs/<dataset>/keys/{ai,gt}/*.csv` is a locked, single-side-decoded key table: one row per
extracted observation, with structural key fields, `treatment_mean`/`control_mean` (used only to
score already-paired rows), `is_figure`, and an `evidence` audit quote. `corpus_mislabels_D2.csv`
lists the 17 wrong-paper PDFs excluded as a sourcing error.

**Structural key columns are generic slots, not fixed semantic types.** The key fields
`treatment_level`, `co_amendment`, `co_amendment_level`, and `timepoint` are generic structural
slots; the decoder packs each corpus's own design coordinates into them, so the column name does not
describe the contents uniformly across datasets. For the biochar dataset they hold the biochar dose
and co-applied amendment; for Loladze 2014 they hold the **element** (`treatment_level` = ca, fe,
zn, n…), the plant **tissue** (`co_amendment` = grain, leaf…), and the **cultivar**
(`co_amendment_level`); the other datasets pack their corresponding coordinates the same way. This
does not affect the comparison: the deposited AI-side decoder and frozen published-reference key
table use the same canonical slot structure, so categorical pairing is like-for-like
(element-to-element, dose-to-dose) regardless of the generic column label. The labels are slots;
the pairing is on the values within matching slots.

The directory name `gt` is retained as an internal file identifier from the original analysis;
throughout the manuscript it denotes the published reference data and does not imply independently
verified ground truth.

## Notes

- **Source PDFs are not redistributed** because of publisher copyright. The finalized extraction
  records under `source_records/` rebuild the AI-side keys; the frozen key tables in `runs/` remain
  the direct inputs to every analysis script.
- **The source-record release is deliberately bounded.** It excludes superseded extraction attempts,
  consensus experiments, broad validation-result dumps, variance-only raw outputs, complete run
  logs, conversational transcripts, and the contaminated March 2026 Boldorini set. The historical
  public `dev-archive-pre-curation` branch contains older development JSON/prompt material and is
  non-authoritative; it is retained only for provenance and no remote history was rewritten.
- **Public source data for the Supplementary Material S7 variance screens** are listed in
  [`PUBLIC_SOURCE_DATA.md`](PUBLIC_SOURCE_DATA.md), including the two comparator article/data pages
  from which any reader can download the source material.
- `decoders/` holds one AI-side decoder per dataset plus its ledger; these build temporary key tables
  from `source_records/`. `verify_source_record_release.py` compares them field-for-field and row-for-row with
  `runs/*/keys/ai/`.
- `INCLUDED_SOURCE_PAPERS.md` is a candidate-input inventory, not an analytical inclusion ledger;
  the analysed-paper counts and bibliographies are in Supplementary Material S6 and S8.
- The round-1 `superseded_v1/` tree is not carried into this deposit; nothing in the
  current manuscript depends on it, and it remains in the original deposit for provenance.
- `requirements.txt` lists dependencies (standard library for the core analyses; `matplotlib` for
  figures; `numpy` and `pandas` for the optional coverage analysis).
- `TESTED_ENVIRONMENT.md` records the exact software versions used for the final audit run.
