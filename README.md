# Reproducibility repository

**Agreement Between AI-Assisted and Human Data Extraction Across Five Agricultural Meta-Analyses: Reading Fidelity versus Analytical Choice**

This repository reproduces every reported number and figure in the manuscript from the deposited
key tables. The comparison is outcome-blind: each side's row metadata is decoded into a canonical
categorical key, pairing is a deterministic categorical operation that never consults outcome
values, multi-row cells are summarised by their mean, and effects are compared as the log response
ratio (percentage change). Two scope-aware reconciliations are applied as method, not result: the
biochar control is harmonized to the absolute baseline (using control means the workflow already
extracted), and Li J paper identifiers are crosswalked by author and year. All scripts are
deterministic and require only the Python standard library, except the figure scripts, which also
require `matplotlib`.

## How to run

```bash
python line_by_line_scope_aware.py     # Part 1: reading fidelity        -> EXPECTED_OUTPUT_LINEBYLINE.txt
python scope_aware_paired_tost.py      # Part 2: paired equivalence TOST -> EXPECTED_OUTPUT_PAIRED_TOST.txt
python scope_aware_aggregate_tost.py   # Part 2: unpaired aggregate      -> EXPECTED_OUTPUT_AGGREGATE_TOST.txt
python reconciliation_analysis.py      # Table 4 / Fig 3 / power / Li J  -> EXPECTED_OUTPUT_RECONCILIATION.txt
python make_bland_altman.py            # Supplement S5 LoA + Figure S5   -> EXPECTED_OUTPUT_BLAND_ALTMAN.txt
python make_fig1_fidelity.py           # Figure 1  -> figures/fig1_fidelity.png
python make_fig2_equivalence.py        # Figure 2  -> figures/fig2_equivalence.png
python make_fig3_reconciliation.py     # Figure 3  -> figures/fig3_reconciliation.png
```

Each analysis script prints to stdout; compare against the locked `EXPECTED_OUTPUT_*.txt`. On Windows:

```powershell
python .\line_by_line_scope_aware.py > out.txt ; fc .\EXPECTED_OUTPUT_LINEBYLINE.txt .\out.txt
```

`run.sh` / `run.ps1` run the full set.

## What each script produces

| Script | Manuscript element | Locked output |
|---|---|---|
| `line_by_line_scope_aware.py` | Table 2 (reading fidelity); Figure 1 | `EXPECTED_OUTPUT_LINEBYLINE.txt` |
| `scope_aware_paired_tost.py` | Table 3 (paired equivalence); Figure 2 | `EXPECTED_OUTPUT_PAIRED_TOST.txt` |
| `scope_aware_aggregate_tost.py` | Part 2 unpaired aggregate (power setup) | `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` |
| `reconciliation_analysis.py` | Table 4; Figure 3; power paragraph; Loladze per-element; Li J | `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| `make_bland_altman.py` | Supplement S5 (Table S5, Figure S5) | `EXPECTED_OUTPUT_BLAND_ALTMAN.txt` |
| `make_fig1_fidelity.py` | Figure 1 | `figures/fig1_fidelity.png` |
| `make_fig2_equivalence.py` | Figure 2 | `figures/fig2_equivalence.png` |
| `make_fig3_reconciliation.py` | Figure 3 | `figures/fig3_reconciliation.png` |

## Datasets and roles

| Directory (`runs/`) | Manuscript label | Role |
|---|---|---|
| `boldorini` | Boldorini et al. 2024 | Validation |
| `biochar_v2` | Li X et al. 2024 | Prospective holdout |
| `hui_v4` | Hui et al. 2025 | Validation (clean corpus after excluding 8 mislabelled PDFs) |
| `li2022_v2` | Li J et al. 2022 | Validation |
| `loladze_v2` | Loladze 2014 | Development-adjacent |

Each `runs/<dataset>/keys/{ai,gt}/*.csv` is a frozen, single-side-decoded key table: one row per
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
does not affect the comparison: the identical decoder is applied to both the AI side and the GT side
of each dataset, so the categorical key pairs like-for-like (element-to-element, dose-to-dose)
regardless of the generic column label. The labels are slots; the pairing is on the values within
matching slots.

## Notes

- **Source PDFs are not redistributed** because of publisher copyright; the deposited key tables in
  `runs/` are the inputs to every script.
- `superseded_v1/` holds the previous (v1) frozen analysis scripts, figures, and pairing tables,
  retained for provenance only; nothing in the current manuscript depends on it.
- `requirements.txt` lists dependencies (standard library for the analysis scripts; `matplotlib`
  for the figure scripts).
