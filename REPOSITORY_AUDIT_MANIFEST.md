# Repository Audit Manifest

Audit date: 2026-08-20

This repository reproduces the reported mean- and effect-based scope-aware analyses. Every
analysis script is deterministic and reproduces its locked `EXPECTED_OUTPUT_*.txt` byte-for-byte.
This deposit reflects two corrections: the AI-side key tables are rebuilt from the extraction the
Methods describes, replacing earlier extraction attempts that had been used by mistake, and the
Boldorini et al. 2024 corpus was re-extracted in August 2026 and its comparison rebuilt. The
reference side is unchanged from the previous deposit. The round-1 `superseded_v1/` tree is not
carried here; it remains in the earlier deposit for provenance.

## Top-level files and folders

| Path | Purpose | Locked output |
|---|---|---|
| `line_by_line_scope_aware.py` | Part 1, conditional numerical agreement (Table 3, Figure 1): outcome-blind categorical-key pairing, values never used to pair. | `EXPECTED_OUTPUT_LINEBYLINE.txt` |
| `scope_aware_paired_tost.py` | Part 2, paired scope-matched equivalence TOST ladder (Table 4, Figure 2). | `EXPECTED_OUTPUT_PAIRED_TOST.txt` |
| `scope_aware_aggregate_tost.py` | Part 2, unpaired "everything in scope" aggregate equivalence (power setup). | `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` |
| `biochar_native_control_tost.py` | Biochar native-control comparison reported beside Table 4. | `EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt` |
| `reconciliation_analysis.py` | Table 5, Figure 3, the power paragraph, the Loladze per-element comparison, and the Li J units/tokens result. | `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| `make_bland_altman.py` | Supplement S5: observation-level limits of agreement (Table S1) and Figure S1. | `EXPECTED_OUTPUT_BLAND_ALTMAN.txt` |
| `make_fig1_fidelity.py` | Figure 1 (conditional numerical agreement scatter; legacy filename retained). | `figures/fig1_fidelity.png` |
| `make_fig2_equivalence.py` | Figure 2 (aggregate equivalence forest). | `figures/fig2_equivalence.png` |
| `make_fig3_reconciliation.py` | Figure 3 (reconciliation two-panel). | `figures/fig3_reconciliation.png` |
| `corpus_mislabels_D2.csv` | Corpus-mislabel manifest (17 wrong-paper PDFs excluded): dataset, filename, actual content, handling, evidence. | — |
| `runs/` | Cleaned AI/reference key tables for the five datasets (`boldorini`, `biochar_v2`, `loladze_v2`, `hui_v4`, `li2022_v2`). Inputs to every script. | — |
| `figures/` | Generated manuscript figures (fig1–fig3, figS1_bland_altman). | — |
| `README.md`, `REPOSITORY_AUDIT_MANIFEST.md` | Reproduction instructions and this manifest. | — |
| `requirements.txt` | Dependencies (standard library for analysis; matplotlib for figures). | — |
| `run.ps1`, `run.sh` | Wrappers that run all deposited core scripts in sequence. | — |
| `LICENSE` | Repository license. | — |
| `round2_additional_analysis/` | Coverage and source-format analysis added in round 2. | Generated CSV tables. |

## Notes

- Source PDFs are not redistributed because of publisher copyright; the `runs/` key tables are the inputs.
- The core analysis scripts reproduce their `EXPECTED_OUTPUT_*.txt` byte-for-byte; the figure scripts run without error and print values matching their captions.
- The coverage/source-format analysis uses only the deposited key tables and reproduces its deposited CSV summaries.
- The Supplementary Material S7 variance-provenance counts rely on source comparator workbooks
  whose row-level provenance inputs are not deposited; they are documented in the manuscript but
  are not rerun by this release.
- Source PDFs, raw model outputs, and prompts are not redistributed because they can contain copyrighted source text or account-provenance material.
