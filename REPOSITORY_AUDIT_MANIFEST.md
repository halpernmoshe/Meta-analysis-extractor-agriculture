# Repository Audit Manifest

Audit date: 2026-06-11

This repository reproduces the single scope-aware analysis reported in the manuscript. Every
analysis script is deterministic and reproduces its locked `EXPECTED_OUTPUT_*.txt` byte-for-byte.
The previous (v1) analysis has been moved, unchanged, into `superseded_v1/` for provenance only.

## Top-level files and folders

| Path | Purpose | Locked output |
|---|---|---|
| `line_by_line_scope_aware.py` | Part 1, reading fidelity (Table 2, Figure 1): bias-free categorical-key pairing, values never used to pair. | `EXPECTED_OUTPUT_LINEBYLINE.txt` |
| `scope_aware_paired_tost.py` | Part 2, paired scope-matched equivalence TOST ladder (Table 3, Figure 2). | `EXPECTED_OUTPUT_PAIRED_TOST.txt` |
| `scope_aware_aggregate_tost.py` | Part 2, unpaired "everything in scope" aggregate equivalence (power setup). | `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` |
| `reconciliation_analysis.py` | Table 4, Figure 3, the power paragraph, the Loladze per-element comparison, and the Li J units/tokens result. | `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| `make_bland_altman.py` | Supplement S5: observation-level limits of agreement (Table S5) and Figure S5. | `EXPECTED_OUTPUT_BLAND_ALTMAN.txt` |
| `make_fig1_fidelity.py` | Figure 1 (reading-fidelity scatter). | `figures/fig1_fidelity.png` |
| `make_fig2_equivalence.py` | Figure 2 (aggregate equivalence forest). | `figures/fig2_equivalence.png` |
| `make_fig3_reconciliation.py` | Figure 3 (reconciliation two-panel). | `figures/fig3_reconciliation.png` |
| `corpus_mislabels_D2.csv` | Corpus-mislabel manifest (17 wrong-paper PDFs excluded): dataset, filename, actual content, handling, evidence. | — |
| `runs/` | Cleaned AI/reference key tables for the five datasets (`boldorini`, `biochar_v2`, `loladze_v2`, `hui_v4`, `li2022_v2`). Inputs to every script. | — |
| `figures/` | Generated manuscript figures (fig1–fig3, figS5_bland_altman). | — |
| `README.md`, `REPOSITORY_AUDIT_MANIFEST.md` | Reproduction instructions and this manifest. | — |
| `requirements.txt` | Dependencies (standard library for analysis; matplotlib for figures). | — |
| `run.ps1`, `run.sh` | Wrappers that run all eight scripts in sequence. | — |
| `LICENSE` | Repository license. | — |
| `superseded_v1/` | Previous (v1) analysis, retained for provenance only. | — |

## Archived (superseded_v1/)

The v1 scope-matched-equivalence scripts (`scope_matched_equivalence.py`, `join_and_score.py`,
`make_figures.py`, `bland_altman_figS5.py`, `make_figS1_flow.py`), their outputs
(`line_by_line_results/`, `EXPECTED_OUTPUT.txt`), the v1 manifest, the orphaned
`source_type_agreement.py` (its Table 7 is not part of the current manuscript), the v1 figures
(`figures/figS1_flow.png`, `figS2_diff_forest.png`, `figS3_margin_grid.png`, `figS4_scatter.png`),
and the v1 frozen pairing tables (`runs_extra/biochar_v3`, `loladze_v3`, `li2022_v4`) are retained
here unchanged. Nothing in the current manuscript depends on them.

## Notes

- Source PDFs are not redistributed because of publisher copyright; the `runs/` key tables are the inputs.
- Every analysis script reproduces its `EXPECTED_OUTPUT_*.txt` byte-for-byte; the figure scripts run without error and print values matching their captions.
