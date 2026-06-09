# Reproducibility Repository for the Environmental Evidence Resubmission

This curated repository contains only the files needed to reproduce the reported analyses in the revised manuscript:

Agreement Between an AI-Assisted, Scaffolded Extraction Workflow and Published Human-Extracted Reference Datasets: A Scope-Matched Equivalence Study Across Five Agricultural Datasets

Legacy false starts, exploratory multi-model consensus code, working audits, local paths, source PDFs, and status trackers are intentionally excluded.

> The previous full development codebase (the extraction tool and exploratory analyses that formerly occupied this repository) is preserved, unchanged, on the `dev-archive-pre-curation` branch and the `v1.0.0` tag, for anyone who needs it.

## Primary Reproduction

Run:

```bash
python scope_matched_equivalence.py
```

This regenerates the central scope-matched equivalence table and paired TOST margin ladder from the cleaned key tables in `runs/`.

Expected output is stored in `EXPECTED_OUTPUT.txt`. On Windows:

```powershell
python .\scope_matched_equivalence.py > my_output.txt
fc .\EXPECTED_OUTPUT.txt .\my_output.txt
```

## Optional Figures

```bash
python make_figures.py          # fig1_concordance.png (cell-level concordance, §3.5.1),
                                # figS2_diff_forest.png (Fig S2), figS3_margin_grid.png (Fig S3)
python bland_altman_figS6.py    # figS6_bland_altman.png (Fig S6) + the numeric 95% LoA table (Table S10)
```

Both write to `figures/` and require `matplotlib`; the primary numeric reproduction does not. Manuscript Figures S1 (flow schematic) and S5 (variance coverage) are not regenerated here (they need author-supplied schematic/dispersion inputs).

## Supporting Analysis (outcome-blind line-by-line matching, §3.3 / Table S6)

The *supporting* (secondary) line-by-line analysis is deposited as canonical outputs and a generic join tool:

- `join_and_score.py` — the deterministic, outcome-blind equality-join tool (no outcome value is used to choose a pairing). Generic CLI: `python join_and_score.py --ai <ai_keys> --gt <gt_keys> --out <dir>`.
- `line_by_line_results/<dataset>/` — the canonical `report.json` (coverage, on-matched agreement) and `classification.csv` (per reference row: MATCH/AMBIGUOUS/NO_MATCH) behind Table S6; see `line_by_line_results/README.md` for per-dataset provenance.
- `bland_altman_figS6.py` reads the bundled blind pairings (`runs/<ds_version>/pairings/`) to reproduce Figure S6 and the numeric limits of agreement (Table S10).

## Official Dataset Labels

| Internal directory | Manuscript label | Role |
|---|---|---|
| `runs/hui_v4` | Hui et al. 2025 | Validation; clean corpus after eight mislabelled PDFs were excluded |
| `runs/li2022_v2` | Li J et al. 2022 | Validation |
| `runs/biochar_v2` | Li X et al. 2024 | Prospective holdout |
| `runs/boldorini` | Boldorini et al. 2024 | Validation |
| `runs/loladze_v2` | Loladze 2014 | Development-adjacent |

The internal directory names are implementation keys only. The manuscript and generated output use the official labels above. The `runs/biochar_v3`, `runs/loladze_v3`, and `runs/li2022_v4` directories hold only the frozen blind *pairings* (and, for `li2022_v4`, the decoded keys) used by the supporting line-by-line/Bland-Altman analysis; the primary analysis uses the `_v2`/`hui_v4` key tables above.

## Included Files

| Path | Purpose |
|---|---|
| `scope_matched_equivalence.py` | Primary analysis script for scope-matched aggregate agreement and paired TOST margin sensitivity |
| `runs/` | Cleaned AI/reference key tables used by the primary analysis |
| `EXPECTED_OUTPUT.txt` | Locked expected output from `scope_matched_equivalence.py` |
| `make_figures.py` | Optional: regenerates fig1_concordance / figS2 / figS3 from the key tables |
| `bland_altman_figS6.py` | Optional: regenerates Figure S6 (Bland-Altman) and the numeric LoA (Table S10) from bundled blind pairings |
| `join_and_score.py` | Deterministic outcome-blind join tool for the supporting line-by-line analysis (§3.3) |
| `line_by_line_results/` | Canonical line-by-line outputs (report.json + classification.csv) behind Table S6 |
| `figures/` | Generated figures (fig1, figS2, figS3, figS6) |
| `run.ps1`, `run.sh` | Convenience wrappers for the primary analysis |
| `requirements.txt` | Notes core/optional dependencies |

## Exclusions

Source PDFs are not redistributed because of publisher copyright. Legacy exploratory code, stale status files, internal audit prompts, and local Dropbox path records are not included because they are not needed to reproduce the resubmitted manuscript and were a source of reviewer confusion.
