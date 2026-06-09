# Repository Audit Manifest

Audit date: 2026-06-09

Inclusion rule: a file is included only if it directly supports a reported analysis, generated figure, or reproducibility check in the resubmitted manuscript.

## Included

| Item | Reason |
|---|---|
| `scope_matched_equivalence.py` | Generates the primary Table S1/Table S5 numbers and the manuscript's main ~3.2 pp agreement claim |
| `runs/*/keys/{ai,gt}/*.csv` | Cleaned key tables consumed by `scope_matched_equivalence.py` |
| `EXPECTED_OUTPUT.txt` | Expected output for byte-level comparison after running the primary script |
| `make_figures.py` and `figures/{fig1_concordance,figS2_diff_forest,figS3_margin_grid}.png` | Optional figure regeneration from the same key tables |
| `bland_altman_figS6.py` and `figures/figS6_bland_altman.png` | Supporting Bland-Altman figure (Fig S6) + numeric LoA (Table S10) from bundled blind pairings |
| `join_and_score.py` | Deterministic outcome-blind join tool for the supporting line-by-line analysis (§3.3) |
| `line_by_line_results/*/` | Canonical line-by-line outputs (report.json, classification.csv) behind Table S6, with provenance README |
| `runs/{biochar_v3,loladze_v3,hui_v4,li2022_v4}/pairings/*.jsonl` and `runs/li2022_v4/keys/` | Frozen blind pairings (and li2022_v4 keys) consumed by the supporting line-by-line/Bland-Altman analysis |
| `run.ps1`, `run.sh` | Convenience wrappers only |
| `README.md`, `requirements.txt` | Reproduction instructions |

## Excluded

| Excluded source | Reason |
|---|---|
| `legacy/` | False starts and exploratory scripts not used for the resubmitted manuscript |
| Internal audit/status files | Author working notes, not submission or reproduction artifacts |
| Source PDFs | Cannot be redistributed because of copyright |
| Local Dropbox paths and old manifests | Not portable and not needed for reproduction |
| Old combined DOCX/COMBINED files | Working bundles with stale claims; replaced by clean journal files |
