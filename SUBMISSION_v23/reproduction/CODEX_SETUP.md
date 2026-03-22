# Codex Reproduction Setup

Reproduce all validation statistics, publication figures, and the Word document
from pre-extracted data. No API keys or PDFs are needed.

## 1. Required Files to Upload

Upload the entire repository, or at minimum these files:

### Scripts (all in repo root or FINAL_SUBMISSION_2026-03-08/)

```
reproduce_everything.py
reproduce_all.py
formal_stats_all_datasets.py
FINAL_SUBMISSION_2026-03-08/generate_figures.py
FINAL_SUBMISSION_2026-03-08/build_docx.py
FINAL_SUBMISSION_2026-03-08/PAPER_FINAL_v19.md
requirements_reproduce.txt
```

### Data Files (critical)

```
output/loladze_agent_replication/validation_llm_10pp.json
output/hui2023_full_35/validation_matches_improved.csv
output/hui2023_full_35/validation_matches.csv
output/li2022_combined/validation_matches_effect_first.csv
output/biochar_extraction/validation_results.json
output/boldorini_extraction/validation_results.json
output/boldorini_extraction/B01_Ali_2018.json
output/boldorini_extraction/B02_Bisseleua_2017.json
output/boldorini_extraction/B03_Borkhataria_2012.json
output/boldorini_extraction/B04_Classen_2014.json
output/boldorini_extraction/B05_Garfinkel_2015.json
output/boldorini_extraction/B06_Garfinkel_2020.json
output/boldorini_extraction/B07_Gras_2016.json
output/boldorini_extraction/B08_Hooks_2003.json
output/boldorini_extraction/B09_Ismoilov_2020.json
output/boldorini_extraction/B10_Lang_2003.json
output/boldorini_extraction/B11_Libran-Embid_2017.json
output/boldorini_extraction/B13_Maas_2013.json
output/boldorini_extraction/B14_Martin_2013.json
output/boldorini_extraction/B15_Mols_2002.json
output/boldorini_extraction/B16_Saunders_2016.json
output/boldorini_extraction/B17_Snyder_2001.json
output/boldorini_extraction/B18_Suenaga_2015.json
output/boldorini_extraction/B19_Vichitbandha_2002.json
```

### Boldorini Ground Truth

Already included in the repo at `data/boldorini_gt.csv`. No external files needed.

## 2. Install Dependencies

```bash
pip install -r requirements_reproduce.txt
```

Or manually:

```bash
pip install numpy scipy pandas matplotlib seaborn python-docx openpyxl
```

## 3. Run Everything

```bash
python reproduce_everything.py
```

This single command runs all four steps in sequence and reports
success/failure for each.

## 4. Expected Output

### Files Created

| File | Description |
|------|-------------|
| `output/reproduction_results.json` | Machine-readable stats verification |
| `output/formal_stats_all_datasets.json` | Formal statistical analysis (all 5 datasets) |
| `FINAL_SUBMISSION_2026-03-08/figures/figure1_architecture.png` | System architecture diagram |
| `FINAL_SUBMISSION_2026-03-08/figures/figure2_scatter_plots.png` | Scatter plots (5 panels) |
| `FINAL_SUBMISSION_2026-03-08/figures/figure3_cross_dataset_comparison.png` | Cross-dataset bar chart |
| `FINAL_SUBMISSION_2026-03-08/figures/figure4_bland_altman.png` | Bland-Altman plots (5 panels) |
| `FINAL_SUBMISSION_2026-03-08/figures/figure5_source_type_accuracy.png` | Source type comparison |
| `FINAL_SUBMISSION_2026-03-08/figures/figure6_aggregate_effects.png` | Aggregate effect sizes |
| `FINAL_SUBMISSION_2026-03-08/PAPER_FINAL_v19.docx` | Word document with embedded figures |

### Console Output

The script prints detailed statistics for each dataset and a cross-dataset
comparison table.  Key numbers to verify:

| Dataset | Obs | r | MAE (pp) |
|---------|-----|---|----------|
| Loladze 2014 | 413 | 0.984 | 1.36 |
| Hui 2023 | 319 | 0.999 | 0.43 |
| Li 2022 | 117 | 0.994 | 1.01 |
| Biochar 2024 | 254 | 0.997 | 1.20 |
| Boldorini 2024 | 46 | 0.972* | 3.06 |

*Boldorini r is on the lnRR scale.

### Runtime

Under 30 seconds on a modern machine (pure computation, no network calls).

## 5. Running Individual Steps

```bash
# Stats verification only
python reproduce_all.py

# Formal stats only
python formal_stats_all_datasets.py

# Figures only
python FINAL_SUBMISSION_2026-03-08/generate_figures.py

# Word document only
python FINAL_SUBMISSION_2026-03-08/build_docx.py
```

## 6. Troubleshooting

- **UnicodeEncodeError**: The scripts already include `sys.stdout.reconfigure(encoding='utf-8')` for Windows. On Linux/Codex this is harmless.
- **Boldorini skipped**: If the external GT CSV is not uploaded, `reproduce_all.py` will skip Boldorini (the other scripts still include it from the validation_results.json).
- **matplotlib backend**: `generate_figures.py` uses `matplotlib.use('Agg')` so no display is needed.
- **Missing python-docx**: `build_docx.py` requires `python-docx`. Install with `pip install python-docx`.
