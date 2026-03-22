# Submission Package -- Meta-Analysis Extractor Validation

This package accompanies the paper:

**"Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets"**

## Contents

| Path | Description |
|------|-------------|
| `PAPER_FINAL_v23.md` | Paper manuscript (Markdown source) |
| `halpern_2026.docx` | Paper manuscript (Word format, original build) |
| `halpern_2026_v2.docx` | Paper manuscript (Word format, revised build) |
| `halpern_2026_v3.docx` | Paper manuscript (Word format, current build) |
| `halpern_2026_supplementary.docx` | Supplementary materials (Word format) |
| `COVER_LETTER.md` | Cover letter |
| `CODE_AVAILABILITY.md` | Code and data availability statement |
| `figures/` | All main publication figures (PNG, 300 DPI) |
| `supplementary/` | Supplementary tables (CSV), figures (PNG), and materials (Markdown) |
| `reproduction/` | Self-contained reproduction package |

## Reproduction

All statistics, figures, and the DOCX can be reproduced from the matched-pair data files included in `reproduction/data/`.

### Requirements

- Python 3.10+
- Dependencies listed in `reproduction/requirements_reproduce.txt`

### Steps

```bash
cd reproduction
pip install -r requirements_reproduce.txt
python reproduce_everything.py
```

### What the reproduction script does

1. **reproduce_all.py** -- Recomputes every statistic reported in the paper (Pearson r, ICC, MAE, TOST, direction agreement, etc.) from raw matched-pair files across all 5 datasets. Verifies each value against paper claims.
2. **formal_stats_all_datasets.py** -- Runs the full formal statistical analysis (Bland-Altman, per-paper tiers, threshold tables).
3. **generate_figures.py** -- Regenerates all publication figures from the validation data.
4. **build_docx.py** -- Rebuilds the Word document from the Markdown source with embedded figures.

### Expected output

- All verification checks pass (0 failures across ~40 checks)
- 5 datasets: Loladze 2014, Hui 2025, Li 2022, Li 2024 (biochar), Boldorini 2024
- Total: ~1,149 matched observations across ~136 papers
- Pearson r = 0.984--0.999, all proportional TOST equivalence tests pass (+-20% margin)
- Figures saved to `reproduction/figures/`
- DOCX saved as `halpern_2026_v3.docx`
- JSON results saved to `reproduction/output/`

### No API keys required

The reproduction package operates entirely on pre-extracted, pre-aligned validation data. No LLM API calls are made.
