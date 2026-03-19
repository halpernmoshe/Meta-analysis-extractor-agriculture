# Reproduction Guide

This document provides step-by-step instructions to reproduce all validation results reported in the paper:

> Halpern, M. (2026). "Breaking the Extraction Bottleneck: A Single AI Agent Achieves Equivalence with Published Meta-Analysis Data Across Three Agricultural Datasets." *Research Synthesis Methods*.

---

## System Requirements

- **Python**: 3.10 or later
- **OS**: Windows 10/11, macOS, or Linux
- **RAM**: 8 GB minimum (16 GB recommended for large datasets)
- **Disk**: ~2 GB for dependencies + output files
- **API keys**: Anthropic (Claude), Google (Gemini), and Moonshot (Kimi) -- see below
- **PDF access**: Source PDFs must be obtained independently (copyright)

---

## 1. Installation

```bash
git clone https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.git
cd Meta-analysis-extractor-agriculture
python -m venv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import anthropic, google.genai, scipy, matplotlib, openpyxl, fitz; print('All dependencies OK')"
```

---

## 2. Environment Setup

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```dotenv
ANTHROPIC_API_KEY=sk-ant-...      # https://console.anthropic.com/
GOOGLE_API_KEY=AIza...             # https://aistudio.google.com/apikey
MOONSHOT_API_KEY=sk-...            # https://platform.moonshot.cn/
```

Not all keys are needed for every step. Validation scripts (Tasks A--F below) require no API keys at all -- they operate on pre-extracted output files included in the repository. API keys are only required if you want to re-run extraction from scratch (Task H).

---

## 3. Obtaining Ground-Truth Data

The three ground-truth datasets are publicly available from the original publications:

| Dataset | Source | Access |
|---|---|---|
| Loladze 2014 | CO2+Dataset.xlsx | https://doi.org/10.7554/eLife.02245 (Supplementary) |
| Hui 2023 | ground.xlsx | Supplementary materials in original article |
| Li 2022 | Supplementary tables | https://doi.org/10.3389/fpls.2022.836702 |

Place ground-truth files at the paths referenced in each validation script, or update the `GT_PATH` variable at the top of each script to point to your local copy.

---

## 4. Obtaining Source PDFs

Source PDFs cannot be redistributed. You must obtain them from the original publishers. The `modules/acquisition/` module supports automated retrieval via Unpaywall where legally permissible. Place PDFs in the appropriate `input*/` directories.

---

## 5. Reproducing Results

The pre-extracted output files are included in the repository under `output/`. You can validate them without re-running extraction.

### Task A: Agent Extraction — Loladze 2014 (Table 1 in paper)

```bash
python validate_agent_extraction.py
```

**Expected output**: r=0.848, MAE=5.4pp, 655 matched observations across 46 papers. ICC(3,1)=0.845. Tier breakdown: 25 Excellent, 11 Good, 9 Fair, 1 Poor.

### Task B: Agent Extraction — Hui 2023 (Table 1 in paper)

```bash
python validate_hui2023_agent.py
```

**Expected output**: r=0.942, MAE=7.4pp, 461 matched observations across 25 papers. ICC(3,1)=0.942.

### Task C: Agent Extraction — Li 2022 (Table 1 in paper)

```bash
python validate_li2022_agent.py
python harmonize_li2022_agent.py
```

**Expected output**: Harmonized HIGH tier r=0.968, MAE=1.6pp, 68 observations across 16 papers. ICC(3,1)=0.966.

### Task D: Agent-Pipeline Agreement (Table 5 in paper, GT-free)

```bash
python agent_pipeline_agreement.py
```

**Expected output**: Loladze r=0.933 (1,205 obs, 44 papers), Hui r=0.971 (185 obs, 20 papers), Li r=0.994 (499 obs, 36 papers). Total: 1,889 obs, 100 papers. Results written to `output/agent_pipeline_agreement.json`.

### Task E: Replication Agreement (Table 4 in paper)

```bash
python validate_replication.py
```

**Expected output**: 1,231 matched observations across 95 papers. Results in `output/replication_agreement.json`.

### Task F: Formal Statistics (Tables 2-3 in paper: ICC, TOST, Bland-Altman, Cohen's d)

Agent formal stats:

```bash
python formal_stats_agent.py
```

**Expected output**: JSON reports in `output/agent_formal_stats/`. ICC(3,1) = 0.845 (Loladze), 0.942 (Hui), 0.966 (Li). TOST ±3pp: all p ≤ 0.047. Cohen's d: 0.016–0.103.

Pipeline formal stats (for Appendix A):

```bash
python formal_statistics.py --dataset loladze
python formal_stats_hui2023.py
python formal_stats_li2022.py
```

**Expected output**: JSON reports in `output/formal_stats/`, `output/hui2023_formal_stats/`, `output/li2022_formal_stats/`.

### Task F2: CR2 Bias-Corrected TOST (Supplementary Table S1)

```bash
python supplementary_cr2_tost.py
```

**Expected output**: `output/supplementary_table_s1_cr2.md` and `.json`. CR2 Hui ±3pp p=0.099 (does not survive correction); all other decisions unchanged.

### Task G: Figure Generation

```bash
python generate_agent_figures.py
```

**Expected output**: 5 PNG files in `output/paper_figures/`:
- `fig1_agent_gt_scatter.png` — Agent vs GT scatter (3 panels)
- `fig2_per_paper_mae.png` — Per-paper MAE distribution
- `fig3_bland_altman.png` — Bland-Altman (3 panels)
- `fig4_gt_free_agreement.png` — Agent-pipeline agreement (3 panels)
- `fig5_error_taxonomy.png` — Error taxonomy (93/3/3%)

Pipeline figures (supplementary):

```bash
python paper_figures.py
```

### Task H: Pipeline Validation (Appendix A)

```bash
python validate_full_46.py --results-dir output/loladze_v3_combined
python validate_hui2023.py
python validate_li2022.py
python programmatic_gt_classifier.py
```

**Expected output**: Pipeline results as reported in Appendix A, Table A1.

---

## 6. Re-Running Extraction From Scratch

Re-running extraction requires API keys and source PDFs. This is not necessary to validate reported results (the extracted outputs are included), but is available for full replication.

### Pipeline extraction (consensus of 3 models)

```bash
python meta_extract.py \
  --config configs/loladze_co2_minerals.json \
  --input-dir /path/to/loladze_pdfs \
  --output-dir output/loladze_rerun
```

Repeat with `hui2023_zinc_wheat.json` and `li2022_biostimulant_yield.json` for the other datasets.

### Agent extraction (single-model)

Agent extraction was performed using Claude Code (Anthropic's CLI agent) with Claude Opus 4.6. The Loladze dataset used a 3-pass approach: extract, text cross-check, fix flagged observations. Hui and Li used single-pass extraction. The agent extraction outputs are in `output/agent_extraction/`, `output/hui2023_agent_extraction/`, and `output/li2022_agent_extraction/`.

---

## 7. Approximate Costs and Time

| Step | API cost | Wall time |
|---|---|---|
| Agent extraction (87 papers) | ~$13 (~$0.15/paper) | ~3 hours |
| Pipeline extraction (50 papers) | ~$17 total (~$0.37/paper) | ~6 hours |
| Validation (no API calls) | $0 | < 1 minute each |
| Formal statistics (no API calls) | $0 | < 1 minute |
| Figure generation (no API calls) | $0 | < 30 seconds |

Total cost to fully replicate all three datasets: approximately $50--70 in API fees.

---

## 8. Troubleshooting

### Windows encoding errors

All scripts include `sys.stdout.reconfigure(encoding='utf-8')` for Windows compatibility. If you see encoding errors, ensure you are running Python 3.10+.

### Path configuration

Validation scripts contain hardcoded ground-truth paths. When running on a different machine, update `GT_PATH` at the top of each validation script to point to your local copies of the ground-truth Excel files. See Section 3 above for download links.

### Missing ground-truth files

If a validation script fails with a file-not-found error, ensure you have placed the ground-truth Excel files at the expected paths. See Section 3 above.

---

## 9. Output Directory Guide

| Directory | Contents |
|---|---|
| `output/agent_extraction/` | Loladze agent extraction JSONs |
| `output/hui2023_agent_extraction/` | Hui agent extraction JSONs |
| `output/li2022_agent_extraction/` | Li agent extraction JSONs |
| `output/agent_formal_stats/` | Agent formal statistics JSON |
| `output/loladze_v3_combined/` | Loladze pipeline consensus JSONs |
| `output/hui2023_full_35/` | Hui pipeline consensus JSONs |
| `output/li2022_combined/` | Li pipeline consensus JSONs + classification |
| `output/formal_stats/` | Loladze pipeline formal statistics JSON |
| `output/hui2023_formal_stats/` | Hui pipeline formal statistics JSON |
| `output/li2022_formal_stats/` | Li pipeline formal statistics JSON |
| `output/paper_figures/` | Publication-ready PNG figures |
| `output/paper_supplementary/` | Supplementary materials |
| `output/sensitivity/` | Leave-one-out sensitivity analysis |
