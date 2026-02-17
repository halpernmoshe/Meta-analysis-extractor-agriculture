# Multi-Model AI Consensus Pipeline for Meta-Analysis Data Extraction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An automated system for extracting quantitative data from scientific papers for meta-analysis, using multi-model LLM consensus (Claude Sonnet 4 + Kimi K2.5 + Gemini 3 Flash).

**Paper**: [Preprint on arXiv](https://arxiv.org/abs/XXXX.XXXXX) (submitted to *Research Synthesis Methods*)

## Key Results

Validated against three published meta-analysis datasets:

| Dataset | Papers | Observations | Pearson r | MAE | Direction | Effect Diff |
|---------|--------|:------------:|:---------:|:---:|:---------:|:-----------:|
| Loladze 2014 (CO2/minerals) | 46 | 635 | 0.669 | 7.9% | 85% | 0.05 pp |
| Hui 2023 (Zn/wheat) | 21 | 279 | 0.950 | 4.95% | 99% | 2.17 pp |
| Li 2022 (biostimulant/yield) | 28 | 163 | 0.453 | 11.6% | 87% | 0.06 pp |

- **TOST equivalence** with human extraction at +/-2 pp (p < 0.001)
- **Paper-level ICC = 0.838** (within human inter-rater range of 0.70-0.90)
- **240-fold API cost reduction** vs manual extraction (~$0.37/paper)
- **73% more observations** from consensus vs best single model
- Alignment-corrected accuracy: **r = 0.876, MAE = 4.3%** (374 well-aligned observations)

## How It Works

```
PDF Input
    |
    v
[1] Challenge-Aware Reconnaissance (Claude Sonnet 4)
    - Detects scanned PDFs, image tables, figure-only data
    - Routes to TEXT, HYBRID, or VISION extraction mode
    |
    v
[2] Dual-Model Extraction
    - Claude Sonnet 4 (primary)
    - Kimi K2.5 (secondary)
    - Identical structured prompts
    |
    v
[3] Consensus Building (2-of-3 voting)
    - Match observations by element/tissue + value tolerance
    - Gemini 3 Flash tiebreaker when consensus < 30%
    |
    v
[4] Post-Processing
    - Duplicate removal, null filtering
    - Effect size computation (% change from control)
    |
    v
Validated Output (JSON/CSV)
```

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

Create a `.env` file with your API keys (see `.env.example`):
```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
MOONSHOT_API_KEY=...
```

### Run Extraction

```bash
# Consensus pipeline with a config file
python consensus_pipeline.py --config configs/loladze_co2_minerals.json

# Single-model extraction (interactive)
python meta_extract.py --input ./input --output ./output --provider google
```

### Validate Results

```bash
# Loladze 2014 validation
python validate_full_46.py

# Hui 2023 validation
python validate_hui2023.py

# Li 2022 validation
python validate_li2022.py

# Formal statistics (TOST, ICC, Bland-Altman, bootstrap CIs)
python formal_statistics.py
python formal_stats_hui2023.py
python formal_stats_li2022.py
```

### Generate Figures

```bash
python paper_figures.py            # Main paper figures
python fig_tost_equivalence.py     # TOST forest + Bland-Altman cross-dataset
```

## Project Structure

```
meta_analysis_extractor/
├── consensus_pipeline.py      # Main consensus extraction pipeline
├── meta_extract.py            # Single-model extraction entry point
├── config.py                  # Model selection, API keys, cost estimates
├── core/
│   ├── orchestrator.py        # Main workflow controller
│   ├── llm.py                 # Unified LLM wrapper (Anthropic + Google + Kimi)
│   └── state.py               # Session state management
├── modules/
│   ├── recon.py               # Challenge-aware reconnaissance
│   ├── extract.py             # Data extraction
│   ├── gap_fill.py            # Targeted gap filling
│   ├── validate.py            # Data validation
│   └── export.py              # JSON/CSV export
├── prompts/                   # LLM prompt templates
├── configs/                   # Meta-analysis configuration files
│   ├── loladze_co2_minerals.json
│   ├── hui2023_zinc_wheat.json
│   └── li2022_biostimulant_yield.json
├── output/
│   ├── paper_figures/         # Publication figures (300 DPI)
│   ├── paper_supplementary/   # Supplementary tables (S1-S4)
│   ├── formal_stats/          # TOST, ICC, Bland-Altman outputs
│   ├── sensitivity/           # LOO sensitivity analysis
│   └── tolerance_sensitivity.csv
│
├── # Validation scripts
├── validate_full_46.py        # Loladze validation (primary)
├── validate_hui2023.py        # Hui validation
├── validate_li2022.py         # Li 2022 validation
├── formal_statistics.py       # TOST, ICC, Bland-Altman, bootstrap
├── sensitivity_loo.py         # Leave-one-out sensitivity
├── tolerance_sensitivity.py   # Matching tolerance sensitivity
├── ablation_analysis.py       # Consensus vs single-model ablation
├── kimi_primary_simulation.py # Kimi-primary architecture simulation
│
├── # Figure generation
├── paper_figures.py           # Main paper figures
├── fig_tost_equivalence.py    # Cross-dataset TOST + Bland-Altman
│
├── # Paper
├── PAPER_FULL_DRAFT.md        # Manuscript (~9,000 words)
└── SPOT_CHECK_LOG.md          # Detailed error diagnosis log
```

## Configuration

The pipeline is fully configuration-driven. To apply it to a new meta-analysis topic, create a JSON config file:

```json
{
  "pico": {
    "population": "Crop plants",
    "intervention": "Biostimulant application",
    "comparison": "Untreated control",
    "outcomes": ["Yield"]
  },
  "extraction": {
    "primary_outcome": "YIELD",
    "effect_type": "percent_change",
    "variance_type": "SE"
  },
  "models": {
    "recon": "claude-sonnet-4",
    "extract_primary": "claude-sonnet-4",
    "extract_secondary": "kimi-k2.5",
    "tiebreaker": "gemini-3-flash-preview"
  }
}
```

All three validation datasets were processed using this configuration approach, changing only the JSON file between runs.

## Cost

| Component | Cost per Paper | Notes |
|-----------|:--------------:|-------|
| Claude Sonnet 4 (recon + extract) | ~$0.28 | 200K context |
| Kimi K2.5 (extract) | ~$0.04 | 256K context |
| Gemini 3 Flash (tiebreaker) | ~$0.05 | 1M context, invoked for ~22% of papers |
| **Total** | **~$0.37** | 46 papers = ~$17 total |

Manual extraction comparison: ~$30/hr x 4 hrs = $120/paper.

## Reproducing Paper Results

Source PDFs cannot be redistributed due to publisher copyright. To reproduce the validation:

1. Obtain PDFs listed in the paper's reference list from your institutional library
2. Place them in `input/` (Loladze), `input_hui2023_full/` (Hui), or `input_li2022/` (Li)
3. Run the consensus pipeline with the corresponding config
4. Run the validation scripts against the published ground-truth datasets

Pre-computed validation outputs (matched observations, formal statistics, and figures) are included in the `output/` directory for inspection without re-running extraction.

## Ground Truth Datasets

| Dataset | Source | Access |
|---------|--------|--------|
| Loladze 2014 | eLife 3:e02245 | [Supplementary dataset](https://elifesciences.org/articles/02245) |
| Hui et al. 2023 | J. Soil Sci. Plant Nutr. | Supplementary materials in original paper |
| Li et al. 2022 | Front. Plant Sci. 13:836702 | [Open access](https://doi.org/10.3389/fpls.2022.836702) |

## Citation

If you use this software, please cite:

```bibtex
@article{author2026multimodel,
  title={Multi-Model AI Consensus Pipeline for Automated Data Extraction
         in Plant Science Meta-Analysis},
  author={Halpern, Moshe},
  journal={Research Synthesis Methods},
  year={2026},
  note={Preprint: arXiv:XXXX.XXXXX}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
