# Meta-Analysis Extractor

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Automated AI extraction of quantitative data from scientific PDFs for meta-analysis. A single general-purpose AI agent (Claude Opus 4.6) reads source PDFs directly and produces structured JSON — no domain-specific prompt templates, few-shot examples, or multi-model consensus required. Validated against three published plant science ground-truth datasets (1,184 observations, 87 papers) with formal ICC, CCC, TOST equivalence testing, and Bland-Altman analysis. An independently developed multi-model consensus pipeline (also included) confirms results across 1,889 observations with all r > 0.93 without ground truth.

**Companion code for:** Halpern, M. (2026). "Breaking the Extraction Bottleneck: A Single AI Agent Achieves Equivalence with Published Meta-Analysis Data Across Three Agricultural Datasets." *Research Synthesis Methods*. Agricultural Research Organization -- Volcani Center, Israel.

---

## Validation Summary (Agent Extraction)

| Dataset | Domain | Papers | Obs | r | CCC | MAE | ICC(3,1) | TOST ±3pp |
|---|---|---|---|---|---|---|---|---|
| Loladze 2014 | CO₂ × plant minerals | 46 | 655 | 0.848 | 0.844 | 5.4 pp | 0.845 | p = 0.003 |
| Hui 2023 | Zinc × wheat | 25 | 461 | 0.942 | 0.942 | 7.4 pp | 0.942 | p = 0.047 |
| Li 2022 | Biostimulants × yield | 16 | 68 | 0.968 | 0.966 | 1.6 pp | 0.966 | p < 0.001 |

Agent cost: ~$0.15/paper | Pipeline cost: ~$0.37/paper

---

## Quick Start

```bash
git clone https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.git
cd Meta-analysis-extractor-agriculture
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # Then add your API keys
```

Run extraction on your own dataset:

```bash
python meta_extract.py \
  --config configs/my_dataset.json \
  --input-dir /path/to/pdfs \
  --output-dir output/my_dataset
```

See [REPRODUCE.md](REPRODUCE.md) for full reproduction instructions.

---

## Repository Structure

```
meta_analysis_extractor/
├── meta_extract.py                 # Main entry point
├── consensus_pipeline.py           # Multi-model consensus logic
├── config.py                       # Model configuration and API routing
├── core/                           # Orchestrator, LLM wrapper, state management
├── modules/                        # Recon, extraction, gap-fill, export
├── prompts/                        # LLM prompt templates
├── configs/                        # Dataset-specific JSON configurations
│   ├── loladze_co2_minerals.json
│   ├── hui2023_zinc_wheat.json
│   └── li2022_biostimulant_yield.json
├── validate_full_46.py             # Loladze 2014 pipeline validation
├── validate_hui2023.py             # Hui 2023 pipeline validation
├── validate_li2022.py              # Li 2022 pipeline validation
├── validate_agent_extraction.py    # Agent extraction validation (Loladze)
├── validate_hui2023_agent.py       # Agent extraction validation (Hui)
├── validate_li2022_agent.py        # Agent extraction validation (Li)
├── agent_pipeline_agreement.py     # GT-free cross-method agreement
├── validate_replication.py         # Run-to-run reproducibility
├── formal_statistics.py            # ICC, TOST, Bland-Altman, Cohen's d
├── formal_stats_agent.py           # Agent formal statistics
├── supplementary_cr2_tost.py       # CR2 bias-corrected TOST analysis
├── generate_agent_figures.py       # Publication figure generation
├── paper_figures.py                # Pipeline figure generation
├── programmatic_gt_classifier.py   # Li 2022 scale-harmonized matching
├── harmonize_li2022_agent.py       # Li 2022 agent scale harmonization
├── output/                         # Pre-computed extraction results
│   ├── loladze_v3_combined/        # Loladze pipeline results
│   ├── hui2023_full_35/            # Hui pipeline results
│   ├── li2022_combined/            # Li pipeline results
│   ├── agent_extraction/           # Agent extraction results
│   ├── paper_figures/              # Publication-ready figures
│   └── formal_stats/              # Statistical analysis outputs
├── data/                           # Ontology and few-shot examples
├── requirements.txt
├── .env.example                    # API key template
├── REPRODUCE.md                    # Full reproduction guide
├── CITATION.cff
└── LICENSE                         # MIT
```

---

## How It Works

### Agent Mode (Paper's Primary Method)
A single AI agent (Claude Opus 4.6) reads each PDF and extracts structured JSON with one natural-language instruction per dataset. No few-shot examples, vision pipelines, or multi-model voting. Cost: ~$0.15/paper.

### Pipeline Mode (Independent Comparator)
Three models (Claude Sonnet 4, Kimi K2.5, Gemini 3 Flash) independently extract each paper. Majority-vote consensus reconciles outputs and assigns confidence tiers (HIGH / MEDIUM / LOW). Cost: ~$0.37/paper.

Both methods are validated against the same ground-truth datasets. Their cross-method agreement (r > 0.93 on 1,889 observations without ground truth) provides circularity-free validation.

---

## Citation

```bibtex
@article{halpern2026extraction,
  title   = {Breaking the Extraction Bottleneck: A Single {AI} Agent Achieves
             Equivalence with Published Meta-Analysis Data Across Three
             Agricultural Datasets},
  author  = {Halpern, Moshe},
  journal = {Research Synthesis Methods},
  year    = {2026},
  note    = {Agricultural Research Organization -- Volcani Center, Israel.
             Code: https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
