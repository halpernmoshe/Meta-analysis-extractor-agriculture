# Meta-Analysis Extractor

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)

Automated multi-model AI pipeline for extracting quantitative data from scientific PDFs for meta-analysis. Three large language models (Claude Sonnet 4, Kimi K2.5, Gemini 2.5 Pro) independently extract each paper; a majority-vote consensus step reconciles their outputs and assigns confidence tiers (HIGH / MEDIUM / LOW) to every observation. Validated against three published meta-analysis ground-truth datasets (921 observations, 92 papers) with formal TOST equivalence testing, ICC reliability, and Bland-Altman analysis. An independent agent-based replication (no shared code or ground truth) confirms results across 1,889 observations with all Pearson r > 0.93.

**Companion code for:** Halpern, M. (2026). "Multi-Model AI Consensus for Confidence-Stratified Data Extraction in Plant Science Meta-Analysis." *Research Synthesis Methods*. Agricultural Research Organization -- Volcani Center, Israel.

---

## Validation Summary

| Dataset | Domain | Papers | Obs | Pearson r | MAE | TOST |
|---|---|---|---|---|---|---|
| Loladze 2014 | CO2 x plant minerals | 46 | 646 | 0.812 | 6.2 pp | p = 0.013 |
| Hui 2023 | Zinc x wheat | 21 | 319 | 0.999 | 0.43 pp | p < 0.001 |
| Li 2022 | Biostimulants x yield | 18 | 110 | 0.999 | 0.32 pp | p < 0.001 |

Cost: ~$0.37/paper | Time saved: 70-75% vs. manual extraction

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

1. **Recon** -- Lightweight LLM pass locates relevant tables, figures, and sections in each PDF.
2. **Extract** -- Three models independently produce structured JSON (means, sample sizes, variance). Papers are routed to TEXT, HYBRID, or VISION mode based on PDF quality.
3. **Consensus** -- Observations are matched across models by treatment identity. Majority vote selects agreed values; a tiebreaker model resolves three-way disagreements.
4. **Output** -- Per-observation confidence tiers, JSON/CSV export, and outlier flagging.

---

## Citation

```bibtex
@article{halpern2026multimodel,
  title   = {Multi-Model {AI} Consensus for Confidence-Stratified Data Extraction
             in Plant Science Meta-Analysis},
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
