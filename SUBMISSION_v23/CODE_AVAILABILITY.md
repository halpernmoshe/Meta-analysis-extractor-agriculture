# Code and Data Availability

**Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets**

Moshe Halpern

---

## Repository

**GitHub:** [https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture](https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture)

**License:** MIT

---

## Installation

### Prerequisites

- Python 3.10+
- An API key for at least one LLM provider:
  - Anthropic (Claude): [https://console.anthropic.com/](https://console.anthropic.com/)
  - Google (Gemini): [https://aistudio.google.com/](https://aistudio.google.com/)
  - Moonshot (Kimi): [https://platform.moonshot.cn/](https://platform.moonshot.cn/)

### Setup

```bash
# Clone the repository
git clone https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.git
cd Meta-analysis-extractor-agriculture

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# or: .\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API key(s):
#   ANTHROPIC_API_KEY=sk-ant-...
#   GOOGLE_API_KEY=AI...
#   MOONSHOT_API_KEY=...
```

---

## Quick Start (3 Steps)

### Step 1: Create a configuration file

Create a JSON config specifying what to extract. See `configs/` for examples.

```json
{
  "name": "My Meta-Analysis",
  "intervention": "Treatment description",
  "control": "Control description",
  "primary_outcomes": ["Outcome variable (units)"],
  "expected_direction": "positive",
  "extraction_priorities": [
    "Extract EVERY treatment combination separately",
    "Get variance (SE/SD) and sample size (n)"
  ],
  "pico": {
    "intervention": {
      "treatment_variable": "TREATMENT",
      "treatment_keywords": ["treatment", "treated"]
    },
    "comparison": {
      "control_definition": "Untreated control",
      "control_keywords": ["control", "CK", "untreated"]
    }
  }
}
```

### Step 2: Extract data from PDFs

```bash
# Place PDFs in an input directory, then run:
python meta_extract.py \
  --input ./my_papers/ \
  --output ./my_output/ \
  --provider anthropic \
  --auto my_config.json
```

### Step 3: Validate against a reference standard

```bash
# If you have ground-truth data:
python run_all_validations.py \
  --config my_config.json \
  --output ./my_output/
```

---

## Repository Structure

```
meta_analysis_extractor/
|
|-- meta_extract.py              # Main entry point for extraction
|-- config.py                    # Model selection, API keys, cost estimates
|-- run_all_validations.py       # Run validation across all datasets
|-- formal_stats_all_datasets.py # Compute formal statistics (ICC, TOST, etc.)
|-- consensus_pipeline.py        # Multi-model consensus extraction
|
|-- core/                        # Core pipeline components
|   |-- orchestrator.py          # Main workflow controller (16 phases)
|   |-- llm.py                   # Unified LLM wrapper (Anthropic + Google + Kimi)
|   |-- state.py                 # Session state with checkpoint/resume
|
|-- modules/                     # Extraction and validation modules
|   |-- recon.py                 # Paper scanning/reconnaissance
|   |-- extract.py               # Data extraction from PDFs
|   |-- figure_extract.py        # Figure/chart data extraction
|   |-- gap_fill.py              # Targeted gap filling for missing data
|   |-- validate.py              # Data validation and quality checks
|   |-- export.py                # JSON/CSV export
|   |-- ground_truth.py          # Ground-truth comparison utilities
|   |-- variance_rescue.py       # Vision-based variance extraction
|   |-- abstract_validator.py    # Abstract-level validation
|   |-- human_validator.py       # Human review interface
|
|-- mcp_server/                  # Model Context Protocol server
|   |-- server.py                # MCP server (7 tools for Claude Desktop)
|   |-- cli.py                   # Standalone CLI interface
|   |-- gt_matcher.py            # LLM-driven GT alignment tool
|
|-- configs/                     # Dataset configuration files
|   |-- loladze_co2_minerals.json
|   |-- hui2023_zinc_wheat.json
|   |-- li2022_biostimulant_yield.json
|   |-- biochar_crop_yield.json
|   |-- boldorini2024_predator_yield.json
|
|-- prompts/                     # LLM prompt templates
|-- data/                        # Reference data and ontologies
|-- output/                      # Extraction results and validation data
```

---

## Major Scripts and Modules

### Extraction

| Script | Description |
|--------|-------------|
| `meta_extract.py` | Main entry point. Runs the full 16-phase extraction pipeline (orientation, PICO definition, recon, extraction, gap fill, validation, export). Supports Claude, Gemini, and Kimi. |
| `core/orchestrator.py` | Workflow controller with checkpoint/resume capability. Saves state after every phase and every 5 papers. |
| `core/llm.py` | Unified LLM client supporting Anthropic, Google, and Moonshot APIs with retry logic and cost tracking. |
| `consensus_pipeline.py` | Multi-model consensus extraction. Runs 2--3 models on the same papers and merges results by fuzzy value matching. |
| `modules/extract.py` | Core extraction module. Sends PDF content + config-driven prompts to the LLM and parses structured JSON output. |
| `modules/figure_extract.py` | Specialized figure/chart extraction using vision capabilities. |
| `modules/gap_fill.py` | Post-extraction gap filling for missing variance, sample size, and moderator values. |
| `modules/variance_rescue.py` | Vision-based variance extraction from table images and figure error bars. |

### Validation

| Script | Description |
|--------|-------------|
| `formal_stats_all_datasets.py` | Computes ICC, TOST (CR2), Bland-Altman, bootstrap CIs, and tier classification across all datasets. Outputs `output/formal_stats_all_datasets.json`. |
| `validate_full_46.py` | Loladze validation: metadata-based matching with Hungarian algorithm, pooling detection. |
| `validate_hui2023.py` | Hui validation: raw-mean matching with tolerance-based alignment. |
| `validate_li2022.py` | Li validation: scale-harmonized matching with unit conversion detection. |
| `validate_boldorini.py` | Boldorini validation: lnRR-based comparison against published effect sizes. |
| `validate_agent_extraction.py` | Agent vs. pipeline agreement (ground-truth-free validation). |
| `validate_replication.py` | Run 1 vs. Run 2 reproducibility comparison. |
| `mcp_server/gt_matcher.py` | LLM-driven alignment tool for matching extracted data to reference standards. |

### Utility

| Script | Description |
|--------|-------------|
| `config.py` | Configuration: model selection, API keys, token limits, cost estimates. |
| `core/state.py` | Session state management with full serialization/deserialization and checkpoint/resume. |
| `modules/export.py` | Export to JSON, CSV, and methods documentation. |

---

## Deployment Modes

The system supports three deployment modes:

| Mode | Interface | Requirements | Use Case |
|------|-----------|-------------|----------|
| **CLI Pipeline** | Command-line (`meta_extract.py`) | Python + API key | Batch processing, automation |
| **MCP Server** | Claude Desktop (7 tools) | Claude Desktop + API key | Interactive GUI-based extraction |
| **Skill** | Claude Code slash command | Claude Code CLI | Single-paper interactive extraction |

All modes share the same core extraction pipeline.

---

## Reproducing Paper Results

To reproduce all validation results reported in the paper:

```bash
# Ensure all PDFs are in the appropriate input directories
# (see configs/ for paths)

# Run formal statistics across all 5 datasets
python formal_stats_all_datasets.py

# Output: output/formal_stats_all_datasets.json
# Contains: ICC, TOST, Bland-Altman, bootstrap CIs, tier counts

# Generate figures
python FINAL_SUBMISSION_2026-03-08/generate_figures.py
```

Pre-computed results are available in:
- `output/formal_stats_all_datasets.json` -- All statistical results
- `output/loladze_v3_combined/` -- Loladze extraction and validation
- `output/hui2023_extraction/` -- Hui extraction and validation
- `output/li2022_combined/` -- Li extraction and validation
- `output/biochar_extraction/` -- Biochar extraction and validation
- `output/boldorini_extraction/` -- Boldorini extraction and validation

---

## API Costs (March 2026 Pricing)

| Provider | Model | Cost per Paper | Notes |
|----------|-------|----------------|-------|
| Anthropic | Claude Opus 4.6 | ~$0.08--0.15 | Used for agent extraction in paper |
| Anthropic | Claude Sonnet 4 | ~$0.04--0.08 | Consensus pipeline |
| Google | Gemini 2.5 Flash | ~$0.02 | Consensus pipeline |
| Moonshot | Kimi K2.5 | ~$0.03 | Consensus pipeline, highest coverage |

Total cost for all five datasets (single-agent extraction): approximately $15--25.

---

## Citation

If you use this tool, please cite:

> Halpern, M. (2026). Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets. *Research Synthesis Methods* (submitted).

---

## Contact

Moshe Halpern
Institute of Soil, Water and Environmental Sciences
Agricultural Research Organization -- Volcani Center
Rishon LeZion 7505101, Israel
