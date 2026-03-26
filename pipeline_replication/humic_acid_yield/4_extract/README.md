# Stage 4: Extract

**Stage name**: LLM data extraction

**Purpose**: Extract treatment means, control means, sample sizes, and variance metrics from downloaded PDFs using the LLM extraction pipeline.

**Expected inputs**:
- `../3_download/pdfs/` — downloaded PDF files
- `../config.json` — PICO and extraction configuration

**Expected outputs**:
- `*_agent.json` — per-paper raw extraction JSON files
- `summary.csv` — all extracted rows in flat CSV format
- `summary_validated.csv` — rows after deterministic QC filter (Stage 5 output written here)
- `extraction_summary.json` — counts, coverage, and error log
