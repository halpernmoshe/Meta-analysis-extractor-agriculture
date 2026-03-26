# Stage 6: LLM Adjudication

**Stage name**: LLM row-level adjudication

**Purpose**: For each QC-passed row, call the LLM adjudicator (adjudicate_llm_universal.py) to make keep / exclude / flag / swap decisions based on PICO criteria. Handles: intervention isolation, outcome disambiguation, comparator identity, estimand verification, T/C swap detection, and plausibility in context.

**Expected inputs**:
- `../5_qc/summary_qc.csv` — QC-passed rows
- `../config.json` — topic PICO and tc_confusion_warnings

**Expected outputs**:
- `adjudication_decisions.jsonl` — per-row LLM decisions with rationale
- `adjudication_summary.json` — decision counts and schema validation report
- `adjudicated_kept.csv` — rows where decision == keep (after swap correction)
