# Stage 5: QC (Hard Filters)

**Stage name**: Deterministic quality control

**Purpose**: Apply hard numeric filters to extracted rows. Flags or removes implausible values, missing means, zero denominators, and out-of-range effects. For humic_acid_yield the topic-specific override flags |effect| > 200% (tighter than the default |lnRR| > 2.0 threshold).

**Expected inputs**:
- `../4_extract/summary.csv` — raw extracted rows

**Expected outputs**:
- `qc_audit.json` — per-row QC decisions with flag reasons
- `summary_qc.csv` — rows passing all hard filters
- `qc_summary.json` — counts of flagged / removed / passed rows
