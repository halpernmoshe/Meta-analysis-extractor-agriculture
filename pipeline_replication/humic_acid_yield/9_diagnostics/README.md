# Stage 9: Diagnostics

**Stage name**: Diagnostic and deviation analysis

**Purpose**: Run post-hoc diagnostics to explain any divergence between pipeline result and benchmark. Includes benchmark-aligned subset analysis (secondary diagnostic only — not the primary result), funnel plots, influence analysis, and T/C swap audit.

**Expected inputs**:
- `../8_synthesize/synthesis_results.json`
- `../7_normalize/summary_normalized.csv`
- `../config.json` — benchmark effect and benchmark_alignment_labels

**Expected outputs**:
- `diagnostics_v2.json` — full diagnostic record: PRISMA flow, QC attrition, adjudication breakdown, aligned vs full comparison
- `funnel_plot.png` — publication bias assessment
- `influence_analysis.csv` — leave-one-out sensitivity per paper
- `deviation_log.md` — narrative explanation of any gap between pipeline and benchmark (required if |pipeline - benchmark| > 5pp)
