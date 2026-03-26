# Stage 8: Synthesis

**Stage name**: Meta-analytic synthesis

**Purpose**: Compute pooled effect size (lnRR, reported as %-change) using random-effects model. Generates primary result and benchmark-aligned subset result. Produces forest plot and subgroup analyses.

**Expected inputs**:
- `../7_normalize/summary_normalized.csv` — normalized, adjudicated rows
- `../config.json` — benchmark effect, moderator definitions

**Expected outputs**:
- `synthesis_results.json` — pooled effect estimate, 95% CI, heterogeneity (I², tau²), k
- `forest_plot.png` — per-paper effect sizes with pooled estimate
- `subgroup_results.json` — by crop type, application method, study setting
- `benchmark_comparison.json` — pipeline result vs Ma et al. 2024 (+12%)
