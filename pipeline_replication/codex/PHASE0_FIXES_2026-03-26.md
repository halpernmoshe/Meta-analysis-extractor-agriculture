# Phase 0 Fixes — 2026-03-26

## Fix 1: Plausibility filter in qc_hard_filters.py

### What was broken

The outlier filter in `qc_hard_filters.py` (Check 6) used `|lnRR| > 2.0` as its only numeric plausibility gate. This threshold corresponds to a raw effect of approximately +639% (increase) or -86% (decrease). However, a +609% effect has lnRR = ln(7.09) ≈ 1.96, which sits **under** the 2.0 ceiling and passes through silently.

The V1 notill_tillage analysis included rows from AbdulsattarAlrijabo 2014 (Iraq drought conditions) with effects ranging from +194% to +609%. All of these passed the lnRR filter because they landed between 1.07 and 1.96. These rows contributed to the wrong-direction result (+1.2% vs benchmark -5.7%) and were only discovered during post-hoc spot-checking.

Additionally, the code listed `pct_extreme` as a secondary check (after `extreme_mask`) in the mask union, so even the percent-change bounds (`EFFECT_PCT_LOWER = -90`, `EFFECT_PCT_UPPER = 500`) were too wide to catch these rows.

### What was changed

File: `pipeline_replication/qc_hard_filters.py`

**Constants block** (lines ~36-41):

| Constant | Old value | New value | Rationale |
|----------|-----------|-----------|-----------|
| `EFFECT_PCT_LOWER` | -90 | **-80** | -80% is a realistic floor for extreme crop failure; below -80% almost always indicates a T/C swap or unit error |
| `EFFECT_PCT_UPPER` | 500 | **200** | No field intervention doubles yield in a single season (+200%); this catches Alrijabo-style outliers that slip under lnRR > 2.0 |
| `EXTREME_LNRR_THRESHOLD` | 2.0 | 2.0 (unchanged) | Retained as secondary check |

**Check 6 logic**: Reordered so `pct_extreme` is evaluated **first** (primary filter), with `extreme_mask` (lnRR) as secondary. Both are unioned — any row triggering either is flagged.

**Audit description**: Updated to record that pct_extreme is primary and explains the V1 Alrijabo lesson.

### New behavior

A row with +609% effect (lnRR = 1.96) will now be flagged by the primary `pct_extreme` check (609 > 200) even though it would pass the secondary `extreme_mask` check (1.96 < 2.0).

A row with -85% effect will now be flagged (−85 < −80). Previously it would not be flagged (−85 > −90).

The filter action remains `flag_only` (not automatic exclusion). Flagged rows are still visible in `summary_qc.csv` with `_qc_outlier = True` for human or LLM adjudication review.

### Why 200% / -80% were chosen

- **+200%**: The highest credible single-season yield response to any field-applied amendment in the agricultural literature. Values above this are almost certainly: wrong units (kg vs. g), T/C swap, factorial arm confusion, or data entry error. The Alrijabo rows ranged from +194% (borderline) to +609% (implausible). Setting the ceiling at +200% flags the implausible cluster while keeping the borderline case for LLM adjudication.
- **-80%**: Near-total yield loss (e.g., herbicide damage, severe drought in a treated plot) can legitimately reach −70% to −75%. Values below −80% are almost always a T/C swap (treatment and control columns reversed). This is the same floor used in the adjudication protocol's contextual plausibility criterion.
