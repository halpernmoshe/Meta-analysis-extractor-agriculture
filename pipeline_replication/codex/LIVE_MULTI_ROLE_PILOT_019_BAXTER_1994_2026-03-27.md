# Live Multi-Role Pilot: 019_Baxter_1994

This note records the first replacement of seeded report-derived role outputs with richer live paper-grounded role outputs in the multi-role full-context prototype.

## What changed

- Replaced the minimal seeded role outputs for `019_Baxter_1994` with richer structured outputs for:
  - `design_agent`
  - `narrative_agent`
  - `table_agent`
  - `figure_agent`
  - `benchmark_agent`
  - `consistency_agent`
- Preserved the shared full-paper context and only updated the role-level `output_schema` blocks.
- Reran `merge_multi_role_pilot_outputs.py`.

## Result

The merged output moved from a single coarse contradiction-bearing target to a richer multi-view representation:

- `n_claim_rows`: 8
- `n_constraints`: 5
- `n_role_contradictions`: 3

Key contradiction pattern:

- `construct_drift`
- `benchmark_comparability_conflict`
- `unit_conflict`

## Practical interpretation

This paper is now represented as a clean example of:

- benchmark construct present in the paper
- benchmark construct located primarily in figures
- numerically explicit table content capturing a different construct
- narrative support aligning with whole-plant uptake/content rather than foliar concentration

That is exactly the kind of paper where a single numeric extractor can look wrong even when it is reading real values correctly.

## Why this matters

The live multi-role version makes the central issue explicit:

- `figure_agent` supports `target::foliar_concentration`
- `table_agent` supports `target::total_content`
- `benchmark_agent` states only the figure-based foliar concentration is benchmark-comparable
- `consistency_agent` identifies the paper as contradiction-bearing rather than simply “failed”

This is a better representation than the earlier seeded summary because it preserves:

- design constraints
- narrative scope limits
- construct drift
- modality mismatch
- benchmark comparability

## Current takeaway

`019_Baxter_1994` is a strong pilot case for the new synthesis idea:

- not just numeric extraction
- not just narrative reading
- but multi-role, full-context, contradiction-aware evidence synthesis

## Next best step

Run the same live replacement on one cleaner paper and one different failure mode paper:

- clean case: `015_Pleijel_2009`
- other failure mode: `035_Oksanen_2005`

Then compare whether the same merger logic distinguishes:

- clean support
- construct drift
- figure-only benchmark targets
- treatment-arm mismatch
