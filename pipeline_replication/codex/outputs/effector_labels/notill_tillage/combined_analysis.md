# Combined Post-Processing + Effector Analysis

This note summarizes the combined test on `notill_tillage` after:

- row-level Codex-style post-processing
- effector normalization from the kept rows

## Result

The combination did not fully solve the benchmark gap.

## Observed Subsets

- `benchmark_aligned` rows: 26
- `benchmark_aligned` papers: 4
- `benchmark_aligned` pooled effect: `-11.56%`
- `benchmark_aligned` 95% CI: `[-15.26, -7.70]`
- Benchmark: `-5.7%`

## Interpretation

The universal post-processing and effector normalization layers improved semantic cleanliness, but the remaining mismatch is still large.

The limiting factor is still mostly context and sample composition, not row-level semantic leakage.
