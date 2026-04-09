# Cross-Batch Status

After tightening report parsing and fixing the false `zero-match` detector, the current state is:

## Audit subset

- clean_support -> clean_support: 11 / 11
- alignment_or_structure_problem -> alignment_or_structure_problem: 92 / 92
- extraction_coverage_problem -> extraction_coverage_problem: 10 / 10

Interpretation:
- The original audited benchmark subset is now cleanly separated again.
- The parser fixes did not destabilize the strongest validated set.

## Held-out subset 1

- clean_support -> clean_support: 28 / 34
- clean_support -> low_support_uncertain: 6 / 34
- alignment_or_structure_problem -> alignment_or_structure_problem: 26 / 26
- extraction_coverage_problem -> extraction_coverage_problem: 3 / 10
- extraction_coverage_problem -> alignment_or_structure_problem: 7 / 10
- unclear -> alignment_or_structure_problem: 4 / 4

Interpretation:
- Strong clean claims remain mostly clean, with the remainder conservatively downgraded to `low_support_uncertain`.
- The alignment bucket generalizes very well.
- The coverage bucket is mixed in this batch; several papers look more like construct/alignment problems than pure extraction limitations.

## Held-out subset 2

- clean_support -> clean_support: 8 / 14
- clean_support -> low_support_uncertain: 5 / 14
- clean_support -> alignment_or_structure_problem: 1 / 14
- alignment_or_structure_problem -> alignment_or_structure_problem: 57 / 59
- alignment_or_structure_problem -> extraction_coverage_problem: 2 / 59
- extraction_coverage_problem -> extraction_coverage_problem: 19 / 19
- unclear -> alignment_or_structure_problem: 13 / 13

Interpretation:
- The second held-out batch was initially the worst-looking one, but most of that was due to weak-label contamination from report parsing bugs.
- After the fixes, the batch looks coherent:
  - truly figure-limited papers stay in coverage
  - mixed papers mostly land in alignment
  - clean papers are either kept clean or downgraded to `low_support_uncertain`

## Main Takeaway

The strongest stable result across batches is:

- `alignment_or_structure_problem` is the dominant and most reproducible failure mode.
- `extraction_coverage_problem` is real, but narrower than it first appeared.
- `low_support_uncertain` is useful as a conservative sink for weak clean claims.

This matches the broader conceptual shift in the project:
- the hard problem is often not "hallucination" in the narrow sense
- it is construct mismatch, forced matching, wrong arm / tissue / timepoint selection, and figure-only target data

## Next Best Step

Move from generic risk flags to explicit construct-drift flags:

- concentration_vs_content
- tissue_mismatch
- arm_mismatch
- timepoint_mismatch
- pooled_vs_subgroup_mismatch
- figure_only_target

These are better aligned with both the local evidence and the broader philosophy/statistics literature.
