# Audit Subset Rule Score

Rows scored: 105

## Confusion
- clean_support -> alignment_or_structure_problem: 1
- clean_support -> clean_support: 8
- clean_support -> low_support_uncertain: 5
- likely_alignment_or_structure_problem -> alignment_or_structure_problem: 57
- likely_alignment_or_structure_problem -> extraction_coverage_problem: 2
- likely_extraction_coverage_problem -> extraction_coverage_problem: 19
- unclear -> alignment_or_structure_problem: 13

## By True Label
- clean_support: {"alignment_or_structure_problem": 1, "clean_support": 8, "low_support_uncertain": 5}
- likely_alignment_or_structure_problem: {"alignment_or_structure_problem": 57, "extraction_coverage_problem": 2}
- likely_extraction_coverage_problem: {"extraction_coverage_problem": 19}
- unclear: {"alignment_or_structure_problem": 13}