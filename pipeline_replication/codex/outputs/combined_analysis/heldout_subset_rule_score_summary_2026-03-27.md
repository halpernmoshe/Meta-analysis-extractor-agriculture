# Audit Subset Rule Score

Rows scored: 74

## Confusion
- clean_support -> clean_support: 28
- clean_support -> low_support_uncertain: 6
- likely_alignment_or_structure_problem -> alignment_or_structure_problem: 26
- likely_extraction_coverage_problem -> alignment_or_structure_problem: 7
- likely_extraction_coverage_problem -> extraction_coverage_problem: 3
- unclear -> alignment_or_structure_problem: 4

## By True Label
- clean_support: {"clean_support": 28, "low_support_uncertain": 6}
- likely_alignment_or_structure_problem: {"alignment_or_structure_problem": 26}
- likely_extraction_coverage_problem: {"alignment_or_structure_problem": 7, "extraction_coverage_problem": 3}
- unclear: {"alignment_or_structure_problem": 4}