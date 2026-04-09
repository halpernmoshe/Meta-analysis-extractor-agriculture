# Four-Way Policy Results

## Policy

The original 3-way policy has now been extended to a 4-way policy:

1. `clean_support`
2. `alignment_or_structure_problem`
3. `extraction_coverage_problem`
4. `low_support_uncertain`

Implementation:

- [score_audit_subset_rules.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/score_audit_subset_rules.py)

## Why Add The Fourth Bucket

The held-out test showed a reasonable but important failure mode:

- some claims were not actually messy or contradicted
- they simply had too little support to be confidently called clean

Examples:

- `Cd`, `Cr`, `Ni`, `Pb` in `006_Azam_2013`
- `Al` in `058_ONeill_1987`

These are weak-support rare-element claims, not obvious alignment failures.

## Results On Original Audit Subset

No change in the original benchmark subset:

- `clean_support -> clean_support`: 11
- `likely_alignment_or_structure_problem -> alignment_or_structure_problem`: 92
- `likely_extraction_coverage_problem -> extraction_coverage_problem`: 10

So the 4-way extension does not destabilize the subset it was built from.

## Results On Held-Out Subset

Held-out summary:

- `clean_support -> clean_support`: 28
- `clean_support -> low_support_uncertain`: 6
- `likely_alignment_or_structure_problem -> alignment_or_structure_problem`: 33
- `unclear -> alignment_or_structure_problem`: 7

This is better than the earlier 3-way version, which had forced those 6 clean-but-weakly-supported claims into the bad bucket.

## Interpretation

The current pipeline now supports a more realistic decision policy:

- `clean_support`: strong enough to trust automatically
- `alignment_or_structure_problem`: likely semantic / matching / design problem
- `extraction_coverage_problem`: likely figure / timepoint / inaccessible-data limitation
- `low_support_uncertain`: not enough evidence to trust, but not clearly a bad extraction either

This is a much more defensible practical outcome than trying to force all claims into only clean vs bad.

## Best Next Step

The next meaningful refinement is to make the fourth bucket explicit in the broader project language:

- not all uncertain claims are hallucinations
- some are simply under-supported

That distinction should carry through into:

- the scoring logic
- the paper framing
- any eventual pipeline QC / adjudication policy
