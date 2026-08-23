# Round-2 additional analyses

This folder contains the post hoc coverage analysis added during the second revision:

- `coverage_structural_complexity.py` reconstructs matched coverage and the descriptive structural-burden/source-format comparison from the deposited key tables.

Run from the repository root:

```bash
python round2_additional_analysis/coverage_structural_complexity.py
```

The reported coverage analysis is fully reproducible from the deposited key tables. The finalized
JSON records that build the AI-side keys are in `source_records/`; source PDFs and superseded or
broader raw experimental archives are not included in this corrected release.
