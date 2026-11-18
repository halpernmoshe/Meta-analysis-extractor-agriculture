# Stage 3: evaluation

These scripts read the key tables in `../2_conversion/keys/` and print their results. They use only
the Python standard library, except `make_bland_altman.py`, which also draws Figure S1 with
matplotlib. They contain no random elements. Run them from any folder, for example:

```bash
python 3_evaluation/scope_aware_paired_tost.py
```

| Script | Manuscript item | Expected output (`expected_outputs/`) |
|---|---|---|
| `line_by_line_scope_aware.py` | Table 3: reading fidelity on matched cells | `EXPECTED_OUTPUT_LINEBYLINE.txt` |
| `scope_aware_paired_tost.py` | Table 4: paired equivalence | `EXPECTED_OUTPUT_PAIRED_TOST.txt` |
| `biochar_native_control_tost.py` | biochar native-control comparison reported with Table 4 | `EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt` |
| `scope_aware_aggregate_tost.py` | secondary unpaired comparison | `EXPECTED_OUTPUT_AGGREGATE_TOST.txt` |
| `reconciliation_analysis.py` | Table 5, Figure 3 values, Loladze per-element and no-selection results, power decomposition, Li J units example (Supplementary S1, S3 and the biochar and Li J parts of S2) | `EXPECTED_OUTPUT_RECONCILIATION.txt` |
| `make_bland_altman.py` | Supplementary S5: Table S1; writes Figure S1 to `../4_figures/figures/` | `EXPECTED_OUTPUT_BLAND_ALTMAN.txt` |
| `round2_additional_analysis/coverage_structural_complexity.py` | coverage results: paper-level burden and figure versus table sources | `EXPECTED_OUTPUT_COVERAGE.txt` |

`python verify.py` (repository root) runs every script and compares its printed output with the
expected file as text, after normalising line endings.

## What the scripts do

**Pairing.** AI and reference rows are paired by categorical key fields only. Outcome values are
never used to decide a match. Where several rows share a key, the cell value is their mean. The key
fields differ by dataset and by analysis; they are set in each script (`keyfn` or equivalent).

**Exclusions.** Papers listed in `../0_sources/corpus_mislabels_D2.csv` are removed by the
`EXCLUDE` sets at the top of each script.

**Reading fidelity** (`line_by_line_scope_aware.py`) reports, per dataset, the number of matched
cells, coverage, Pearson r and mean absolute error in percentage points. Where the reference stores
raw means, r uses the raw treatment means and the mean absolute error uses the percentage-change
effects derived from them. Where the reference stores effects, both use effects.

**Paired equivalence** (`scope_aware_paired_tost.py`). Each matched cell gives an AI-minus-reference
difference in mean log response ratio. The differences are averaged within each study, and the
per-study values are averaged across studies. The standard error is the sample standard deviation of
the per-study values divided by the square root of the number of studies. The 90% interval is the
mean plus or minus 1.645 standard errors. A dataset passes at margin m (5, 10, 15 or 20%) when the
interval lies inside plus or minus m times the absolute reference effect (the reference log response
ratio averaged within and then across studies). All of
this is computed on the log scale. The difference and interval are displayed as relative
response-ratio differences, 100 × (exp(d) − 1). They are not percentage-point differences between
the displayed AI and reference effects.

**Unpaired comparison** (`scope_aware_aggregate_tost.py`) restricts both sides to papers that both
sides contain, removes AI rows outside the reference dataset's scope, and averages each side's log
response ratios within and then across studies. The interval for the difference uses the two
standard errors combined (square root of the sum of squares) and 1.645. The same margins apply.

**Biochar control definition.** The AI records use a matched control (the co-amendment background
held fixed). The reference uses an absolute control. For Table 4, and for the biochar effects in
Table 3, both sides' effects are expressed against the absolute control. Each side's treatment mean
is divided by that side's own control mean from the rows with co-amendment level zero for the same
paper, crop and time point; on the AI side these control means were already in the extraction. The
first zero-co-amendment row for the same paper, crop and time point is used as the baseline. This
step was informed by the reference definition.
`biochar_native_control_tost.py` repeats the paired test with each side's own control (204 cells,
16 studies), and `reconciliation_analysis.py` block (A) reports the AI native-control effect on the
same 152 cells used in Table 4.

**Boldorini comparison units.** The raw-mean fidelity analysis and Figure S1 use paper, outcome,
crop, treatment level, co-amendment, co-amendment level, time point and unit (9 matched cells). The
paired test uses paper, outcome, crop and treatment level (16 matched cells). Both keys are
outcome-blind. The 9-cell result does not identify the source of the 16-cell difference.

## Round-2 additional analysis

`round2_additional_analysis/coverage_structural_complexity.py` rebuilds matched status with the same
keys and pooling rules, and describes coverage against structural features of the reference records
and against figure versus table sources. It writes these files beside itself, and prints the report:

- `coverage_structural_complexity_report.md` (the printed report)
- `coverage_structural_complexity_summary.csv`, `coverage_structural_complexity_cells.csv`,
  `coverage_structural_complexity_papers.csv`, `coverage_structural_complexity_associations.csv`,
  `coverage_source_format_summary.csv`

The analysis is descriptive and was added post hoc during the second revision.
