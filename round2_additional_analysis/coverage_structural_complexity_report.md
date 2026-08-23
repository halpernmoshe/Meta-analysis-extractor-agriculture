# Exploratory structural-complexity analysis of coverage

Matched status was reconstructed using the exact outcome-blind structural keys and dataset-specific pooling rules used for manuscript Table 2. Outcome values were not used to form matches. Complexity was not scored. The analysis used transparent structural-burden proxies available in the published-reference key records.

| Dataset | All reference cells | Same-paper AI records present | Same-paper AI records absent | Matched cells | Overall coverage | Coverage when same-paper AI records present |
|---|---:|---:|---:|---:|---:|---:|
| Boldorini et al. 2024 | 47 | 46 | 1 | 9 | 19% | 20% |
| Li X et al. 2024 | 517 | 517 | 0 | 204 | 39% | 39% |
| Hui et al. 2025 | 36 | 34 | 2 | 33 | 92% | 97% |
| Loladze 2014 | 605 | 558 | 47 | 177 | 29% | 32% |
| Li J et al. 2022 | 172 | 35 | 137 | 35 | 20% | 100% |

Same-paper presence means that the final published crosswalk identified at least one AI record from that paper. Absence can reflect a paper outside the processed corpus or an unresolved paper identifier; it is not classified as extraction difficulty.

The structural-burden comparison below is restricted to reference cells from papers with same-paper AI records. The unmatched column therefore excludes cells from absent or unresolved papers.

| Dataset | Same-paper reference cells | Matched | Unmatched | Within-paper coverage | Cells/paper, matched | Cells/paper, unmatched | Rows/cell, matched | Rows/cell, unmatched | Multirow cells, matched | Multirow cells, unmatched |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Boldorini et al. 2024 | 46 | 9 | 37 | 20% | 2.00 | 4.00 | 1.00 | 1.00 | 0.0% | 0.0% |
| Li X et al. 2024 | 517 | 204 | 313 | 39% | 16.00 | 27.00 | 1.00 | 1.00 | 2.9% | 2.6% |
| Hui et al. 2025 | 34 | 33 | 1 | 97% | 2.00 | 2.00 | 6.00 | 1.00 | 93.9% | 0.0% |
| Loladze 2014 | 558 | 177 | 381 | 32% | 34.00 | 22.00 | 1.00 | 1.00 | 14.7% | 5.2% |
| Li J et al. 2022 | 35 | 35 | 0 | 100% | 1.00 | NA | 6.00 | NA | 94.3% | NA |

Values are medians unless shown as percentages. A multirow cell contains more than one original published-reference row under the final comparison key. This multiplicity does not imply that the reason for aggregation was documented.

## Paper-level descriptive associations

| Dataset | Papers | Spearman rho: reference cells vs match rate | Spearman rho: reference rows vs match rate |
|---|---:|---:|---:|
| Boldorini et al. 2024 | 18 | 0.01 | 0.01 |
| Li X et al. 2024 | 27 | -0.15 | -0.16 |
| Hui et al. 2025 | 18 | -0.05 | -0.30 |
| Loladze 2014 | 41 | -0.03 | 0.02 |
| Li J et al. 2022 | 35 | NA | NA |

These correlations are restricted to papers with same-paper AI records and are descriptive only. Papers, cells, and structural slots differ across datasets; no pooled test, significance test, or composite difficulty score was used. Reporting quality and extraction uncertainty are not consistently encoded in the published-reference datasets and are therefore not tested here.
