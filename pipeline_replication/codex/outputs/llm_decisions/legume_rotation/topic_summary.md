# LLM Adjudication Summary: legume_rotation

**Date:** 2026-03-26  |  **Total rows:** 542

## Decision Counts

| Decision | Count |
|----------|-------|
| exclude | 171 |
| flag | 91 |
| keep | 280 |
| swap | 0 |

## Effect Sizes

| Metric | Value |
|--------|-------|
| LLM kept rows | 280 |
| LLM mean effect (unweighted) | 23.3% |
| Keyword adjudicator effect | 17.8% |
| Benchmark (Zhao et al. 2022) | 20.0% |

## Top Exclusion Reasons (LLM)

| Reason | Count |
|--------|-------|
| non_yield_outcome | 68 |
| no_rotation_signal | 64 |
| missing_means | 27 |
| intercropping_not_rotation | 12 |
| extreme_effect | 5 |
| straw_yield | 3 |

## Disagreements with Keyword Adjudicator

- **Total disagreements:** 218
- **LLM excludes / KW kept:** 97
- **LLM keeps / KW excluded:** 30
- **LLM flags / KW kept:** 43

### Top LLM-excludes-KW-kept cases

| row_id | outcome | reason |
|--------|---------|--------|
| 102 | proso millet grain yield | no_rotation_signal |
| 103 | proso millet grain yield | no_rotation_signal |
| 104 | proso millet grain yield | no_rotation_signal |
| 105 | proso millet grain yield | no_rotation_signal |
| 106 | proso millet grain yield | no_rotation_signal |
| 107 | proso millet grain yield | no_rotation_signal |
| 108 | proso millet grain yield | no_rotation_signal |
| 109 | proso millet grain yield | no_rotation_signal |
| 110 | proso millet grain yield | no_rotation_signal |
| 111 | proso millet grain yield | no_rotation_signal |

### Top LLM-keeps-KW-excluded cases

| row_id | outcome | reason excluded by KW |
|--------|---------|----------------------|
| 82 | Rainfall use efficiency (RUE) | KW: exclude |
| 83 | Rainfall use efficiency (RUE) | KW: exclude |
| 167 | Maize grain yield | KW: exclude |
| 168 | Maize grain yield | KW: exclude |
| 169 | Maize grain yield | KW: exclude |