# LLM Adjudication Summary: biochar_crop_yield

**Date:** 2026-03-26  |  **Total rows:** 629

## Decision Counts

| Decision | Count |
|----------|-------|
| exclude | 232 |
| flag | 58 |
| keep | 339 |
| swap | 0 |

## Effect Sizes

| Metric | Value |
|--------|-------|
| LLM kept rows | 339 |
| LLM mean effect (unweighted) | 13.8% |
| Keyword adjudicator effect | 7.3% |
| Benchmark (Ye et al. 2020) | 16.0% |

## Top Exclusion Reasons (LLM)

| Reason | Count |
|--------|-------|
| non_yield_outcome | 130 |
| intervention_mismatch | 67 |
| missing_or_zero_means | 35 |
| straw_or_biological_yield | 31 |
| per_plant_unit | 25 |
| extreme_effect | 2 |

## Disagreements with Keyword Adjudicator

- **Total disagreements:** 331
- **LLM excludes / KW kept:** 116
- **LLM keeps / KW excluded:** 145
- **LLM flags / KW kept:** 23

### Top LLM-excludes-KW-kept cases

| row_id | outcome | reason |
|--------|---------|--------|
| 13 | Wheat dry matter yield | intervention_mismatch |
| 14 | Wheat dry matter yield | intervention_mismatch |
| 15 | Wheat dry matter yield | intervention_mismatch |
| 16 | Wheat dry matter yield | intervention_mismatch |
| 17 | Wheat dry matter yield | intervention_mismatch |
| 18 | Wheat dry matter yield | intervention_mismatch |
| 19 | Wheat dry matter yield | intervention_mismatch |
| 20 | Wheat dry matter yield | intervention_mismatch |
| 21 | Plant N uptake | non_yield_outcome |
| 22 | Plant N uptake | non_yield_outcome |

### Top LLM-keeps-KW-excluded cases

| row_id | outcome | reason excluded by KW |
|--------|---------|----------------------|
| 133 | Lettuce shoot dry weight | KW: exclude |
| 145 | Corn grain yield | KW: exclude |
| 236 | Grain yield per hill | KW: exclude |
| 237 | Grain yield per hill | KW: exclude |
| 238 | Grain yield per hill | KW: exclude |