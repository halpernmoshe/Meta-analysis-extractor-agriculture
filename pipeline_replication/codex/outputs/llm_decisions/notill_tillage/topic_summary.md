# LLM Adjudication Summary: notill_tillage

**Date:** 2026-03-26  |  **Total rows:** 881

## Decision Counts

| Decision | Count |
|----------|-------|
| exclude | 252 |
| flag | 249 |
| keep | 380 |
| swap | 0 |

## Effect Sizes

| Metric | Value |
|--------|-------|
| LLM kept rows | 380 |
| LLM mean effect (unweighted) | 6.1% |
| Keyword adjudicator effect | 2.7% |
| Benchmark (Pittelkow et al. 2015) | -5.7% |

## Top Exclusion Reasons (LLM)

| Reason | Count |
|--------|-------|
| non_yield_outcome | 90 |
| reduced_till_not_notill | 87 |
| straw_yield | 73 |
| missing_means | 59 |
| not_notill | 10 |
| yield_component | 6 |
| extreme_effect | 4 |

## Disagreements with Keyword Adjudicator

- **Total disagreements:** 365
- **LLM excludes / KW kept:** 24
- **LLM keeps / KW excluded:** 90
- **LLM flags / KW kept:** 123

### Top LLM-excludes-KW-kept cases

| row_id | outcome | reason |
|--------|---------|--------|
| 375 | Rice effective panicle number (EP) | non_yield_outcome |
| 376 | Rice effective panicle number (EP) | non_yield_outcome |
| 377 | Rice effective panicle number (EP) | non_yield_outcome |
| 381 | Rice 1000-grain weight | non_yield_outcome |
| 382 | Rice 1000-grain weight | non_yield_outcome |
| 383 | Rice 1000-grain weight | non_yield_outcome |
| 387 | Wheat effective panicle number (EP) | non_yield_outcome |
| 388 | Wheat effective panicle number (EP) | non_yield_outcome |
| 389 | Wheat effective panicle number (EP) | non_yield_outcome |
| 393 | Wheat 1000-grain weight | non_yield_outcome |

### Top LLM-keeps-KW-excluded cases

| row_id | outcome | reason excluded by KW |
|--------|---------|----------------------|
| 69 | number of spikes per m2 | KW: exclude |
| 70 | number of spikes per m2 | KW: exclude |
| 71 | number of spikes per m2 | KW: exclude |
| 94 | Grain yield | KW: exclude |
| 95 | Grain yield | KW: exclude |