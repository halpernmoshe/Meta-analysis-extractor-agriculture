# LLM Adjudication Summary: organic_yield_gap

**Date:** 2026-03-26  |  **Total rows:** 590

## Decision Counts

| Decision | Count |
|----------|-------|
| exclude | 222 |
| flag | 53 |
| keep | 315 |
| swap | 0 |

## Effect Sizes

| Metric | Value |
|--------|-------|
| LLM kept rows | 315 |
| LLM mean effect (unweighted) | -12.0% |
| Keyword adjudicator effect | -4.9% |
| Benchmark (Ponisio et al. 2015) | -19.2% |

## Top Exclusion Reasons (LLM)

| Reason | Count |
|--------|-------|
| not_organic | 86 |
| missing_means | 77 |
| non_yield_outcome | 58 |
| comparator_unclear | 27 |
| straw_yield | 16 |
| per_plant | 9 |
| possible_swap_large_positive | 1 |
| yield_component | 1 |

## Disagreements with Keyword Adjudicator

- **Total disagreements:** 288
- **LLM excludes / KW kept:** 84
- **LLM keeps / KW excluded:** 120
- **LLM flags / KW kept:** 11

### Top LLM-excludes-KW-kept cases

| row_id | outcome | reason |
|--------|---------|--------|
| 11 | Okra number of fruits per plant | non_yield_outcome |
| 12 | Cowpea number of pods per plant | non_yield_outcome |
| 277 | Cacao dry bean yield | not_organic |
| 278 | Cacao dry bean yield | not_organic |
| 279 | Plantain marketable yield (Musa x paradi | not_organic |
| 280 | Banana marketable yield (Musa x paradisi | not_organic |
| 281 | Coffee dry parchment yield (Coffea arabi | not_organic |
| 387 | Silage maize shoot biomass yield | not_organic |
| 388 | Silage maize shoot biomass yield | not_organic |
| 389 | Winter barley grain yield | not_organic |

### Top LLM-keeps-KW-excluded cases

| row_id | outcome | reason excluded by KW |
|--------|---------|----------------------|
| 69 | Human metabolizable energy (HME) - grain | KW: exclude |
| 70 | Human metabolizable energy (HME) - grain | KW: exclude |
| 71 | Human metabolizable energy (HME) - total | KW: exclude |
| 72 | Human metabolizable energy (HME) - total | KW: exclude |
| 73 | Human metabolizable energy (HME) - total | KW: exclude |