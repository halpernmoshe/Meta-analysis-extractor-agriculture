# LLM Adjudication Summary: mycorrhiza_yield

**Date:** 2026-03-26  |  **Total rows:** 588

## Decision Counts

| Decision | Count |
|----------|-------|
| exclude | 264 |
| flag | 15 |
| keep | 309 |
| swap | 0 |

## Effect Sizes

| Metric | Value |
|--------|-------|
| LLM kept rows | 309 |
| LLM mean effect (unweighted) | 26.0% |
| Keyword adjudicator effect | 31.4% |
| Benchmark (Hoeksema et al. 2010) | 23.0% |

## Top Exclusion Reasons (LLM)

| Reason | Count |
|--------|-------|
| non_yield_outcome | 213 |
| not_amf | 31 |
| missing_means | 19 |
| extreme_effect | 15 |
| root_biomass_not_yield | 1 |

## Disagreements with Keyword Adjudicator

- **Total disagreements:** 194
- **LLM excludes / KW kept:** 47
- **LLM keeps / KW excluded:** 100
- **LLM flags / KW kept:** 6

### Top LLM-excludes-KW-kept cases

| row_id | outcome | reason |
|--------|---------|--------|
| 100 | Number of leaves | non_yield_outcome |
| 109 | Chlorophyll a content | non_yield_outcome |
| 110 | Chlorophyll b content | non_yield_outcome |
| 112 | Soil organic matter | non_yield_outcome |
| 116 | Soil pH | non_yield_outcome |
| 131 | Root fresh weight (yield) | non_yield_outcome |
| 132 | Root fresh weight (yield) | non_yield_outcome |
| 133 | Root fresh weight (yield) | non_yield_outcome |
| 134 | Root fresh weight (yield) | non_yield_outcome |
| 135 | Root fresh weight (yield) | non_yield_outcome |

### Top LLM-keeps-KW-excluded cases

| row_id | outcome | reason excluded by KW |
|--------|---------|----------------------|
| 0 | Leaflet area | KW: exclude |
| 1 | Leaflet area | KW: exclude |
| 2 | Leaflet thickness | KW: exclude |
| 3 | Leaflet thickness | KW: exclude |
| 4 | Palisade cell number per 100 μm leaflet  | KW: exclude |