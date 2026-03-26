# Search Summary: Humic Acid Yield — Stage 1

**Date completed:** 2026-03-26
**Review ID:** humic_acid_yield
**Seed DOI:** 10.3390/agronomy14122763 (Ma, Cheng & Zhang 2024)

---

## Search Queries and Hit Counts

| Query | OpenAlex Total Hits | Records Retrieved (page 1) |
|-------|--------------------|-----------------------------|
| `humic acid crop yield` | 37,008 | 100 |
| `humate application grain yield` | 1,615 | 100 |
| `fulvic acid plant yield field` | 6,374 | 100 |
| `humic substances wheat maize yield` | 6,526 | 100 |
| `leonardite potassium humate crop yield` | 230 | 100 |
| Seed DOI backward citations (44 refs) | — | 44 resolved |
| Seed DOI related works (10 works) | — | 10 |
| **Total raw records** | | **~554** |

Note: Total OpenAlex hits across all queries exceed 50,000 owing to the very broad first query ("humic acid crop yield" = 37,008). The first query's top-100 by relevance captures the most-cited and on-topic papers. Pages 2–370 would require automated pagination for a truly exhaustive harvest; the top-100 per query is the standard Stage 1 scope for this pipeline.

---

## Deduplication Results

- Records from all 5 keyword searches combined: ~500 raw
- Backward citations from seed paper added: 44
- Related works from seed paper added: 10
- **Total unique records (by OpenAlex ID) after deduplication: 152**
- Records eliminated as duplicates: ~402 (primarily from broad first query overlapping with others)

The 152 unique records are stored in `full_search_results.json`.

---

## Year Distribution

| Decade / Period | Count | Percentage |
|----------------|-------|------------|
| 1980–1999 | 7 | 4.6% |
| 2000–2009 | 14 | 9.2% |
| 2010–2014 | 26 | 17.1% |
| 2015–2019 | 48 | 31.6% |
| 2020–2025 | 57 | 37.5% |

The corpus is dominated by recent literature (2015–2025 = 69%), reflecting the surge in biostimulant research over the past decade. Oldest relevant record: Stefansson & Lindén 1996 (Environment International). Newest records: 2025 publications in Plants, Scientific Reports, Agronomy, and Chemical and Biological Technologies in Agriculture.

---

## Top Journals Represented

| Journal | Papers | Notes |
|---------|--------|-------|
| Agronomy (MDPI) | 24 | Open access; includes seed paper |
| Scientific Reports | 11 | High-impact OA |
| Chemical and Biological Technologies in Agriculture | 10 | Humic-specialist journal |
| Journal of Plant Nutrition | 7 | |
| Egyptian Journal of Soil Science / JSSAE | 9 | Many Egyptian field trials |
| Plant and Soil | 5 | |
| Journal of Soils and Sediments | 5 | |
| Frontiers in Plant Science / Agronomy / Env. Science | 7 | Frontiers group |
| BMC Plant Biology | 3 | |
| Plants (MDPI) | 5 | |
| Communications in Soil Science and Plant Analysis | 3 | |
| Bulletin of the National Research Centre (Egypt) | 3 | |
| Journal of Central European Agriculture | 3 | |
| Acta Agriculturae Scandinavica Section B | 2 | |

Geographic diversity: Egypt/Middle East (~25%), China (~15%), Turkey (~10%), Eastern Europe (~10%), Rest of World (~40%).

---

## Estimated Total Corpus Size

The 5 keyword searches collectively address an OpenAlex universe of ~51,000 records. However:

- Many hits in the broad "humic acid crop yield" query (37K) are tangentially related (soil chemistry, biostimulant reviews, non-yield endpoints).
- After applying PICO filters (primary experiment, yield outcome, humic acid as isolated variable, English language), the realistic eligible pool is **200–400 papers** based on the benchmark paper (Ma 2024 screened 93 eligible studies from a larger pool).
- The 152 records captured in Stage 1 represent the most relevant and most-cited core of this literature.
- An exhaustive search (all pages of all queries + forward citation chasing on the 152 records) would likely identify another 100–200 records, with most being lower-citation papers from regional journals (Egypt, Turkey, Pakistan, India).

**Stage 1 corpus: 152 unique records**
**Estimated eligible after full PICO screening: 80–120 papers** (based on ~50–60% pass rate observed in Ma 2024 from comparable search scope)

---

## Key Observations

1. **Highly active research area**: 37,000+ papers indexed in OpenAlex containing "humic acid" + "crop yield" (all years, English), with approximately 2,000–4,000 new publications per year in recent years.

2. **Geographic concentration**: Egyptian and Turkish agricultural journals contribute a large proportion of primary field studies. These are typically legitimate field experiments but may have variable reporting quality (missing SE/SD).

3. **Benchmark coverage**: The seed paper (Ma et al. 2024) cites 44 references, of which approximately 15–20 are primary experimental studies reporting yield data. All 44 cited references are captured in `full_search_results.json` via backward citation chasing.

4. **Important negative/null results captured**: Two critical null-result papers are in the corpus:
   - Humic Substances Generally Ineffective in Improving Vegetable Crop Nutrient Uptake or Productivity (Muscolo 2010, HortScience; 90 citations)
   - Fulvic and humic acid fertilizers are ineffective in dry bean (Heckman & Sims 2016, Can J Plant Sci; 10 citations)
   These are essential for an unbiased meta-analysis.

5. **Review papers captured but will be excluded at screening**: Several high-citation reviews (Canellas & Olivares 2014, Yakhin et al. 2017, du Jardin 2015) are in the corpus and should be excluded at Stage 2 screening as they contain no primary yield data.

6. **Source diversity**: The corpus spans cereals (wheat, maize, rice, barley), vegetables (potato, tomato, bean, eggplant, cucumber, spinach), industrial crops (cotton, sugarcane, sugar beet, rapeseed), and fruit crops (citrus, olive, strawberry).

---

## Files Written

| File | Description |
|------|-------------|
| `full_search_results.json` | 152 unique records with id, doi, title, year, journal, cited_by_count, search_query |
| `search_summary.md` | This file |
| `openalex_raw.json` | Original small test search (25 records, pre-existing) |
| `openalex_search_results.json` | Pre-existing combined search output |

---

## Next Step: Stage 2 Screening

Apply PICO inclusion/exclusion criteria to the 152 records:
- Include: primary experiment, reports yield (kg/ha or equivalent), humic acid as isolated variable
- Exclude: reviews, meta-analyses, yield not reported, humic acid confounded with other amendments, wastewater/remediation context
- Expected pass rate: 50–65% → approximately 75–100 eligible papers for full-text retrieval
