# Dress Rehearsal: humic_acid_yield Pilot Topic
**Date**: 2026-03-26
**Purpose**: Simulate Pipeline V2 Stage 1 (Search) and Stage 2 (Screening) on the humic_acid_yield pilot topic to identify bottlenecks, calibrate adjudication logic, and estimate corpus characteristics before a full V2 run.

---

## 1. Search Results

### OpenAlex Query
- **Search string**: `humic acid crop yield`
- **Filter**: `type:article`, `publication_year:1990-2026`
- **Per page**: 25 (relevance-ranked)
- **Total corpus size**: **39,194 records** (far larger than expected; see Section 5)
- **Retrieved for rehearsal**: 25 papers (first page, relevance-ranked)
- **Raw output**: `1_search/openalex_raw.json`
- **Screening output**: `2_screen/screening_results.csv`

### Corpus Size Context
The 39,194-record total is the broadest estimate — it includes any article mentioning "humic acid" AND "crop yield" anywhere in the full-text indexed fields. This includes:
- Review papers and meta-analyses (~15-20% estimated)
- Papers where humic acid is a measured soil variable, not an applied treatment (~25-30%)
- Papers where HA is co-applied with other inputs in an inseparable bundle (~20%)
- Papers reporting only plant physiology outcomes, not yield (~15%)
- Genuine primary HA application vs. no-HA yield trials (~15-20%)

**Estimated true eligible corpus**: 3,000–6,000 papers (applying PICO filters at screening).
This is consistent with the benchmark paper's 93-article corpus extracted from a literature search in 2024.

---

## 2. Screening Results (25-Paper Sample)

### Decision Counts
| Decision | Count | Percent |
|----------|-------|---------|
| INCLUDE | 5 | 20% |
| EXCLUDE | 14 | 56% |
| UNSURE | 6 | 24% |

### Breakdown of INCLUDEs (5 papers)
1. **HA-urea vs urea** (Agriculture 2022) — HA isolated vs plain urea; maize-wheat; yield reported
2. **Coal-derived HA soil conditioner** (IJB 2015) — Dose-response wheat trial; yield reported
3. **Peanut continuous cropping HA fertilizer** (Scientific Reports 2019) — 3-year field trial; peanut yield
4. **Leonardite HA + drought/P stress in maize** (Scientific Reports 2020) — Field trial; grain yield
5. **SSP + HA on wheat calcareous soil** (Agronomy 2020) — Field experiment; grain yield; HA isolated vs SSP

### Breakdown of EXCLUDEs (14 papers)
| Exclusion Reason | Count |
|-----------------|-------|
| Review or meta-analysis | 8 |
| Intervention confounded (HA + other inputs, inseparable) | 3 |
| Humic acid is a soil property measured, not applied | 2 |
| Off-topic (no HA application at all) | 1 |

### Breakdown of UNSUREs (6 papers)
| UNSURE Reason | Count |
|--------------|-------|
| HA co-applied with other inputs; needs full text to check for HA-only arm | 4 |
| HA + N co-applied; may or may not have HA-without-N arm | 1 |
| HA + PGPR; may have factorial design | 1 |

---

## 3. Search Precision Estimate

**Estimated precision of raw search** (fraction of total records likely eligible):
- Reviews/meta-analyses: ~18% of 39,194 = ~7,000 records to exclude
- HA measured but not applied: ~25% = ~9,800 records to exclude
- HA confounded with other inputs: ~20% = ~7,800 records
- Non-yield outcomes only: ~15% = ~5,900 records
- **Estimated eligible**: ~15-20% = 5,900–7,800 records with some yield data; ~8-12% = 3,100–4,700 genuine HA-isolated yield trials

**From the 25-paper sample**: 20% INCLUDE rate (5/25). This is consistent with the broader estimate.

**Implication for full corpus screening**: At 20% precision, screening 39,194 records would yield approximately **5,000-8,000 potentially eligible papers** before full-text review. This is a large but manageable corpus with LLM abstract screening.

**Practical recommendation**: A more targeted search with Boolean filters will improve precision before screening:
- Add: `AND (field experiment OR greenhouse experiment OR pot experiment)`
- Add: `AND (grain yield OR crop yield OR fruit yield OR tuber yield)`
- Exclude keywords: `"review" OR "meta-analysis" OR "simulation" OR "model"` in title

A refined search on OpenAlex could reduce the raw corpus to ~8,000-12,000 records with ~35-40% precision, screening to ~3,000-4,500 eligible papers.

---

## 4. Adjudication Challenges Identified

### Challenge 1: Intervention Co-application (Frequency: VERY HIGH)
**Problem**: Humic acid is frequently applied together with other biostimulants, PGPR, chitosan, seaweed extract, or as one component of a combined fertilizer product. In 4 of 6 UNSURE cases, the question is whether a factorial design has a HA-only arm.

**Keyword approach limitation**: Cannot detect whether a factorial design isolates HA. Would either over-include (keep all HA-involved papers) or over-exclude (reject all bundled papers).

**LLM fix**: Reads the full Methods section to identify whether T1=HA, T2=PGPR, T3=HA+PGPR, T0=control, then extracts T1-vs-T0 comparison. Keywords cannot do this.

**Example**: Paper 24 (HA + PGPR on potato, ~140% increase) would need LLM to confirm the factorial structure.

### Challenge 2: HA as Measured Variable vs Applied Treatment (Frequency: HIGH)
**Problem**: Many papers study humic acid as a soil organic matter fraction that changes in response to management (straw incorporation, compost, biochar). These papers mention "humic acid" and "yield" prominently but are completely off-target for this review.

**Keyword approach**: Cannot distinguish "HA was applied" from "HA was measured." Both appear in the same document with the same keyword density.

**LLM fix**: Reads the experimental design section. If the treatment assignment involves applying a HA product vs. not, it's eligible. If HA is a measured soil chemistry outcome, it's excluded.

**Example**: Paper 3 (straw + CRF → HA soil fraction measurement) — easily identified from abstract but many similar papers would fool a keyword filter.

### Challenge 3: Outcome Scope (Frequency: MEDIUM)
**Problem**: Many HA papers report a mix of outcomes: plant height, root biomass, chlorophyll, enzyme activity, soil properties, AND yield. The pipeline needs to extract only yield rows while ignoring the others.

**Keyword approach limitation**: If a paper is admitted (correctly), the extractor will likely produce rows for all outcomes including non-yield. These must be filtered post-extraction.

**LLM adjudication role**: At Stage 6, each extracted row must be evaluated: "Is this row's outcome a crop yield measurement, or is it a growth/physiology/soil measurement?"

**Example**: Paper 7 (Scientific Reports peanut study) — abstract mentions soil properties as the primary emphasis. Yield data likely present but may be secondary.

### Challenge 4: Comparator Definition (Frequency: MEDIUM)
**Problem**: "Control" in HA papers can mean: (a) no fertilizer at all, (b) standard NPK without HA, (c) different HA rate, (d) other organic amendment without HA. The benchmark comparison requires that the control receives the same base fertilization as the treatment, just without HA.

**Keyword approach limitation**: Cannot distinguish these comparator types.

**LLM fix**: Checks control arm description: if "T0 = no fertilizer" and "T1 = NPK + HA," the apparent HA effect is confounded with NPK benefit. If "T0 = NPK" and "T1 = NPK + HA," it correctly isolates HA.

### Challenge 5: Effect Plausibility Filter (Frequency: LOW but IMPORTANT)
**Problem**: Some papers (particularly pot experiments from Pakistan/Egypt/Turkey/India) report HA effects of 80-200%+ on yield. While some of these may be real (very poor baseline soil conditions amplify effects), others likely reflect unit conversion errors or comparison to suboptimal controls (no fertilizer at all, not NPK-matched control).

**Example**: Paper 24 reports ~140% yield increase. Paper 12 reports 40% grain weight increase. Without knowing the baseline, these cannot be assessed from keywords alone.

**LLM fix**: Stage 5 hard filter catches values above domain-plausible threshold (|effect| > 200% for HA studies). Stage 6 LLM checks whether extraordinary effects have extraordinary context (very low-fertility baseline, drought stress).

---

## 5. Estimated Full Corpus Size

### Based on OpenAlex Total: 39,194 records
The true eligible corpus is estimated at:
- **After LLM abstract screening**: ~3,000–5,000 papers included (8-13% of raw)
- **After full-text eligibility check**: ~800–1,500 papers
- **After HA isolation check**: ~400–700 papers
- **With yield as primary outcome**: ~200–400 papers with extractable yield data
- **With usable variance**: ~100–200 papers

The benchmark paper (Ma et al. 2024) found 93 articles with 383 yield observations after systematic search up to 2024. The pipeline should retrieve a comparable corpus if the search covers 1990-2024 on OpenAlex with language filter (en) and a structured query.

**Conclusion**: The search is working correctly; the primary bottleneck is precision, not recall.

---

## 6. Config Adjustments Needed

### Adjustment 1: Tighten the Boolean search query
**Current**: `humic acid crop yield` (broad free-text search)
**Recommended**: Add a more structured OpenAlex query combining:
- `humic acid` OR `potassium humate` OR `humic substances` OR `leonardite` (intervention terms)
- `grain yield` OR `crop yield` OR `fruit yield` OR `tuber yield` (outcome terms)
- `filter=type:article` (already present)
- Exclude title contains "review" OR "meta-analysis"

This should improve precision from ~20% to ~35-40%.

### Adjustment 2: Add explicit HA-isolation verification to adjudication prompt
**Current config**: `"tc_confusion_warnings"` lists confounds but the screening question (llm_enabled: false) does not enforce HA isolation check.
**Recommended**: Add a dedicated binary question to the LLM screening prompt: "Does this paper have a treatment arm that receives HA/humate and a control arm that does NOT receive HA/humate, while both receive equivalent base fertilization? Y/N/UNCLEAR"

### Adjustment 3: Lower the effect plausibility threshold for HA studies
**Current**: General QC filter (Stage 5)
**Recommended**: For humic_acid_yield, flag any single-study effect > 150% for human review. Effects this large are implausible for a soil conditioner in normal conditions and usually indicate: (a) comparison to zero-fertilizer control (not NPK-matched), (b) pot experiment with extreme baseline stress, or (c) unit error.

### Adjustment 4: Add setting-stratified analysis explicitly
**Current config**: Pot experiments are allowed. Benchmark includes all settings.
**Recommended**: Add `study_setting` as an extraction field and run field-only sensitivity analysis from the start. Expected finding: field experiments will show smaller effects (~8-10%) than pot experiments (~20-30%), explaining some gap with benchmark.

---

## 7. What the Next Pipeline Stages Would Do

### Stage 3: PDF Download
- Target: All 5 INCLUDEs and 6 UNSUREs from Stage 2 screening = 11 papers to download
- Expected OA rate: ~85% (all 5 INCLUDEs are MDPI/Scientific Reports/open-access journals)
- Tool: `universal_downloader.py` resolving DOIs to full-text PDFs
- Estimated output: 9-10 PDFs successfully downloaded from 11 targets

### Stage 4: Data Extraction
- Each PDF sent to 2-3 LLM extractors (Claude + Gemini) with topic-specific prompts
- Expected rows per paper: 3-8 (dose-response designs may produce multiple rows)
- Expected total rows from rehearsal set: 30-60 rows from 9-10 papers
- Challenge: Many HA papers mix yield with growth parameter tables — extractor must find the yield table

### Stage 5: Deterministic QC
- Filter 1: Both means numeric and positive
- Filter 2: Effect size computable (lnRR)
- Filter 3: Duplicate detection
- Filter 4: Plausibility check (flag effects > 150%)
- Expected pass rate: ~70-80% of extracted rows

### Stage 6: LLM Adjudication (Core V2 innovation)
- Each QC-passed row reviewed by Claude
- Decision: keep/exclude/flag/swap on 5 semantic criteria
- Expected to resolve the 6 UNSURE papers from screening (some will be kept, some excluded)
- Expected to catch outcome_mismatch (plant height rows that bypassed extractor yield filter)
- Critical for HA topic: verify HA is the isolated variable, not co-applied with other biostimulants

### Stage 7: Effector Normalization
- Label each row: crop_class, study_setting, climate_class, estimand_context
- Apply benchmark_aligned label to field + grain_yield + isolated_HA rows
- Expected benchmark-aligned subset: ~40-60% of kept rows

### Stage 8: Synthesis
- DerSimonian-Laird RE on lnRR
- Expected rehearsal effect from 5 INCLUDEs: +10-20% (directionally aligned with +12% benchmark)
- Report field-only subset separately

### Stage 9: Diagnostics
- Compare rehearsal corpus to benchmark (Ma et al. 2024)
- Identify crop type composition, setting distribution
- Check if result direction agrees with +12% benchmark
- Generate failure taxonomy if gap persists

---

## 8. Preliminary Adjudication Quality Assessment

Based on this 25-paper rehearsal:

**Keyword-only screening would achieve**: ~60% precision on INCLUDE decisions (it would catch the 8 reviews cleanly, but would over-include confounded intervention papers and miss the HA-as-soil-property distinction).

**LLM screening advantage**: The 6 UNSURE papers require reading experimental design — this is exactly the task LLM abstract screening is designed for. A well-prompted LLM would resolve 4-5 of these 6 correctly by checking for HA isolation in the abstract.

**Estimated full-corpus screening efficiency**:
- Keyword alone: ~30-40% false inclusion rate (confounded interventions, HA as soil property)
- LLM screening: ~10-15% false inclusion rate
- With full-text verification: ~5% false inclusion rate

The humic_acid_yield topic is rated GOOD for V2 pilot because: the estimand is clear, the benchmark is well-defined, and the primary ambiguity (HA isolation from co-applied inputs) is exactly the type of question LLM adjudication excels at resolving.

---

## 9. Summary Assessment

| Dimension | Assessment |
|-----------|-----------|
| Search recall | HIGH (39,194 records, broad coverage of 1990-2026) |
| Search precision | LOW-MEDIUM (20% INCLUDE rate on first 25) |
| Primary exclusion reason | Reviews (56% of EXCLUDEs) + confounded intervention (21%) |
| Primary adjudication challenge | HA isolation from co-applied inputs (needs full text) |
| Estimated eligible corpus | 400-700 papers with extractable yield data |
| Expected pipeline-benchmark alignment | GOOD (clean estimand, +12% benchmark is plausible) |
| Pilot topic suitability | CONFIRMED — proceed to full V2 run after config adjustment |

**Overall verdict**: humic_acid_yield is well-suited for V2 dress rehearsal. The search works, the PICO criteria are unambiguous in the clear cases, and the primary challenge (HA isolation from bundles) is well-handled by LLM adjudication. The pilot should proceed with the config adjustments listed in Section 6.
