# Pipeline V2 Full Testing Plan
Date: 2026-03-26

---

## Revised Execution Order (2026-03-26)

REVISED PLAN (supersedes phases below where they conflict):

Phase A — V1 LLM Re-adjudication (ALL 6 existing extracted topics)
  Purpose: Extract all possible lessons before touching preregistered topics
  Topics: legume_rotation, mycorrhiza_yield, organic_yield_gap, notill_tillage, biochar_crop_yield, intercropping_yield
  Method: Claude Code reads existing extracted rows, makes semantic keep/exclude/flag/swap decisions
  Output: Full LLM decisions + comparison vs keyword + updated effect sizes
  No new paper reading. No new extraction. Fast.

Phase B — humic_acid_yield Pipeline Test (NOT preregistered)
  Purpose: Test full 9-stage pipeline on a clean new topic
  Results: Used to fix pipeline bugs only — NOT counted in P1/P2 success criteria
  Proceed only after Phase A lessons are incorporated into pipeline config

Phase C — Preregistered Topics (5 topics, confirmatory)
  amf_inoculation_yield, biochar_tropical_yield, elevated_co2_face_yield, legume_rotation (V2 rerun), cover_crop_corn_yield
  Proceed only after Phase B validates the pipeline
  Results count toward P1/P2 success criteria

---

## The V1 Lessons That Shape This Plan

Every major design decision in this testing plan is anchored to a specific V1 failure. This section provides the failure inventory that motivates the plan. It is not exhaustive — the full diagnostic record is in SPOT_CHECK_REPORT_2026-03-26.md, BENCHMARK_ALIGNED_ANALYSIS_2026-03-26.md, and AUDIT_REPORT_2026-03-26.md — but it names the failures that directly constrain how testing must be organized.

### V1 Failure 1: Wrong direction on notill_tillage (+1.2% vs benchmark -5.7%)

Two compounding causes: (a) AbdulsattarAlrijabo 2014 contributed rows with 144–609% effects that the numeric plausibility filter did not catch, because the threshold was too permissive; (b) the intervention definition was too broad, mixing strict zero-till with strip-till, reduced-till, and conservation agriculture. Keyword filters could not distinguish these practices.

**Implication for testing**: Every topic must have a topic-specific plausibility threshold verified during Phase 0. The adjudicator dry-run test must include at least one implausible row to verify it is flagged.

### V1 Failure 2: Semantic over-inclusion in organic_yield_gap and mycorrhiza_yield

The organic topic retained 197 rows via a category called "topic_exclude_outcome" that appears to have been a false-exclusion filter — meaning the filter incorrectly removed field-scale yield outcomes because their labels looked like quality traits or system metrics. The mycorrhiza topic excluded 112 rows as "outcome_mismatch" that were legitimate yield proxies (marketable fruit weight, tuber dry matter) but used non-standard language. Keyword matching cannot reliably handle outcome label heterogeneity.

**Implication for testing**: The LLM adjudicator dry-run test must include rows with non-standard yield labels (e.g., "marketable fresh weight per plant," "total harvestable biomass per area") to verify the LLM includes them rather than excluding them as non-yield outcomes.

### V1 Failure 3: Estimand mismatch in intercropping_yield (-3.1% vs benchmark +22.0%)

The benchmark measures system productivity (LER). The pipeline measured per-component crop yield. These are not the same quantity, so no amount of row-level filtering can produce a comparable estimate. The intercropping topic was removed from V2 for this reason. The V2 replacement topics (humic_acid, amf_inoculation, biochar_tropical) were scored specifically on estimand clarity.

**Implication for testing**: Benchmark spec completeness must be verified before running any topic. Each benchmark spec must include an explicit "known estimand traps" section. This is checked in Phase 0 against the frozen specs.

### V1 Failure 4: Benchmark-aligned subset filtering is a diagnostic, not a correction

Subset filtering moved the biochar estimate from +6.66% to +23.8% (helpful; benchmark is +16%). But it worsened notill (+1.2% → +6.7%) and tripled the mycorrhiza overshoot (+29.26% → +74.7%). The aligned-subset analysis is run unconditionally in Stage 9, but it is never the primary result. For four of six V1 topics, the root cause of divergence was upstream of filtering.

**Implication for testing**: The Phase 4 evaluation must use the preregistered primary analyses (direction agreement, CI overlap) as the confirmatory check. The benchmark-aligned subset is evaluated as a secondary diagnostic only. Testing protocols must not substitute aligned-subset results for primary results.

### V1 Failure 5: Keyword adjudication could not detect comparator identity, T/C swaps, or aggregation level

The keyword adjudicator correctly routed papers and excluded non-yield outcomes but failed systematically on: (a) verifying that the "control" arm was genuinely a no-treatment baseline rather than a different treatment variant; (b) detecting treatment/control arm reversals; (c) distinguishing per-plant from per-area yield. All three require reading the row in context, which requires LLM judgment.

**Implication for testing**: The LLM adjudicator dry-run test (Phase 0) must include test cases for each of the five failure categories: intervention isolation, outcome disambiguation, comparator identity, estimand verification, and plausibility with context.

### V1 Failure 6: The adjudicator was only dry-run tested before the architecture was frozen

The `adjudicate_llm_universal.py` script was tested in dry-run mode (no actual API calls), meaning the LLM's behavior on real rows was never observed before the architecture freeze. This creates risk: the live first run may reveal prompt failures, schema parsing errors, or unexpected LLM behavior. The audit report (AUDIT_REPORT_2026-03-26.md, Risk 4) flags this explicitly.

**Implication for testing**: Phase 1 (humic_acid_yield pilot) is explicitly the first live test of the LLM adjudicator. The pilot's purpose is precisely to catch prompt failures and schema issues before the full 6-topic run. Phase 1 results govern whether the full run proceeds.

### V1 Failure 7: API keys expired before the live run could proceed

Both Anthropic and Google API keys were out of credit or expired as of 2026-03-26. This is a purely operational blocker but it is the most urgent item on the path to any V2 results. It must be resolved before Phase 1 starts.

**Implication for testing**: Phase 0 includes API key validation as the first check. No subsequent phase begins until this is resolved.

### V1 Failure 8: The humic_acid_yield pilot had no end-to-end run before the architecture was frozen

The dress rehearsal covered Stages 1-2 (search and screen) only. Stages 3-9 were not run. The frozen architecture was written based on design intent, not observed behavior. This means the first live run of Stages 3-9 must be treated as a discovery exercise that may surface bugs, not as a confirmatory evaluation.

**Implication for testing**: Phase 1 is explicitly a go/no-go gate. Phase 2 does not begin until Phase 1 has produced a synthesized result that was inspected for obvious failures.

---

## Overview: 4 Phases

| Phase | Name | Topics | Purpose | Unblocks |
|-------|------|--------|---------|----------|
| 0 | Pre-flight Checks | All 6 | Verify everything works before first real run | Phase 1 |
| 1 | Pilot: humic_acid_yield | humic_acid_yield | Full end-to-end test; go/no-go gate | Phase 2 and 3 |
| 2 | Carried-Forward Topics | legume_rotation, elevated_co2_face_yield, cover_crop_corn_yield | Decide what to reuse vs. re-run; run V2 pipeline | Phase 4 |
| 3 | New Topics | amf_inoculation_yield, biochar_tropical_yield | Full pipeline from scratch | Phase 4 |
| 4 | Evaluation and Reporting | All 6 | Preregistered analyses; forest plots; deviation log | Final paper |

---

## Phase 0 — Pre-flight Checks

**Purpose**: Confirm that all preconditions for a valid V2 run are met before touching any topic data. Every item here is a blocker; nothing in Phase 1 starts until the pre-flight checklist is complete.

**Why this phase exists**: V1 lost time to silent failures that were only discovered during runs (expired API keys, malformed configs, missing benchmark numbers). Pre-flight catches these before they corrupt results.

### 0.1 API Key Validation

**What to check**: Both Anthropic and Google API keys are active and have sufficient credit.

**How to check**:
1. Run `adjudicate_llm_universal.py --provider anthropic --dry-run False --test-row` with a single synthetic row. Verify an actual API call is made and a structured JSON decision is returned.
2. Repeat with `--provider google`.
3. Run the extraction test call for at least one provider (Claude Sonnet or Gemini Pro) with a small text snippet.

**Success criteria**: Both dry-run=False API calls succeed without authentication errors. At least one adjudication call returns the frozen output schema with all 8 fields populated.

**Failure mode from V1**: Both API keys expired before the live run could proceed (AUDIT_REPORT Risk 1).

**V2 fix**: This check is mandatory before any run proceeds. If either key fails, stop and top up credit / regenerate key before continuing.

---

### 0.2 LLM Adjudicator Dry-Run on Synthetic Test Cases

**What to check**: The adjudicator correctly handles the five failure categories identified in V1 and returns the frozen output schema.

**How to prepare**: Create a file `codex/adjudication_test_cases.json` with at least one test row per failure category. Minimum required test cases:

| Test ID | Failure Category | Expected Decision | Test Row Content |
|---------|-----------------|-------------------|-----------------|
| TC-01 | Intervention isolation | exclude | HA + seaweed extract combined; cannot isolate HA effect |
| TC-02 | Outcome disambiguation | keep | "marketable fresh weight per plant" for a tomato experiment |
| TC-03 | Comparator identity | exclude | Control arm is zero-fertilizer, not NPK-matched no-HA |
| TC-04 | Estimand verification | exclude | Yield reported per plant, not per area |
| TC-05 | Plausibility check | flag | Treatment effect of +350% in pot study with zero-fertilizer control |
| TC-06 | T/C swap detection | swap_treatment_control | treatment_mean < control_mean by 40%; treatment described as "no HA" |
| TC-07 | Correct keep | keep | Clean HA application, NPK-matched control, grain yield per area |
| TC-08 | Implausible notill-style outlier | exclude | +600% yield increase from a tillage change (for notill topic test only) |

**How to run**: `python codex/adjudicate_llm_universal.py --provider anthropic --input codex/adjudication_test_cases.json --dry-run False`

**Success criteria**: TC-01 → exclude; TC-02 → keep; TC-03 → exclude; TC-04 → exclude; TC-05 → flag or exclude; TC-06 → swap_treatment_control; TC-07 → keep. All 8 output schema fields present in every response. No JSON parse failures.

**Failure mode from V1**: Adjudicator was only dry-run tested; live behavior on real rows was unknown before the freeze (AUDIT_REPORT Risk 4).

**V2 fix**: Live test on 8 synthetic cases with known expected decisions before any real data is processed.

---

### 0.3 Benchmark Spec Completeness Check

**What to check**: All 6 benchmark specs contain the required fields with specific numbers, not placeholders.

**How to check**: For each of the 6 topic directories, verify the benchmark spec contains:

| Required Field | Must Have |
|---------------|-----------|
| Benchmark paper citation | Author(s), Year, Journal, DOI |
| Published pooled effect | Exact number with units (e.g., "+12% crop yield increase") |
| Effect CI or k | Either 95% CI or number of studies/observations |
| Intervention definition | What counts as the treatment; what is excluded |
| Comparator definition | What counts as the control |
| Primary outcome definition | What outcome qualifies; what does not |
| Study setting restrictions | Field only? Any? Pot included? |
| At least one "known estimand trap" | Documented |

**Success criteria**: All 6 specs have all fields above. Any field marked "TBC" in the preregistration appendix (amf_inoculation DOI, biochar_tropical DOI, elevated_co2 DOI, cover_crop DOI) must be resolved with the exact value before continuing.

**Failure mode from V1**: Benchmark numbers were sometimes approximate or only described in notes; the audit report flagged that "exact benchmark values being targeted" should be locked before any V2 analysis (AUDIT_REPORT Risk 7).

**V2 fix**: No topic proceeds past Phase 0 until its benchmark spec has exact numbers.

---

### 0.4 Config Validation

**What to check**: Each of the 6 topic config.json files has all required fields from the PIPELINE_V2_ARCHITECTURE spec.

**Required fields per config**:
- `review_id` (string, unique)
- `pico.population.search_terms` (non-empty array)
- `pico.intervention.description` (non-empty string)
- `pico.comparator.description` (non-empty string)
- `pico.outcome.primary.description` (non-empty string)
- `tc_confusion_warnings` (array, may be empty)
- `extraction_priorities` (array, at least one entry)
- `benchmark.published_pooled_effect.estimate` (numeric, not null)
- `benchmark.published_pooled_effect.unit` (string)
- `search.date_range.start` and `search.date_range.end` (years as integers)

**How to run**: Write or use a validation script: `python codex/validate_configs.py --config-dir .` that checks each JSON against the required schema.

**Success criteria**: All 6 configs pass validation with no missing required fields. No config has a placeholder value in `benchmark.published_pooled_effect.estimate`.

---

### 0.5 Directory Structure Scaffolding

**What to check**: All 6 topic directories exist and contain the 9 stage subdirectories needed for a full run.

**Required structure for each topic**:
```
{topic}/
  1_search/
  2_screen/
  3_download/
  4_extract/
  5_qc/
  6_adjudication/
  7_effectors/
  8_synthesis/
  9_diagnostics/
  config.json
  benchmark_spec.md
```

**How to check**: Verify each directory exists and that config.json and benchmark_spec.md are present. Create missing stage directories.

**Success criteria**: All 6 topics have complete directory structure. No runs fail due to missing output directories.

---

### 0.6 Plausibility Filter Verification

**What to check**: The Stage 5 deterministic QC correctly flags rows with extreme effects per the frozen threshold (|lnRR| > 2.0, i.e., effect > +619% or < -86%), and that topic-specific overrides are configured where needed.

**Why**: The notill_tillage failure was partly caused by AbdulsattarAlrijabo 2014 contributing 144–609% effects that passed the plausibility filter (SPOT_CHECK_REPORT, Topic 4). The frozen spec sets |lnRR| > 2.0 as default but permits topic-specific overrides. The humic_acid topic has an explicit override (|effect| > 200%) to catch extreme pot-study values.

**How to check**: Construct 3 test rows for each of: (a) a value just below the default threshold (e.g., +600%), (b) a value above the default threshold (e.g., +700%), (c) a notill-style value (e.g., +200% that should pass default but would be caught by a tighter notill threshold). Run `qc_hard_filters.py` on these rows. Verify the outputs are correctly flagged.

**Success criteria**: Default threshold correctly flags |effect| > 619%. humic_acid override correctly flags |effect| > 200%. Rows at threshold boundary behave as expected.

---

### Phase 0 Completion Gate

Before Phase 1 begins, all six checks must pass:

- [ ] 0.1: Both API keys validated with live calls
- [ ] 0.2: All 8 adjudicator test cases return expected decisions
- [ ] 0.3: All 6 benchmark specs have exact numbers (no TBCs)
- [ ] 0.4: All 6 configs pass schema validation
- [ ] 0.5: All 6 topic directories have complete 9-stage structure
- [ ] 0.6: Plausibility filter correctly flags test rows

Record the date/time of completion in STATUS_LOG.md before proceeding.

---

## Phase 1 — Pilot: humic_acid_yield

**Purpose**: Run the full 9-stage pipeline on humic_acid_yield end-to-end. This topic was never run in V1 and was selected precisely because it is clean, well-scoped, and OA-rich. It serves as the go/no-go gate for the full V2 run.

**Why this topic first**: humic_acid was chosen as the pilot because: (a) the benchmark paper (Ma, Cheng & Zhang 2024) is fully OA (MDPI); (b) the estimand is clean (HA applied to soil or as foliar treatment vs. no-HA control); (c) the expected direction is unambiguous (+12%); (d) it was never tested in V1, so there are no V1 artifacts to confuse interpretation of V2 results.

**Why it is a gate**: The AUDIT_REPORT (Section 5, Risk 3) notes: "If the pilot reveals extraction problems... the architecture may need adjustment — but it is now frozen." A pilot run that reveals systematic extraction failures or prompt failures requires a deviation log entry before proceeding. The testing plan must be able to distinguish "pilot revealed fixable implementation bugs" from "pilot revealed structural failures requiring deviation."

---

### Stage 1 — Literature Search

**What it does**: Queries OpenAlex API using config search terms; retrieves up to 40,000 records; deduplicates by DOI.

**How to run**: `python pipeline_replication/1_search.py --config humic_acid_yield/config.json --output humic_acid_yield/1_search/`

**Expected output**: `humic_acid_yield/1_search/results.json` with query metadata and raw results. The dress rehearsal retrieved 25 results from a total OpenAlex corpus of 37,008–39,194 hits. A full retrieval with pagination should return thousands of records.

**Success criteria**:
- results.json exists and is valid JSON
- Record count > 500 (dress rehearsal confirmed 37K+ total hits in OpenAlex)
- Every record has doi, title, abstract_inverted_index, and publication_year fields (even if some are null)
- Query metadata logged (search terms, date, total_hits)

**V1 failure addressed**: No specific V1 failure at this stage. The universal downloader was identified as a retained strength.

---

### Stage 2 — Abstract Screening

**What it does**: LLM screens each title+abstract against PICO criteria; returns INCLUDE / EXCLUDE / UNSURE per record.

**How to run**: `python pipeline_replication/2_screen.py --config humic_acid_yield/config.json --input humic_acid_yield/1_search/results.json --output humic_acid_yield/2_screen/`

**Expected output**: `humic_acid_yield/2_screen/screening_results.csv` with columns: doi, title, year, decision, reason, intervention_match, comparator_match, outcome_match.

**Success criteria**:
- All records from Stage 1 have a screening decision (no unscreened records)
- INCLUDE rate between 10-40% (dress rehearsal found 20% from first 25 results; full corpus may differ)
- UNSURE rate < 30% (high UNSURE rate suggests PICO definition needs tightening)
- No systematic screening of review articles as INCLUDE (dress rehearsal found 57% of initial results were reviews; they should be excluded)
- Dominant exclusion reasons logged and plausible (e.g., "HA measured, not applied"; "co-applied with other biostimulants"; "review article")

**Quality check to run after**:
1. Spot-check 10 random INCLUDE decisions: does each title+abstract describe a primary study that applied humic acid as a treatment and measured crop yield?
2. Spot-check 10 random EXCLUDE decisions: are any of these false exclusions (legitimate HA+yield studies excluded by overly tight screening)?
3. Count review articles in INCLUDE: should be near zero.

**V1 failure addressed**: In V1, screening was not checked for over-exclusion of yield proxies. Here, the spot-check at step 2 above specifically looks for false negatives.

---

### Stage 3 — PDF Download

**What it does**: Resolves DOIs from Stage 2 INCLUDE list to OA full-text PDFs; downloads from multiple sources with retry logic.

**How to run**: `python "claude universal metaanalysis pipeline/pdf_downloader.py" --input humic_acid_yield/2_screen/screening_results.csv --output humic_acid_yield/3_download/`

**Expected output**: `humic_acid_yield/3_download/` containing PDFs + `download_log.json` with per-paper success/failure status.

**Success criteria**:
- Download success rate >= 50% of INCLUDE papers (benchmark is OA-published MDPI; many cited papers should be OA)
- download_log.json records each paper's outcome (success/failure/error code)
- No crash without completing the download queue (the universal downloader has retry logic)

**Quality check to run after**:
1. Inspect download_log.json: what percentage of papers failed? What is the dominant failure source (publisher block, no OA link, DOI resolution failure)?
2. Open 3 random downloaded PDFs: do they correspond to the expected paper (title/author match)?
3. Check for zero-byte or corrupted PDFs in the download folder.

**V1 failure addressed**: The universal downloader is a retained V1 strength; no systematic failure is expected here. The check is for operational issues (expired links, rate limiting).

---

### Stage 4 — Data Extraction

**What it does**: Sends each downloaded PDF to multi-model extraction (Claude + Gemini); merges consensus outputs into summary.csv.

**How to run**: `python pipeline_replication/4_extract.py --config humic_acid_yield/config.json --input humic_acid_yield/3_download/ --output humic_acid_yield/4_extract/`

**Expected output**: `humic_acid_yield/4_extract/summary.csv` with one row per treatment-control comparison per paper, using the frozen extraction schema.

**Success criteria**:
- At least 60% of downloaded PDFs have at least one extracted row
- Every row in summary.csv has paper_id, treatment_mean, control_mean, outcome, and treatment_description fields (other fields may be null but these are required)
- No extraction model returns 0 rows across all papers (would indicate prompt failure)
- Consensus merge does not produce an error for any paper

**Quality check to run after**:
1. For 5 papers, manually open the PDF and verify the extracted treatment_mean and control_mean match the values visible in the primary table. This is an accuracy spot-check.
2. Check the distribution of `confidence` field: what percentage are high/medium/low? A majority of "low" confidence suggests the extraction prompt is struggling.
3. Check for rows where treatment_description and control_description are both null (a sign the consensus merge dropped descriptions).
4. Verify at least one row per paper has outcome containing a yield-related term (grain, yield, biomass, weight, t/ha, kg/ha).

**V1 failure addressed**: V1 sometimes extracted biomass instead of mineral concentrations because the extraction prompt did not clearly prioritize the target outcome. The humic_acid extraction prompt must explicitly prioritize crop yield (grain, harvestable, dry matter per area) over soil organic matter, root biomass, and plant height.

---

### Stage 5 — Deterministic QC

**What it does**: Applies hard mathematical filters to summary.csv; flags rows with missing means, extreme effects, duplicates, or non-independence; computes lnRR; outputs summary_qc.csv.

**How to run**: `python codex/qc_hard_filters.py --input humic_acid_yield/4_extract/summary.csv --output humic_acid_yield/5_qc/ --config humic_acid_yield/config.json`

**Expected output**: `humic_acid_yield/5_qc/summary_qc.csv` and `humic_acid_yield/5_qc/qc_audit.json`

**Success criteria**:
- qc_audit.json records count of rows excluded for each reason (missing means, extreme effect, duplicate, etc.)
- No rows with treatment_mean <= 0 or control_mean <= 0 pass through to adjudication (lnRR is undefined for zero/negative means)
- The humic_acid plausibility override is active: rows with |effect| > 200% are flagged
- Duplicate detection runs and reports any duplicates found
- The lnRR column is populated for all rows with both means present

**Quality check to run after**:
1. Inspect qc_audit.json: are the exclusion counts plausible? (e.g., "missing_means: 30" would be high for a clean topic)
2. Inspect the flagged extreme values: do any look like legitimate HA effects? (A +250% effect in a severely depleted soil might be real; the flag is for inspection, not automatic exclusion)
3. Verify no rows in summary_qc.csv have lnRR = NaN (indicates a calculation error)

**V1 failure addressed**: The notill plausibility filter was too permissive (passed 144-609% effects). The humic_acid config includes a tighter threshold (|effect| > 200%) verified in Phase 0. This stage check verifies it is applied correctly.

---

### Stage 6 — LLM Semantic Adjudication

**What it does**: Sends each QC-passed row to Claude with the topic brief; receives structured keep/exclude/flag/swap decisions; outputs adjudicated_kept.csv.

**How to run**: `python codex/adjudicate_llm_universal.py --provider anthropic --input humic_acid_yield/5_qc/summary_qc.csv --config humic_acid_yield/config.json --output humic_acid_yield/6_adjudication/`

**Expected output**: `humic_acid_yield/6_adjudication/adjudication_decisions.jsonl` and `humic_acid_yield/6_adjudication/adjudicated_kept.csv`

**Success criteria**:
- Every row in summary_qc.csv has a decision in adjudication_decisions.jsonl (no missing rows)
- All 8 fields of the frozen adjudication output schema are present in every decision record
- No JSON parse failures (the script should log parse failures; 0 failures expected)
- Retention rate between 30-80% (too low = over-filtering; too high = the adjudicator is not doing semantic work)
- At least some rows are excluded with exclusion_reason = "intervention bundled with other inputs" (this is the primary HA challenge identified in the dress rehearsal)
- At least some rows are kept with a clear rationale string

**Quality check to run after**:
1. Spot-check 10 excluded rows: do the rationale_short strings give specific, defensible reasons?
2. Spot-check 10 kept rows: do treatment_description and control_description fields confirm a real HA vs. no-HA contrast?
3. Check for any rows where needs_tc_swap = true: examine the original row to verify the swap is correct.
4. Count decisions by decision type: keep, exclude, flag, swap. Are the proportions sensible?

**V1 failure addressed**: This is the core V2 innovation. V1 used keywords that could not: (a) isolate HA from bundled biostimulant products, (b) verify that "foliar fresh weight" is a yield proxy, (c) identify zero-fertilizer controls, (d) detect T/C swaps, (e) flag implausible effects in context. All five must be demonstrated in the Stage 6 outputs.

---

### Stage 7 — Effector Normalization

**What it does**: Assigns canonical labels to kept rows: crop class, study setting, climate class, soil class, management class, and estimand context.

**How to run**: `python codex/normalize_effectors_universal.py --provider anthropic --input humic_acid_yield/6_adjudication/adjudicated_kept.csv --output humic_acid_yield/7_effectors/`

**Expected output**: `humic_acid_yield/7_effectors/effector_labels.jsonl`

**Success criteria**:
- All 6 effector fields present for every kept row
- Crop class distribution is plausible (grain cereals and vegetables likely dominant for HA studies)
- Study setting distribution shows both field and pot categories (both are present in the HA literature)
- At least some rows labeled `benchmark_aligned` (field + direct yield + isolated HA + NPK-matched control)

**Quality check to run after**:
1. Count rows by normalized_study_setting: field vs. pot vs. greenhouse. Is the breakdown consistent with what is known about the HA literature (mix of field and pot studies)?
2. Count rows by normalized_estimand_context: benchmark_aligned vs. partially_aligned vs. misaligned. What fraction is benchmark-aligned?
3. Verify that rows with source_type = "pot" are labeled normalized_study_setting = "pot" (consistency check between extraction and normalization).

---

### Stage 8 — Synthesis

**What it does**: Pools kept rows using DerSimonian-Laird random effects on lnRR; reports pooled effect, 95% CI, I², tau², k; compares to benchmark.

**How to run**: `python pipeline_replication/synthesize.py --input humic_acid_yield/6_adjudication/adjudicated_kept.csv --output humic_acid_yield/8_synthesis/synthesis_results.json --benchmark-effect 12.0 --benchmark-ci-lower 8.0 --benchmark-ci-upper 16.0`

**Expected output**: `humic_acid_yield/8_synthesis/synthesis_results.json` with pooled effect, CI, I², k, benchmark comparison.

**Success criteria**:
- k >= 10 (at least 10 independent effect sizes for pooling to be meaningful)
- Synthesis runs without error
- Pooled effect is expressed in percentage change (not lnRR raw value)
- 95% CI is reported
- Benchmark comparison fields are populated: direction_agrees (boolean), ci_includes_benchmark (boolean), absolute_gap_pp (float)

**Quality check to run after**:
1. Check k against the benchmark (benchmark used k=93 articles, 383 yield observations). The pipeline will have fewer due to OA limitations; k=20-60 would be acceptable.
2. Check the variance coverage rate: what percentage of pooled rows had usable variance (SD, SE, LSD, CI)? If < 30%, the weighted estimate may be driven by a small subset of rows.
3. Compare direction_agrees: expected TRUE (positive direction, matching +12% benchmark).
4. Note the absolute_gap_pp: expected target is < 10pp (i.e., pipeline estimate in the range of +2% to +22%).

---

### Stage 9 — Diagnostics

**What it does**: Runs leave-one-out analysis, benchmark-aligned subset, high-confidence-only subset, table-only subset, funnel plot, composition comparison, and failure taxonomy.

**How to run**: `python pipeline_replication/run_diagnostics.py --input humic_acid_yield/ --output humic_acid_yield/9_diagnostics/`

**Expected output**: `humic_acid_yield/9_diagnostics/diagnostics_report.md` and supporting figures.

**Success criteria**:
- All 7 automatic diagnostics run and produce outputs
- Funnel plot and Egger's test outputs exist (even if asymmetry is not significant)
- Composition comparison identifies crop class and setting distributions
- No diagnostic crashes

**Quality check to run after**:
1. Identify the most influential paper via LOO analysis: does removing any single paper change the direction of the pooled effect?
2. Compare benchmark-aligned subset to full estimate: does restricting to field + direct yield improve or worsen agreement with benchmark?
3. Inspect composition comparison: what crop classes dominate? Does the composition match the benchmark's stated scope?

---

### Phase 1 Go/No-Go Criteria

**Proceed to Phase 2 if ALL of these hold**:
- Stage 4: >= 60% of papers have extracted rows
- Stage 6: Adjudication runs without JSON parse failures; retention rate is 30-80%
- Stage 6: At least one HA-isolation exclusion is found (intervention bundled with other inputs)
- Stage 8: k >= 10 and pooled effect direction is positive
- Stage 8: absolute_gap_pp < 15pp (i.e., pipeline estimate between -3% and +27%)
- Stage 9: No single paper dominates to the point of changing direction

**Fix the pipeline first if ANY of these holds**:
- Stage 6 produces > 5% JSON parse failures
- Stage 6 retention rate < 15% (over-filtering) or > 90% (adjudicator not working)
- Stage 8 direction is negative (wrong direction for HA benchmark)
- Stage 8 k < 5 (inadequate corpus; re-examine download and screening stages)
- Any stage crashes and cannot be restarted without code changes

**What to do if fixes are needed**:
1. Log the specific failure in STATUS_LOG.md and DEVIATIONS_LOG.md (if it requires architectural change).
2. Distinguish implementation bugs (prompt formatting, schema parsing, API call structure) from architectural failures (the adjudication logic itself is wrong).
3. Fix implementation bugs without deviation log entry.
4. Fix architectural issues with a deviation log entry describing what changed, why, and the expected impact on preregistered analyses.
5. Re-run the affected stage and re-run Phase 0 tests that are relevant to the fix before continuing.

---

## Phase 2 — Carried-Forward Topics

**Topics**: legume_rotation, elevated_co2_face_yield, cover_crop_corn_yield

**Why these three are "carried forward"**: These topics existed in V1 (V1 results in the preregistration appendix). They have existing downloaded corpora and existing extracted data. The question is what, if anything, can be reused.

### What to Reuse vs. Re-Run

| Stage | Reuse V1 output? | Reason |
|-------|-----------------|--------|
| Stage 1 (search) | Optional | V1 search results are usable if date range matches V2 config; re-run if search terms changed significantly |
| Stage 2 (screening) | NO | V1 screening was keyword-based or an earlier LLM version; re-screen with V2 LLM screening protocol |
| Stage 3 (download) | YES | Downloaded PDFs are the same regardless of pipeline version; reuse the V1 download directory to avoid re-downloading |
| Stage 4 (extraction) | MAYBE | If V1 used the same multi-model consensus extraction with the same schema, reuse. If extraction schema changed or models changed significantly, re-extract. |
| Stage 5 (QC) | NO | Re-run; the QC script was updated with the frozen V2 hard filters |
| Stage 6 (adjudication) | NO — NEVER | This is the V2 innovation. V1 used keyword adjudication. V2 must use LLM adjudication. Reusing V1 adjudication decisions would defeat the purpose of V2. |
| Stage 7 (effectors) | NO | Re-run with the frozen V2 effector schema |
| Stage 8 (synthesis) | NO | Re-run after adjudication |
| Stage 9 (diagnostics) | NO | Re-run after synthesis |

**Key lesson**: DO NOT reuse V1 adjudication decisions. The entire point of V2 is that LLM adjudication replaces keyword adjudication. Reusing V1 adjudication would make the V2 evaluation a re-analysis of V1-filtered data, not a V2 evaluation.

### Per-Topic Instructions

#### legume_rotation

**V1 status**: GOOD direction, 17.7% vs 20.0% benchmark. Carried forward to test whether V2 LLM adjudication improves magnitude accuracy.

**V1 issues to address**:
- 82 opaque "low_confidence" exclusions with no documented rationale
- Comparator identity (true untreated monoculture vs. different rotation) was never verified
- Legume yield vs. subsequent cash crop yield was not explicitly verified at row level

**What to run**: Use V1 download directory for Stage 3. Re-run Stage 2 screening if the screen was keyword-based. Re-run Stages 5-9 with V2 pipeline.

**Specific check**: After Stage 6, count how many rows were previously excluded as "low_confidence" in V1. How does the LLM adjudicator handle these? If it keeps most of them, the V1 threshold was too conservative. If it excludes them with specific reasons, document the reason distribution.

**Expected improvement**: V2 LLM adjudication should improve magnitude accuracy (reduce the 2.3pp gap). Direction should remain correct.

#### elevated_co2_face_yield

**V1 status**: This topic was included in the V2 preregistration but was not in the 4-topic V1 audit set. Its V1 results are referenced in the preregistration as baseline for the S3 improvement analysis.

**Primary challenge**: The benchmark (Ainsworth & Long 2021) distinguishes FACE studies from OTC (open-top chamber) studies. OTC studies typically show larger yield responses to elevated CO2 than FACE studies. Including OTC data in a FACE-benchmark comparison would inflate the pipeline estimate. Stage 6 LLM adjudication must verify study type.

**Specific check**: After Stage 6, what fraction of excluded rows are excluded for "OTC/chamber study, not FACE"? If this fraction is high, it demonstrates the adjudicator is correctly enforcing the study-setting criterion that V1 keywords could not.

**Expected direction**: Positive for C3 crops (+8-15%); near-zero for C4 crops. The pipeline may not have enough C4 papers to estimate a separate C4 effect; the primary analysis will pool all crops.

#### cover_crop_corn_yield

**V1 status**: This topic is a hard null-result case. The benchmark is -1% to +3% (near-zero). The purpose of this topic in V2 is to test whether the pipeline avoids spurious positive effects.

**Primary challenge**: Cover crop studies often report multiple outcomes in the same paper: cover crop biomass (which should be excluded), subsequent corn yield (which should be included), and sometimes soil outcomes. Stage 6 must exclude cover crop biomass rows and include only subsequent corn grain yield.

**Specific check**: After Stage 6, verify that rows labeled with outcome containing "cover" or "biomass" or "termination" are excluded unless they represent the subsequent corn yield comparison.

**Expected result**: Pipeline pooled effect near zero (within -5% to +8% range). A strongly positive pooled effect would indicate outcome contamination (cover crop productivity rather than subsequent yield). A strongly negative pooled effect would indicate comparator confusion (comparing corn to no-corn, not corn-after-cover to corn-without-cover).

---

### Phase 2 Execution Order

Run topics in this order, in parallel where possible:

1. **Start legume_rotation first** (V1 GOOD result; most likely to succeed; good calibration point for LLM adjudicator behavior)
2. **Start elevated_co2_face_yield in parallel** (large corpus, independent of legume)
3. **Run cover_crop_corn_yield after observing Stage 6 behavior in legume and co2** (null-result topic; want to understand how adjudicator handles low-effect rows before testing null case)

**Parallelism note**: Stages 3, 4, 5 are independent across topics and can run in parallel. Stage 6 requires API calls and should not run three topics simultaneously (API rate limits). Run Stage 6 for one topic at a time.

---

## Phase 3 — New Topics

**Topics**: amf_inoculation_yield, biochar_tropical_yield

**Why these are "new"**: Neither topic had an end-to-end V1 pipeline run. amf_inoculation replaces mycorrhiza_yield (which was too broad); biochar_tropical replaces biochar_crop_yield (which was too broad for field-only studies). Both have explicitly scoped estimands that V1 did not have.

### What to Watch For Based on Phase 1 Lessons

After Phase 1 (humic_acid_yield) completes, the testing team will know:
- Whether the adjudicator correctly handles bundled interventions (HA + seaweed analogy → AMF + PGPR for amf_inoculation; biochar + compost for biochar_tropical)
- What the typical retention rate looks like for a clean topic
- Whether Stage 4 extraction correctly prioritizes harvestable yield over biomass and soil metrics

Apply those lessons here.

#### amf_inoculation_yield

**Benchmark**: Wu et al. 2022, PeerJ, +23% yield increase in field conditions

**Primary challenge**: AMF inoculation products are frequently combined with PGPR (plant-growth-promoting rhizobacteria), Trichoderma, or biostimulants. The benchmark is AMF-only. Stage 6 must exclude bundled inoculation treatments. This is directly analogous to the HA isolation challenge in Phase 1.

**Specific checks**:
1. After Stage 6, count rows excluded for "AMF bundled with other inoculants or PGPR." This should be a prominent exclusion reason.
2. After Stage 8, check whether the pipeline estimate is in the range of 15-35% (consistent with AMF field effects). An estimate > 50% would suggest pot-study contamination (AMF effects are much larger in controlled low-P conditions).
3. If estimate is well above benchmark: stratify by normalized_study_setting. Compare field-only estimate to full-pool estimate. A field-only estimate closer to +23% would confirm that pot-experiment inflation is the cause of the gap.

**Failure mode to anticipate**: This topic specifically replaced V1 mycorrhiza_yield, which showed a +29% → +74% paradox when filtered to "aligned" rows (BENCHMARK_ALIGNED_ANALYSIS). The cause was design-amplification: tightly controlled studies have larger AMF effects than field studies. V2 Stage 6 should enforce the "rainfed field conditions" restriction from the Wu et al. benchmark spec.

#### biochar_tropical_yield

**Benchmark**: Jeffery et al. 2017, Environmental Research Letters, +25% in tropical soils

**Primary challenge**: The benchmark is the tropical subgroup of a global biochar meta-analysis. Temperate biochar studies show near-zero effects (Jeffery et al. found essentially no temperate effect). Including temperate studies would dilute the estimate toward zero. Stage 6 must enforce the tropical/subtropical setting restriction.

**Specific checks**:
1. After Stage 7 effector normalization, count rows by normalized_climate_class: how many are tropical, subtropical, temperate? The synthesis should be restricted to tropical + subtropical rows.
2. After Stage 8, verify that the synthesis was run on tropical+subtropical rows only (not the full pool).
3. The benchmark-aligned subset in Stage 9 should show tropical rows only; compare this subset to the full-pool estimate and the benchmark.

**Failure mode to anticipate**: V1 biochar_crop_yield included global scope studies and produced +6.66% for the full pool, vs. +16.0% benchmark (which was itself the full-scope benchmark, not the tropical subgroup). Using the tropical subgroup benchmark (+25%) makes this a harder test but a more valid one. If the pipeline cannot restrict to tropical settings at Stage 6 or 7, the test is not comparable to the benchmark.

**Execution note**: Run amf_inoculation and biochar_tropical in parallel with Phase 2 topics (after Phase 1 validates the adjudicator is working).

---

## Phase 4 — Evaluation and Reporting

**Purpose**: Apply all preregistered analyses to the results from Phases 1-3; produce the full comparison table, forest plots, and deviation log; write the reporting summary.

### 4.1 Primary Confirmatory Analyses

These analyses are defined in the preregistration (PREREGISTRATION_V2_2026-03-26.md Section 4.1). They are evaluated after all 6 topics have completed Stages 8-9.

**P1 — Direction Agreement**:
For each topic, record: direction of pipeline pooled effect (positive or negative), direction of benchmark, direction_agrees (boolean). Count topics where direction_agrees = True. Apply thresholds:
- Primary success: >= 5/6
- Partial success: >= 4/6
- Failure: < 4/6

**P2 — CI Overlap**:
For each topic, record: pipeline 95% CI lower, pipeline 95% CI upper, benchmark point estimate. Compute ci_includes_benchmark = (CI_lower <= benchmark <= CI_upper). Count topics where ci_includes_benchmark = True. Apply thresholds:
- Primary success: >= 3/6
- Partial success: >= 2/6
- Failure: <= 1/6

**Important**: Both P1 and P2 must be assessed using the synthesis output from Stage 8 (adjudicated kept rows, DL random effects). Not from the benchmark-aligned subset (Stage 9). The preregistration is explicit about this.

### 4.2 Secondary Exploratory Analyses

**S1 — Absolute Gap**: For each topic, compute |pipeline estimate - benchmark estimate| in percentage points. No formal threshold; target is <= 10pp for >= 4/6 topics.

**S2 — Benchmark-Aligned Subset**: For each topic, compare aligned-subset estimate (Stage 9) to full-pool estimate (Stage 8) to benchmark. This is a diagnostic. The preregistered hypothesis (exploratory) is that alignment improves agreement in >= 3/6 topics.

**S3 — V1→V2 Improvement** (carried-forward topics only): For legume_rotation, elevated_co2_face_yield, cover_crop_corn_yield, compare V1 absolute gap (from preregistration appendix) to V2 absolute gap. Target: V2 gap < V1 gap for >= 2/3 topics.

### 4.3 Summary Table

Produce a single comparison table across all 6 topics:

| Topic | Pipeline Effect (%) | 95% CI | Benchmark (%) | Direction | CI Overlap | Abs Gap (pp) | V1 Effect (%) |
|-------|--------------------|----|--------------|-----------|------------|-----------|--------------|
| humic_acid_yield | ... | ... | +12% | ... | ... | ... | — |
| amf_inoculation_yield | ... | ... | +23% | ... | ... | ... | — |
| biochar_tropical_yield | ... | ... | +25% | ... | ... | ... | — |
| legume_rotation | ... | ... | +20% | ... | ... | ... | +17.7% |
| elevated_co2_face_yield | ... | ... | ~+10-15% | ... | ... | ... | ... |
| cover_crop_corn_yield | ... | ... | -1 to +3% | ... | ... | ... | ... |

### 4.4 Forest Plots

For each topic, produce a forest plot showing:
- Individual paper effect sizes with 95% CIs
- Pooled estimate with 95% CI (diamond)
- Benchmark point estimate (dashed vertical line)

Forest plots go in `9_diagnostics/` for each topic and in a combined figure for the paper.

### 4.5 Deviation Log

During the V2 run, any change to the frozen architecture requires a deviation log entry. The log is maintained in `codex/DEVIATIONS_LOG.md` (create if it doesn't exist).

Required fields per deviation:
- Date of deviation
- Stage(s) affected
- Description of what changed
- Reason (why the change was necessary)
- Expected impact on preregistered analyses (P1 and P2)
- Who authorized the change (or "automatic due to implementation bug fix")

At Phase 4, compile all deviation log entries and assess whether any deviation materially affects the interpretation of P1 or P2.

### 4.6 Final Report Structure

The final report must include, in this order:

1. **Overview**: What V2 is; what was preregistered; when the architecture was frozen.
2. **Primary Results**: P1 direction agreement (n of 6 topics correct); P2 CI overlap (n of 6 topics). Pass/partial/fail verdict against preregistered thresholds.
3. **Per-Topic Results**: For each topic: pipeline estimate, benchmark, direction, CI overlap, absolute gap. Brief narrative of any notable Stage 6 adjudication findings.
4. **Secondary Results**: S1 absolute gaps; S2 benchmark-aligned subset findings; S3 V1→V2 improvement.
5. **Failure Analysis**: For topics where direction or CI overlap failed, apply the failure taxonomy (extraction error vs. corpus composition vs. estimand mismatch vs. OA access limitation).
6. **Deviation Log Summary**: Any deviations from the frozen protocol, with impact assessment.
7. **Limitations**: What structural limits remain even after V2 improvements (OA corpus bias, missing papers, irreducible benchmark-composition mismatches).

---

## Stage-by-Stage Reference

Condensed one-page reference for each stage.

---

### Stage 1: Literature Search

**Inputs**: `{topic}/config.json` (search terms, date range)
**Output**: `{topic}/1_search/results.json`
**Script**: `pipeline_replication/1_search.py` (or equivalent search module)
**Key V1 failure**: None at search stage; search was functional in V1.
**V2 fix**: Broader date range and topic-specific Boolean construction from config.
**Quality check**: Count total records returned; verify record fields are complete; spot-check 5 titles for relevance.

---

### Stage 2: Abstract Screening

**Inputs**: `{topic}/1_search/results.json`, `{topic}/config.json` (PICO criteria)
**Output**: `{topic}/2_screen/screening_results.csv`
**Script**: `pipeline_replication/2_screen.py`
**Key V1 failure**: V1 used keyword screening that could not detect HA isolation, outcome type, study setting from abstracts.
**V2 fix**: LLM screening with PICO criteria from config.
**Quality check**: Spot-check 10 INCLUDE and 10 EXCLUDE decisions; verify INCLUDE rate is 10-40%; verify no review articles in INCLUDE.

---

### Stage 3: PDF Download

**Inputs**: `{topic}/2_screen/screening_results.csv` (INCLUDE decisions)
**Output**: `{topic}/3_download/` (PDFs), `download_log.json`
**Script**: `claude universal metaanalysis pipeline/pdf_downloader.py`
**Key V1 failure**: None; the universal downloader is a retained V1 strength.
**V2 fix**: No change; use same universal downloader.
**Quality check**: Check download_log.json for success rate; open 3 PDFs to verify content matches expected paper.

---

### Stage 4: Data Extraction

**Inputs**: PDFs from Stage 3, `{topic}/config.json`
**Output**: `{topic}/4_extract/summary.csv`
**Script**: Multi-model consensus extraction
**Key V1 failure**: Wrong outcome prioritized (biomass over grain yield; mineral concentration over yield); high "low confidence" rate.
**V2 fix**: Extraction prompt explicitly prioritizes target outcome class (grain/harvest yield) using config `extraction_priorities`; `tc_confusion_warnings` highlight common extraction mistakes.
**Quality check**: Verify >= 60% of papers have rows; spot-check 5 papers against PDFs; check confidence distribution.

---

### Stage 5: Deterministic QC

**Inputs**: `{topic}/4_extract/summary.csv`, `{topic}/config.json`
**Output**: `{topic}/5_qc/summary_qc.csv`, `qc_audit.json`
**Script**: `codex/qc_hard_filters.py`
**Key V1 failure**: notill_tillage plausibility filter was too permissive (passed 144-609% effects from AbdulsattarAlrijabo 2014).
**V2 fix**: Default threshold |lnRR| > 2.0 (i.e., effect > +619%); topic-specific override in config (e.g., humic_acid: |effect| > 200%).
**Quality check**: Inspect qc_audit.json exclusion counts; examine all flagged extreme-value rows; verify lnRR column is populated for all complete rows.

---

### Stage 6: LLM Semantic Adjudication

**Inputs**: `{topic}/5_qc/summary_qc.csv`, `{topic}/config.json` (topic brief)
**Output**: `{topic}/6_adjudication/adjudication_decisions.jsonl`, `adjudicated_kept.csv`
**Script**: `codex/adjudicate_llm_universal.py`
**Key V1 failure**: Keyword adjudicator could not handle intervention isolation, outcome label heterogeneity, comparator identity, estimand verification, or plausibility with context (SPOT_CHECK_REPORT, all 4 topics).
**V2 fix**: Claude LLM reads each row in context against topic brief; produces structured 8-field decision; excludes by default if ambiguous.
**Quality check**: Verify 0 JSON parse failures; spot-check 10 excluded and 10 kept rows; verify retention rate 30-80%; verify at least one exclusion per failure category relevant to the topic.

---

### Stage 7: Effector Normalization

**Inputs**: `{topic}/6_adjudication/adjudicated_kept.csv`
**Output**: `{topic}/7_effectors/effector_labels.jsonl`
**Script**: `codex/normalize_effectors_universal.py`
**Key V1 failure**: Without canonical labels, benchmark-aligned subgroup analyses could not be run systematically.
**V2 fix**: Universal effector schema produces 6 canonical labels per row; enables benchmark-aligned subset analysis in Stage 9.
**Quality check**: Verify all 6 fields present per row; check setting and climate distribution; verify some rows labeled benchmark_aligned.

---

### Stage 8: Synthesis

**Inputs**: `{topic}/6_adjudication/adjudicated_kept.csv`
**Output**: `{topic}/8_synthesis/synthesis_results.json`
**Script**: `pipeline_replication/synthesize.py` or equivalent
**Key V1 failure**: Synthesis pooled wrong rows because adjudication (keyword) was inadequate.
**V2 fix**: Synthesis receives only LLM-adjudicated kept rows; variance conversion hierarchy is frozen and applies consistently.
**Quality check**: Verify k >= 10; check variance coverage rate; verify direction and approximate magnitude; inspect benchmark comparison fields.

---

### Stage 9: Diagnostics

**Inputs**: `{topic}/` (all stage outputs)
**Output**: `{topic}/9_diagnostics/diagnostics_report.md`
**Script**: `pipeline_replication/run_diagnostics.py`
**Key V1 failure**: No automatic diagnostics in V1; benchmark-aligned subsets were run ad hoc and sometimes used as correction mechanisms rather than diagnostics.
**V2 fix**: 7 automatic diagnostics run unconditionally; benchmark-aligned subset is diagnostic only, never the primary result.
**Quality check**: Verify all 7 diagnostics produced outputs; identify most influential paper; check whether any single crop class dominates composition.

---

## Known Risks and Mitigations

### Risk 1: API keys not yet refreshed (BLOCKER)

**Description**: Both Anthropic and Google API keys are expired/out of credit as of 2026-03-26. Stage 6 and Stage 4 (multi-model extraction) cannot proceed.

**Mitigation**: This is a Phase 0 prerequisite. No Phase 1 work begins until 0.1 (API key validation) passes. Top up Anthropic credit and renew Google API key immediately.

**Escalation**: If only one provider is available, V2 can proceed with single-model extraction and single-provider adjudication, but this must be logged as a deviation.

---

### Risk 2: LLM adjudicator behavior on live data is unknown

**Description**: `adjudicate_llm_universal.py` was tested only in dry-run mode. The live behavior on real rows may surface prompt failures, schema parsing errors, or unexpected model behavior that dry-run cannot anticipate (AUDIT_REPORT, Risk 4).

**Mitigation**: Phase 0 test (0.2) uses 8 synthetic test cases with known expected decisions on live API calls. Phase 1 (humic_acid pilot) is the first live test on real extracted data and functions as the go/no-go gate. The Phase 1 go/no-go criteria explicitly include "Stage 6 produces > 5% JSON parse failures" as a failure condition.

---

### Risk 3: notill_tillage-style outliers in new topics

**Description**: The most visible V1 failure was the AbdulsattarAlrijabo unit error (144-609% effects) passing undetected. New topics may have analogous papers with physically implausible effects from unit errors, misread tables, or T/C swaps.

**Mitigation**: Phase 0 check (0.6) validates the plausibility filter with test rows. Topic configs include domain-specific plausibility thresholds (default |lnRR| > 2.0; humic_acid overrides to |effect| > 200%). Stage 6 LLM adjudication includes "plausibility with context" as criterion 5. Stage 9 LOO analysis flags any single paper whose removal changes direction.

---

### Risk 4: Tropical restriction in biochar_tropical may reduce k below usable threshold

**Description**: The biochar_tropical benchmark is the tropical/subtropical subgroup of Jeffery et al. 2017. Restricting to tropical rows only may leave k < 10 after adjudication, making the pooled estimate unreliable.

**Mitigation**: Before running Stage 8 on the tropical-restricted set, count the number of tropical+subtropical rows. If k < 10, document this as a structural OA limitation rather than an extraction failure. Consider whether subtropical rows should also be included (the benchmark spec should specify this). If k is genuinely too low, flag this topic as "OA access limitation" in the failure taxonomy.

---

### Risk 5: cover_crop_corn_yield near-zero benchmark is hard to confirm directionally

**Description**: The benchmark is -1% to +3%. Any pipeline result in the range -10% to +10% is arguably consistent. This makes P1 direction agreement ambiguous for this topic: what counts as "the right direction" for a near-zero effect?

**Mitigation**: The preregistration specifies "expected direction: near-zero positive." For the purpose of P1, define success as the pipeline 95% CI including zero (i.e., the pipeline correctly identifies the null or near-null result). A pipeline estimate of +15% or -15% (far from zero) would be counted as a direction failure because it implies a spurious effect. Document this interpretation in the deviation log if it deviates from a strict sign-match interpretation.

---

### Risk 6: V2 architecture freeze happened before a full dress rehearsal

**Description**: The PIPELINE_V2_FROZEN_2026-03-26.md document notes "dress rehearsal completed" in its version history, but only Stages 1-2 were run (search and screen). Stages 3-9 were not run before freezing. This creates risk that the pilot reveals issues requiring architectural changes (AUDIT_REPORT, Risk 2).

**Mitigation**: The Phase 1 go/no-go criteria are designed to catch this. Any issue in Phases 1-3 that requires changing a frozen element must be logged as a deviation before proceeding. The deviation log format captures the impact on preregistered analyses. The scientific integrity of the preregistration is maintained as long as all deviations are documented and their impact disclosed.

---

### Risk 7: Success criteria are ambitious relative to V1 performance

**Description**: V1 achieved 4/6 correct direction. V2 primary success threshold is 5/6. This requires converting at least one V1 failure into a success. The two V1 direction failures were notill (+1.2% vs -5.7%) and intercropping (-3.1% vs +22%). Both are replaced in V2: notill replaced by cover_crop_corn_yield; intercropping replaced by amf_inoculation_yield and biochar_tropical_yield. So V2 is not simply re-running V1 topics — it has a different topic set designed to reduce structural failures.

**Mitigation**: The topic selection process explicitly scored all 6 V2 topics on estimand clarity, intervention clarity, OA feasibility, and setting coherence. The new topics score 34-37 out of 40. If V2 still fails to achieve 5/6 direction agreement, this is scientifically important and will be reported as such.

---

### Risk 8: Benchmark exact numbers not yet confirmed for 4 of 6 topics

**Description**: The preregistration appendix lists 4 benchmark DOIs as "TBC" (amf_inoculation, biochar_tropical, elevated_co2, cover_crop). The exact pooled effects and CIs from these papers must be confirmed before any synthesis comparison is run (AUDIT_REPORT, Risk 7).

**Mitigation**: Phase 0 check (0.3) requires all 6 benchmark specs to have exact numbers before any pipeline run begins. This check will catch TBC placeholders.

---

## What Claude Code Does vs What Scripts Do

This section documents the division of labor between Claude Code acting as an LLM semantic adjudicator and Python scripts doing deterministic computation.

### Claude Code / LLM Roles

Claude Code or another LLM performs tasks where the question is "what does this mean?" or requires reading text in context:

| Task | Stage | Why LLM |
|------|-------|---------|
| Abstract screening: Does this paper meet PICO criteria? | Stage 2 | Requires interpreting title+abstract semantics; keyword matching misses paraphrases |
| Row adjudication: Does the treatment match the intervention definition? | Stage 6 | Requires reading treatment_description in context against a semantic intervention spec |
| Row adjudication: Is this a real yield outcome or a proxy/quality trait? | Stage 6 | Outcome labels are heterogeneous; "marketable fresh weight" requires judgment |
| Row adjudication: Is the control a genuine no-treatment comparator? | Stage 6 | Comparator identity requires reading the experimental design context |
| Row adjudication: Is the effect size plausible given study context? | Stage 6 | Plausibility requires domain knowledge and context, not arithmetic |
| T/C swap detection: Is treatment/control orientation reversed? | Stage 6 | Requires understanding which arm the table labels as "treatment" |
| Effector normalization: crop class, setting, climate, estimand context | Stage 7 | Ontology mapping from free-text fields requires semantic understanding |
| Failure taxonomy: what category explains a benchmark mismatch? | Stage 9 | Diagnosis of mismatches requires reasoning about study design and scope |

**Execution**: These tasks are run by calling Claude (or Gemini) through the adjudication and normalization scripts. Claude Code is the human-in-the-loop reviewer who reads the diagnostic reports and spot-checks the decisions. Claude Code does NOT manually override individual adjudication decisions — it checks that the system is working correctly and logs deviations if it is not.

### Python Script Roles

Python scripts perform tasks where the question has a formula or is purely deterministic:

| Task | Stage | Why Script |
|------|-------|----------|
| Construct OpenAlex API query from config | Stage 1 | Boolean logic on search terms; deterministic |
| Deduplicate records by DOI | Stage 1 | Exact string matching |
| Download PDFs from URLs with retry | Stage 3 | HTTP requests; no semantic content needed |
| Check both means are present and numeric | Stage 5 | Python isinstance() check |
| Check both means are positive | Stage 5 | Arithmetic comparison |
| Compute lnRR | Stage 5 | Formula: ln(treatment/control) |
| Detect duplicates within same paper | Stage 5 | String + numeric comparison |
| Flag rows where |lnRR| > threshold | Stage 5 | Arithmetic comparison |
| Convert SE to SD (SD = SE * sqrt(n)) | Stage 8 | Formula |
| Convert LSD to SD | Stage 8 | Formula |
| Convert CV to SD (SD = CV * mean / 100) | Stage 8 | Formula |
| DerSimonian-Laird random effects | Stage 8 | Statistical algorithm |
| Compute 95% CI, I², tau² | Stage 8 | Statistical algorithm |
| Leave-one-out influence analysis | Stage 9 | Iterative pooling |
| Funnel plot / Egger's test | Stage 9 | Statistical test |
| Composition comparison tabulation | Stage 9 | Counting and percentages |

**Rule of thumb** (from CLAUDE_HANDOFF.md): "If there is a formula, do it programmatically. If the question is 'what does this row mean?', use an LLM."

---

## Timeline

### Dependencies and Critical Path

The critical path is: Phase 0 → Phase 1 → Phase 2 + Phase 3 (parallel) → Phase 4.

Phase 2 and Phase 3 can run in parallel with each other after Phase 1 validates the LLM adjudicator is working. However, the API rate limits mean Stage 6 should not be run on more than 2 topics simultaneously.

### Rough Time Estimates Per Phase

| Phase | Estimated Duration | Notes |
|-------|-------------------|-------|
| Phase 0 | 1-2 hours | Mostly verification tasks; API key refresh may take longer if credit processing delays |
| Phase 1 (humic_acid) | 4-8 hours | First live run; allow extra time for diagnosing unexpected Stage 6 behavior |
| Phase 2 (3 topics, parallel) | 8-12 hours | Stage 6 serial; other stages parallel |
| Phase 3 (2 topics, parallel) | 6-10 hours | Can run alongside Phase 2 after Phase 1 validates adjudicator |
| Phase 4 (evaluation) | 3-5 hours | Mostly script runs + report writing |
| **Total** | **~1-3 days** | Depending on API latency and any deviation fixes needed |

### What Blocks What

```
Phase 0: API key refresh (0.1) → BLOCKS all downstream phases
Phase 0: Benchmark spec TBCs (0.3) → BLOCKS Phase 4 evaluation for affected topics
Phase 1: Stage 6 live test → BLOCKS Phase 2 and Phase 3 start
Phase 1: Go/no-go verdict → BLOCKS full run if failed (fix required first)
Phase 2/3: Stage 6 completion → BLOCKS Stage 8 per topic
Phase 2/3: Stage 8 completion → BLOCKS Phase 4 P1/P2 evaluation
```

### What Can Parallelize

- Phase 2 (legume, co2, cover_crop) and Phase 3 (amf, biochar) can proceed in parallel after Phase 1 clears.
- Within a topic, Stages 1-5 can run while Stage 3-5 of another topic are also running (download and QC are not API-rate-limited).
- Stage 9 diagnostics for any topic can begin as soon as Stage 8 for that topic completes.
- Benchmark spec TBC resolution can be done in parallel with Phase 1 (it does not require API calls; it requires reading the benchmark papers).

---

## Appendix: Pre-flight Checklist (Summary)

Before any Phase 1 work begins, confirm all six items with a timestamp:

```
[ ] 0.1 API key validation — DATE: ___________
[ ] 0.2 Adjudicator test cases (8 rows) — DATE: ___________
[ ] 0.3 All 6 benchmark specs have exact numbers — DATE: ___________
[ ] 0.4 All 6 configs pass schema validation — DATE: ___________
[ ] 0.5 All 6 topic directories have 9-stage structure — DATE: ___________
[ ] 0.6 Plausibility filter correctly flags test rows — DATE: ___________
```

Record the completion of Phase 0 in STATUS_LOG.md before starting Phase 1.
