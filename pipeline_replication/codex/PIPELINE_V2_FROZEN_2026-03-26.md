# Pipeline V2 Frozen Architecture Snapshot
**Version note**: Pipeline V2 architecture frozen 2026-03-26 prior to preregistration.
**Based on**: `PIPELINE_V2_ARCHITECTURE.md` (Draft 1.0, 2026-03-25)
**Status**: FROZEN — do not modify this document after 2026-03-26 without creating a deviation log entry.

---

## Purpose of This Document

This document captures the exact state of the Pipeline V2 architecture as frozen before any V2 results were produced. It exists to make the evaluation of V2 results independent of post-hoc design changes. Any deviation from this specification must be logged in `DEVIATIONS_LOG.md` with date, rationale, and expected impact on results.

---

## Section 1: Frozen Architecture — 9 Stages

### Stage 1: Literature Search
- **Tool**: OpenAlex API (`https://api.openalex.org/works`)
- **Query construction**: Topic config `pico.intervention.search_terms` + `pico.outcome.primary.search_terms` joined as Boolean OR within each group, AND between groups
- **Filters**: `type:article`, `publication_year: {config.search.date_range.start}-{config.search.date_range.end}`
- **Fields retrieved**: `id, doi, title, abstract_inverted_index, publication_year, primary_location, authorships`
- **Deduplication**: By DOI (remove exact-match duplicates)
- **Output**: `1_search/results.json` with query metadata and full results
- **Target corpus size**: 5,000-10,000 records for most topics (up to 40,000 for broad topics like HA)
- **Configurable**: search terms, date range, language filter — all from topic config

### Stage 2: Abstract Screening
- **Tool**: LLM (fast model — Gemini Flash or Claude Haiku for cost efficiency)
- **Input**: title + abstract for each record from Stage 1
- **Decision schema**: INCLUDE / EXCLUDE / UNSURE with reason
- **PICO evaluation**: Does the abstract indicate (1) the right intervention, (2) the right comparator, (3) the right outcome type?
- **Output**: `2_screen/screening_results.csv` (columns: doi, title, year, decision, reason, intervention_match, comparator_match, outcome_match)
- **Expected pass rate**: 10-30% of raw corpus
- **Configurable**: PICO criteria from topic config

### Stage 3: PDF Download
- **Tool**: `universal_downloader.py` (proven V1 asset — do not replace)
- **Input**: DOI list from Stage 2 (INCLUDE decisions only)
- **Sources**: Unpaywall, OpenAlex OA link, PubMed Central, direct publisher, Sci-Hub fallback
- **Output**: `3_download/` directory with PDFs + `download_log.json`
- **Expected download success rate**: 60-85% (varies by topic OA coverage)
- **Configurable**: none (universal download logic)

### Stage 4: Data Extraction
- **Tool**: Multi-model consensus extraction (Claude Sonnet + Gemini Pro, minimum 2 models)
- **Input**: Downloaded PDFs + topic config (intervention, comparator, outcome definitions)
- **Extraction prompt**: topic-specific, derived from config `extraction_priorities` + `tc_confusion_warnings`
- **Output per model**: structured JSON per paper with observation-level rows
- **Consensus**: majority rules on numeric values; tiebreaker = model with highest global match rate on this topic's pilot papers
- **Output**: `4_extract/summary.csv` (one row per comparison per paper per model, then merged)
- **Extraction schema** (frozen):
```json
{
  "paper_id": "string",
  "outcome": "free text",
  "outcome_unit": "string",
  "source_type": "table|figure|text",
  "table_or_figure": "string",
  "treatment_mean": "float",
  "control_mean": "float",
  "treatment_n": "integer or null",
  "control_n": "integer or null",
  "variance_type": "SE|SD|LSD|CI|CV|null",
  "variance_value": "float or null",
  "se_treatment": "float or null",
  "sd_treatment": "float or null",
  "se_control": "float or null",
  "sd_control": "float or null",
  "treatment_description": "string",
  "control_description": "string",
  "confidence": "high|medium|low",
  "notes": "string",
  "title": "string",
  "moderators": {
    "mod_crop_type": "string or null",
    "mod_experiment_type": "string or null",
    "mod_climate_zone": "string or null",
    "mod_soil_type": "string or null",
    "mod_application_method": "string or null"
  }
}
```
- **Configurable**: extraction targets, moderators, TC confusion prompts — all from topic config

### Stage 5: Deterministic QC (Hard Filters)
- **Tool**: `qc_hard_filters.py` (Python, no LLM)
- **Input**: `summary.csv` from Stage 4
- **Checks performed** (frozen):
  1. Both treatment and control means present and numeric (float, not string)
  2. At least one mean is positive (lnRR requires log of positive ratio)
  3. Both means positive (required for lnRR calculation)
  4. CV bounds check: flag if CV > 100% or CV < 0.5% (likely unit error)
  5. Duplicate detection: same paper + same outcome + same means within 0.1% = likely duplicate; keep one
  6. Non-independence flag: multiple rows from same paper with same outcome → flag for paper-level aggregation
  7. Extreme effect check: |lnRR| > 2.0 (i.e., effect > +619% or < -86%) → flag as implausible
  8. Effect size computation: `lnRR = ln(treatment_mean / control_mean)`; simple percentage change as backup
- **Output**: `summary_qc.csv` + `qc_audit.json`
- **Configurable**: extreme effect threshold is topic-specific (default |lnRR| > 2.0; humic_acid overrides to |effect| > 200%)

### Stage 6: LLM Semantic Adjudication
- **Model**: Claude (claude-sonnet-4-20250514 or equivalent current Anthropic model)
- **Input**: QC-passed rows + topic config brief
- **Prompt structure**: system prompt with topic brief + row JSON → structured decision JSON
- **Adjudication output schema** (frozen):
```json
{
  "row_id": "string",
  "decision": "keep|exclude|flag|swap_treatment_control",
  "intervention_match": "yes|partial|no",
  "comparator_match": "yes|partial|no",
  "outcome_match": "yes|partial|no",
  "estimand_match": "yes|partial|no",
  "needs_tc_swap": "boolean",
  "normalized_outcome_class": "grain_yield|harvest_yield|biomass|quality_trait|system_productivity|component_yield|other",
  "normalized_study_setting": "field|greenhouse|pot|mixed|unknown",
  "normalized_estimand_class": "string",
  "exclusion_reason": "string or null",
  "rationale_short": "1-2 sentences"
}
```
- **Decision policy** (frozen):
  - `keep` if intervention, comparator, and outcome all >= `partial`
  - `exclude` if any of intervention, comparator, outcome = `no`
  - `flag` if estimand = `partial` or row semantics are ambiguous
  - `swap_treatment_control` if T/C clearly reversed relative to config
  - Default: exclude ambiguous rows rather than include
- **Output**: `adjudication_decisions.jsonl` + `adjudicated_kept.csv`
- **Script**: `codex/adjudicate_llm_universal.py`
- **Configurable**: topic brief content (from config) — decision logic is universal and fixed

### Stage 7: Effector Normalization
- **Tool**: LLM (Claude) + regex patterns
- **Input**: Kept rows from Stage 6
- **Effector output schema** (frozen):
```json
{
  "row_id": "string",
  "normalized_crop_class": "grain_cereal|legume|vegetable|root_tuber|oilseed|fiber|tree_crop|grass_forage|null",
  "normalized_study_setting": "field|greenhouse|pot|mixed|unknown",
  "normalized_climate_class": "temperate|tropical|subtropical|semi_arid|arid|mediterranean|boreal|unknown",
  "normalized_soil_class": "string or null",
  "normalized_management_class": "residue_rotation|residue_only|rotation_only|standard|unknown",
  "normalized_estimand_context": "benchmark_aligned|partially_aligned|misaligned|unknown"
}
```
- **Output**: `effector_labels.jsonl`
- **Script**: `codex/normalize_effectors_universal.py`
- **Configurable**: which moderators are topic-relevant (from config `extraction.moderators`)

### Stage 8: Synthesis
- **Method**: DerSimonian-Laird random effects meta-analysis on lnRR
- **Variance conversion hierarchy** (frozen):
  1. SD + n → variance of lnRR directly (`1/n_t + 1/n_c` times mean SD approximation)
  2. SE + n → SD = SE × √n → variance
  3. LSD + n → SE_diff = LSD / (t_crit × √2); SD from SE_diff × √n
  4. CI → SD from CI width / (2 × z_crit) × √n
  5. CV + mean → SD = CV × mean / 100
  6. Missing variance → row included in unweighted mean but excluded from DL weighted analysis
- **Primary output**: pooled effect %, 95% CI, I², tau², k
- **Secondary output**: paper-level aggregated estimate (non-independence sensitivity)
- **Benchmark comparison**: direction agreement, CI overlap, absolute difference in pp
- **Output**: `synthesis_results.json`

### Stage 9: Diagnostics & Reporting
- **Automatic diagnostics** (all run unconditionally):
  1. Leave-one-out influence analysis
  2. Benchmark-aligned subset estimate (field-only, direct yield-only rows per benchmark spec)
  3. High-confidence-only estimate (confidence=high rows only)
  4. Table-only estimate (source_type=table rows only)
  5. Funnel plot / Egger's test for small-study effects
  6. Composition comparison: crop class distribution, setting distribution, climate distribution vs benchmark
  7. Failure taxonomy: if pipeline disagrees with benchmark, classify likely cause
- **Output**: `diagnostics_report.md` + supporting figures

---

## Section 2: Frozen Adjudication Design

### LLM vs Keyword — Which Stages Use Which

| Stage | Method | Rationale |
|-------|--------|-----------|
| Stage 2: Abstract Screening | LLM | Keywords cannot detect HA isolation, outcome type, study setting from abstract |
| Stage 5: Hard Filters | Deterministic code | No semantics needed; purely numeric checks |
| Stage 6: Semantic Adjudication | LLM | Core V2 innovation; handles 5 failure categories keywords cannot |
| Stage 7: Effector Normalization | LLM + regex | Crop/setting labels need semantic understanding; regex for structured fields |

### The 5 Criteria Where LLM Replaces Keywords (Frozen)

1. **Intervention isolation** — Is the treatment arm a pure HA application, or is HA bundled with other inputs in an inseparable combination? Keywords cannot read experimental design tables.

2. **Outcome label heterogeneity** — Does "marketable fresh yield per plant" or "total dry shoot biomass g/m²" count as crop yield for this benchmark? Keywords cannot assess whether an outcome label is a valid yield proxy.

3. **Comparator identity** — Is the control arm a proper no-HA comparison (same base fertilization, just without HA), or is it zero-fertilizer, another biostimulant, or another HA rate? Keywords cannot read comparator arm descriptions.

4. **Estimand verification** — Is this row measuring the same quantity as the benchmark? For example, does the row measure per-plant yield (not benchmark-aligned) or per-area yield (benchmark-aligned)?

5. **Plausibility check with context** — Is a +300% HA effect in a pot trial implausible (zero-fertilizer control) or possibly real (severely degraded soil, very low baseline)? LLM can read the Methods context; keywords cannot.

---

## Section 3: Frozen Schema Definitions

### 3.1 Extraction Fields (see Stage 4 schema above — frozen)

### 3.2 Adjudication Output Fields (see Stage 6 schema above — frozen)

### 3.3 Effector Label Fields (see Stage 7 schema above — frozen)

### 3.4 Outcome Class Ontology (Frozen)
- `grain_yield` — harvested grain (wheat, maize, rice, barley, sorghum)
- `harvest_yield` — other harvestable crop product (fruit, tuber, pod, seed, fiber)
- `biomass` — total/shoot/plant dry matter (not harvestable-portion-specific)
- `quality_trait` — protein concentration, mineral content, sugar content, energy value
- `system_productivity` — LER, total system yield, equivalent yield
- `component_yield` — individual crop yield within a mixed system
- `other` — any yield-adjacent metric not covered above

### 3.5 Study Setting Ontology (Frozen)
- `field` — field experiments, on-farm trials, multi-location trials
- `greenhouse` — controlled environment with natural or artificial light
- `pot` — container experiments (with soil substrate)
- `hydroponic` — soilless system (excluded from most analyses)
- `mixed` — study includes both field and controlled-environment components
- `unknown` — insufficient information to determine

### 3.6 Estimand Context Labels (Frozen)
- `benchmark_aligned` — row measures exactly what the benchmark measures (field, direct yield, isolated treatment, appropriate comparator)
- `partially_aligned` — row is related but differs in one dimension (e.g., pot not field, yield per plant not per area)
- `misaligned` — row measures something structurally different from the benchmark target
- `unknown` — insufficient information to classify

---

## Section 4: Frozen Success Criteria

### Primary (Confirmatory — must be assessed before data collected)

| Criterion | Description | Success Threshold |
|-----------|-------------|------------------|
| Direction agreement | Pipeline pooled effect has same sign as benchmark | ≥ 5/6 topics |
| CI overlap | Pipeline 95% CI includes benchmark point estimate | ≥ 3/6 topics |

**Primary success** = both thresholds met.
**Partial success** = ≥ 4/6 direction agreement OR ≥ 2/6 CI overlap.
**Failure** = < 4/6 direction agreement.

### Secondary (Exploratory — assessed after data collected, no success threshold)

| Criterion | Description |
|-----------|-------------|
| Absolute gap | |pipeline - benchmark| in percentage points (report per topic) |
| Benchmark-aligned subset | Does field-only / direct-yield-only filter improve agreement? |
| V1→V2 improvement | For carried-forward topics, is V2 absolute gap smaller than V1 keyword gap? |
| Failure taxonomy | Can disagreements be classified into diagnostic categories? |

### Process Metrics (Reported but not pass/fail)

| Metric | Description |
|--------|-------------|
| Download coverage | % of screened-included papers successfully downloaded |
| Extraction completeness | % of downloaded papers with ≥1 extracted row |
| Variance coverage | % of adjudicated-kept rows with usable variance for DL weighting |
| Row retention rate | % of extracted rows passing Stage 6 adjudication |

---

## Section 5: What IS Frozen vs. What Is Configurable

### Frozen (cannot change without deviation log entry)
- The 9-stage architecture and their sequence
- The adjudication output schema (8 fields)
- The outcome class ontology
- The estimand context labels
- The primary success criteria and thresholds
- The variance conversion hierarchy
- The DerSimonian-Laird synthesis method
- The 6 preregistered topics
- The benchmark papers for each topic (see preregistration document)

### Configurable (can change per topic without deviation)
- Search terms and date range (from topic config)
- Extraction moderators (from topic config)
- Topic-specific plausibility thresholds (e.g., |effect| > 200% for HA topics)
- LLM model version (Anthropic upgrades from sonnet-4 to newer equivalent are permitted)
- Download retry logic and fallback sources (implementation detail)
- Number of extraction models used (minimum: 2)

---

## Version History

| Date | Change | Author |
|------|--------|--------|
| 2026-03-25 | Architecture spec drafted (PIPELINE_V2_ARCHITECTURE.md) | Claude Code (autonomous) |
| 2026-03-26 | Architecture frozen (this document); dress rehearsal completed | Claude Code (autonomous) |
