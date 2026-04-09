# Pipeline V2 Architecture Specification

## Version
Draft 1.0 — 2026-03-25

## Purpose

Pipeline V2 is a universal autonomous meta-analysis replication system.
Given a topic configuration and benchmark specification, it proceeds from
literature search to pooled effect estimation without manual paper-by-paper curation.

V2 incorporates all lessons from V1 development and stress-testing.

---

## Design Principles

1. **Broad extraction, strict post-processing.** The extractor casts a wide net.
   The post-extraction layer decides what enters synthesis.

2. **LLM for semantics, code for math.** If there is a formula, do it in code.
   If the question is "what does this row mean?", use an LLM.

3. **Config-driven universality.** Every topic-specific behavior comes from the config,
   not from hardcoded rules. The same code runs for any topic.

4. **Benchmark comparison, not benchmark worship.** The benchmark is an external
   reference synthesis. Disagreement is informative, not necessarily failure.

5. **Preregistered evaluation.** The topic set, primary analyses, and success criteria
   are frozen before seeing results.

---

## Architecture Overview

```
Input: Topic Config + Benchmark Spec
  |
  v
Stage 1: Literature Search (OpenAlex API)
  |
  v
Stage 2: Abstract Screening (LLM-based)
  |
  v
Stage 3: PDF Download (Universal Downloader)
  |
  v
Stage 4: Data Extraction (Multi-model consensus)
  |
  v
Stage 5: Deterministic QC (Hard filters)
  |
  v
Stage 6: LLM Semantic Adjudication (Claude Opus 4.6)
  |
  v
Stage 7: Effector Normalization (LLM + regex)
  |
  v
Stage 8: Synthesis (DerSimonian-Laird RE)
  |
  v
Stage 9: Diagnostics & Reporting
```

---

## Stage 1: Literature Search

**Input:** Topic config with search terms
**Tool:** OpenAlex API
**Output:** `1_search/results.json` (5,000-10,000 records)

Process:
- Query OpenAlex using topic search terms
- Retrieve title, abstract, DOI, publication year, journal, OA status
- Deduplicate by DOI

---

## Stage 2: Abstract Screening

**Input:** Search results
**Tool:** LLM (fast model for cost efficiency)
**Output:** `2_screen/screening_results.csv`

Process:
- LLM screens each title+abstract against PICO criteria
- Binary include/exclude with confidence score
- Pass rate: typically 10-30%

---

## Stage 3: PDF Download

**Input:** Screened paper list
**Tool:** Universal Downloader (retained from V1 — proven strength)
**Output:** `3_download/` (PDFs)

Process:
- Resolve DOIs to OA full-text URLs
- Download from multiple sources (publisher, PMC, repository)
- Retry with fallback sources
- Log download success/failure rates

The universal downloader is a key asset. Do not regress to older download logic.

---

## Stage 4: Data Extraction

**Input:** Downloaded PDFs + topic config
**Tool:** Multi-model extraction (Claude + Gemini + Kimi, consensus)
**Output:** `4_extract/summary.csv`

Process:
- Send each PDF to 2-3 LLM extractors with topic-specific prompts
- Each extractor returns structured JSON: treatment/control means, n, variance, outcome, moderators
- Consensus engine merges outputs (majority rules, with tiebreaker)
- Broad extraction — capture all potentially relevant comparisons

Extraction schema per row:
```json
{
  "paper_id": "string",
  "outcome": "free text describing what was measured",
  "outcome_unit": "string",
  "source_type": "table|figure|text",
  "table_or_figure": "Table 2",
  "treatment_mean": 12.5,
  "control_mean": 10.2,
  "treatment_n": 4,
  "control_n": 4,
  "variance_type": "SE|SD|LSD|CI|CV",
  "variance_value": 1.3,
  "se_treatment": 1.3,
  "sd_treatment": null,
  "se_control": 1.1,
  "sd_control": null,
  "treatment_description": "free text",
  "control_description": "free text",
  "confidence": "high|medium|low",
  "notes": "free text",
  "title": "paper title",
  "moderators": {
    "mod_crop_type": "wheat",
    "mod_experiment_type": "field",
    "mod_climate_zone": "temperate",
    "...": "..."
  }
}
```

---

## Stage 5: Deterministic QC (Hard Filters)

**Input:** `summary.csv`
**Output:** `summary_qc.csv` + `qc_audit.json`

Programmatic checks (no LLM needed):

1. **Structural completeness**
   - Both means present and numeric
   - At least one mean is positive (for lnRR)
   - Both means positive for ratio-based effect sizes

2. **Variance integrity**
   - SE/SD/LSD conversion to common SD
   - CV bounds check (flag if CV > 100% or CV < 1%)
   - Missing variance classification (present / imputable / missing)

3. **Duplicate detection**
   - Same paper + same outcome + same treatment/control means = likely duplicate
   - Same paper + table row + figure row with matching values = text/table duplicate
   - Pooled summary + individual years both present = non-independence

4. **Effect size computation**
   - lnRR = ln(treatment_mean / control_mean)
   - Variance of lnRR from SD and n
   - Simple percentage change as backup
   - Flag extreme values (|lnRR| > 2, i.e., >600% or <-86%)

5. **Provenance tracking**
   - Each row retains source_type, table_or_figure, confidence
   - Paper-level metadata preserved

---

## Stage 6: LLM Semantic Adjudication

**Input:** QC-passed rows + topic config
**Model:** Claude Opus 4.6
**Output:** `adjudication_decisions.jsonl` + `adjudicated_kept.csv`

This is the core of V2. The LLM sees each row alongside a compact topic brief
and makes structured semantic judgments.

### Prompt Structure

```
System: You are validating extracted meta-analysis rows after PDF extraction.
Your job is to decide whether each row should be kept, excluded, flagged,
or treatment/control swapped.

Topic Brief: {{generated from config}}

Row: {{extracted row fields as JSON}}

For each row, judge:
1. Does the treatment match the configured intervention?
2. Does the control match the configured comparator?
3. Does the outcome match the configured primary outcome?
4. Does the row match the benchmark estimand?
5. Is there evidence treatment and control were swapped?

Return JSON only.
```

### Output Schema

```json
{
  "row_id": "string",
  "decision": "keep|exclude|flag|swap_treatment_control",
  "intervention_match": "yes|partial|no",
  "comparator_match": "yes|partial|no",
  "outcome_match": "yes|partial|no",
  "estimand_match": "yes|partial|no",
  "needs_tc_swap": false,
  "normalized_outcome_class": "grain_yield|harvest_yield|biomass|quality_trait|system_productivity|other",
  "normalized_study_setting": "field|greenhouse|pot|mixed|unknown",
  "normalized_estimand_class": "string",
  "exclusion_reason": "string or null",
  "rationale_short": "1-2 sentences"
}
```

### Decision Policy

- `keep` if intervention, comparator, and outcome all match >= `partial`
- `exclude` if intervention or comparator is `no`
- `exclude` if outcome is `no`
- `flag` if estimand is `partial` or row meaning is ambiguous
- `swap_treatment_control` if T/C clearly reversed vs config
- Be strict: better to exclude ambiguous rows than include off-target ones

---

## Stage 7: Effector Normalization

**Input:** Kept rows from adjudication
**Tool:** LLM (Claude) + regex patterns
**Output:** `effector_labels.jsonl`

Normalize benchmark-relevant moderators:

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

Purpose:
- Enable benchmark-aligned secondary analyses
- Explain composition differences between pipeline sample and benchmark
- Support pre-specified subgroup analyses

---

## Stage 8: Synthesis

**Input:** Adjudicated + labeled rows
**Method:** DerSimonian-Laird random effects on lnRR
**Output:** `synthesis_results.json`

### Primary Analysis
- Pool all kept rows using inverse-variance weighting
- Report: pooled effect %, 95% CI, I^2, tau^2, k (number of effect sizes)
- Compare to benchmark: direction match, CI overlap, absolute difference

### Variance Handling
Conversion hierarchy:
1. SD + n -> variance of lnRR directly
2. SE + n -> SD = SE * sqrt(n) -> variance
3. LSD + n -> SE_diff = LSD / (t_crit * sqrt(2)), SD = SE_diff * sqrt(n)
4. CI -> SD from CI width
5. CV + mean -> SD = CV * mean / 100
6. Missing variance -> row included in simple mean but excluded from weighted analysis

### Non-Independence Handling
- Flag multi-row papers
- Compute paper-level average for sensitivity analysis
- Report both observation-level and paper-level pooled estimates

---

## Stage 9: Diagnostics & Reporting

**Output:** Comprehensive diagnostic report per topic

### Automatic Diagnostics

1. **Paper influence analysis**
   - Leave-one-out sensitivity (drop each paper, recalculate)
   - Identify most influential papers

2. **Benchmark alignment analysis**
   - Benchmark-aligned subset: field-only, direct-yield-only
   - Compare aligned subset to benchmark

3. **Quality sensitivity**
   - High-confidence-only pooled estimate
   - Table-only pooled estimate (no figure-extracted rows)

4. **Funnel plot / Egger's test**
   - Publication bias / small-study effects

5. **Composition comparison**
   - Crop class distribution: pipeline vs benchmark
   - Setting distribution: pipeline vs benchmark
   - Climate distribution: pipeline vs benchmark
   - Management context distribution

6. **Failure taxonomy**
   When pipeline disagrees with benchmark, classify the likely cause:
   - Extraction/filtering error (fixable)
   - Corpus composition difference (structural)
   - Estimand mismatch (structural)
   - OA access limitation (structural)

---

## Input Specification: Topic Config

Each topic is defined by a JSON config with these required fields:

```json
{
  "review_id": "string",
  "title": "string",
  "research_question": "string",
  "pico": {
    "population": {
      "description": "string",
      "search_terms": ["..."],
      "exclude_terms": ["..."]
    },
    "intervention": {
      "description": "string",
      "search_terms": ["..."]
    },
    "comparator": {
      "description": "string",
      "search_terms": ["..."]
    },
    "outcome": {
      "primary": {
        "description": "string",
        "search_terms": ["..."]
      }
    }
  },
  "tc_confusion_warnings": ["..."],
  "extraction_priorities": ["..."],
  "benchmark": {
    "source": "Author et al. Year (Journal DOI)",
    "published_pooled_effect": {
      "estimate": -19.2,
      "ci_lower": -21.5,
      "ci_upper": -16.9,
      "unit": "percent_change",
      "notes": "string"
    }
  }
}
```

---

## Input Specification: Benchmark Spec

A structured encoding of the benchmark paper's operational definitions.
See `BENCHMARK_SPEC_TEMPLATE.md` for the full template.

Key fields:
- Intervention definition (required features, excluded variants)
- Comparator definition (required features, excluded variants)
- Outcome hierarchy (what counts, what doesn't)
- Study setting restrictions
- Critical moderators
- Known estimand traps
- Subgroup logic for secondary analyses

---

## Canonical Ontologies

### Outcome Classes
- `grain_yield` — harvested grain (wheat, maize, rice, barley)
- `harvest_yield` — other harvested crop product (fruit, tuber, pod, seed)
- `biomass` — total/shoot/plant dry matter
- `quality_trait` — protein, mineral concentration, energy content
- `system_productivity` — LER, equivalent yield, combined system output
- `component_yield` — individual crop yield within a mixed system
- `other`

### Study Settings
- `field` — field experiments, on-farm trials
- `greenhouse` — greenhouse, glasshouse, screenhouse
- `pot` — pot experiments, container trials
- `mixed` — study includes both field and pot components
- `unknown`

### Climate Classes
- `tropical`, `subtropical`, `temperate`, `mediterranean`
- `semi_arid`, `arid`, `boreal`, `unknown`

### Estimand Contexts
- `benchmark_aligned` — row measures exactly what the benchmark measures
- `partially_aligned` — row is related but not identical
- `misaligned` — row measures something different from benchmark target
- `unknown`

---

## Success Criteria (For Preregistered Evaluation)

### Primary
1. **Direction agreement:** Pipeline pooled effect has same sign as benchmark
   (success threshold: >= 4/6 topics or >= 80% of topics)

2. **CI overlap:** Pipeline 95% CI includes the benchmark point estimate
   (success threshold: >= 3/6 topics or >= 50% of topics)

### Secondary
3. **Absolute difference:** |pipeline - benchmark| in percentage points
4. **Replication ratio:** pipeline / benchmark
5. **Benchmark-aligned subset performance:** Does the aligned subset improve agreement?
6. **Failure taxonomy:** Can failures be diagnostically explained?

### Process Metrics
7. **Row retention rate:** % of extracted rows passing adjudication
8. **Download coverage:** % of screened papers successfully downloaded
9. **Extraction completeness:** % of papers with >0 extracted rows
10. **Variance coverage:** % of rows with usable variance for weighting

---

## What V2 Fixes vs V1

| Problem in V1 | Fix in V2 |
|---------------|-----------|
| Semantic over-inclusion | LLM adjudication stage |
| Wrong outcome leakage | Canonical outcome ontology |
| Wrong intervention/comparator | Config-driven semantic matching |
| Estimand mismatch | Explicit estimand classification |
| Study setting mismatch | Setting normalization + sensitivity |
| No duplicate control | Programmatic duplicate detection |
| No non-independence handling | Paper-level aggregation option |
| No benchmark alignment analysis | Effector-based alignment labeling |
| Post-hoc topic-specific rules | Universal config-driven architecture |

---

## Implementation Status

| Component | Status | Location |
|-----------|--------|----------|
| Literature search | Working (V1) | `universal_downloader.py` |
| Abstract screening | Working (V1) | Pipeline stage 2 |
| PDF download | Working (V1) | `universal_downloader.py` |
| Data extraction | Working (V1) | Pipeline stage 4 |
| Deterministic QC | Prototype | `pico_validate.py` |
| LLM adjudication | Prototype (keyword) | `codex/adjudicate_universal.py` |
| LLM adjudication | Design only (Claude) | `codex/CLAUDE_UNIVERSAL_POSTPROCESS_PROMPT.md` |
| Effector normalization | Prototype | `codex/normalize_effectors_universal.py` |
| Synthesis | Working | `resynthesize_all.py` |
| Diagnostics | Partial | `replication_formal_stats.py` |
| Benchmark spec | Template only | `codex/BENCHMARK_SPEC_TEMPLATE.md` |
| Topic scoring | Template ready | `codex/score_topic_candidates.py` |

---

## Next Steps

1. Upgrade adjudication from keyword-based to Claude Opus 4.6 LLM-based
2. Integrate benchmark spec ingestion into extraction prompts
3. Add non-independence detection and paper-level sensitivity
4. Select new prospective topic set using scoring criteria
5. Preregister V2 evaluation
6. Run V2 on preregistered topics
7. Write paper centered on V2 results
