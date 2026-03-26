# Universal Pipeline Improvements — 2026-03-26

## What V1 Taught Us (Applies to ALL Topics)

The full LLM adjudication run over 3,460 rows across 6 topics revealed three classes of
systematic extraction errors that are not topic-specific. They arise from structural features
of scientific papers (what gets reported) and keyword-based matching limitations (what the
pipeline treats as a match). These errors contaminate ANY synthesis topic, regardless of the
target intervention or crop.

### Error Class 1 — Yield Components Passed as Yield

Extraction models consistently extract table rows reporting yield components
(1000-grain weight, grains per spike, hectoliter weight, ear length) as primary yield outcomes
because these rows appear in the same tables as grain yield and use the word "yield" in
compound noun phrases. Keyword filters cannot distinguish compound terms.

**Scope observed in V1:** ~168 rows in notill_tillage, ~111 rows in biochar_crop_yield,
present in all 6 topics. Total contamination: estimated >300 rows across the run.

### Error Class 2 — Morphological and Quality Traits Passed as Yield

Papers about agronomic interventions (mycorrhiza, biochar, no-till, organic) routinely report
plant height, SPAD/chlorophyll, LAI, tiller number, and nutrient concentrations alongside
yield. Keyword adjudicators pass these because the papers are correctly identified as
on-topic — the PICO screening is correct, but the extracted *rows* are off-estimand.

**Scope observed in V1:** ≥47 plant height rows in mycorrhiza_yield alone; widespread in
biochar_crop_yield. Nutrient uptake rows appear in legume_rotation and biochar topics.

### Error Class 3 — Confounded Interventions Not Flagged

Comparisons where the treatment arm contains additional inputs beyond the focal intervention
(e.g., no-till + cover crop vs conventional fallow; mycorrhiza + NPK vs untreated control)
cannot isolate the effect of the target intervention. Keyword matching cannot detect
asymmetric inputs — it only checks whether the intervention term appears at all.

**Scope observed in V1:** Most severe in notill_tillage (~confounded rows contributed to
sign error in that topic's pooled estimate vs benchmark).

---

## The Three Universal Fixes Applied Today

### Fix 1 — Universal Adjudication Prompt Extension (`adjudicate_llm_universal.py`)

**Location:** SYSTEM_PROMPT, inserted after Decision Policy, before Output Format.

**What was added:** A new section titled "UNIVERSAL EXTRACTION ERROR PATTERNS" containing
five numbered checks the LLM must apply to every row regardless of topic config:

1. Yield components ≠ yield (1000-grain weight, grains/spike, ear length, etc.)
2. Morphological traits ≠ yield (plant height, LAI, SPAD, tiller number, root biomass)
3. Quality traits ≠ yield (nutrient concentration, protein %, oil %, unless config
   explicitly includes them as primary outcomes)
4. Confounded intervention — T/C must differ in exactly one agronomic factor
5. Per-plant values without area conversion — flag, do not auto-exclude

Each check specifies the exact decision fields to set (outcome_match, normalized_outcome_class,
exclusion_reason, decision) so the LLM output is consistent and parseable.

**Why this is a pipeline fix, not a config fix:** These checks apply identically across all
agronomic yield topics. A config-level fix would require duplicating the same language in
every config.json and trusting that future topics add it. The prompt is the shared enforcement
point.

### Fix 2 — Universal Non-Yield QC Flag (`qc_hard_filters.py`)

**Location:** New Check 8, inserted before the Summary block in `run_qc()`.

**What was added:** A regex-based flag (`_qc_possible_non_yield`) that scans
`outcome_variable` against 18 patterns covering:
- Plant height, stem diameter, LAI, tiller number, SPAD
- Root length/biomass/weight
- 1000-grain weight, hectoliter weight, grains per spike/panicle, ear/spike length
- Mycorrhizal colonisation, germination rate
- Nutrient concentration/content/uptake, protein/oil/starch content

This flag is **NOT an auto-exclusion** — it surfaces rows for LLM adjudication. The LLM
makes the final call (a nutrient-concentration row might legitimately belong in a mineral-
nutrition topic even though it matches the non-yield pattern). The QC filter's job is to
ensure these rows reach the LLM rather than being silently passed through.

**Why this is a pipeline fix, not a config fix:** The regex patterns are universal — they
describe what is never harvestable yield regardless of topic. A config blacklist would need
to be exhaustively maintained per topic and would inevitably miss patterns in future topics.

**Audit output:** Check 8 appears in `qc_audit.json` with a count of flagged rows and the
number of patterns checked, so every run is auditable.

### Fix 3 — This Document

**Why document at the pipeline level:** Future contributors adding new topics need to know
that these checks exist and are intentionally universal. Without this document, the temptation
is to work around Check 8 flags by adding topic-specific blacklists, which defeats the
purpose of pipeline-level enforcement.

---

## Why These Are Pipeline-Level Fixes, Not Config-Level Fixes

A config-level fix (adding a `non_yield_exclusions` list to each config.json) would:

1. **Require maintenance per topic** — Each new topic must remember to include the list.
2. **Be inconsistently applied** — Topics without the list would silently pass contaminated rows.
3. **Miss generalisation** — The error patterns are *structurally identical* across all
   agronomic yield topics; they are not topic-specific knowledge.
4. **Obscure the lesson** — Hiding the fix inside per-topic configs makes it invisible to
   anyone auditing the pipeline as a whole.

The pipeline scripts (`adjudicate_llm_universal.py`, `qc_hard_filters.py`) are the correct
place because they run on *every* topic, every time, with no per-topic override needed.

---

## What Remains Legitimately Topic-Specific

The following should remain in config.json files and should NOT be lifted to pipeline level:

| Config field | Why it stays in config |
|---|---|
| `outcome_description` | What counts as the primary outcome differs by topic (LER for intercropping; grain yield for rotation) |
| `intervention_description` | The focal treatment is by definition topic-specific |
| `tc_confusion_warnings` | T/C swap patterns depend on how papers describe the specific intervention |
| `benchmark_source` / `benchmark_notes` | The reference meta-analysis is specific to each topic |
| `expected_direction` | Sign of expected effect is topic-specific |
| Quality metrics as primary outcomes | A mineral-nutrition topic legitimately includes nutrient concentration; a yield topic does not — this distinction requires a config flag |
| Estimand clarification (e.g., rotation vs legume yield) | The research question framing is topic-specific; the pipeline cannot infer it |

The rule of thumb: if the same sentence could appear verbatim in every topic's config, it
belongs in the pipeline instead.

---

## What Remains Unresolved (Future Work)

1. **Intercropping estimand mismatch** — The pipeline compares component-crop yield vs
   sole-crop yield of the same species. The benchmark (+22%) uses LER (Land Equivalent Ratio).
   Fix requires config-level estimand change to LER, not a universal pipeline fix.

2. **Organic/notill benchmark gap** — Part of the gap is structural (transitional vs
   certified organic; short-term vs long-term no-till). These are sampling/inclusion criteria
   issues, not extraction errors, and must be addressed in the topic config's `include_criteria`.

3. **Per-plant conversion** — Check 8 flags per-plant rows but cannot automatically convert
   them using planting density (which would require reading the paper). The LLM adjudicator
   currently flags these; future work could add a density-lookup step for papers that provide
   planting density in the moderator fields.

4. **Variance type detection** — LSD-only papers remain in the dataset with `variance_missing`
   status. The LSD → SD conversion formula exists in `qc_hard_filters.py` but requires `n`
   (sample size). Papers that omit `n` but report LSD cannot be weighted without imputation.

---

*Written: 2026-03-26*
*Author: pipeline improvement session — universal fixes only, no topic-specific config changes*
