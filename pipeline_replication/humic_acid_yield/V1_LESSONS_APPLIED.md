# V1 Lessons Applied — humic_acid_yield

**Written:** 2026-03-26
**Source:** LLM_VS_KEYWORD_COMPARISON_2026-03-26.md (6 topics, 3,460 rows adjudicated)
**Purpose:** Document how each V1 lesson was translated into concrete changes before the humic_acid_yield pilot run.

---

## V1 Lessons and How Each Was Applied

### Lesson 1 — Yield components pass as yield
**What V1 showed:** Keywords match compound terms containing "yield" or "grain" — e.g.,
`1000-grain weight`, `hectoliter weight`, `grain weight per plant`, `ear length`. The LLM
adjudicator correctly excluded these as `component_yield`, not harvestable area yield.
This was the single most frequent error class across all 6 topics (>150 rows affected),
with the worst cases in notill_tillage (~6 rows) and biochar_crop_yield (~111 rows of
yield components + plant height combined).

**Applied to humic_acid_yield:**
1. `config.json` — `extraction_priorities` item 6 strengthened: now reads
   "DO NOT extract yield components, growth traits, or physiological measurements."
2. `config.json` — Added `non_yield_exclusions` list (40 specific terms) covering
   1000-grain weight, hectoliter weight, grains per spike, ear/spike length, tiller number,
   SPAD, chlorophyll a/b, LAI, root length/biomass/dry weight, nutrient concentration/uptake,
   NUE, enzyme activity, germination rate, and others.
3. `benchmark_spec.md` — Added `## Extraction Blacklist` section (17-row table) with
   explicit "why excluded" rationale for each measurement. Includes a humic acid-specific
   note about NUE/N-uptake co-measurement, which is extremely common in HA literature.
4. `adjudicate_llm_universal.py` — Added `## COMMON EXTRACTION ERRORS TO CATCH` to the
   system prompt. Error 1 specifically instructs the adjudicator to set
   `normalized_outcome_class="component_yield"` and `exclusion_reason="yield_component"`
   for these cases.

---

### Lesson 2 — Non-yield plant traits (plant height, chlorophyll, P uptake)
**What V1 showed:** `plant height` was extracted and kept in >47 rows in mycorrhiza_yield
alone; chlorophyll content, number of leaves, soil organic matter, and soil pH were all
passed by the keyword filter. In mycorrhiza_yield, the LLM exclusion of 213 `non_yield_outcome`
rows brought the pooled effect from +31.4% (keyword, biased by growth traits) to +26.0%
(LLM, closer to the Hoeksema benchmark of +23%).

**Applied to humic_acid_yield:**
1. `config.json` `non_yield_exclusions` covers all plant physiological traits observed
   in V1: SPAD, chlorophyll a/b, LAI, photosynthesis rate, stomatal conductance, water
   use efficiency, enzyme activity, soil OM, soil pH, microbial biomass, colonization %.
2. `benchmark_spec.md` `## Extraction Blacklist` calls out SPAD/chlorophyll explicitly
   with the note that HA papers almost universally co-measure these alongside yield.
3. `benchmark_spec.md` Section 5 `excluded_outcomes` (pre-existing) plus the new
   Blacklist table are now consistent and mutually reinforcing.
4. `adjudicate_llm_universal.py` Error 1 in the new section covers this class broadly
   via `outcome_match="no"`, `normalized_outcome_class="component_yield"`.

---

### Lesson 3 — Per-plant values without area conversion cause upward bias
**What V1 showed:** In biochar_crop_yield, 25 rows with `per_plant_unit` were excluded by
the LLM adjudicator. In organic_yield_gap, 9 `per_plant` rows were excluded. Per-plant
values from pot experiments inflate apparent effects because pot density is often higher
than field density, and pot experiments tend to show larger fertilizer responses.

**Applied to humic_acid_yield:**
1. `config.json` `pico.outcome.unit` already lists `g/plant` as acceptable. This is
   intentional for this topic (many legitimate HA studies use pot designs), but the
   unit is now tracked explicitly to enable sensitivity analysis.
2. `benchmark_spec.md` Section 6 `setting_notes` flags: "Sensitivity analysis: field-only
   subset vs all settings."
3. `adjudicate_llm_universal.py` Error 2 instructs the adjudicator to flag per-plant
   rows with `normalized_outcome_class="biomass"` rather than silently keeping them, so
   they can be analysed as a separate stratum.

---

### Lesson 4 — Confounded interventions (T/C isolation failure)
**What V1 showed:** In notill_tillage, 123 rows were flagged by the LLM for confounded
interventions (no-till + cover crop vs conventional; no-till + mulch vs conventional),
preventing isolation of the tillage effect. In biochar_crop_yield, 67 rows were excluded
for `intervention_mismatch`. These caused the notill and biochar pooled effects to diverge
from benchmark.

**Applied to humic_acid_yield:**
1. `config.json` — Added `intervention_isolation_check` key with explicit instruction:
   "The humic acid treatment must be the ONLY differentiating factor between T and C.
   Flag any row where HA is combined with seaweed extract, amino acids, or microbial
   inoculants without a matching control arm that also receives those additives."
2. `config.json` `tc_confusion_warnings` (pre-existing) already covered HA+NPK vs NPK;
   the new `intervention_isolation_check` extends this to multi-component biostimulant
   products (a known HA literature problem).
3. `benchmark_spec.md` Section 3 `intervention_ambiguity_notes` covers "biostimulant"
   bundles; Section 4 `comparator_ambiguity_notes` covers factorial HA × NPK designs.
4. `adjudicate_llm_universal.py` Error 3 explicitly instructs the adjudicator to look
   at `treatment_description` and `control_description` for asymmetric inputs, and to
   set `intervention_match="no"/"partial"` with `exclusion_reason="confounded_intervention"`.

---

### Lesson 5 — Intercropping/legume estimand confusion (topic-specific, not directly applicable)
**What V1 showed:** In legume_rotation, 12 rows described simultaneous intercropping, not
rotation, and 64 rows captured the legume's OWN yield rather than the subsequent cereal
yield. These are topic-specific to rotation estimands.

**Applied to humic_acid_yield:** Not directly applicable — humic_acid_yield has a simpler,
single-estimand structure (HA vs no-HA on direct crop yield). The general principle
(LLM should verify the estimand matches the benchmark's research question) is already
embedded in `adjudicate_llm_universal.py` Decision Policy point 4 ("Does the row match
the benchmark estimand?") and the new Error 3 check.

---

## What Risks Remain Despite These Fixes

### Risk 1 — NUE masquerading as yield in paper titles and table headers
Several HA papers titling themselves as "yield and NUE study" have tables with
`Grain yield`, `NUE`, `N uptake` in adjacent columns. The extraction model may
capture all three. The `non_yield_exclusions` list covers NUE explicitly, but the
LLM adjudicator must still parse each row's `outcome` field carefully. If extraction
labels an NUE row as "yield" in the `outcome` field, the adjudicator may miss it.
**Residual risk: medium.**

### Risk 2 — Biostimulant bundle products
The HA literature contains many commercial products (e.g., Humiforte, Huminrich,
Terra-Sorb) that contain humic acid plus other active ingredients. If authors do not
disclose the full composition, neither the extraction model nor the adjudicator can
reliably detect confounding. The `intervention_isolation_check` helps but cannot catch
undisclosed bundling.
**Residual risk: medium — mitigated by flagging (`flag` rather than `exclude`).**

### Risk 3 — Unit heterogeneity in vegetable/fruit crop studies
Vegetable and fruit yield is often reported in `t/ha fresh weight`, `kg/plant`, or
`number of fruits × average fruit weight`. These are legitimate yield metrics but may
not align with the benchmark's grain-yield-centric database. The per-plant flag (Lesson 3
fix) will surface these for sensitivity analysis, but the unweighted pooled effect may
still be influenced by this heterogeneity.
**Residual risk: low-medium — addressed via sensitivity analysis in benchmark_spec.md.**

### Risk 4 — Variance type confusion (SE vs SD)
The `non_yield_exclusions` and adjudication improvements address outcome classification,
but not variance quality. If papers report LSD without SE/SD values, or report SE when
SD is needed for lnRR, the effect size distribution will be noisy. This is a known issue
from the broader pipeline (see CLAUDE.md variance sections). The config already sets
`variance_handling.missing_strategy = "exclude"` and includes LSD in acceptable types.
**Residual risk: medium — standard pipeline limitation, same as V1.**

---

## What These Fixes Are Expected to Prevent

| Fix | Expected prevention |
|-----|-------------------|
| `non_yield_exclusions` in config.json | Stops 1000-grain weight, SPAD, chlorophyll, root biomass from entering extraction pool |
| `intervention_isolation_check` in config.json | Stops confounded HA+seaweed or HA+microbe bundles from inflating effect estimate |
| `## Extraction Blacklist` in benchmark_spec.md | Gives LLM adjudicator a concrete lookup table during per-row review |
| `## COMMON EXTRACTION ERRORS TO CATCH` in adjudicate_llm_universal.py | Standardises how all future topics (including humic_acid_yield) are adjudicated; prevents the adjudicator from being permissive on yield components |
| Per-plant flagging (Error 2 in adjudication prompt) | Preserves per-plant rows for sensitivity analysis without silently inflating the pooled effect |

**Expected net effect:** Pooled humic acid yield estimate should be closer to the
Ma et al. 2024 benchmark (+12%) than it would have been with V1 pipeline defaults.
The main residual uncertainty is the biostimulant bundle issue and unit heterogeneity
across crop types.
