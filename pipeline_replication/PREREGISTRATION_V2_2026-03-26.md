# Pipeline V2: Autonomous Meta-Analysis Replication — Preregistration Document

**Title**: Pipeline V2 Evaluation: Autonomous Meta-Analysis Replication Across Six Agricultural Topics
**Preregistration Date**: 2026-03-26
**Document Status**: FROZEN — do not modify after 2026-03-26
**Frozen Architecture Reference**: `codex/PIPELINE_V2_FROZEN_2026-03-26.md`

---

## Section 1 — Study Overview

### 1.1 What Pipeline V2 Does

Pipeline V2 is a universal autonomous meta-analysis replication system. Given a topic configuration and a published benchmark meta-analysis, it proceeds from OpenAlex literature search to pooled effect estimation without manual paper-by-paper curation. The pipeline applies:

1. Broad literature retrieval (OpenAlex API)
2. LLM-based abstract screening against PICO criteria
3. PDF download via the universal downloader
4. Multi-model consensus data extraction from PDFs
5. Deterministic quality control (hard mathematical filters)
6. LLM semantic adjudication (the core V2 innovation — replaces keyword-based filtering)
7. Canonical effector labeling for benchmark-aligned subgroup analysis
8. DerSimonian-Laird random-effects synthesis
9. Automated diagnostics and failure classification

### 1.2 Why It Matters

Systematic reviews and meta-analyses underpin evidence-based agricultural policy. They are expensive (months of expert labor), hard to update, and subject to publication bias. An autonomous pipeline that reliably replicates the direction and approximate magnitude of published meta-analyses would dramatically reduce the time and cost of evidence synthesis, enable continuous updating, and allow rapid screening of emerging intervention questions.

### 1.3 What V2 Claims

V2 makes two confirmatory claims evaluated by preregistered primary analyses:

**P1 - Direction agreement**: Pipeline V2 will correctly identify the sign (positive/negative) of the benchmark published pooled effect for at least 4 of 5 preregistered confirmatory topics.

**P2 - CI overlap**: Pipeline V2 95% confidence interval will include the benchmark point estimate for at least 3 of 5 preregistered confirmatory topics.

These claims are evaluated against published benchmark meta-analyses chosen before any V2 results were produced.

**Note on evaluation set**: The 5 confirmatory topics are: amf_inoculation_yield, biochar_tropical_yield, elevated_co2_face_yield, legume_rotation, and cover_crop_corn_yield. humic_acid_yield is a non-preregistered pilot test (see Section 3, Topic 1 and the Pilot vs Confirmatory Distinction section below). Its results do not count toward P1 or P2.

### 1.4 What V2 Does Not Claim

V2 does not claim to precisely replicate effect size magnitudes. Structural gaps (papers behind paywalls, idiosyncratic study selection in the original meta-analysis) are expected and documented. The goal is directional concordance and approximate magnitude agreement, not exact numerical replication.

---

## Section 2 — Pipeline Architecture

Full architecture is frozen in `codex/PIPELINE_V2_FROZEN_2026-03-26.md`. Summary below.

### 2.1 Stage Overview

| Stage | Name | Method | Key Design Decision |
|-------|------|--------|---------------------|
| 1 | Literature Search | OpenAlex API (deterministic) | Broad retrieval; precision handled downstream |
| 2 | Abstract Screening | LLM (Gemini Flash) | Semantic PICO matching; replaces keyword filter |
| 3 | PDF Download | Universal Downloader (deterministic) | Proven V1 asset; multi-source retry |
| 4 | Data Extraction | Multi-model consensus (Claude + Gemini) | Broad extraction; strict post-processing |
| 5 | Deterministic QC | Python code (no LLM) | Math checks, duplicate detection, plausibility flags |
| 6 | LLM Adjudication | Claude (LLM) | 5-criterion semantic keep/exclude/flag/swap decision |
| 7 | Effector Normalization | Claude + regex (hybrid) | Canonical labels for benchmark-aligned subgroup analysis |
| 8 | Synthesis | DerSimonian-Laird RE (deterministic) | lnRR, 95% CI, I2, tau2, k |
| 9 | Diagnostics | Python code (deterministic) | LOO influence, composition comparison, failure taxonomy |

### 2.2 Core V2 Innovation: LLM Semantic Adjudication (Stage 6)

Stage 6 reviews each extracted row against the topic PICO and benchmark estimand, making structured decisions: keep / exclude / flag / swap_treatment_control.

LLM adjudication handles 5 failure categories that keyword filters cannot:
1. Intervention isolation (is the intervention isolated, or bundled with other inputs?)
2. Outcome label heterogeneity (does the outcome label qualify as the target estimand?)
3. Comparator identity (is the control NPK-matched vs. zero-fertilizer?)
4. Estimand verification (per-plant vs. per-area yield)
5. Contextual plausibility (is an extreme effect credible given study context?)

---

## Section 3 — Pre-registered Topics (6)

Topics were selected using an 8-dimension scoring system before any V2 extraction results were produced. Benchmark papers were identified before topic selection. Scoring details: `codex/outputs/topic_candidates/scored_candidates.csv`.

---

### Topic 1: humic_acid_yield (PIPELINE VALIDATION TEST — NOT CONFIRMATORY)

> **IMPORTANT**: humic_acid_yield is NOT part of the preregistered confirmatory evaluation. It is a pipeline validation test run BEFORE the preregistered evaluation begins, after all lessons are learned from V1 LLM re-adjudication. Its results will be used solely to fix pipeline bugs and will NOT be included in P1/P2 success criteria. The 5 preregistered confirmatory topics are Topics 2–6.

- **Config**: `humic_acid_yield/config.json`
- **Benchmark paper**: Ma, Cheng & Zhang (2024). "The Impact of Humic Acid Fertilizers on Crop Yield and Nitrogen Use Efficiency: A Meta-Analysis." *Agronomy* (MDPI) 14(12):2763.
- **Benchmark DOI**: 10.3390/agronomy14122763
- **Benchmark effect**: +12% crop yield increase (k=93 articles, 383 yield observations)
- **Estimand**: Exogenous humic acid application vs. no-HA control; primary outcome = crop yield (grain, fruit, tuber, or harvestable biomass)
- **Expected direction**: Positive (yield increase)
- **Known traps**: HA co-applied with other biostimulants (seaweed, PGPR, chitosan); HA as measured soil variable not applied treatment; reviews dominate search results; pot experiments inflate effects vs. field
- **Why selected**: Fully OA benchmark (MDPI); clean estimand; never tested in V1; pipeline validation test topic for V2 operational refinement before confirmatory run

---

### Topic 2: amf_inoculation_yield

- **Config**: `amf_inoculation_yield/config.json`
- **Benchmark paper**: Wu et al. (2022). "Mycorrhizal fungi improve the yield of legumes and non-legumes in field experiments." *PeerJ*.
- **Benchmark effect**: +23% yield increase [95% CI: 16-30%] (k=21 field studies, n=546)
- **Estimand**: AMF inoculation vs. non-inoculated control; primary outcome = crop yield in rainfed field conditions
- **Expected direction**: Positive (yield increase)
- **Known traps**: Mixed mycorrhizal products; AMF + PGPR combinations; pot experiments inflate effects; inoculum establishment failure under high soil P
- **Why selected**: PeerJ fully OA; rainfed restriction creates cleaner estimand; +23% is a strong testable signal

---

### Topic 3: biochar_tropical_yield

- **Config**: `biochar_tropical_yield/config.json`
- **Benchmark paper**: Jeffery et al. (2017). "Biochar boosts tropical but not temperate crop yields." *Environmental Research Letters* (ERL / IOP Publishing).
- **Benchmark effect**: +25% yield increase in tropical soils (benchmark tropical subgroup estimate)
- **Estimand**: Biochar soil amendment vs. unamended control; primary outcome = crop yield in tropical or subtropical settings
- **Expected direction**: Positive for tropical soils
- **Known traps**: Temperate studies show near-zero effect (must use tropical subgroup); biochar + compost combinations; multiple application rates per study; old papers may report only % change
- **Why selected**: ERL fully OA (IOP); tropical restriction creates testable subgroup; demonstrates pipeline subgroup analysis capability

---

### Topic 4: legume_rotation

- **Config**: `legume_rotation/config.json`
- **Benchmark paper**: Zhao et al. (2022). "Legume-based rotations increase crop yields and reduce N2O emissions." *Nature Communications*.
- **Benchmark DOI**: 10.1038/s41467-022-28412-9
- **Benchmark effect**: +20% yield increase in post-legume cash crops (k=116 studies)
- **Estimand**: Cash crop following legume vs. cash crop in continuous monoculture; primary outcome = cash crop yield (NOT legume yield)
- **Expected direction**: Positive (yield increase in subsequent cash crop)
- **Known traps**: Extracting legume yield instead of subsequent cash crop; simultaneous intercropping vs. sequential rotation; multi-year cumulative vs. single-year yield
- **Why selected**: Carried forward from V1 (direction correct at +17.7%); Nature Comms fully OA; V2 expected to improve magnitude accuracy

---

### Topic 5: elevated_co2_face_yield

- **Config**: `elevated_co2_face_yield/config.json`
- **Benchmark paper (PRIMARY)**: Long SP, Ainsworth EA, Leakey ADB, Nosberger J, Ort DR (2006). "Food for Thought: Lower-Than-Expected Crop Yield Stimulation with Rising CO2 Concentrations." *Science* 312(5782):1918-1921. DOI: 10.1126/science.1114722. PMID: 16809532. Open access.
- **Benchmark paper (SECONDARY)**: Ainsworth & Long (2021). "30 years of free-air carbon dioxide enrichment (FACE): What have we learned about future crop productivity and its potential for adaptation?" *Global Change Biology* 27(1):27-49. DOI: 10.1111/gcb.15375. PMID: 33135850. Closed access; CI not obtainable. Retained as secondary reference.
- **Benchmark effect**: ~+8% for FACE C3 grain cereals (wheat, rice); ~+13% for FACE C3 legumes (soybean); ~0% for C4 crops (maize, sorghum). Key finding: FACE shows ~50% less yield stimulation than prior enclosure studies. Long et al. 2006 is a perspective/synthesis article; formal 95% CI not reported. Benchmark range for CI overlap evaluation: approximately +5% to +13% for FACE C3 cereals.
- **Estimand**: Elevated CO2 (~+200 ppm above ambient) vs. ambient CO2; FACE studies only; primary outcome = grain yield; C3 crop focus
- **Expected direction**: Positive for C3 crops; near-zero for C4 crops
- **Known traps**: CO2 x nutrient interaction confounds; elevated ozone co-treatments; per-plant vs. per-area yield; OTC studies inflate effect vs. FACE (excluded by design)
- **Study design restriction**: FACE (Free-Air CO2 Enrichment) experiments only; OTC and chamber studies excluded to match benchmark scope. This is a pre-registered design decision.
- **Why selected**: Policy-relevant IPCC topic; 30 years of FACE data = large corpus; C3/C4 split tests subgroup analysis; carried forward from V1
- **Benchmark update note**: Updated 2026-03-26, pre-run. Primary benchmark changed from Ainsworth & Long 2021 to Long et al. 2006 because Long et al. 2006 is FACE-specific (matches pipeline scope restriction), open-access with obtainable benchmark range, and the canonical FACE cereal reference. Ainsworth & Long 2021 retained as secondary. Change made before any pipeline data collection for this topic.

---

### Topic 6: cover_crop_corn_yield

- **Config**: `cover_crop_corn_yield/config.json`
- **Benchmark paper**: Marcillo & Miguez (2017). "Corn yield response to winter cover crops: An updated meta-analysis." *Journal of Soil and Water Conservation* (JSWC).
- **Benchmark effect**: -1% to +3% (near-zero; slight positive in most conditions)
- **Estimand**: Corn grain yield following winter cover crop vs. no cover crop; US/Canada primary focus
- **Expected direction**: Near-zero positive (potentially null)
- **Known traps**: Confusing cover crop biomass with subsequent corn yield; including cash-crop-as-cover studies
- **Why selected**: Near-zero benchmark tests whether pipeline avoids spurious effects; hard case that constrains false-positive rate; carried forward from V1

---

## Section 4 — Pre-specified Analyses

All analyses were specified before any V2 extraction results were produced. No post-hoc analyses will be conducted without clearly labeling them as exploratory.

### 4.1 Primary Confirmatory Analyses

Evaluated on the 5 preregistered confirmatory topics only (Topics 2–6: amf_inoculation_yield, biochar_tropical_yield, elevated_co2_face_yield, legume_rotation, cover_crop_corn_yield). humic_acid_yield results are excluded from P1 and P2.

**P1 - Direction Agreement**
- Hypothesis: Pipeline V2 produces a pooled effect with the same sign as the benchmark for >= 4/5 confirmatory topics.
- Analysis: Compare sign of pipeline pooled effect vs. benchmark reported effect, per topic.
- Success threshold: >= 4/5 topics
- Partial success: >= 3/5 topics
- Failure: < 3/5 topics

**P2 - CI Overlap**
- Hypothesis: Pipeline V2 95% CI includes the benchmark point estimate for >= 3/5 confirmatory topics.
- Analysis: Check whether benchmark point estimate falls within [pipeline CI lower, pipeline CI upper].
- Success threshold: >= 3/5 topics
- Partial success: >= 2/5 topics
- Failure: <= 1/5 topics

### 4.2 Secondary Exploratory Analyses

**S1 - Absolute Gap**

For each topic: compute |pipeline pooled estimate - benchmark point estimate| in percentage points. No formal success threshold. Target for reporting: <= 10pp in >= 4/5 confirmatory topics.

**S2 - Benchmark-Aligned Subset**

For each topic: filter to rows labeled benchmark_aligned by Stage 7 effector normalization (field setting + direct yield outcome + isolated intervention + appropriate comparator). Compare aligned-subset pooled estimate vs. full-dataset estimate vs. benchmark. Hypothesis (exploratory): alignment filter improves agreement in >= 3/5 confirmatory topics.

**S3 - V2 vs V1 Improvement (Carried-Forward Topics)**

For 3 topics carried forward from V1 (legume_rotation, elevated_co2_face_yield, cover_crop_corn_yield): record V1 absolute gap and V2 absolute gap. Target: V2 gap < V1 gap for >= 2/3 carried-forward topics.

---

## Section 5 — Success / Failure Criteria

**Primary Success**: Both P1 (>= 4/5 direction) AND P2 (>= 3/5 CI overlap).

**Partial Success**:
- >= 4/5 direction AND >= 2/5 CI overlap, OR
- >= 3/5 direction AND >= 3/5 CI overlap.

**Failure**: < 3/5 direction agreement, OR both P1 and P2 below partial threshold.

**Evaluation set**: 5 confirmatory topics (Topics 2–6). humic_acid_yield is excluded from success criteria.

**Reporting commitment**: Results will be reported regardless of direction. All 5 confirmatory topics will be run and reported. humic_acid_yield results will be reported as an appendix for transparency. Failure results are as scientifically valuable as success results.

---

## Pilot vs Confirmatory Distinction

*(Added 2026-03-26 to clarify evaluation scope)*

### humic_acid_yield is a pipeline validation test, not a preregistered topic

humic_acid_yield is run as a full-pipeline test BEFORE the preregistered confirmatory evaluation begins. Its purpose is to surface pipeline bugs, calibrate adjudication prompts, and validate that Stages 3–9 work end-to-end on a clean new topic. It is NOT part of the preregistered evaluation set.

**humic_acid_yield results will NOT be included in P1/P2 success criteria.** The 5 confirmatory topics (Topics 2–6) are the evaluation set.

### Execution order

The pipeline will be run in this sequence:

**Phase A — V1 LLM Re-adjudication** (before any new topic is run): Claude Code reads all 6 existing V1 extracted-row datasets and applies LLM semantic adjudication to learn from the V1 failures before any new extraction begins.

**Phase B — humic_acid_yield pipeline test**: Full Stages 3–9 run on humic_acid_yield after Phase A lessons are incorporated. Results used to fix bugs only. Proceed only after Phase A is complete.

**Phase C — Preregistered confirmatory topics**: amf_inoculation_yield, biochar_tropical_yield, elevated_co2_face_yield, legume_rotation (V2 rerun), cover_crop_corn_yield. Run only after Phase B validates the pipeline. Results count toward P1/P2.

### Why this order

Running confirmatory topics before learning from V1 data would risk repeating known failure modes. The Phase A re-adjudication extracts all available lessons from already-extracted data (no new API costs, no new papers) before the first dollar is spent on confirmatory extraction.

---

## Section 6 — LLM Adjudication Protocol

### 6.1 Stage A: Deterministic Pre-Check (No LLM)

1. Both means present and numeric -> pass
2. Both means positive (lnRR requires positive ratio) -> pass or flag
3. Missing or non-numeric mean -> exclude without LLM call

### 6.2 Stage B: LLM Semantic Adjudication

Each Stage-A-passed row is sent to Claude with a compact topic brief (from config: PICO, TC confusion warnings, outcome definitions) plus the row as structured JSON.

The LLM evaluates 5 criteria:
1. Does the treatment match the configured intervention?
2. Does the control match the configured comparator?
3. Does the outcome match the configured primary outcome?
4. Does the row match the benchmark estimand?
5. Is there evidence treatment and control were swapped?

Decision policy (frozen):
- keep if intervention, comparator, and outcome all match >= partial
- exclude if any of intervention, comparator, outcome = no
- flag if estimand = partial or row semantics ambiguous
- swap_treatment_control if T/C clearly reversed relative to config
- Default: exclude ambiguous rows (false exclusion preferred over false inclusion)

### 6.3 Where LLM Replaces Keywords

LLM replaces keyword-based filtering for: intervention isolation, outcome disambiguation, estimand verification, comparator identity, contextual plausibility. Keywords retained only for Stage A structural checks.

---

## Section 7 — Deviation Policy

### 7.1 Allowable Without Deviation Log

- LLM model version update (Claude Sonnet 4 -> newer equivalent) without prompt change
- Download retry logic changes (implementation detail)
- Additional exploratory analyses clearly labeled as post-hoc

**Model plan**: Initial Phase C runs use claude-sonnet-4-20250514. If P1 or P2 outcomes are borderline (within 1 topic of threshold), a confirmatory re-run using the most capable available Anthropic model (Opus 4.6) will be conducted and both runs reported. This is not considered a deviation.

### 7.2 Requires Deviation Log Entry (in codex/STATUS_LOG.md)

Any change to: stage architecture, adjudication decision policy or output schema, primary success criteria, topic set, benchmark assignments, variance conversion hierarchy, synthesis method.

Log format: date, stage(s) affected, description of change, reason, expected impact on preregistered analyses.

### 7.3 Non-Reporting Commitment

Results will be reported regardless of direction. All 5 confirmatory topics will be run and reported. humic_acid_yield pilot results will be reported as a separate appendix for transparency.

---

## Section 8 — V1 Development Context

### 8.1 V1 Results (Keyword Pipeline, 2026-03-25)

V1 was development work run iteratively. Topics were not preregistered. Results below are for reference only and do not contribute to V2 confirmatory evaluation.

**Note on legume_rotation prior results**: The raw V1 extracted data for legume_rotation was produced on 2026-03-25, one day before this preregistration was written. A preliminary synthesis using keyword adjudication (+15.7%) was also visible at preregistration time. The Phase C confirmatory run for legume_rotation will use a fresh Stage 4 re-extraction under the V2 auditable extraction protocol (extract_stage4_universal.py), ensuring the confirmatory result is independent of any prior run. The 2026-03-25 result is treated as a V1 development data point only.

| Topic | V1 Pooled Effect | Benchmark | Direction | Abs Gap |
|-------|-----------------|-----------|-----------|---------|
| organic_yield_gap | -4.9% | -19.2% | CORRECT | 14.3pp |
| notill_tillage | +1.2% | -5.7% | WRONG | 6.9pp |
| mycorrhiza_yield | +29.3% | +23.0% | CORRECT | 6.3pp |
| legume_rotation | +17.7% | +20.0% | CORRECT | 2.3pp |
| biochar_crop_yield | +6.7% | +16.0% | CORRECT | 9.3pp |
| intercropping_yield | -3.1% | +22.0% | WRONG | 25.1pp |

V1 direction agreement: 4/6 topics (67%). V2 primary target: >= 4/5 confirmatory topics (see Section 5).

### 8.2 V1 Lessons Motivating V2 Design

1. Semantic over-inclusion (intercropping, notill): keyword filters accepted rows measuring wrong outcomes
2. Estimand mismatch: keywords could not detect per-plant vs. per-area, sequential vs. simultaneous
3. No T/C swap detection: occasional treatment/control reversals inflated/deflated estimates
4. Composition mismatch: pipeline sample differed from benchmark corpus
5. No duplicate control: some papers contributed overlapping rows

V2 addresses all five via LLM adjudication, effector normalization, and deterministic QC.

---

## Section 9 — Timeline

| Milestone | Date |
|-----------|------|
| V2 architecture drafted | 2026-03-25 |
| Topic scoring and selection | 2026-03-25 |
| Topic configs and benchmark specs written | 2026-03-25 |
| Dress rehearsal (humic_acid_yield search + screen) | 2026-03-26 |
| **Preregistration frozen** | **2026-03-26** |
| **Phase 1 — V1 LLM re-adjudication (all 6 existing topics)** | TBD |
| **Phase 2 — humic_acid_yield pipeline test (Stages 3-9)** | TBD (after Phase 1) |
| **Phase 3 — Preregistered confirmatory topics (5 topics)** | TBD (after Phase 2) |
| V2 results paper | TBD |

**Phase 1** covers LLM semantic adjudication of all 6 previously-extracted V1 topics (legume_rotation, mycorrhiza_yield, organic_yield_gap, notill_tillage, biochar_crop_yield, intercropping_yield). No new papers are read; no new extraction is run. Purpose: extract all lessons from existing data before touching any preregistered topic.

**Phase 2** is the humic_acid_yield pipeline validation test. Full Stages 3–9 run after Phase 1 lessons are incorporated. Results used to fix pipeline bugs. NOT part of the confirmatory evaluation.

**Phase 3** covers the 5 preregistered confirmatory topics (Topics 2–6). Run only after Phase 2 validates the end-to-end pipeline. Results count toward P1/P2 success criteria.

If any phase reveals a bug requiring architectural change, this will be logged as a deviation in codex/STATUS_LOG.md before proceeding.

---

## Appendix A — Benchmark Papers Summary

| Topic | Benchmark | Year | Journal | Effect | DOI |
|-------|-----------|------|---------|--------|-----|
| humic_acid_yield | Ma, Cheng & Zhang | 2024 | Agronomy (MDPI) | +12% | 10.3390/agronomy14122763 |
| amf_inoculation_yield | Wu et al. | 2022 | PeerJ | +23% | TBC |
| biochar_tropical_yield | Jeffery et al. | 2017 | ERL (IOP) | +25% tropics (95% CI: ~+15% to +35%, graphical read from Fig 1) | 10.1088/1748-9326/aa67bd |
| legume_rotation | Zhao et al. | 2022 | Nature Comm. | +20% | 10.1038/s41467-022-28412-9 |
| elevated_co2_face_yield | Long et al. (PRIMARY) | 2006 | Science | ~+8% FACE cereals; ~+13% FACE legumes; ~0% C4; range ~+5-13% (no formal CI; perspective article) | 10.1126/science.1114722 |
| elevated_co2_face_yield | Ainsworth & Long (SECONDARY) | 2021 | Global Change Biol. | ~+18% all C3 crops pooled; CI not obtainable (closed access) | 10.1111/gcb.15375 |
| cover_crop_corn_yield | Marcillo & Miguez | 2017 | JSWC | -1% to +3% | TBC |

TBC = To be confirmed from journal website.

---

## Appendix B — Topic Scoring Summary

| Topic | Total Score (max 40) | Notes |
|-------|----------------------|-------|
| legume_rotation | 37 | Highest scored; Nature Comms OA |
| elevated_co2_face_yield | 36 | Gold-standard FACE corpus |
| amf_inoculation_yield | 35 | PeerJ OA; rainfed field restriction |
| biochar_tropical_yield | 35 | ERL OA; tropical subgroup |
| humic_acid_yield | 34 | MDPI OA; pilot topic |
| cover_crop_corn_yield | 34 | Hard null-result benchmark |
