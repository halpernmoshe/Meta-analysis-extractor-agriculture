# Claude Handoff

## Purpose Of This Note

This is a handoff for continued work on the meta-analysis replication project.
It summarizes:

- what the current project is
- what was learned from the existing replication runs
- what Codex tested under `codex/`
- what conclusions were reached
- what Claude should do next

All new work in this phase was kept under the `codex` folder.

## Project State

The submitted manuscript in `SUBMISSION_v23/SUBMISSION_CLEAN` is the extraction-validation paper.

The `pipeline_replication` folder is the end-to-end autonomous replication project:

- search
- screening
- OA download
- extraction
- post-extraction validation
- synthesis

The original goal was to replicate published benchmark meta-analyses from seed DOIs and OA-accessible literature.

Important reminder from the user:

- Claude's universal downloader worked much better than all previous downloaders.
- Treat the universal downloader as a retained strength when designing Pipeline v2.
- Do not regress to the older downloader logic unless there is a specific reason to test it.

## Key Conceptual Realization

The major issue is usually **not** raw numeric hallucination.

The dominant failure modes are:

- wrong outcome included
- wrong intervention/comparator interpretation
- wrong study setting
- wrong estimand
- benchmark mismatch in construct definition
- corpus composition differences

This means the pipeline needs strong **LLM-based post-extraction adjudication**.

The user also clarified that LLM post-processing was the breakthrough that made the submitted extraction paper work. That strongly supports making LLM post-processing central in Pipeline v2.

## What Was Found In The Current Topic Set

### High-Level Topic Status

- `legume_rotation`: strongest success
- `mycorrhiza_yield`: qualified success
- `organic_yield_gap`: primary miss, but strong crop-composition / construct-alignment issues
- `notill_tillage`: primary miss, but strong management / context mismatch issues
- `biochar_crop_yield`: primary miss, but field vs pot / study-design mismatch issues
- `intercropping_yield`: estimand mismatch (`LER` / system productivity vs component crop yield)

### Important Correction

The benchmark papers should **not** automatically be treated as more representative than the pipeline-derived sample.

The benchmark is a published external reference synthesis, not metaphysical truth.

Correct framing:

- agreement with benchmark = successful reproduction of published reference synthesis
- disagreement = divergence from published reference synthesis
- disagreement does **not** prove the benchmark is more representative

Possible interpretations of disagreement:

1. pipeline extraction/filtering error
2. accessible corpus differs from benchmark corpus
3. benchmark study mix differs from pipeline study mix
4. pipeline and benchmark estimate different constructs

### Estimand Clarification

An **estimand** is the exact quantity the synthesis is trying to estimate.

This is more precise than the topic label.

Examples:

- Topic: `intercropping yield`
  - possible estimands:
    - component maize yield
    - component soybean yield
    - total system productivity
    - `LER` / land-equivalent ratio

- Topic: `no-till`
  - possible estimands:
    - grain yield
    - straw yield
    - total biomass
    - effect of strict no-till
    - effect of broader conservation agriculture

Two analyses can address the same topic but still disagree because they estimate different estimands.

This matters especially for:

- `intercropping`: `LER/system productivity` vs component crop yield
- `organic`: harvested yield vs quality or concentration traits leaking into the dataset
- `no-till`: grain yield vs broader productivity outcomes, and strict no-till vs broader tillage categories

## What We Learned About Validation Logic

The paper should not claim:

- “we proved the pipeline is true because it matched the benchmark”

Instead, the valid claims are:

- the pipeline can generate coherent syntheses from primary literature
- the pipeline sometimes reproduces published reference syntheses
- LLM post-processing is necessary to make extraction outputs benchmark-comparable
- when the pipeline disagrees with the benchmark, the disagreement is often structured and interpretable

So the benchmark is useful for:

- external comparison
- convergent validity
- reproducibility testing

not for claiming ultimate truth.

## What Was Tested Under `codex`

### 1. Project summary and planning notes

Created:

- `DISCUSSION_SUMMARY.md`
- `PIPELINE_V2_STRATEGY.md`
- `CLAIMS_WE_CAN_DEFEND.md`
- `FIXABLE_VS_STRUCTURAL_LIMITS.md`
- `WAYS_TO_NARROW_THE_GAP.md`
- `BENCHMARK_REPRESENTATIVENESS_NOTE.md`

### 2. Universal LLM-first post-processing design

Created:

- `UNIVERSAL_POSTPROCESS_DESIGN.md`
- `CLAUDE_UNIVERSAL_POSTPROCESS_PROMPT.md`
- `build_universal_llm_postprocess_inputs.py`

This packages extracted rows for universal semantic adjudication using only:

- topic config
- extracted row fields

### 3. Topic-specific strict prototype

Created:

- `prototype_strict_postprocess.py`
- `TIGHTER_POST_EXTRACTION_RULES.md`

Important:

This topic-specific strict prototype is exploratory only.
It is **not** the intended final universal solution.

### 4. Codex adjudication test on two difficult topics

Tested:

- `organic_yield_gap`
- `notill_tillage`

Using:

- validated rows
- universal config-driven adjudication logic
- Codex workers writing `keep/exclude/flag/swap` decisions under `codex/outputs/codex_decisions`

Combined with:

- `apply_codex_decisions_and_resynthesize.py`

Result:

- cleaning helped remove semantic leakage
- but did **not** solve the benchmark gaps by itself

Organic:

- validated rows: 378
- kept after Codex pass: 270
- pooled effect still far from benchmark overall

No-till:

- validated rows: 605
- kept after Codex pass: 299
- pooled effect moved closer to zero, but still did not match benchmark

Conclusion:

Universal LLM post-processing is necessary, but not sufficient for the hardest misses.

### 5. Effector normalization

Created:

- `CLAUDE_EFFECTOR_NORMALIZATION_PROMPT.md`
- `build_effector_review_batches.py`
- `combine_llm_postprocess_and_effectors.py`

Codex workers labeled benchmark-relevant effectors on already-kept rows for:

- `organic_yield_gap`
- `notill_tillage`

Result:

- the combined post-processing + effector route helped interpret the misses
- some narrow subsets got closer
- but the difficult topics were still not robustly “fixed”

Organic:

- one narrow slice got close to benchmark
- the broad benchmark-aligned slice still did not fully solve the topic

No-till:

- benchmark-aligned subset after both layers still missed benchmark materially

Conclusion:

Combined LLM post-processing + effector normalization is useful and justified, but still not enough to guarantee success on hard topics.

### 6. Universal diagnostics

Created:

- `universal_postprocess_diagnostics.py`

Purpose:

- annotate validated rows with universal classes:
  - outcome class
  - study setting
  - estimand class
  - intervention class
  - benchmark alignment class
  - variance presence
  - duplicate key

This is for diagnosis, not silent data rewriting.

### 7. Benchmark spec work

Created:

- `BENCHMARK_SPEC_TEMPLATE.md`
- `WHY_BENCHMARK_SPEC_MATTERS.md`

Main idea:

The benchmark paper should be converted prospectively into an explicit structured spec capturing:

- intervention definition
- comparator definition
- outcome hierarchy
- study-setting restrictions
- moderator structure
- subgroup logic
- known estimand traps

This is to improve construct alignment prospectively, not to chase benchmark numbers post hoc.

## What We Concluded Strategically

### Most Important Conclusion

The current topic set was useful for learning, but it is not the right place to keep trying to extract a strong confirmatory claim.

Instead:

1. use current topics as development/stress-test cases
2. optimize the pipeline into a real Pipeline v2
3. select a better topic set prospectively
4. preregister again
5. run Pipeline v2 prospectively

### Why

Because continuing to optimize inside the old preregistration would blur confirmatory and exploratory work.

### Important Related Point

The user is **not attached to the current topics**.

That is a major advantage.

It means:

- current topics can remain internal learning cases
- the next paper can be built around a new topic set chosen prospectively to give Pipeline v2 a fair evaluation

## What Pipeline V2 Should Include Before Re-Preregistration

Pipeline v2 should include, at minimum:

- benchmark spec ingestion
- universal LLM post-extraction adjudication
- universal effector normalization
- outcome canonicalization
- study-setting normalization
- intervention/comparator normalization
- estimand classification
- variance diagnostics
- duplicate control
- benchmark-aligned secondary analyses defined in advance

## What Should Be Done Programmatically Vs By LLM

### Programmatic

Use deterministic code for:

- require means / positivity checks
- unit parsing where explicit
- SE/SD/LSD/CI conversion
- effect-size and variance calculation
- duplicate detection
- provenance tracking
- exact pairing when keys match
- audit logs

### LLM

Use LLMs for semantic judgment:

- does the row really match intervention?
- comparator?
- primary outcome?
- benchmark estimand?
- canonical outcome class
- canonical study setting
- canonical intervention class
- canonical effector labels
- benchmark-alignment tagging

Rule of thumb:

- if there is a formula, do it programmatically
- if the question is “what does this row mean?”, use an LLM

## What Should Go In The Future Paper

If the goal is that **v2** becomes the preregistered working system, then the final paper should center on v2, not v1.

v1 should be treated as:

- internal development work
- failure analysis
- architecture discovery

The final paper should instead say:

- earlier development work revealed recurring failure modes
- these motivated Pipeline v2
- Pipeline v2 was then preregistered and evaluated prospectively

## What Claude Should Do Next

When available again, Claude should proceed in this order.

### 1. Treat current topics as development cases only

Do not try to turn the current topic set into the main confirmatory paper result.

Use the already-downloaded papers from the current topics as the primary retrospective development bench for Pipeline v2.

That means:

- run v2 on the already-downloaded corpora for the existing topics
- use those reruns to test whether v2 actually improves semantic targeting, estimand handling, and post-processing
- treat those runs as internal development / stress testing / retrospective validation
- do **not** treat those reruns as the new confirmatory evidence

### 2. Freeze Pipeline v2 architecture

Produce a single explicit v2 architecture spec including:

- benchmark spec
- universal LLM adjudication
- universal effector normalization
- canonical outcome/intervention/setting/estimand classes
- diagnostics

### 3. Build a broad candidate topic list

Do this before choosing “good” topics.

### 4. Score topic candidates prospectively

Use:

- `TOPIC_SELECTION_CRITERIA.md`
- `topic_scoring_template.csv`
- `score_topic_candidates.py`

Good candidate features:

- clear estimand
- clear intervention/comparator
- limited study-setting heterogeneity
- good OA feasibility
- benchmark paper with explicit methods
- low estimand-trap risk

### 5. Pick one small pilot topic

Use it to test the full v2 workflow end to end:

- benchmark spec creation
- extraction prompt
- universal LLM adjudication
- effector normalization
- synthesis
- reporting

### 6. Only after that, select the full v2 topic set

Select prospectively and preregister.

### 7. Then run the actual v2 evaluation

That should be the basis of the next paper.

In other words, the recommended sequence is:

1. build Pipeline v2
2. test v2 on the already-downloaded old-topic corpora
3. learn what v2 fixes and what still fails
4. freeze the v2 architecture
5. choose a new topic set prospectively
6. preregister
7. run the real v2 evaluation on the new preregistered topics

## Operational Request

Because prior work was interrupted by token limits and restart events, Claude should keep a short rolling status log under `codex` while working.

That log should record:

- what was completed
- what was partially completed
- what should be resumed next
- any blockers or failures encountered

The goal is to make recovery from another interruption fast and reliable.

## Immediate Practical Files Claude Should Read First

Under `codex`, Claude should start with:

- `PIPELINE_V2_STRATEGY.md`
- `TOPIC_SELECTION_CRITERIA.md`
- `BENCHMARK_SPEC_TEMPLATE.md`
- `UNIVERSAL_POSTPROCESS_DESIGN.md`
- `WHY_BENCHMARK_SPEC_MATTERS.md`
- `CLAIMS_WE_CAN_DEFEND.md`
- `FIXABLE_VS_STRUCTURAL_LIMITS.md`
- `IMMEDIATE_NEXT_ACTIONS.md`

## Final Bottom Line

The current work was useful.

It established that:

- the pipeline idea is real
- LLM post-processing is essential
- some topics work
- the hardest misses persist even after cleanup
- those misses are often structural or construct-level, not just row-level mistakes

That is enough to justify moving to a true Pipeline v2 development phase followed by a new preregistration.
