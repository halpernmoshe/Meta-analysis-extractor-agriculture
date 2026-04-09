# Statistical Diagnostics For Hallucination Risk

## Purpose

This memo proposes a way to detect likely extraction hallucinations or severe extraction errors **without requiring humans to reread every primary paper**.

The key idea is:

- publication bias can often be detected statistically without re-reading all studies
- likewise, some hallucinations or extraction failures may be detectable **indirectly** through statistical and structural inconsistency

This will not provide absolute proof that a row is true or false.

But it may provide a strong **risk-scoring framework** that identifies suspicious rows for:

- second-pass extraction
- second-model review
- targeted human audit

This memo is written with the existing project in mind, using data already available from:

- the submitted extraction-validation paper
- prior validation datasets
- the `pipeline_replication` topic runs
- repeated / partial extraction runs already on disk

## Core Claim

There may not be a single test for “hallucination” analogous to Egger’s test for publication bias.

However, a **battery of statistical and structural diagnostics** can make hallucinated or badly extracted rows much easier to detect.

This is best framed as:

- anomaly detection
- measurement-integrity assessment
- extraction-risk scoring

not as a direct proof-of-falsehood engine.

## Why This Is Plausible

A hallucinated extraction often has one or more of these properties:

- it is arithmetically inconsistent
- it does not fit the table/figure structure of the paper
- it has implausible units or outcome labels
- it behaves unlike other rows from the same paper
- it is unstable across reruns
- it is not recovered by an independent method
- it sits outside the empirical distribution of valid rows for that topic

Those are all things we may be able to measure.

## Proposed Diagnostic Families

### 1. Arithmetic Consistency Checks

These are the simplest and highest-value diagnostics.

Examples:

- Does `effect_pct` match `treatment_mean` and `control_mean`?
- Are means positive when lnRR is claimed?
- Are SE / SD / N combinations mathematically coherent?
- If `variance_type = SE`, does implied SD look plausible given `n`?
- If LSD or CI is reported, do derived uncertainty estimates behave sensibly?

Why useful:

- hallucinated rows often fail basic arithmetic coherence
- extraction mistakes also frequently show up here

These should remain deterministic and fully programmatic.

### 2. Table / Figure Structural Consistency

The extracted row should make structural sense relative to the source object.

Examples:

- Does the cited `table_or_figure` actually exist?
- Does the paper contain the named outcome and unit?
- Is the number of extracted rows plausible given the factorial design?
- Do repeated combinations imply impossible duplication?
- Does the extraction create rows that do not map onto visible treatment arms?

Why useful:

- hallucinations often invent rows or combinations that the paper structure does not support

This is partly deterministic and partly semantic.

### 3. Distributional Anomaly Detection

Valid scientific data have topic-specific empirical distributions.

Hallucinated or severely wrong rows may look statistically odd.

Potential signals:

- too many rounded values
- strange terminal-digit patterns
- repeated template-like means across different papers
- implausibly smooth ratios
- unusual variance-to-mean relationships
- effect sizes far outside the topic’s normal range

Possible tests:

- z-score / robust MAD outlier detection
- clustering / isolation forest on numeric features
- digit-distribution summaries
- frequency of duplicated numeric tuples

Why useful:

- not all extreme rows are false, but false rows are often statistically unusual

### 4. Paper-Level Coherence Checks

Rows from the same paper should usually tell a coherent story.

Suspicion signals:

- one row has a huge effect while all neighboring rows are flat
- significance labels contradict the extracted uncertainty
- the paper narrative says “no effect” but extracted values imply a very large precise effect
- one row has units or scales unlike all others in the same paper

Why useful:

- hallucinations and row misreads often break within-paper coherence before they break corpus-level patterns

This suggests a paper-level anomaly score, not just a row-level one.

### 5. Cross-Run Stability

If the same paper is extracted multiple times independently, stable rows are more trustworthy than unstable ones.

Signals:

- exact agreement across reruns
- near agreement after unit normalization
- high instability on one row while the rest of the paper is stable

Why useful:

- hallucinated or weakly grounded rows are often less stable than genuine readings

Existing project advantage:

- the current repo already contains multiple runs from both the current paper and earlier work
- that gives a natural substrate for estimating row-level stability

### 6. Cross-System Convergence

If two structurally different systems converge, that is strong evidence.

Examples:

- agent extractor vs consensus pipeline
- text extraction vs figure-derived extraction
- Claude-based run vs alternative-model run

Signals:

- agreement on means
- agreement on sign
- agreement on inclusion/exclusion
- disagreement concentrated in particular source types or outcomes

Why useful:

- this is one of the best non-human alternatives to direct rereading

### 7. Topic-Specific Prior Plausibility

Each topic has a prior landscape of plausible values.

Examples:

- no-till single-study effects above +200% are likely suspicious
- AMF effects in pot/drought studies may be much larger than field means
- biological yield, root biomass, and 1000-grain weight behave differently from grain yield

This suggests learning topic-specific plausibility ranges from:

- validated historical datasets
- prior benchmark datasets
- accepted rows from earlier runs

Why useful:

- not because prior ranges define truth
- but because they help identify rows that deserve heightened scrutiny

### 8. Corpus-Level Pattern Monitoring

If a run has too many suspicious rows, the problem may be systematic rather than local.

Examples:

- a sudden spike in very high effects
- abrupt change in source-type mix
- huge increase in figure-derived rows
- severe drop in variance coverage
- shift in extracted outcome-class composition

This suggests stop-run logic:

- if suspicious-row rate exceeds threshold, stop and audit before synthesis

## A Practical Proposal: Hallucination Risk Score

Build a per-row `hallucination_risk_score` from several components.

Candidate components:

1. arithmetic inconsistency
2. missing provenance / weak provenance
3. extreme effect size
4. uncommon unit/outcome combination
5. figure-only extraction
6. rerun instability
7. cross-system disagreement
8. paper-level incoherence
9. topic-prior outlier status

Then classify:

- `low risk`
- `medium risk`
- `high risk`

Recommended actions:

- low risk: pass automatically
- medium risk: second-pass extraction or second-model review
- high risk: targeted human audit

## Data Already Available To Build This

The project already has strong data resources for this work.

### From the extraction-validation paper

- matched observations against published reference datasets
- per-paper outputs
- repeated runs
- multiple datasets with different difficulty profiles

These can be used to estimate:

- which rows are stable
- which row types are most error-prone
- how error relates to source type, variance, and effect size

### From prior pipeline runs

- multiple extraction runs
- partial reruns
- topic-specific corpora
- adjudication outputs
- normalization outputs
- diagnostics reports

These can be used to estimate:

- topic-specific plausibility
- paper-level coherence
- corpus-level anomaly signatures

## Candidate First Implementation

The first implementation should be simple and empirical.

### Phase 1: Deterministic risk features

For every row, compute:

- `effect_pct_recomputed_matches`
- `variance_coherent`
- `source_type_is_figure`
- `missing_n`
- `missing_variance`
- `effect_pct_abs`
- `outcome_class`
- `unit_class`
- `paper_internal_outlier`

### Phase 2: Stability features

Where repeated runs exist, compute:

- `rerun_same_sign`
- `rerun_abs_delta_pct`
- `rerun_same_unit`
- `rerun_same_outcome_class`

### Phase 3: Cross-system features

Where an independent system exists, compute:

- `other_system_found_row`
- `other_system_same_sign`
- `other_system_abs_delta_pct`

### Phase 4: Simple classifier or scoring rule

Start with a transparent rule-based score before moving to ML.

Example:

- +3 if arithmetic inconsistency
- +2 if figure-derived
- +2 if extreme effect for topic
- +2 if rerun instability
- +2 if cross-system disagreement
- +1 if paper-level outlier
- +1 if missing variance

Then threshold for:

- auto-pass
- re-extract
- human review

## Why This Matters Philosophically

This approach directly supports the broader validation philosophy of the project.

The point is not to produce one magical truth test.

The point is to show that hallucinations, like publication bias, can often be inferred **indirectly** from patterns that are inconsistent with well-behaved scientific data.

That helps move the project from:

- “trust the extractor”

to:

- “trust the validation system that constrains the extractor”

This is a much stronger epistemic position.

## Immediate Next Steps

1. Inventory all available repeated-run data from the current paper and prior runs.
2. Define a first set of row-level deterministic risk features.
3. Build a merged dataset of:
   - accepted rows
   - known-bad rows
   - unstable rows
   - cross-system-disagreeing rows
4. Test whether these features actually predict:
   - extraction disagreement
   - benchmark divergence
   - adjudication exclusion
5. Produce a first `hallucination_risk_score` prototype.

## Final Bottom Line

Yes, statistical diagnostics for hallucination risk are plausible.

They will not prove truth directly.

But they may become a powerful part of a cumulative validation framework by:

- identifying suspicious rows
- prioritizing limited review effort
- reducing dependence on blanket human rereading
- strengthening the argument that the system is doing real science rather than merely generating plausible text
