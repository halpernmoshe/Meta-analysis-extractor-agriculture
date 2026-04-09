# Pipeline V2: What We Learned And How To Choose Better Agricultural Topics

## What V2 Needs To Do To Really Work

The main lesson from the current project is that Pipeline v2 must be built around **semantic discipline**, not just better numeric extraction.

The downloader appears to be a retained strength.
The raw extraction is often numerically usable.
The bigger failures are usually caused by:

- wrong estimand
- wrong outcome class
- wrong comparator
- wrong study setting
- non-independent rows
- mismatch between topic label and benchmark operational definition

So v2 should be designed as:

1. broad extraction
2. deterministic structural QC
3. strong LLM semantic adjudication
4. canonical row labeling
5. strict synthesis using only rows that match the target estimand

## What Must Be First-Class In V2

### 1. Canonical row labels

Every extracted row should get structured labels for:

- outcome class
- estimand class
- intervention class
- comparator class
- study setting
- independence status
- benchmark-alignment status

Without this, synthesis remains too permissive.

### 2. Non-independence handling

V2 should explicitly identify and manage:

- pooled averages
- repeated contrasts against the same control
- multiple years from the same experiment
- component rows plus system-level rows from the same study
- repeated subgroup summaries that are not independent observations

### 3. Strong semantic adjudication

Programmatic code should do:

- numeric sanity checks
- effect-size calculations
- variance calculations and conversions
- duplicate detection
- provenance tracking

LLMs should do:

- does this row match the intended intervention?
- does this control match the intended comparator?
- is this the primary outcome or only a related trait?
- what estimand does this row actually measure?
- is this row benchmark-comparable?

### 4. Built-in diagnostics

Each topic run should automatically produce:

- paper influence diagnostics
- likely-off-target burden
- non-independence burden
- table-only sensitivity
- high-confidence-only sensitivity
- field-only sensitivity where relevant
- benchmark-aligned secondary summaries

## Why The Current Data Still Helped

The current topics were useful because they exposed the failure modes clearly.

What they showed:

- `intercropping_yield` failed mainly because of estimand mismatch
- `organic_yield_gap` still contains outcome leakage and system-productivity contamination
- `notill_tillage` is less about obvious bad rows and more about intervention/context drift
- `biochar_crop_yield` is strongly affected by study setting and outcome class
- `mycorrhiza_yield` is heterogeneous and somewhat paper-sensitive
- `legume_rotation` worked relatively well but still shows contamination from neighboring constructs

This means v2 should be tested first on the already-downloaded old-topic corpora as a retrospective development bench.
That is the right place to find out whether the new architecture actually improves the hard parts.

## Which Agricultural Topics Are Better For V2

The user wants the next preregistered topic set to remain agricultural.
That is sensible.

But not all agricultural topics are equally good validation topics for a universal pipeline.

### Better agricultural topics

The best agricultural topics for v2 are topics with:

- one clear intervention
- one clear comparator
- one clear primary outcome
- little estimand ambiguity
- limited setting heterogeneity
- benchmark papers with explicit definitions
- high OA feasibility
- relatively standardized reporting in primary papers

Good topic pattern:

- one amendment or inoculant vs no amendment
- one cropping practice vs a clearly defined baseline
- direct harvested yield as the primary outcome
- mostly field trials or a clearly predeclared setting class

### Worse agricultural topics

The worst topics for v2 validation are topics with:

- system-level estimands hidden under simple topic labels
- many related but non-equivalent outcomes
- intervention definitions that sprawl across multiple practices
- extreme dependence on study setting
- benchmark papers whose inclusion logic is complex but not obvious from the title

Bad topic pattern:

- topic label sounds simple but hides several estimands
- component yield and system productivity get mixed
- intervention blends several management practices
- benchmark depends heavily on moderators that are rarely reported cleanly

## How To Think About The Current Agricultural Topics

### Stronger current topics

- `legume_rotation`
  - comparatively clear topic
  - already close to working
  - still needs contamination control

- `biochar_crop_yield`
  - agriculturally important
  - intervention is understandable
  - main issue is setting and outcome-class discipline
  - likely salvageable with stronger row ontology

### Moderate current topics

- `organic_yield_gap`
  - important and publishable
  - but the term hides many possible comparisons and outcome classes
  - still usable, but not a simple clean validation topic

- `mycorrhiza_yield`
  - intervention concept is clear
  - but the literature is heterogeneous and sometimes co-intervention heavy
  - workable if v2 becomes strict about setting and outcome class

### Weak current topics for validation

- `notill_tillage`
  - highly confounded with residue, rotation, conservation agriculture, moisture context, and crop system
  - still interesting scientifically
  - not an ideal clean validation topic

- `intercropping_yield`
  - the clearest estimand trap
  - should not be used as a primary clean validation topic unless the target is explicitly restricted to one system-level estimand such as `LER`

## What A Better Agricultural Validation Topic Looks Like

A stronger v2 validation topic would be something like:

- one crop or narrow crop group
- one treatment type
- one direct yield endpoint
- mostly one setting class
- benchmark paper with explicit inclusion and moderator rules

The goal is not to choose topics that are easy in a trivial sense.
The goal is to choose topics that are fair tests of whether the universal architecture can correctly identify and synthesize the intended estimand.

## Recommended Next Move

1. Use the existing downloaded agricultural corpora as a retrospective v2 test bench.
2. See which current topics are improved most by:
   - semantic adjudication
   - non-independence control
   - canonical row labels
3. After that, choose a new agricultural topic set that avoids the worst estimand traps.
4. Then preregister the real v2 evaluation.

## Bottom Line

If v2 is going to really work, it should be strongest on agricultural topics that are:

- semantically narrow
- outcome-clean
- comparator-clean
- setting-coherent
- benchmark-explicit

That is the most realistic path to a preregistered v2 paper that actually succeeds.
