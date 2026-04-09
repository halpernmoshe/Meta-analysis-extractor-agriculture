# Why A Benchmark Spec Matters

## Short Version

A topic label is not enough.

Labels like:

- `no-till yield`
- `intercropping yield`
- `organic yield gap`

hide crucial operational choices that determine what a synthesis is actually estimating.

If those choices are explicit in the benchmark paper, they should be captured prospectively in a structured benchmark spec and fed into both extraction and post-processing.

## Why This Is Important

Two syntheses can address the “same topic” but still estimate different things because they differ in:

- intervention definition
- comparator definition
- outcome hierarchy
- study-setting restrictions
- moderator coding
- subgroup logic
- estimand definition

If the pipeline does not know those choices explicitly, it can produce a coherent synthesis that is still not benchmark-comparable.

## What A Benchmark Spec Fixes

### 1. Better extraction prompts

The extractor can be told up front:

- what definitely counts
- what definitely does not count
- which moderators must be captured
- which ambiguities are dangerous

### 2. Better post-processing

The validator can judge rows against explicit definitions rather than vague topic names.

### 3. Better benchmark-aligned secondary analyses

If the benchmark paper reports important subgroup logic, that can be translated into a prospective secondary analysis plan instead of post hoc fishing.

## Why This Is Not The Same As Cheating

This is only defensible if used to capture **explicit methodological definitions**, not to chase the benchmark number.

Good use:

- copying the benchmark's explicit inclusion logic
- copying the benchmark's explicit estimand definition
- copying the benchmark's explicit moderator structure

Bad use:

- changing filters after seeing which settings move your pooled estimate toward the benchmark

So the benchmark spec is a way to improve construct alignment, not a way to force agreement.

## Why It Matters For This Project

Several current misses likely came from hidden operational mismatches:

- `intercropping`: system productivity vs component crop yield
- `no-till`: strict no-till vs reduced/conservation tillage
- `biochar`: field-focused benchmark vs mixed field/pot extraction
- `organic`: crop-composition and diversification structure

These are exactly the kinds of mismatches a benchmark spec can reduce in future runs.
