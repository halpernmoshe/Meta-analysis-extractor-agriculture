# Fixable Vs Structural Limits

## Fixable By Better Pipeline Design

These are realistic engineering targets.

### 1. Better universal LLM adjudication

- stronger model (`Claude Opus 4.6`)
- better prompt structure
- explicit keep / exclude / flag / swap schema
- benchmark-alignment tagging

### 2. Outcome canonicalization

- map rows into yield, biomass, quality, concentration, system productivity, component yield

### 3. Intervention/comparator normalization

- strict no-till vs reduced tillage
- organic vs other low-input systems
- biochar vs mixed amendment systems

### 4. Study-setting normalization

- field vs greenhouse vs pot

### 5. Estimand normalization

- system productivity vs component crop yield
- direct harvest yield vs biomass proxy

### 6. Variance diagnostics

- missing variance classification
- SE/SD ambiguity checks
- suspicious weighting flags

### 7. Duplicate / near-duplicate control

- repeated rows
- text/table duplicates
- pooled and raw rows both entering synthesis

### 8. Better moderator extraction

- crop class
- study setting
- climate class
- management context
- pairing variables for derived metrics

### 9. Confidence-aware sensitivity analyses

- exclude low-confidence rows
- table-only sensitivity
- figure-only diagnostics

## Structural Limits

These are unlikely to disappear with post-processing alone.

### 1. OA download bottleneck

If the paper is not accessible, no downstream refinement can recover it.

### 2. Nonrepresentative accessible corpus

The accessible literature may systematically differ from the benchmark corpus.

### 3. Benchmark not commensurate with current accessible sample

Even a clean extracted sample can differ in:

- crop mix
- climate mix
- management context
- study setting
- publication era

### 4. Missing papers not found or not downloaded

This is a search/download coverage problem, not a post-processing problem.

### 5. Missing key information never extracted

If the current extraction outputs do not contain the fields needed for pairing or derived metrics, post-processing cannot fully reconstruct them.

### 6. Estimand mismatch in the literature itself

Intercropping is the clearest example:

- benchmark estimand: system productivity (`LER`)
- extracted literature: mostly component crop yields

That is not just a processing problem.

## Practical Meaning

The right paper message is not “everything can be fixed with better prompts.”

The right message is:

- some problems are fixable with better pipeline design
- some are inherent to autonomous OA-only replication
- the important result is distinguishing those two classes clearly
