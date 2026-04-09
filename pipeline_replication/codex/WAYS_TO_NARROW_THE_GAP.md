# Ways To Narrow The Gap

This note focuses on practical strategies that could increase the number of successful topic replications.

## Highest-Leverage Near-Term Moves

### 1. Use `Claude Opus 4.6` for universal post-extraction adjudication

This is the clearest next upgrade because semantic adjudication was already the breakthrough in the submitted extraction paper.

Expected gain:

- fewer off-target rows
- cleaner benchmark alignment
- better intervention/comparator discrimination

### 2. Add universal effector normalization with the same model

Use the LLM to normalize:

- crop class
- study setting
- climate context
- management context
- estimand context

Expected gain:

- cleaner benchmark-aligned secondary analyses
- better explanation of misses
- better ability to pre-specify benchmark-comparable subsets

### 3. Improve search and download coverage

This may produce larger gains than better filtering for some topics.

Expected gain:

- reduce OA composition bias
- recover missing benchmark-like studies
- increase paper diversity within topics

Priority ideas:

- stronger citation chasing
- benchmark-seed recovery audit
- more OA source coverage
- better download retry logic

### 4. Extract missing benchmark-critical moderators explicitly

Many gaps persist because the current rows do not always contain:

- field vs pot
- crop class
- rotation / residue context
- study pairing keys
- system-vs-component estimand markers

Expected gain:

- cleaner benchmark alignment
- fewer fragile heuristics later

## Topic-Specific Gap-Narrowing Ideas That Still Preserve Universality

These are not hand-tuned rules. They are generic pipeline improvements that happen to matter for current topics.

### 1. Canonical outcome ontology

A universal ontology would stop many off-target rows before synthesis:

- harvest yield
- biomass
- quality trait
- nutrient concentration
- system productivity
- component crop yield

### 2. Canonical intervention ontology

Examples:

- strict no-till
- reduced tillage
- organic system
- low-input but non-organic
- biochar amendment
- AMF inoculation

This would reduce semantic drift without writing topic-specific code.

### 3. Canonical study-setting ontology

Examples:

- field
- greenhouse
- pot
- chamber
- mixed

This would especially help topics whose benchmark corpus is mostly field-based.

## Harder But Potentially Important

### 1. Better variance recovery

If weighting is wrong, pooled effects can drift even when rows are otherwise correct.

### 2. Robust handling of non-independence

Many papers contribute multiple related rows. Better clustering or multi-level handling could improve pooled estimates.

### 3. Derived metric reconstruction where possible

This matters most for intercropping and other system-level metrics.

### 4. Benchmark-aware secondary synthesis plans

Keep preregistered primary analyses intact, but predefine a universal set of secondary benchmark-alignment summaries:

- field-only
- table-only
- high-confidence only
- direct-harvest outcomes only
- benchmark-aligned estimand only

This avoids ad hoc subgroup fishing after seeing results.

## Most Promising Practical Strategy

If the goal is to show more successful replications rather than only explain failures, the best sequence is:

1. stronger universal LLM adjudication
2. universal effector normalization
3. better download / search coverage
4. richer moderator extraction
5. benchmark-aligned secondary analyses defined in advance

## Important Caveat

Some remaining gaps may not be reducible enough to convert all failures into successes.

That is acceptable.

The stronger paper is the one that shows:

- where the pipeline succeeds
- where better post-processing helps
- where remaining failures persist despite better post-processing
- which of those failures are structural
