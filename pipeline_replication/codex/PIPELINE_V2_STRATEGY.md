# Pipeline V2 Strategy

## Why Move To Pipeline V2

The current preregistered topics were useful stress tests, but they also exposed several failure modes that are now understood more clearly:

- semantic over-inclusion after extraction
- intervention-definition drift
- outcome/estimand mismatch
- study-setting mismatch
- benchmark-alignment problems
- OA-accessible sample composition differences

At this point, the highest-value move is not to keep pushing the current preregistered topic set toward a confirmatory claim. The highest-value move is to optimize the pipeline, freeze a revised universal architecture, and then preregister a fresh evaluation.

## Why A New Topic Set Is Better

If the next study is meant to test `Pipeline v2`, it should not inherit avoidable topic-level traps from the older exploratory/stress-test set.

The current topics remain useful as:

- internal development cases
- diagnostic failure examples
- robustness tests

But they are not necessarily the best confirmatory evaluation set for the next preregistration.

## Better Evaluation Principle

The next preregistered study should test whether the improved universal pipeline works on topics that are:

- scientifically meaningful
- benchmarked in published meta-analyses
- feasible for OA-only autonomous replication
- not dominated by avoidable estimand ambiguity

This is not cherry-picking if the topic set is chosen **before** the next run using explicit criteria and then preregistered.

## What Pipeline V2 Should Include Before Re-Preregistration

### Core Architecture

- universal extraction
- universal LLM post-extraction adjudication
- universal effector normalization
- benchmark spec ingestion
- outcome canonicalization
- study-setting normalization
- intervention/comparator normalization
- estimand classification

### Diagnostics

- duplicate detection
- variance diagnostics
- confidence-aware sensitivity analysis
- benchmark-aligned secondary summaries

## What The Next Validation Should Try To Show

Not that the pipeline discovers “truth,” but that it:

- produces coherent syntheses from primary literature
- aligns with published reference syntheses when construct alignment is good
- fails in interpretable ways when alignment is poor
- performs better than Pipeline v1 under a cleaner, prospectively selected topic set

## Practical Recommendation

Use the current topics to design Pipeline v2.
Do not rely on them as the main confirmatory topic set for the next preregistration.

Instead:

1. freeze the new architecture
2. select a new topic set prospectively using explicit scoring criteria
3. preregister the evaluation
4. run Pipeline v2 on those topics
