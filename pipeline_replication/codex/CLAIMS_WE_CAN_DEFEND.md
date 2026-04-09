# Claims We Can Defend

## Main Claim

A universal autonomous meta-analysis replication pipeline can proceed from topic configuration to pooled effect estimation without manual paper-by-paper curation, and LLM-based post-extraction adjudication is a necessary component of making the extracted evidence benchmark-comparable.

## Strong Claims

These are the claims most defensible from the current evidence.

### 1. End-to-end autonomous replication is feasible

The pipeline successfully performs:

- literature search
- screening
- OA download
- extraction
- post-extraction validation
- synthesis

across multiple agricultural topics.

### 2. LLM post-processing is necessary

The dominant failure mode is often not wrong numbers, but semantically wrong rows:

- wrong outcome
- wrong intervention
- wrong comparator
- wrong study setting
- wrong estimand

This justifies a universal LLM-based adjudication layer after extraction.

### 3. The pipeline can genuinely replicate some topics

At least some preregistered topics are genuine successes or qualified successes:

- `legume_rotation`
- `mycorrhiza_yield`

### 4. Persistent failures remain even after semantic cleanup

This is important because it shows the remaining gaps are not just extraction noise.

### 5. Replication failures can be diagnostically informative

When the pipeline still misses the benchmark after cleanup, the miss often reveals:

- sample-composition mismatch
- benchmark-alignment mismatch
- OA-accessible corpus bias
- estimand mismatch

## Claims To Avoid

### 1. Avoid claiming that post-processing alone solves replication

Current tests do not support that.

### 2. Avoid claiming that all failures are extraction failures

The data do not support that interpretation.

### 3. Avoid replacing preregistered primary analyses with benchmark-aligned slices

Those are useful diagnostics or sensitivity analyses, not replacements.

### 4. Avoid claiming universal reliable replication across all topics

The current evidence supports conditional success, not blanket reliability.

## Recommended Framing

The best framing is:

1. universal autonomous replication is possible
2. LLM adjudication is essential
3. some topics replicate well
4. some topics fail for reasons that remain after semantic cleanup
5. those remaining failures identify the real limits of OA-only autonomous synthesis
