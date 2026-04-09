# Universal Post-Extraction Design

## Goal

Create a universal post-extraction layer that can run on any topic using only:

- the original topic config / prompt content
- the extracted row fields
- optional per-paper metadata already present in extraction output

It should not rely on topic-specific custom code written after observing the results.

## Why This Is Needed

The main failure mode observed so far is not fake numbers. It is semantic over-inclusion:

- wrong outcome
- wrong intervention
- wrong comparator
- wrong estimand
- wrong study setting

Those failures can be handled by a universal post-processing stage that asks, row by row:

1. Does this row match the configured intervention?
2. Does this row match the configured comparator?
3. Does this row match the configured primary outcome?
4. Does this row match the benchmark estimand implied by the config?
5. Is the row usable for synthesis?

## Architecture

## Stage A: Hard Universal Checks

These are deterministic and topic-agnostic.

- require both means
- require positive means when using ratio-based effect sizes
- require numeric treatment/control means
- detect likely duplicates
- keep original confidence and source-type fields
- flag missing variance rather than immediately dropping

This stage is intentionally lightweight.
It should not try to solve semantic eligibility by regex alone.

## Stage B: Config-Driven Semantic Screening

Use only fields already present in the topic config:

- `pico.population.description`
- `pico.intervention.description`
- `pico.intervention.search_terms`
- `pico.comparator.description`
- `pico.comparator.search_terms`
- `pico.outcome.primary.description`
- `pico.outcome.primary.search_terms`
- `tc_confusion_warnings`
- `extraction_priorities`
- `benchmark.source` and `benchmark.published_pooled_effect.notes` if present

The semantic post-processor should decide whether each row is:

- `keep`
- `exclude`
- `flag_for_review`
- `swap_treatment_control`

## Stage C: LLM Adjudication

This is the main universal improvement and should be treated as the core of the system, not an optional add-on.

The LLM sees:

- a compact topic brief synthesized from config
- one extracted row
- title / notes / row descriptions

The LLM returns structured judgments:

- intervention match: `yes`, `partial`, `no`
- comparator match: `yes`, `partial`, `no`
- outcome match: `yes`, `partial`, `no`
- estimand match: `yes`, `partial`, `no`
- likely T/C swap: `yes` / `no`
- decision: `keep`, `exclude`, `flag`
- short rationale
- normalized labels for:
  - `outcome_class`
  - `study_setting`
  - `estimand_class`

## Why LLM Helps

A config-driven regex layer alone is too brittle. It fails on:

- synonyms
- compositional descriptions
- rows whose meaning is implicit in treatment labels
- outcome leakage like quality traits vs yield
- system productivity vs component yield

An LLM can make these semantic judgments while remaining universal if the prompt is fixed and the topic-specific inputs come only from config.

This matches the user’s experience from the submitted paper: LLM-based post-processing was the key breakthrough that turned extraction outputs into benchmark-comparable validated data.

## Production Model

Prototype work can be packaged under `codex`, but the intended production model is:

- `Claude Opus 4.6`

That means the design should avoid model-specific hacks and use a stable structured-output schema.

## Recommended Schema

```json
{
  "row_id": "topic::paper_id::row_index",
  "decision": "keep",
  "intervention_match": "yes",
  "comparator_match": "yes",
  "outcome_match": "yes",
  "estimand_match": "partial",
  "needs_tc_swap": false,
  "normalized_outcome_class": "grain_yield",
  "normalized_study_setting": "field",
  "normalized_estimand_class": "component_yield",
  "exclusion_reason": null,
  "rationale_short": "Outcome is grain yield and comparison matches config, but estimand is component yield rather than system productivity."
}
```

## Universal Decision Policy

Default policy:

- hard filters only remove rows that are structurally unusable
- the LLM is the authoritative semantic adjudicator
- `keep` only if intervention, comparator, and outcome all match at least `partial`, and none is `no`
- `exclude` if intervention or comparator is `no`
- `exclude` if outcome is `no`
- `flag` if estimand is `partial` or row meaning is ambiguous
- `swap_treatment_control` if the row clearly violates the configured T/C direction

## Advantages

- universal across topics
- auditable
- config-driven
- can be run after extraction with no prompt redesign for each topic
- lets the extractor stay broad while making synthesis input strict

## Current Codex Deliverable

This folder includes a prototype packager that:

- reads `config.json`
- reads extracted rows
- creates compact semantic review inputs for a future Claude Opus adjudication pass
- writes outputs under `codex/outputs`
