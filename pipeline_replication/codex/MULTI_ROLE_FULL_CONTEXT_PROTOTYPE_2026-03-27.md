# Multi-Role Full-Context Prototype

Goal: set up a concrete prototype for the idea that multiple agents should read the full paper but with different roles, then compare outputs for convergence and contradiction.

## Proposed roles

1. `design_agent`
   - intervention/comparator structure
   - valid arms
   - tissues
   - timepoints
   - units
   - factorial structure

2. `narrative_agent`
   - abstract/result/conclusion claims
   - direction and qualitative strength
   - significance language

3. `table_agent`
   - table-derived numeric claims
   - means, variance, n
   - source table and row identifiers

4. `figure_agent`
   - figure-only targets
   - caption-derived claims
   - whether target data is graph-only

5. `benchmark_agent`
   - what construct the benchmark appears to use
   - what rows seem benchmark-comparable
   - likely mismatches

6. `consistency_agent`
   - contradictions across the five roles
   - final evidence profile

## Reusable local assets

- universal LLM review input structure:
  - `outputs/universal_llm_inputs/*/llm_review_inputs.jsonl`
- existing review batch structures:
  - `outputs/validated_review_batches/*`
  - `outputs/effector_review_batches/*`
- row-audit outputs:
  - `outputs/row_audit/*/row_audit.jsonl`
- adjudication outputs:
  - `outputs/llm_decisions/*`
  - `outputs/codex_decisions/*`
- current claim-level merged feature tables:
  - `outputs/combined_analysis/*_claim_features_merged_2026-03-27.csv`

## Prototype plan

For a small batch of papers:
- create one prompt template per role
- give each role the full paper context
- require structured JSON output
- merge the role outputs into one contradiction / consilience table

## Suggested first pilot papers

- `019_Baxter_1994`
- `026_Seneweera_1997`
- `035_Oksanen_2005`
- `015_Pleijel_2009`

These span:
- concentration vs content
- tissue mismatch
- arm mismatch
- figure-only targets
- one clean control

## Output target

One row per `paper_id x claim_key` with role-specific fields:
- design constraints
- narrative direction
- table grounding
- figure-only indicator
- benchmark comparability
- contradiction list
- consilience verdict
