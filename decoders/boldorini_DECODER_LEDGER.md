# Boldorini decoder ledger — rebuild 2026-08-19

**Read `boldorini_CONTAMINATION_EVIDENCE.md` first. These keys are not validation evidence.**

## Source
`01_INPUTS_FROZEN/boldorini/*.json` — 18 files, 48 observations, copied from
`meta_analysis_extractor/output/boldorini_extraction` (2026-03-18). Flat `observations[]`,
no multi-model fields. The values in this source were hand-authored against the reference.

## Decoder
Written fresh (`02_DECODERS/boldorini/decode_boldorini.py`) because the submitted pipeline had
**no AI-side decoder script for this dataset** — `runs/boldorini/keys/ai` was produced inline by
an LLM decode agent (`decoder=ai-decode`), so that hop was never reproducible from deposited code.
Only `decode_gt.py` existed. This is itself a reproducibility gap in the submitted repository.

## Field mapping (source only; no outcome value informs any key field)
| Key column | Source |
|---|---|
| paper_id | `paper_id` |
| outcome_canonical | constant `crop_yield_under_predator` |
| crop | top-level `crop`, snake_cased |
| treatment_level | design rule below, from `treatment_type` + `moderators.predator_group` |
| co_amendment | constant `none` (see change 1) |
| co_amendment_level | constant `0` |
| timepoint | year/season token parsed from `data_source` + descriptions + moderators |
| aggregation_level | constant `single_cell` |
| unit_canonical | `unit` verbatim |
| control_token | constant `absolute_control` |
| treatment_mean / control_mean | copied verbatim |
| is_figure | 1 when `source_type` starts with "fig" |

`treatment_level` rule, evaluated per paper from source fields only:
1. more than one distinct `treatment_type` in the paper -> `<predator_group>_<treatment_type>`
2. else more than one distinct `predator_group` -> `<predator_group>`
3. else -> `<treatment_type>`
4. else -> `<predator_group>` or `na`

## Changes vs the deposited (consensus-era) AI keys
1. **`co_amendment` set to `none` throughout.** The deposited AI keys packed the predator group
   into `co_amendment` on 5 rows (`bird`, `beetle`, `invertebrate`, `vertebrate`), while the GT side
   carries `none` on all 47 rows. Those 5 AI rows were therefore structurally incapable of pairing.
   A predator group is part of the treatment, not a co-applied amendment, so it belongs in
   `treatment_level`. This is a correctness fix, made without reference to any value.

## Record arithmetic
`records_in = 48`, `rows_out = 48`, `excluded = 0`. Decoder re-run twice: byte-identical.

## Vocabulary diff vs GT structural reference
| Column | Present on AI only | Present on GT only |
|---|---|---|
| treatment_level | (none) | `birds_exclusion`, `invertebrates_exclusion` |
| co_amendment | (none) | (none) |
| timepoint | (none) | (none) |
| unit_canonical | (none) | `count` |

The two GT-only `treatment_level` tokens are B08_Hooks_2003 and B19_Vichitbandha_2002, where the
GT decoder appended the design token but rule 2 above did not; `count` belongs to `Liere_2015`,
a paper with no AI-side source file at all. No AI-only tokens remain, so the rebuilt keys align
with the reference vocabulary better than the deposited ones did.

## Open item
Boldorini needs genuine re-extraction from its 18 source PDFs, blind to the reference, before any
Boldorini number can support a claim. That is an extraction task, not a decode task, and is out of
scope for this rebuild.
