# Claude Stage 4 Recovery Note

## Purpose

This note is operational, not conceptual.

It records what was learned about interrupted Stage 4 extraction runs that used the validated Claude Code extractor:

- `pipeline_replication/extract_stage4_agent.py`
- `claude universal metaanalysis pipeline/agent_extract.py`

The main lesson is that many `_ZERO.json` files are **not** real “no data found” papers. Some are merely quota / usage-limit artifacts and must be retried.

## Core Rule

Do not treat every `_ZERO.json` extraction file as evidence that the paper truly contains no usable data.

For Stage 4 runs using the Claude Code extractor, `_ZERO.json` files can mean two different things:

1. **Quota / usage failure artifact**
2. **Genuine empty extraction after a real read**

These must be separated before making any progress claims, paper counts, or downstream stage decisions.

## How To Tell The Difference

Check the `_ZERO.json` file itself, especially:

- `raw_output_preview`
- token usage fields inside the embedded Claude response
- runtime duration

### Invalid Zero: Retry Required

If `raw_output_preview` contains:

- `You're out of extra usage`

then the zero is **not valid evidence** that the paper had no usable data.

This means Claude failed before actually processing the paper.

Common signatures:

- `result` contains `You're out of extra usage`
- `input_tokens: 0`
- `output_tokens: 0`
- very short runtime

These files must be treated as **failed extractions** and retried.

### Plausible Genuine Zero: Do Not Automatically Retry

If the `_ZERO.json` file shows:

- a normal model run
- nonzero token usage
- `result: "[]"`

then the extraction at least actually ran.

That does **not** prove the paper is truly off-target, but it does mean the zero is a real extraction outcome rather than a quota crash.

These should be left alone unless there is a specific reason to manually audit them.

## Verified Examples

### Quota-Failure Zero

Examples checked directly:

- `amf_inoculation_yield/4_extract/per_paper/Aghili_2014_Green_Manure_Addition_to_Soil_agent_ZERO.json`
- `legume_rotation/4_extract/per_paper/ABationo_2000_Rotation_and_nitrogen_fertilizer_effects_agent_ZERO.json`

These contained:

- `You're out of extra usage`
- zero token usage

Therefore they are invalid as evidence of “no data.”

### Genuine Empty Run

Example checked directly:

- `elevated_co2_face_yield/4_extract/per_paper/Ahmad_2023_MH_21_A_NOVEL_HIGH_agent_ZERO.json`

This contained:

- normal model runtime
- nonzero token usage
- `result: "[]"`

So this is a plausible genuine empty extraction, not a quota artifact.

## Recovery Counts Already Verified

| Topic | `_agent.json` | `_ZERO.json` | `_ERROR.json` | usage-failure zeros |
|-------|---------------|--------------|---------------|---------------------|
| `amf_inoculation_yield` | 18 | 89 | 1 | 86 |
| `legume_rotation` | 10 | 279 | 1 | 250 |
| `elevated_co2_face_yield` | 12 | 33 | 1 | 10 |
| `biochar_tropical_yield` | 0 | 0 | 0 | 0 |
| `cover_crop_corn_yield` | 0 | 0 | 0 | 0 |

Interpretation:

- `amf_inoculation_yield` is mostly interrupted, not genuinely empty
- `legume_rotation` is mostly interrupted, not genuinely empty
- `elevated_co2_face_yield` is mixed: some real zeros, some quota failures
- `biochar_tropical_yield` and `cover_crop_corn_yield` have not meaningfully started Stage 4

## Required Procedure Before Resuming Stage 4

Before restarting any extraction job:

1. Enumerate `4_extract/per_paper/*_ZERO.json`
2. Split them into:
   - usage-failure zeros
   - genuine-run zeros
3. Retry only the usage-failure zeros plus any papers with `_ERROR.json`
4. Preserve existing successful `_agent.json` outputs
5. Do not report `_ZERO.json` counts as “papers with no usable observations” unless the zero was a genuine run

## Required Procedure Before Reporting Progress

When summarizing topic progress, distinguish:

- successfully extracted papers
- genuine zero-observation papers
- retry-required quota-failure papers
- not-yet-started papers

Do not collapse those categories into a single “processed” count.

## Immediate Implication For Current Project State

The confirmatory V2 extraction set is still incomplete.

At minimum, the following topics still require substantial Stage 4 recovery or completion:

- `amf_inoculation_yield`
- `legume_rotation`
- `elevated_co2_face_yield`
- `biochar_tropical_yield`
- `cover_crop_corn_yield`

Any narrative suggesting those extraction runs are complete should be treated as provisional until the quota-failure retry pass is done.
