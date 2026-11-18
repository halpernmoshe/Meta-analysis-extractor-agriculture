# Stage 1: AI extraction

## How the extraction was done

Extraction used Claude Code with the model Claude Opus 4.6, on a consumer subscription, not through
the API. Agents read each source PDF and wrote one JSON record per paper. The records were written
on these dates:

| Dataset | Records written |
|---|---|
| Loladze 2014 | 16 March 2026 |
| Hui et al. 2025 | 16 March 2026 |
| Li J et al. 2022 | 16 March 2026 |
| Li X et al. 2024 (biochar) | 18 March 2026 |
| Boldorini et al. 2024 | 20 August 2026 |

The extraction protocol is deposited as a Claude Code skill. It was written on 18 March 2026 to formalise the procedure used in the preceding extraction sessions, and was used for the Biochar corpus. The earlier corpora were extracted by Claude Code agents under instructions given in-session.

The skill's example configuration files state published effect sizes and expected directions. The
example configurations for Loladze, Hui and Li J were written on 18 March 2026, after those corpora
had been extracted.

The Predator-biocontrol (Boldorini) corpus was re-extracted by the same route on 20 August 2026, under a written prompt. That prompt is deposited in `protocol/boldorini_2026-08/EXTRACTION_PROMPT.md`. It is the text given to the extraction session, with two local folder paths replaced by placeholders.

A hosted language model does not return identical output when it is run again. The deposited records
are the extraction outputs analysed in the paper, and all downstream steps run deterministically
from them. Source PDFs are not redistributed (publisher copyright).

## Contents

| Path | Content |
|---|---|
| `protocol/extract-meta-analysis/` | The Claude Code skill, unedited: `SKILL.md`, `examples/` (configurations and an example output), `scripts/` (`summarize.py`, `validate.py`), `templates/` (observation schema). |
| `protocol/boldorini_2026-08/EXTRACTION_PROMPT.md` | The prompt used for the August 2026 re-extraction of the Boldorini corpus (local folder paths replaced). |
| `records/<dataset>/` | The deposited JSON records, one per paper. |
| `records/SHA256SUMS.txt` | SHA-256 digest, byte size and repository path of each record. |

The skill's `scripts/` are part of the protocol as written. They are not used by the analyses in this
repository, and `scripts/validate.py` needs NumPy and SciPy; SciPy is not in `requirements.txt`.

## The records

| Folder | Dataset | Records | Observation list key | Notes |
|---|---|---:|---|---|
| `biochar/` | Li X et al. 2024 | 28 | `observations` | |
| `boldorini/` | Boldorini et al. 2024 | 18 | `observations` | August 2026 re-extraction; replaces an earlier set that is not used. |
| `hui/` | Hui et al. 2025 | 37 | `consensus_observations` | Corpus and row exclusions are applied by the decoder. |
| `li_j/` | Li J et al. 2022 | 49 | `consensus_observations` | |
| `loladze/` | Loladze 2014 | 46 | `consensus_observations` | |
| **Total** | | **178** | | 2,494,170 bytes |

In the Loladze, Hui and Li J records the observation list is stored under the key
`consensus_observations`. This is a field name retained from an earlier file format so that existing
validation scripts could read the records. Each record is the output of a single extraction, not a
merge of several models or passes.

The bibliographic fields inside the records (title, authors, journal, DOI) were written by the
extraction step and may be inexact. `0_sources/source_papers.csv` is the authoritative citation list.

Some records belong to papers that were later excluded because the PDF held a different paper from
the one named; see `0_sources/corpus_mislabels_D2.csv`.

## Checking the records

`python verify.py` (repository root) checks every record against `SHA256SUMS.txt` (digest and byte
size) and confirms that no record is missing or extra.
`python 2_conversion/ai_side/verify_source_record_release.py` runs the same record check and the
AI-side decoder rebuild on their own.
