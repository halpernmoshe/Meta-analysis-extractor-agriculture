# Finalized extraction source records

This directory contains the complete corrected JSON artifact set underlying the five-dataset
analysis: 178 paper extraction records plus 6 retained biochar pipeline auxiliaries. The paper
records are the source inputs from which the deposited AI key tables under `runs/*/keys/ai/` were
built. The auxiliaries are not decoded. These files are not source PDFs or the authoritative
published-reference tables; some biochar auxiliaries repeat comparator-derived fields for audit and
variance/matching provenance.

| Directory | Dataset | JSON files | Bytes | Relationship to final analysis |
|---|---|---:|---:|---|
| `biochar/` | Li X et al. 2024 | 34 | 617,518 | Frozen March 2026 set: 28 paper extractions plus 6 pipeline auxiliaries. The decoder accepts only objects with a top-level `paper_id` and `observations` list; the six auxiliaries are retained for a complete, checksummed provenance set but do not produce key rows. |
| `boldorini/` | Boldorini et al. 2024 | 18 | 184,865 | Corrected August 2026 re-extraction used by the final analysis (80 decoded rows). This replaces the earlier March hand-authored/contaminated set. |
| `hui/` | Hui et al. 2025 | 37 | 538,547 | Frozen March 2026 extraction set. Corpus and row exclusions are applied by the decoder and documented in `decoders/hui_DECODER_LEDGER.md`. |
| `li_j/` | Li J et al. 2022 | 49 | 719,848 | Frozen March 2026 extraction set. Analysis uses published-reference paper identifiers only for the disclosed author-year crosswalk; two same-author/year collisions are excluded after source-title checks. |
| `loladze/` | Loladze 2014 | 46 | 764,780 | Frozen March 2026 extraction set. All 1,646 source records are emitted as AI key rows; unpairable coordinates remain visible rather than being dropped. |
| **Total** |  | **184** | **2,825,558** |  |

`SHA256SUMS.txt` records the SHA-256 digest and byte size of every JSON file. From the repository
root, run:

```bash
python verify_source_record_release.py
```

The verification parses every JSON, checks the manifest, runs all five deposited decoders in a
temporary directory, and requires the generated AI keys to be content-identical to the frozen analysis
inputs after normalizing CSV line endings. This preserves exact fields, ordering, and row values while
accommodating the historical CRLF line endings in the frozen Boldorini keys. The expected result is
170 generated CSV files containing 3,151 rows. Hui's frozen analysis
input is the decoder's documented `strict` variant; `hui_method_field_first` is generated only as a
sensitivity variant.

## Relationship to the frozen reference keys

The JSON files describe the AI extraction side. The independently supplied published-reference
side remains frozen under each `runs/<dataset>/keys/gt/` directory. Decoding creates categorical AI
keys; subsequent analysis scripts pair those keys to the reference keys without using outcome values
to choose matches. Li J reads the reference paper-ID vocabulary for its disclosed crosswalk, and the
final Boldorini joiner reads the reference's structural treatment-token form. Those comparator-informed
steps are explicit in their decoder ledgers.

## What is not provided

- No full source PDFs are included because of publisher copyright.
- No credentials, account/session metadata, conversational transcripts, or embedded PDF/base64 data
  are present in this release set.
- Superseded extraction attempts, consensus experiments, validation-result dumps, variance-only raw
  model outputs, and the contaminated March 2026 Boldorini records are excluded because the final
  analysis does not use them.
- Prompts and complete run logs are not part of this source-record deposit. A standalone variance-only
  methods prompt already tracked under `round2_additional_analysis/` is retained as protocol
  documentation; its raw experimental outputs remain excluded.

The public `dev-archive-pre-curation` branch is a historical development snapshot and contains older
raw JSON/prompt material. It is not the source of the corrected release inputs and must not be used to
reconstruct the reported analysis. It remains available for provenance; Git history and remote refs
have not been rewritten.
