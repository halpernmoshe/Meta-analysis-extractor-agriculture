# File map

Every file in the repository, by folder: 741 files.

| Path | Content | Files |
|---|---|---:|
| `README.md`, `verify.py`, `run_all.ps1`, `run_all.sh`, `requirements.txt`, `LICENSE`, `.gitignore`, `.gitattributes` | Entry point, checks, wrappers, setup and licence. | 8 |
| `0_sources/` | README, `comparator_datasets.csv`, `source_papers.csv`, `corpus_mislabels_D2.csv`. | 4 |
| `0_sources/raw/` | Published data files for Loladze, Li J, Li X (biochar) and Boldorini, each with `ATTRIBUTION.txt`; `hui/DOWNLOAD.txt`. | 9 |
| `1_extraction/README.md` | How the extraction was done. | 1 |
| `1_extraction/protocol/extract-meta-analysis/` | Claude Code extraction skill (SKILL.md, 5 example files, 2 scripts, 1 schema). | 9 |
| `1_extraction/protocol/boldorini_2026-08/` | `EXTRACTION_PROMPT.md`, the prompt for the August 2026 Boldorini re-extraction. | 1 |
| `1_extraction/records/` | 178 JSON extraction records (biochar 28, boldorini 18, hui 37, li_j 49, loladze 46) and `SHA256SUMS.txt`. | 179 |
| `2_conversion/README.md` | Key-table format, decoders and builders. | 1 |
| `2_conversion/ai_side/` | Five AI-side decoders and `verify_source_record_release.py`. | 6 |
| `2_conversion/reference_side/` | README, four reference-side builders, and `row_trace/` (one trace per dataset). | 10 |
| `2_conversion/keys/<dataset>/ai/` | AI-side key tables: biochar_v2 28, boldorini 18, hui_v4 29, li2022_v2 49, loladze_v2 46. | 170 |
| `2_conversion/keys/<dataset>/gt/` | Published-reference key tables: biochar_v2 28, boldorini 19, hui_v4 26, li2022_v2 181, loladze_v2 50. | 304 |
| `3_evaluation/` | README and six analysis scripts. | 7 |
| `3_evaluation/expected_outputs/` | Expected printed output of the seven analysis scripts. | 7 |
| `3_evaluation/round2_additional_analysis/` | Coverage script, its five CSV tables and its report. | 7 |
| `4_figures/` | README and three figure scripts. | 4 |
| `4_figures/expected_outputs/` | Values printed by the Figure 2 and Figure 3 scripts. | 2 |
| `4_figures/figures/` | Figures 1, 2, 3 and S1. | 4 |
| `docs/` | This file, `CHANGELOG.md`, `TESTED_ENVIRONMENT.md`. | 3 |
| `docs/decoder_ledgers/` | One decoding log per AI-side decoder (the Boldorini one is `boldorini_march_style_DECODER_LEDGER.md`). | 5 |

The decoder ledgers and comments inside the decoders use the previous folder names
(`source_records/`, `runs/<dataset>/keys/`, `decoders/`); `CHANGELOG.md` gives the mapping.
