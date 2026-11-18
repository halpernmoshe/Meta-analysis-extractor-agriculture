# Stage 0: sources

This folder records where every input comes from: the five published reference datasets and the
primary papers in each corpus.

## The five published reference datasets

| Dataset | Data location | Data licence | File | SHA-256 | In this repository |
|---|---|---|---|---|---|
| Loladze I. 2014. eLife 3:e02245. https://doi.org/10.7554/eLife.02245 | Dryad, https://doi.org/10.5061/dryad.6356f (version 1, 2014-05-16) | CC0 1.0 | `CO2 Dataset.xlsx` | `67f8591aadad25fc27f5e9c40b05c1da49b2d3bb2e4a6b881af289aaa7fbaf06` | included: `raw/loladze/` |
| Hui X, Luo L, Chen Y, Palta JA, Wang Z. 2025. Nature Communications 16:3913. https://doi.org/10.1038/s41467-025-58397-y | Publisher supplement (Supplementary Data 1-6): https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-58397-y/MediaObjects/41467_2025_58397_MOESM5_ESM.xlsx | No separate data licence; article CC BY-NC-ND 4.0 | `41467_2025_58397_MOESM5_ESM.xlsx` | `2e028151275ee1c8c626eb908bdd838d4aa957d6674b4da805624144e18ac634` | not included; download it and place it in `raw/hui/` (see `raw/hui/DOWNLOAD.txt`) |
| Li J, Van Gerrewey T, Geelen D. 2022. Frontiers in Plant Science 13:836702. https://doi.org/10.3389/fpls.2022.836702 | Publisher supplement, Data Sheet 2: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.836702/full#supplementary-material | No separate data licence; article CC BY 4.0 | `Data_Sheet_2.XLSX` | `cb871f5855303305cdf42798b531d4b5cff9d931bfbb54b3a92830453821835b` | included: `raw/li_j/` |
| Li X, Wu D, Liu X, et al. 2024. Scientific Data 11:57. https://doi.org/10.1038/s41597-023-02867-9 | figshare, https://doi.org/10.6084/m9.figshare.24781764.v1 (collection https://doi.org/10.6084/m9.figshare.c.6622375.v1) | CC0 | `BiocharDS_V1.0.csv` | `808acad395c0cc3e69e70238ad235c07343f8947054a73d34a47f22331bc7372` | included: `raw/biochar/` |
| Boldorini GX, McCary MA, Romero GQ, et al. 2024. Proceedings of the Royal Society B 291:20232522. https://doi.org/10.1098/rspb.2023.2522 | figshare, https://doi.org/10.6084/m9.figshare.25283296.v1 (collection https://doi.org/10.6084/m9.figshare.c.7090120.v1) | CC BY 4.0 | `rspb20232522_si_005.csv` | `23a0ee8789422bb81eb51b7605e0721c9f0031a1de2de263167fb00f233c09db` | included: `raw/boldorini/` |

Each included file is unmodified and has an `ATTRIBUTION.txt` beside it (citation, location,
licence, SHA-256). The files were downloaded from the locations above on 18 November 2026. For
Loladze, Hui, Li X and Boldorini the downloaded file is identical (SHA-256) to the file used in the
analysis. For Li J the analysis read a copy that had been re-saved in Excel; its bytes differ from
the publisher's file, but every cell value and formula is the same, and the reference-side
conversion gives identical key tables from the publisher's file (see
`2_conversion/reference_side/`).

`comparator_datasets.csv` gives the details for each dataset: data-availability statement, data
and article licences with their sources, version, sheets and row counts, and how the download was
checked.

## Primary papers: `source_papers.csv`

`source_papers.csv` lists the 150 primary papers analysed in the five corpora (Supplementary
Material S8), one row per paper, with the AI-side key table and the reference key table that belong to it.
Each row gives the first author, year, title, journal and DOI,
and `doi_source` says where the DOI came from:

| DOI source | Papers |
|---|---:|
| Supplementary Material S8 | 93 |
| Extraction record, confirmed in Crossref | 13 |
| Li J published dataset (`doi` column), checked in Crossref | 2 |
| Crossref lookup (title, first author and year all agree) | 33 |
| No DOI found | 9 |

The bibliographic fields inside the extraction records (`1_extraction/records/`) were written by
the extraction step and may be inexact. `source_papers.csv` is the authoritative citation list.

Source PDFs are not redistributed (publisher copyright).

## Other files

`corpus_mislabels_D2.csv` lists 17 source PDFs that turned out to contain a different paper from
the one named. They were excluded as sourcing errors. The exclusions are applied in the decoders
and analysis scripts; this file records them and is not read by any script.
