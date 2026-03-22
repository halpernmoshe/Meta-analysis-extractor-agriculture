# Submission Audit Notes

Date: 2026-03-22

Scope reviewed:
- Main manuscript source: `PAPER_FINAL_v23.md`
- Main submission files: `halpern_2026.docx`, `halpern_2026_v2.docx`
- Supplementary files: `supplementary/`, `halpern_2026_supplementary.docx`
- Reproduction package: `reproduction/`
- Package metadata: `README.md`

## Overall assessment

The statistical content appears internally consistent, but the submission package is not fully ready as assembled. The main issues are packaging and document consistency rather than core numerical results.

## High-priority issues to fix

### 1. Main manuscript only includes 3 of the 6 main figures

Files:
- `PAPER_FINAL_v23.md`
- `halpern_2026.docx`
- `halpern_2026_v2.docx`

What I found:
- The manuscript source contains figure placeholders only for `FIGURE 1`, `FIGURE 2`, and `FIGURE 3`.
- The `figures/` folder contains 6 main figure files:
  - `figure1_architecture.png`
  - `figure2_scatter_plots.png`
  - `figure3_cross_dataset_comparison.png`
  - `figure4_bland_altman.png`
  - `figure5_source_type_accuracy.png`
  - `figure6_aggregate_effects.png`
- The generated main `.docx` files contain only 3 embedded images.

Why this matters:
- The submission package includes 6 figure assets, but the main paper currently only places 3 of them into the manuscript export.
- Reviewers/editors may see missing figures or a mismatch between the figure folder and the manuscript.

Recommended change:
- Add figure placements/captions for all 6 main figures in the manuscript source and regenerate the main `.docx`.

### 2. Figure numbering/captions do not match the actual figure files

Files:
- `PAPER_FINAL_v23.md`
- `reproduction/generate_figures.py`
- `reproduction/build_docx.py`

What I found:
- The manuscript currently describes:
  - Figure 1 as scatter plots
  - Figure 2 as per-paper MAE distribution
  - Figure 3 as Bland-Altman plots
- The actual generated figure set in `reproduction/generate_figures.py` is:
  - Figure 1: system architecture
  - Figure 2: scatter plots
  - Figure 3: cross-dataset validation metrics
  - Figure 4: Bland-Altman
  - Figure 5: source-type accuracy
  - Figure 6: aggregate effect reproduction
- There is no generated per-paper MAE figure matching the current Figure 2 caption in the manuscript.

Why this matters:
- The text and the figure assets are not aligned.
- Even if all files are present, the wrong images may be attached to the wrong captions.

Recommended change:
- Decide on the final 6-figure set, then align the manuscript figure callouts, captions, and docx build mapping to that exact set.

### 3. Supplementary material is duplicated as placeholders inside the main manuscript

Files:
- `PAPER_FINAL_v23.md`
- `halpern_2026.docx`
- `halpern_2026_v2.docx`

What I found:
- The main manuscript includes:
  - `# Appendix A`
  - then a `# Supplementary Material` section with placeholder lines for `Table S1-S4` and `Figure S1-S3`
- The separate supplementary package already exists as:
  - `supplementary/SUPPLEMENTARY_MATERIALS.md`
  - `halpern_2026_supplementary.docx`

Why this matters:
- The main paper `.docx` currently contains supplementary placeholders rather than full supplementary content.
- This makes the main manuscript look incomplete and duplicates material that should likely be submitted separately.

Recommended change:
- Remove the supplementary placeholder section from the main manuscript or replace it with a one-line note pointing to the separate supplement.

### 4. Supplementary title is inconsistent with the main manuscript

Files:
- `supplementary/SUPPLEMENTARY_MATERIALS.md`

What I found:
- Main manuscript title:
  - `A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets`
- Supplementary title currently says:
  - `Achieves Equivalence with Human Coders Across Five Independent Meta-Analysis Datasets`

Why this matters:
- Title mismatch makes the package look version-mixed.

Recommended change:
- Make the supplementary title exactly match the final main manuscript title or a clearly approved short form.

### 5. `Hui 2023` vs `Hui 2025` naming drift across the supplement/reproduction outputs

Files:
- `supplementary/SUPPLEMENTARY_MATERIALS.md`
- `supplementary/Table_S1_TOST_results.csv`
- `supplementary/Table_S2_per_paper_agreement.csv`
- `supplementary/Table_S3_variance_recovery.csv`
- `supplementary/Table_S4_agent_replication.csv`
- `reproduction/output/*.json`
- several reproduction scripts still refer to `Hui 2023`

What I found:
- The main paper consistently uses `Hui 2025`.
- The supplement and some reproduction outputs still use `Hui 2023`.

Why this matters:
- This looks like a stale version merge and can raise questions about which dataset/publication year is actually being validated.

Recommended change:
- Standardize the label everywhere in submission-facing materials.
- If the reproduction code intentionally uses an internal folder name like `hui2023_full_35`, keep that internal path if needed, but do not expose inconsistent naming in the final documents.

### 6. Supplementary Table S4 is malformed/inconsistent with the main manuscript

Files:
- `supplementary/SUPPLEMENTARY_MATERIALS.md`
- `supplementary/Table_S4_agent_replication.csv`
- `PAPER_FINAL_v23.md`

What I found:
- Main paper Table 9 reports:
  - Loladze 2014: 41 papers, 665 matched obs, r = 0.816, effect diff 0.09
  - Hui 2025: 24 papers, 362 matched obs, r = 0.946, effect diff 6.31
  - Li 2022: 30 papers, 204 matched obs, r = 0.849, effect diff 0.23
- Supplementary Table S4 instead shows:
  - Loladze row with `1231` matched obs and `95` papers, which are the overall totals across all replicated datasets, not Loladze-only
  - Li row with missing paper/obs counts
  - Hui row mostly blank and still labeled `Hui 2023`

Why this matters:
- This is a visible table-level inconsistency between the main paper and the supplement.

Recommended change:
- Rebuild Table S4 from the same source used for main Table 9 and confirm row-level values before regenerating the supplementary `.docx`.

## Medium-priority issues

### 7. README is stale relative to the actual package

File:
- `README.md`

What I found:
- README title does not match the current manuscript title.
- README references `PAPER_FINAL_v23.docx`, but the top-level package actually contains:
  - `halpern_2026.docx`
  - `halpern_2026_v2.docx`
  - `halpern_2026_supplementary.docx`

Why this matters:
- Makes the package look disorganized and version-mixed.

Recommended change:
- Update README file names and title to match the actual top-level submission files.

### 8. Reproduction/build script output handling is brittle when the docx is open

Files:
- `reproduction/build_docx.py`

What I found:
- Running the full reproduction pipeline succeeded for stats and figures but failed on the final docx write because `halpern_2026.docx` was locked:
  - `PermissionError: [Errno 13] Permission denied`

Why this matters:
- This is not a scientific issue, but it makes last-minute rebuilds fragile.

Recommended change:
- Either close the `.docx` before rebuilding or add a fallback output filename when the default file is open.

## What appears to be okay

- Core stats reproduced successfully from the reproduction package.
- Reproduction checks reported all manuscript numerical checks passing within tolerance.
- All main figure PNGs are present.
- All supplementary figure PNGs are present.
- Supplementary `.docx` contains 3 embedded supplementary images.
- Main `.docx` contains Appendix A text.

## Recommended submission checklist

1. Finalize the canonical manuscript title.
2. Standardize `Hui 2025` vs `Hui 2023` across all submission-facing documents.
3. Decide the final 6 main figures and align manuscript captions to the actual assets.
4. Regenerate the main `.docx` so all 6 main figures are embedded.
5. Remove placeholder-only supplementary content from the main manuscript.
6. Fix and regenerate Supplementary Table S4.
7. Regenerate `halpern_2026_supplementary.docx`.
8. Update `README.md` to match the real package contents.
9. Do one final manual open-check of:
   - main `.docx`
   - supplementary `.docx`
   - figures folder
   - supplementary folder

## Verification notes

I did not modify the manuscript, supplement, figures, or scripts.
I only reviewed the package and produced this audit note.
