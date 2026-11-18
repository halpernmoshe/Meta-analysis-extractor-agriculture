# Reference-side conversion

These files convert each published reference dataset (`0_sources/raw/`) into the reference key
tables in `../keys/<dataset>/gt/`. They write to a new temporary folder by default, or to the folder
given with `--out`, and never into `../keys/`. `python verify.py` (repository root) runs them and
compares the output with the deposited tables after line-ending normalisation.

| Dataset | Script | Status | Result against the deposited tables |
|---|---|---|---|
| Loladze 2014 | `build_reference_keys_loladze.py` | original conversion script, with paths parameterised | 754 of 754 rows identical (50 files) |
| Li J et al. 2022 | `build_reference_keys_li_j.py` | original conversion script, with paths parameterised | 1,108 of 1,108 rows identical (181 files), from the publisher's file |
| Hui et al. 2025 | `build_reference_keys_hui.py` | first step reconstructed in November 2026; second step original | 546 of 546 rows identical (26 files); needs the workbook to be downloaded |
| Boldorini et al. 2024 | `build_reference_keys_boldorini.py` | original conversion script, with paths parameterised | 47 row ids and all means identical; 45 rows identical in every column; 2 rows differ in `crop` and `evidence` (below) |
| Li X et al. 2024 (biochar) | none | cannot be re-run as a script (below) | every row traced to the published file (`row_trace/`) |

The scripts need openpyxl (Loladze, Li J, Hui) and pandas (Hui).

## Per dataset

**Loladze 2014 and Li J et al. 2022.** The key tables are regenerated exactly from the published
file by the original conversion scripts. Only the input and output paths were changed, so that
they come from the command line.

**Hui et al. 2025.** The key tables are regenerated exactly. The conversion had two steps. The
first step (workbook to one combined table of the grain-zinc rows of Supplementary Data 2 to 4) is a
November 2026 reconstruction, because the original script for it was lost. The second step
(combined table to key tables) is the reference-side part of the original script. The workbook is
not redistributed; download it as described in `0_sources/raw/hui/DOWNLOAD.txt`. Without it,
`verify.py` reports this check as SKIPPED. The second step decides which publications get a key
table from the file names of the AI-side key tables (34 names, listed in the script); it does not
read any AI-side value. Its `paper_id` is the first author's surname and the year; for four ids
(`rehman_2018`, `wang_2012`, `zhang_2012`, `zhang_2017`) this joins rows from two different
publications.

**Boldorini et al. 2024.** The key tables are regenerated from the published file, with one known
difference. After the script ran, the crop labels of two rows (Garfinkel 2020,
`gt__B06_Garfinkel_2020__id_44` and `__id_45`, corn and soybean) were corrected by hand. The script
does not encode that correction, so the deposited tables differ from the script output in those
two rows' `crop` and `evidence` fields. All row ids and all treatment and control means are the
same. The published file has no unit column. It has a `study_year` column, which the script copies
into the `evidence` field but does not use as the time point. The unit of all 47 rows, the time
point of 17 rows and the treatment level of 26 rows were set through per-row overrides in the
script (`setov`); otherwise the time point is `pooled` and the treatment level is the published
predator group. The script's comments give the basis for each override: some cite the source paper
(for example a table number), and several state that the token was chosen to match the AI-side key
table.

**Li X et al. 2024 (biochar).** The published file omits design descriptors that distinguish rows within a paper (for example the level of a co-applied amendment). The reference key tables for this dataset were prepared by reading each source paper alongside the published file: each published row was located in its source paper using that row's own published values, and its design descriptors were recorded. The agent was instructed not to consult the AI-side data. This step was carried out by a language-model agent and cannot be re-run as a script. A deterministic unit-harmonisation step followed. It converted reference rows to kg/ha for papers whose AI-side key table declares kg/ha, using each reference row's own declared unit (×1000 for t/ha, ×100 for q/ha). This is a rule on unit labels: it reads the declared unit of the AI-side table, not its values. The rule was applied to a list of 19 papers and converted rows in 13 of them. Every treatment and control mean in the Biochar reference key tables is the value in the published file, after the unit conversion where one applies; `row_trace/` gives the published row behind each key row.

The trace covers all 552 Biochar reference key rows. 468 match one published row exactly; 28 match
exactly two published rows that hold identical values (14 duplicated pairs); 56 match after a
factor of 1000 (the key table holds kg/ha where the published file holds t/ha). Together they are
552 distinct published rows, which are all the crop-yield rows of the 28 studies. In 86 of the
552 rows (5 papers) the crop label in the key table differs from the published file's `CropType`
(ignoring case, spaces and underscores); these labels were taken from the source papers, and the
`evidence` field of each such row quotes the paper. Most are refinements, for example a broad class
such as 'vegetable' recorded as the specific crop (amaranth, bok choy, coriander, water spinach,
rapeseed), or rice and wheat recorded separately for grain and biomass. In two papers the label was
corrected (wheat recorded as barley; potato recorded as mash bean). The
unit-harmonisation step converted 258 rows from t/ha to kg/ha (×1000) and 12 rows from q/ha to
kg/ha (×100). The same step also changed other fields in 27 rows: `co_amendment_level` in 12,
`is_figure` in 12 and `co_amendment` in 3. Its script is not included, because the step before it
cannot be re-run.

## Hand-written mapping tables

Each script contains hand-written tables that are part of the conversion and are visible in the
script: paper-identifier crosswalks, vocabulary maps and per-row overrides.

| Script | Tables |
|---|---|
| `build_reference_keys_loladze.py` | `REF_TO_PAPER`: 51 entries mapping a published reference to a paper id; `TISSUE_MAP`: 16 entries |
| `build_reference_keys_li_j.py` | `STUDY_TO_AI`: 33 published study numbers mapped to paper ids (other studies keep an id `gt_studyNN`); `PBS_NORMALIZE`: 7 entries |
| `build_reference_keys_hui.py` | `AI_KEY_FILENAMES`: 34 file names; `EXCLUDE`: 1 paper |
| `build_reference_keys_boldorini.py` | `PAPER_ID`: 19 entries; `CROP_DEFAULT`: 13 entries; 37 per-row override calls (`setov`) covering all 47 rows, which set treatment level, time point, unit, crop, control token, figure flag, source location and evidence notes |

The paper-identifier crosswalks map each published study to the identifier of the corresponding
paper in the corpus, so that the two sides can be paired. Comments in the scripts give the reason
for individual entries.

## Row traces

`row_trace/<dataset>_row_trace.csv` gives, for every reference key row, the published file, sheet
and row (or record) it comes from, the published values used, the stored means and a match flag.
For Biochar it also gives the source location recorded in the key table and the unit conversion
applied. `verify.py` checks that every deposited reference key row appears once in its trace with a
match flag and the same means.
