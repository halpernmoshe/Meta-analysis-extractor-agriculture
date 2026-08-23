# biochar — AI-side decoder ledger (independent rebuild, 2026-08-19)

Dataset: **biochar** (Li X 2024). Outcome: `crop_yield`.
Deliverable decoder: `02_DECODERS/biochar/decode_biochar.py`
Output keys: `03_KEYS/ai_rebuilt/biochar/*.csv` (28 files, 446 rows)
Machine-generated companion log: `06_LEDGER/biochar_DECODER_AUDIT.md`
Determinism: decoder run twice; the 28 CSVs are **byte-identical** across runs.
Combined SHA-256 of the 28 CSVs (concatenated in filename order):
`9ff298002a10d87c5e82e1ae6790b936cf230220df1d3d2eee50f43a9242b3af`

---

## 0. One-line summary

The AI side was rebuilt from the frozen March 2026 single-model Claude JSONs
only. **All 28 GT papers are covered** — no paper had to be excluded. The
rebuild produces **446 rows vs the deposited 500**; the entire 54-row
difference is attributable to exactly the three papers that were re-extracted
from PDFs on 2026-05-31 and spliced into the deposited keys
(`016_Li_B_2016`, `063_Asai_2009`, `145_Omara_2020`). Every other paper
reproduces its deposited row count exactly. `control_token` free text has
been closed to the GT's two-token vocabulary.

---

## 1. Source

| item | value |
|---|---|
| source dir | `01_INPUTS_FROZEN/biochar/` |
| `.json` files present | **34** |
| accepted as paper extractions | **28** |
| rejected as pipeline artefacts | **6** |
| source observations across the 28 papers | **432** |
| source mtime | 2026-03-18 (inside the declared window) |
| May 2026 re-extractions used? | **No** — not read, not spliced |
| deposited AI key tables consulted for values? | **No** — only the
means-stripped `00_SPEC/vocab_reference/biochar_ai_structural.csv`, and only
to recover the *encoding convention* for the four papers that had no
submitted decoder script (§3, STAGE C) |

### 1.1 Input filter (structural, per orchestrator correction)

A file is treated as a paper extraction **iff** it is a JSON object with a
truthy top-level `paper_id` **and** a top-level `observations` list. No
filename is hardcoded. The six rejected files were opened only far enough to
evaluate those two structural keys; **nothing else in them was read** (several
are known to contain GT column mappings and reference values, which are out of
scope for an outcome-blind AI-side decode).

| rejected file | reason |
|---|---|
| `alignment.json` | non-extraction pipeline artefact, not a paper |
| `llm_alignment_v2.json` | non-extraction pipeline artefact, not a paper |
| `llm_alignment_v3.json` | non-extraction pipeline artefact, not a paper |
| `validation_moderator_matched.json` | non-extraction pipeline artefact, not a paper |
| `variance_imputation_report.json` | non-extraction pipeline artefact, not a paper |
| `variance_recovered.json` | non-extraction pipeline artefact, not a paper |

---

## 2. Decoders started from

The submitted AI-side chain for biochar turned out to be **four** artefacts,
not two. The prompt named (1) and (4); (2) and (3) were found by inspection
and are load-bearing — without them 12 of the 28 papers have no decode branch
at all, and the AI side would be pre-patch while the frozen GT side is
post-patch.

| # | submitted artefact | papers | role |
|---|---|---|---|
| 1 | `…/matching/runs/biochar/keys/ai/_decode_ai.py` | 16 | per-paper decode: 081, 082, 101, 116, 126, 130, 133, 145, 153, 166, 184, 193, 207, 219, 223, 227 |
| 2 | `…/matching/runs/biochar/decode_ai_batch.py` | 8 | per-paper decode: 001, 007, 016, 021, 041, 063, 077, 078 |
| 3 | `…/matching/runs/biochar/patch_iter0.py` | 11 | post-decode token canonicalisation; **has both AI-side and GT-side halves** |
| 4 | `…/matching/runs/biochar_v2/_convert_ai_csv_to_jsonl.py` | 25 | CSV → JSONL carry-over, with `EXCLUDE = {016, 063, 145}` |

**No submitted script exists for `229_Shi_2022`, `231_Zhang_2021`,
`234_Malik_2018`, `242_Liu_2014`.** Their deposited AI key rows were
hand-authored. Reconstructed here as STAGE C.

### 2.1 Verification that `03_KEYS/gt/biochar` is the *post-patch* GT

`patch_iter0.py` wrote in place to both sides. The frozen GT confirms it ran:
`041_Guerena_2013` GT `co_amendment_level = 90pct` (patched from `97.2`),
`166_Haefele_2011` GT `timepoint = irri_2005ws` and `unit_canonical = kg/ha`
(patched from `t/ha`), `133_Pandit_2018` GT `is_figure = 1` on 12 of 15 rows,
`184_Yeboah_2018` GT `co_amendment = nitrogen` at level 0. Also
`03_KEYS/gt/biochar/*.csv` is row-for-row, field-for-field identical to
`00_SPEC/vocab_reference/biochar_gt_structural.csv` (552 rows, 0 diffs).
**Therefore the AI-side half of `patch_iter0.py` is part of the submitted AI
decode chain and is reproduced here as STAGE D.** The GT-side half is *not*
reproduced (GT is frozen as submitted and already carries it).

---

## 3. Structure of the rebuilt decoder

| stage | provenance | what it does |
|---|---|---|
| A | `decode_ai_batch.py` | decodes the 8 "generic-schema" papers |
| B | `_decode_ai.py` | decodes the 16 "labelled-schema" papers |
| C | **new** | decodes the 4 papers with no submitted script |
| D | `patch_iter0.py` (AI half only) | token canonicalisation / coordinate recovery from each row's own fields |
| E | **new** | declared-unit **string** normalisation (notation synonyms only) |
| F | **new** | `control_token` → GT closed vocabulary |

### 3.1 Change log — every change, with reason

| # | change | reason | class |
|---|---|---|---|
| C1 | `SRC` repointed from `output/biochar_extraction` to `01_INPUTS_FROZEN/biochar` | the one variable the rebuild changes | (a) input path |
| C2 | `OUT` repointed to `03_KEYS/ai_rebuilt/biochar`; `decoder` tag set to `rebuild_2026-08-19/biochar` (was `ai-decode`) | spec §Output contract | (a) |
| C3 | Stages A and B merged into one script with a shared `HEADER`/`add()` writer | the submission split the same job over two scripts writing to the same directory; merging is required to emit one CSV per paper deterministically | (c) mechanical |
| C4 | Both `sig3()` (from `_decode_ai.py`) and `round3()` (from `decode_ai_batch.py`) retained side by side | the two submitted decoders round differently (`round(f, 2-d)` vs `%g`). Unifying them would silently change `treatment_level` strings on the STAGE A papers. Kept per-paper-faithful. | (c) faithfulness |
| C5 | STAGE C written for 229 / 231 / 234 / 242 | no submitted script existed; without it 73 of 446 rows are missing. Coordinates come from each record's own explicit `treatment`/`control`/`crop`/`season`/`biochar_rate_t_ha`/`fertilizer`/`lime_rate_pct`/`yield_unit`/`data_source` fields. | (b) schema adapter |
| C6 | **`234_Malik_2018` schema adapter**: the records carry *no* `treatment_mean` field. Each record explicitly supplies two outcomes (`*_mean_biomass`, `*_mean_grain`) → one key row per outcome. | An independent inventory reported this paper's March JSON as "all-null means". That is a **false alarm caused by a field-name assumption**: the means are present, under `treatment_mean_biomass` / `treatment_mean_grain`. All 28 rows carry means. | (b) |
| C7 | STAGE D reproduces only the AI-side half of `patch_iter0.py`; the GT-side half is omitted | GT is frozen as submitted and already patched (§2.1); re-applying would double-convert | (c) |
| C8 | `patch_iter0.fix_234` folded into STAGE C (it rebuilt `treatment_level` as `<feedstock>_<pct>pct` and set `co_amendment=quicklime`; STAGE C emits both directly). Retained as a runtime `assert`. | avoids a redundant string-surgery pass | (c) |
| C9 | **STAGE E, new**: unit-string normalisation, notation-only | see §3.2 | (c) normalisation defect |
| C10 | **STAGE F, new**: `control_token` closed to the GT vocabulary | explicitly requested; see §3.3 | (c) |
| C11 | Explicit structural input filter + per-file exclusion tally | orchestrator correction; also satisfies spec rule 5 (no silent drops) | (a) |
| C12 | CSVs written with `lineterminator="\n"` and files sorted by `paper_id` | byte-reproducibility across runs and platforms | (c) determinism |

**Not changed** (deliberately): every per-paper coordinate rule, every
rounding call, every `to_kgha()` conversion, and every hardcoded
`unit_canonical` in stages A/B is byte-for-byte the submitted logic.

### 3.2 STAGE E — unit-string normalisation (34 rows, 3 papers)

Notation-only. Strips parenthetical qualifiers, collapses whitespace, and maps
**dimensionally identical** spellings onto one token. No value is rescaled;
no factor other than 1 is involved. Operates on the unit string the source
record itself declared, so it is outcome-blind.

| paper | rows | before | after | justification |
|---|---|---|---|---|
| `016_Li_B_2016` | 6 | `t/ha (fresh weight total over 9 crops)` | `t/ha` | parenthetical is a qualifier, not a unit |
| `078_Wang_2012` | 16 | `g pot-1` | `g/pot` | same unit, different notation; `g/pot` is the spelling used elsewhere in this corpus on both sides |
| `231_Zhang_2021` | 12 | `Mg/ha` | `t/ha` | 1 Mg ≡ 1 tonne (exact identity, factor 1) |

Deliberately **not** normalised: `t/ha/yr` (229, 3 rows — genuinely per-year,
not the same dimension as `t/ha`) and `x10^4 kg/ha` (242, 6 rows — a
scale-factored unit whose 6 rows have null means, so there is nothing to
convert; left as declared rather than relabelled).

### 3.3 STAGE F — `control_token` closed to the GT vocabulary (111 rows)

**The defect.** GT uses only `{absolute_control, cofactor_matched_control}`.
`decode_ai_batch.py` line `ctok = cdesc.strip()` wrote the **raw control
description** into `control_token` for its 8 papers, and `_decode_ai.py`
invented a third token `co_factor_present_unmatched` for `153_Wei_2022`.
In the deposited keys this produced 28 distinct `control_token` values, 25 of
them free text (e.g. `Mineral fertilizer (MF), Indre, maize 2019`,
`No biochar, 90% N, 2007`). In the March-only rebuild the same defect affects
103 free-text rows (all 8 STAGE A papers, because 016 and 063 are no longer
overwritten by the May splice) plus the 8 `co_factor_present_unmatched` rows
= **111 rows**.

`control_token` is deliberately **not** part of the match key, so closing it
changes **no pairing whatsoever**. It only makes the recorded
control-definition concordance measurable.

**The rule** (deterministic; reads only the row's own control description,
carried in its own `evidence` field, plus its own decoded
`co_amendment_level`):

| rule | condition | result |
|---|---|---|
| F1 | control records **no** co-amendment / fertiliser background (zero or absent) | `absolute_control` |
| F2 | control records the co-amendment at the **same level** this row carries | `cofactor_matched_control` |
| F3 | control records a co-amendment background at a **different** level, or the row carried the non-closed token `co_factor_present_unmatched` | `absolute_control` (the closed vocabulary has no unmatched-cofactor token; logged as `demoted_unmatched_cofactor`) |

Result: 41 (paper, source-string) groups closed; the full row-level table is
in `biochar_DECODER_AUDIT.md`. Post-closure the AI side uses **only** the two
GT tokens (238 `absolute_control` / 208 `cofactor_matched_control` vs GT
449 / 103).

Two papers where the closure disagrees with GT's own reading, recorded as
concordance divergence (not pairing changes):
`021_Nobile_2022` — GT calls the mineral-fertiliser reference
`cofactor_matched_control` (18 rows); rule F3 demotes it to
`absolute_control`, because the treatment's co-amendment (compost 8 t/ha) is
absent from that control. `041_Guerena_2013` — GT calls the "No biochar, 90 % N"
reference `absolute_control` (12 rows); rule F2 makes it
`cofactor_matched_control`, because the control carries the same 90 % N
background the treatment row records.

---

## 4. Field mapping (raw JSON field → key column)

The March corpus is **not** one schema. It is four schema families, seven
papers each — a batching artefact of the March extraction run. This is why the
submitted decode needed per-paper branches.

| family | papers | mean fields | dose field | crop | time |
|---|---|---|---|---|---|
| **A** generic | 001, 007, 016, 021, 041, 063, 077 | `treatment_mean`, `control_mean` | `moderators.biochar_rate_tha` | `moderators.crop` | parsed from `treatment_description` |
| **B** labelled | 078, 081, 082, 101, 116, 126, 130 | `treatment_mean`, `control_mean` | `biochar_rate_tha` / `biochar_rate_pct` | `crop` | `season` |
| **C1** obs_id + SE | 133, 145, 153, 166, 184, 193, 207 | `treatment_mean`, `control_mean` | `biochar_rate_t_ha` / `_Mg_ha` | `crop` | `year` / `season` |
| **C2** text labels | 219, 223, 227, 229, 231, 234, 242 | `treatment_mean` / `control_mean`, **except 234** (`*_mean_biomass`, `*_mean_grain`) | `biochar_rate_t_ha` / `biochar_rate_pct` | `crop` | `season` |

Column-by-column:

| key column | derived from | notes |
|---|---|---|
| `row_id` | `paper_id` + `obs_id` (families B/C1) or enumeration index (A/C2); `234` appends `_biomass` / `_grain` | not a key |
| `side` | constant `ai` | |
| `paper_id` | top-level `paper_id` | identical token vocabulary to GT (28/28, zero diff) |
| `outcome_canonical` | constant `crop_yield` | |
| `crop` | `moderators.crop` or `crop`, lowercased, snake_cased, parenthetical cultivar stripped | |
| `treatment_level` | biochar dose (`biochar_rate_tha` / `_t_ha` / `_Mg_ha` / `_pct`), 3 s.f. | blank for `077` (dose is % of soil mass, no t/ha equivalent in the record) |
| `co_amendment` | `fertilizer`, `moderators.P_source`, `lime_rate_pct`, or parsed from the treatment description | |
| `co_amendment_level` | same sources, 3 s.f.; `0` when no co-amendment | |
| `timepoint` | `year` → `y<YYYY>`; `season` → `season<N>` / `season<YYYY_YYYY>`; site/cultivar folded in for 063/145/166 per STAGE D | `pooled` when the record carries no time coordinate |
| `aggregation_level` | `single_cell`, or `pooled` for records the source itself labels a mean, or `documented_pooled` (101) | |
| `unit_canonical` | `unit` / `yield_unit`, then STAGE B `to_kgha()` where the submitted decoder converted, then STAGE E notation normalisation | |
| `control_token` | control description → closed vocabulary (STAGE F) | **not** a match key |
| `treatment_mean`, `control_mean` | `treatment_mean` / `control_mean` (or `*_biomass` / `*_grain` for 234), unit-converted only where the submitted decoder did | copied through; never used to decide a key |
| `source_locator` | `data_source` (or the submitted decoder's hardcoded table label) | |
| `is_figure` | `1` iff `source_type == "figure"` (STAGE C also accepts `"fig"` in `data_source`) | |
| `evidence` | the record's own treatment/control labels + descriptors | audit trail |
| `decoder` | constant `rebuild_2026-08-19/biochar` | |

---

## 5. Record arithmetic

**`records_in = rows_out − expansions + exclusions`**
`432 = 446 − 14 + 0`. **Zero source records dropped.**

The only expansion is `234_Malik_2018`, where each of 14 records explicitly
carries two outcomes (biomass and grain) → 28 rows.

### 5.1 Per-paper: March records in, key rows out, GT rows

| paper_id | March records in | key rows out | GT rows | key-coordinate overlap with GT | coverage note |
|---|---|---|---|---|---|
| 001_Adekiya_2019 | 12 | 12 | 12 | 12 | full |
| 007_Gathorne-Hardy_2009 | 4 | 4 | 20 | 0 | March is a conference abstract; only 4 of 25 factorial cells extracted, all means null; AI `timepoint=pooled` vs GT `y2008` |
| 016_Li_B_2016 | 6 | 6 | 42 | 0 | March read Fig 4c *totals over 9 crops* (1 row per N×biochar cell); GT reads Table 5 per-crop per-season |
| 021_Nobile_2022 | 19 | 19 | 18 | 0 | GT folds site+biochar-feedstock into `co_amendment`; AI uses `compost`/`8` |
| 041_Guerena_2013 | 16 | 16 | 12 | 12 | AI has 4 extra rows (the 1 t/ha-annually arm) |
| 063_Asai_2009 | 18 | 18 | 27 | 10 | see §7.1 (unit) and §7.2 (site/cultivar) |
| 077_Zhang_J_2019 | 12 | 12 | 12 | 0 | dose is % of soil mass; `treatment_level` blank vs GT `5/10/15/20` |
| 078_Wang_2012 | 16 | 16 | 20 | 0 | AI reads pot basis (`g/pot`), GT field basis (`t/ha`); AI `timepoint=pooled` vs GT `upland`/`paddy` |
| 081_Deenik_2010 | 14 | 14 | 12 | 0 | AI `crop=corn` vs GT `maize`; AI `co_amendment_level=present` vs GT `100`/`200` |
| 082_Jose_2013 | 18 | 18 | 18 | 0 | AI dose in % w/w (`0.5/1/2.5`) and `g/pot`; GT in t/ha (`11.25/22.50/56.25`) |
| 101_Liang_Feng_2014 | 9 | 9 | 30 | 6 | March = 4-season totals (÷4 to documented mean per STAGE D); GT = per-year rows. **March means are present, not null.** |
| 116_Farrell_2014 | 18 | 18 | 14 | 0 | AI `co_amendment=phosphorus_dap` vs GT `phosphorus` |
| 126_Arif_2017 | 16 | 16 | 14 | 0 | AI `aggregation_level=pooled` vs GT `single_cell` |
| 130_Azeem_2019 | 16 | 16 | 16 | 16 | full |
| 133_Pandit_2018 | 30 | 30 | 15 | 15 | AI has 15 extra `mustard` rows with no GT counterpart |
| 145_Omara_2020 | 12 | 12 | 24 | 12 | March carries only the N-matched contrast; GT (and the May re-extraction) also carry the 0-biochar absolute contrast |
| 153_Wei_2022 | 12 | 12 | 16 | 12 | full on the 12 shared cells |
| 166_Haefele_2011 | 28 | 28 | 36 | 14 | GT has the RH raw-husk arm (`treatment_level=49.5`, 18 rows, non-biochar); AI has 6 site-pooled rows and 4 dry-season IRRI rows GT lacks |
| 184_Yeboah_2018 | 9 | 9 | 15 | 9 | GT has the 6-row straw arm (`treatment_level=0`, non-biochar) |
| 193_Islami_2011 | 12 | 12 | 16 | 12 | full on the 12 shared cells |
| 207_Liu_2019 | 36 | 36 | 36 | 16 | GT splits 18 rows to `co_amendment=none`; AI keys all 36 as `nitrogen` |
| 219_Xie_2021 | 12 | 12 | 42 | 0 | AI dose is per-season (`2.25/6.75/11.2`), GT cumulative (`4.5/13.5/22.5`); AI `timepoint=season7`/`pooled`, GT `y2012…y2018` |
| 223_Dong_2019 | 16 | 16 | 16 | 16 | full |
| 227_Niu_2017 | 12 | 12 | 12 | 0 | AI `timepoint=y2014` vs GT `pooled`; AI `nitrogen` vs GT `npk_fertilizer` |
| 229_Shi_2022 | 21 | 21 | 18 | 18 | AI has 3 extra `annual_total` pooled rows (Table 2) |
| 231_Zhang_2021 | 12 | 12 | 15 | 0 | AI keys the fertiliser regime into `co_amendment` (`chemical_fertilizer`/`240_kgN_ha`); GT keys the biochar dose into `co_amendment_level` (`1.5`) |
| 234_Malik_2018 | 14 | 28 | 12 | 12 | AI has 14 extra `wheat_biomass` rows + 2 lime-only rows (`treatment_level=0`) |
| 242_Liu_2014 | 12 | 12 | 12 | 0 | AI `timepoint=season2011_2012`/`y2012` vs GT `season1`/`season2` |
| **TOTAL** | **432** | **446** | **552** | **192** | |

"key-coordinate overlap" = multiset intersection on the 7-field match key
(`outcome_canonical, crop, treatment_level, co_amendment, co_amendment_level,
timepoint, aggregation_level`). It is a *coverage* diagnostic only; it is not
a match rate and no value comparison was performed.

### 5.2 GT papers that could NOT be covered from March

**None.** All 28 GT papers have decodable March records. In particular, the
three papers the deposited keys re-extracted in May are all present and
decodable from March:

| paper | March records | decodable? | how handled here |
|---|---|---|---|
| `016_Li_B_2016` | 6 (family A) | yes | decoded from March Fig-4c totals via STAGE A. 6 rows, not the deposited 54. |
| `063_Asai_2009` | 18 (family A) | yes | decoded from March Table 3 via STAGE A + STAGE D `fix_063`. 18 rows, not the deposited 12 (March covers 8 site codes; the May version kept only Exp 1 / HK1). |
| `145_Omara_2020` | 12 (family C1) | yes | decoded from March via STAGE B + STAGE D `fix_145`. 12 rows, not the deposited 24 (March carries only the N-matched contrast). |

The `_convert_ai_csv_to_jsonl.py` `EXCLUDE = {016, 063, 145}` set is therefore
**not** carried over: it existed solely to make room for the May splice, and
there is no May splice here.

### 5.3 The other two papers the independent inventory flagged

| paper | flag | verdict |
|---|---|---|
| `234_Malik_2018` | "March JSON has all-null means" | **False alarm.** The means are present under `treatment_mean_biomass` / `control_mean_biomass` / `treatment_mean_grain` / `control_mean_grain`. There is no `treatment_mean` key, which is what a field-name-based check would miss. All 28 rebuilt rows carry means. |
| `101_Liang_Feng_2014` | "carries values absent from the March run" | **Not confirmed.** All 9 March records carry `treatment_mean` and `control_mean`. What differs is *granularity*: March holds 4-season totals (9 rows), the deposited GT holds per-year rows (30). The deposited AI side for this paper is 9 rows — i.e. the March granularity — so the deposited AI keys for 101 appear to be the March decode, unchanged. |

### 5.4 Rows with null means (retained, not dropped)

19 rows carry blank `treatment_mean`/`control_mean` because the source record
reports only `effect_pct`: `007_Gathorne-Hardy_2009` (4), `219_Xie_2021` (6),
`227_Niu_2017` (3), `242_Liu_2014` (6, the sweet-potato arm). The submitted
decoders emitted these same rows blank. They are key rows, counted in the
arithmetic, and are simply unscorable.

---

## 6. Deposited 500 rows vs rebuilt 446 — full attribution

| paper_id | deposited AI rows | March-only rows | delta | why |
|---|---|---|---|---|
| `016_Li_B_2016` | 54 | 6 | **−48** | deposited rows are the 2026-05-31 PDF re-extraction (per-crop × per-season, Table 5); March holds 6 Fig-4c totals |
| `063_Asai_2009` | 12 | 18 | **+6** | deposited rows are the May re-extraction (Exp 1 / HK1 only, 12 rows); March covers 8 site codes (18 rows) |
| `145_Omara_2020` | 24 | 12 | **−12** | deposited rows are the May re-extraction (both N-matched and absolute contrasts); March carries only the N-matched contrast |
| **all other 25 papers** | **410** | **410** | **0** | reproduced exactly, paper by paper |
| **TOTAL** | **500** | **446** | **−54** | |

The 90 rows the prompt identified as post-window (54 + 12 + 24) are replaced
by the 36 rows those three papers yield from March. `500 − 90 + 36 = 446`. ✔
No other paper contributes to the difference — the 25-paper subtotal matches
row-for-row.

---

## 7. Full vocabulary diff vs the GT structural reference

Per spec, this is the **coverage diagnostic**. No value was forced to match.
Counts are rows.

### `paper_id`
AI-only: **none**. GT-only: **none**. 28/28 papers share the token vocabulary.

### `outcome_canonical`
AI-only: none. GT-only: none. Both sides are `crop_yield` on every row.

### `control_token`
AI-only: **none**. GT-only: **none**. After STAGE F both sides use exactly
`{absolute_control, cofactor_matched_control}`. Distribution differs
(AI 238/208 vs GT 449/103) — that is concordance, not vocabulary.

### `aggregation_level`
| side | value | rows |
|---|---|---|
| AI-only | `pooled` | 31 |
| GT-only | — | — |
AI `pooled` occurs on rows the *source record itself* labels a mean
(`166` season="mean" ×6, `219` "mean" ×6, `126` 2-yr pooled ×16, `229`
3-year-mean ×3). GT uses `single_cell` (546) and `documented_pooled` (6) only.
`101_Liang_Feng_2014` is the one paper where both sides say
`documented_pooled` (STAGE D converts the AI 4-season total to the mean).

### `unit_canonical`
| side | value | rows | papers |
|---|---|---|---|
| AI-only | `t/ha/yr` | 3 | 229 (Table 2 annual-total rows) |
| AI-only | `x10^4 kg/ha` | 6 | 242 (sweet-potato arm; means are null) |
| GT-only | — | — | — |

Papers where both sides use in-vocabulary units but **different** ones
(these are the unit-agreement failures the join will flag):

| paper | AI | GT | cause |
|---|---|---|---|
| `063_Asai_2009` | `t/ha` (18) | `kg/ha` (27) | **artefact of freezing GT** — see §8, open item 1 |
| `078_Wang_2012` | `g/pot` (16) | `t/ha` (20) | genuine: AI read the pot-basis figure, GT the field-basis table |
| `082_Jose_2013` | `g/pot` (18) | `t/ha` (18) | genuine: same |

All other 25 papers agree on `unit_canonical`.

### `crop`
| side | value | rows | note |
|---|---|---|---|
| AI-only | `mustard` | 15 | 133_Pandit second crop; GT extracted maize only |
| AI-only | `corn` | 8 | 081_Deenik; GT says `maize`. **Synonym, not casing** — deliberately not remapped (see §8, open item 2) |
| AI-only | `vegetable` | 6 | 016_Li_B; March read the 9-crop total, GT the per-crop rows |
| AI-only | `annual_total` | 3 | 229_Shi Table 2 annual total |
| AI-only | `wheat+maize_combined` | 3 | 101_Liang_Feng rotation total |
| GT-only | `amaranth` | 12 | 016_Li_B per-crop rows |
| GT-only | `bok_choy` | 12 | 016_Li_B |
| GT-only | `water_spinach` | 12 | 016_Li_B |
| GT-only | `coriander` | 6 | 016_Li_B |
| GT-only | `rice_biomass` | 2 | 078_Wang tissue split |
| GT-only | `rice_grain` | 2 | 078_Wang |
| GT-only | `wheat_grain` | 2 | 078_Wang |

Casing/snake_case is consistent on both sides; no casing defect found for this
dataset (unlike Hui). `234_Malik_2018`'s `wheat_biomass` (14 AI rows) is in the
shared vocabulary, so it does not appear above.

### `treatment_level`
| side | value | rows | note |
|---|---|---|---|
| AI-only | `` (blank) | 12 | 077_Zhang_J: dose is % of soil mass, no t/ha in the record |
| AI-only | `1` | 10 | 041_Guerena 1 t/ha-annually arm (4); 082_Jose 1 % w/w (6) |
| AI-only | `0.5` | 6 | 082_Jose % w/w |
| AI-only | `2.25` | 4 | 219_Xie per-season dose |
| AI-only | `6.75` | 4 | 219_Xie per-season dose |
| AI-only | `11.2` | 4 | 219_Xie — **also a 3-s.f. rounding artefact of the submitted `sig3()`**: 11.25 → 11.2 |
| GT-only | `49.5` | 18 | 166_Haefele RH raw husk (non-biochar arm) |
| GT-only | `4.5` / `13.5` / `22.5` | 14 / 14 / 14 | 219_Xie cumulative dose (= 2 × the AI per-season dose) |
| GT-only | `11.25` / `22.50` / `56.25` | 6 / 6 / 6 | 082_Jose in t/ha (AI uses % w/w) |

`234_Malik_2018`'s composite levels (`sludge_2pct`, `straw_4pct`, …) are in
the shared vocabulary on both sides.

### `co_amendment`
| side | value | rows |
|---|---|---|
| AI-only | `compost` | 19 (021) |
| AI-only | `mineral_fertilizer` | 12 (082) |
| AI-only | `phosphorus_dap` | 12 (116) |
| AI-only | `chemical_fertilizer` | 9 (231) |
| AI-only | `chemical_fertilizer_dap` | 2 (126) |
| AI-only | `lime_npk_fertilizer` | 2 (081) |
| GT-only | `mineral_fertilization` | 18 (082 — AI says `mineral_fertilizer`) |
| GT-only | `phosphorus` | 15 (116 — AI says `phosphorus_dap`) |
| GT-only | `straw_nitrogen` | 6 (184 non-biochar straw arm) |
| GT-only | `npk_fertilizer` | 6 (227 — AI says `nitrogen`) |
| GT-only | `compost_*_biochar_{indre,oise,haut_rhin}` | 2 each, 18 total (021: GT folds site **and** biochar feedstock into the co-amendment token) |
| GT-only | `nitrogen_phosphorus` | 1 (063) |

### `co_amendment_level`
| side | value | rows |
|---|---|---|
| AI-only | `240_kgN_ha` | 9 (231) |
| AI-only | `20pct_N_replaced` | 3 (231) |
| AI-only | `present` | 5 (081) |
| AI-only | `conventional` | 2 (016) |
| AI-only | `1.33conventional` | 2 (016) |
| GT-only | `1.5` | 9 (231 — GT puts the *biochar* dose here) |
| GT-only | `313` / `417` | 6 each (016 per-crop N rates) |
| GT-only | `333` / `600` / `800` | 4 each (016) |
| GT-only | `80` | 4 (007) |
| GT-only | `10` / `20` | 4 each (007) |
| GT-only | `240` | 2 (207) |

Note: AI-only `present` (081) and the categorical 016 levels are **the
submitted decoders' own non-numeric encodings**, reproduced faithfully. They
violate the spec's "numeric levels as 3-s.f. decimal strings" rule, but the
records genuinely do not state a number, and inventing one would not be blind.

### `timepoint`
| side | value | rows |
|---|---|---|
| AI-only | `season2011_2012` | 6 (242) |
| AI-only | `irri_2006ds` / `irri_2007ds` / `irri_2008ds` / `irri_2008ws` | 2 each (166 dry-season + 2008ws cells GT lacks) |
| AI-only | `HK2` `HK3` `LO2` `LO3` `LS` `SN` `SO` | 1 each (063 site codes outside GT's Table-3 scope; left uppercase because STAGE D's `fix_063` only rewrites HK1 rows) |
| GT-only | `season1` | 30 |
| GT-only | `season2` | 12 |
| GT-only | `season3` … `season6` | 6 each |
| GT-only | `paddy` / `upland` | 12 / 8 (078) |
| GT-only | `lo1_apo` / `lo1_vieng` | 6 each (063) |
| GT-only | `hk4` | 3 (063) |
| GT-only | `season1_bcA` / `season1_bcB` | 6 each (082) |
| GT-only | `siniloan_2008ws` / `ubon_2008ws` | 4 each (166) |
| GT-only | `exp1_unresolved` | 1 (063) |

Papers whose AI rows sit at `pooled` where GT has a real time coordinate:
007 (4), 016 (6), 078 (16), 081 (14), 082 (18). This is the largest single
source of key non-overlap in the rebuild.

---

## 8. Open items (unresolved, stated plainly)

1. **`063_Asai_2009` unit mismatch is an artefact of freezing GT, not of the
   March source.** `patch_iter0.canon_gt_unit()` converted a GT file from
   `t/ha` to `kg/ha` **only if the AI side for that paper was already
   `kg/ha`**. The deposited AI side for 063 was the *May* re-extraction, which
   was in `kg/ha`; so GT 063 was converted. The March AI side declares `t/ha`.
   With GT frozen as submitted, my 18 AI rows (`t/ha`) cannot agree on
   `unit_canonical` with the 27 GT rows (`kg/ha`), and the join will flag all
   of them `unit_mismatch` even though 10 of them share the full 7-field key.
   I did **not** apply a `×1000` conversion to fix this: it would be an edit
   whose only motivation is to make the two sides agree, which the spec
   forbids. **Recommendation for the orchestrator:** treat 063's unit
   disagreement as a known GT-freezing artefact when attributing biochar
   attrition, not as a March-vs-May reading difference. A one-line
   declared-unit conversion (`t/ha → kg/ha`, ×1000, blind to values) would
   remove it if the orchestrator judges that in scope.
2. **`crop`: `corn` (AI, 081_Deenik, 8 rows) vs `maize` (GT, 6 rows).** A pure
   synonym. The submitted `clean_crop()` does not map it and I did not add a
   mapping, because remapping a crop token onto the GT spelling is forcing a
   value to match. Flagged rather than fixed.
3. **`219_Xie_2021` `11.25 → 11.2`.** The submitted `sig3()` rounds 11.25 to
   3 s.f. as `11.2`, while GT carries `11.25`. Even after the per-season vs
   cumulative dose definition is reconciled, this row would still miss on
   `treatment_level` for a rounding reason. Not fixed (it is the submitted
   rounding function, reproduced verbatim); flagged.
4. **The four hand-authored papers (229, 231, 234, 242) are a reconstruction,
   not a reproduction.** No submitted script exists for them. Their STAGE C
   encodings were derived from each March record's own explicit fields, and
   cross-checked for *convention* against the means-stripped
   `biochar_ai_structural.csv` (structural columns only). Two deliberate
   departures from the deposited encoding, both in non-key columns:
   `source_locator` carries the raw `data_source` string rather than a
   tidied label (e.g. `Fig. 1a (approximate)` not `Fig 1a`), and 234's
   `row_id` uses `__ai__<i>_biomass` / `_grain`. Neither affects pairing.
5. **`231_Zhang_2021` keys a different factor than GT.** The AI side puts the
   fertiliser regime in `co_amendment` / `co_amendment_level`
   (`chemical_fertilizer` / `240_kgN_ha`); GT puts the *biochar dose* in
   `co_amendment_level` (`1.5`) with `co_amendment=none`. This is a design
   disagreement, not a defect on either side, and it drives all 12 rows to
   zero key overlap. Not reconcilable without consulting GT.
6. **`aggregation_level = pooled` (31 AI rows) has no GT counterpart.** GT
   only ever uses `single_cell` or `documented_pooled`. The 31 AI rows are
   source-labelled means. Whether they should have been emitted as
   `documented_pooled` (as `101` was, via STAGE D) is a protocol question the
   submission left inconsistent; I reproduced the submitted behaviour.

---

## 9. Compliance with the hard rules

| rule | status |
|---|---|
| 1 outcome-blind | Held. No key field reads `treatment_mean`, `control_mean`, `effect_pct`, or any GT value. `effect_pct` appears only inside 219's `evidence` string, which the submitted decoder also did and which is not a key. |
| 2 no value matching | Held. The deposited AI key CSVs (`runs/biochar*/keys/ai`, `runs/biochar_v2/jsonl/ai`) were never opened. Only `00_SPEC/vocab_reference/biochar_ai_structural.csv` (means stripped) was read, and only for structural columns. |
| 3 start from the submitted decoder | Held. Stages A/B/D are the submitted logic; changes are itemised in §3.1. Stage C is new because no submitted script existed. |
| 4 deterministic | Held. stdlib only, no randomness, no LLM at runtime; two runs byte-identical (SHA-256 above). |
| 5 no silent drops | Held. `432 = 446 − 14 expansions + 0 exclusions`; 6 non-paper files excluded with reasons. |
| 6 don't touch protected paths | Held. Writes only to `02_DECODERS/biochar/`, `03_KEYS/ai_rebuilt/biochar/`, `06_LEDGER/`. `04_ANALYSIS/_AS_SUBMITTED/` and `03_KEYS/gt/` untouched (GT read-only, verified identical to the reference). |
| no equivalence/TOST/fidelity analysis | Held. Only coverage and vocabulary diagnostics; no mean was compared against anything. |
