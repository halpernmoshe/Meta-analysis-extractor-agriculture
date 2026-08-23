# Loladze — AI-side decoder ledger (independent rebuild, 19 Aug 2026)

Dataset: `loladze`. Side rebuilt: **AI only**. GT side untouched.
Exactly one substantive variable changes vs the submitted analysis: the AI-side
**source** is now the frozen March-2026 **single-model Claude** agent JSONs instead
of the multi-model consensus folder.

---

## 1. Provenance

| Item | Value |
|---|---|
| Submitted decoder started from | `meta_analysis_extractor/SUBMISSION_Environmental_Evidence/resubmission/matching/runs/loladze_v2/keys/ai/_decode_ai_loladze_v3.py` |
| ...its SHA-256 | `86ee14b46014d9d9e29ff9e5d7a7af7e284994db4aa4f42d0db115fa878f09ca` |
| Submitted decoder's `SRC` (replaced) | `meta_analysis_extractor/output/loladze_v3_combined/*_consensus.json` (multi-model consensus) |
| Rebuild decoder | `02_DECODERS/loladze/decode_loladze.py` |
| ...its SHA-256 | `a71a29f461ba16bc0b536c2106a02b5b3a9b3ffb52c6d955d0efb94fd38d415b` |
| New source | `01_INPUTS_FROZEN/loladze/*_agent.json` — 46 files, 1646 records |
| Source aggregate SHA-256 (name+bytes, sorted) | `a4853ef024c132963ac3ca78c6a60491906d9bdf2ea7e0d7b4bcab9a6a49cbff` |
| Output | `03_KEYS/ai_rebuilt/loladze/*.csv` — 46 files, 1646 rows |
| Output aggregate SHA-256 (name+bytes, sorted) | `af06e4f9529a3af4c9e51252ca9626371b59b754305f5133a2cd5b13421c4094` |
| Determinism check | decoder executed twice; aggregate SHA-256 identical both runs (byte-identical output confirmed) |
| Runtime deps | Python stdlib only (`csv, glob, json, os, re`). No network, no LLM, no randomness. |

**Source-schema confirmation.** The record list in these files sits under the legacy
key `consensus_observations`. This is inherited naming only: all 46 files were
checked and **none** contains `claude_obs`, `kimi_obs`, `gemini_obs`, `tiebreaker`,
or `disagreements`. Observation-field census across 1646 records: `element`,
`tissue`, `treatment_mean`, `control_mean`, `effect_pct`, `unit`, `data_source`,
`treatment_description`, `control_description`, `moderators` on 1646/1646;
`n` 1576; `species` 193; `treatment_se`/`control_se` 125; `site` 70;
`se_treatment`/`se_control` 26; `notes` 22.

**Outcome-blindness attestations.**
- No key column is derived from `treatment_mean`, `control_mean`, or `effect_pct`.
  `ratio_effect()` writes the row's own recorded ratio into the value column
  `treatment_mean`; it is never consulted to build, choose, or drop a coordinate.
- `022_Blank_2011_agent.json` carries a top-level `ground_truth_comparison` block
  containing Loladze GT effect sizes for that paper. **The decoder never reads that
  key.** Recorded here so the omission is auditable rather than merely asserted.
- No GT key table and no deposited AI key table was joined against. `03_KEYS/gt/`
  was read only to enumerate `paper_id` tokens; `00_SPEC/vocab_reference/
  loladze_gt_structural.csv` (no outcome values) was read only for the §6 vocabulary
  comparison, after the keys were already written.

---

## 2. Every change vs the submitted decoder, with reason

| # | Change | Class | Reason |
|---|---|---|---|
| **1** | `SRC` repointed to `01_INPUTS_FROZEN/loladze`; glob `*_consensus.json` → `*_agent.json` | (a) input path | The mandated variable change. Filename convention differs in the frozen run. |
| **2a** | New `build_recon(d)` schema adapter | (b) schema adapter | The submitted paper-level CO2 fallback read `recon["treatment_definition"]`, `recon["control_definition"]`, and `recon["raw_response"]` (regex'd for a `"co2_levels"` object). **The frozen files have no `recon` block at all**; the same paper-level information sits in the top-level keys `co2_elevated`, `co2_ambient`, `co2_levels`. The adapter re-points the *identical* fallback at them and adds no new inference. Multi-level `co2_levels` dicts (`elevated_1`/`elevated_2`/…) are deliberately **not** collapsed to a single elevated value, because choosing among them would be a guess. Effect: 92 rows that would otherwise be `co2_unresolved` acquire a CO2 token (all of `043_Natali_2009`, 70 rows, plus 22 `034_Johnson_1997` rows that route to `pooled`). |
| **2b** | New `moderator_view(o)` schema adapter; promotes observation-level `species` and `site` into the moderator dict, `moderators` winning any collision | (b) schema adapter | `species_base()` and `level_suffixes()` read `species` and `site` **out of `moderators`**. In the frozen schema `043_Natali_2009` (70 rows) puts exactly those two fields on the *observation* instead. Without the adapter all 70 Natali rows lose their species token and their GT-aligned `duke`/`ornl` suffix. Only these two named keys are promoted — no other observation-level field — so pooling detection and every other suffix rule see the same key-set the submitted decoder saw. `034_Johnson_1997` has both locations populated (123 rows); `moderators` wins, so its output is unchanged. |
| **2c** | `recon["is_fig_only"] = False` (constant) | (b) schema adapter, lossy | The submitted decoder OR-ed a paper-level `recon.is_fig_only` flag into `is_figure`. **The frozen schema has no counterpart field.** Rather than invent one (e.g. by string-matching `notes` for "Figure"), it is set False and row-level detection via `data_source` is left exactly as submitted. See open item OI-3. |
| **3** | `"se"` added to `ELEMENT_SYMBOLS`; `"selenium": "se"` added to `ELEMENT_WORD` | (c) genuine parse defect | The submitted closed mineral list **omitted selenium**, so source rows whose `element` field literally reads `"Se"` parsed to an empty `treatment_level` and could never pair. Selenium is a mineral nutrient and **occurs in the GT structural vocabulary** (1 GT row). This is a closed-list parse defect fixed by correctly parsing the source string; no value was consulted. Effect: **3 rows** (`014_Lieffering_2004` ×2, `010_Li_2010` ×1). Verified no collateral re-mapping: re-running with the two `# [REBUILD-CHANGE 3]` lines removed changes the element histogram in exactly two cells — blank 32→35, `se` 3→0 — and nothing else. **Deliberately NOT added:** `"h"` (hydrogen, 4 rows — `006_Azam_2013` ×2, `048_Khan_2013` ×2), because hydrogen is not a mineral nutrient and is absent from the GT vocabulary. |
| **4** | Emits canonical-schema CSV directly (18 columns, `csv.DictWriter`) instead of JSONL | (d) output plumbing | The submitted pipeline was decoder → `jsonl/ai/*.jsonl` → `matching/keys_from_jsonl.py` → `keys/ai/*.csv`. The rebuild spec's deliverable is the CSV. Column order, `DictWriter` quoting, `newline=""` (CRLF), and `encoding="utf-8"` are identical to `keys_from_jsonl.py`, so the bytes are what that two-step pipeline would have produced. |
| **4b** | `DECODER` constant `"ai-decode-v3"` → `"rebuild_2026-08-19/loladze"` | (d) spec requirement | `DECODER_SPEC.md` §Output contract mandates `decoder = rebuild_2026-08-19/<dataset>`. |
| **5** | Added an unpairability/diagnostic tally printed at the end of the run | (d) reporting only | Required by spec rule 5 (no silent drops). Does not touch any emitted field. |

**Nothing else changed.** `element_symbol`, `tissue_token`, `slug`, `SPECIES_NORM`,
`species_base`, `level_suffixes`, `time_suffixes`, `PPM_RE`, `_ppm_from_text`,
`_only_int`, `co2_contrast`, `is_pooled`, `ratio_effect`, `fmt_num`,
`NON_MINERAL_HINTS`, the row-assembly block, the evidence string, and the
`co2_unresolved` / `pooled` branching are the submitted v3 code verbatim.

**Casing / normalization check (spec §Normalization).** No casing normalization was
needed on this dataset: the submitted decoder already lowercases every key column
it emits (`element_symbol` → lowercase symbol; `tissue_token` → lowercased token;
`slug()` → lowercase snake_case; `outcome_canonical`, `crop`, `unit_canonical`,
`control_token` are lowercase constants). Verified on the output: **0 rows** contain
an uppercase character in any of the seven non-`paper_id` key columns
(`outcome_canonical`, `crop`, `treatment_level`, `co_amendment`,
`co_amendment_level`, `timepoint`, `aggregation_level`), nor in `unit_canonical` /
`control_token`. `paper_id` deliberately retains mixed case because that is the GT
token vocabulary itself; verified that all 46 emitted `paper_id` values are exact
`03_KEYS/gt/loladze/*.csv` filename stems.

---

## 3. Field mapping (raw JSON → key column)

| Key column | Source | Transform |
|---|---|---|
| `row_id` | `paper_id` + observation index | `"%s__ai__%d"` (index = position in `consensus_observations`) |
| `side` | — | constant `ai` |
| `paper_id` | top-level `paper_id` | verbatim; token vocabulary already identical to the GT side (46/46 tokens present in `03_KEYS/gt/loladze/`) |
| `outcome_canonical` | — | constant `mineral_concentration` |
| `crop` | — | constant `na` |
| `treatment_level` | `obs.element` | `element_symbol()` → lowercase element symbol (closed mineral list) |
| `co_amendment` | `obs.tissue` | `tissue_token()` → `grain` / `foliar` / `above_ground` / `edible` / else a slug (documented attrition) |
| `co_amendment_level` | `obs.moderators.{cultivar\|clone\|ecotype\|species}` (precedence in that order; `species` also from observation level via adapter 2b) **+** `__`-joined factor suffixes from `obs.moderators` via `level_suffixes()` (N level, P level, leaf/needle age, `inner_nm`, site `duke`/`ornl`, soil `basalt`/`rhyolite`, `kplus`/`kminus`) | `slug()` + `SPECIES_NORM` common-name normalization |
| `timepoint` | `obs.treatment_description` / `obs.control_description` → `obs.moderators.CO2_*` → adapter-2a paper-level `co2_elevated`/`co2_ambient`/`co2_levels` | `co2_contrast()` → `eco2_<e>_amb_<a>`, plus `__`-joined time suffixes from `time_suffixes()` (`y<YYYY>`, `y<a>_<b>`, season, `h<YYYY>`, `doy<N>`); `pooled` when `is_pooled()`; `co2_unresolved` when the ppm pair is undeterminable |
| `aggregation_level` | `is_pooled()` over `obs.moderators` values + species-string pooling flag | `pooled` / `single_cell` |
| `unit_canonical` | — | constant `ratio` |
| `control_token` | — | constant `ambient_co2` (not part of the match key) |
| `treatment_mean` | `obs.effect_pct`, else `(obs.treatment_mean − obs.control_mean)/obs.control_mean` | `ratio_effect()` → ratio, 6 dp, `fmt_num()`. **Value column only.** |
| `control_mean` | — | empty (GT side likewise carries the contrast as a single ratio) |
| `source_locator` | `obs.data_source` | verbatim |
| `is_figure` | `obs.data_source` contains `fig` (case-insensitive) | 0/1 |
| `evidence` | assembled audit string | element/tissue/species/CO2/suffixes/descriptions/unit/locator + notes |
| `decoder` | — | constant `rebuild_2026-08-19/loladze` |

---

## 4. Record arithmetic

```
files in                    46
records in                1646   (sum of len(consensus_observations))
rows out                  1646
hard drops                   0
  => records_in (1646) = rows_out (1646) + hard_drops (0)
```

**No record was dropped.** Every one of the 1646 source records became exactly one
key row. Rows that cannot pair for structural reasons are still emitted (with the
reason written into `evidence`) so the join can classify them, per spec rule 5.

### Unpairability tally by reason (rows emitted, not dropped)

Reasons are not mutually exclusive — a row can carry more than one blocker.

| Reason | Rows | Notes |
|---|---|---|
| `treatment_level` empty — non-mineral variable | 26 | `lignin` 10, `TNC` 10 (`004_Finzi_2001`); `protein` 4 (`002_Ziska_1997`), 2 (`049_Singh_2013`) |
| `treatment_level` empty — not in closed mineral list | 6 | `H` 2 (`006_Azam_2013`), 2 (`048_Khan_2013`); `oil` 2 (`049_Singh_2013`) |
| `co_amendment` outside GT closed list | 100 | `total_plant` 40 (`047_Rodenkirchen_2009`), `vegetation_content` 24 + `litter` 22 (`034_Johnson_1997`), `total_yield` 14 (`036_Schenk_1997`) |
| `timepoint` = `co2_unresolved*` | 238 | `034_Johnson_1997` 101, `017_Fangmeier_2002` 62, `038_Newbery_1995` 40, `018_Al-Rawahy_2013` 21, `049_Singh_2013` 14 |
| `co_amendment_level` blank (diagnostic) | 622 | see §5; GT also has 90 blank-`co_amendment_level` rows, so blank is not automatically fatal |
| `is_figure = 1` (separate scoring tier, not an exclusion) | 412 | GT has 0 figure-tier rows for this dataset |
| `treatment_mean` empty (value column, not a key) | 1 | effect undeterminable from `effect_pct` or own means |
| **Rows with no structural blocker** | **861** | element parsed, tissue in GT vocab, `co_amendment_level` non-blank, CO2 pair resolved |
| **…of which table-tier (`is_figure=0`)** | **676** | |

### Papers covered

| Quantity | N |
|---|---|
| GT papers (`03_KEYS/gt/loladze/*.csv`) | 50 |
| Frozen single-model source files | 46 |
| Papers in the rebuilt AI keys | 46 |
| Shared with GT | **46 / 50 (92%)** |
| AI-side papers absent from GT | 0 |

GT papers with **no** frozen AI source — confirmed, and **nothing was substituted**
for any of them:

- `024_Nowak_2002`
- `029_Kuehny_1991`
- `030_Wroblewitz_2013`
- `033_Johnson_2003`

Exactly the four named in the brief. Every `paper_id` token in the rebuilt keys is
byte-identical to a GT filename stem, so pairing is not blocked by tokenization.

**Coverage differences vs the deposited (consensus-derived) AI keys** — reported
because they change what the orchestrator can compare, not to argue a result:

| Paper | Deposited AI keys | Rebuilt AI keys |
|---|---|---|
| `039_Heagle_1993` | file present but **empty** (0 data rows) | **132 rows** |
| `010_Li_2010` | file present but **empty** (0 data rows) | **21 rows** |
| `024_Nowak_2002`, `029_Kuehny_1991`, `030_Wroblewitz_2013`, `033_Johnson_2003` | present, with rows | **absent** (no frozen source) |

Deposited AI keys: 50 files / 1834 rows / **48** papers with rows (`010_Li_2010` and
`039_Heagle_1993` were empty in the submitted run, per `runs/loladze_v2/FLOW.md` §1).
Rebuilt AI keys: 46 files / 1646 rows / **46** papers with rows — no empty files.
Net: the rebuild loses 4 papers it has no source for, and gains real rows on 2 papers
the submitted run could not extract at all.

---

## 5. The empty-`treatment_level` (missing element) resolution

**Question posed:** the deposited consensus-derived AI keys had 173 of 1834 rows with
an empty `treatment_level` and 568 with an empty `co_amendment_level`. Does the frozen
single-model source do the same, and why?

### Empty `treatment_level` — resolved

| Run | Empty `treatment_level` | Share |
|---|---|---|
| Deposited AI keys (consensus source, as submitted) | **173 / 1834** | 9.4% |
| Rebuilt, frozen source, submitted element list unchanged | **35 / 1646** | 2.1% |
| Rebuilt, frozen source, after change #3 (selenium) | **32 / 1646** | **1.9%** |

**Cause.** It is not a missing field: `element` is present on **1646/1646** frozen
records. Every one of the 35 pre-fix empties is an `element` **string that the closed
mineral list legitimately or defectively rejects**:

| `element` string | Rows | Paper(s) | Verdict |
|---|---|---|---|
| `lignin` | 10 | `004_Finzi_2001` | correct rejection — not a mineral |
| `TNC` | 10 | `004_Finzi_2001` | correct rejection — total non-structural carbohydrate |
| `protein` | 6 | `002_Ziska_1997` (4), `049_Singh_2013` (2) | correct rejection — not a mineral |
| `oil` | 2 | `049_Singh_2013` | correct rejection — not a mineral |
| `H` | 4 | `006_Azam_2013` (2), `048_Khan_2013` (2) | correct rejection — hydrogen, not a mineral nutrient; absent from GT vocabulary |
| `Se` | 3 | `014_Lieffering_2004` (2), `010_Li_2010` (1) | **parse defect** — selenium missing from the submitted closed list; **fixed** (change #3) |

So the 9.4% → 1.9% drop is a property of the **source**, not of the decoder: the
frozen single-model extraction records a clean element symbol on essentially every
row, whereas the multi-model consensus merge evidently produced far more element
strings the same parser could not resolve. The only decoder-side contribution is the
3-row selenium fix. Nothing was inferred from any value: the fix is the addition of
one chemical symbol and one element name to a closed vocabulary.

The residual 32 rows (1.9%) are all **genuinely non-mineral outcome variables**
(lignin, TNC, protein, oil) or a non-mineral element (H). They are correctly outside
`mineral_concentration` scope, are emitted with the reason in `evidence`, and would
have no GT counterpart even if forced through — GT contains no `lignin`, `tnc`,
`protein`, `oil`, or `h` rows.

### Empty `co_amendment_level` — characterized, not "fixed"

| Run | Empty `co_amendment_level` |
|---|---|
| Deposited AI keys | 568 / 1834 (31.0%) |
| Rebuilt | **622 / 1646 (37.8%)** |
| GT side (for reference) | 90 / 754 (11.9%) |

**Cause.** `species_base()` reads `cultivar` → `clone` → `ecotype` → `species` **from
the row's `moderators`**. For 622 rows none of those four keys is present in
`moderators`. Concentrated in 17 papers:

`047_Rodenkirchen_2009` 120, `040_Pfirrmann_1996` 88, `026_Seneweera_1997` 78,
`017_Fangmeier_2002` 62, `041_Mjwara_1996` 62, `013_Keutgen_2001` 40,
`038_Newbery_1995` 40, `048_Khan_2013` 30, `018_Al-Rawahy_2013` 21,
`028_Mishra_2011` 21, `031_Pal_2003` 15, `049_Singh_2013` 14, `058_ONeill_1987` 14,
`046_Porter_1984` 10, `051_Niu_2013` 4, `037_de_2000` 2, `022_Blank_2011` 1.

All 17 of those papers **do** carry a paper-level `species` string (and
`026_Seneweera_1997` a paper-level `cultivar` = `Jarrah`), so a paper-level fallback
would fill all 622. **That fallback was NOT added**, for two reasons:

1. It is a *logic* change, not a schema adaptation. The field the submitted decoder
   reads (`moderators.species`) exists in the frozen schema; it is simply unpopulated
   on these rows. Adding a paper-level fallback would change decode behaviour beyond
   the one variable this rebuild is allowed to move. (Contrast adapter 2b, where the
   same semantic field genuinely *moved* to the observation level.)
2. It would not reliably close the gap anyway. `040_Pfirrmann_1996` paper-level
   species is `Picea abies (Norway spruce)` → `picea_abies`, whereas the GT token for
   that paper is `karst`; `013_Keutgen_2001` → `citrus_madurensis` has no GT
   counterpart token. Some papers would gain a matching token (`041_Mjwara_1996` →
   `phaseolus_vulgaris_l_cv_contender` vs GT `contender` — still no match;
   `018_Al-Rawahy_2013` → GT `victor`, also no match). Adding it risks looking like
   vocabulary tuning while delivering little.

Logged as open item **OI-1** for the orchestrator to decide.

---

## 6. Vocabulary comparison vs the GT structural reference

Source: `00_SPEC/vocab_reference/loladze_gt_structural.csv` (754 rows, no outcome
values) vs the 1646 rebuilt AI rows. **No value was forced to match.**

### Constant columns — exact agreement

| Column | AI | GT | AI-only | GT-only |
|---|---|---|---|---|
| `outcome_canonical` | `mineral_concentration` 1646 | `mineral_concentration` 754 | — | — |
| `crop` | `na` 1646 | `na` 754 | — | — |
| `unit_canonical` | `ratio` 1646 | `ratio` 754 | — | — |
| `control_token` | `ambient_co2` 1646 | `ambient_co2` 754 | — | — |
| `aggregation_level` | `single_cell` 1554, `pooled` 92 | `single_cell` 677, `pooled` 77 | — | — |

### `treatment_level` (element) — **vocabulary now identical**

25 distinct symbols on each side; **0 AI-only, 0 GT-only**. (Before change #3 the AI
side lacked `se`, which GT has; the deposited AI keys additionally carried 173 blank
rows.)

| element | AI rows | GT rows |
|---|---|---|
| `p` | 207 | 79 |
| `n` | 190 | 78 |
| `ca` | 162 | 72 |
| `zn` | 137 | 69 |
| `k` | 151 | 65 |
| `fe` | 116 | 62 |
| `mg` | 136 | 58 |
| `mn` | 108 | 55 |
| `cu` | 124 | 52 |
| `s` | 73 | 30 |
| `c` | 50 | 19 |
| `na` | 15 | 19 |
| `b` | 40 | 17 |
| `cd` | 39 | 13 |
| `pb` | 10 | 12 |
| `ni` | 10 | 12 |
| `mo` | 12 | 11 |
| `al` | 9 | 7 |
| `co` | 9 | 7 |
| `cr` | 3 | 6 |
| `v` | 7 | 6 |
| `si` | 1 | 2 |
| `ba` | 1 | 1 |
| `sr` | 1 | 1 |
| `se` | 3 | 1 |
| *(blank)* | 32 | 0 |

Answer to the brief's question 2: **there is no element GT has that the AI side
lacks, and none the AI side has that GT lacks.** The remaining asymmetry is
row-count density (the AI extractor emits ~2.2× more rows per element), not
vocabulary.

### `co_amendment` (tissue)

| token | AI rows | GT rows |
|---|---|---|
| `foliar` | 859 | 392 |
| `above_ground` | 363 | 129 |
| `edible` | 164 | 82 |
| `grain` | 160 | 151 |
| `litter` | 22 | **0** |
| `total_plant` | 40 | **0** |
| `vegetation_content` | 24 | **0** |
| `total_yield` | 14 | **0** |

AI-only: `litter`, `total_plant`, `vegetation_content`, `total_yield` (100 rows) —
tissues the AI extractor recorded that lie outside the GT four-token closed list.
Left as-is (documented attrition), exactly as the submitted decoder did.
GT-only: none.

### `co_amendment_level` (species/cultivar + factor suffixes)

| | AI | GT |
|---|---|---|
| distinct tokens | 62 | 82 |
| shared tokens | **33** | 33 |
| rows on a shared token | 1016 | 393 |
| blank | 622 | 90 |

Largest AI-only tokens: `nc_s` 66, `nc_r` 66, `q_geminata` 45, `q_myrtifolia` 45,
`wuxiangjing_14` 36, `yangmai_14` 36, `pooled` 28, `sporobolus_kentrophyllus` 26,
`pinus_sylvestris` 26, `combined` 22, `maravilla_de_verano` 20, `calluna_vulgaris`
20, `pinus_taeda__duke` 20, `batavia_rubia_munguia` 19,
`batavia_rubia_munguia__inner_nm` 19, `pinus_sylvestris__1yr_old` 19.

Largest GT-only tokens: `bintje__o3` 16, `bintje__nf` 16, `astra` 14, `eureka` 14,
`laws__unfertilized` 11, `regal__ncr` 11, `regal__ncs` 11, `karst__kminus__needles`
11, `karst__kplus__needles` 11, `karst` 11, `brm__inner_nm` 10, `mv__inner_nm` 10,
`theresa__n100` 10, `theresa__n50` 10, `batis__n100` 10, `batis__n50` 10,
`flindersia_brayleyana` 10, `pinus_taeda__1yr_old` 10.

Three distinct causes, all left uncorrected on principle:
1. **Abbreviation convention.** GT abbreviates some cultivars (`brm` / `mv`,
   `nc_s` → `ncs`, `nc_r` → `ncr`, `q_geminata` → GT spells `quercus_geminata`
   elsewhere); the AI slugs the full string it read from the paper.
2. **Factor routing.** GT routes an O3 / fertilizer factor into
   `co_amendment_level` (`bintje__o3`, `bintje__nf`, `laws__unfertilized`) where the
   AI moderator set exposes it under a key name `level_suffixes()` does not recognize.
3. **Granularity.** `pinus_taeda__duke` (AI, 20 rows, 0-yr + 1-yr needles pooled
   under the site suffix) vs `pinus_taeda__1yr_old` (GT, 10 rows) — a real
   needle-age granularity difference in `043_Natali_2009`.

### `timepoint` (CO2 pair token + time suffixes)

| | AI | GT |
|---|---|---|
| distinct tokens | 56 | 56 |
| shared tokens | **27** | 27 |
| rows on a shared token | 847 | 430 |
| `co2_unresolved*` | 238 | 0 |
| `pooled` | 92 | 77 |

Largest AI-only tokens: `co2_unresolved` 224, `eco2_700_amb_400` 120,
`eco2_750_amb_350` 112, `eco2_693_amb_362` 45, `eco2_490_amb_380` 44,
`eco2_600_amb_380` 44, `eco2_560_amb_360__y1997` 33, `eco2_550_amb_370` 30.

Largest GT-only tokens: `eco2_730_amb_380` 30, `eco2_725_amb_375` 22,
`eco2_685_amb_385` 22, `eco2_620_amb_300` 20, `eco2_582_amb_382` 20,
`eco2_537_amb_387__y2004_2006` 16, `eco2_680_amb_375` 16, `eco2_680_amb_360` 16,
`eco2_550_amb_370__doy247` 13.

This is the same dominant attrition class the submitted run documented: the two
sides frequently **record different CO2 ppm values** for the same experiment
(`043_Natali_2009`: AI reads one paper-wide `570/370` from the text, GT carries
per-site `582/382`, `549/393`, `730/380`). Reconciling them would require choosing a
coordinate to make values agree, which the protocol prohibits. Left as honest
attrition.

### `is_figure`

| | AI | GT |
|---|---|---|
| 0 (table tier) | 1234 | 754 |
| 1 (figure tier) | 412 | 0 |

The 412 figure-tier AI rows come from `data_source` strings that name a figure.
Per paper: `047_Rodenkirchen_2009` 120, `025_Guo_2011` 72, `041_Mjwara_1996` 62,
`014_Lieffering_2004` 24, `028_Mishra_2011` 21, `007_Woodin_1992` 20,
`032_Kanowski_2001` 20, `010_Li_2010` 18, `026_Seneweera_1997` 18, `027_Peet_1986` 15,
`016_Fernando_2012a` 10, `008_Campbell_2002` 6, `051_Niu_2013` 4, `037_de_2000` 2.
Deposited AI keys had only 166 figure rows; the frozen single-model source records
figure provenance on far more rows. Since GT has no figure tier, these rows cannot
enter the headline table-tier statistic either way — but the count is materially
different from the submission and the orchestrator should know it.

---

## 7. Output manifest (46 files, 1646 rows)

`sha256` truncated to 16 hex chars.

| file | rows | sha256 |
|---|---|---|
| `001_Ma_2007.csv` | 16 | `529d2bcf06aaf523` |
| `002_Ziska_1997.csv` | 8 | `03f708dcf65c917f` |
| `003_Baslam_2012.csv` | 76 | `e8af89eadf0f48d8` |
| `004_Finzi_2001.csv` | 70 | `98cb69f6a7d5365c` |
| `005_Niinemets_1999.csv` | 10 | `f216c540d73a3cb1` |
| `006_Azam_2013.csv` | 29 | `43a504172a54d425` |
| `007_Woodin_1992.csv` | 20 | `86d7669051a299f2` |
| `008_Campbell_2002.csv` | 6 | `44e8ae03df958bf2` |
| `009_Barnes_1992.csv` | 24 | `704cd299ec8f3098` |
| `010_Li_2010.csv` | 21 | `65e1bd9821912b24` |
| `011_Huluka_1994.csv` | 10 | `8cda9f874770f875` |
| `012_Wu_2004.csv` | 4 | `9b10bfdc4e6d862b` |
| `013_Keutgen_2001.csv` | 60 | `5843d7a1ea9010e9` |
| `014_Lieffering_2004.csv` | 24 | `d0cab0c98d22a730` |
| `015_Pleijel_2009.csv` | 8 | `2fa5b057a8db4214` |
| `016_Fernando_2012a.csv` | 11 | `ee5ef2b0852b7afd` |
| `017_Fangmeier_2002.csv` | 62 | `a441345a6d08bd66` |
| `018_Al-Rawahy_2013.csv` | 21 | `7221a500a6bfe3fb` |
| `019_Baxter_1994.csv` | 15 | `3029eb522164b3b5` |
| `020_Overdieck_1993.csv` | 28 | `7f7201a784846081` |
| `021_Wilsey_1994.csv` | 26 | `1c284cf67db6d256` |
| `022_Blank_2011.csv` | 29 | `3bfff6b748e0e57d` |
| `025_Guo_2011.csv` | 72 | `316ad371a60a308a` |
| `026_Seneweera_1997.csv` | 78 | `7b226cb867b847dd` |
| `027_Peet_1986.csv` | 15 | `07435f30d7f0430d` |
| `028_Mishra_2011.csv` | 21 | `3cf9f23d04f06d58` |
| `031_Pal_2003.csv` | 15 | `f649d55649c51bd9` |
| `032_Kanowski_2001.csv` | 20 | `f11fe8c054486097` |
| `034_Johnson_1997.csv` | 123 | `fa704c750552fc84` |
| `035_Oksanen_2005.csv` | 11 | `a42482588944a9d5` |
| `036_Schenk_1997.csv` | 42 | `fbccd8e546dc4563` |
| `037_de_2000.csv` | 2 | `15a5e7e26317b1f1` |
| `038_Newbery_1995.csv` | 40 | `d23087434c93bb6f` |
| `039_Heagle_1993.csv` | 132 | `dc14c0f236ee72e0` |
| `040_Pfirrmann_1996.csv` | 88 | `f2a0008a3ffed9b5` |
| `041_Mjwara_1996.csv` | 62 | `9a82e0f34bf1a03e` |
| `042_Luomala_2005.csv` | 45 | `cdfa9191297811d2` |
| `043_Natali_2009.csv` | 70 | `8361b0411b54b187` |
| `044_Housman_2012.csv` | 30 | `4e5e1c56612db11c` |
| `046_Porter_1984.csv` | 10 | `7eafa171465745a2` |
| `047_Rodenkirchen_2009.csv` | 120 | `0e66c58ed0952baf` |
| `048_Khan_2013.csv` | 30 | `8d85d07c1cd6e77a` |
| `049_Singh_2013.csv` | 14 | `5c9e4d289b79af30` |
| `050_Polley_2011.csv` | 10 | `71d78f478f3dd0ff` |
| `051_Niu_2013.csv` | 4 | `5430e6306db92b4b` |
| `058_ONeill_1987.csv` | 14 | `553ceec4b226ff04` |

---

## 8. Open items (unresolved, stated plainly)

**OI-1 — 622 blank `co_amendment_level` rows (37.8% of the AI side).**
All 17 affected papers carry a usable paper-level `species` (one also a paper-level
`cultivar`), so a paper-level fallback would fill every blank. I did **not** add it:
it is a decode-logic change rather than a schema adaptation, and spot-checking shows
it would still fail to produce the GT token for several of the biggest contributors
(`040_Pfirrmann_1996` → `picea_abies` vs GT `karst`; `041_Mjwara_1996` →
`phaseolus_vulgaris_l_cv_contender` vs GT `contender`). **Orchestrator decision
needed**: leave as-is (current state), or authorize a paper-level species/cultivar
fallback as a declared second variable and re-run.

**OI-2 — 238 `co2_unresolved` rows in 5 papers.** Each is genuinely
under-specified in the frozen source, and I declined every available guess:
- `034_Johnson_1997` (101): `co2_levels = {ambient: "ambient", elevated: "ambient + 350 uL/L"}` — the ambient ppm is never stated numerically.
- `017_Fangmeier_2002` (62): row descriptions give elevated `680 ul/l` but the control is `"Ambient CO2 (NF)"`. The paper-level `notes` string does contain "Ambient CO2 ~370-404 ul/l", but the submitted decoder never reads `notes` and mining free text for a ppm range would be new inference.
- `038_Newbery_1995` (40): `co2_elevated = "ambient + 250 ppm"` — a **delta**, not a level; `co2_ambient = "ambient air"`. Adapter 2a correctly leaves this unresolved rather than emitting `eco2_250`.
- `018_Al-Rawahy_2013` (21) and `049_Singh_2013` (14): elevated ppm present, control described only qualitatively (`"charcoal-filtered air"`, `"nonfiltered air, ambient CO2"`).

**OI-3 — no paper-level figure flag in the frozen schema.** The submitted decoder
OR-ed `recon.is_fig_only` into `is_figure`; the frozen files have no equivalent, so
figure-tier assignment rests solely on row-level `data_source`. The rebuilt side has
412 figure rows vs 166 in the deposited keys. I could not determine whether any part
of that 412 vs 166 gap is attributable to the missing paper-level flag versus simply
richer `data_source` provenance in the single-model run, because establishing that
would require reading the consensus source this rebuild is meant to replace.

**OI-4 — selenium fix is a judgement call.** Change #3 is the one edit that is not
strictly "a bug that blocks execution". It affects 3 rows and it is what makes the AI
and GT element vocabularies exactly coincide (25 = 25). If the orchestrator prefers a
zero-tolerance reading of spec rule 3, delete the two `# [REBUILD-CHANGE 3]` lines
and re-run: `treatment_level` blanks go 32 → 35 and the AI side loses `se`
(GT retains 1 `se` row). Flagged rather than buried.

**OI-5 — four GT papers have no AI counterpart** (`024_Nowak_2002`,
`029_Kuehny_1991`, `030_Wroblewitz_2013`, `033_Johnson_2003`). Confirmed absent from
the frozen single-model run; nothing was substituted. Their GT rows will necessarily
be NO_MATCH on coverage grounds. Not resolvable at the decoder layer.

**No equivalence, TOST, or fidelity analysis was run.** This ledger stops at verified
key tables, per the brief.
