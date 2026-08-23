# Li J 2022 (plant biostimulants) — AI-side decoder ledger, rebuild 2026-08-19

## 1. Source

`01_INPUTS_FROZEN/li_j/*_agent.json` — **49 files, 1053 records**, copied from
`meta_analysis_extractor/output/li2022_agent_extraction` (March 2026, single-model
Claude agent run). Records live under the legacy key `consensus_observations`; no
`claude_obs` / `kimi_obs` / `gemini_obs` / `tiebreaker_used` / `disagreements` field
occurs in any file (verified in `README.md`: 132/132 SHA-256 match, zero multi-model
fields).

This replaces the deposited AI source `output/li2022_combined/*_consensus.json`
(three-model consensus). The deposited AI key tables were **not read**; only the
outcome-stripped structural reference `00_SPEC/vocab_reference/li_j_ai_structural.csv`
was consulted, and only for structural-column comparison.

## 2. Decoder started from

`meta_analysis_extractor/decode_li2022_ai.py` (27 814 bytes, mtime 2026-05-31 13:21).
This is the only AI-side Li-2022 decoder in the repository; its `DECODER` constant
`claude-opus-4-8/decode_li2022_ai_v2` is the string carried on every row of the
deposited key table, so it is the submitted decoder.

Adapted to `02_DECODERS/li_j/decode_li_j.py`. Everything between the "VERBATIM" banners
— `nfkc`, `norm_text`, `sig3`, `NONYIELD_SUBSTR`, `YIELD_POS`, `YIELD_GENERIC`,
`is_yield`, `CROP_MAP`, `infer_crop`, `infer_pbs_category`, `infer_method`, `DOSE_KEYS`,
`DOSE_RE`, `infer_dose`, `infer_frequency`, `_unit_token`, `canon_unit` — is copied
character-for-character.

## 3. Changes made, with reasons

| # | Change | Reason |
|---|---|---|
| 1 | `SRC` repointed from `output/li2022_combined` to `01_INPUTS_FROZEN/li_j`; output written as the canonical 18-column CSV directly instead of JSONL | The one intended variable. The submitted pipeline wrote JSONL and then ran `matching/keys_from_jsonl.py`; that step is inlined here using the same `csv.DictWriter`, so quoting of the comma- and quote-bearing `evidence` field is identical. |
| 2 | **Schema adapter** `adapt_record()` (field renames / relocation only) | The frozen records use different field names and a flatter shape. See §4. Creates no information, drops no record, touches no number. |
| 3 | `paper_text` built from the frozen paper-level fields `title / species / crop / experiment_type / location` | The submitted decoder built it from the consensus `recon` block (`treatment_definition / control_definition / extraction_guidance`). The frozen files have no `recon`; these are the equivalent paper-level own-metadata fields. Same role, same provenance class (`crop_basis=paper_title`), no values. |
| 4 | Outcome-label separator normalization `_` → ` ` before `is_yield()` | The frozen agent writes some endpoint labels snake_cased (`total_biomass`, `fruit_fresh_weight`, `runner_yield`) where the consensus source used spaces. `is_yield`'s vocabulary is space-separated, so without this the *same* endpoint classifies differently depending only on punctuation. **Effect: 462 → 464 yield rows** (`total_biomass` and `fruit_fresh_weight`, both in `agriculture-10-00618-v2`). Applied uniformly, blind to every value. |
| 5 | **Paper crosswalk** onto the GT `paper_id` vocabulary | The frozen files are named with the clean corpus id (`006_Alabdulla_2019`) while the reference side stores the corpus id plus the source-PDF title fragment (`006_Alabdulla_2019_Effect of foliar application of humic ac`). Without a crosswalk no row could pair. Rule in §5. Structural tokens only. |
| 6 | Crop: a fourth rung appended after the three submitted rungs — the paper's own top-level `crop` / `species` field, normalized with the submitted rung-1 normalizer | See §7. The frozen source states the crop; discarding it is a decode loss, not a property of the source. |
| 7 | Crop token separator: emitted with spaces (`mung bean`) rather than snake_case (`mung_bean`) | The GT vocabulary for this dataset writes multi-word crops with spaces on all 1108 rows. This is the same silent non-matching defect the spec flags for Hui casing. Formatting only — the token is unchanged. See §7. |
| 8 | `is_figure` derived from the row's own `data_source` (`\bfig(\.|ure|s)?\b`) instead of the hard-coded `0` | The canonical schema requires figure-read rows be flagged and scored in a separate tier. **Effect: 8 rows**, all `strawberry_chitosan` ("estimated from Figure 2"). That paper has no GT counterpart, so this cannot change any pairing. |
| 9 | `timepoint` = constant `pooled` | The recovered script assigns a per-paper sequential `pair<N>` index, but every row of the AI key table actually deposited carries `pooled`, and so does every one of the 1108 reference rows (GT `timepoint` vocabulary = `{pooled}`). Emitting pair indices would make the match key structurally unable to pair on a column where the reference has exactly one value. The per-row year / season / frequency tokens the decoder did read are preserved in `evidence`. |
| 10 | `fold()` also folds stroke/bar letters (`ł→l`, `ø→o`, `đ→d`, `ß→ss`, `æ→ae`, `œ→oe`, `ð→d`, `þ→th`, `ı→i`) | Blocking bug in the crosswalk: `ł` carries no combining mark, so NFKD leaves it intact and `Głosek-Sobieraj` (reference) never compared equal to `Glosek-Sobieraj` (frozen id). Without this, 32/49 papers crosswalked instead of 33. |
| 11 | `main(outdir=…, emit=False, verbose=…)` parameters added | Harness only, so the sensitivity runs in §7 can collect rows without overwriting the delivered keys. The default call is byte-identical to before the parameters were added (aggregate SHA-256 unchanged: `8e240978…cd46b`). |

Not changed, deliberately:

* `is_yield()`'s endpoint vocabulary, `infer_pbs_category()`'s category vocabulary,
  `DOSE_KEYS`, `canon_unit()`'s unit tables. Retuning any of these against a reference
  whose category / crop / dose vocabulary I can see is exactly the drift the blindness
  rules forbid. Two consequences are visible in §8 and are reported, not fixed:
  `dt/ha` and `g/plant DM` fall through to unresolved unit tokens, and
  `027_Chen_2021` classifies as `protein_hydrolysate`.

## 4. Field mapping (source → key column)

No key column is derived from, conditioned on, or selected using `treatment_mean`,
`control_mean`, `effect_pct`, `variance_value`, `variance_type` or `n`.

| Key column | Source |
|---|---|
| `row_id` | `<crosswalked paper_id>__ai__<index of the record in consensus_observations>` |
| `side` | constant `ai` |
| `paper_id` | crosswalked GT token (§5), else the frozen file id |
| `outcome_canonical` | constant `yield` (rows failing `is_yield` are excluded, not relabelled) |
| `crop` | `infer_crop` on row `moderators.crop`/`.species` → endpoint label → paper text → (new) paper-level `crop`/`species`; emitted in the GT separator convention |
| `treatment_level` | `infer_pbs_category(treatment_description, moderators)` |
| `co_amendment` | `infer_method(treatment_description, moderators)` (application method) |
| `co_amendment_level` | `infer_dose(treatment_description, moderators)` |
| `timepoint` | constant `pooled` (change 9) |
| `aggregation_level` | constant `single_cell` |
| `unit_canonical` | `canon_unit(unit)` — declared-dimension only, never value-fit |
| `control_token` | constant `absolute_control` |
| `treatment_mean` / `control_mean` | `treatment_mean` / `control_mean` × the declared-unit conversion factor, copied through |
| `source_locator` | `data_source` |
| `is_figure` | figure cue in `data_source` (change 8) |
| `evidence` | endpoint label, treatment description, full moderator dict, declared unit, crop basis, frequency token, conversion factor, `n`, `variance_type`, control description, crosswalk rule |
| `decoder` | constant `rebuild_2026-08-19/li_j` |

### Schema adapter (change 2), exact mapping

| Field the submitted decoder reads | Frozen agent field(s) |
|---|---|
| `element` | `outcome` |
| `treatment_description` | `treatment_description` ‖ `treatment_label` ‖ `description` |
| `control_description` | `control_description` ‖ `control_label` |
| `moderators` (dict) | `moderators` when present (581 records), **merged with** every other record-level key that is not one of the core observation fields below (the other 472 records carry the same descriptors as flat sibling keys: `biostimulant_type`, `year`, `cultivar`, `application_method`, `rate`, `biostimulant_rate`, `crop`, `timing`, `season`, `setting`, `experiment`, `nitrogen_level`, `year_of_observation`) |
| `moderators['dose']` | `moderators['rate']` ‖ `moderators['biostimulant_rate']` when no `dose` key exists — a pure key alias, so the submitted `DOSE_KEYS` list is left untouched |

Deliberately **excluded** from the moderator merge, because the inference functions
substring-match a JSON blob of the moderators and admitting any outcome or statistical
field there would break outcome-blindness: `outcome`, `tissue`, `treatment_mean`,
`control_mean`, `effect_pct`, `ln_rr`, `n`, `unit`, `variance_value`, `variance_type`,
`treatment_variance`, `control_variance`, `significance`, `data_source`, `note(s)`,
`confidence`, `observation_id`, `grim_valid`, `cv_reasonable`, `direction_expected`,
and the description/label fields (handled explicitly above).

## 5. Crosswalk rule

Structural tokens only — never a crop, product, unit, dose or value. Ladder, first rung
that yields candidates must yield exactly one, otherwise the paper stays unmapped:

* **R1** folded-alphanumeric equality of the two id strings → 3 papers
  (`agriculture-10-00618-v2`, `plants-09-01633`, `sustainability-11-02171`).
* **R2** (first-author surname, 4-digit year) equality, parsed out of the id string
  itself after stripping a leading corpus index; surnames may be a prefix of one
  another at ≥4 characters, so `al-tawaha` matches `al-tawaha-et-al` → 27 papers.
* **R3** folded-alphanumeric substring containment, minimum length 8, which resolves
  accession-style ids (`S0304423819306703` inside `1-s2.0-S0304423819306703-main`) and
  `Azarpour` inside `article1400838000_Azarpour et al` → 3 papers.

Folding = lowercase + NFKD accent strip + the stroke/bar table of change 10.

**Result: 33 of 49 frozen papers crosswalked, and those 33 are exactly the 33 named
paper_id tokens the GT side carries** (the other 148 GT studies are `gt_studyNN`
placeholders — the GT decoder identified only the 33 papers that the AI corpus
contained, and I did not attempt to name any `gt_studyNN`, which would mean re-deriving
identities the as-submitted GT decoder deliberately left unresolved). No collisions, no
ambiguous matches. The 16 unmapped papers keep their own frozen id; they have no
reference counterpart and cannot pair.

Alias note for readers comparing against the deposited AI keys: the deposited side used
the full PDF filename stem where the frozen side uses a short id, e.g.
`BR-18-165` = `1542-1558-15(3)2018 BR-18-165`, `BR-1503` = `604-615-14(3)2017BR-1503`,
`Alrubaiee_2023` = `AlrubaieeAlsulaiman2023FullManuscript`,
`Effect_Seaweed_Extracts` = `Effect_of_Different_Seaweed_Extracts_and`,
`S1878818119307637` = `1-s2.0-S1878818119307637-main`,
`ascophyllum_carrot_2014` = `2014-ascophyllum-extract-application-…`,
`strawberry_chitosan` = `strawberryandchitosan`. For the 33 crosswalked papers both
sides now speak the same token.

Key-file names mirror the reference key filenames for crosswalked papers, so
`03_KEYS/ai_rebuilt/li_j/` and `03_KEYS/gt/li_j/` correspond 1:1 by name.

## 6. Record arithmetic

```
files in                 49
records in             1053
rows out                464
excluded                589
check       1053 == 464 + 589   ✓
```

| Exclusion reason | Records |
|---|---|
| `not_yield_outcome` — endpoint is not a harvested yield per the submitted `is_yield()` classifier | 589 |

Excluded endpoints, grouped for readability (105 distinct labels):

| Endpoint family | Records |
|---|---|
| antioxidant / secondary metabolite (phenolics, flavonoids, FRAP, ABTS, anthocyanins, proline …) | 186 |
| compositional quality (protein, fat, starch, sugars, nitrate, NDF/ADF, essential-oil content …) | 183 |
| morphometric / count (plant height, spike number, leaf area, tuber size, fibre indices …) | 92 |
| non-yield organ mass / growth (herb fresh/dry weight, root fresh weight, trunk growth …) | 76 |
| processing / physiological (falling number, wet gluten, sedimentation, bread volume, harvest index, chlorophyll …) | 38 |
| other | 14 |

Zero rows have a missing mean. Zero rows have an empty `crop`. 8 rows carry
`is_figure=1`. Three papers emit a header-only file: `006_Alabdulla_2019` (0 source
records — the agent wrote an `exclusion_reason`: the treatments are microbial
biofertilizers plus NPK, "not biostimulants per Li 2022 definition"),
`090_Kocira_2020` (64 records, all antioxidant/quality endpoints, no yield endpoint
extracted) and `S1878818119309879` (16 records: fresh weight, dry weight, plant height,
essential-oil content).

Determinism: the decoder was run twice from a cleaned output directory;
`diff -r` reports no difference. Aggregate SHA-256 over
`sorted(basename) + file bytes` = `8e240978d572e368d500f1bc589e6d5284e062cabfa571935d04c13623fcd46b`.

## 7. The crop decision

The deposited AI side had `crop` **empty on 367 of 576 rows** while the reference
carries 74 distinct crops, so most rows were structurally incapable of pairing. The
submitted crop ladder is row-moderator → endpoint label → CROP_MAP scan of the paper
text, and CROP_MAP has only 30 entries, so any crop the paper states plainly but that is
absent from that whitelist (oat, cardoon, cucumber, zinnia …) came out empty.

The frozen source states the crop at paper level in its own `crop` and `species` fields.
I therefore **append a fourth rung** that uses those fields, normalized by the submitted
rung-1 normalizer, and **emit the crop token in the reference's separator convention**
(spaces, not underscores — the GT side writes `common bean`, `mung bean`, `sweet pepper`
on all 1108 rows).

Rows out are 464 in every variant — crop is a key column, not a filter, so populating it
changes which rows *pair*, not how many exist. Reported both ways:

| Variant | crop empty | distinct crop tokens | tokens shared with GT | AI rows on a GT crop token | AI-only crop tokens | (paper, outcome, crop, category, method) keys shared with GT |
|---|---:|---:|---:|---:|---:|---:|
| **A** submitted ladder + snake_case (= submitted logic) | 62 (13.4%) | 25 | 16 | 313 (67.5%) | 9 | 25 keys / 128 AI rows |
| **B** submitted ladder + GT space form | 62 (13.4%) | 25 | 18 | 356 (76.7%) | 7 | 26 keys / 131 AI rows |
| **C** + paper-level crop, snake_case | 0 | 29 | 19 | 359 (77.4%) | 10 | 25 keys / 128 AI rows |
| **D** + paper-level crop + GT space form — **delivered** | 0 | 29 | 21 | 402 (86.6%) | 8 | 26 keys / 131 AI rows |

**Decision: variant D**, chosen on faithfulness, not on agreement. The frozen JSON
states the crop for every one of the 49 papers; leaving `crop` empty when the source
records it is a decode loss that manufactures an artificial non-match, and a separator
convention is a formatting choice that the reference has already fixed for this dataset.
The delivered variant does score better on the vocabulary overlap, but so would any
variant that populated the field correctly; the 62 rows that A leaves empty are simply
oat, cardoon, cucumber, zinnia, festulolium, sand lucerne, orchard grass and Italian
ryegrass rows whose crop is written in the source in plain words. Note also that the
improvement over the *deposited* side (367 empty of 576 → 62 empty of 464 even in
variant A) comes mostly from the frozen source carrying paper-level crop metadata at
all, not from change 6.

Crop tokens on the AI side that do not occur on the GT side, after variant D
(8 tokens, 62 rows): `festulolium` 2, `italian ryegrass` 12, `orchard grass` 2,
`sand lucerne` 3, `shallot` 6, `sugar beet` 9, `timothy grass` 12,
`zinnia ornamental` 16. Two of these are near-misses that I did **not** force:
GT writes `suger beet` (a typo in the published reference) and `timothy`.

## 8. Vocabulary diff vs `00_SPEC/vocab_reference/li_j_gt_structural.csv`

464 AI rows vs 1108 GT rows.

| Column | AI-only | GT-only |
|---|---|---|
| `outcome_canonical` | (none) | (none) — both `yield` |
| `crop` | 8 tokens / 62 rows (listed in §7) | 53 tokens / 472 rows — crops in studies with no AI source file at all (onion 33, garlic 28, bean 53, blackgram 19, chamomile 21, pea 16, …) |
| `treatment_level` | `amino_acid` 55, `microbial` 26, `other` 88 | `phosphite` 18 |
| `co_amendment` | `none` 75 | (none) |
| `co_amendment_level` | 75 tokens / **all 464 rows** | 32 tokens / **all 1108 rows** — **zero overlap** |
| `timepoint` | (none) | (none) — both `pooled` |
| `aggregation_level` | (none) | (none) — both `single_cell` |
| `unit_canonical` | 15 tokens / all 464 rows | `unresolved` / all 1108 rows — **zero overlap** |
| `control_token` | (none) | (none) — both `absolute_control` |
| `is_figure` | `1` on 8 rows | (none) |
| shared `crop` (21) | apple, barley, broccoli, cardoon, carrot, celery, cotton, cucumber, eggplant, grape, maize, mung bean, oat, olive, potato, soybean, strawberry, sugarcane, sweet pepper, tomato, wheat | |
| shared `treatment_level` (6) | chitosan, humic, plant_extract, protein_hydrolysate, seaweed, silicon | |
| shared `co_amendment` (3) | foliar, seed, soil | |

Two of these gaps are structural properties of the dataset that the submitted manuscript
already discloses, and I have not tried to close them:

* **`unit_canonical` cannot agree.** The human reference stores a normalised scale —
  `unresolved` on all 1108 rows — while the AI side stores the real declared unit
  (`t/ha` 225, `g/plant` 88, `dt_ha` 60, `ton_feddan` 18, `g_plantdm` 18, `kg_fad` 16 …).
  Units are therefore unusable as a matching constraint here.
* **`co_amendment_level` cannot agree.** The reference stores a *within-study normalised*
  dose (0.01 – 1, with 731 rows at exactly `1`); the AI side stores the paper's declared
  dose with its unit (`3g/l`, `1.5dm3/ha`, `250ppm`, …). Overlap is zero by construction.

Consequently the full 8-field match key produces **0 shared keys**, exactly as the
submitted analysis found ("a normalised reference scale and divergent product and crop
tokens leave one structural row per study"), which is why the submitted comparison was
run on **effects, at study level**. Dropping `co_amendment_level`, `timepoint` and
`aggregation_level` from the key leaves **26 shared (paper, outcome, crop, category,
method) coordinates covering 131 AI rows and 75 GT rows**.

## 9. Coverage of the reference set

| | |
|---|---:|
| GT studies | 181 |
| GT studies crosswalked to a frozen AI paper | **33** |
| … of which have ≥1 rebuilt AI row | 31 |
| GT rows in the 33 crosswalked studies | 235 (21.2% of 1108) |
| GT rows in the 31 studies that actually have AI rows | 205 (18.5% of 1108) |
| Rebuilt AI rows that sit in a crosswalked (pairable) paper | 320 of 464 (69.0%) |
| Rebuilt AI rows in papers with no reference counterpart | 144 of 464 (31.0%) |

Per-paper rows, crosswalked papers (AI rows / GT rows):
006_Alabdulla_2019 0/6 · 009_Ali_2019 6/4 · 027_Chen_2021 12/24 · 029_Ciepiela_2019 12/2 ·
058_Fichhof_2018 1/2 · 062_Glosek-Sobieraj_2018 60/6 · 067_Grabowska_2012 12/6 ·
086_Knapowski_2019 14/4 · 088_Kocira_2019 12/12 · 090_Kocira_2020 0/24 ·
091_Kocira_2018 12/36 · 094_Kowalska_2021 6/4 · 095_Kuisma_1989 9/3 ·
100_Lola-Luz_2014 6/6 · 105_Mattner_2018 4/3 · 106_Matysiak_2018 12/6 ·
110_Michalak_2016 5/5 · 120_Pohl_2019 10/12 · 121_Popescu_2018 12/12 ·
124_Pramanick_2016 13/8 · 125_Prochazka_2015 4/6 · 126_Prokkola_2007 15/6 ·
131_Rahman_2018 4/4 · 153_Soppelsa_2018 10/7 · 158_Sulakhudin_2019 6/3 ·
175_Wilczewski_2018 9/1 · Azarpour 14/2 · S0304423819306703 6/6 ·
S0304423820302417 2/2 · agriculture-10-00618-v2 4/1 · al-tawaha_2011 18/8 ·
plants-09-01633 2/2 · sustainability-11-02171 8/2.

Unmapped frozen papers (AI rows): 002_Abdel-Mawgoud_2010 2 · 064_Godlewska_2016 4 ·
065_Godlewska_2018 3 · 111_Mondal_2013 8 · 116_Nurdiawati_2019 8 ·
127_Radkowski_2018 12 · Alrubaiee_2023 12 · BR-1503 15 · BR-18-165 16 ·
Effect_Seaweed_Extracts 22 · L3638 8 · S1878818119307637 12 · S1878818119309879 0 ·
ali 6 · ascophyllum_carrot_2014 8 · strawberry_chitosan 8.

## 10. Scope difference vs the deposited (consensus-era) AI side

Structural row counts only; no value was compared. Both sides have 46 papers carrying
rows, but the rebuild has **464 rows against the deposited 576**, and the difference is
not uniform. Papers where the single-model March run extracted materially *less*:
`1-s2.0-S1878818119307637-main` 36→12, `029_Ciepiela_2019` 46→12,
`Effect_of_Different_Seaweed_Extracts_and` 32→22, `090_Kocira_2020` 24→0,
`1542-1558-15(3)2018 BR-18-165` 22→16, `065_Godlewska_2018` 18→3,
`027_Chen_2021` 28→12, `006_Alabdulla_2019` 12→0, `120_Pohl_2019` 22→10,
`106_Matysiak_2018` 23→12, `sustainability-11-02171` 17→8. Papers where it extracted
materially *more*: `062_Glosek-Sobieraj_2018` 45→60,
`article1400838000_Azarpour et al` 1→14, `al-tawaha_2011` 9→18,
`095_Kuisma_1989` 2→9, `002_Abdel-Mawgoud_2010` 0→2, `strawberryandchitosan` 0→8,
`2014-ascophyllum-…` 1→8. This is the AI-side scope change the rebuild exists to expose;
it is reported, not corrected.

## 11. Open items

1. **`027_Chen_2021` classifies as `protein_hydrolysate`, not `seaweed`.** Its moderator
   reads "cultured kelp enzymatic hydrolysate"; the submitted `infer_pbs_category`
   seaweed list contains `kelpak` but no bare `kelp`, so the `hydrolysate` token in the
   later protein-hydrolysate branch wins. This blocks pairing on 12 AI rows against the
   24-row reference study with the largest shared row count. I did **not** add a `kelp`
   token: the reference's category vocabulary is visible to me, and retuning the
   classifier against it is precisely the drift the blindness rules forbid. Flagged for a
   pre-registered vocabulary decision instead.
2. **`dt/ha` (60 rows) and `g/plant DM` (18 rows) fall through `canon_unit` to
   unresolved tokens** (`dt_ha`, `g_plantdm`), so their means are copied without
   conversion. Because the analysis compares treatment/control *ratios*, a constant unit
   factor cancels, so no effect estimate is affected — but the `unit_canonical` column is
   not clean for those rows. Same reasoning as item 1 for why the unit table was not
   extended.
3. **Corpus-provenance discrepancy at slot `006_Alabdulla_2019`.** The frozen agent
   record for that file reports `title = "Effect of Mineral-Biofertilizer on
   Physiological Parameters and Yield"`, authors `Al-Freeh, Alabdulla, Huthily` (2019),
   and returned 0 observations with an exclusion reason; the reference token for the same
   corpus slot is `006_Alabdulla_2019_Effect of foliar application of humic ac` and the
   reference rows are oat × foliar humic acid. A second corpus file,
   `AlrubaieeAlsulaiman2023FullManuscript` (frozen `Alrubaiee_2023`), *is* an oat ×
   foliar-humic-acid paper with 12 yield rows. Either the slot holds a different article
   than the reference cites, or the two PDFs overlap. The crosswalk was left on
   author+year as specified, and the discrepancy is recorded here rather than resolved by
   matching on crop/treatment tokens, which the brief explicitly rules out. It belongs in
   `matching/CORPUS_ERRORS.md` alongside the Hui `11_Zhao_2020` case.
4. **148 of 181 GT studies remain `gt_studyNN`.** They have no AI source file, so this
   is a corpus-scope fact, not a decode failure — but it means any headline built on this
   dataset rests on at most 33 of 181 reference studies (205 of 1108 reference rows have
   a pairable AI counterpart at all).
5. The recovered submitted decoder script emits `timepoint = "pair<N>"` while the
   deposited keys carry `pooled`. The script in the repository is therefore **not** the
   exact revision that produced the deposited keys; no other copy exists. Change 9
   reproduces the deposited (and GT-compatible) behaviour, but the provenance gap in the
   submitted repository is itself unresolved.
6. **The reference side's `paper_id` vocabulary is itself conditioned on the AI side.**
   33 of the 181 GT tokens are not identifiers of the published Li 2022 supplement at
   all; they are the *consensus-era AI PDF filenames*, complete with truncated article
   titles (`006_Alabdulla_2019_Effect of foliar application of humic ac`,
   `1-s2.0-S0304423819306703-main`, `article1400838000_Azarpour et al`), while the other
   148 are anonymous `gt_studyNN` placeholders. Whether a GT study got a real name
   therefore depended on what the AI corpus happened to contain. This is the same class
   of finding as `GT_SIDE_WAS_CONDITIONED_ON_AI_SIDE.md` (biochar), and it is the reason
   an author+year crosswalk is needed on this dataset at all. It does not move any value
   — a paper id is not a measurement — but "each side decoded independently and blind to
   the other" is not what the deposited Li-2022 key tables implement either.
