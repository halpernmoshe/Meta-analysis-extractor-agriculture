# hui — AI-side decoder ledger (independent rebuild, 2026-08-19)

Dataset: **Hui et al. 2025**, wheat grain-Zn biofortification. Outcome `grain_zn` (mg/kg).
Decoder: `02_DECODERS/hui/decode_hui.py`, tag `rebuild_2026-08-19/hui`.

**Two key sets are emitted**, identical in every column except `treatment_level`, so that the
effect of the source change and the effect of the parser change can be attributed separately
(see §10):

| Key set | Variant | `treatment_level` rule | Files | Rows |
|---|---|---|---|---|
| `03_KEYS/ai_rebuilt_strict/hui/` | `strict` | `treatment_description` only — literal port of `gen_ai_keys.py::app_type`. **Like-for-like baseline.** | 29 | **515** |
| `03_KEYS/ai_rebuilt/hui/` | `method_field_first` | record's own explicit Zn-application-method field first, descriptor union as fallback | 29 | **515** |

Both apply the mandated casing normalisation. **40 of 515 rows (7.8 %) differ** between them;
§10 gives the full breakdown and the pairing consequence.

Exactly one variable changed vs the submission: the AI-side **source** is now the frozen
March-2026 single-model Claude agent JSONs (`01_INPUTS_FROZEN/hui/*_agent.json`, 37 files,
748 records) instead of the multi-model consensus folder
`meta_analysis_extractor/output/hui2023_full_35/*_consensus.json`. No GT file is read by
the decoder; no analysis/TOST/fidelity computation was run here.

---

## 1. Submitted decoders started from

| Stage | Path |
|---|---|
| 1 (named base) | `SUBMISSION_Environmental_Evidence/resubmission/matching/runs/hui/gen_ai_keys.py` |
| 1 — sibling scripts of the same chain | `runs/hui/_gen_ai_keys.py`, `runs/hui/decode_ai_batch.py`, `runs/hui/_decode_ai.py` |
| 1b — documented key canonicalisations | `runs/hui/diagnosis_iter0.md` (root causes 1 & 2), `runs/hui_v2/reemit_ai_aggregation.py` |
| 2 (corpus cleaning + canonical paper_id) | `runs/hui_v4/build_hui_v4.py` |

**Important provenance finding.** The submitted AI side was not produced by one script.
`gen_ai_keys.py` carries a hard-coded `PAPERS` list of 8 papers; three sibling scripts in
the same directory cover 8 papers each with **divergent** rules (different `is_grain_zn`
unit tests, different `app_type` keyword sets and — crucially — different *casing*, see
§6), and two further papers (`Rashid_2019`, `Zhang_2017`) were produced by a script that is
no longer in the repository. The deposited AI key table is therefore a heterogeneous union
of ≥5 decoders. The rebuild spec requires a single deliverable decoder, so this rebuild
implements **one** decoder = the named base (`gen_ai_keys.py`) **plus the union of the
sibling scripts' documented rules**, applied uniformly to all 37 files. Every such rule is
attributed inline in the code and listed in §2. This unification necessarily changes some
individual rows relative to the deposited keys and is the largest single deviation in this
rebuild; it is stated here rather than hidden.

---

## 2. Every change made, with reason

| # | Change | Class | Reason |
|---|---|---|---|
| C1 | `AI_DIR` → `01_INPUTS_FROZEN/hui`, glob `*_agent.json`; record list still read from the key `consensus_observations` | (a) input path | The frozen single-model files inherit the legacy key name. |
| C2 | Hard-coded 8-paper `PAPERS` list → sorted glob over all 37 frozen files | (a)+(c) | `gen_ai_keys.py` alone covers 8/34 papers; spec deliverable is one decoder for the dataset. Sorted iteration also guarantees determinism. |
| C3 | **Schema adapter — two record layouts.** 26 files nest descriptors under `moderators`; 5 files (`69_Ramzan_2020`, `70_Rehman_2018`, `82_Torun_2001`, `84_Yilmaz_1998`, `Dong_2018`) inline the same descriptors at record level. Added `get()`/`moderator_blob()` to read either. Also `tissue` ⇄ `plant_part`, `obs_id` ⇄ `observation_id`, and top-level `recon` (old) → `title/authors/year/species/journal/...` (frozen) | (b) schema | Without the adapter the 5 flat files lose their tissue field (`tissue` absent ⇒ `tissue != "grain"`) and **all 56 of their records would be silently dropped**, including 3 papers that are in the GT (ramzan_2020, rehman_2018, yilmaz_1998). |
| C4 | `paper_id` token derived from the frozen top-level `paper_id` **label** (numeric prefix stripped, surname reduced to letters, lowercased, `_<year>` appended) using build_hui_v4's own `derive_paper_id` construction | (b) schema | 8 of 37 frozen filenames are publisher slugs (`fpls-08-00281`, `s11104-016-2815-3`, `pse_pse-201308-0003`, `41598_2018_Article_25247`, `agronomy-10-01566-v2`, `HarevstPlus_Zouetal2012`, `fpls-10-00426`, `s11104-015-2758-0`), so build_hui_v4's *filename*-based rule fails. The JSON's own `paper_id` label supplies the author+year token. Crosswalk in §4. Structural tokens only — no value was consulted. |
| C5 | Corpus cleaning: 8 mislabelled papers excluded at key-build stage (`EXCLUDE_MISLABELLED`) | preserved + extended | `build_hui_v4.py` excluded only `zhao_2020` at build time; the other 7 were excluded downstream by **every** submitted analysis script (`line_by_line_scope_aware.py`, `scope_aware_paired_tost.py`, `scope_aware_aggregate_tost.py`, `make_fig1_fidelity.py`, `make_fig2_equivalence.py`, `make_bland_altman.py` all carry the identical 8-token set). The manuscript calls hui_v4 "clean corpus after excluding 8 mislabelled PDFs". Reasons transcribed from `04_ANALYSIS/_AS_SUBMITTED/corpus_mislabels_D2.csv`. Downstream results are unaffected by *where* the exclusion is applied because the analysis scripts re-exclude the same tokens. Per-paper log in §5. |
| C6 | `treatment_level` (Zn application-method axis): **primary** signal is the record's own explicit method field (`application_method` \| `zn_method` \| `zn_application_method`); descriptor keyword decode is the **fallback**. **Applies to the `method_field_first` key set ONLY** — the `strict` key set does not use the method field at all (§10) | (b) schema + accuracy | All 748 frozen records carry one of these three fields with a clean closed vocabulary (`soil`, `foliar`, `soil+foliar`, `seed`, `seed priming`, `seed coating`, `seed biofortification`, `seed+foliar`, `foliar + seed`, `foliar+pesticide`, `nutrient solution`, `fertigation`, `none`). The submitted sibling `_gen_ai_keys.py` already folded moderator text into its `app_type` blob, so moderator-based method decoding is within the submitted chain. It removes free-text mis-parses that the descriptor-only rule makes on co-factor clauses — e.g. `66_PahlavanRad_2009` "40 kg ZnSO4/ha, **Fe 1% foliar**" decodes to `Foliar` from the descriptor but `Soil` (correct) from `zn_method`. |
| C7 | (`method_field_first` set only) Descriptor fallback = union of the four siblings' keyword sets (`foliar`/`spray`/`leaf`; `soil`/`broadcast`; `seed`/`coat`/`prim`), plus their dose-form fallbacks (`%` w/v without `ha` ⇒ Foliar, from `decode_ai_batch.py`; numeric Zn rate with no method word ⇒ Soil, from `gen_ai_keys.py`), with two explicit precedence rules: negated mentions (`no soil Zn`) are stripped before detection, and an explicit soil application outranks a seed-Zn co-factor (`"Soil Zn at 23 kg/ha, seed Zn 355 ng/seed"` ⇒ `Soil`, as the submitted `_decode_ai.py` did for `84_Yilmaz_1998`) | union | `gen_ai_keys.py`'s narrow `has_seed` test ("seed priming"/"priming" only) mis-decodes `38_Yilmaz_1997` "Seed treatment 30% ZnSO4" as `Soil`; the broad test alone mis-decodes `84_Yilmaz_1998` as `Seed`. The precedence rules reproduce what the submitted chain achieved per-paper. |
| C8 | **Casing normalised to the GT vocabulary** (`Soil`, `Foliar`, `Soil+Foliar`; AI-surplus `Seed`, `Seed+Foliar` in the same casing) | mandated defect fix | See §6. |
| C9 | Unit scope test widened to `re.search(r"mg\s*[ /]?\s*kg", unit)` (the sibling `decode_ai_batch.py` form) alongside `ug/g`, `µg/g`, `ppm`; area-basis (`/ha`) still rejected as *content* not concentration | union / (c) | `gen_ai_keys.py`'s literal set does not recognise `mg/kg DW` (20 frozen records, papers `82_Torun_2001`, `84_Yilmaz_1998`), which would have been silently dropped. |
| C10 | `control_token`: keyword union of all four siblings (adds `nil`, `0 kg`, `0 mg`, `zn 0`, `0 zn`, `no zinc`, `no micronutrient`, `distilled water`, `deionized`, `foliar dw`, `deficient`, `low zn`); a comparator that itself carries a Zn application ⇒ `co_factor_present_unmatched` (rule from `_gen_ai_keys.py`); residual ⇒ `other` (base-script behaviour, a legal §4 token that routes to human review) | union | `gen_ai_keys.py`'s 6-keyword list leaves 68 "Nil Zn" and 20 "0 kg Zn/ha" controls as `other`. `control_token` is **not** a match-key field; it is reported as concordance only. |
| C11 | `co_amendment` / `co_amendment_level`: lime (from `lime_rate`/`lime_treatment`, `gen_ai_keys.py`), nitrogen (`decode_ai_batch.py` rule, with the frozen field names `nitrogen_kg_ha`, `n_rate_kg_ha`, `n_rate` added to `nitrogen_rate`/`nitrogen_level`), sucrose (`_decode_ai.py`, 3.0 % w/v tank mix) | (b) schema | Same axes as the submitted chain; only the moderator spellings differ in the frozen files. |
| C12 | `crop` decoded from the top-level `species` string (Triticum⇒wheat, Oryza⇒rice, Hordeum⇒barley, Zea⇒maize) | union | `gen_ai_keys.py` hard-codes `wheat`; the siblings `_gen_ai_keys.py`/`_decode_ai.py` decode crop, and the deposited AI side carries rice/maize/barley. See §7 for the outcome. |
| C13 | `timepoint` emitted as `""` for every row | preserved submitted canonicalisation | `runs/hui/diagnosis_iter0.md` root cause 1: the Hui GT compilation has no year/site column and the source PDFs are absent, so `timepoint` is undeterminable on the GT side and was blanked on the AI side (24/34 files edited) as coordinate canonicalisation, not value fitting. The *decoded* token (`y2013`, `pooled`, …) is still computed and written into the `evidence` column as `decoded_timepoint(blanked):…` so nothing is lost. |
| C14 | `aggregation_level` emitted as `single_cell` for every row | preserved submitted canonicalisation | `diagnosis_iter0.md` root cause 2 (blanked on both sides, GT value was a hard-coded default) followed by `runs/hui_v2/reemit_ai_aggregation.py` (re-emitted as `single_cell` to match the GT convention). Net submitted state = `single_cell` on all rows, which is what the deposited keys contain. Decoded value retained in `evidence` as `decoded_aggregation(canonicalised):…`. |
| C15 | `zhang_2012` scope note preserved: `treatment_level=""` because the focal axis is N rate with identical basal ZnSO4 on both arms | preserved | Verbatim from the submitted `_gen_ai_keys.py`; the frozen file's own `note` states "All plots received basal 30 kg/ha ZnSO4·7H2O. The treatment factor is N rate, not Zn rate." (Moot in practice: `zhang_2012` is also mislabel-excluded.) |
| C16 | `row_id` = `<canon_token>__ai__<index-in-source-list>`; `decoder` = `rebuild_2026-08-19/hui` | stage 2 shape + spec | `build_hui_v4.py` re-keyed row ids to `<canon>__ai__<suffix>`; index is over the full `consensus_observations` list (base-script behaviour) so a row traces back to its source record. |
| C17 | Determinism hardening: sorted `glob`, stale `*.csv` purge before writing, `sort_keys=True` on the moderator JSON embedded in `evidence` | (c) | Re-run must be byte-identical. Verified for both key sets (§9). |
| C18 | **Unit conversion `mg/100g` → `mg/kg` (×10)** applied in **both** key sets, with the mean values converted. Implemented with `Decimal` arithmetic (exact decimal, no float drift) and rendered as a plain decimal string. The factor is chosen from the **unit string alone** | protocol §5 | `unit_canonical` is a match-key field that must agree, so an unconverted `mg/100 g` row would be unpairable on a units technicality. ×10 is arithmetic, not judgement. Deliberately **not** validated against magnitudes: `unit_canonical` is a key field and spec rule 1 forbids conditioning a key field on any mean. See §5 for the (nil) effect and §8 item 5 for the ambiguity assessment. |
| C19 | Decoder restructured into `decode(variant, out_dir)` + a variant loop, emitting both key sets in one deterministic invocation; `evidence` now carries `variant:` and `unit_conversion:` audit tokens | (c) | Needed to produce the strict/method-field pair from one file (the spec's single-decoder deliverable). **Regression-verified**: re-running the pre-refactor decoder and diffing all 17 key/scored columns against the refactored `ai_rebuilt` set gives **zero differences** on 515/515 rows — only `evidence` changed. |

Not changed, deliberately: means are copied through **verbatim** (`str()` of the JSON value,
`None` ⇒ empty). The **only** exception is the `mg/100g → mg/kg` unit conversion (C18), which
is a pure decimal rescale decided from the unit string; every other unit in the corpus
(`mg/kg`, `mg/kg DW`, `ug/g`, `ppm`) is numerically identical to the canonical unit and needs
no conversion.

---

## 3. Field mapping (raw JSON → key column)

| Key column | Source | Rule |
|---|---|---|
| `row_id` | derived | `<paper token>__ai__<index in consensus_observations>` |
| `side` | constant | `ai` |
| `paper_id` | top-level `paper_id` (fallback: filename stem minus `_agent`) | strip numeric prefix → deaccent → surname letters lowercased + `_<year>` |
| `outcome_canonical` | constant | `grain_zn` (scope of this dataset) |
| `crop` | top-level `species` (+ record `species`/`species_note`/`wheat_type`) | genus/common-name map |
| `treatment_level` | `application_method` \| `zn_method` \| `zn_application_method` (record or `moderators`); fallback `treatment_description` | `METHOD_MAP` → GT casing; fallback = keyword decode |
| `co_amendment` | `lime_rate`/`lime_treatment`; `nitrogen_rate`/`nitrogen_level`/`nitrogen_kg_ha`/`n_rate_kg_ha`/`soil_N_rate`/`n_rate`; `sucrose` in `treatment_description` | `lime` / `nitrogen` / `sucrose` / `none` |
| `co_amendment_level` | same fields | first number as a plain decimal string; `0` when none or "no lime" |
| `timepoint` | `year`/`growing_season`/`years` | decoded, then canonicalised to `""` (C13); decoded token preserved in `evidence` |
| `aggregation_level` | moderators + descriptor ("mean of", "averaged", "pooled") | decoded, then canonicalised to `single_cell` (C14); decoded token preserved in `evidence` |
| `unit_canonical` | `unit` | constant `mg/kg` for all in-scope rows (`mg/kg`, `mg/kg DW`, `ug/g` are numerically identical); non-conforming units excluded, not converted |
| `control_token` | `control_description` | §4 closed vocabulary (C10) |
| `treatment_mean` | `treatment_mean` | verbatim |
| `control_mean` | `control_mean` | verbatim |
| `source_locator` | `data_source` | verbatim (empty on the 5 flat-layout files, which carry no `data_source`) |
| `is_figure` | `data_source` | `1` if it matches `/fig/i` |
| `evidence` | descriptors, element, tissue, unit, n, obs id, method provenance, full moderator dict, the two canonicalised-away tokens | audit trail |
| `decoder` | constant | `rebuild_2026-08-19/hui` |

---

## 4. paper_id crosswalk (structural tokens only — author + year, never values)

37 frozen source files → 37 distinct canonical tokens. **All 26 GT paper tokens are present
in the source set** (0 GT papers unrepresented at file level).

| Frozen file | JSON `paper_id` label | canonical token | records | in GT? |
|---|---|---|---|---|
| `pse_pse-201308-0003_agent.json` | Bharti_2013 | `bharti_2013` | 40 | GT |
| `44_Cakmak_1997_agent.json` | 44_Cakmak_1997 | `cakmak_1997` | 78 | GT (excluded) |
| `fpls-08-00281_agent.json` | Chattha_2017 | `chattha_2017` | 24 | GT |
| `42_Curtin_2008_agent.json` | 42_Curtin_2008 | `curtin_2008` | 28 | ai-only |
| `41598_2018_Article_25247_agent.json` | Dapkekar_2018 | `dapkekar_2018` | 4 | ai-only |
| `49_Dawar_2022_agent.json` | 49_Dawar_2022 | `dawar_2022` | 0 | ai-only |
| `Dong_2018_agent.json` | Dong_2018 | `dong_2018` | 12 | GT (excluded) |
| `50_Erdal_2002_agent.json` | 50_Erdal_2002 | `erdal_2002` | 20 | GT |
| `52_Forster_2018_agent.json` | 52_Forster_2018 | `forster_2018` | 11 | GT |
| `46_Ghasal_2017_agent.json` | 46_Ghasal_2017 | `ghasal_2017` | 8 | GT |
| `s11104-015-2758-0_agent.json` | Gomez-Coronado_2016 | `gomezcoronado_2016` | 9 | ai-only |
| `53_Grant_1998_agent.json` | 53_Grant_1998 | `grant_1998` | 0 | ai-only |
| `58_Kalayci_1999_agent.json` | 58_Kalayci_1999 | `kalayci_1999` | 82 | GT |
| `59_Khoshgoftarmanesh_2013_agent.json` | 59_Khoshgoftarmanesh_2013 | `khoshgoftarmanesh_2013` | 16 | GT (excluded) |
| `61_Kumar_2018_agent.json` | 61_Kumar_2018 | `kumar_2018` | 0 | GT (excluded) |
| `Li_2013_agent.json` | Li_2013 | `li_2013` | 23 | GT (excluded) |
| `Liu_2014_agent.json` | Liu_2014 | `liu_2014` | 2 | GT (excluded) |
| `fpls-10-00426_agent.json` | Liu_2019 | `liu_2019` | 20 | GT |
| `62_Morshedi_2012_agent.json` | 62_Morshedi_2012 | `morshedi_2012` | 0 | ai-only |
| `63_Mosavian_2021_agent.json` | 63_Mosavian_2021 | `mosavian_2021` | 24 | ai-only |
| `65_Oliver_1994_agent.json` | 65_Oliver_1994 | `oliver_1994` | 0 | ai-only |
| `66_PahlavanRad_2009_agent.json` | 66_PahlavanRad_2009 | `pahlavanrad_2009` | 10 | GT |
| `68_Peck_2008_agent.json` | 68_Peck_2008 | `peck_2008` | 18 | GT |
| `s11104-016-2815-3_agent.json` | Ram_2016 | `ram_2016` | 48 | ai-only |
| `69_Ramzan_2020_agent.json` | 69_Ramzan_2020 | `ramzan_2020` | 8 | GT |
| `Rashid_2019_agent.json` | Rashid_2019 | `rashid_2019` | 38 | GT |
| `70_Rehman_2018_agent.json` | 70_Rehman_2018 | `rehman_2018` | 16 | GT |
| `82_Torun_2001_agent.json` | 82_Torun_2001 | `torun_2001` | 15 | ai-only |
| `21_Wang_2012_agent.json` | 21_Wang_2012 | `wang_2012` | 18 | GT |
| `05_Yang_2011_agent.json` | 05_Yang_2011 | `yang_2011` | 29 | GT |
| `38_Yilmaz_1997_agent.json` | 38_Yilmaz_1997 | `yilmaz_1997` | 40 | GT |
| `84_Yilmaz_1998_agent.json` | 84_Yilmaz_1998 | `yilmaz_1998` | 5 | GT |
| `03_Zhang_2012_agent.json` | 03_Zhang_2012 | `zhang_2012` | 12 | GT (excluded) |
| `Zhang_2017_agent.json` | Zhang_2017 | `zhang_2017` | 0 | GT |
| `11_Zhao_2020_agent.json` | 11_Zhao_2020 | `zhao_2020` | 6 | ai-only (excluded) |
| `HarevstPlus_Zouetal2012_agent.json` | Zou_2012 | `zou_2012` | 68 | GT |
| `agronomy-10-01566-v2_agent.json` | Zulfiqar_2020 | `zulfiqar_2020` | 16 | GT |

Note on identity: `44_Cakmak_1997` and `s11104-015-2758-0` are the **same publication**
(Gómez-Coronado et al. 2016, Plant Soil 401:331-346) reached under two labels — the
mislabelled one and the correctly named one. The mislabelled token is excluded (C5); the
correctly named token survives as an AI-only paper, which is the right handling.

**Papers covered:** 26/26 GT papers present as source files; **18/26** yield ≥1 rebuilt key
row. The 8 that yield none: `cakmak_1997`, `dong_2018`, `khoshgoftarmanesh_2013`,
`kumar_2018`, `li_2013`, `liu_2014`, `zhang_2012` (all mislabel-excluded, C5) and
`zhang_2017` (frozen source contains 0 records — the agent recorded that the paper studies
irrigation × N and reports no Zn data). Five AI-only papers contribute rows: `curtin_2008`,
`dapkekar_2018`, `gomezcoronado_2016`, `mosavian_2021`, `ram_2016`.

---

## 5. Record arithmetic

Identical for **both** key sets (they differ only in the value of one column, never in which
rows are emitted):

```
files in                 37
records in              748
rows out                515
excluded                233
check                   748 = 515 + 233   OK
key CSVs written         29   (23 with >=1 row, 6 header-only)

rows unit-converted mg/100g -> mg/kg                : 0
convertible records inside mislabel-excluded papers : 16
treatment_level case-normalised rows                : 0  (both variants emit GT casing by construction)
```

**On the unit conversion (C18).** The conversion is implemented and active, but it converts
**0 emitted rows**, and no row in either key set ever carried `unit_canonical = mg/100 g`:
`unit_canonical` is the constant `mg/kg` on all 515 rows in both sets. All 16 `mg/100g`
records in the corpus belong to `59_Khoshgoftarmanesh_2013`, which is removed one step
earlier by the mislabelled-PDF exclusion (C5), so they were counted under
`paper_excluded_mislabelled_pdf[khoshgoftarmanesh_2013]` and never reached the unit test.
There were therefore never 16 rows sitting unpairable on a units technicality. The conversion
now guarantees the correct behaviour if that paper is ever reinstated, and it is the only
non-verbatim treatment any mean receives.

### Exclusions by reason

| Reason | Records | Detail |
|---|---:|---|
| `paper_excluded_mislabelled_pdf` | **149** | 8 papers, per-paper table below |
| `non_grain_tissue:shoot` | 64 | `42_Curtin_2008` 22, `82_Torun_2001` 15, `38_Yilmaz_1997` 20, `21_Wang_2012` 6, `84_Yilmaz_1998` 1 |
| `non_grain_tissue:straw` | 14 | `46_Ghasal_2017` 4, `fpls-10-00426`/liu_2019 10 |
| `non_Zn_element` | 6 | `21_Wang_2012` grain Fe |
| **total** | **233** | |

Papers whose *whole file* was excluded (reason text from `corpus_mislabels_D2.csv`):

| Token | File | Records lost | Documented reason |
|---|---|---:|---|
| `cakmak_1997` | `44_Cakmak_1997_agent.json` | 78 | PDF is Plant Soil (2016) 401:331-346, Portuguese INIAV wheat lines, Elvas 2010-2013 (= Gómez-Coronado et al. 2016) |
| `li_2013` | `Li_2013_agent.json` | 23 | PDF is Impa, Morete et al., J. Exp. Bot. 2013, "Zn uptake, translocation and grain Zn loading in rice" |
| `khoshgoftarmanesh_2013` | `59_Khoshgoftarmanesh_2013_agent.json` | 16 | filename-vs-content mismatch (B2 source-verification) |
| `dong_2018` | `Dong_2018_agent.json` | 12 | filename-vs-content mismatch (B2 source-verification) |
| `zhang_2012` | `03_Zhang_2012_agent.json` | 12 | filename-vs-content mismatch (B2 source-verification) |
| `zhao_2020` | `11_Zhao_2020_agent.json` | 6 | PDF is Mirbolook et al., Commun. Soil Sci. Plant Anal.; header "A. MIRBOLOOK ET AL." |
| `liu_2014` | `Liu_2014_agent.json` | 2 | PDF is Uddin, Kaczmarczyk & Vincze (barley hordein transcripts & Zn, Aarhus) |
| `kumar_2018` | `61_Kumar_2018_agent.json` | 0 | filename-vs-content mismatch (B2 source-verification) |

Source files with **0 records** (agent-side scope determination, quoted from each file's own
`note`/`notes`; header-only CSVs written, no rows):

| Token | File | Agent's stated reason |
|---|---|---|
| `dawar_2022` | `49_Dawar_2022_agent.json` | reports yield and yield components only; no grain or tissue Zn concentration |
| `grant_1998` | `53_Grant_1998_agent.json` | reports grain yield and grain **Cd**; no grain Zn concentration |
| `morshedi_2012` | `62_Morshedi_2012_agent.json` | reports grain protein and yield under Zn × K; no grain Zn concentration |
| `oliver_1994` | `65_Oliver_1994_agent.json` | dependent variable is **Cd** in grain, not Zn |
| `zhang_2017` | `Zhang_2017_agent.json` | irrigation × N study; no Zn treatment and no Zn concentration |
| `kumar_2018` | `61_Kumar_2018_agent.json` | GWAS across 330 lines; uniform ZnSO4 to all plots, so no treatment–control contrast (also mislabel-excluded) |

### Rows out per paper (23 papers with rows)

`kalayci_1999` 82 · `zou_2012` 68 · `ram_2016` 48 · `bharti_2013` 40 · `rashid_2019` 38 ·
`yang_2011` 29 · `chattha_2017` 24 · `mosavian_2021` 24 · `erdal_2002` 20 · `yilmaz_1997` 20 ·
`peck_2008` 18 · `rehman_2018` 16 · `zulfiqar_2020` 16 · `forster_2018` 11 · `liu_2019` 10 ·
`pahlavanrad_2009` 10 · `gomezcoronado_2016` 9 · `ramzan_2020` 8 · `curtin_2008` 6 ·
`wang_2012` 6 · `dapkekar_2018` 4 · `ghasal_2017` 4 · `yilmaz_1998` 4.
Header-only: `dawar_2022`, `grant_1998`, `morshedi_2012`, `oliver_1994`, `torun_2001`
(all 15 records are shoot Zn), `zhang_2017`.

---

## 6. The casing defect — quantified and fixed

**The defect.** The deposited AI keys emit `treatment_level` in mixed case, because the
submitted AI side is a union of ≥5 sibling decoders and one family of them
(`decode_ai_batch.py`, plus the unlocated script that produced `Rashid_2019`) returns
lowercase tokens while the others return GT casing. The GT side carries only `Soil`,
`Foliar`, `Soil+Foliar`. `join_and_score.py` keys on `treatment_level` exactly, so every
lowercase AI row was structurally unable to pair.

Measured on the deposited AI key table (`00_SPEC/vocab_reference/hui_ai_structural.csv`,
395 rows):

| paper_id | deposited token | rows | case-normalised cell exists on GT side? |
|---|---|---:|---|
| `rashid_2019` | `soil` | 28 | **yes** (GT `rashid_2019`/`Soil`, 19 rows) |
| `mosavian_2021` | `soil` | 8 | no (AI-only paper) |
| `yang_2011` | `foliar` | 8 | no (GT `yang_2011` has only `Soil`) |
| `kalayci_1999` | `soil` | 4 | **yes** (GT `kalayci_1999`/`Soil`, 78 rows) |
| `khoshgoftarmanesh_2013` | `foliar` | 4 | **yes** (GT `khoshgoftarmanesh_2013`/`Foliar`, 40 rows) |
| `pahlavanrad_2009` | `soil` | 2 | **yes** (GT `pahlavanrad_2009`/`Soil`, 2 rows) |
| `pahlavanrad_2009` | `foliar` | 1 | **yes** (GT `pahlavanrad_2009`/`Foliar`, 1 row) |
| **total** | | **55 / 395 (13.9 %)** | **39 rows** sat in a GT-existing cell |

So in the submitted analysis, **55 AI rows (13.9 % of the AI side) could never pair on
casing alone, and 39 of them had a same-paper GT counterpart cell waiting** — a silent
coverage loss, not a data disagreement. `Seed`/`Seed+Foliar` and `''` are a different thing:
they are genuine AI-side surplus, not casing.

**The fix.** `decode_hui.py` emits the GT vocabulary's casing by construction: every method
token goes through `METHOD_MAP` (whose values are `Soil` / `Foliar` / `Soil+Foliar` /
`Seed` / `Seed+Foliar`), and a final guard re-canonicalises any token that is not already in
that casing (counter printed as `treatment_level case-normalised rows`; it is **0** in the
rebuild, i.e. no path can emit a lowercase token).

**Rows recovered.** In the rebuild, the papers that the lowercase-emitting decoders owned
contribute **183 rows** (`kalayci_1999` 82, `rashid_2019` 38, `yang_2011` 29,
`mosavian_2021` 24, `pahlavanrad_2009` 10) — all now in GT casing where the submitted chain
would have written lowercase. Of those, **115 rows land in a (paper_id, treatment_level)
cell that exists on the GT side** and are therefore pairing-eligible instead of structurally
dead: `kalayci_1999`/`Soil` 82, `rashid_2019`/`Soil` 19, `pahlavanrad_2009`/`Soil` 6,
`yang_2011`/`Soil` 5, `pahlavanrad_2009`/`Foliar` 3.

The same count for the `strict` key set is **135 rows** (`kalayci_1999`/`Soil` 82,
`rashid_2019`/`Soil` 38, `pahlavanrad_2009`/`Foliar` 6, `yang_2011`/`Soil` 5,
`pahlavanrad_2009`/`Soil` 4) — higher only because the strict rule mis-codes 19 of
`rashid_2019`'s seed-applied rows as `Soil` (see §10); the casing fix itself is identical in
both sets, which is why `case_normalised` is 0 in both.

*(Headline: 115 rebuilt rows are pairing-eligible thanks to the casing fix in the
`method_field_first` set, 135 in the `strict` set; the equivalent defect cost the submitted
analysis 55 rows, 39 of them in live GT cells. The two numbers
differ because the frozen single-model source yields a different row population per paper —
e.g. `kalayci_1999` 82 rows here vs 4 deposited. Whether a pairing-eligible row ends up
MATCH or AMBIGUOUS is for the orchestrator's join to decide; no join was run here.)*

---

## 7. Vocabulary comparison vs the GT structural reference

GT reference: `00_SPEC/vocab_reference/hui_gt_structural.csv`, 546 rows / 26 papers
(identical to `03_KEYS/gt/hui/*.csv`, 546 rows). Rebuilt AI: 515 rows / 23 papers.

| Key column | Rebuilt AI values | GT values | AI-only (count) | GT-only (count) |
|---|---|---|---|---|
| `outcome_canonical` | `grain_zn` 515 | `grain_zn` 546 | — | — |
| `crop` | `wheat` 515 | `wheat` 546 | — | — |
| `treatment_level` (`method_field_first`) | `Soil` 271, `Foliar` 127, `Soil+Foliar` 74, `Seed` 37, `Seed+Foliar` 5, `''` 1 | `Soil` 314, `Foliar` 158, `Soil+Foliar` 74 | `Seed` 37, `Seed+Foliar` 5, `''` 1 | — |
| `treatment_level` (`strict`) | `Soil` 295, `Foliar` 133, `Soil+Foliar` 74, `Seed` 10, `''` 3 | `Soil` 314, `Foliar` 158, `Soil+Foliar` 74 | `Seed` 10, `''` 3 | — |
| `co_amendment` | `none` 482, `nitrogen` 30, `lime` 3 | `none` 546 | `nitrogen` 30, `lime` 3 | — |
| `co_amendment_level` | `0` 490, `75` 6, `150` 6, `225` 6, `10` 3, `120` 2, `240` 2 | `0` 546 | `75` 6, `150` 6, `225` 6, `10` 3, `120` 2, `240` 2 | — |
| `timepoint` | `''` 515 | `''` 546 | — | — |
| `aggregation_level` | `single_cell` 515 | `single_cell` 546 | — | — |
| `unit_canonical` | `mg/kg` 515 | `mg/kg` 546 | — | — |
| `control_token` (not a key field) | `absolute_control` 515 | `absolute_control` 546 | — | — |
| `is_figure` (not a key field) | `0` 480, `1` 35 | `0` 546 | `1` 35 (`yang_2011` 15, `zulfiqar_2020` 16, `ghasal_2017` 4 — quarantine tier) | — |
| `paper_id` | 23 tokens | 26 tokens | `curtin_2008` 6, `dapkekar_2018` 4, `gomezcoronado_2016` 9, `mosavian_2021` 24, `ram_2016` 48 | `cakmak_1997` 5, `dong_2018` 2, `khoshgoftarmanesh_2013` 60, `kumar_2018` 4, `li_2013` 12, `liu_2014` 10, `zhang_2012` 9, `zhang_2017` 50 |

**Cell-level overlap on (paper_id, treatment_level)** — the only two axes that discriminate
in this dataset after C13/C14:

- shared cells: **33** (357 AI rows ↔ 393 GT rows)
- AI-only cells: **16** (158 AI rows) — `ram_2016`/Foliar 48, `mosavian_2021`/Soil 24,
  `rashid_2019`/Seed 19, `yang_2011`/Foliar 19, `zulfiqar_2020`/Seed 8, `chattha_2017`/Seed 6,
  `curtin_2008`/Soil 6, `yang_2011`/Soil+Foliar 5, `yilmaz_1997`/Seed 4,
  `yilmaz_1997`/Seed+Foliar 4, `dapkekar_2018`/Foliar 4, `gomezcoronado_2016` ×3 cells 9,
  `forster_2018`/Seed+Foliar 1, `pahlavanrad_2009`/`''` 1
- GT-only cells: **17** (153 GT rows) — dominated by the mislabel-excluded papers
  (`khoshgoftarmanesh_2013` 60, `zhang_2017` 50, `li_2013` 12, `liu_2014` 10, `zhang_2012` 9,
  `cakmak_1997` 5, `kumar_2018` 4, `dong_2018` 2) plus `rehman_2018`/Foliar 1.

Per the spec this diagnostic is **not** to be "fixed" by forcing values to match, and it has
not been.

### Crop scope note

GT `crop` is `wheat` on all 546 rows. The deposited AI side carried `rice` 21, `maize` 6,
`barley` 3 — genuine scope expansion, not errors, and the decoder here still decodes
rice/barley/maize from `species`. In this rebuild, however, **0 non-wheat rows survive**: the
frozen corpus's only non-wheat sources are `Li_2013` (Oryza, 23 records) and `Liu_2014`
(Hordeum, 2 records), and **both are on the mislabelled-PDF exclusion list**; the frozen
corpus contains no maize at all (the deposited `wang_2012` maize rows and `rashid_2019` rice
rows come from the multi-model consensus extraction, which read those multi-site papers
differently). Net: rebuilt `crop` = `wheat` 515/515, so no AI-only crop cells arise. Had the
non-wheat papers not been mislabel-excluded, 25 rows would have fallen out as AI-only cells.

---

## 8. Open items (unresolved, stated plainly)

1. **Sibling-decoder unification is a real deviation.** The submitted AI side is a union of
   ≥5 scripts with divergent rules, one of which is no longer in the repository
   (`Rashid_2019`, `Zhang_2017`). One uniform decoder cannot reproduce all of them row for
   row. I chose the named base plus the documented union (§2); a different reviewer could
   defensibly choose a stricter base and would get a somewhat different row population.
2. **Method-field-first decoding (C6) is the largest logic change — now bounded, not open.**
   It is better grounded than free-text parsing and demonstrably fixes real mis-parses, but the
   submitted `gen_ai_keys.py` parsed descriptors, so it is a genuine improvement rather than a
   like-for-like port. Its entire effect is now measured and isolated in the `strict` key set:
   **40 of 515 rows (7.8 %)**, in 7 papers. See §10. All affected rows are traceable via
   `evidence` (`app_type_src:` records which rule fired, and `variant:` which set the row
   belongs to).
3. **`forster_2018`, 1 row → `Seed+Foliar`.** The source's own
   `zn_application_method` is `"foliar + seed"` for the row
   "KCl + Zn-EDTA, KCl 9 kg Cl/ha with seed + foliar Zn-EDTA 1.1 kg Zn/ha Feekes 10". Reading
   the descriptor, the *seed*-placed component is the KCl, not the Zn, so this is arguably
   `Foliar`. I honoured the source's structural label rather than second-guessing it; the row
   consequently becomes an AI-only cell. Flagged, not silently re-labelled.
4. **`pahlavanrad_2009`, 1 row with blank `treatment_level`.** `zn_method = "none"`
   ("Zn 0, Fe 1% foliar" — the Fe-only cell of a Zn × Fe factorial). Emitted with a blank
   token and counted; it cannot pair, correctly.
5. **`mg/100g` — RESOLVED (C18), and no row was ambiguous.** The ×10 conversion is now
   applied in both key sets. **No row was left unconverted on ambiguity grounds**, and the
   ambiguity assessment was made from structural evidence only: all 16 records in
   `59_Khoshgoftarmanesh_2013` carry the single identical unit string `mg/100g`, come from a
   single source locator (`Figure 3`), and the source file's own `notes` field states verbatim
   *"Unit: mg Zn per 100g grain"* — an explicit, unanimous declaration, so there is nothing to
   adjudicate. I did **not** cross-check the conversion against the magnitude of the means:
   `unit_canonical` is a match-key field and spec rule 1 forbids conditioning a key field on
   any mean, so a "does this look like mg/kg?" test would have been a rule-1 violation. Net
   effect on output: **zero rows**, because that paper is mislabel-excluded (see §5). Residual
   caveat: all 16 rows are figure-estimated (`is_figure=1`, quarantine tier) rather than
   table-read, so if the paper is ever reinstated they should enter the figure tier, not the
   headline.
6. **`brown rice` tissue (13 records, `Li_2013`) is dropped** by the strict `tissue == "grain"`
   test inherited from the submitted decoders, even though brown rice *is* the grain. Moot
   here (`li_2013` is mislabel-excluded), but it is a latent scope bug in the inherited rule.
7. **Where the 8-paper exclusion is applied.** `build_hui_v4.py` excluded 1 of 8 at key-build
   time and left the other 7 to the analysis scripts; this rebuild applies all 8 at key-build
   time, so `03_KEYS/ai_rebuilt/hui/` contains no file for them. Downstream numbers are
   unaffected (every submitted analysis script carries the identical 8-token `EXCLUDE`), but
   anyone diffing file lists against the deposited `runs/hui_v4/keys/ai/` will see 7 fewer
   files. Row counts that *would* have been emitted for them are in §5.
8. **`co_amendment = nitrogen` on 30 rows** (`mosavian_2021` 24, `yang_2011` 6) makes those
   rows unpairable against a GT side that is `none`/`0` throughout. This faithfully ports
   `decode_ai_batch.py`'s rule to the frozen field names; the alternative (ignoring the N
   axis) would merge N levels into one key and produce same-side duplicate keys, which the
   protocol treats as AMBIGUOUS. Neither route yields a clean match, so the choice is
   presentational, but it is a choice.
9. **`torun_2001` yields 0 rows** (all 15 frozen records are shoot Zn). The paper title is
   "…Grain Yield and **Shoot** Concentrations of Zinc…", so this is correct scope behaviour,
   but it means an AI-only paper contributes nothing.
10. **`curtin_2008` co-amendment asymmetry.** The 3 lime-treated grain rows carry a
    `lime_rate` moderator and so decode as `lime`/`10`, while the 3 no-lime rows carry no such
    moderator and decode as `none`/`0` — the same axis labelled two ways within one paper. The
    inherited rule keys on moderator *presence*, so it cannot see the "no lime" arm as lime
    level 0. `curtin_2008` is AI-only (no GT counterpart), so nothing downstream changes, but
    the rule would need a descriptor fallback to be correct in general. Separately, these
    Curtin rows are a Cu+Mn+Zn micronutrient mix against a no-micronutrient control, which
    protocol §4 arguably codes `mineral_mix` rather than `absolute_control`; the inherited
    keyword list assigns `absolute_control`.

---

## 9. Reproduction

```
python 02_DECODERS/hui/decode_hui.py
```
One invocation writes **both** key sets and prints both record-arithmetic reports plus the
variant diff. Pure stdlib, no arguments, paths resolved relative to the script.

Two consecutive runs produced byte-identical output for all 29 CSVs in **each** set (per-file
SHA-256 unchanged):

| Key set | Aggregate SHA-256 over (filename + file digest), sorted |
|---|---|
| `03_KEYS/ai_rebuilt_strict/hui/` | `117a582c621519955681f452c14699a5a51cd0b4df9180f4f327d2b9400a054b` |
| `03_KEYS/ai_rebuilt/hui/` | `8faf8e5b79261cf5be93f3f23c5043a6e2c7284dae8e82e92b5c98e00eefabca` |

Column-wise diff between the two sets: `treatment_level` differs on 40 rows, `evidence`
differs on all 515 (it records `variant:` and the rule that fired); **all 16 other columns are
identical on all 515 rows**, and the `row_id` sets are equal. Regression check against the
pre-refactor decoder: zero differences across all 17 key/scored columns on 515/515 rows.

No GT file, no deposited-key value, and no analysis script is read at runtime.

---

## 10. Bounding the `treatment_level` change — `strict` vs `method_field_first`

Purpose: the rebuild is designed so that exactly one variable changes versus the submission
(the AI-side source). The method-field rule (C6) would have been a second, unquantified change
riding along, making a moved number unattributable. Both variants are therefore emitted, and
they differ in **nothing** but the `treatment_level` rule.

- `strict` = `treatment_description` only, a literal port of `gen_ai_keys.py::app_type`
  (its exact keyword sets, precedence and kg/ha fallback), plus the casing guard.
- `method_field_first` = the record's own explicit Zn-application-method field when present,
  with the four-sibling descriptor union as fallback, plus the casing guard.

Verified: same 515 `row_id`s, same values in all 16 other columns, both byte-stable on re-run.

### How many rows differ: **40 / 515 (7.8 %)**, in 7 papers

| Paper | Rows differing | Rows in that paper | GT counterpart? |
|---|---:|---:|---|
| `rashid_2019` | 19 | 38 | yes |
| `yilmaz_1997` | 8 | 20 | yes |
| `zulfiqar_2020` | 4 | 16 | yes |
| `curtin_2008` | 3 | 6 | no (AI-only paper) |
| `pahlavanrad_2009` | 3 | 10 | yes |
| `dapkekar_2018` | 2 | 4 | no (AI-only paper) |
| `forster_2018` | 1 | 11 | yes |

### By transition (strict → method_field_first)

| Transition | Rows | Driver |
|---|---:|---|
| `Soil` → `Seed` | 27 | `gen_ai_keys.py`'s `has_seed` test matches only `"seed priming"` / `"priming"`, so it misses `"Seed treatment 30% ZnSO4 solution"`, `"Zn seed coating (1.25 g Zn/kg seed)"` and `"High-Zn biofortified seeds (no soil Zn)"`; those rows fall through to its `digit + zn ⇒ Soil` fallback |
| `Foliar` → `Seed+Foliar` | 5 | same missed `seed` keyword, in rows that also say `foliar` (`38_Yilmaz_1997` "Seed 30% ZnSO4 + Foliar 0.4% ZnSO4") |
| `<blank>` → `Soil` | 3 | `curtin_2008` "Cu+Mn+Zn applied, ±lime" has no method word and no digit, so the strict fallback yields `""`; the method field says `soil` |
| `Foliar` → `Soil` | 2 | `pahlavanrad_2009` "40 / 80 kg ZnSO4/ha, **Fe 1% foliar**" — the descriptor's `foliar` belongs to the Fe co-factor, not the Zn; `zn_method` says `soil` |
| `Soil` → `Foliar` | 2 | `dapkekar_2018` "Urea + Zn-CNP nanoparticles (40 / 4 mg/L Zn)" — a mg/L spray with no method word; `application_method` says `foliar` |
| `Foliar` → `<blank>` | 1 | `pahlavanrad_2009` "Zn 0, Fe 1% foliar" — the Fe-only cell of a Zn × Fe factorial; `zn_method` is `none`, i.e. no Zn was applied |

### Pairing consequence — the part that matters for choosing a headline

Cell overlap against the GT structural reference on `(paper_id, treatment_level)`:

| Key set | Shared cells | AI rows in shared cells | AI-only cells | AI rows in AI-only cells |
|---|---:|---:|---:|---:|
| `strict` | 33 | **390** | 13 | 125 |
| `method_field_first` | 33 | **357** | 16 | 158 |

Of the 40 differing rows:

- **35 land in a GT-existing cell under `strict`** — `rashid_2019`/`Soil` 19,
  `yilmaz_1997`/`Soil` 4, `yilmaz_1997`/`Foliar` 4, `zulfiqar_2020`/`Soil` 4,
  `pahlavanrad_2009`/`Foliar` 3, `forster_2018`/`Foliar` 1.
- **2 land in a GT-existing cell under `method_field_first`** — `pahlavanrad_2009`/`Soil` 2.

So `strict` shows ~33 more rows sitting in pairable cells. **Those extra rows are
mis-coordinated, and this can be shown without looking at a single mean.** In 33 of the 35,
the strict rule assigns a soil- or foliar-only coordinate to a treatment whose own descriptor
says it is seed-applied (`"Seed treatment 30% ZnSO4 solution"`, `"Zn seed coating"`,
`"High-Zn biofortified seeds (no soil Zn)"`, `"Seed 30% ZnSO4 + Foliar 0.4% ZnSO4"`), so they
would pair against GT `Soil` / `Foliar` cells that describe a different intervention. The 2
rows the method-field rule adds are the opposite case: `pahlavanrad_2009`'s soil-applied Zn
rows that the strict rule pushed into `Foliar` because the descriptor's `foliar` belongs to
the Fe co-factor.

**Recommendation, stated as a caveat rather than a decision.** `strict` is the correct
like-for-like baseline and is the right set for the headline *if* the headline must isolate the
source change — but publishing it means importing ~33 known-false pairings that inflate
apparent coverage (390 vs 357 rows in shared cells, +9 %). `method_field_first` is the more
faithful coordinate assignment. The cleanest presentation is probably the headline on `strict`
with this section cited as the sensitivity, plus an explicit note that the `strict` coverage
figure is ~33 rows optimistic for a documented parser reason. Whichever is chosen, the
difference is now measured rather than entangled with the source change. No equivalence, TOST
or fidelity computation was run here — that is the orchestrator's step.
