# Pipeline V2: Prospective Topic Recommendations

## Date
2026-03-25

## Selection Methodology

18 candidate agricultural meta-analysis topics were scored on 8 dimensions (each 1-5):

1. **Estimand clarity** — Is the primary outcome unambiguous?
2. **Intervention clarity** — Is the intervention binary and well-defined?
3. **Comparator clarity** — Is the control condition obvious?
4. **Setting consistency** — Are study settings homogeneous?
5. **OA feasibility** — Can we access >60% of primary literature via OA?
6. **Benchmark richness** — Does the benchmark report subgroups, forest plots, obs counts?
7. **Moderator extractability** — Can moderators be reliably extracted from abstracts/text?
8. **Low estimand trap risk** — Is there a risk of measuring the wrong thing?

Minimum threshold for V2 inclusion: total >= 33 AND no dimension scored 1 or 2.

---

## Tier 1: Recommended for V2 (score >= 34, no weak dimensions)

| Rank | Topic | Score | Benchmark | Published Effect | Key Strength |
|------|-------|-------|-----------|-----------------|--------------|
| 1 | legume_rotation_yield | 37 | Zhao et al. 2022 (Nat Comm) | +20% | Already V1-validated; massive OA corpus; Nature Comms fully OA |
| 2 | elevated_co2_face_yield | 36 | Ainsworth & Long 2021 (GCB) | variable by crop | Gold-standard FACE; well-known corpus; clear estimand |
| 3 | amf_inoculation_yield_rainfed | 35 | Wu et al. 2022 (PeerJ) | +23% [16-30%] (k=21, n=546) | PeerJ fully OA; rainfed only; clear AMF vs non-AMF |
| 4 | biochar_tropical_yield | 35 | Jeffery et al. 2017 (ERL) | +25% (tropics) | ERL fully OA; binary comparison; tropical subgroup clean |
| 5 | zn_biofortification_wheat | 35 | Wang et al. 2025 (Nat Comm) | ~+30% Zn conc | OA; very clear intervention; Zn concentration primary |
| 6 | humic_acid_yield | 34 | Ma et al. 2024 (Agronomy MDPI) | +12% (k=93) | MDPI fully OA; very clear intervention; clean estimand |
| 7 | cover_crop_corn_yield | 34 | Marcillo & Miguez 2017 (JSWC) | -1% to +3% | Clear subsequent-crop yield; US/Canada focus |

---

## Tier 2: Acceptable but with known risks (score 30-33)

| Rank | Topic | Score | Risk |
|------|-------|-------|------|
| 8 | plastic_mulch_yield | 33 | OA feasibility = 2 (Elsevier + Chinese journals) |
| 9 | biological_seed_treatment | 31 | Broad intervention (many organisms); intervention clarity = 3 |
| 10 | notill_vs_conventional | 31 | Already V1-tested; intervention definition drifts |
| 11 | deficit_irrigation_veg | 30 | Continuous intervention (deficit level varies) |

---

## Tier 3: Not recommended for V2 (score < 30 or fatal weakness)

| Topic | Score | Fatal weakness |
|-------|-------|----------------|
| organic_yield_gap | 28 | Intervention too broad; outcome leakage; already V1-tested |
| intercropping_yield_ler | 27 | Estimand trap (LER vs component yield) scored 1 |
| biostimulant_yield_all | 26 | Intervention too broad (estimand = 3, intervention = 2) |
| salinity_rice_yield | 24 | Continuous stressor; low OA |
| nitrogen_fertilizer_yield | 22 | Continuous intervention; no binary comparison |
| maize_density_yield | 22 | Continuous intervention; no binary comparison |
| compost_amendment_yield | 21 | Broad intervention; low OA; co-primary outcomes |

---

## Recommended V2 Topic Set (6 topics)

For preregistered V2 evaluation, select **6 topics** — 3 prospective (never tested) + 3 carried forward from V1 (with V2 upgrades):

### Prospective (new)
1. **amf_inoculation_yield_rainfed** (score 35) — Clean binary intervention, fully OA benchmark
2. **biochar_tropical_yield** (score 35) — Narrower than V1 biochar (tropics only), fully OA
3. **humic_acid_yield** (score 34) — MDPI fully OA, very clean estimand, never attempted

### Carried forward (V1 → V2 upgrade)
4. **legume_rotation_yield** (score 37) — Best V1 performer; V2 should improve further
5. **elevated_co2_face_yield** (score 36) — Gold-standard benchmark; tests mineral + yield
6. **cover_crop_corn_yield** (score 34) — Clean comparison; moderate OA but US-focused

### Rationale for this set
- **3 fully OA** (amf, biochar_tropical, humic_acid) — tests whether OA access removes the corpus composition barrier
- **3 with moderate OA** (legume, co2, cover_crop) — tests how much the pipeline can achieve with realistic OA rates
- **Mix of effect sizes** — positive (amf +23%, legume +20%, biochar +25%), small/null (cover_crop ±3%), negative (co2 yield varies)
- **Mix of V1 experience** — 3 brand new, 3 with prior results to compare V1→V2 improvement
- **No topic scored below 3 on any dimension** — all are structurally sound

---

## Pilot Topic for V2 Dress Rehearsal

**Recommended pilot: humic_acid_yield**

Reasons:
1. Fully OA benchmark (MDPI) — no access barriers
2. Very clean intervention (humic acid vs no humic acid) — minimal semantic confusion
3. Moderate corpus size (~100-200 papers) — fast to run end-to-end
4. Score 34 with no weak dimensions — representative of V2 target difficulty
5. Never tested before — genuinely prospective
6. Clear estimand (crop yield, %) — low risk of outcome leakage
7. Single benchmark paper (Olk et al. 2024) — straightforward comparison

Alternative pilot: **amf_inoculation_yield_rainfed** (score 35, also fully OA, but larger corpus ~500 papers may be slower for a dress rehearsal).

---

## V2 Success Criteria (preregistered)

### Primary
1. **Direction agreement**: Pipeline pooled effect has same sign as benchmark in >= 5/6 topics
2. **CI overlap**: Pipeline 95% CI includes benchmark point estimate in >= 3/6 topics

### Secondary
3. **Absolute difference**: |pipeline - benchmark| < 10 pp in >= 4/6 topics
4. **Benchmark-aligned subset**: Aligned subset improves agreement vs full sample
5. **V1→V2 improvement**: Carried-forward topics show smaller |gap| in V2 than V1
6. **Failure taxonomy**: All disagreements can be diagnostically classified

### Process Metrics
7. **Download coverage**: >= 50% of screened papers successfully downloaded
8. **Extraction completeness**: >= 80% of downloaded papers yield >0 extracted rows
9. **Adjudication retention**: 40-70% of extracted rows pass LLM adjudication
10. **Variance coverage**: >= 30% of kept rows have usable variance for weighting

---

## Next Steps

1. Write benchmark specs for the 6 selected topics (using BENCHMARK_SPEC_TEMPLATE.md)
2. Write topic configs (JSON) for the 3 new prospective topics
3. Run V2 dress rehearsal on humic_acid_yield
4. Preregister V2 evaluation (freeze topic set + success criteria)
5. Run V2 on all 6 topics
6. Analyze results and write V2 paper
