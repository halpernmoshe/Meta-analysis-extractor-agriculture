# Benchmark Spec: Elevated CO2 Effects on Crop Yield (FACE)

## 1. Benchmark Identity

### Primary Benchmark

- benchmark_citation: Long et al. 2006
- benchmark_title: Food for Thought: Lower-Than-Expected Crop Yield Stimulation with Rising CO2 Concentrations
- benchmark_year: 2006
- benchmark_journal: Science
- benchmark_doi: 10.1126/science.1114722
- benchmark_pmid: 16809532
- benchmark_volume_pages: vol 312(5782):1918-1921, June 30, 2006
- benchmark_authors: Long SP, Ainsworth EA, Leakey ADB, Nosberger J, Ort DR
- benchmark_open_access: yes (PubMed PMID 16809532)

### Secondary Benchmark (Retained as Reference)

- benchmark_secondary_citation: Ainsworth & Long 2021
- benchmark_secondary_title: 30 years of free-air carbon dioxide enrichment (FACE): What have we learned about future crop productivity and its potential for adaptation?
- benchmark_secondary_year: 2021
- benchmark_secondary_journal: Global Change Biology
- benchmark_secondary_doi: 10.1111/gcb.15375
- benchmark_secondary_pmid: 33135850
- benchmark_secondary_volume_pages: vol 27(1):27-49, January 2021
- benchmark_secondary_open_access: no (Wiley GCB, closed access; CI not obtainable)

## 2. Core Research Question

- benchmark_question: How much does elevated CO2 increase crop yield in FACE field experiments, specifically for C3 grain cereals?
- target_estimand: Crop grain yield or aboveground biomass under elevated vs ambient CO2 in FACE experiments
  - Primary focus on C3 cereals (wheat, rice) and legumes (soybean)
  - C4 crops (maize, sorghum) expected ~0% response
- benchmark_key_finding: FACE experiments show ~50% less yield stimulation than prior enclosure studies; C3 cereals in FACE approximately +8%; soybean approximately +13%

## 3. Intervention Definition

- intervention_label: Elevated atmospheric CO2
- intervention_required_features:
  - Experimentally elevated CO2 concentration (typically 500-700 ppm)
  - FACE, OTC, or controlled environment chamber
  - Treatment maintained for significant portion of growing season
- intervention_excluded_variants:
  - Natural CO2 springs or volcanic vents
  - Short-term (<1 week) CO2 pulse experiments
  - CO2 applied to soil only (not atmospheric enrichment)
- intervention_ambiguity_notes:
  - Both FACE and OTC are acceptable
  - Chamber experiments acceptable but label separately
  - CO2 x temperature or CO2 x drought factorials: extract CO2 effect at each level

## 4. Comparator Definition

- comparator_label: Ambient CO2
- comparator_required_features:
  - Same crop, same field/chamber
  - Ambient atmospheric CO2 (~370-420 ppm depending on year of study)
- comparator_excluded_variants:
  - Sub-ambient CO2 treatments
- comparator_ambiguity_notes:
  - In FACE, "ambient" means the surrounding unmodified atmosphere
  - In OTC, ambient control may be an open-top chamber without CO2 addition

## 5. Outcome Definition

- primary_outcome_label: Crop grain yield
- acceptable_primary_outcomes:
  - Grain yield (kg/ha, g/plant, g/m2)
  - Seed yield
  - Total aboveground biomass (if grain yield not reported)
- excluded_outcomes:
  - Photosynthesis rate (Asat, Vcmax)
  - Stomatal conductance
  - Mineral/nutrient concentration (mg/g)
  - Water use efficiency
  - Leaf area index
  - Root biomass
  - Respiration rate
- acceptable_units: kg/ha, g/plant, g/m2, Mg/ha, t/ha
- outcome_hierarchy:
  1. Grain/seed yield
  2. Total aboveground biomass
  3. Dry matter per plant

## 6. Study Setting

- allowed_settings:
  - field (FACE, OTC)
  - greenhouse (controlled CO2 chambers)
- excluded_settings:
  - Growth chamber with unrealistic conditions
  - In-vitro
- setting_notes:
  - FACE is gold standard for realistic field conditions
  - OTC acceptable but noted as slightly different environment
  - Greenhouse/chamber: include but label separately

## 7. Study Design / Eligibility

- included_study_types:
  - FACE experiments
  - Open-top chamber (OTC) experiments
  - Greenhouse CO2 enrichment
- excluded_study_types:
  - Reviews, meta-analyses
  - Modeling/simulation
  - Natural CO2 gradient studies
- special_design_rules:
  - FACE typically has n=3-4 rings — true replicates are rings, not plants
  - Multi-year FACE: extract each year separately if yields differ

## 8. Critical Moderators

- moderator_1:
  - name: Photosynthetic pathway (C3 vs C4)
  - why_it_matters: C3 crops respond strongly, C4 crops show ~0% yield response
  - expected_levels: C3, C4
  - whether_required_for_alignment: yes (C3-only for primary benchmark comparison)

- moderator_2:
  - name: Crop species
  - why_it_matters: Wheat, rice, soybean may differ in CO2 response
  - expected_levels: wheat, rice, soybean, maize, sorghum, barley, cotton, potato
  - whether_required_for_alignment: no

- moderator_3:
  - name: CO2 concentration
  - why_it_matters: Higher CO2 = larger effect
  - expected_levels: 475-525 ppm, 525-575 ppm, 575-625 ppm, >625 ppm
  - whether_required_for_alignment: no

- moderator_4:
  - name: Experiment type
  - why_it_matters: FACE more realistic than OTC/chamber
  - expected_levels: FACE, OTC, greenhouse_chamber
  - whether_required_for_alignment: yes (FACE-only for strictest benchmark comparison)

## 9. Benchmark Subgroup Logic

### Primary Benchmark (Long et al. 2006)

- subgroup_1:
  - definition: C3 grain cereals (wheat, rice), FACE experiments only
  - reported_effect: approximately +8% yield increase at ~+200 ppm CO2 (FACE)
  - ci_95: Not formally reported; Long et al. 2006 is a perspective/synthesis article, not a formal statistical meta-analysis with computed CIs. The commonly cited FACE range for cereals is approximately +5% to +13%.
  - notes: PRIMARY benchmark target for P2 CI overlap evaluation. Range approximately +5-13% serves as the informal CI for benchmark comparison.

- subgroup_2:
  - definition: C3 legumes (soybean), FACE experiments only
  - reported_effect: approximately +13% yield increase in FACE
  - notes: Higher response than cereals. If pipeline includes soybeans, pooled estimate will be above +8%.

- subgroup_3:
  - definition: C4 crops (maize, sorghum), FACE experiments
  - reported_effect: ~0% yield change (no increase)
  - notes: Important negative control — pipeline should replicate null effect. Matches Long et al. 2006 finding.

- subgroup_4:
  - definition: All C3 crops pooled, FACE and enclosure combined (prior enclosure estimates)
  - reported_effect: ~+15-17% (enclosure estimates, which Long et al. 2006 shows are approximately double the FACE results)
  - notes: These are the pre-FACE enclosure estimates; NOT the appropriate benchmark for this pipeline (FACE-only restriction).

### Secondary Benchmark (Ainsworth & Long 2021, retained for reference)

- subgroup_1_secondary:
  - definition: C3 crops only, FACE experiments (all C3 crops pooled, 2021 update)
  - reported_effect: ~+18% yield at ~+200 ppm CO2 under non-stress conditions (k=186 independent studies of 18 C3 crop species, 14 FACE sites, 5 continents)
  - ci_95: Not reported in abstract; full text closed access (Wiley GCB). CI not obtainable.
  - notes: SECONDARY reference only. Higher than Long 2006 because it includes post-2006 data and all C3 crops (legumes + root crops + cereals pooled).

- subgroup_2_secondary:
  - definition: By crop functional group (cereals vs legumes vs root crops) — Ainsworth & Long 2021
  - reported_effect: Legumes and root crops > cereals. Rice highest-potential cultivars: +35% vs. average +14%.
  - notes: Confirms cereal FACE effect is below the 18% headline.

## 10. Known Estimand Traps

- trap_1: Extracting photosynthesis parameters instead of yield (very common in CO2 literature)
- trap_2: Extracting mineral concentration changes (different construct than yield)
- trap_3: Confusing biomass with grain yield (CO2 may increase biomass more than grain)
- trap_4: Pooling C3 and C4 crops (dilutes the C3 yield effect toward zero)
- trap_5: FACE replication is RINGS not PLANTS — n=3 or n=4, not n=hundreds

## 11. Prompt Consequences

- extraction_priority_1: Search for "yield" or "grain" in table headers — NOT photosynthesis or gas exchange
- extraction_priority_2: Record CO2 concentration level and whether FACE/OTC/chamber
- extraction_priority_3: Record C3/C4 status of crop (critical moderator)

## 12. Post-Processing Consequences

- keep_rules:
  - Keep if outcome is crop yield AND treatment is elevated CO2 AND control is ambient CO2
- exclude_rules:
  - Exclude if outcome is photosynthesis, stomatal conductance, or gas exchange
  - Exclude if outcome is mineral concentration
  - Exclude if no ambient CO2 control
  - Exclude if study duration < 1 month
- flag_rules:
  - Flag if n seems unrealistically large for FACE (likely plant-level, not ring-level)
  - Flag if CO2 x stress factorial and stress treatment is extreme
- benchmark_alignment_labels:
  - benchmark_aligned: FACE, C3 crop, grain yield, ~550 ppm
  - partially_aligned: OTC or chamber, or total biomass instead of grain
  - misaligned: C4 crop pooled with C3, photosynthesis parameter, mineral concentration

## 13. Provenance

- who_created_spec: Claude (autonomous)
- date_created: 2026-03-25
- updated_1: 2026-03-26 (DOI corrected, exact title confirmed, benchmark numbers updated from abstract)
- updated_2: 2026-03-26 (PRIMARY BENCHMARK CHANGED — Ainsworth & Long 2021 replaced by Long et al. 2006 as primary; Ainsworth & Long 2021 retained as secondary reference)
- created_before_results_seen: yes (no V1 replication of this specific topic)
- reason_for_primary_change: Long et al. 2006 (Science 312:1918, PMID 16809532) is FACE-specific (matches pipeline FACE-only scope restriction), open-access via PubMed, and the canonical reference for FACE cereal yield effects. Ainsworth & Long 2021 is closed access (Wiley GCB) and its 95% CI is not obtainable, preventing formal P2 evaluation. Long et al. 2006 provides an obtainable benchmark range (~+5-13% FACE cereals) enabling CI overlap assessment. Change made before any pipeline data collection for this topic.
- notes: Long et al. 2006 is a perspective/synthesis article (Science 312:1918-1921), not a formal statistical meta-analysis. It does not report formal 95% CIs, but the implied range for FACE C3 cereals (~+5-13%) is derivable from the text and widely cited in the literature. Elevated CO2 FACE literature is one of the best-studied areas in plant science.

## 14. Scope Restriction (Pre-registered)

Scope restriction: FACE experiments only. OTC and chamber studies are excluded from this pipeline topic to match the benchmark's FACE-only scope. This is a pre-registered design decision.

Rationale: OTC (open-top chambers) are known to give systematically higher CO2 response effects than FACE because of less turbulent gas mixing and modified microclimate inside the chamber. Including OTC studies would bias the pooled estimate upward relative to the Ainsworth & Long (2021) FACE-only benchmark. Growth chambers and greenhouses are excluded for the same reason — they produce non-field-realistic conditions. Only FACE experiments (outdoor field plots with free-air CO2 fumigation rings) are eligible for this topic.
