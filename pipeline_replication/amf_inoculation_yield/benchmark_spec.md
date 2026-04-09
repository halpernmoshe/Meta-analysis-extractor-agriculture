# Benchmark Spec: AMF Inoculation Effects on Crop Yield

## 1. Benchmark Identity

- benchmark_citation: Wu et al. 2022
- benchmark_title: Arbuscular mycorrhizal fungi increase crop yields by improving biomass under rainfed condition: a meta-analysis
- benchmark_year: 2022
- benchmark_journal: PeerJ
- benchmark_doi: 10.7717/peerj.12861
- benchmark_k: 21 articles, 546 paired observations
- benchmark_scope: RAINFED conditions only (non-irrigated)

## 2. Core Research Question

- benchmark_question: Does AMF inoculation increase crop yield compared to non-inoculated controls?
- target_estimand: Direct crop yield (grain, shoot biomass, fruit, tuber)
  - NOT mycorrhizal colonization rate, root parameters, or nutrient uptake
  - Yield measured as mass per unit area or mass per plant

## 3. Intervention Definition

- intervention_label: AMF inoculation
- intervention_required_features:
  - Deliberate inoculation with arbuscular mycorrhizal fungi (Glomeromycota)
  - Any AMF species: Glomus, Rhizophagus, Funneliformis, Claroideoglomus, Gigaspora, etc.
  - Single or mixed-species inoculants
  - Commercial or laboratory-produced inoculum
- intervention_excluded_variants:
  - Ectomycorrhizal fungi (ECM) — different phylum, different mechanism
  - Native AMF diversity studies without inoculation treatment
  - Mycorrhizal helper bacteria alone (without AMF)
  - Studies of AMF colonization without yield measurement
- intervention_ambiguity_notes:
  - "Biofertilizer" products may contain AMF + bacteria — include only if AMF is the primary component
  - Some studies use sterilized soil to eliminate native AMF — acceptable (common design)

## 4. Comparator Definition

- comparator_label: Non-inoculated control
- comparator_required_features:
  - Same crop, same soil, same fertilization regime
  - No AMF inoculant added
  - May receive sterilized inoculant carrier (mock inoculation) or nothing
- comparator_excluded_variants:
  - Comparison between two different AMF species (without non-AMF control)
  - Comparison between AMF and another biofertilizer
- comparator_ambiguity_notes:
  - In factorial designs (AMF x P), use AMF vs no-AMF at the SAME P level
  - Some controls have native AMF present — this is acceptable (reflects real conditions)

## 5. Outcome Definition

- primary_outcome_label: Crop yield
- acceptable_primary_outcomes:
  - Grain yield (kg/ha, g/plant)
  - Shoot dry biomass (g/plant, kg/ha)
  - Fruit yield (g/plant, kg/ha)
  - Tuber yield (g/plant, kg/ha)
  - Total aboveground biomass
- excluded_outcomes:
  - Mycorrhizal colonization percentage
  - Root colonization intensity
  - Spore density
  - Root length, root biomass, root:shoot ratio
  - Nutrient concentration (N, P, K, Zn)
  - Nutrient uptake (mg/plant)
  - Chlorophyll content, photosynthesis rate
  - Plant height (unless sole yield proxy in tropical smallholder context)
  - Stomatal conductance
  - Soil enzyme activity
- acceptable_units: kg/ha, g/plant, g/pot, Mg/ha, t/ha
- outcome_hierarchy:
  1. Grain/seed yield per unit area
  2. Fruit/tuber yield per unit area
  3. Shoot biomass per plant
  4. Total dry matter per plant

## 6. Study Setting

- allowed_settings:
  - field
  - greenhouse
  - pot
- excluded_settings:
  - in-vitro / tissue culture
  - hydroponic (AMF requires soil/substrate)
- setting_notes:
  - Field studies preferred for benchmark alignment
  - Pot experiments very common in AMF literature — include but label
  - Greenhouse studies intermediate between field and pot

## 7. Study Design / Eligibility

- included_study_types:
  - RCBD, CRD, factorial, split-plot
  - Field trials, pot experiments, greenhouse trials
- excluded_study_types:
  - Reviews, meta-analyses
  - Observational surveys of natural AMF
  - Modeling studies
- special_design_rules:
  - AMF x P factorial: extract AMF vs no-AMF at each P level separately

## 8. Critical Moderators

- moderator_1:
  - name: AMF species
  - why_it_matters: Different species have different efficacy
  - expected_levels: Glomus mosseae, Rhizophagus irregularis, Funneliformis mosseae, mixed, etc.
  - whether_required_for_alignment: no

- moderator_2:
  - name: Host crop type
  - why_it_matters: Crop dependency on AMF varies (cereals < legumes < vegetables typically)
  - expected_levels: grain_cereal, legume, vegetable, root_tuber, tree_crop
  - whether_required_for_alignment: no

- moderator_3:
  - name: Soil P availability
  - why_it_matters: AMF benefits greatest at low soil P; effect diminishes at high P
  - expected_levels: low (<10 mg/kg), medium (10-25 mg/kg), high (>25 mg/kg)
  - whether_required_for_alignment: no

- moderator_4:
  - name: Study setting
  - why_it_matters: Pot experiments may overestimate field-scale AMF benefits
  - expected_levels: field, greenhouse, pot
  - whether_required_for_alignment: yes (field-only for benchmark-aligned subset)

## 9. Benchmark Subgroup Logic

- subgroup_1:
  - definition: Field experiments only
  - reported_effect: To be confirmed from Wu et al. 2022
  - notes: Expected to show lower effect than pot studies

- subgroup_2:
  - definition: By crop type (cereal vs legume vs vegetable)
  - reported_effect: To be confirmed
  - notes: Vegetables/legumes expected to show larger AMF response

- subgroup_3:
  - definition: By soil P level (low vs high)
  - reported_effect: To be confirmed
  - notes: Low-P soils expected to show stronger AMF benefit

## 10. Known Estimand Traps

- trap_1: Extracting colonization rates instead of yield — colonization is the mechanism, not the outcome
- trap_2: Including ectomycorrhizal studies (ECM fungi are different from AMF)
- trap_3: Confusing P-fertilizer effect with AMF effect in factorial designs
- trap_4: Extracting nutrient uptake (mg P/plant) as if it were yield
- trap_5: Including studies where "AMF" refers to natural colonization observation, not inoculation treatment

## 11. Prompt Consequences

- extraction_priority_1: Search for "yield", "biomass", "dry weight", "harvest" in table headers
- extraction_priority_2: When colonization AND yield are both reported, ONLY extract yield rows
- extraction_priority_3: Record AMF species name from Methods section

## 12. Post-Processing Consequences

- keep_rules:
  - Keep if outcome is crop yield/biomass AND treatment is AMF inoculation AND control is non-inoculated
- exclude_rules:
  - Exclude if outcome is colonization, root traits, nutrient concentration
  - Exclude if treatment is ECM or other non-AM fungi
  - Exclude if no proper non-inoculated control
- flag_rules:
  - Flag if AMF is part of multi-organism inoculant
  - Flag if sterilized vs non-sterilized soil design (may inflate apparent AMF effect)
- benchmark_alignment_labels:
  - benchmark_aligned: field experiment, crop yield, AMF inoculation vs non-inoculated
  - partially_aligned: pot/greenhouse, or biomass per plant
  - misaligned: colonization, nutrient uptake, or no inoculation treatment

## 13. Provenance

- who_created_spec: Claude (autonomous)
- date_created: 2026-03-25
- created_before_results_seen: yes
- notes: +23% [16%, 30%], 21 articles, 546 obs. RAINFED ONLY scope. Subgroups: wheat +34%, chickpea +18%, N-fixing +29%, non-N-fixing +20%. 13 crop species. lnRR as effect metric.
