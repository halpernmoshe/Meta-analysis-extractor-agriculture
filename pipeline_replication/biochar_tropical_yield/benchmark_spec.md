# Benchmark Spec: Biochar Effects on Crop Yield in Tropical Soils

## 1. Benchmark Identity

- benchmark_citation: Jeffery et al. 2017
- benchmark_title: Biochar boosts tropical but not temperate crop yields
- benchmark_year: 2017
- benchmark_journal: Environmental Research Letters
- benchmark_doi: 10.1088/1748-9326/aa67bd
- benchmark_k: ~50-80 studies (exact count to confirm from paper)

## 2. Core Research Question

- benchmark_question: Does biochar soil amendment increase crop yield, and does the effect differ between tropical and temperate regions?
- target_estimand: Direct crop yield (grain, fruit, tuber, biomass)
  - Focus on TROPICAL subset for this replication
  - NOT soil properties, carbon sequestration, or nutrient retention

## 3. Intervention Definition

- intervention_label: Biochar soil amendment
- intervention_required_features:
  - Application of biochar (pyrolyzed biomass) to soil
  - Any feedstock: wood, crop residue, manure, sewage sludge, etc.
  - Any pyrolysis temperature
  - Any application rate
- intervention_excluded_variants:
  - Activated carbon (different product, different mechanism)
  - Hydrochar (hydrothermal carbonization, not pyrolysis)
  - Charcoal used as fuel or heating, not soil amendment
  - Biochar used only as growing medium (soilless/hydroponic)
- intervention_ambiguity_notes:
  - "Charcoal" in some papers refers to biochar — include if applied to soil for crop growth
  - Biochar + compost combinations: include if biochar effect can be isolated (vs compost-only control)

## 4. Comparator Definition

- comparator_label: Unamended control (no biochar)
- comparator_required_features:
  - Same crop, same soil, same base fertilization
  - No biochar added
- comparator_excluded_variants:
  - Comparison between two biochar rates without a zero control
  - Comparison between biochar and compost (without no-amendment control)
- comparator_ambiguity_notes:
  - In factorial designs (biochar x NPK), use biochar+NPK vs NPK (isolate biochar)
  - "CK" in Chinese papers typically means unamended control

## 5. Outcome Definition

- primary_outcome_label: Crop yield
- acceptable_primary_outcomes:
  - Grain yield (kg/ha, t/ha)
  - Fruit/tuber yield (kg/ha)
  - Total aboveground biomass (if harvestable portion)
  - Dry matter yield
- excluded_outcomes:
  - Soil organic carbon
  - Soil pH, CEC, nutrient availability
  - Soil respiration, microbial biomass
  - Root biomass or root length
  - Plant height (growth parameter, not yield)
  - Nutrient concentration in plant tissue
  - Greenhouse gas emissions
- acceptable_units: kg/ha, Mg/ha, t/ha, g/plant, g/pot
- outcome_hierarchy:
  1. Grain/seed yield per unit area
  2. Fruit/tuber yield per unit area
  3. Total dry biomass per unit area
  4. Yield per plant/pot (if area-based not available)

## 6. Study Setting

- allowed_settings:
  - field (preferred for benchmark alignment)
  - greenhouse
  - pot
- excluded_settings:
  - hydroponic / soilless
  - remediation-only studies (no crop growth)
- setting_notes:
  - Field-only subset for benchmark-aligned analysis
  - Geographic filter: TROPICAL/SUBTROPICAL only (latitude ~30N to 30S)
  - Exclude temperate, boreal, mediterranean sites

## 7. Study Design / Eligibility

- included_study_types:
  - RCBD, CRD, factorial, split-plot
  - Multi-year field trials
- excluded_study_types:
  - Reviews, meta-analyses
  - Modeling/simulation
  - Remediation studies without crop yield
- special_design_rules:
  - Multiple biochar rates: extract each rate vs same zero control as separate rows
  - Multiple crops in same trial: extract each crop separately

## 8. Critical Moderators

- moderator_1:
  - name: Biochar feedstock
  - why_it_matters: Wood vs crop residue vs manure biochars have different properties
  - expected_levels: wood, crop_residue, manure, sewage_sludge, mixed, other
  - whether_required_for_alignment: no

- moderator_2:
  - name: Application rate (t/ha)
  - why_it_matters: Dose-response relationship; very high rates may suppress yield
  - expected_levels: continuous
  - whether_required_for_alignment: no

- moderator_3:
  - name: Soil pH
  - why_it_matters: Biochar (alkaline) expected to benefit acidic soils more
  - expected_levels: acidic (<6.5), neutral (6.5-7.5), alkaline (>7.5)
  - whether_required_for_alignment: no

- moderator_4:
  - name: Crop type
  - why_it_matters: Different crops respond differently to biochar
  - expected_levels: grain_cereal, legume, vegetable, root_tuber
  - whether_required_for_alignment: no

- moderator_5:
  - name: Soil texture
  - why_it_matters: Sandy soils may benefit more (improved water retention)
  - expected_levels: sandy, loamy, clay
  - whether_required_for_alignment: no

## 9. Benchmark Subgroup Logic

- subgroup_1:
  - definition: Tropical latitudes only (<=35 degrees latitude)
  - reported_effect: +25% (95% CI: approximately +15% to +35%, read from Figure 1 forest plot; text states "approximately 25%"; exact tabular bounds in supplementary MetaWin output)
  - k: 527 observations from 62 independent publications
  - notes: This IS the primary analysis for this topic. Temperate subgroup: approximately -3% (CI crosses zero).

- subgroup_2:
  - definition: By soil pH (acidic vs neutral/alkaline)
  - reported_effect: To be confirmed from Jeffery 2017
  - notes: Acidic soils expected to show stronger biochar benefit

- subgroup_3:
  - definition: By biochar feedstock
  - reported_effect: To be confirmed
  - notes: Wood biochar may differ from crop residue biochar

## 10. Known Estimand Traps

- trap_1: Including temperate studies — this topic is TROPICAL ONLY
- trap_2: Extracting soil properties (SOC, pH, CEC) instead of crop yield
- trap_3: Including remediation studies where yield is secondary to pollutant cleanup
- trap_4: Confusing biochar with compost or organic matter amendment
- trap_5: Including "activated carbon" studies (different product)

## 11. Prompt Consequences

- extraction_priority_1: Check study location — MUST be tropical or subtropical
- extraction_priority_2: Extract yield outcomes only, not soil properties
- extraction_priority_3: Record biochar feedstock, rate, and pyrolysis temperature from Methods

## 12. Post-Processing Consequences

- keep_rules:
  - Keep if outcome is crop yield AND treatment is biochar AND control is no-biochar AND location is tropical/subtropical
- exclude_rules:
  - Exclude if temperate/boreal/mediterranean location
  - Exclude if outcome is soil property, not crop yield
  - Exclude if treatment is not biochar (compost, fertilizer, lime)
  - Exclude if remediation study without crop yield
- flag_rules:
  - Flag if biochar combined with other amendments and effect cannot be isolated
  - Flag if location is borderline subtropical/temperate
- benchmark_alignment_labels:
  - benchmark_aligned: tropical field experiment, crop yield, biochar vs no-biochar
  - partially_aligned: tropical pot/greenhouse, or subtropical borderline
  - misaligned: temperate location, soil property, or non-biochar amendment

## 13. Provenance

- who_created_spec: Claude (autonomous)
- date_created: 2026-03-25
- created_before_results_seen: yes
- notes: ERL is fully open access (CC-BY). PDF read 2026-03-26. Tropical point estimate +25% confirmed from paper text. CI bounds (~+15% to +35%) are graphical reading from Figure 1 forest plot — the text body says only "approximately 25%" and "significantly increase". Exact tabular CI values are in the supplementary MetaWin output docx (stacks.iop.org/ERL/12/053001/mmedia). Grand mean overall: +13% (k=1125 obs, 109 publications). Feedstock subgroups (tropical): Nutrient biochars +70%, Structure biochars +19%.
