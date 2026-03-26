# Benchmark Spec: Humic Acid Effects on Crop Yield

## 1. Benchmark Identity

- benchmark_citation: Ma, Cheng & Zhang 2024
- benchmark_title: The Impact of Humic Acid Fertilizers on Crop Yield and Nitrogen Use Efficiency: A Meta-Analysis
- benchmark_year: 2024
- benchmark_journal: Agronomy (MDPI)
- benchmark_doi: 10.3390/agronomy14122763
- benchmark_k: 93 articles, 383 yield observations (479 total including NUE)

## 2. Core Research Question

- benchmark_question: Does exogenous humic substance application increase crop yield?
- target_estimand: Direct crop yield (grain, fruit, tuber, or harvestable biomass)
  - NOT plant height, root length, chlorophyll content, or nutrient uptake
  - Yield measured as mass per unit area (kg/ha, t/ha) or mass per plant (g/plant)

## 3. Intervention Definition

- intervention_label: Humic substance application
- intervention_required_features:
  - Exogenous application of humic acid, fulvic acid, or humate products
  - May be applied to soil (drench, incorporation) or foliage (spray)
  - Includes commercial humic products (e.g., leonardite extracts, K-humate)
  - Includes fulvic acid (subset of humic substances)
- intervention_excluded_variants:
  - Compost or vermicompost where humic acid is not isolated as the variable
  - Biochar (different mechanism, different product)
  - Organic matter amendment where HA is not the active principle
  - Microbial inoculants that happen to contain some humic substances
- intervention_ambiguity_notes:
  - "Biostimulant" studies may include humic acid as one component among many — include only if HA is the primary/sole active ingredient
  - Some products labeled "humic acid" may contain significant fulvic acid fraction — include both
  - Purity and source vary widely; this is acceptable

## 4. Comparator Definition

- comparator_label: No humic substance application (untreated control)
- comparator_required_features:
  - Same crop, same field/pot, same base fertilization
  - No humic acid, fulvic acid, or humate product added
  - May receive water, carrier solution, or equivalent volume of non-HA solution
- comparator_excluded_variants:
  - Comparison between two different humic acid rates (without a zero control)
  - Comparison between humic acid and another biostimulant
- comparator_ambiguity_notes:
  - In factorial designs (HA x NPK), use the HA vs no-HA comparison at the SAME NPK level
  - If only "full fertilizer" vs "full fertilizer + HA" is available, that is acceptable (isolates HA effect)

## 5. Outcome Definition

- primary_outcome_label: Crop yield
- acceptable_primary_outcomes:
  - Grain yield (kg/ha, t/ha, g/plant)
  - Fruit yield (kg/ha, kg/plant, g/plant)
  - Tuber yield (kg/ha, t/ha)
  - Seed yield (kg/ha)
  - Total aboveground biomass (if harvestable crop product)
  - Dry matter yield of harvested portion
- excluded_outcomes:
  - Plant height
  - Root length, root biomass, root volume
  - Chlorophyll content (SPAD)
  - Nutrient concentration or uptake
  - Soil properties (organic carbon, CEC, pH)
  - Germination rate
  - Number of fruits/pods (without weight)
  - Leaf area index
  - Photosynthesis rate
  - Enzyme activity
- acceptable_units: kg/ha, Mg/ha, t/ha, g/plant, g/pot, kg/plant, g/m2
- outcome_hierarchy:
  1. Grain/seed yield per unit area (preferred)
  2. Fruit/tuber yield per unit area
  3. Yield per plant (if area-based not available)
  4. Total dry biomass (last resort)

## 6. Study Setting

- allowed_settings:
  - field
  - greenhouse
  - pot (acceptable for this topic since many HA studies are pot-based)
- excluded_settings:
  - hydroponic (soilless — HA acts on soil, mechanism different)
  - in-vitro / tissue culture
- setting_notes:
  - Field studies are preferred for benchmark alignment
  - Pot/greenhouse studies are common and acceptable but should be labeled
  - Sensitivity analysis: field-only subset vs all settings

## 7. Study Design / Eligibility

- included_study_types:
  - Randomized complete block design (RCBD)
  - Completely randomized design (CRD)
  - Factorial experiments (if HA effect can be isolated)
  - Split-plot designs
- excluded_study_types:
  - Reviews and meta-analyses (no primary data)
  - Observational studies (no experimental control)
  - Pure modeling/simulation studies
  - Surveys of farmer practices
- special_design_rules:
  - Factorial designs: extract HA vs no-HA at each level of the other factor, OR extract the HA main effect if reported

## 8. Critical Moderators

- moderator_1:
  - name: Crop type
  - why_it_matters: HA effects may differ by crop category (cereals vs vegetables vs legumes)
  - expected_levels: grain_cereal, legume, vegetable, root_tuber, oilseed, tree_crop, grass_forage
  - whether_required_for_alignment: no (include all crops)

- moderator_2:
  - name: Application method
  - why_it_matters: Soil vs foliar application may have different efficacy
  - expected_levels: soil_drench, foliar_spray, seed_treatment, fertigation, soil_incorporation
  - whether_required_for_alignment: no

- moderator_3:
  - name: Application rate
  - why_it_matters: Dose-response relationship expected; very low or very high rates may differ
  - expected_levels: continuous (mg/L for foliar, kg/ha for soil)
  - whether_required_for_alignment: no

- moderator_4:
  - name: Study setting
  - why_it_matters: Field vs pot may show different magnitudes
  - expected_levels: field, greenhouse, pot
  - whether_required_for_alignment: yes (field-only for benchmark-aligned subset)

- moderator_5:
  - name: Humic acid source
  - why_it_matters: Leonardite vs compost-derived vs commercial may differ
  - expected_levels: leonardite, peat, compost_extract, vermicompost_extract, commercial_product, unspecified
  - whether_required_for_alignment: no

## 9. Benchmark Subgroup Logic

- subgroup_1:
  - definition: Field experiments only
  - reported_effect: To be determined from benchmark paper
  - notes: Expected to be closer to benchmark overall effect

- subgroup_2:
  - definition: By crop type (cereals vs vegetables vs legumes)
  - reported_effect: To be determined from benchmark paper
  - notes: Vegetables may show larger response than cereals

- subgroup_3:
  - definition: By application method (soil vs foliar)
  - reported_effect: To be determined from benchmark paper
  - notes: Both methods are common; may not differ substantially

## 10. Known Estimand Traps

- trap_1: Extracting plant height or root length instead of yield — these are NOT yield outcomes
- trap_2: Including studies where HA is confounded with other biostimulants (seaweed, amino acids) in a multi-component product
- trap_3: Confusing compost application (which contains some HA) with pure/isolated HA application
- trap_4: Including nutrient concentration responses as "yield" — they are quality traits, not yield

## 11. Prompt Consequences

- extraction_priority_1: Search for "yield", "harvest", "grain", "fruit", "tuber" in table headers — these are the target outcomes
- extraction_priority_2: When a study has both yield and growth parameters, ONLY extract yield rows
- extraction_priority_3: Record the humic acid product name, source material, and application rate from Methods section

## 12. Post-Processing Consequences

- keep_rules:
  - Keep if outcome is crop yield (any form) AND treatment is HA/fulvic AND control is no-HA
  - Keep if outcome is biomass of harvested portion
- exclude_rules:
  - Exclude if outcome is plant height, root length, chlorophyll, nutrient uptake
  - Exclude if treatment is compost/biochar/other organic amendment (not isolated HA)
  - Exclude if no proper control (comparing two HA rates without zero)
  - Exclude if study is hydroponic
- flag_rules:
  - Flag if HA is part of a multi-component biostimulant product
  - Flag if treatment includes additional microorganisms
  - Flag if outcome is "biomass" but unclear if harvestable portion
- benchmark_alignment_labels:
  - benchmark_aligned: field experiment, crop yield, isolated HA treatment vs no-HA control
  - partially_aligned: pot/greenhouse, or yield per plant (not per area), or HA+NPK vs NPK
  - misaligned: growth parameter, nutrient concentration, or HA confounded with other inputs

## 13. Provenance

- who_created_spec: Claude (autonomous, from scored_candidates.csv and topic config)
- date_created: 2026-03-25
- created_before_results_seen: yes (no V2 extraction has been run on this topic)
- notes: Pilot topic for V2 dress rehearsal. Benchmark is Ma et al. 2024 (93 articles, 383 yield obs). Key moderators: crop type (cash>upland>paddy), soil pH (6-8 optimal), N rate (100-200 kg/ha), precipitation (>300mm), temperature (>10C).
