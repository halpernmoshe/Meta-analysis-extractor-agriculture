# Benchmark Spec: Cover Crop Effects on Corn Yield

## 1. Benchmark Identity

- benchmark_citation: Marcillo & Miguez 2017
- benchmark_title: Corn yield response to winter cover crops: An updated meta-analysis
- benchmark_year: 2017
- benchmark_journal: Journal of Soil and Water Conservation
- benchmark_doi: 10.2489/jswc.72.3.226

## 2. Core Research Question

- benchmark_question: Do winter cover crops affect subsequent corn (maize) grain yield?
- target_estimand: Corn grain yield after cover crop vs after fallow
  - NOT cover crop biomass, NOT soil properties, NOT soybean yield

## 3. Intervention Definition

- intervention_label: Winter cover crop before corn
- intervention_required_features:
  - Cover crop grown during winter/off-season before corn planting
  - Any species: cereal rye, hairy vetch, crimson clover, radish, annual ryegrass, mixtures
  - Cover crop terminated before or at corn planting
- intervention_excluded_variants:
  - Perennial cover (not terminated — becomes living mulch, different construct)
  - Cover crop as intercrop during corn growing season
  - Green manure incorporated without a subsequent corn crop
- intervention_ambiguity_notes:
  - "Catch crop" in European literature = cover crop — include
  - Legume cover crops (vetch, clover) vs grass covers (rye) may show different effects
  - Cover crop mixtures (e.g., rye + vetch) are acceptable

## 4. Comparator Definition

- comparator_label: No cover crop (winter fallow / bare soil)
- comparator_required_features:
  - Same field, same corn variety, same fertilization
  - No cover crop during winter (bare/fallow)
- comparator_excluded_variants:
  - Comparison between two different cover crop species without a fallow control
  - Comparison between cover crop termination timings (without no-cover control)
- comparator_ambiguity_notes:
  - "Conventional" may mean no cover crop — verify from paper context
  - Winter weed growth on fallow is acceptable (not truly bare, but no planted cover)

## 5. Outcome Definition

- primary_outcome_label: Corn (maize) grain yield
- acceptable_primary_outcomes:
  - Corn grain yield (kg/ha, Mg/ha, bu/ac)
  - Maize grain yield
- excluded_outcomes:
  - Cover crop biomass production
  - Soil N, organic matter, or soil health metrics
  - Weed suppression metrics
  - Water infiltration or runoff
  - Soybean yield (even if in same rotation study)
  - Corn silage yield (different harvest, different moisture)
  - Corn stover yield
- acceptable_units: kg/ha, Mg/ha, bu/ac, t/ha
- outcome_hierarchy:
  1. Corn grain yield per unit area (bu/ac or Mg/ha)
  2. Corn total dry matter (only if grain yield not reported)

## 6. Study Setting

- allowed_settings:
  - field (required — cover crop rotation meaningless in pots)
- excluded_settings:
  - greenhouse, pot, growth chamber
- setting_notes:
  - Field experiments only
  - Primarily US Midwest and Southeast, but global studies acceptable

## 7. Study Design / Eligibility

- included_study_types:
  - Field rotation experiments
  - Long-term cover crop trials
  - On-farm trials
  - Strip-trial comparisons
- excluded_study_types:
  - Reviews, meta-analyses
  - Modeling studies
  - Greenhouse/pot experiments
- special_design_rules:
  - Multi-year data: extract each year separately
  - Multiple cover crop species: extract each vs same fallow control
  - Tillage x cover crop factorial: extract cover vs no-cover at each tillage level

## 8. Critical Moderators

- moderator_1:
  - name: Cover crop species
  - why_it_matters: Legume covers (vetch, clover) contribute N; grass covers (rye) may immobilize N
  - expected_levels: cereal_rye, hairy_vetch, crimson_clover, radish, annual_ryegrass, mixture, other
  - whether_required_for_alignment: no

- moderator_2:
  - name: Cover crop type
  - why_it_matters: Legume vs grass vs brassica have different mechanisms
  - expected_levels: grass, legume, brassica, mixture
  - whether_required_for_alignment: no

- moderator_3:
  - name: Tillage system
  - why_it_matters: No-till cover crop systems may have different effects than conventional till
  - expected_levels: no_till, conventional_till, reduced_till
  - whether_required_for_alignment: no

- moderator_4:
  - name: N fertilizer rate on corn
  - why_it_matters: Cover crop N effect most visible at low N rates
  - expected_levels: 0, low (<100 kg/ha), medium (100-175), high (>175)
  - whether_required_for_alignment: no

- moderator_5:
  - name: Region
  - why_it_matters: US Midwest vs Southeast vs other regions may differ
  - expected_levels: US_midwest, US_southeast, US_northeast, Canada, Europe, other
  - whether_required_for_alignment: no

## 9. Benchmark Subgroup Logic

- subgroup_1:
  - definition: By cover crop type (grass vs legume vs mixture)
  - reported_effect: Legume covers expected to show positive effect; grass covers neutral/slightly negative
  - notes: Key biological distinction

- subgroup_2:
  - definition: By tillage system (no-till vs conventional)
  - reported_effect: To be confirmed from Marcillo & Miguez 2017
  - notes: No-till + cover crop is the most common management combination

## 10. Known Estimand Traps

- trap_1: Extracting COVER CROP BIOMASS instead of CORN YIELD
- trap_2: Extracting soybean yield from a corn-soybean rotation study (want corn years only)
- trap_3: Extracting corn silage yield (whole plant, different from grain yield)
- trap_4: Confusing cover crop effects on soil properties with effects on corn yield
- trap_5: Including perennial living mulch studies (different construct from annual cover crop)

## 11. Prompt Consequences

- extraction_priority_1: The outcome is CORN (MAIZE) GRAIN YIELD, not cover crop biomass
- extraction_priority_2: Look for "fallow" or "no cover" as the control treatment
- extraction_priority_3: Record cover crop species, tillage system, and N rate

## 12. Post-Processing Consequences

- keep_rules:
  - Keep if outcome is corn/maize grain yield AND treatment is cover crop AND control is fallow/no cover
- exclude_rules:
  - Exclude if outcome is cover crop biomass, soil property, weed count
  - Exclude if outcome is soybean yield
  - Exclude if corn silage (not grain)
  - Exclude if no fallow control
  - Exclude if greenhouse/pot experiment
- flag_rules:
  - Flag if "living mulch" or perennial cover (different construct)
  - Flag if cover crop and no-cover differ in tillage (confounded)
- benchmark_alignment_labels:
  - benchmark_aligned: field, corn grain yield, annual cover crop vs fallow, same tillage
  - partially_aligned: field, corn total biomass, or cover crop species comparison without fallow
  - misaligned: cover crop biomass, soil property, soybean yield, pot experiment

## 13. Provenance

- who_created_spec: Claude (autonomous)
- date_created: 2026-03-25
- created_before_results_seen: yes (no V1 replication; cover_crop_soybean was different topic)
- notes: Cover crop effects on corn yield are near zero — this tests the pipeline's ability to replicate a null/small effect. JSWC has moderate OA.
