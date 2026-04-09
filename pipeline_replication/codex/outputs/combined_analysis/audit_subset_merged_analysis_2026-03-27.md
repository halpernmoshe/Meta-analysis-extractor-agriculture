# Merged Audit Subset Analysis

Total claims: 113

## clean_support
- Claims: 11
- Mean disagreement count: 0.0
- Mean strict support: 3.0
- Mean relaxed support: 3.0
- Mean risk-flag count: 3.273
- Mean construct-drift count: 0.636
- Mean report-channel count: 2.273
- Paper root causes: {"extraction_coverage_limitation": 7, "none": 4}
- Papers: 020_Overdieck_1993, 031_Pal_2003

Feature true-rates:
- drift_concentration_vs_content: 0.0
- drift_tissue_mismatch: 0.0
- drift_arm_mismatch: 0.0
- drift_timepoint_mismatch: 0.0
- drift_pooled_vs_subgroup_mismatch: 0.0
- drift_figure_only_target: 0.636
- warning_timepoint_risk: 0.0
- warning_figure_only_risk: 0.0
- warning_averaging_risk: 0.0
- warning_factorial_risk: 1.0
- warning_multi_condition_risk: 0.0
- report_timepoint_conflict: 0.0
- report_averaging_conflict: 0.0
- report_factorial_conflict: 0.0
- report_figure_digitization_limit: 0.636
- report_mentions_results_text: 0.0
- report_mentions_abstract: 0.0

## likely_alignment_or_structure_problem
- Claims: 92
- Mean disagreement count: 2.283
- Mean strict support: 1.467
- Mean relaxed support: 2.489
- Mean risk-flag count: 5.543
- Mean construct-drift count: 0.837
- Mean report-channel count: 2.043
- Paper root causes: {"matching_alignment_artifact": 42, "none": 50}
- Papers: 002_Ziska_1997, 003_Baslam_2012, 004_Finzi_2001, 005_Niinemets_1999, 007_Woodin_1992, 016_Fernando_2012a, 017_Fangmeier_2002, 021_Wilsey_1994, 040_Pfirrmann_1996, 044_Housman_2012, 051_Niu_2013

Feature true-rates:
- drift_concentration_vs_content: 0.0
- drift_tissue_mismatch: 0.0
- drift_arm_mismatch: 0.0
- drift_timepoint_mismatch: 0.0
- drift_pooled_vs_subgroup_mismatch: 0.652
- drift_figure_only_target: 0.185
- warning_timepoint_risk: 0.0
- warning_figure_only_risk: 0.185
- warning_averaging_risk: 0.37
- warning_factorial_risk: 0.837
- warning_multi_condition_risk: 0.337
- report_timepoint_conflict: 0.435
- report_averaging_conflict: 0.402
- report_factorial_conflict: 0.902
- report_figure_digitization_limit: 0.663
- report_mentions_results_text: 0.207
- report_mentions_abstract: 0.022

## likely_extraction_coverage_problem
- Claims: 10
- Mean disagreement count: 0.0
- Mean strict support: 3.0
- Mean relaxed support: 3.0
- Mean risk-flag count: 5.0
- Mean construct-drift count: 2.0
- Mean report-channel count: 2.0
- Paper root causes: {"extraction_coverage_limitation": 10}
- Papers: 011_Huluka_1994

Feature true-rates:
- drift_concentration_vs_content: 0.0
- drift_tissue_mismatch: 0.0
- drift_arm_mismatch: 0.0
- drift_timepoint_mismatch: 1.0
- drift_pooled_vs_subgroup_mismatch: 0.0
- drift_figure_only_target: 1.0
- warning_timepoint_risk: 1.0
- warning_figure_only_risk: 1.0
- warning_averaging_risk: 0.0
- warning_factorial_risk: 1.0
- warning_multi_condition_risk: 0.0
- report_timepoint_conflict: 1.0
- report_averaging_conflict: 0.0
- report_factorial_conflict: 1.0
- report_figure_digitization_limit: 1.0
- report_mentions_results_text: 1.0
- report_mentions_abstract: 0.0
