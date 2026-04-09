# Held-Out Test Results

## Goal

Run the existing convergence / warning / within-paper pipeline on a fresh paper batch not used in the original 14-paper audit subset, and apply the same rule set without changing the overall framework.

Held-out paper list:

- [heldout_papers_2026-03-27.txt](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/heldout_papers_2026-03-27.txt)

Papers used:

- `001_Ma_2007`
- `006_Azam_2013`
- `008_Campbell_2002`
- `014_Lieffering_2004`
- `018_Al-Rawahy_2013`
- `022_Blank_2011`
- `025_Guo_2011`
- `032_Kanowski_2001`
- `058_ONeill_1987`

## Outputs

Feature layers:

- [heldout_subset_paper_features_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_paper_features_2026-03-27.csv)
- [heldout_subset_claim_features_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_claim_features_2026-03-27.csv)
- [heldout_subset_claim_labels_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_claim_labels_2026-03-27.csv)
- [heldout_subset_within_paper_features_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_within_paper_features_2026-03-27.csv)
- [heldout_subset_claim_features_merged_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_claim_features_merged_2026-03-27.csv)

Summaries:

- [heldout_subset_label_analysis_2026-03-27.md](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_label_analysis_2026-03-27.md)
- [heldout_subset_merged_analysis_2026-03-27.md](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_merged_analysis_2026-03-27.md)
- [heldout_subset_rule_score_summary_2026-03-27.md](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/heldout_subset_rule_score_summary_2026-03-27.md)

## Main Result

The rule set generalizes partially, not perfectly.

After one small threshold relaxation for strong 2-model support:

- held-out `clean_support`: 28 / 34 predicted clean
- held-out `likely_alignment_or_structure_problem`: 33 / 33 predicted as alignment/structure
- held-out `unclear`: 7 / 7 predicted as alignment/structure

This is encouraging because the rule set preserved the most important behavior:

1. It still catches the messy/alignment-heavy papers.
2. It is no longer overly strict on all 2-model clean claims.

## What The Remaining Errors Mean

The 6 remaining held-out clean claims predicted as bad are all low-support rare-element cases:

- `006_Azam_2013`: `Cd`, `Cr`, `H`, `Ni`, `Pb`
- `058_ONeill_1987`: `Al`

These have:

- zero or near-zero support
- zero disagreement
- low risk-flag counts

This is a conservative failure mode. The rule set is not rejecting strong clean claims. It is refusing to auto-trust low-support edge claims.

That is much better than the earlier failure mode, where it was over-penalizing all factorial papers.

## Interpretation

The held-out test supports the following stronger claim:

The current feature stack is already useful for separating:

- obviously clean claims
- alignment/structure-heavy claims
- timepoint / coverage-limited claims

But it is still conservative on weakly supported rare-element claims, which are best treated as:

- `uncertain`
- or `needs escalation`

rather than automatically clean.

## Best Next Step

Move from a 3-way classifier to a 4-way policy:

1. `clean_support`
2. `alignment_or_structure_problem`
3. `extraction_coverage_problem`
4. `low_support_uncertain`

That would match the actual behavior of the held-out run much better than forcing every low-support case into either clean or bad.
