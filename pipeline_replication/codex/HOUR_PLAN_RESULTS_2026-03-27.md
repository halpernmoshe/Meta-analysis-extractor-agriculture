# Hour Plan Results

This note records the concrete outputs from the current hour of work.

## Objective Used

Use the existing 14-paper / 113-claim audit subset as the working benchmark and test whether the current feature stack is already sufficient to separate:

- clean support
- alignment/structure problems
- extraction coverage problems

## What Was Built

### Merged analysis layer

Built with [analyze_merged_audit_subset.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/analyze_merged_audit_subset.py):

- [audit_subset_claim_features_merged_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/audit_subset_claim_features_merged_2026-03-27.csv)
- [audit_subset_merged_analysis_2026-03-27.json](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/audit_subset_merged_analysis_2026-03-27.json)
- [audit_subset_merged_analysis_2026-03-27.md](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/audit_subset_merged_analysis_2026-03-27.md)

### First rule-based score

Built with [score_audit_subset_rules.py](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/score_audit_subset_rules.py):

- [audit_subset_rule_scored_2026-03-27.csv](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/audit_subset_rule_scored_2026-03-27.csv)
- [audit_subset_rule_score_summary_2026-03-27.json](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/audit_subset_rule_score_summary_2026-03-27.json)
- [audit_subset_rule_score_summary_2026-03-27.md](/C:/Users/moshe/Dropbox/Testing%20metaanalyis%20program/meta_analysis_extractor/pipeline_replication/codex/outputs/combined_analysis/audit_subset_rule_score_summary_2026-03-27.md)

## Main Result

The simple rule set cleanly separates the current audit subset:

- `clean_support -> clean_support`: 11
- `likely_alignment_or_structure_problem -> alignment_or_structure_problem`: 92
- `likely_extraction_coverage_problem -> extraction_coverage_problem`: 10

This is not a claim that the general problem is solved. It is a claim that the current feature stack is already strong enough to distinguish the three regimes on the benchmark subset built from prior audits.

## Why The Rule Set Works

The useful components are:

1. `strict support`
2. `relaxed support`
3. `claim disagreement count`
4. `timepoint / figure limitation`
5. `averaging / multi-condition / factorial structure`

The crucial distinction is:

- coverage cases can look highly convergent under both strict and relaxed support, but also carry timepoint + figure limitations
- alignment cases often show either disagreement or a gap between relaxed and strict support, plus averaging/factorial/multi-condition structure
- clean cases have strong support, zero disagreement, and no timepoint/averaging warning pattern

## Current Weaknesses

1. The within-paper feature layer is still noisy and is built from report text, not raw paper sections.
2. The rule set is partly circular because the audit subset itself was chosen from known problem types.
3. This is a benchmark subset, not a prospective held-out evaluation.

## Best Next Step

The next meaningful test is not more tuning inside this subset.

It is:

1. apply the same merged feature pipeline to a fresh held-out set of papers
2. see whether the same rules still separate:
   - clean support
   - alignment/structure problems
   - coverage limitations

That is the point where this becomes evidence for a real risk model rather than a tidy retrospective taxonomy.
