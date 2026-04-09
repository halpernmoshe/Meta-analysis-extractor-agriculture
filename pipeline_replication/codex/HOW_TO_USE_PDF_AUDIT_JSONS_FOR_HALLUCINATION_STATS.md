## Purpose

This memo explains how to use the PDF-based audit artifacts as weak labels and features for the convergence / hallucination-risk statistics.

## The Audit Artifacts That Matter

1. Per-paper consensus JSONs
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\claude_kimi_full_comparison\*_consensus.json`
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\claude_kimi_comparison_full\*_consensus.json`

Each file contains:
- `recon` block: a PDF-grounded reconstruction (warnings, treatment/control definition, factorial structure, variance source, extraction guidance)
- `consensus_observations`: structured values both models agree on
- `disagreements`: structured items present in one model only
- `claude_obs`, `kimi_obs`, `matched_obs`
- sometimes `verification_flags`

2. Explicit PDF-based audit writeups
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\concordant_error_audit.md`
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\concordant_error_audit_v2.md`
- `C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\model_comparison\claude_review.json`

These files contain explicit judgments about correctness or root cause when models agree but differ from GT.

## Why These Are Useful

They are not just comparisons. They include:
- PDF-grounded context
- the known failure modes (alignment vs extraction vs aggregation)
- an explicit root-cause taxonomy

This gives us two things:

1. Weak labels for “type of error,” not just “error vs no error”
2. Features that predict those errors (factorial structure, multiple CO2 levels, missing variance, etc.)

## How To Turn Them Into Labels

### Paper-level labels

Use the audits to label paper risk type:
- `alignment_risk_high` if recon warns about multiple treatments, factorials, ambiguous comparator
- `extraction_risk_high` if recon indicates missing variance or data only in figures
- `aggregation_risk_high` if audit cites year-level or condition averaging as the root cause

### Claim-level labels

For each claim in a paper:
- `supported_by_consensus` if it appears in `consensus_observations`
- `model_disagreement` if it appears in `disagreements`
- `audit_flagged` if the concordant audit lists the paper/element and root cause

### Row-level proxy labels

Only for aligned subsets:
- `low_risk` if consensus observation exists and paper has no high-risk recon warnings
- `high_risk` if consensus exists but concordant audit lists that paper/element as a mismatch
- `uncertain` otherwise

This avoids pretending that we have a perfect binary “true/false” label.

## Features Extractable Directly From These JSONs

From each `*_consensus.json`:
- `n_recon_warnings`
- `has_factorial_structure`
- `has_tc_confusion`
- `variance_confidence`
- `n_disagreements`
- `claude_obs`, `kimi_obs`
- `matched_obs`
- `agreement_ratio = matched_obs / max(claude_obs, kimi_obs)`
- `consensus_count = len(consensus_observations)`
- `disagreement_count = len(disagreements)`

From each `disagreements` item:
- `disagreement_type` (`claude_only` or `kimi_only`)
- `missing_control` or `missing_variance`
- `tissue`, `element`, `unit`, `data_source`
- `confidence` of the extraction

From each `consensus_observation`:
- `data_source` (table vs figure)
- `confidence`
- `has_variance`
- `effect_pct`

From concordant audit documents:
- error category: `alignment_artifact`, `temporal_point_mismatch`, `aggregation_mismatch`, `extraction_gap`
- paper + element + tissue patterns that predict error

## How To Use The Concordant Error Audits

These are especially valuable because they show:
- all models agreed
- extracted numbers were consistent
- yet GT mismatch remained

That means the error is **not** hallucination. It is:
- alignment mismatch, or
- aggregation mismatch, or
- extraction gap (data not accessible)

So:
- if a case appears in `concordant_error_audit_v2.md`, you should **not** treat it as hallucination even if it disagrees with GT
- instead mark it as “alignment risk” or “aggregation risk”

This will prevent your model from learning that “agreement = wrong.”

## Practical Use In The Convergence Framework

1. External convergence features come directly from `consensus_observations` vs `disagreements`.
2. Internal convergence features come from `recon` warnings and guidance:
   - presence of multiple CO2 levels
   - factorial structure
   - complex sampling (years, seasons, timepoints)
3. Root-cause categories from the concordant audits become error-type labels.

This lets you build a triage model:

- `hallucination risk` is mostly about low convergence + weak provenance
- `alignment risk` is about high convergence + high structural complexity
- `extraction gap risk` is about missing data sources (figures only, missing variance)

## Minimal Implementation Steps

1. Parse a sample of `*_consensus.json` into a dataframe with:
   - paper_id
   - consensus_count
   - disagreement_count
   - agreement_ratio
   - recon warning count
   - recon flags (factorial, multi-CO2, missing variance)

2. Add concordant audit flags:
   - if paper + element appears in `concordant_error_audit*.md`, add `alignment_risk = 1`

3. Use that as the first “risk score” dataset.

This is low-effort and immediately useful for building the first version of the hallucination-risk stats.
