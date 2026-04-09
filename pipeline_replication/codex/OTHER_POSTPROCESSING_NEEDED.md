# Other Post-Processing That Is Still Needed

## 1. Universal Outcome Canonicalization

Map extracted outcomes into a small ontology:

- grain yield
- harvested yield
- biomass
- quality trait
- nutrient concentration
- system productivity
- component crop yield

This should happen before synthesis so that clearly non-comparable outcomes never enter the pooled estimate.

## 2. Universal Study-Setting Normalization

Normalize study context into:

- field
- greenhouse
- pot
- mixed
- unknown

This matters because several benchmark papers are field-focused while the extracted corpus mixes field and pot studies.

## 3. Universal Intervention/Comparator Normalization

The post-processor should normalize treatment and control labels into a structured comparison class, for example:

- strict no-till vs conventional tillage
- reduced tillage vs conventional tillage
- organic vs conventional
- biochar vs no biochar
- AMF vs non-AMF

This would prevent semantic drift from entering the synthesis.

## 4. Estimand Normalization

The pipeline should explicitly classify rows by estimand:

- benchmark-aligned
- partially aligned
- misaligned

This is especially important for intercropping, where `LER` and component crop yield should never be pooled together in the primary benchmark comparison.

## 5. Variance Integrity Checks

Add a post-processing layer for:

- impossible or suspicious variance values
- SE vs SD ambiguity
- missing variance classification
- outlier variance diagnostics

This matters because even when the row is semantically correct, the weighting can still be wrong.

## 6. Duplicate / Near-Duplicate Detection

Rows should be screened for:

- repeated extraction of the same comparison
- same comparison reported in table and text
- repeated years or pooled summaries duplicated as raw observations

This is a universal problem in agricultural papers with dense supplementary reporting.

## 7. Pairing Validation For Derived Metrics

When metrics like `LER` must be reconstructed, the post-processor should verify:

- same paper
- same site-year
- same treatment condition
- same N level / row arrangement / density context

Without this, derived benchmark-aligned metrics are too fragile.

## 8. Benchmark-Aligned Secondary Summaries

Primary preregistered analyses should remain intact.
But the post-processing layer should also produce secondary benchmark-aligned summaries using normalized effectors, so the paper can distinguish:

- failure from bad row selection
- failure from sample composition
- failure from estimand mismatch

## 9. Confidence-Aware Downweighting Or Triage

Rather than binary exclusion only, the post-processor should consider:

- table vs figure source
- high / medium / low confidence
- flagged ambiguity

These can be used either for sensitivity analyses or for manual-review prioritization.
