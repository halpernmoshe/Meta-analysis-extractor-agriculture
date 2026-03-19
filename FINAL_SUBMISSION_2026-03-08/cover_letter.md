# Cover Letter

Dear Editors of Research Synthesis Methods,

I am pleased to submit for your consideration the manuscript entitled **"Multi-Model AI Consensus for Confidence-Stratified Data Extraction in Plant Science Meta-Analysis"** for publication in Research Synthesis Methods.

## Why this manuscript matters for RSM readers

Data extraction remains the primary bottleneck in evidence synthesis, consuming 2-8 hours per paper with single-extractor error rates approaching 18%. While large language models (LLMs) have shown promise for categorical study characteristics (>90% accuracy), they achieve only 26-36% accuracy on the continuous quantitative outcomes that meta-analysis actually pools. This paper addresses this fundamental gap.

## Key contributions

1. **First validation on agricultural/ecological data.** To our knowledge, this is the first AI extraction system validated against published plant science meta-analysis datasets, addressing the observation by Clark et al. (2025) that 17 of 19 generative AI systematic review studies focus on clinical/biomedical settings.

2. **The Three-Barrier Model with programmatic circularity breaking.** We introduce a conceptual framework decomposing extraction *validation* into three distinct challenges: the Reading Barrier (table comprehension), the Granularity Barrier (analytical sub-selection concordance), and the Provenance Barrier (reference-standard integrity). Crucially, the classification of discrepancy sources is performed by a fully programmatic algorithm using only observable data properties (zero-error fraction, direction agreement, MAE thresholds)---not by LLM auditors---breaking the "LLMs confirming LLMs" circularity that otherwise undermines validation credibility. This framework has immediate methodological implications for how future AI extraction benchmarks should be designed and interpreted.

3. **Multi-model consensus as a confidence predictor.** Building on Khan et al. (2025), we demonstrate that inter-model agreement predicts extraction accuracy with sufficient reliability to support automated triage: ~75% of observations are auto-validated at MAE < 5%, while the flagged minority concentrates 95% of large errors.

4. **Formal equivalence testing.** We are the first AI extraction study to apply TOST, ICC, and Bland-Altman agreement analyses, providing a statistical framework appropriate for RSM's methodologically rigorous readership.

5. **Practical impact.** The pipeline reduces human extraction time by an estimated 70–75% at a cost of ~$0.37 per paper (January–March 2026 API pricing), shifting the bottleneck from manual data entry to automated triage.

## Validation scope

The pipeline was validated against three published datasets totaling 921 matched observations across 92 papers:
- Loladze 2014 (CO2/mineral concentrations, development case study: ICC = 0.870, r = 0.886, MAE = 4.36 pp on 413 unambiguous observations)
- Hui et al. 2023 (zinc biofortification/wheat, independent holdout: ICC = 0.999, r = 0.999, MAE = 0.43 pp, 308 observations)
- Li et al. 2022 (biostimulants/agronomic outcomes, cross-domain: ICC = 0.949, r = 0.951, MAE = 2.30 pp, 200 observations)

Aggregate meta-analytic effects were reproduced to within 0.05-0.84 percentage points across all datasets. On the independent holdout, 81% of observations had zero error.

## Declarations

This manuscript has not been published elsewhere and is not under consideration at another journal. All code, data, and validation scripts are publicly available. I confirm that appropriate ethical standards have been followed in the conduct of this research.

I believe this work will be of significant interest to RSM's readership and look forward to your consideration.

Sincerely,

Moshe Halpern
Institute of Soil, Water and Environmental Sciences
Agricultural Research Organization -- Volcani Center, Israel
