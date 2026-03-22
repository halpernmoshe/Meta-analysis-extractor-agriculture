# Cover Letter

Dear Editors of *Research Synthesis Methods*,

I am pleased to submit the manuscript entitled **"Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets"** for consideration in *Research Synthesis Methods*.

## Summary

This manuscript presents the first validation study of AI-extracted continuous quantitative data against multiple independent published meta-analysis datasets with formal equivalence testing. A single AI agent (Claude Opus 4.6) extracted effect-size data from source PDFs across five agricultural meta-analyses spanning diverse domains (CO2/plant minerals, zinc biofortification, biostimulants, biochar, predator biocontrol), totaling 1,149 matched observations from 136 papers. All five datasets passed proportional TOST equivalence testing at +-20% of each dataset's mean absolute effect, and aggregate effects were reproduced within 0.01--1.61 pp of published values.

## Key Contributions

1. **First multi-dataset equivalence validation of AI extraction.** Previous studies have evaluated AI extraction accuracy on single datasets or with informal metrics. This is the first to apply ICC, proportional TOST with cluster-robust standard errors, and Bland-Altman analysis across five independent datasets, including a fully prospective holdout (biochar, r = 0.997).

2. **LLM-driven alignment as a methodological contribution.** We demonstrate that most apparent extraction error in validation studies is actually alignment error -- the failure to correctly match extracted observations to reference-standard rows. LLM-driven alignment improved the Li 2024 (biochar) correlation from r = 0.377 to r = 0.997 without changing any extracted values.

3. **Run-to-run reproducibility.** Independent duplicate runs on three datasets (1,231 matched observations across 95 papers) confirm extraction stability, with aggregate effects stable within 0.09--0.23 pp on two of three datasets.

4. **Source type transparency.** Every observation is labeled by data source (table vs. figure), revealing that table-sourced data is 5.5x more precise than figure-sourced data -- a finding with direct implications for reporting standards and extraction tool design.

## Why *Research Synthesis Methods*

RSM is the leading methodological journal for evidence synthesis, and this work addresses the journal's core readership: researchers developing and evaluating tools for systematic review and meta-analysis. The manuscript introduces a statistical validation framework (equivalence testing, cross-method agreement, source-type labeling) that is directly applicable to future evaluations of AI extraction tools. The findings are consistent with the 2025 Cochrane--Campbell--JBI--CEE joint position statement calling for domain-specific validation studies before AI tools can be recommended for routine use in evidence synthesis.

## Declarations

- This manuscript has not been published elsewhere and is not under consideration at another journal.
- There are no conflicts of interest to declare.
- The author has no financial relationship with Anthropic, Google, or Moonshot AI beyond standard API usage.
- All code, extracted data, and validation scripts are publicly available at [https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture](https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture).
- Ground-truth datasets are from published open-access sources (Loladze 2014, eLife; Hui et al. 2025, Nature Communications; Li et al. 2022, Frontiers in Plant Science; Li et al. 2024, Scientific Data; Boldorini et al. 2024, Proc. R. Soc. B).
- No human subjects were involved in this research.

I look forward to your consideration.

Sincerely,

Moshe Halpern
Institute of Soil, Water and Environmental Sciences
Agricultural Research Organization -- Volcani Center
Rishon LeZion 7505101, Israel
