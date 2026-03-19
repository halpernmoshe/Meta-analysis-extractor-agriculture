Dr. [Editor-in-Chief]
Research Synthesis Methods
[Date]

Dear Editor,

We are pleased to submit our manuscript, "Breaking the Extraction Bottleneck: A Single AI Agent Achieves Equivalence with Published Meta-Analysis Data Across Three Agricultural Datasets," for consideration as an Original Research Article in *Research Synthesis Methods*.

**Why this manuscript fits RSM.** Data extraction remains the primary bottleneck in evidence synthesis. While recent studies — including Khraisha et al. (2024) and Kataoka et al. (2026), both published in RSM — have explored LLM-based extraction, they report limited accuracy on continuous numerical outcomes (26–75%) and lack formal equivalence testing. Our study addresses this gap directly, validating a single AI agent against three independent ground-truth datasets with the statistical rigor RSM readers expect.

**Three novel contributions:**

1. **Formal equivalence testing of AI-extracted continuous data.** We apply ICC, Lin's CCC, cluster-robust TOST, and Bland-Altman analysis — standard method-comparison tools — to AI extraction for the first time. The agent achieves ICC = 0.845–0.966 across datasets, with all Cohen's d < 0.20 and TOST equivalence confirmed at ±3 pp.

2. **Multi-dataset validation with true holdouts.** Rather than validating against a single benchmark, we test against three published meta-analyses spanning different agricultural domains, effect-size ranges, and reporting conventions. Two datasets (Hui 2023, Li 2022) are fully independent holdouts — neither was seen during development.

3. **Ground-truth-free validation via cross-method agreement.** We demonstrate that two structurally independent extraction methods (differing in model family, architecture, and prompts) converge on the same values for 1,889 observations across 100 papers (all r > 0.93), providing a circularity-free validation framework that scales beyond curated reference standards.

**Additional contributions** include an error taxonomy revealing that alignment ambiguity — not reading errors — dominates extraction discrepancies (97% vs. 3%), a practical Extraction Equivalence Testing (EET) protocol for researchers deploying AI tools, and a cost analysis showing three orders of magnitude reduction relative to manual methods.

The manuscript responds directly to the Cochrane/Campbell/JBI/CEE joint position statement (2025) calling for validation evidence before AI tools can be recommended for routine use in evidence synthesis. We believe RSM is the ideal venue for this work, given the journal's focus on methodological innovation in systematic review practice.

The manuscript is approximately 8,500 words with 5 figures, 7 tables, and an appendix. All code, configuration files, and validation scripts are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.

This manuscript has not been published elsewhere and is not under consideration at any other journal. The author declares no conflicts of interest.

Thank you for your consideration.

Sincerely,

Moshe Halpern
Institute of Soil, Water and Environmental Sciences
Agricultural Research Organization — Volcani Center
Rishon LeZion 7505101, Israel
