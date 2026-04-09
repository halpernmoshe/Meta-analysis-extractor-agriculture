Moshe Halpern
Institute of Soil, Water and Environmental Sciences
Agricultural Research Organization -- Volcani Center
Rishon LeZion 7505101, Israel
Email: hmoshe@volcani.agri.gov.il

6 April 2026

The Editors
*Environmental Evidence*

Dear Editors,

I am writing to submit the manuscript titled **"Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets"** for consideration as a Methodology article in *Environmental Evidence*.

Data extraction remains the primary bottleneck in systematic reviews and meta-analyses, consuming weeks of researcher time with single-extractor error rates reaching 17.7%. Despite rapid advances in large language models, existing LLM-based systems achieve only 26--36% accuracy on continuous numerical outcomes, and no study has validated AI-extracted continuous data against multiple independent datasets using formal equivalence testing. This manuscript addresses that gap directly.

A single AI agent extracted treatment means, control means, sample sizes, and variance measures from source PDFs across five published agricultural meta-analyses spanning zinc biofortification, biostimulant efficacy, biochar amendments, predator biocontrol, and elevated CO2 effects on plant mineral nutrition. Across these five datasets, the agent produced 1,149 matched observations from 136 papers. Pearson correlations with published reference standards ranged from 0.984 to 0.999. Proportional TOST equivalence testing confirmed statistical equivalence for all five datasets (all p < 0.05), and aggregate effects were reproduced within 0.01--1.61 percentage points of published values. Independent duplicate runs confirmed extraction stability.

The manuscript introduces several methodological contributions of broad relevance to the evidence synthesis community:

- **Proportional TOST equivalence framework.** Conventional TOST with fixed margins is inappropriate when effect sizes span orders of magnitude. The proportional variant scales equivalence margins to effect-size magnitude, providing a rigorous, domain-general test for AI-human agreement on continuous outcomes.

- **LLM-driven alignment and error decomposition.** Separating extraction error from matching error revealed that much of what appears to be extraction error is actually alignment error -- matching correct values to the wrong reference-standard row. In one dataset, correcting alignment alone raised the correlation from 0.377 to 0.997 without changing any extracted values. This finding has implications for how all AI extraction validation studies should be designed and interpreted.

- **Source-type stratification.** Labeling each observation by its source (table vs. figure) showed that table-sourced data achieved 5.5x lower median error than figure-estimated data, providing a practical quality signal for downstream meta-analysts.

- **Extraction Equivalence Testing (EET) protocol.** The manuscript proposes a standardized protocol for validating AI extraction tools, combining proportional TOST, ICC, Bland-Altman analysis, and source-type reporting into a reproducible testing framework.

I believe *Environmental Evidence* is the appropriate venue for this work for several reasons. First, the journal publishes methodology papers that advance the practice of evidence synthesis, and this manuscript introduces four generalizable methodological tools for the field. Second, the journal is managed by the Collaboration for Environmental Evidence (CEE), which -- together with Cochrane, Campbell, and JBI -- issued a 2025 joint position statement calling for validated AI tools in evidence synthesis. This manuscript responds directly to that call by providing the first multi-dataset equivalence validation of AI-extracted continuous data. Third, all five validation datasets fall within the agricultural and environmental domains central to the journal's scope. Finally, while the methods are validated in agricultural science, they are domain-general and applicable across any field that conducts meta-analyses of continuous outcomes.

The manuscript is approximately 8,000 words including references. All code and data are publicly available on GitHub. This manuscript has not been published previously and is not under consideration at any other journal. There are no competing interests and no specific funding to declare. I am the sole author.

Thank you for considering this submission. I look forward to your response.

Sincerely,

Moshe Halpern
