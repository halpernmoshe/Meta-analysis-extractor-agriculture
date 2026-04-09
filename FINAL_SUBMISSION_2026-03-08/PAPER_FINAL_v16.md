# AI-Extracted Effect Sizes Achieve Statistical Equivalence with Published Meta-Analysis Data Across Three Plant Science Datasets

**Moshe Halpern** ^ORCID^

Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization — Volcani Center, Israel

*Correspondence:* Moshe Halpern, Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization — Volcani Center, Rishon LeZion 7505101, Israel.

---

# Abstract

**Background:** Data extraction is the primary bottleneck in meta-analysis, with 17.7% error rates for single extractors (Buscemi et al., 2006) and errors in 66.8% of published meta-analyses (Mathes et al., 2017). LLMs achieve only 26--36% accuracy on continuous outcomes (Jansen et al., 2025). No study has validated AI-extracted continuous data against multiple independent datasets with formal equivalence testing.

**Methods:** A single AI agent (Claude Opus 4.6) extracted data from source PDFs across three plant science meta-analyses: Loladze 2014 (CO2/minerals, 46 papers), Hui 2025 (zinc biofortification, 30 papers, 461 observations), and Li 2022 (biostimulants, 50 papers). We validated using ICC(3,1), Lin's CCC, cluster-robust TOST (CR2 with Satterthwaite df), and Bland-Altman analysis. Variance (SE/SD) extraction was not validated and remains an open challenge.

**Results:** On the non-circular Loladze subset (447 observations matched entirely on metadata), r = 0.891, ICC = 0.890, MAE = 3.4 pp (percentage points), direction agreement = 94%. Hui: ICC = 0.942 (461 obs). Li raw matching yielded r = 0.446 (166 obs) due to unit-scale mismatches; after programmatic scale harmonization (no values changed), r = 0.968 on 68 observations. CR2 TOST confirmed equivalence at +/-3 pp for Loladze and Li (p = 0.011 and p < 0.001 respectively); Hui did not reach equivalence at +/-3 pp (p = 0.099) or +/-2 pp (p = 0.184). Li passed at +/-2 pp (p = 0.004). All Cohen's d < 0.07. Cross-method agreement with a structurally independent pipeline yielded r > 0.93 on 1,889 observations without ground truth.

**Conclusions:** A single AI agent achieves agreement with published reference data sufficient for aggregate pooling, reproducing effects within 0.22--0.50 pp. The approach enables previously infeasible syntheses by reducing extraction from months to hours.

**Keywords:** meta-analysis, data extraction, large language models, AI agent, agriculture, equivalence testing, TOST, concordance

---

# 1. Introduction

Meta-analysis is the cornerstone of evidence-based practice in agricultural science, but the systematic review process is both slow and fragile. Borah et al. (2017) found that the mean time to complete and publish a systematic review is 67.3 weeks, and Shojania et al. (2007) showed that 23% of reviews already need updating within two years of publication. The primary bottleneck is not statistical analysis but data extraction: trained researchers must manually identify, read, and record quantitative values from source publications.

This bottleneck is both expensive and error-prone. Schmidt et al. (2025) estimate 2--8 hours per paper for manual extraction. Buscemi et al. (2006) demonstrated that single-extractor error rates reach 17.7%, falling to 8.8% only under costly dual-extraction protocols -- underpinning the Cochrane Handbook's recommendation for dual independent extraction (Higgins et al., 2023). At the meta-analysis level, Mathes et al. (2017) found that 66.8% of published meta-analyses contain at least one data extraction error. For agricultural meta-analyses, which may span hundreds of papers with complex factorial designs, the extraction burden can extend to months of researcher time.

Agricultural experiments present distinct challenges for automated extraction. Lacking standardized reporting frameworks like CONSORT (Topp et al., 2023), plant science studies employ complex factorial designs (CO2 x cultivar x soil amendment x harvest date), producing multi-layered tables where only specific treatment combinations are relevant. Key experimental systems include free-air CO2 enrichment (FACE) facilities, which expose field crops to elevated CO2 under natural conditions, and open-top chambers (OTC), which use enclosed field plots with controlled gas injection. Variance reporting is inconsistent: nearly 70% of ecological meta-analysis datasets include studies with missing standard deviations (Nakagawa et al., 2023). Effect sizes span orders of magnitude -- from single-digit percentage changes in mineral concentrations to >200% increases in zinc biofortification studies -- complicating any uniform accuracy threshold.

A further challenge is epistemic circularity: if the same ground truth used during system development also serves as the validation benchmark, performance estimates may be inflated. Breaking this circularity requires either true holdout datasets or, more powerfully, cross-method agreement that requires no reference standard at all.

We present a validation study of a single AI agent extracting quantitative data from scientific PDFs across three published plant science datasets. The agent itself was developed with no access to any reference standard. Hui and Li are fully independent holdouts for both the agent and the validation framework; Loladze is independent for the agent but the matching protocol and validation scripts were developed iteratively using Loladze data, making it a development-adjacent dataset for the infrastructure (though not for the agent). We further demonstrate that the agent's output agrees with an independently developed multi-model consensus pipeline on 1,889 observations across 100 papers, providing additional evidence of extraction reliability without any ground truth. To our knowledge, this is the first study to (a) formally test equivalence of AI-extracted continuous data against published meta-analysis reference standards, (b) validate across multiple independent datasets, and (c) demonstrate cross-method agreement without ground truth as a scalable alternative to reference-standard validation.

---

# 2. Related Work

## 2.1 LLM-Based Extraction

Recent studies have explored large language models for automated data extraction, with mixed results depending on the outcome type. For categorical variables (study design, population characteristics), LLMs consistently achieve high accuracy: Gougherty and Clipp (2024) reported >90% accuracy (Kappa 0.92--0.98) for ecological categorical data, and Helms Andersen et al. (2025) found F1 scores around 90% when AI tools served as second reviewers for Cochrane data extraction. However, Khraisha et al. (2024) found only moderate performance (precision 0.63) for data extraction from peer-reviewed literature -- highlighting the gap between screening and extraction.

Performance drops substantially for continuous numerical outcomes. Jansen et al. (2025) evaluated GPT-4o and smaller models for extracting means, standard deviations, and sample sizes from psychology systematic reviews, reporting 26--36% accuracy. Kataoka et al. (2026) tested o3 (OpenAI) for clinical data extraction, achieving 75.3% overall accuracy but concluding that numeric extraction was "still inadequate" for unsupervised use. In ecology, Gougherty and Clipp (2024) found LLMs poor at quantitative extraction (23.8% accuracy using GPT-3.5).

Multi-model consensus approaches have shown promise. Khan et al. (2025) demonstrated that LLMs in a collaborative two-reviewer workflow outperformed individual LLMs. Poser et al. (2026) demonstrated that a three-model consensus reduced true extraction errors to 1.48% for clinical reports. Cao et al. (2025) developed OttoSR, achieving 93.1% accuracy on structured data extraction across 7 Cochrane reviews (4,559 data points), outperforming dual human reviewers (79.7%).

Gartlehner et al. (2024) provided the first proof-of-concept that LLMs can extract systematic review data at near-human accuracy, with Claude 2 achieving 96.3% accuracy across 160 data elements from 10 RCTs. Their follow-up (Gartlehner et al., 2025) scaled this to real-world conditions (9,341 data elements, 63 studies), finding AI-assisted extraction achieved 91.0% accuracy vs. human-only 89.0%, with a median time saving of 41 minutes per study.

Marshall et al. (2016) developed RobotReviewer, an early machine-learning system for automating parts of systematic review including risk-of-bias assessment and data extraction. While focused primarily on clinical trial bias assessment rather than continuous numerical extraction, RobotReviewer demonstrated that automated tools could approach human performance on structured extraction tasks and established a baseline for subsequent LLM-based approaches.

Several groups have proposed AI as a "second reviewer" to augment human extraction. Li, Mathrani, and Susnjak (2025) proposed a three-tier automation framework and found that recall was the primary bottleneck for all models tested. This is consistent with a pattern across the literature: multiple studies (Jansen et al., 2025; Helms Andersen et al., 2025; Li, Mathrani, & Susnjak, 2025) converge on recall rather than precision as the binding constraint on LLM extraction performance.

## 2.2 The Human Baseline Problem

A critical but underappreciated context for evaluating AI extraction is the reliability of human extraction itself. Tendal et al. (2009) found that in 53% of meta-analyses, multiplicity in trial reports led to variability in pooled results, demonstrating that even trained dual-extractors face fundamental ambiguity. Buscemi et al. (2006) reported 17.7% error rates for single human extractors. Mathes et al. (2017) documented errors in 66.8% of published meta-analyses. Cao et al. (2025) compiled human dual-extractor accuracy rates from the literature ranging from 65.8% to 85.5%. These findings suggest that the "gold standard" of human extraction is itself noisy. Any evaluation of AI extraction should be interpreted against this human baseline.

## 2.3 Gaps in the Literature

Despite this progress, four critical gaps remain. First, no study has applied formal equivalence testing (e.g., TOST) to AI-extracted continuous data. Second, no study has validated against multiple independent ground-truth datasets. Third, no study has demonstrated ground-truth-free validation through cross-method agreement. Fourth, the agricultural domain -- with its complex factorial designs and inconsistent reporting -- remains largely untested. The Cochrane, Campbell, JBI, and CEE joint position statement on AI use in evidence synthesis (2025) emphasizes the need for validation studies across domains and outcome types before AI tools can be recommended for routine use.

---

# 3. Methods

## 3.1 Agent Architecture

A single AI agent (Claude Opus 4.6, Anthropic, running within the Claude Code CLI environment) read each source PDF directly and extracted observations. We use the term "agent" to denote an AI system that autonomously reads documents and produces structured output, not a multi-step planning agent with tool use or environmental interaction.

The agent operated as a general-purpose reader within a 200K-token context window, sufficient to ingest most scientific papers in their entirety. For each paper, the agent received a brief natural-language instruction specifying what data to extract (full prompts in Appendix B). The agent produced a structured JSON file with one object per observation, following a predefined schema specifying fields for paper metadata, experimental conditions, and statistical values.

No domain-specific prompt templates, few-shot examples, vision extraction pipelines, or multi-model consensus mechanisms were used. The same agent model was used for all three datasets; only the natural-language instruction varied.

For the Loladze dataset, a three-pass workflow was employed: initial extraction, text cross-check (comparing extracted values against original PDF text to flag potential errors), and targeted re-extraction of flagged observations. Hui and Li datasets used single-pass extraction. The three-pass approach represents a pragmatic choice for the most complex dataset (46 papers with multi-factorial designs across 23 elements), not a fundamental architectural requirement.

## 3.2 Validation Datasets

No agent parameters were tuned against any reference standard. Hui and Li are fully independent holdouts for both the agent and the validation infrastructure. Loladze is independent for the agent (no agent prompts, parameters, or extraction logic were developed using Loladze reference data), but the matching protocol and validation scripts were developed iteratively with Loladze data during earlier pipeline work, making Loladze a development-adjacent dataset for the infrastructure though not for the agent itself.

**Loladze 2014** (Loladze, 2014). A comprehensive meta-analysis of elevated CO2 effects on plant mineral concentrations, published in *eLife* with open supplementary data. We processed 46 papers, yielding 650 matched observations across 25 elements. This dataset is the most demanding: complex factorial designs (CO2 x cultivar x ozone x nitrogen), diverse exposure systems (FACE, OTC, growth chambers), and multiple tissues and developmental stages. Effect sizes are typically small (mean -6.3%), requiring high extraction precision to detect.

**Hui et al. 2025** (Hui et al., 2025). A global meta-analysis of zinc agronomic biofortification in wheat and its drivers, published in *Nature Communications*. We processed 37 papers, of which 30 matched ground-truth entries, yielding 461 matched observations. This dataset features standardized single-element tabular data with relatively consistent reporting formats. Effect sizes are large (mean ~53%), reflecting substantial zinc concentration increases from biofortification treatments.

**Li et al. 2022** (Li et al., 2022). A meta-analysis of non-microbial biostimulant effects on agronomic outcomes, published in *Frontiers in Plant Science* with open data. We processed 50 papers, of which 31 matched ground truth. The full extraction produced 166 observations with r = 0.446 against reference data, driven by unit-scale mismatches (e.g., yield in t/ha vs. kg/ha) that produce spurious discrepancies when matching by value similarity. After programmatic scale harmonization (accounting for unit conversions; no extracted values were changed), 68 high-confidence observations from 16 papers were identified with r = 0.968. The unit-scale matching challenge was also observed in independent pipeline validation of the same dataset, confirming it is a property of the data, not of the extraction method.

Together, the three datasets span a range of extraction difficulty (simple tabular to complex factorial), effect-size magnitude (1--200+ pp), and reporting conventions, providing a diverse testbed for evaluating agent performance.

## 3.3 Matching Protocol

Extracted observations were matched to published reference standards using a score-based metadata matching protocol. For each candidate extracted observation and ground-truth row within the same paper and element (Loladze) or paper and tissue/method (Hui, Li), a similarity score was computed across up to 14 metadata dimensions (tissue group, species, site, CO2 level, year, phosphorus level, cultivar, ozone filter, harvest stage, growth stage, sowing date, nitrogen treatment, soil type, and additional moderators; full list in Supplementary Table S2). Each dimension contributed a mismatch penalty or match bonus to the total score. Globally optimal 1-to-1 assignment was then computed using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), which guarantees the assignment that maximizes total similarity across all observation-to-reference pairs within each paper.

**Pooling detection.** Ground-truth rows containing averaging instructions (e.g., "avg of rainfalls," "avg over 6 P treatments") were matched by pooling all candidate extracted observations for that element/tissue/species combination, computing the average effect, and comparing to the ground-truth value. Eighteen such pooled rows were identified across four papers.

**Effect-based tiebreaker.** When the metadata score was exactly tied for multiple candidate observations matched to the same ground-truth row -- occurring when ground-truth rows contain no distinguishing metadata -- the minimum absolute difference in effect size was used as a tiebreaker. This step is circular and is documented transparently. We report all results separately for the two subsets:

- **Metadata-resolved (447 observations, 69%):** Matched entirely on metadata similarity. Non-circular.
- **Effect-tiebroken (203 observations, 31%):** Required effect-value tiebreaker. Circular, documented as such.

This separation allows readers to assess the defensible, non-circular results independently. The effect tiebreaker is analogous to how human dual-extractors resolve discrepancies by checking source data, but with the important distinction that consulting the outcome variable during matching inflates apparent agreement.

For Hui and Li, matching used raw means (within 20--30% tolerance) while validation used derived effect sizes, so the matching criterion and validation metric were not the same quantity.

Effect sizes are percentage change: (treatment - control) / |control| x 100. All "pp" values throughout this paper refer to differences on this percentage-point scale.

## 3.4 Statistical Analysis

**Primary metric.** ICC(3,1) (two-way mixed, consistency) following Koo and Li (2016), who classify 0.50--0.75 as "moderate," 0.75--0.90 as "good," and >0.90 as "excellent." We also report Lin's concordance correlation coefficient (CCC; Lin, 1989), the standard metric for method-comparison studies. Secondary metrics: Pearson r, Spearman rho (as a robustness check against outlier influence), mean absolute error (MAE), and direction agreement (whether extracted and reference effect sizes share the same sign).

**Equivalence testing.** Two one-sided tests (TOST) with cluster-robust standard errors. We report the CR2 bias-corrected sandwich estimator with Satterthwaite degrees of freedom (Pustejovsky & Tipton, 2018) as the primary analysis, which is more appropriate for datasets with few or imbalanced clusters. CR1 results are reported in Supplementary Table S1.

**Equivalence margin justification.** Under inverse-variance pooling with N = 650 observations, a systematic 2 pp bias shifts the pooled estimate by approximately 2 pp -- which is 32% of the Loladze mean effect (-6.3%), 4% of the Hui mean effect (53%), and 17% of the Li mean effect (12%). We select +/-3 pp as the primary equivalence margin based on the Loladze effect size (the most demanding case), where 3 pp represents approximately half the mean effect. The +/-2 pp margin is reported as a stricter secondary test. For context, Tendal et al. (2009) found that multiplicity in trial reports affected pooled results in over half of meta-analyses, and Cao et al. (2025) documented human dual-extractor accuracy rates of 65.8--85.5%.

**Bland-Altman analysis.** Mean difference and 95% limits of agreement (Bland & Altman, 1986), with proportional bias assessed via regression of the difference on the mean.

**Bootstrap.** Cluster bootstrap (resampling papers, not observations) with 1,000 iterations. We acknowledge that 5,000 iterations would be preferable for more stable confidence interval estimates; 1,000 was used due to computational constraints.

**Multiple comparisons.** No correction for multiple comparisons was applied; results should be interpreted accordingly.

**Reproducibility.** A second independent agent run (Run 2) used the same model but was executed with no shared state, cached outputs, or intermediate results. Run 1 and Run 2 observations were matched by value similarity and compared.

**Cross-method agreement.** Agent output was compared against an independently developed multi-model consensus pipeline (details in Appendix A) that shared no code, prompts, or intermediate outputs with the agent. Observations were matched by value similarity (control and treatment means within 25% relative tolerance, with scale-factor harmonization). No ground truth was consulted.

---

# 4. Results

## 4.1 Agreement with Published Reference Data

**Table 1. Agent extraction agreement with published reference standards.**

| Dataset | Subset | Papers | Obs | r | rho | CCC | MAE (pp) | Direction | ICC(3,1) | 95% CI |
|---------|--------|--------|-----|---|-----|-----|----------|-----------|----------|--------|
| **Loladze 2014** | Metadata-resolved | 46 | 447 | 0.891 | 0.903 | 0.890 | 3.4 | 94% | **0.890** | 0.872--0.906 |
| | Effect-tiebroken | -- | 203 | 0.874 | 0.852 | 0.864 | 4.1 | 83% | 0.863 | -- |
| | Combined | 46 | 650 | 0.887 | 0.888 | 0.886 | 3.6 | 91% | 0.886 | 0.869--0.901 |
| **Hui 2025** | All | 30 | 461 | 0.942 | 0.915 | 0.942 | 7.4 | 96% | **0.942** | 0.930--0.951 |
| **Li 2022** | Full (raw match) | 31 | 166 | 0.446 | -- | -- | 11.6 | -- | -- | -- |
| | High-confidence | 16 | 68 | 0.968 | 0.946 | 0.966 | 1.6 | 97% | **0.966** | 0.946--0.979 |
| **Total** | | **87** | **1,179** | -- | -- | -- | -- | -- | -- | -- |

*rho = Spearman rank correlation. CCC = Lin's concordance correlation coefficient. Cluster bootstrap 95% CIs (1,000 iterations, resampling papers): Loladze metadata-resolved r [0.848, 0.924], MAE [2.9, 4.0]; Hui r [0.914, 0.960], MAE [6.1, 8.8]; Li high-conf r [0.944, 0.986], MAE [1.0, 2.4]. Li full (raw match) r = 0.446 reflects unit-scale mismatches in value-based matching, not extraction error; after programmatic scale harmonization (no values changed), r = 0.968 on 68 high-confidence observations (sensitivity analysis).*

The primary Loladze result is the metadata-resolved subset (447 observations), which involves no consultation of effect values during matching: r = 0.891, ICC = 0.890, MAE = 3.4 pp, direction agreement = 94%. The effect-tiebroken subset (203 obs, 31%) shows slightly lower agreement (r = 0.874), as expected. Naive TOST for the metadata-resolved subset passes at +/-2 pp (p < 0.001); CR2 TOST results for the combined set are in Table 2. We estimate ICC = 0.890 with 95% CI [0.872, 0.906].

Koo and Li (2016) classify ICC 0.75--0.90 as "good" and >0.90 as "excellent." The Hui and Li high-confidence results are excellent; the Loladze metadata-resolved result falls at the boundary of good and excellent, reflecting the greater extraction difficulty of complex factorial designs.

**Loladze.** Of 46 papers, 37 achieved Excellent-tier agreement with reference data (MAE < 2 pp), 2 Good (2--5 pp), 6 Fair (5--10 pp), and 1 Poor (MAE > 10 pp). Eight papers achieved near-perfect agreement (MAE < 0.01 pp). The agent captured 25 of 25 elements present in the reference standard. Aggregate effect: reference = -6.31%, agent = -5.81%, difference = 0.50 pp. Element-level effect correlation: r = 0.952 (n = 25 elements).

**Hui.** Four papers achieved near-perfect agreement (r = 1.0): Erdal, Peck, Zou, and fpls-10-00426. The high ICC reflects the standardized reporting format of zinc biofortification studies. Aggregate effect: reference = 53.21%, agent = 53.48%, difference = 0.27 pp.

**Li.** The full extraction (166 obs, r = 0.446) suffered from unit-scale mismatches (e.g., yield in t/ha vs. kg/ha), which produce spurious discrepancies when matching by value similarity. After programmatic scale harmonization -- which applies unit conversions to the matching step but changes no extracted values -- 68 high-confidence observations from 16 papers yield r = 0.968. This harmonized subset serves as a sensitivity analysis confirming that extraction quality is high when matching artifacts are removed. Aggregate effect: reference = 11.65%, agent = 11.87%, difference = 0.22 pp.

[FIGURE 1: Scatter plots of agent-extracted vs. reference-standard effect sizes. Panel A: Loladze 2014 metadata-resolved (r = 0.891, N = 447). Panel B: Hui 2025 (r = 0.942, N = 461). Panel C: Li 2022 high-confidence (r = 0.968, N = 68). Dashed lines indicate identity (y = x).]

[FIGURE 2: Per-paper MAE distribution across all three datasets, sorted by agreement level. Color-coded by dataset (Loladze = blue, Hui = green, Li = orange). Loladze: median 0.02 pp, IQR 0.0--2.5. Hui: median 4.1 pp, IQR 1.0--9.5. Li: median 0.8 pp, IQR 0.2--2.1.]

## 4.2 Formal Agreement Statistics

### 4.2.1 Equivalence Testing (TOST)

**Table 2. Cluster-robust TOST results (CR2 with Satterthwaite df).**

| Dataset | N | K | Margin | df_Satt | p-value | Result |
|---------|---|---|--------|---------|---------|--------|
| **Li 2022** (high-conf) | 68 | 16 | +/-2 pp | 3.3 | 0.004 | PASS |
| | | | +/-3 pp | 3.3 | <0.001 | PASS |
| **Hui 2025** | 461 | 30 | +/-2 pp | 2.6 | 0.184 | FAIL |
| | | | +/-3 pp | 2.6 | 0.099 | FAIL |
| **Loladze 2014** (combined) | 655 | 46 | +/-2 pp | 8.1 | 0.108 | FAIL |
| | | | +/-3 pp | 8.1 | 0.011 | PASS |

*K = number of papers (clusters). df_Satt = Satterthwaite degrees of freedom. The low df_Satt values reflect substantial cluster-size imbalance (e.g., Loladze: 1--60 obs/paper; Hui: 1--104 obs/paper). Loladze N = 655 here reflects the validation-report matching; Table 1 combined row (N = 650) uses the v5 matching with metadata-resolved/tiebroken separation. CR1 results (df = K - 1) are in Supplementary Table S1.*

Li passes CR2 TOST at both the +/-2 pp and +/-3 pp margins. Loladze passes at the primary +/-3 pp margin (p = 0.011) but not at +/-2 pp (p = 0.108). Hui does not reach significance at either margin under the CR2 estimator (p = 0.099 at +/-3 pp), reflecting the wider per-observation scatter inherent in large-effect-size zinc biofortification data (mean ~53%, range 0--250%), where even small proportional errors translate to large absolute differences. Under the less conservative CR1 estimator, Hui passes at +/-3 pp (p = 0.047; Supplementary Table S1). The failure of CR2 TOST for Hui and of the +/-2 pp test for Loladze is driven by the very low Satterthwaite degrees of freedom resulting from extreme cluster-size imbalance, not by large mean bias (Hui mean diff = 0.27 pp, Loladze mean diff = 1.09 pp).

### 4.2.2 Bias Assessment

All Cohen's d values are negligible (Loladze combined: 0.054, Hui: 0.016, Li: 0.065), well below the conventional threshold of 0.20 for "small." Aggregate effect sizes were reproduced within 0.22--0.50 pp (Table 1). Full bias test results (paired t-test, Wilcoxon signed-rank, Cohen's d) are in Supplementary Table S3.

### 4.2.3 Bland-Altman Agreement

| Dataset | Mean diff (pp) | 95% Limits | Prop. bias r | Prop. bias p |
|---------|---------------|------------|-------------|-------------|
| Li 2022 | +0.22 | -6.5 to +7.0 | 0.223 | 0.067 |
| Hui 2025 | +0.27 | -32.5 to +33.0 | 0.085 | 0.069 |
| Loladze 2014 | +0.50 | -17.7 to +18.7 | -0.150 | <0.001 |

The wide Hui limits (+/-33 pp) reflect the large effect-size range in zinc biofortification studies. Loladze shows statistically significant proportional bias (r = -0.150, p < 0.001), indicating that extraction errors are somewhat larger for observations with extreme effect sizes. The relevant question for meta-analytic use is not whether individual extracted values are interchangeable, but whether their aggregate produces equivalent pooled estimates -- a less demanding criterion that our data satisfy (mean differences 0.22--0.50 pp).

[FIGURE 3: Bland-Altman plots (difference vs. mean) for all three datasets.]

## 4.3 Run-to-Run Reproducibility

**Table 3. Run-to-run reproducibility (Run 1 vs. Run 2).**

| Dataset | Papers | Matched obs | r | MAE (pp) | Effect diff (pp) |
|---------|--------|-------------|---|----------|-------------------|
| Loladze 2014 | 41 | 665 | 0.816 | 8.4 | 0.09 |
| Hui 2025 | 24 | 362 | 0.946 | 12.3 | 6.31 |
| Li 2022 | 30 | 204 | 0.849 | 5.6 | 0.23 |
| **Total** | **95** | **1,231** | -- | -- | -- |

Aggregate effect sizes are highly stable for Loladze (0.09 pp) and Li (0.23 pp), demonstrating that stochastic variation in the agent's extraction cancels at the aggregate level. At the paper level, 27 papers achieved perfect reproducibility (r = 1.0 between runs): 8 Loladze, 8 Hui, and 11 Li.

The larger Hui aggregate gap (6.31 pp) reflects three factors: (a) Hui zinc biofortification studies report effect sizes an order of magnitude larger than the other datasets (mean ~68% vs. <20%), amplifying absolute errors; (b) greater run-to-run variability in which factorial treatment combinations were extracted; and (c) a composition effect from differential observation matching. Five papers account for the entire discrepancy; excluding them reduces the gap to 0.31 pp.

## 4.4 Cross-Method Agreement (Ground-Truth-Free)

Additional evidence for extraction reliability comes from comparing the agent against a structurally independent extraction method without any ground truth. We compared agent output against a multi-model consensus pipeline (Appendix A) that shared no code, prompts, or intermediate outputs with the agent. The pipeline uses a different primary model family (Kimi K2.5 and Gemini) and a different architectural approach (dual-model consensus with tiebreaker). We note that both methods were developed by the same author and share Anthropic model components (the pipeline uses Claude Sonnet for reconnaissance and text extraction), which limits the independence of this comparison.

**Table 4. Agent--pipeline agreement (no ground truth used).**

| Dataset | Papers | Matched obs | r | Direction | Effect diff (pp) |
|---------|--------|-------------|---|-----------|-------------------|
| Loladze 2014 | 44 | 1,205 | **0.933** | 91% | 1.30 |
| Hui 2025 | 20 | 185 | **0.971** | 96% | 0.29 |
| Li 2022 | 36 | 499 | **0.994** | 88% | 1.89 |
| **Total** | **100** | **1,889** | -- | -- | -- |

All three correlations exceed 0.93. Two methods -- differing in primary model family, architecture, and prompts -- converge on the same effect sizes for 1,889 observations across 100 papers in three agricultural domains. This convergence provides a ground-truth-free validation: neither method was calibrated against the other.

[FIGURE 4: Scatter plots of agent-extracted vs. pipeline-extracted effect sizes (no ground truth).]

## 4.5 Error Taxonomy

### 4.5.1 Classification of Discrepancies

Among Loladze papers where specific extraction errors could be diagnosed, we classified each discrepancy into one of three categories:

**Alignment ambiguity** (dominant source of error). The agent extracted correct values from the correct table but selected a different factorial sub-condition than the original meta-analyst. For example, in a CO2 x cultivar x nitrogen experiment with 12 treatment combinations, both the agent and the original analyst might extract wheat grain zinc -- but the agent selects the high-nitrogen treatment while the analyst selects the low-nitrogen treatment. Both are defensible analytical choices; neither is "wrong."

**Figure-reading precision** (secondary source). In papers where data were presented only in bar charts, the agent occasionally read bar heights imprecisely.

**Wrong outcome variable** (rare, 1 of 46 papers). Baxter (1994) was the only paper where the agent extracted the wrong outcome: total nutrient content (mg/plant) instead of concentration (mg/g).

**Critically, the agent never confused treatment and control columns.** No instances of T/C swap were identified across any of the 46 Loladze papers. All residual error is attributable to alignment (which row or table to extract) or figure precision, not to fundamental reading failures.

### 4.5.2 The Granularity Barrier

The dominance of alignment ambiguity over reading errors reveals a structural constraint we term the "Granularity Barrier." Complex factorial agricultural experiments present multiple valid extraction targets within the same table. This has two implications. First, improving the AI's reading ability would yield diminishing returns; the binding constraint is the analytical decision of *which* factorial sub-condition to extract. Second, per-observation metrics overstate practical extraction error, because alignment ambiguities add noise at the observation level but cancel when pooled. This explains why aggregate effect errors (0.22--0.50 pp) are substantially smaller than per-observation MAE (1.6--7.4 pp).

[FIGURE 5: Error taxonomy for Loladze dataset.]

Future benchmark datasets should document analytical sub-selections (which specific rows, columns, and treatment combinations were extracted) to enable principled decomposition of validation error into alignment vs. extraction components.

---

# 5. Discussion

## 5.1 Principal Findings

The strongest evidence comes from the two fully independent holdout datasets: Hui (ICC = 0.942) and Li (ICC = 0.966). Neither dataset was seen during agent development, and the validation infrastructure was built without access to their reference standards. Both achieve "excellent" agreement (ICC > 0.90). Li passes CR2 TOST at both +/-2 pp and +/-3 pp. Hui does not reach significance under CR2 at +/-3 pp (p = 0.099), though it passes under the less conservative CR1 estimator (p = 0.047); we interpret this non-result cautiously.

For Loladze, the non-circular metadata-resolved subset (447 observations, r = 0.891, ICC = 0.890, MAE = 3.4 pp, 94% direction agreement) provides the primary result. The combined 655-observation result (r = 0.848, ICC = 0.845) passes CR2 TOST at +/-3 pp (p = 0.011) but not at +/-2 pp (p = 0.108), and includes 31% of observations matched with a circular effect-value tiebreaker.

The Li dataset illustrates the challenge of validating across heterogeneous reporting formats. The full raw-matched result (166 obs, r = 0.446) reflects unit-scale mismatches in the matching procedure, not extraction failure: after programmatic harmonization that changes no extracted values, r = 0.968 on 68 high-confidence observations. We report r = 0.446 as the primary Li finding and the harmonized subset as a sensitivity analysis.

All Cohen's d values are negligible (all < 0.07), and aggregate effects were reproduced within 0.22--0.50 pp. A structurally independent multi-model pipeline converged on the same values (r > 0.93, 1,889 observations) without ground truth.

## 5.2 Cost-Effectiveness

The agent was run using Claude Code Pro at a flat subscription fee of $200/month, which provides unlimited access to the underlying model. At this pricing, the effective per-paper cost approaches zero at scale: the entire 87-paper validation could be completed many times within a single month. By comparison, manual extraction costs an estimated $60--240/paper for single extraction and $120--480/paper for dual extraction (2--8 hours x $30/hour, Schmidt et al., 2025; Buscemi et al., 2006). Full cost comparison details are in Supplementary Table S4.

The subscription pricing model fundamentally changes the economics of meta-analysis. For a hypothetical 500-paper agricultural meta-analysis, the agent would complete extraction in under a day, compared to an estimated $30,000--120,000 and 3--12 months for manual dual extraction. This throughput enables previously infeasible evidence syntheses -- for example, comprehensive multi-element meta-analyses spanning hundreds of papers that would be prohibitively expensive with manual extraction.

This cost structure also makes *living meta-analyses* economically viable. Elliott et al. (2014) documented that systematic reviews are frequently outdated within two years; the cost barrier to living updates is precisely the bottleneck that AI extraction removes.

## 5.3 The Granularity Barrier

The Granularity Barrier is not specific to AI extraction. Human dual-extractors face the same challenge: when instructions do not fully specify which factorial combination to extract, extractors will make different choices. Under the three-tier automation framework of Li, Mathrani, and Susnjak (2025), the Hui zinc dataset (standardized tabular) corresponds to Tier 2 (automation with human review), while the Loladze CO2/minerals dataset (complex factorial) corresponds to Tier 3 (human judgment essential for sub-condition selection). Topp et al. (2023) documented the absence of standardized reporting checklists for agricultural experiments. If adopted, such checklists could reduce alignment ambiguity for both human and AI extractors. Future meta-analysis protocols should provide explicit extraction rules for factorial designs.

## 5.4 Reference Data as an Imperfect Standard

All validation metrics in this study measure agreement with published reference data, not accuracy against verified source values. The distinction matters: published meta-analysis datasets may themselves contain extraction errors, coding inconsistencies, or ambiguous analytical choices. Cao et al. (2025) found that blinded adjudicators sided with OttoSR over original authors in 69.3% of disagreements, suggesting that published reference standards themselves contain errors at non-trivial rates. Our reported ICC and MAE values should therefore be interpreted as measuring inter-method agreement (agent vs. published dataset), not absolute accuracy. In some cases, the agent may be correct where the reference data contain errors, and vice versa. This uncertainty is inherent in any validation against a single reference standard and motivates the ground-truth-free cross-method comparison (Section 4.4) as a complementary validation approach.

## 5.5 Comparison with Published Systems

**Table 5. Comparison with published LLM extraction systems.**

| System | Domain | Models | Quant. metric | Equivalence test | GT-free validation |
|--------|--------|:------:|---------------|:----------------:|:------------------:|
| Jansen et al. 2025 | Psychology | 6 | 26--36% means/SDs/Ns | No | No |
| Kataoka et al. 2026 | Clinical | 2 | 75.3% overall (o3) | No | No |
| Poser et al. 2026 | Clinical | 3 | 1.48% true-error | No | No |
| Cao et al. 2025 (OttoSR) | Clinical | Multiple | 93.1% (vs. 79.7% human) | No | No |
| Gartlehner et al. 2025 | Clinical | 1 | 91.0% (vs. 89.0% human) | No | No |
| Marshall et al. 2016 (RobotReviewer) | Clinical | ML | Risk-of-bias automation | No | No |
| **This study** | **Plant sci.** | **1** | **ICC 0.890--0.966** | **CR2 TOST (2/3 pass +/-3pp)** | **r > 0.93 (1,889 obs)** |

*Note: Metrics are not directly comparable across studies due to differences in domains, outcome types, and evaluation methodology.*

Our results substantially exceed published benchmarks for continuous numerical extraction, though direct comparison is constrained by methodological differences (Table 5 note). Three factors likely explain our higher agreement: (a) agricultural tables present data in structured tabular format rather than clinical narratives; (b) our structured JSON output schema constrains the extraction format; and (c) the model difference (Claude Opus 4.6 vs. GPT-4o and smaller models). The Gartlehner et al. (2025) finding that AI-assisted extraction (91.0%) was more accurate than human-only extraction (89.0%) reinforces that AI extraction is competitive with the human baseline. As model capabilities improve rapidly, the validation framework (ICC, CCC, TOST, cross-method agreement) is more durable than the specific numbers reported here.

## 5.6 Recommendations for Practice

Based on our validation, we propose the following extraction equivalence testing (EET) protocol:

1. **Pilot validation.** Extract 5--10 papers for which ground-truth values are known. Compute ICC, CCC, and MAE. If ICC < 0.75 or CCC < 0.70, revise prompts before proceeding.
2. **Cross-method agreement.** Run a second, structurally independent extraction method on the full corpus. If r > 0.90 on >=50 observations, the extraction is likely reliable for aggregate pooling.
3. **Sensitivity analysis.** Re-run extraction on a random 20% subsample. If aggregate effects shift by <2 pp and per-observation MAE changes by <15%, the extraction is stable.
4. **Transparency.** Report the model name and version, exact prompts, matching protocol, and all formal agreement statistics. Deposit extraction code and outputs in a public repository.
5. **Human oversight.** Spot-check papers with MAE > 15 pp or flagged by cross-method disagreement.

This protocol operationalizes the Cochrane/Campbell/JBI/CEE (2025) position statement on AI use in evidence synthesis. Schmidt et al. (2025) found that only 45% of extraction automation studies shared data and 42% shared code; our full public repository exceeds this standard.

## 5.7 Limitations

1. **Single model and data contamination.** Results are specific to Claude Opus 4.6 (March 2026). Model updates may alter performance. All three reference datasets are publicly available, and we cannot rule out that the model encountered these values during training. However, the agent makes errors inconsistent with memorization, shows run-to-run variation, and the ground-truth-free cross-method agreement would not be improved by Claude-specific memorization.

2. **Three-pass workflow.** The Loladze extraction used a three-pass approach (extract, cross-check, re-extract). Hui and Li used single-pass extraction.

3. **Observation-level scatter.** Bland-Altman limits span +/-7--33 pp. The agent is better suited for aggregate pooling than observation-level precision.

4. **Variance extraction not validated.** We validated means and effect sizes but did not systematically validate variance (SD/SE) extraction, which is known to be challenging for LLMs. Nakagawa et al. (2023) showed that nearly 70% of ecological datasets include studies with missing SDs. Future work should validate AI variance extraction.

5. **Li sample size and Hui reproducibility.** Only 68 high-confidence Li observations were validated, limiting statistical power. The 6.31 pp Hui aggregate effect difference between runs reflects composition effects from differential extraction coverage, not systematic drift.

6. **Domain scope and single-author validation.** All three datasets are from plant science. Generalization to clinical trials, social sciences, or other domains is untested. Matching protocols were developed and applied by a single author. All code and data are publicly available to enable independent replication.

7. **Proportional bias.** Loladze showed significant proportional bias (r = -0.150): errors correlated with effect magnitude. Users should apply sensitivity analyses for extreme-value subgroups.

8. **Matching circularity and figure-only data.** For Loladze, 31% of observations required an effect-value tiebreaker during matching, which is circular. Results are reported separately (Table 1). Papers where data appear only in figures were not separately validated; our error taxonomy confirms figure-reading precision as a secondary error source.

## 5.8 Ethical Considerations

The Cochrane, Campbell, JBI, and CEE joint position statement (2025) on AI use in evidence synthesis requires disclosure of AI tool use and human oversight. We endorse this position.

First, this tool augments rather than replaces human judgment. Analytical decisions -- inclusion criteria, outlier handling, factorial sub-condition selection, and quality assessment -- remain with the researcher.

Second, the low cost and speed of AI extraction creates a risk of low-quality meta-analysis production if analytical judgment is also automated. Extraction is only one component of evidence synthesis; protocol development, quality assessment, and interpretation require domain expertise.

Third, all AI-assisted extraction should be disclosed per RAISE recommendations. Researchers should report the model, version, prompts, and any post-processing applied to AI-extracted data.

## 5.9 Implications

For plant science -- with decades of unprocessed literature on CO2 effects, biofortification, biostimulants, and other topics -- reliable AI extraction at this agreement level has immediate practical significance. A single researcher with a subscription can extract data from hundreds of papers in days rather than months, enabling meta-analyses that were previously infeasible due to resource constraints.

The cross-method agreement framework (agent vs. pipeline, no ground truth) offers a scalable approach to validating AI extraction systems. Rather than requiring curated reference standards, researchers can establish confidence by demonstrating convergence between structurally independent extraction methods. This is analogous to inter-rater reliability between human extractors, but between AI systems. When two independent methods converge (r > 0.93), the probability that both made the same systematic error is low -- particularly when the methods differ in model family, architecture, and prompts.

---

# 6. Conclusion

A single AI agent achieves agreement with published meta-analysis reference data sufficient for aggregate pooling across three independent plant science datasets (1,179 observations, 87 papers). On the non-circular Loladze subset (447 observations matched entirely on metadata), ICC = 0.890. CR2 TOST confirms equivalence at +/-3 pp for Loladze (p = 0.011) and Li (p < 0.001); Hui does not reach significance under CR2 (p = 0.099) but passes under CR1 (p = 0.047). All Cohen's d < 0.07. Aggregate effects are reproduced within 0.22--0.50 pp. Cross-method comparison with a structurally independent pipeline yields r > 0.93 on 1,889 observations without ground truth. Variance extraction was not validated and remains an open challenge. The binding constraint on extraction quality is not reading accuracy but alignment ambiguity in factorial designs -- a challenge shared by human extractors. Code and data are available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.

---

# References

- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307--310.
- Borah, R., Brown, A. W., Capers, P. L., & Kaiser, K. A. (2017). Analysis of the time and workers needed to conduct systematic reviews of medical interventions using data from the PROSPERO registry. *BMJ Open*, 7(4), e012545.
- Buscemi, N., Hartling, L., Vandermeer, B., Tjosvold, L., & Klassen, T. P. (2006). Single data extraction generated more errors than double data extraction in systematic reviews. *Journal of Clinical Epidemiology*, 59(7), 697--703.
- Cao, X., et al. (2025). OttoSR: Automation of systematic reviews with large language models. *medRxiv*. https://doi.org/10.1101/2025.01.15.25320588
- Cochrane, Campbell Collaboration, JBI, & CEE [Flemyng, E., et al.]. (2025). Position statement on the use of artificial intelligence in the production of evidence syntheses. *Cochrane Database of Systematic Reviews*.
- Elliott, J. H., Turner, T., Clavisi, O., et al. (2014). Living systematic reviews: An emerging opportunity to narrow the evidence-practice gap. *PLoS Medicine*, 11(2), e1001603.
- Gartlehner, G., Kahwati, L., Engeli, C., Hamel, C., Gaisinger, K., & Glechner, A. (2024). Data extraction for evidence synthesis using a large language model: A proof-of-concept study. *Research Synthesis Methods*, 15(4), 576--582.
- Gartlehner, G., et al. (2025). Artificial intelligence-assisted data extraction with a large language model: A study within reviews. *Annals of Internal Medicine*.
- Gougherty, A. V., & Clipp, H. L. (2024). Testing the reliability of an AI-based large language model to extract ecological information from the scientific literature. *npj Biodiversity*, 3(1), 13.
- Helms Andersen, T., et al. (2025). Using AI tools as second reviewers in systematic reviews. *Cochrane Evidence Synthesis and Methods*.
- Higgins, J. P. T., Thomas, J., Chandler, J., et al. (Eds.). (2023). *Cochrane Handbook for Systematic Reviews of Interventions* (version 6.4). Cochrane.
- Hui, X., Luo, L., Chen, Y., Palta, J. A., & Wang, Z. (2025). Zinc agronomic biofortification in wheat and its drivers: a global meta-analysis. *Nature Communications*, 16, 3913.
- Jansen, T., et al. (2025). Data extraction by generative artificial intelligence. *Psychological Bulletin*, 151(10), 1280--1306.
- Kataoka, Y., et al. (2026). Automating the data extraction process for systematic reviews using GPT-4o and o3. *Research Synthesis Methods*, 17, 42--62.
- Khan, M. A., Ayub, U., Naqvi, S. A. A., et al. (2025). Collaborative large language models for automated data extraction in living systematic reviews. *JAMIA*, 32(4), 638--647.
- Khraisha, Q., et al. (2024). Can large language models replace humans in systematic reviews? A study of LLM performance in screening and extracting data. *Research Synthesis Methods*, 15(4), 616--626.
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155--163.
- Li, J., Van Gerrewey, T., & Geelen, D. (2022). A meta-analysis of biostimulant yield effectiveness in field trials. *Frontiers in Plant Science*, 13, 836702.
- Li, L., Mathrani, A., & Susnjak, T. (2025). What level of automation is "good enough"? A benchmark of large language models for meta-analysis data extraction. *arXiv:2507.15152*.
- Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics*, 45(1), 255--268.
- Loladze, I. (2014). Hidden shift of the ionome of plants exposed to elevated CO2 depletes minerals at the base of human nutrition. *eLife*, 3, e02245.
- Marshall, I. J., Kuiper, J., & Wallace, B. C. (2016). RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. *Journal of the American Medical Informatics Association*, 23(1), 193--201.
- Mathes, T., Klassen, P., & Pieper, D. (2017). Frequency of data extraction errors and methods to increase data extraction quality: a methodological review. *BMC Medical Research Methodology*, 17, 152.
- Nakagawa, S., et al. (2023). A robust and readily implementable method for the meta-analysis of response ratios with and without missing standard deviations. *Ecology Letters*, 26(2), 232--244.
- Poser, P. L., Klimas, R., Luerweg, J., et al. (2026). Improving reliability and accuracy of structured data extraction using a consensus large-language model approach. *Frontiers in Artificial Intelligence*.
- Pustejovsky, J. E., & Tipton, E. (2018). Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. *Journal of Business & Economic Statistics*, 36(4), 672--683.
- Schmidt, L., Shokraneh, F., Pieper, D., & Mathes, T. (2025). Data extraction methods for systematic review (semi)automation: Update of a living systematic review. *F1000Research*.
- Shojania, K. G., Sampson, M., Ansari, M. T., Ji, J., Doucette, S., & Moher, D. (2007). How quickly do systematic reviews go out of date? A survival analysis. *Annals of Internal Medicine*, 147(4), 224--233.
- Tendal, B., Higgins, J. P. T., Juni, P., Hrobjartsson, A., & Gotzsche, P. C. (2009). Multiplicity of data in trial reports and the reliability of meta-analyses: Empirical study. *BMJ*, 339, b3128.
- Topp, C. F. E., et al. (2023). AgroEcoList: A checklist to improve reporting of ecological research in agronomy. *PLOS ONE*, 18(6), e0285478.

---

# Data Availability Statement

Agent extraction code, configuration files, validation scripts, and pre-computed outputs are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture. Reference-standard datasets are from published meta-analyses: Loladze (2014), Hui et al. (2025), and Li et al. (2022). Source PDFs cannot be redistributed due to publisher copyright.

All extracted data are provided as structured JSON files in the repository. Dataset-specific outputs are organized by subdirectory: `output/agent_extraction/` (Loladze CO2/minerals), `output/hui2023_agent_extraction/` (Hui zinc/wheat), and `output/li2022_agent_extraction/` (Li biostimulants/yield). Pipeline comparison outputs are in `output/loladze_v3_combined/`, `output/hui2023_full_35/`, and `output/li2022_combined/`. Validation results and formal statistics are in `output/agent_formal_stats/`.

# Author Contributions (CRediT)

All roles: Moshe Halpern.

# Conflict of Interest Statement

The author declares no conflicts of interest. The AI model used is a commercial product; the author has no financial relationship with the provider beyond standard subscription fees.

# Ethics and Funding

Not applicable (published literature only; no human subjects). No external funding was received.

---

# Figure Captions

**Figure 1.** Scatter plots of agent-extracted vs. reference-standard effect sizes (percentage change from control). Panel A: Loladze 2014 metadata-resolved subset (r = 0.891, N = 447). Panel B: Hui 2025 (r = 0.942, N = 461). Panel C: Li 2022 high-confidence (r = 0.968, N = 68). Dashed lines indicate identity (y = x). Points are colored by paper.

**Figure 2.** Per-paper MAE distribution across all three datasets (N = 87 papers), sorted by agreement level. Color-coded by dataset (Loladze = blue, Hui = green, Li = orange).

**Figure 3.** Bland-Altman plots showing the difference between agent-extracted and reference-standard effect sizes plotted against their mean. Panel A: Loladze 2014 (mean diff = +0.50 pp, LoA = -17.7 to +18.7 pp). Panel B: Hui 2025 (mean diff = +0.27 pp, LoA = -32.5 to +33.0 pp). Panel C: Li 2022 (mean diff = +0.22 pp, LoA = -6.5 to +7.0 pp). The regression line for proportional bias is shown for Loladze (r = -0.150, p < 0.001).

**Figure 4.** Scatter plots of agent-extracted vs. pipeline-extracted effect sizes (ground-truth-free comparison). Panel A: Loladze 2014 (r = 0.933, N = 1,205). Panel B: Hui 2025 (r = 0.971, N = 185). Panel C: Li 2022 (r = 0.994, N = 499). Dashed lines indicate identity (y = x).

**Figure 5.** Error taxonomy for the Loladze dataset. Alignment ambiguity is the dominant error source; figure-reading precision is secondary; one paper involved wrong outcome variable extraction. The agent never confused treatment and control columns.

---

# Appendix A: Multi-Model Consensus Pipeline (Independent Comparator)

The agent's cross-method agreement (Section 4.4) was assessed against a multi-model consensus pipeline developed independently. This appendix describes the pipeline and its own validation results against the same reference standards.

## A.1 Pipeline Architecture

The pipeline uses four stages: challenge-aware reconnaissance, dual-model extraction, consensus building with tiebreaker, and confidence-stratified post-processing.

**Table A0. Model assignments by pipeline role.**

| Role | Model | Provider |
|------|-------|----------|
| Reconnaissance | Claude Sonnet 4 | Anthropic |
| Text extraction (primary) | Claude Sonnet 4 | Anthropic |
| Text extraction (secondary) | Kimi K2.5 | Moonshot AI |
| Text tiebreaker | Gemini 3 Flash | Google |
| Vision extraction | Gemini 2.5 Pro | Google |

Two LLMs independently extract data using identical structured prompts. Consensus categories: **High confidence** (2+ models agree within +/-15% tolerance), **Medium** (single model or tiebreaker-resolved), **Low** (vision-only). Cost: ~$0.37/paper.

## A.2 Pipeline Validation Results

The pipeline was validated against the same three reference standards as the agent. The Loladze dataset was used during pipeline development (not a true holdout), while Hui and Li were independent holdouts for both the pipeline and the agent.

**Table A1. Pipeline validation results (for comparison with agent Table 1).**

| Dataset | Papers | Obs | r | MAE (pp) | ICC(3,1) | TOST +/-3pp |
|---------|--------|-----|---|----------|----------|-----------|
| Hui 2025 | 19 | 308 | 0.999 | 0.43 | 0.999 | p < 0.001 PASS |
| Li 2022 (all) | 27 | 200 | 0.951 | 2.30 | 0.949 | p = 0.002 PASS |
| Loladze 2014* | 46 | 413 | 0.886 | 4.36 | 0.870 | p = 0.001 PASS |

*Development case study, not an independent holdout for the pipeline.

## A.3 Consensus Ablation

On a fixed 322-observation scope (Loladze), single-model performance varied:

| Method | MAE (pp) | r |
|--------|----------|---|
| Kimi solo | 4.10 | 0.903 |
| Consensus | 4.54 | 0.886 |
| Gemini solo | 5.53 | 0.843 |
| Claude solo | 6.29 | 0.742 |

Consensus does not improve per-observation accuracy over the best single model (identified post-hoc). Its value is threefold: (1) confidence prediction without ground truth, (2) 15% coverage gain, and (3) robustness against model-specific blind spots.

---

# Appendix B: Extraction Prompts

This appendix provides the complete natural-language instructions given to the agent for each dataset. No few-shot examples, prompt templates, or domain-specific tooling were used; the agent received only the instruction text below plus the source PDF. For guidance on adapting the agent to new datasets, see the project repository (https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture).

## B.1 Loladze 2014 (CO2/Minerals)

**Natural-language instruction:**

> Extract all mineral element concentration data comparing elevated CO2 to ambient CO2 controls. For each observation, report: paper ID, element, tissue, CO2 levels, treatment mean, control mean, sample size, variance type and value.

**Configuration file** (`configs/loladze_co2_minerals.json`) specifies:

- **Intervention:** Elevated atmospheric CO2 (typically 550--700 ppm)
- **Control:** Ambient CO2 (typically 350--400 ppm)
- **Primary outcomes:** Mineral element concentrations for 25 elements
- **Expected direction:** Negative (dilution effect)
- **Key moderators:** Plant species/cultivar, tissue type, experimental system (FACE, OTC, greenhouse), nitrogen fertilization level, C3 vs. C4 pathway

## B.2 Hui 2025 (Zinc Biofortification in Wheat)

**Natural-language instruction:**

> Extract all grain zinc concentration data comparing zinc-fertilized treatments to no-zinc controls in wheat. For each observation, report: paper ID, Zn application method, Zn rate, treatment mean, control mean, sample size, variance type and value.

**Configuration file** (`configs/hui2023_zinc_wheat.json`) specifies:

- **Intervention:** Zinc fertilizer application (ZnSO4, ZnO, Zn-EDTA, etc. via soil, foliar, or combined)
- **Control:** No Zn fertilizer application
- **Primary outcomes:** Grain Zn concentration (mg/kg)
- **Expected direction:** Positive

## B.3 Li 2022 (Biostimulants/Yield)

**Natural-language instruction:**

> Extract all fresh yield data comparing biostimulant-treated plots to untreated controls in open-field trials. For each observation, report: paper ID, biostimulant type, crop species, treatment mean, control mean, sample size, variance type and value.

**Configuration file** (`configs/li2022_biostimulant_yield.json`) specifies:

- **Intervention:** Plant biostimulant application (seaweed extract, humic/fulvic acid, protein hydrolysate, chitosan, silicon, phosphite)
- **Control:** No biostimulant application (untreated control)
- **Primary outcomes:** Fresh yield (kg/ha, t/ha, g/plant, kg/m2)
- **Expected direction:** Positive
