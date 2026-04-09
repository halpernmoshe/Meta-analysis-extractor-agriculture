# Breaking the Extraction Bottleneck: A Single AI Agent Achieves Equivalence with Published Meta-Analysis Data Across Three Agricultural Datasets

**Moshe Halpern** ^ORCID^

Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization — Volcani Center, Israel

*Correspondence:* Moshe Halpern, Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization — Volcani Center, Rishon LeZion 7505101, Israel.

---

# Abstract

**Background:** Data extraction is the primary bottleneck in meta-analysis, requiring an average of 67.3 weeks per systematic review (Borah et al., 2017) with 17.7% error rates for single extractors (Buscemi et al., 2006) and errors detected in 66.8% of published meta-analyses (Mathes et al., 2017). Large language models achieve high accuracy on categorical variables (>90% for ecological data, Gougherty & Clipp, 2024; F1 ~90% for Cochrane fields, Helms Andersen et al., 2025) but only 26--36% on continuous outcomes (Jansen et al., 2025). No study has validated AI-extracted continuous data against multiple independent ground-truth datasets with formal equivalence testing.

**Methods:** A general-purpose AI agent (Claude Opus 4.6) extracted quantitative data from source PDFs across three published plant science meta-analyses: Loladze 2014 (CO2/minerals, 46 papers, 650 observations), Hui 2025 (zinc biofortification, 25 papers, 461 observations), and Li 2022 (biostimulants, 16 papers, 68 observations). Extracted observations were matched to published reference standards using a score-based metadata matching protocol with Hungarian algorithm assignment; results are reported separately for the non-circular metadata-resolved subset (69% of Loladze observations) and the effect-tiebroken subset (31%). We validated using intraclass correlation coefficients (ICC(3,1)), Lin's concordance correlation coefficient (CCC), cluster-robust two one-sided tests (TOST), and Bland-Altman analysis. We further compared against a structurally independent multi-model pipeline on 1,889 observations across 100 papers without ground truth.

**Results:** Pearson r ranged from 0.887 (Loladze) to 0.968 (Li); CCC from 0.886 to 0.966. Cluster-robust TOST confirmed equivalence at +/-2 pp for all three datasets (all p < 0.014). Cohen's d: 0.006--0.065 (negligible). Aggregate effects reproduced within 0.22--0.50 pp. On the non-circular Loladze subset (447 observations, metadata-resolved only), r = 0.891, MAE = 3.4 pp, direction agreement = 94%. Ground-truth-free cross-method comparison yielded r > 0.93 across all datasets. Error analysis revealed that the agent never confused treatment and control columns; residual error was dominated by alignment ambiguity (which factorial sub-condition) and figure-reading precision.

**Conclusions:** A single AI agent achieves equivalence with published meta-analysis data for aggregate pooling, reproducing overall effects within 0.22--0.50 pp across three independent datasets. At approximately $200/month flat cost enabling ~100 meta-analyses, the approach makes previously infeasible evidence syntheses tractable.

**Keywords:** meta-analysis, data extraction, large language models, AI agent, agriculture, equivalence testing, TOST, concordance

---

# 1. Introduction

Meta-analysis is the cornerstone of evidence-based practice in agricultural science, but the systematic review process is both slow and fragile. Borah et al. (2017) found that the mean time to complete and publish a systematic review is 67.3 weeks, and Shojania et al. (2007) showed that 23% of reviews already need updating within two years of publication -- with 7% obsolete at the time of publication. The primary bottleneck is not statistical analysis but data extraction: trained researchers must manually identify, read, and record quantitative values from source publications.

This bottleneck is both expensive and error-prone. Schmidt et al. (2025) estimate 2--8 hours per paper for manual extraction. Buscemi et al. (2006) demonstrated that single-extractor error rates reach 17.7%, falling to 8.8% only under costly dual-extraction protocols -- underpinning the Cochrane Handbook's recommendation for dual independent extraction (Higgins et al., 2023). At the meta-analysis level, Mathes et al. (2017) found that 66.8% of published meta-analyses contain at least one data extraction error. Gartlehner et al. (2025) confirmed that single-investigator extraction requires approximately 107 minutes per study, with errors identified in up to 67% of meta-analyses. For agricultural meta-analyses, which may span hundreds of papers with complex factorial designs, the extraction burden can extend to months of researcher time.

Agricultural experiments present distinct challenges for automated extraction. Lacking standardized reporting frameworks like CONSORT (Topp et al., 2023), plant science studies employ complex factorial designs (CO2 x cultivar x soil amendment x harvest date), producing multi-layered tables where only specific treatment combinations are relevant. Variance reporting is inconsistent: nearly 70% of ecological meta-analysis datasets include studies with missing standard deviations (Nakagawa et al., 2023). Effect sizes span orders of magnitude -- from single-digit percentage changes in mineral concentrations to >200% increases in zinc biofortification studies -- complicating any uniform accuracy threshold.

A further challenge is epistemic circularity: if the same ground truth used during system development also serves as the validation benchmark, performance estimates may be inflated. Breaking this circularity requires either true holdout datasets or, more powerfully, cross-method agreement that requires no reference standard at all.

We present a validation study of a single AI agent extracting quantitative data from scientific PDFs across three published plant science datasets. The agent itself was developed with no access to any reference standard. Hui and Li are fully independent holdouts for both the agent and the validation framework; Loladze is independent for the agent but the matching protocol and validation scripts were developed iteratively using Loladze data, making it a development-adjacent dataset for the infrastructure (though not for the agent). We further demonstrate that the agent's output agrees with an independently developed multi-model consensus pipeline on 1,889 observations across 100 papers, providing a circularity-free validation without any ground truth. To our knowledge, this is the first study to (a) formally test equivalence of AI-extracted continuous data against published meta-analysis reference standards, (b) validate across multiple independent datasets, and (c) demonstrate cross-method agreement without ground truth as a scalable alternative to reference-standard validation.

---

# 2. Related Work

## 2.1 The Manual Extraction Burden

Data extraction remains the most labor-intensive phase of systematic review. Borah et al. (2017) found that systematic reviews take a mean of 67.3 weeks to complete, with a mean of 5 authors per review. Buscemi et al. (2006) found that single extractors make errors in 17.7% of fields, with dual extraction reducing the rate to 8.8% at double the cost (49 minutes additional per study). Shojania et al. (2007) showed that the median survival time of a systematic review before a signal for updating is 5.5 years, but 23% of reviews need updating within 2 years. Schmidt et al. (2025), in a living systematic review of extraction methods, estimate 2--8 hours per paper depending on complexity. For a 200-paper agricultural meta-analysis, this translates to 400--1,600 hours of skilled labor -- a cost that effectively limits the scope and frequency of evidence synthesis.

## 2.2 LLM-Based Extraction

Recent studies have explored large language models for automated data extraction, with mixed results depending on the outcome type. For categorical variables (study design, population characteristics), LLMs consistently achieve high accuracy: Gougherty and Clipp (2024) reported >90% accuracy (Kappa 0.92--0.98) for ecological categorical data, and Helms Andersen et al. (2025) found F1 scores around 90% when AI tools served as second reviewers for Cochrane data extraction, with recall lowest for review-specific variables (77--80%). However, Khraisha et al. (2024) found only moderate performance (precision 0.63) for data extraction from peer-reviewed literature, despite >90% accuracy on categorical screening tasks -- highlighting the gap between screening and extraction.

Performance drops substantially for continuous numerical outcomes. Jansen et al. (2025) evaluated GPT-4o and smaller models for extracting means, standard deviations, and sample sizes from psychology systematic reviews, reporting 26--36% accuracy -- insufficient for meta-analytic use. Accuracy correlated positively with model size and with more detailed variable descriptions. Kataoka et al. (2026) tested o3 (OpenAI) for clinical data extraction, achieving 75.3% overall accuracy (higher for string than numeric variables) but concluding that numeric extraction was "still inadequate" for unsupervised use. In ecology, Gougherty and Clipp (2024) found LLMs poor at quantitative extraction (23.8% accuracy using GPT-3.5); the improvement in quantitative extraction accuracy in our study likely reflects both model advancement (GPT-3.5 to Claude Opus 4.6) and methodological differences (conversational interface vs. structured JSON output with full-document context).

Multi-model consensus approaches have shown promise. Qiao et al. (2025) demonstrated that LLMs in a collaborative two-reviewer workflow (GPT-4-turbo + Claude-3-Opus) outperformed individual LLMs, with concordant responses between models highly likely to be correct. Khan et al. (2025) used paired LLMs for living systematic reviews, reporting 0.25% hallucination rates when models agreed. Poser et al. (2026) demonstrated that a three-model consensus reduced true extraction errors to 1.48% for clinical reports -- comparable to human neurologist error rates (~2%) -- and that consensus overcame a ceiling effect in single-model accuracy improvement. Cao et al. (2025) developed OttoSR, achieving 93.1% accuracy on structured data extraction across 7 Cochrane reviews (4,559 data points, 495 studies), outperforming dual human reviewers (79.7%). Notably, blinded adjudicators sided with OttoSR over original authors in 69.3% of disagreements, suggesting that published "ground truth" data may itself contain errors.

Gartlehner et al. (2024) provided the first proof-of-concept that LLMs can extract systematic review data at near-human accuracy, with Claude 2 achieving 96.3% accuracy across 160 data elements from 10 RCTs. Their follow-up (Gartlehner et al., 2025) scaled this to real-world conditions (9,341 data elements, 63 studies), finding AI-assisted extraction achieved 91.0% accuracy vs. human-only 89.0%, with a median time saving of 41 minutes per study -- the strongest evidence to date that LLMs can match or exceed human extraction under operational conditions.

Several groups have proposed AI as a "second reviewer" to augment human extraction. Li, Mathrani, and Susnjak (2025) proposed a three-tier automation framework and found that recall -- the completeness of extracted information -- was the primary bottleneck for all models tested. Delgado-Chaves et al. (2025) showed that LLMs can substantially reduce manual screening workload, with GPT-4o achieving best performance (MCC = 0.349) and open-source models (Llama 3.1) showing competitive performance. This is consistent with a pattern across the literature: multiple studies (Jansen et al., 2025; Helms Andersen et al., 2025; Li, Mathrani, & Susnjak, 2025) converge on recall rather than precision as the binding constraint on LLM extraction performance.

## 2.3 The Human Baseline Problem

A critical but underappreciated context for evaluating AI extraction is the reliability of human extraction itself. Tendal et al. (2009) found that in 53% of meta-analyses, multiplicity in trial reports -- multiple measurement scales, timepoints, and subgroups -- led to variability in pooled results, demonstrating that even trained human dual-extractors face fundamental ambiguity in what to extract from complex studies. Buscemi et al. (2006) reported 17.7% error rates for single human extractors. Mathes et al. (2017) documented error rates of 17.0% at the study level and 66.8% at the meta-analysis level -- meaning two-thirds of published meta-analyses contain at least one extraction error. Cao et al. (2025) compiled human dual-extractor accuracy rates from the literature ranging from 65.8% to 85.5%. These findings suggest that the "gold standard" of human extraction is itself noisy, particularly for continuous outcomes from complex study designs. Any evaluation of AI extraction should be interpreted against this human baseline: perfect agreement with a reference standard is not achievable even by trained humans.

## 2.4 Gaps in the Literature

Despite this progress, four critical gaps remain. First, no study has applied formal equivalence testing (e.g., TOST) to AI-extracted continuous data -- accuracy is typically reported as percentage agreement or correlation, which cannot establish statistical equivalence. Second, no study has validated against multiple independent ground-truth datasets, making it impossible to assess generalizability. Third, no study has demonstrated ground-truth-free validation through cross-method agreement, which is essential for scaling beyond the handful of datasets with published reference standards. Fourth, the agricultural domain -- with its complex factorial designs and inconsistent reporting -- remains largely untested. The Cochrane, Campbell, JBI, and CEE joint position statement on AI use in evidence synthesis (2025) emphasizes the need for validation studies across domains and outcome types before AI tools can be recommended for routine use.

---

# 3. Methods

## 3.1 Agent Architecture

A single AI agent (Claude Opus 4.6, Anthropic, running within the Claude Code CLI environment) read each source PDF directly and extracted observations. We use the term "agent" to denote an AI system that autonomously reads documents and produces structured output, not a multi-step planning agent with tool use or environmental interaction.

The agent operated as a general-purpose reader within a 200K-token context window, sufficient to ingest most scientific papers in their entirety. For each paper, the agent received a brief natural-language instruction specifying what data to extract. For example, the Loladze instruction read: "Extract all mineral element concentration data comparing elevated CO2 to ambient CO2 controls. For each observation, report: paper ID, element, tissue, CO2 levels, treatment mean, control mean, sample size, variance type and value." The agent produced a structured JSON file with one object per observation, following a predefined schema specifying fields for paper metadata, experimental conditions, and statistical values.

No domain-specific prompt templates, few-shot examples, vision extraction pipelines, or multi-model consensus mechanisms were used. The same agent model was used for all three datasets; only the natural-language instruction varied.

For the Loladze dataset, a three-pass workflow was employed: initial extraction, text cross-check (comparing extracted values against original PDF text to flag potential errors), and targeted re-extraction of flagged observations. This produced nine cross-check batch reports and nine re-extractions. Hui and Li datasets used single-pass extraction. The three-pass approach represents a pragmatic choice for the most complex dataset (46 papers with multi-factorial designs across 23 elements), not a fundamental architectural requirement.

**Cost.** The agent was run using Claude Code Pro at a flat subscription fee of $200/month, which provides unlimited access to the underlying model. At this pricing, the effective per-paper cost approaches zero at scale: the entire 87-paper validation could be completed many times within a single month. By comparison, manual extraction costs an estimated $60--240/paper for single extraction and $120--480/paper for dual extraction (2--8 hours x $30/hour, Schmidt et al., 2025; Buscemi et al., 2006). At subscription rates, a single researcher could feasibly extract data for approximately 100 meta-analyses per month.

## 3.2 Validation Datasets

No agent parameters were tuned against any reference standard. Hui and Li are fully independent holdouts for both the agent and the validation infrastructure. Loladze is independent for the agent (no agent prompts, parameters, or extraction logic were developed using Loladze reference data), but the matching protocol and validation scripts were developed iteratively with Loladze data during earlier pipeline work, making Loladze a development-adjacent dataset for the infrastructure though not for the agent itself.

**Loladze 2014** (Loladze, 2014). A comprehensive meta-analysis of elevated CO2 effects on plant mineral concentrations, published in *eLife* with open supplementary data. We processed 46 papers, yielding 650 matched observations across 25 elements. This dataset is the most demanding: complex factorial designs (CO2 x cultivar x ozone x nitrogen), diverse exposure systems (free-air CO2 enrichment [FACE], open-top chambers [OTC], growth chambers), and multiple tissues and developmental stages. Effect sizes are typically small (mean -6.3%), requiring high extraction precision to detect.

**Hui et al. 2025** (Hui et al., 2025). A global meta-analysis of zinc agronomic biofortification in wheat and its drivers, published in *Nature Communications*. We processed 37 papers, of which 25 matched ground-truth entries, yielding 461 matched observations. This dataset features standardized single-element tabular data with relatively consistent reporting formats. Effect sizes are large (mean ~53%), reflecting substantial zinc concentration increases from biofortification treatments.

**Li et al. 2022** (Li et al., 2022). A meta-analysis of non-microbial biostimulant effects on agronomic outcomes, published in *Frontiers in Plant Science* with open data. We processed 50 papers, of which 31 matched ground truth. After programmatic scale harmonization (accounting for unit conversions such as t/ha to kg/ha), 68 high-confidence observations from 16 papers were identified. The raw extraction (166 obs, r = 0.446) suffered from unit-scale mismatches identical to those observed in independent pipeline validation of the same dataset, confirming that the matching challenge is a property of the data, not of the extraction method.

Together, the three datasets span a range of extraction difficulty (simple tabular to complex factorial), effect-size magnitude (1--200+ pp), and reporting conventions, providing a diverse testbed for evaluating agent performance.

## 3.3 Matching Protocol

Extracted observations were matched to published reference standards using a score-based metadata matching protocol. For each candidate extracted observation and ground-truth row within the same paper and element (Loladze) or paper and tissue/method (Hui, Li), a similarity score was computed across up to 14 metadata dimensions:

1. **Tissue group** normalization (leaf, grain, root, shoot, etc.)
2. **Species** matching with common-to-Latin name mapping
3. **Site** matching (e.g., Duke FACE, ORNL, SERC)
4. **CO2 level** numeric matching
5. **Year** and year-range matching
6. **Phosphorus level** numeric and categorical (high/low P)
7. **Cultivar/variety** matching
8. **Ozone filter** (O3 vs. NF)
9. **Harvest stage** (intermediate vs. final)
10. **Growth stage** matching
11. **Sowing date/season**
12. **Nitrogen treatment**
13. **Soil type/potassium treatment**
14. **Additional moderators** (rainfall, needle age, temperature, etc.)

Each dimension contributed a mismatch penalty or match bonus to the total score. Globally optimal 1-to-1 assignment was then computed using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), which guarantees the assignment that maximizes total similarity across all observation-to-reference pairs within each paper.

**Pooling detection.** Ground-truth rows containing averaging instructions (e.g., "avg of rainfalls," "avg over 6 P treatments") were matched by pooling all candidate extracted observations for that element/tissue/species combination, computing the average effect, and comparing to the ground-truth value. Fourteen such pooled rows were identified across three papers (Seneweera 1997, Mishra 2011, Newbery 1995, Housman 2012).

**Effect-based tiebreaker.** When the metadata score was exactly tied for multiple candidate observations matched to the same ground-truth row -- occurring when ground-truth rows contain no distinguishing metadata (blank "Additional Info" column) -- the minimum absolute difference in effect size was used as a tiebreaker. This step is circular and is documented transparently. We report all results separately for the two subsets:

- **Metadata-resolved (447 observations, 69%):** Matched entirely on metadata similarity. Non-circular.
- **Effect-tiebroken (203 observations, 31%):** Required effect-value tiebreaker. Circular, documented as such.

This separation allows readers to assess the defensible, non-circular accuracy (metadata-resolved) independently of the overall result. The effect tiebreaker is analogous to how human dual-extractors resolve discrepancies by checking source data, but with the important distinction that consulting the outcome variable during matching inflates apparent agreement.

For Hui and Li, matching used raw means (within 20--30% tolerance) while validation used derived effect sizes, so the matching criterion and validation metric were not the same quantity.

Effect sizes are percentage change: (treatment - control) / |control| x 100. All "pp" values throughout this paper refer to differences on this percentage-point scale.

## 3.4 Statistical Analysis

**Primary metric.** ICC(3,1) (two-way mixed, consistency) following Koo and Li (2016), who classify 0.50--0.75 as "moderate," 0.75--0.90 as "good," and >0.90 as "excellent." We report ICC(3,1) because the agent represents a single fixed rater applied systematically, not a random sample of raters (Shrout & Fleiss, 1979). We also report Lin's concordance correlation coefficient (CCC; Lin, 1989), the standard metric for method-comparison studies, which simultaneously captures both precision (correlation) and accuracy (bias). Secondary metrics: Pearson r, Spearman rho (as a robustness check against outlier influence), mean absolute error (MAE), and direction agreement (whether extracted and reference effect sizes share the same sign).

**Equivalence testing.** Two one-sided tests (TOST) with both naive and cluster-robust standard errors. Cluster-robust SEs use the CR1 sandwich estimator, clustering by paper, with t(K-1) degrees of freedom, where K is the number of papers. We note that the CR2 bias-corrected estimator with Satterthwaite degrees of freedom (Pustejovsky & Tipton, 2018) is more appropriate for datasets with few or imbalanced clusters. Supplementary Table S1 reports CR2 results for all datasets. We selected equivalence margins based on practical meta-analytic considerations: in inverse-variance pooled meta-analysis, shifts of 1--2 pp in individual effect sizes produce negligible change in pooled estimates when averaging over dozens of observations. For context, Tendal et al. (2009) found that multiplicity in trial reports affected pooled results in over half of meta-analyses, and Cao et al. (2025) documented human dual-extractor accuracy rates of 65.8--85.5%.

**Bland-Altman analysis.** Mean difference and 95% limits of agreement (Bland & Altman, 1986), with proportional bias assessed via regression of the difference on the mean. We note that CCC = r x C_b (Lin, 1989), where C_b is a bias correction factor; CCC values closely matching ICC(3,1) confirms that the bias component is negligible.

**Paired bias tests.** Paired t-test, Wilcoxon signed-rank test, and Cohen's d for systematic bias assessment.

**Reproducibility.** A second independent agent run (Run 2) used the same model but was executed with no shared state, cached outputs, or intermediate results. Run 1 and Run 2 observations were matched by value similarity and compared.

**Cross-method agreement.** Agent output was compared against an independently developed multi-model consensus pipeline (details in Appendix A) that shared no code, prompts, or intermediate outputs with the agent. Observations were matched by value similarity (control and treatment means within 25% relative tolerance, with scale-factor harmonization). No ground truth was consulted.

---

# 4. Results

## 4.1 Extraction Accuracy Against Ground Truth

**Table 1. Agent extraction accuracy against published reference standards.**

| Dataset | Papers | Obs | r | rho | CCC | MAE (pp) | Direction | ICC(3,1) | 95% CI |
|---------|--------|-----|---|-----|-----|----------|-----------|----------|--------|
| Loladze 2014 | 46 | 650 | 0.887 | 0.888 | 0.886 | 3.6 | 91% | **0.886** | 0.869--0.901 |
| Hui 2025 | 25 | 461 | 0.942 | 0.915 | 0.942 | 7.4 | 96% | **0.942** | 0.930--0.951 |
| Li 2022 (high-conf) | 16 | 68 | 0.968 | 0.946 | 0.966 | 1.6 | 98% | **0.966** | 0.946--0.979 |
| **Total** | **87** | **1,179** | -- | -- | -- | -- | -- | -- | -- |

*rho = Spearman rank correlation. CCC = Lin's concordance correlation coefficient. Bootstrap 95% CIs (1,000 iterations): Loladze r [0.840, 0.920], MAE [3.1, 4.2]; Hui r [0.914, 0.960], MAE [6.1, 8.8]; Li r [0.944, 0.986], MAE [1.0, 2.4].*

Koo and Li (2016) classify ICC 0.75--0.90 as "good" and >0.90 as "excellent." CCC values closely mirror ICC(3,1), confirming that the agreement is robust to the choice of metric. Spearman rho values track Pearson r closely (e.g., 0.888 vs. 0.887 for Loladze), confirming that outlier influence does not inflate the Pearson results. The Li and Hui results are excellent; the Loladze result falls at the boundary of good and excellent, reflecting the greater extraction difficulty of complex factorial designs.

**Table 1a. Loladze matching resolution: non-circular vs. effect-tiebroken subsets.**

| Subset | N | % | r | MAE (pp) | Direction | Circular? |
|--------|---|---|---|----------|-----------|-----------|
| Metadata-resolved | 447 | 69% | 0.891 | 3.4 | 94% | No |
| Effect-tiebroken | 203 | 31% | 0.874 | 4.1 | 83% | Yes (last resort) |
| Pooled (GT-averaged) | 14 | 2% | 0.986 | -- | -- | No |
| All | 650 | 100% | 0.887 | 3.6 | 91% | Partially |

The non-circular subset (447 observations matched entirely on metadata) achieves r = 0.891, MAE = 3.4 pp, and 94% direction agreement -- comparable to or better than the full dataset. The effect-tiebroken subset shows slightly lower performance (r = 0.874, MAE = 4.1 pp), as expected given that effect-based matching inflates apparent agreement for well-matched pairs but cannot correct for genuinely misaligned observations. Crucially, the non-circular subset demonstrates that extraction quality is strong even without any circularity in matching.

**Loladze.** Of 46 papers, 37 achieved Excellent-tier accuracy (MAE < 2 pp), 2 Good (2--5 pp), 6 Fair (5--10 pp), and 1 Poor (MAE > 10 pp). Ten papers achieved perfect extraction (MAE = 0.00 pp): Wilsey 1994, Blank 2011, Ziska 1997, Porter 1984, Schenk 1997, Keutgen 2001, Pleijel 2009, Niinemets 1999, Wu 2004, and O'Neill 1987. The agent captured 25 of 25 elements present in the reference standard, with high coverage across Ca, Fe, Zn, Mg, N, P, K, S, Mn, Cu, and trace elements. Aggregate effect: GT = -6.31%, agent = -5.81%, difference = 0.50 pp. Element-level effect correlation: r = 0.952 (n = 25 elements).

**Hui.** Four papers achieved perfect extraction (r = 1.0): Erdal, Peck, Zou, and fpls-10-00426. The high ICC reflects the standardized reporting format of zinc biofortification studies: most papers present single-element data in well-structured tables. Aggregate effect: GT = 53.21%, agent = 53.48%, difference = 0.27 pp.

**Li.** The progression from raw matching (r = 0.446, 166 obs) to scale-harmonized high-confidence matching (r = 0.968, 68 obs) changed no extracted values -- only the matching methodology. The raw-match degradation resulted from unit-scale mismatches (e.g., yield in t/ha vs. kg/ha), which produce spurious discrepancies when matching by value similarity. Aggregate effect: GT = 11.65%, agent = 11.87%, difference = 0.22 pp.

[FIGURE 1: Scatter plots of agent-extracted vs. reference-standard effect sizes. Panel A: Loladze 2014 (r = 0.887, N = 650). Panel B: Hui 2025 (r = 0.942, N = 461). Panel C: Li 2022 high-confidence (r = 0.968, N = 68). Dashed lines indicate identity (y = x).]

[FIGURE 2: Per-paper MAE distribution across all three datasets, sorted by accuracy. Color-coded by dataset (Loladze = blue, Hui = green, Li = orange). Loladze: median 0.02 pp, IQR 0.0--2.5. Hui: median 4.1 pp, IQR 1.0--9.5. Li: median 0.8 pp, IQR 0.2--2.1.]

## 4.2 Formal Agreement Statistics

### 4.2.1 Equivalence Testing (TOST)

**Table 2. Cluster-robust TOST results.**

| Dataset | N | K | Margin | p-value | Result |
|---------|---|---|--------|---------|--------|
| **Li 2022** | 68 | 16 | +/-1 pp | 0.000 | PASS |
| | | | +/-2 pp | <0.001 | PASS |
| **Hui 2025** | 461 | 25 | +/-2 pp | 0.014 | PASS |
| | | | +/-3 pp | <0.001 | PASS |
| **Loladze 2014** | 650 | 46 | +/-2 pp | <0.001 | PASS |
| | | | +/-3 pp | <0.001 | PASS |

All three datasets now pass cluster-robust TOST at +/-2 pp. The Loladze improvement from the previous matching protocol (which failed at +/-2 pp with p = 0.091) reflects the score-based metadata matching with Hungarian algorithm assignment, which reduced aggregate bias from 1.09 pp to 0.50 pp. Under CR2 with Satterthwaite degrees of freedom (Supplementary Table S1), results may be slightly more conservative for datasets with cluster imbalance.

### 4.2.2 Bias Assessment

**Table 3. Systematic bias tests.**

| Dataset | Mean diff (pp) | Paired t | p | Cohen's d | Interpretation |
|---------|---------------|----------|---|-----------|----------------|
| Loladze 2014 | +0.50 | 1.38 | 0.168 | 0.054 | negligible |
| Hui 2025 | +0.27 | 0.35 | 0.726 | 0.006 | negligible |
| Li 2022 | +0.22 | 0.53 | 0.596 | 0.065 | negligible |

All Cohen's d < 0.10 (well below the conventional threshold of 0.20 for "small"), indicating negligible practical bias in all three datasets. The Loladze bias is no longer statistically significant (p = 0.168), reflecting the improved matching that reduced aggregate effect difference from 1.09 pp to 0.50 pp.

### 4.2.3 Bland-Altman Agreement

| Dataset | Mean diff (pp) | 95% Limits | Prop. bias r | Prop. bias p |
|---------|---------------|------------|-------------|-------------|
| Li 2022 | +0.22 | -6.5 to +7.0 | 0.223 | 0.067 |
| Hui 2025 | +0.27 | -32.5 to +33.0 | 0.085 | 0.069 |
| Loladze 2014 | +0.50 | -17.7 to +18.7 | -0.150 | <0.001 |

The wide Hui limits (+/-33 pp) reflect the large effect-size range in zinc biofortification studies, where some studies report >200% increases from foliar zinc application. Loladze shows statistically significant proportional bias (r = -0.150, p < 0.001), indicating that extraction errors are somewhat larger for observations with extreme effect sizes. Bland and Altman (1986) note that wide limits of agreement indicate poor interchangeability at the individual level. Here, the relevant question is not whether individual extracted values are interchangeable, but whether their aggregate produces equivalent pooled estimates -- a less demanding criterion that our data satisfy (mean differences 0.22--0.50 pp).

[FIGURE 3: Bland-Altman plots (difference vs. mean) for all three datasets. Panel A: Loladze 2014. Panel B: Hui 2025. Panel C: Li 2022. Horizontal lines show mean difference (solid) and 95% limits of agreement (dashed). Proportional bias regression line shown for Loladze.]

## 4.3 Run-to-Run Reproducibility

**Table 4. Run-to-run reproducibility (Run 1 vs. Run 2).**

| Dataset | Papers | Matched obs | r | MAE (pp) | Effect diff (pp) |
|---------|--------|-------------|---|----------|-------------------|
| Loladze 2014 | 41 | 665 | 0.816 | 8.4 | 0.09 |
| Hui 2025 | 24 | 362 | 0.946 | 12.3 | 6.31 |
| Li 2022 | 30 | 204 | 0.849 | 5.6 | 0.23 |
| **Total** | **95** | **1,231** | -- | -- | -- |

Aggregate effect sizes are highly stable for Loladze (0.09 pp) and Li (0.23 pp), demonstrating that stochastic variation in the agent's extraction cancels at the aggregate level. At the paper level, 27 papers achieved perfect reproducibility (r = 1.0 between runs): 8 Loladze papers, 8 Hui papers, and 11 Li papers.

The larger Hui aggregate gap (6.31 pp vs. <0.3 pp for Loladze and Li) reflects three factors: (a) Hui zinc biofortification studies report effect sizes an order of magnitude larger than the other datasets (mean ~68% vs. <20%), amplifying absolute errors; (b) greater run-to-run variability in which factorial treatment combinations were extracted (15 of 24 papers differed by >3 observations between runs); and (c) the matched-pair aggregate gap (6.31 pp) substantially exceeds the unmatched aggregate gap (0.23 pp), indicating a composition effect from differential observation matching. Five papers (Cakmak 1997, Kalayci 1999, Yilmaz 1997, Li 2013, fpls-10-00426) account for the entire discrepancy; excluding them reduces the gap to 0.31 pp.

The overall pattern -- high aggregate stability with moderate per-observation noise -- is consistent with alignment ambiguity (which factorial sub-condition to extract) rather than reading errors. This distinction is explored further in Section 4.5.

## 4.4 Cross-Method Agreement (Ground-Truth-Free)

The strongest epistemic warrant comes from comparing the agent against a structurally independent extraction method without any ground truth. We compared agent output against a multi-model consensus pipeline (Appendix A) that shared no code, prompts, or intermediate outputs with the agent. The pipeline uses a different primary model family (Kimi K2.5 and Gemini) and a different architectural approach (dual-model consensus with tiebreaker).

**Table 5. Agent--pipeline agreement (no ground truth used).**

| Dataset | Papers | Matched obs | r | Direction | Effect diff (pp) |
|---------|--------|-------------|---|-----------|-------------------|
| Loladze 2014 | 44 | 1,205 | **0.933** | 91% | 1.30 |
| Hui 2025 | 20 | 185 | **0.971** | 96% | 0.29 |
| Li 2022 | 36 | 499 | **0.994** | 88% | 1.89 |
| **Total** | **100** | **1,889** | -- | -- | -- |

All three correlations exceed 0.93. Two largely independent methods -- differing in primary model family, architecture, prompts, and implementation -- converge on the same effect sizes for 1,889 observations across 100 papers in three agricultural domains. Overall effect sizes differ by only 0.29--1.89 pp between methods. This convergence provides a circularity-free validation: neither method was calibrated against the other, and no ground truth was consulted.

[FIGURE 4: Scatter plots of agent-extracted vs. pipeline-extracted effect sizes (no ground truth). Panel A: Loladze 2014 (r = 0.933, N = 1,205). Panel B: Hui 2025 (r = 0.971, N = 185). Panel C: Li 2022 (r = 0.994, N = 499). Dashed lines indicate identity (y = x).]

## 4.5 Error Taxonomy

### 4.5.1 Classification of Discrepancies

Among Loladze papers where specific extraction errors could be diagnosed, we classified each discrepancy into one of three categories:

**Alignment ambiguity** (dominant source of error). The agent extracted correct values from the correct table but selected a different factorial sub-condition than the original meta-analyst. For example, in a CO2 x cultivar x nitrogen experiment with 12 treatment combinations, both the agent and the original analyst might extract wheat grain zinc -- but the agent selects the high-nitrogen treatment while the analyst selects the low-nitrogen treatment. Both are defensible analytical choices; neither is "wrong."

**Figure-reading precision** (secondary source). In papers where data were presented only in bar charts, the agent occasionally read bar heights imprecisely. For example, Rodenkirchen (2009) contained near-zero-effect observations for the *T. submollis* subgroup where treatment and control bars were nearly equal height, leading to rounding errors. Guo (2011) showed near-zero Cd effects at baseline soil levels -- an instance where the ground truth itself records 0.0% change, and any small imprecision in bar reading produces a proportionally large relative error.

**Wrong outcome variable** (rare, 1 of 46 papers). Baxter (1994) was the only paper where the agent extracted the wrong outcome: total nutrient content (mg/plant) from Table 1 instead of concentration (mg/g) from the figures. This represents an outcome-variable selection error rather than a reading error.

**Critically, the agent never confused treatment and control columns.** No instances of T/C swap were identified across any of the 46 Loladze papers. No misread numbers from tabular data were documented. All residual error is attributable to alignment (which row or table to extract) or figure precision, not to fundamental reading failures.

### 4.5.2 The Granularity Barrier

The dominance of alignment ambiguity over reading errors reveals a structural constraint we term the "Granularity Barrier." Complex factorial agricultural experiments present multiple valid extraction targets within the same table. When a meta-analyst specifies "extract wheat grain zinc under elevated CO2," the instruction is under-determined if the paper reports data for 3 cultivars x 2 nitrogen levels x 2 harvest dates. Different extractors -- human or AI -- will make different analytical sub-selections.

This finding has two implications. First, improving the AI's reading ability would yield diminishing returns; the binding constraint is the analytical decision of *which* factorial sub-condition to extract. Second, per-observation accuracy metrics (MAE, Bland-Altman limits) overstate the practical extraction error, because alignment ambiguities add noise at the observation level but cancel when pooled across many observations. This explains why aggregate effect errors (0.22--0.50 pp) are substantially smaller than per-observation MAE (1.6--7.4 pp).

[FIGURE 5: Error taxonomy for Loladze dataset. Categories: alignment ambiguity (dominant), figure-reading precision (secondary), wrong outcome variable (1 paper). Inset: example of alignment ambiguity from a factorial table.]

Future benchmark datasets should document analytical sub-selections (which specific rows, columns, and treatment combinations were extracted) to enable principled decomposition of validation error into alignment vs. extraction components.

---

# 5. Discussion

## 5.1 Principal Findings

The strongest evidence comes from the two fully independent holdout datasets: Hui (ICC = 0.942, CCC = 0.942) and Li (ICC = 0.966, CCC = 0.966). Neither dataset was seen during agent development, and the validation infrastructure was built without access to their reference standards. Both achieve "excellent" agreement (ICC > 0.90) and pass cluster-robust equivalence testing at +/-2 pp. These results establish that a single general-purpose AI agent can extract continuous quantitative data from scientific PDFs with sufficient accuracy for aggregate meta-analytic pooling.

The Loladze dataset presents a more nuanced picture. The full-dataset results (r = 0.887, ICC = 0.886, MAE = 3.6 pp) represent "good to excellent" agreement, substantially improved from previous matching approaches (r = 0.848, MAE = 5.4 pp). The improvement derives entirely from the score-based metadata matching with Hungarian algorithm assignment -- not from any change in extraction. The non-circular subset (447 observations, r = 0.891, MAE = 3.4 pp, 94% direction agreement) provides the most defensible headline number, as it involves no consultation of effect values during matching. The effect-tiebroken subset (203 observations, r = 0.874) is reported transparently for completeness. TOST now passes at +/-2 pp for all three datasets (all p < 0.014), eliminating the previous limitation where Loladze failed the +/-2 pp threshold.

Across all three datasets, Cohen's d ranged from 0.006 to 0.065 (all negligible), and aggregate effect sizes were reproduced within 0.22--0.50 pp -- well within the range of human dual-extractor disagreement on continuous data, where accuracy rates of 65.8--85.5% have been documented across multiple studies (Cao et al., 2025). A structurally independent multi-model pipeline (Appendix A) converged on the same values (r > 0.93, 1,889 observations) without ground truth, and the pipeline's own formal statistics against the same reference standards are consistent with the agent's (Appendix A, Table A1).

## 5.2 Cost-Effectiveness

**Table 6. Cost comparison of extraction methods.**

| Method | Cost/paper | Time/paper | Accuracy | Source |
|--------|-----------|------------|----------|--------|
| Single human extractor | $60--240 | 2--8 hours | 82.3% field-level* | Buscemi et al., 2006; Schmidt et al., 2025 |
| Dual human extraction | $120--480 | 4--16 hours | 91.2% field-level* | Buscemi et al., 2006 |
| AI-assisted human | ~$30 + API | ~1 hour | 91.0% field-level | Gartlehner et al., 2025 |
| AI agent (this study) | ~$2/paper** | ~2 min | ICC 0.89--0.97 | -- |

*Field-level accuracy (% of data fields without errors). **Based on $200/month subscription amortized over ~100 papers/month; effective cost decreases with volume. ICC is not directly comparable to field-level accuracy; it measures agreement on continuous effect sizes rather than discrete correctness.

The subscription pricing model fundamentally changes the economics of meta-analysis. At $200/month flat cost with capacity for approximately 100 meta-analyses, the effective cost per paper approaches zero at scale. For a hypothetical 500-paper agricultural meta-analysis, the agent would complete extraction in under a day, compared to an estimated $30,000--120,000 and 3--12 months for manual dual extraction.

It is important to note that humans also make errors, at non-trivial rates. Mathes et al. (2017) found errors in 66.8% of published meta-analyses. The relevant comparison is therefore not AI accuracy vs. perfection, but AI accuracy vs. human accuracy -- and on this comparison, the agent's performance (ICC 0.89--0.97, aggregate errors 0.22--0.50 pp) is competitive with the human baseline.

This cost structure also makes *living meta-analyses* economically viable for the first time. Elliott et al. (2014) documented that systematic reviews are frequently outdated within two years of publication; Shojania et al. (2007) confirmed that 23% need updating within two years. The cost barrier to living updates is precisely the bottleneck that AI extraction removes. A living systematic review that updates extraction from 200 new papers annually would cost effectively nothing with a flat-fee subscription, compared to $24,000--96,000/year for manual dual extraction -- a difference that explains why living reviews remain rare despite their recognized value.

## 5.3 The Granularity Barrier

The Loladze dataset illustrates a challenge inherent in any extraction system applied to complex factorial data. When the agent extracts correct numbers from the correct table but selects different factorial sub-conditions than the original meta-analyst, the resulting discrepancy is a legitimate analytical disagreement, not an extraction error. This explains why per-observation MAE (3.6 pp) is substantially larger than aggregate effect error (0.50 pp): alignment ambiguities add noise at the observation level but cancel when pooled.

The Granularity Barrier is not specific to AI extraction. Human dual-extractors face the same challenge: when instructions do not fully specify which factorial combination to extract, extractors will make different choices. The difference is that human discrepancies are resolved through discussion, while AI discrepancies are revealed only through validation. This finding connects to a broader pattern in the literature: multiple studies identify recall (completeness) rather than precision as the primary bottleneck for LLM extraction (Jansen et al., 2025; Li, Mathrani, & Susnjak, 2025; Helms Andersen et al., 2025). Our alignment ambiguity finding is essentially a recall problem at the factorial sub-condition level -- the agent extracts correct values but may select different subsets.

Under the three-tier automation framework of Li, Mathrani, and Susnjak (2025), the Hui zinc dataset (standardized tabular) corresponds to Tier 2 (automation with human review), while the Loladze CO2/minerals dataset (complex factorial) corresponds to Tier 3 (human judgment essential for sub-condition selection). Topp et al. (2023) documented the absence of standardized reporting checklists for agricultural experiments. If adopted, such checklists could specify which factorial sub-conditions to report, reducing alignment ambiguity for both human and AI extractors. Future meta-analysis protocols should provide explicit extraction rules for factorial designs (e.g., "extract the highest nitrogen treatment if multiple levels are reported").

## 5.4 Comparison with Published Systems

**Table 7. Comparison with published LLM extraction systems.**

| System | Domain | Models | Quant. accuracy | Equivalence test | GT-free validation |
|--------|--------|:------:|-----------------|:----------------:|:------------------:|
| Jansen et al. 2025 | Psychology | 6 | 26--36% means/SDs/Ns | No | No |
| Kataoka et al. 2026 | Clinical | 2 | 75.3% overall (o3); lower for numeric | No | No |
| Khan et al. 2025 | Clinical | 2 | 0.25% halluc. (concordant) | No | No |
| Poser et al. 2026 | Clinical | 3 | 1.48% true-error | No | No |
| Gougherty & Clipp 2024 | Ecology | 1 | 23.8% quant. (GPT-3.5) | No | No |
| Cao et al. 2025 (OttoSR) | Clinical | Multiple | 93.1% (vs. 79.7% human) | No | No |
| Gartlehner et al. 2025 | Clinical | 1 | 91.0% (vs. 89.0% human) | No | No |
| **This study** | **Plant sci.** | **1** | **ICC 0.886--0.966** | **TOST < 0.014** | **r > 0.93 (1,889 obs)** |

*Note: Accuracy metrics are not directly comparable across studies due to differences in domains, outcome types, denominators, and evaluation methodology.*

Our results substantially exceed published benchmarks for continuous numerical extraction, though direct comparison is constrained by differences in domain, outcome types, and evaluation methodology (Table 7 note). Jansen et al. (2025) reported 26--36% accuracy for extracting means, standard deviations, and sample sizes from psychology systematic reviews -- not computed effect sizes, but the raw components. Three factors likely explain our substantially higher accuracy: (a) agricultural tables present data in structured tabular format rather than clinical narratives; (b) our structured JSON output schema constrains the extraction format; and (c) the model difference (Claude Opus 4.6 vs. GPT-4o and smaller models). Kataoka et al. (2026) concluded that LLM-based numeric extraction was "still inadequate" for unsupervised use. Our results suggest this conclusion may be domain-dependent: agricultural tabular data, with explicit numerical entries in structured tables, may be more amenable to LLM extraction than clinical trial data reported in narrative text with complex outcome hierarchies.

The Gartlehner et al. (2025) results deserve particular attention: in a real-world evaluation across 6 systematic reviews, AI-assisted extraction (91.0%) was more accurate than human-only extraction (89.0%), with a median saving of 41 minutes per study. Our work extends this finding by demonstrating formal equivalence on continuous numerical data -- not just categorical fields -- and across a much larger observation count (1,179 vs. ~150 in Gartlehner's evaluation).

The OttoSR results (Cao et al., 2025) similarly reframe the evaluation question: their finding that blinded adjudicators sided with OttoSR over original authors in 69.3% of disagreements suggests that the "ground truth" against which AI systems are validated may itself contain errors at rates comparable to AI extraction errors.

It is worth noting that many comparison systems were evaluated on harder tasks or used older models. As model capabilities improve rapidly, the specific accuracy numbers reported here are less important than the validation framework: ICC, CCC, TOST, and cross-method agreement provide a reusable toolkit for evaluating any future extraction system.

## 5.5 Recommendations for Practice

Based on our validation, we propose the following extraction equivalence testing (EET) protocol for researchers deploying AI extraction in meta-analyses:

1. **Pilot validation.** Extract 5--10 papers for which ground-truth values are known. Compute ICC, CCC, and MAE. If ICC < 0.75 or CCC < 0.70, revise prompts before proceeding.
2. **Cross-method agreement.** Run a second, structurally independent extraction method (different model, different prompts) on the full corpus. Compute inter-method r. If r > 0.90 on >=50 observations, the extraction is likely reliable for aggregate pooling.
3. **Sensitivity analysis.** Re-run extraction on a random 20% subsample. If aggregate effects shift by <2 pp and per-observation MAE changes by <15%, the extraction is stable.
4. **Transparency.** Report the model name and version, exact prompts, matching protocol, and all formal agreement statistics (ICC, CCC, TOST). Deposit extraction code and outputs in a public repository.
5. **Human oversight.** Spot-check papers with MAE > 15 pp or flagged by cross-method disagreement. The AI extracts; the researcher verifies outliers and makes analytical decisions.

This protocol operationalizes the Cochrane/Campbell/JBI/CEE (2025) position statement, which articulates five principles for AI use in evidence synthesis. We note explicit compliance: (1) the researcher bears ultimate responsibility for all analytical decisions; (2) RAISE reporting recommendations are followed; (3) methodological rigor is maintained through formal equivalence testing; (4) human oversight is built in through spot-checking (point 5 above); and (5) the agent extracts data but does not make inclusion/exclusion or quality assessment judgments. The Cochrane Handbook (Higgins et al., 2023) mandates dual independent extraction to control error; our cross-method agreement framework -- comparing a single agent against a structurally independent pipeline -- operationalizes this principle for AI extraction. Schmidt et al. (2025) found that only 45% of extraction automation studies shared data and 42% shared code; our full public repository exceeds this standard.

## 5.6 Limitations

1. **Single model.** Results are specific to Claude Opus 4.6 (March 2026). Model updates may alter performance, and model deprecation renders results tied to a specific model version. Users should re-validate when updating models.

2. **Three-pass workflow.** The Loladze extraction used a three-pass approach (extract, cross-check, re-extract), which is more involved than truly zero-shot. Hui and Li used single-pass extraction.

3. **Observation-level scatter.** Bland-Altman limits span +/-7--33 pp depending on the dataset. The agent is better suited for aggregate pooling than observation-level precision.

4. **Variance extraction.** We validated means and effect sizes but did not systematically validate variance (SD/SE) extraction, which is known to be challenging for LLMs. Nakagawa et al. (2023) showed that nearly 70% of ecological datasets include studies with missing SDs, and proposed imputation methods based on pooled coefficients of variation that could be applied downstream of AI extraction. Future work should validate AI variance extraction and assess whether the combination of AI-extracted means with imputed variances produces equivalent meta-analytic results.

5. **Li sample size.** Only 68 high-confidence Li observations were validated, limiting statistical power for that dataset.

6. **Hui reproducibility.** The 6.31 pp aggregate effect difference between runs reflects composition effects from differential extraction coverage, not systematic extraction drift.

7. **Domain scope.** All three datasets are from plant science. Generalization to clinical trials, social sciences, or other domains with different reporting conventions is untested.

8. **Single-author validation.** Matching protocols were developed and applied by a single author. All code and data are publicly available to enable independent replication.

9. **Proportional bias.** Loladze showed significant proportional bias (r = -0.150): errors correlated with effect magnitude. Users should apply sensitivity analyses for extreme-value subgroups.

10. **Data contamination.** Claude Opus 4.6 was trained on internet data through early 2025. All three reference datasets are publicly available: Loladze's supplementary data is hosted on eLife's website; Hui and Li published in Frontiers journals with open data. We cannot rule out that the model encountered these values during training. However, three observations argue against memorization as the primary driver of performance: (a) the agent makes errors -- alignment ambiguities, wrong factorial conditions, occasional outcome-variable selection errors -- that would not occur with memorized data; (b) run-to-run variation (Table 4) shows the agent is actively processing documents rather than recalling fixed values; and (c) the ground-truth-free agreement between the agent and a pipeline using different model families (Kimi and Gemini) would not be improved by Claude-specific memorization.

11. **Non-English papers.** All source papers were in English. Performance on non-English publications is untested.

12. **Figure-only data.** Papers where quantitative data appear only in figures (not tables) were not separately validated. The agent can read figures but accuracy may differ from tabular extraction; our error taxonomy confirms figure-reading precision as a secondary error source.

13. **Matching circularity.** For the Loladze dataset, 31% of observations (203/650) required an effect-value tiebreaker during matching, which is circular. Results for these observations are reported separately (Table 1a) and should be interpreted accordingly. The non-circular subset (69%, 447 observations) is the defensible headline for Loladze.

## 5.7 Ethical Considerations

The Cochrane, Campbell, JBI, and CEE joint position statement (2025) on AI use in evidence synthesis requires disclosure of AI tool use and human oversight of AI-generated outputs. We endorse this position and note several considerations for responsible deployment of AI extraction tools.

First, this tool augments rather than replaces human judgment. Analytical decisions -- inclusion criteria, outlier handling, factorial sub-condition selection, and quality assessment -- remain with the researcher. The agent extracts data; the researcher decides what to do with it.

Second, the low cost and speed of AI extraction creates a risk of low-quality meta-analysis "mills" if analytical judgment is also automated. We emphasize that extraction is only one component of evidence synthesis; protocol development, quality assessment, and interpretation require domain expertise that AI currently cannot provide.

Third, all AI-assisted extraction should be disclosed per RAISE (Reporting AI use in Systematic reviews and meta-analyses in Evidence synthesis) recommendations. Schmidt et al. (2025), in a living review of 117 extraction automation publications, found a trend of decreasing reporting quality for quantitative results; our study addresses this gap by reporting the full suite of method-comparison metrics (ICC, CCC, TOST, Bland-Altman). Researchers should report the model, version, prompts, and any post-processing applied to AI-extracted data.

## 5.8 Implications

For plant science -- with decades of unprocessed literature on CO2 effects, biofortification, biostimulants, and other topics -- reliable AI extraction at this accuracy level has immediate practical significance. A single researcher with a subscription can now extract data from hundreds of papers in days rather than months, enabling meta-analyses that were previously infeasible due to resource constraints. If even 10% of the estimated 600,000 researcher-hours spent annually on manual data extraction in agricultural sciences (based on ~300 published agricultural meta-analyses/year x 200 papers x 5 hours/paper) could be automated at equivalent quality, the freed capacity would correspond to approximately 30 full-time equivalent researchers redirected toward analysis and interpretation.

The cross-method agreement framework (agent vs. pipeline, no ground truth) offers a scalable approach to validating AI extraction systems. Rather than requiring curated reference standards -- which are expensive to create and limited in availability -- researchers can establish confidence by demonstrating convergence between structurally independent extraction methods. This is analogous to inter-rater reliability between human extractors, but between AI systems operating on the same source documents. Khan et al. (2025) showed that when two LLMs agree, the hallucination rate drops to 0.25%. Qiao et al. (2025) demonstrated that multi-LLM collaboration outperforms individual models, with concordant responses highly likely to be correct. Our cross-method agreement (r > 0.93 between agent and pipeline) provides an analogous guarantee at the dataset level: when two independent methods converge, the probability that both made the same systematic error is low -- particularly when the methods differ in model family, architecture, and prompts, and when the error is not driven by ambiguity in the source document itself.

---

# 6. Conclusion

A single AI agent reading scientific PDFs directly achieves ICC = 0.886--0.966 against published reference standards across three independent plant science meta-analysis datasets (1,179 observations, 87 papers). On the non-circular Loladze subset (447 observations matched entirely on metadata), r = 0.891, MAE = 3.4 pp, and direction agreement = 94%. Cluster-robust TOST (CR1) confirms equivalence at +/-2 pp for all three datasets (all p < 0.014). Aggregate effects are reproduced within 0.22--0.50 pp (all Cohen's d < 0.07). Run-to-run reproducibility is high at the aggregate level (0.09--0.23 pp for Loladze and Li). Ground-truth-free comparison with a structurally independent multi-model pipeline yields r > 0.93 on 1,889 observations across 100 papers. Error analysis reveals that the agent never confused treatment and control columns; residual error is dominated by alignment ambiguity (which factorial sub-condition) and figure-reading precision, not reading failures.

At approximately $200/month flat cost enabling extraction at scale, the approach reduces extraction cost by approximately three orders of magnitude relative to manual methods while achieving formal statistical equivalence. The cross-method agreement framework provides a scalable, ground-truth-free validation paradigm applicable beyond the three datasets studied here.

The error taxonomy reveals a finding with implications beyond AI validation: the binding constraint on extraction accuracy is not the quality of the extractor -- human or AI -- but the precision of the extraction protocol. When factorial experimental designs produce multiple defensible extraction targets, no extractor can be "correct" without unambiguous instructions specifying which sub-condition to select. Future meta-analysis guidelines should address this granularity problem explicitly, for the benefit of both human and AI extractors.

For agricultural science, where the volume of unprocessed primary literature far exceeds the capacity for manual extraction, AI agents offer a path to more comprehensive, more timely, and more reproducible evidence synthesis.

Code, configuration files, and validation scripts are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.

---

# References

- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307--310.
- Borah, R., Brown, A. W., Capers, P. L., & Kaiser, K. A. (2017). Analysis of the time and workers needed to conduct systematic reviews of medical interventions using data from the PROSPERO registry. *BMJ Open*, 7(4), e012545.
- Buscemi, N., Hartling, L., Vandermeer, B., Tjosvold, L., & Klassen, T. P. (2006). Single data extraction generated more errors than double data extraction in systematic reviews. *Journal of Clinical Epidemiology*, 59(7), 697--703.
- Cao, X., et al. (2025). OttoSR: Automation of systematic reviews with large language models. *medRxiv*. https://doi.org/10.1101/2025.01.15.25320588
- Cochrane, Campbell Collaboration, JBI, & CEE [Flemyng, E., et al.]. (2025). Position statement on the use of artificial intelligence in the production of evidence syntheses. *Cochrane Database of Systematic Reviews*.
- Delgado-Chaves, F. M., et al. (2025). Transforming literature screening: The emerging role of large language models in systematic reviews. *PNAS*, 122(3).
- Elliott, J. H., Turner, T., Clavisi, O., et al. (2014). Living systematic reviews: An emerging opportunity to narrow the evidence-practice gap. *PLoS Medicine*, 11(2), e1001603.
- Gartlehner, G., Kahwati, L., Engeli, C., Hamel, C., Gaisinger, K., & Glechner, A. (2024). Data extraction for evidence synthesis using a large language model: A proof-of-concept study. *Research Synthesis Methods*, 15(4), 576--582.
- Gartlehner, G., et al. (2025). Artificial intelligence-assisted data extraction with a large language model: A study within reviews. *Annals of Internal Medicine*.
- Gougherty, A. V., & Clipp, H. L. (2024). Testing the reliability of an AI-based large language model to extract ecological information from the scientific literature. *npj Biodiversity*, 3(1), 13.
- Helms Andersen, T., et al. (2025). Using AI tools as second reviewers in systematic reviews. *Cochrane Evidence Synthesis and Methods*.
- Higgins, J. P. T., Thomas, J., Chandler, J., et al. (Eds.). (2023). *Cochrane Handbook for Systematic Reviews of Interventions* (version 6.4). Cochrane.
- Hui, X., Luo, L., Chen, Y., Palta, J. A., & Wang, Z. (2025). Zinc agronomic biofortification in wheat and its drivers: a global meta-analysis. *Nature Communications*, 16, 3913.
- Jansen, T., et al. (2025). Data extraction by generative artificial intelligence. *Psychological Bulletin*, 151(10), 1280--1306.
- Jonnalagadda, S. R., Goyal, P., & Huffman, M. D. (2015). Automating data extraction in systematic reviews: a systematic review. *Systematic Reviews*, 4, 78.
- Kataoka, Y., et al. (2026). Automating the data extraction process for systematic reviews using GPT-4o and o3. *Research Synthesis Methods*, 17, 42--62.
- Khan, M. A., Ayub, U., Naqvi, S. A. A., et al. (2025). Collaborative large language models for automated data extraction in living systematic reviews. *JAMIA*, 32(4), 638--647.
- Khraisha, Q., et al. (2024). Can large language models replace humans in systematic reviews? A study of LLM performance in screening and extracting data. *Research Synthesis Methods*, 15(4), 616--626.
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155--163.
- Li, J., Van Gerrewey, T., & Geelen, D. (2022). A meta-analysis of biostimulant yield effectiveness in field trials. *Frontiers in Plant Science*, 13, 836702.
- Li, L., Mathrani, A., & Susnjak, T. (2025). What level of automation is "good enough"? A benchmark of large language models for meta-analysis data extraction. *arXiv:2507.15152*.
- Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics*, 45(1), 255--268.
- Loladze, I. (2014). Hidden shift of the ionome of plants exposed to elevated CO2 depletes minerals at the base of human nutrition. *eLife*, 3, e02245.
- Mathes, T., Klassien, P., Janzen, T., Jackel, A., Peinemann, F., Becker, A., et al. (2017). Frequency of data extraction errors and methods to increase data extraction quality: a methodological review. *BMC Medical Research Methodology*, 17, 152.
- Nakagawa, S., et al. (2023). A robust and readily implementable method for the meta-analysis of response ratios with and without missing standard deviations. *Ecology Letters*, 26(2), 232--244.
- Poser, P. L., Klimas, R., Luerweg, J., et al. (2026). Improving reliability and accuracy of structured data extraction using a consensus large-language model approach. *Frontiers in Artificial Intelligence*.
- Pustejovsky, J. E., & Tipton, E. (2018). Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. *Journal of Business & Economic Statistics*, 36(4), 672--683.
- Qiao, Y., Shen, Y., Wang, Y., et al. (2025). Collaborative large language models for automated data extraction in living systematic reviews. *Journal of the American Medical Informatics Association*, 32(4), 638--647.
- Schmidt, L., Shokraneh, F., Pieper, D., & Mathes, T. (2025). Data extraction methods for systematic review (semi)automation: Update of a living systematic review. *F1000Research*.
- Shojania, K. G., Sampson, M., Ansari, M. T., Ji, J., Doucette, S., & Moher, D. (2007). How quickly do systematic reviews go out of date? A survival analysis. *Annals of Internal Medicine*, 147(4), 224--233.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420--428.
- Tendal, B., Higgins, J. P. T., Juni, P., Hrobjartsson, A., & Gotzsche, P. C. (2009). Multiplicity of data in trial reports and the reliability of meta-analyses: Empirical study. *BMJ*, 339, b3128.
- Topp, C. F. E., et al. (2023). AgroEcoList: A checklist to improve reporting of ecological research in agronomy. *PLOS ONE*, 18(6), e0285478.
- Tsafnat, G., Glasziou, P., Choong, M. K., Dunn, A., Galgani, F., & Coiera, E. (2014). Systematic review automation technologies. *Systematic Reviews*, 3, 74.

---

# Data Availability Statement

Agent extraction code, configuration files, validation scripts, and pre-computed outputs are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture. Reference-standard datasets are from published meta-analyses: Loladze (2014), Hui et al. (2025), and Li et al. (2022). Source PDFs cannot be redistributed due to publisher copyright.

All extracted data are provided as structured JSON files in the repository. Each observation is a JSON object containing fields for `paper_id`, `element` or `outcome_variable`, `treatment_mean`, `control_mean`, `effect_pct` (percentage change from control), `variance_type`, `treatment_variance`, `control_variance`, `treatment_n`, `control_n`, and relevant moderators. Dataset-specific outputs are organized by subdirectory: `output/agent_extraction/` (Loladze CO2/minerals), `output/hui2023_agent_extraction/` (Hui zinc/wheat), and `output/li2022_agent_extraction/` (Li biostimulants/yield). Pipeline comparison outputs are in `output/loladze_v3_combined/`, `output/hui2023_full_35/`, and `output/li2022_combined/`. Validation results and formal statistics are in `output/agent_formal_stats/`.

# Author Contributions (CRediT)

All roles: Moshe Halpern.

# Conflict of Interest Statement

The author declares no conflicts of interest. The AI model used is a commercial product; the author has no financial relationship with the provider beyond standard subscription fees.

# Ethics and Funding

Not applicable (published literature only; no human subjects). No external funding was received.

---

# Figure Captions

**Figure 1.** Scatter plots of agent-extracted vs. reference-standard effect sizes (percentage change from control). Panel A: Loladze 2014 (r = 0.887, N = 650). Panel B: Hui 2025 (r = 0.942, N = 461). Panel C: Li 2022 high-confidence (r = 0.968, N = 68). Dashed lines indicate identity (y = x). Points are colored by paper.

**Figure 2.** Per-paper MAE distribution across all three datasets (N = 87 papers), sorted by accuracy. Color-coded by dataset (Loladze = blue, Hui = green, Li = orange). Observation counts (n) are annotated on each bar. Loladze: 37 papers at Excellent tier (MAE < 2 pp), with 10 papers achieving perfect extraction (MAE = 0.00 pp).

**Figure 3.** Bland-Altman plots showing the difference between agent-extracted and reference-standard effect sizes plotted against their mean. Panel A: Loladze 2014 (mean diff = +0.50 pp, LoA = -17.7 to +18.7 pp). Panel B: Hui 2025 (mean diff = +0.27 pp, LoA = -32.5 to +33.0 pp). Panel C: Li 2022 (mean diff = +0.22 pp, LoA = -6.5 to +7.0 pp). Horizontal solid lines show mean difference; dashed lines show 95% limits of agreement. The regression line for proportional bias is shown for Loladze (r = -0.150, p < 0.001).

**Figure 4.** Scatter plots of agent-extracted vs. pipeline-extracted effect sizes (ground-truth-free comparison). Panel A: Loladze 2014 (r = 0.933, N = 1,205). Panel B: Hui 2025 (r = 0.971, N = 185). Panel C: Li 2022 (r = 0.994, N = 499). Dashed lines indicate identity (y = x). Neither method was calibrated against the other.

**Figure 5.** Error taxonomy for the Loladze dataset. Alignment ambiguity -- the agent and original meta-analyst selected different factorial sub-conditions from the same table -- is the dominant error source. Figure-reading precision is secondary. Only one paper (Baxter 1994) involved extraction of the wrong outcome variable. The agent never confused treatment and control columns.

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

The pipeline was validated against the same three reference standards as the agent. Note: the Loladze dataset was used during pipeline development (not a true holdout), while Hui and Li were independent holdouts for both the pipeline and the agent.

**Table A1. Pipeline validation results (for comparison with agent Table 1).**

| Dataset | Papers | Obs | r | MAE (pp) | ICC(3,1) | TOST +/-2pp | TOST +/-3pp |
|---------|--------|-----|---|----------|----------|-----------|-----------|
| Hui 2025 | 19 | 308 | 0.999 | 0.43 | 0.999 | p < 0.001 PASS | p < 0.001 PASS |
| Li 2022 (all) | 27 | 200 | 0.951 | 2.30 | 0.949 | p = 0.052 FAIL | p = 0.002 PASS |
| Loladze 2014* | 46 | 413 | 0.886 | 4.36 | 0.870 | p = 0.020 PASS | p = 0.001 PASS |

*Development case study, not an independent holdout for the pipeline.

**Key comparison.** On the two datasets that are independent holdouts for both methods:
- **Hui**: Pipeline ICC = 0.999, Agent ICC = 0.942. The pipeline achieves higher per-observation accuracy on this standardized dataset, likely due to multi-model cross-validation.
- **Li**: Pipeline ICC = 0.949, Agent ICC = 0.966. The agent achieves comparable or slightly better results on the biostimulant dataset.

Both methods achieve negligible bias (all Cohen's d < 0.20) and strong aggregate effect reproduction. The pipeline's main advantage is its built-in confidence stratification, which concentrates 95% of large errors in flagged observations. The agent's advantages are simplicity and lower cost.

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

# Appendix B: Extraction Prompts and Adaptation Guide

This appendix provides the complete natural-language instructions given to the agent for each dataset, along with the dataset-specific configuration files that define PICO criteria and extraction priorities. No few-shot examples, prompt templates, or domain-specific tooling were used; the agent received only the instruction text below plus the source PDF.

## B.1 Extraction Prompts

### B.1.1 Loladze 2014 (CO2/Minerals)

**Natural-language instruction:**

> Extract all mineral element concentration data comparing elevated CO2 to ambient CO2 controls. For each observation, report: paper ID, element, tissue, CO2 levels, treatment mean, control mean, sample size, variance type and value.

**Configuration file** (`configs/loladze_co2_minerals.json`) specifies:

- **Intervention:** Elevated atmospheric CO2 (typically 550--700 ppm)
- **Control:** Ambient CO2 (typically 350--400 ppm)
- **Primary outcomes:** Mineral element concentrations for 25 elements (N, P, K, Ca, Mg, Fe, Zn, Mn, Cu, S, B, Mo, Na, Cd, Cr, Ni, Pb, Se, Si, V, Al, Sr, Ba, Co, Cl)
- **Expected direction:** Negative (dilution effect)
- **Key moderators:** Plant species/cultivar, tissue type (grain, leaf, root, shoot), experimental system (FACE, OTC, greenhouse), nitrogen fertilization level, C3 vs. C4 pathway
- **Extraction priorities:** Extract every row from tables with no pooling across cultivars, years, or treatments; prioritize mineral concentration data over biomass; record variance (SE/SD) and sample size (n); record tissue type and specific CO2 levels (ppm)
- **T/C confusion warnings:** Ambient = control; water stress treatments are not CO2 treatments; if effect is strongly positive (+50%), verify treatment/control assignment; AMF treatments create factorial designs requiring both levels extracted separately

### B.1.2 Hui 2025 (Zinc Biofortification in Wheat)

**Natural-language instruction:**

> Extract all grain zinc concentration data comparing zinc-fertilized treatments to no-zinc controls in wheat. For each observation, report: paper ID, Zn application method, Zn rate, treatment mean, control mean, sample size, variance type and value.

**Configuration file** (`configs/hui2023_zinc_wheat.json`) specifies:

- **Intervention:** Zinc fertilizer application (ZnSO4, ZnO, Zn-EDTA, etc. via soil, foliar, or combined)
- **Control:** No Zn fertilizer application (CK)
- **Primary outcomes:** Grain Zn concentration (mg/kg)
- **Expected direction:** Positive
- **Key moderators:** Zn application method (soil, foliar, soil+foliar), fertilizer type, application rate (kg Zn/ha), spraying concentration and timing, country, soil available Zn (mg/kg), soil pH
- **Extraction priorities:** Extract every treatment combination separately; control = zero-Zn treatment; for factorial designs (Zn rate x timing x form), extract each combination vs. its control; record LSD where reported; also extract grain yield when available alongside Zn concentration
- **T/C confusion warnings:** Control is the zero-Zn treatment, not untreated plants; foliar vs. soil application are different treatments, not replicates

### B.1.3 Li 2022 (Biostimulants/Yield)

**Natural-language instruction:**

> Extract all fresh yield data comparing biostimulant-treated plots to untreated controls in open-field trials. For each observation, report: paper ID, biostimulant type, crop species, treatment mean, control mean, sample size, variance type and value.

**Configuration file** (`configs/li2022_biostimulant_yield.json`) specifies:

- **Intervention:** Plant biostimulant application (seaweed extract, humic/fulvic acid, protein hydrolysate, chitosan, silicon, phosphite)
- **Control:** No biostimulant application (untreated control)
- **Primary outcomes:** Fresh yield (kg/ha, t/ha, g/plant, kg/m2)
- **Expected direction:** Positive
- **Key moderators:** Biostimulant type (SWE, PHs, PE, HFA, Chi, Si, Phi), crop species, application method (soil drench, foliar spray, seed treatment), application rate, climate zone, number of applications
- **Extraction priorities:** Extract every treatment combination separately; control = untreated/no-biostimulant; for multiple rates or timings, extract each vs. control; focus on fresh yield as primary outcome
- **T/C confusion warnings:** Watch for different yield units (g/plant vs. kg/ha vs. t/ha); some papers test multiple biostimulant products -- extract each vs. control

## B.2 Adapting for New Datasets

The agent requires only two inputs to extract data from a new meta-analysis domain: (1) a natural-language instruction and (2) a JSON configuration file. No code changes are needed.

**Step 1: Write the natural-language instruction.** The instruction should specify in one to two sentences: what outcome variable to extract, what comparison to make (treatment vs. control), and what fields to report for each observation. The three examples in B.1 illustrate the pattern: "Extract all [outcome] data comparing [treatment] to [control]. For each observation, report: [field list]."

**Step 2: Create a configuration file.** Copy one of the existing configuration files (e.g., `configs/loladze_co2_minerals.json`) and modify the following fields:

- **`intervention` and `control`**: Define the treatment and control conditions in plain language.
- **`primary_outcomes`**: List the outcome variables of interest. For single-outcome meta-analyses (e.g., yield), this is a single item. For multi-element studies (e.g., mineral concentrations), list all elements.
- **`expected_direction`**: Set to "positive" or "negative" based on the expected treatment effect. This is used for quality checks, not for extraction.
- **`important_moderators`**: List study-level and treatment-level variables to record (e.g., species, application rate, soil type).
- **`extraction_priorities`**: Specify granularity expectations (e.g., "extract each cultivar separately, do not pool across years").
- **`tc_confusion_warnings`**: List domain-specific pitfalls for treatment/control misidentification.
- **`pico`**: Define keywords for identifying treatment and control groups in source papers.

**Step 3: Validate on a pilot set.** Extract from 3--5 papers with known values and compare against the source tables. Common issues to check: (a) treatment/control swap (verify the direction of effect matches expectations), (b) unit mismatches (e.g., mg/kg vs. g/kg), (c) missing factorial conditions (verify that all treatment combinations are extracted, not just main effects).

**Key adaptation points by domain:**

- **Different outcome variable:** Change `primary_outcomes` and the natural-language instruction. The agent will look for whatever outcome you specify.
- **Different element/analyte list:** For multi-element studies, list all target elements in `primary_outcomes`. The agent normalizes element names (e.g., "grain Zn concentration (mg/kg)" maps to "Zn").
- **Different experimental designs:** Update `tc_confusion_warnings` to flag domain-specific factorial structures (e.g., CO2 x N, Zn rate x timing, biostimulant x irrigation).
- **Different variance conventions:** If the target domain predominantly uses LSD or CV rather than SE/SD, note this in `extraction_priorities` so the agent records the appropriate variance type.
- **Different effect-size scales:** The pipeline computes effect sizes as percentage change from control. For ratio-based outcomes (e.g., odds ratios), post-processing adjustments would be needed.
