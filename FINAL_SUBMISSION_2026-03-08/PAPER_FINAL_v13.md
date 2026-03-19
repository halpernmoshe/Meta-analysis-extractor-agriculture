# Breaking the Extraction Bottleneck: A Single AI Agent Achieves Equivalence with Published Meta-Analysis Data Across Three Agricultural Datasets

**Moshe Halpern** ^ORCID^

Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization — Volcani Center, Israel

*Correspondence:* Moshe Halpern, Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization — Volcani Center, Rishon LeZion 7505101, Israel.

---

# Abstract

**Background:** Data extraction is the primary bottleneck in meta-analysis (2–8 hours/paper, 17.7% error rate). Large language models achieve >90% accuracy on categorical variables but only 26–36% on continuous outcomes. No study has validated AI-extracted continuous data against multiple independent ground-truth datasets with formal equivalence testing.

**Methods:** A general-purpose AI agent (Claude Opus 4.6) extracted quantitative data from source PDFs across three published plant science meta-analyses: Loladze 2014 (CO₂/minerals, 46 papers), Hui 2023 (zinc biofortification, 25 papers), and Li 2022 (biostimulants, 16 papers). Hui and Li are fully independent holdouts; Loladze is independent for the agent but not the validation framework. We validated against published reference standards (1,184 observations, 87 papers) and compared against a structurally independent multi-model pipeline without ground truth (1,889 observations, 100 papers) using intraclass correlation coefficients (ICC(3,1)), Lin's concordance correlation coefficient (CCC), cluster-robust two one-sided tests (TOST), and Bland-Altman analysis.

**Results:** ICC(3,1) ranged from 0.845 (Loladze) to 0.966 (Li); CCC closely matched (0.844–0.966). Cluster-robust TOST confirmed equivalence at ±3 pp for all datasets (p ≤ 0.047). Cohen's d: 0.016–0.103 (negligible). Aggregate effects reproduced within 0.22–1.09 pp. Ground-truth-free cross-method comparison yielded r > 0.93 across all datasets. Alignment ambiguity dominated reading errors (93% vs. 3% of diagnosable discrepancies). Cost: ~$0.15/paper.

**Conclusions:** A single AI agent achieves equivalence with published meta-analysis data for aggregate pooling. Cross-method agreement without ground truth provides a scalable, circularity-free validation framework. At three orders of magnitude less cost than manual extraction, the approach makes previously infeasible meta-analyses tractable.

**Keywords:** meta-analysis, data extraction, large language models, AI agent, agriculture, equivalence testing, TOST, concordance

---

# 1. Introduction

Meta-analysis is the cornerstone of evidence-based practice in agricultural science. Loladze (2014) synthesized 1,481 mineral concentration measurements from 130 species to reveal the hidden nutritional cost of elevated CO₂. Hui et al. (2023) aggregated zinc biofortification trials across dozens of wheat studies to quantify application-method effects. Li et al. (2022) pooled biostimulant field trials to estimate yield responses. In each case, the primary bottleneck was not statistical analysis but data extraction: trained researchers must manually identify, read, and record quantitative values from source publications.

This bottleneck is both expensive and error-prone. Schmidt et al. (2025) estimate 2–8 hours per paper for manual extraction. Buscemi et al. (2006) demonstrated that single-extractor error rates reach 17.7%, falling to 8.8% only under costly dual-extraction protocols — underpinning the Cochrane Handbook's recommendation for dual independent extraction (Higgins et al., 2023). For agricultural meta-analyses, which may span hundreds of papers with complex factorial designs, the extraction burden can extend to months of researcher time.

Agricultural experiments present distinct challenges for automated extraction. Lacking standardized reporting frameworks like CONSORT (Topp et al., 2023), plant science studies employ complex factorial designs (CO₂ × cultivar × soil amendment × harvest date), producing multi-layered tables where only specific treatment combinations are relevant. Variance reporting is inconsistent: nearly 70% of ecological meta-analysis datasets include studies with missing standard deviations (Nakagawa et al., 2023). Effect sizes span orders of magnitude — from single-digit percentage changes in mineral concentrations to >200% increases in zinc biofortification studies — complicating any uniform accuracy threshold.

A further challenge is epistemic circularity: if the same ground truth used during system development also serves as the validation benchmark, performance estimates may be inflated. Breaking this circularity requires either true holdout datasets or, more powerfully, cross-method agreement that requires no reference standard at all.

We present a validation study of a single AI agent extracting quantitative data from scientific PDFs across three published plant science datasets. The agent itself was developed with no access to any reference standard. Hui and Li are fully independent holdouts for both the agent and the validation framework; Loladze is independent for the agent but the matching protocol and validation scripts were developed iteratively using Loladze data, making it a development-adjacent dataset for the infrastructure (though not for the agent). We further demonstrate that the agent's output agrees with an independently developed multi-model consensus pipeline on 1,889 observations across 100 papers, providing a circularity-free validation without any ground truth. To our knowledge, this is the first study to (a) formally test equivalence of AI-extracted continuous data against published meta-analysis reference standards, (b) validate across multiple independent datasets, and (c) demonstrate cross-method agreement without ground truth as a scalable alternative to reference-standard validation.

---

# 2. Related Work

## 2.1 The Manual Extraction Burden

Data extraction remains the most labor-intensive phase of systematic review. Buscemi et al. (2006) found that single extractors make errors in 17.7% of fields, with dual extraction reducing the rate to 8.8% at double the cost. Schmidt et al. (2025), in a living systematic review of extraction methods, estimate 2–8 hours per paper depending on complexity. For a 200-paper agricultural meta-analysis, this translates to 400–1,600 hours of skilled labor — a cost that effectively limits the scope and frequency of evidence synthesis.

## 2.2 LLM-Based Extraction

Recent studies have explored large language models for automated data extraction, with mixed results depending on the outcome type. For categorical variables (study design, population characteristics, risk of bias), LLMs consistently achieve >90% accuracy (Khraisha et al., 2024). However, performance drops substantially for continuous numerical outcomes.

Jansen et al. (2025) evaluated GPT-4 for extracting effect sizes from psychology papers, reporting 26–36% accuracy for continuous outcomes — insufficient for meta-analytic use. Kataoka et al. (2026) tested o3 (OpenAI) for clinical data extraction, achieving 75% accuracy on numerical fields but concluding that the approach was "still inadequate" for unsupervised use. In ecology, Gougherty and Clipp (2024) found LLMs reliable for categorical data but poor at quantitative extraction (23.8% accuracy).

Multi-model consensus approaches have shown promise. Khan et al. (2025) used paired LLMs for living systematic reviews, reporting 0.25% hallucination rates when models agreed. Poser et al. (2026) demonstrated that a three-model consensus reduced true extraction errors to 1.48%. Cao et al. (2025) developed OttoSR, achieving 93.1% accuracy on structured data extraction through automated pipeline orchestration.

Several groups have proposed AI as a "second reviewer" to augment human extraction. Helms Andersen et al. (2025) evaluated AI tools in Cochrane reviews, finding high agreement with human reviewers on structured fields. Li, Mathrani, and Susnjak (2025) asked "what level of automation is 'good enough'?" and proposed a framework for evaluating AI extraction quality against human baselines.

## 2.3 The Human Baseline Problem

A critical but underappreciated context for evaluating AI extraction is the reliability of human extraction itself. Tendal et al. (2009) found that when two trained extractors independently extracted continuous data from the same trial reports, they agreed on the same numerical value only 53% of the time — with discrepancies arising from multiplicity in reported outcomes, timepoints, and subgroups. Buscemi et al. (2006) reported 17.7% error rates for single human extractors. These findings suggest that the "gold standard" of human extraction is itself noisy, particularly for continuous outcomes from complex study designs. Any evaluation of AI extraction should be interpreted against this human baseline: perfect agreement with a reference standard is not achievable even by trained humans.

## 2.4 Gaps in the Literature

Despite this progress, four critical gaps remain. First, no study has applied formal equivalence testing (e.g., TOST) to AI-extracted continuous data — accuracy is typically reported as percentage agreement or correlation, which cannot establish statistical equivalence. Second, no study has validated against multiple independent ground-truth datasets, making it impossible to assess generalizability. Third, no study has demonstrated ground-truth-free validation through cross-method agreement, which is essential for scaling beyond the handful of datasets with published reference standards. Fourth, the agricultural domain — with its complex factorial designs and inconsistent reporting — remains largely untested. The Cochrane, Campbell, JBI, and CEE joint position statement on AI use in evidence synthesis (2025) emphasizes the need for validation studies across domains and outcome types before AI tools can be recommended for routine use.

---

# 3. Methods

## 3.1 Agent Architecture

A single AI agent (Claude Opus 4.6, Anthropic, running within the Claude Code CLI environment) read each source PDF directly and extracted observations. We use the term "agent" to denote an AI system that autonomously reads documents and produces structured output, not a multi-step planning agent with tool use or environmental interaction.

The agent operated as a general-purpose reader within a 200K-token context window, sufficient to ingest most scientific papers in their entirety. For each paper, the agent received a brief natural-language instruction specifying what data to extract. For example, the Loladze instruction read: "Extract all mineral element concentration data comparing elevated CO₂ to ambient CO₂ controls. For each observation, report: paper ID, element, tissue, CO₂ levels, treatment mean, control mean, sample size, variance type and value." The agent produced a structured JSON file with one object per observation, following a predefined schema specifying fields for paper metadata, experimental conditions, and statistical values.

No domain-specific prompt templates, few-shot examples, vision extraction pipelines, or multi-model consensus mechanisms were used. The same agent model was used for all three datasets; only the natural-language instruction varied.

For the Loladze dataset, a three-pass workflow was employed: initial extraction, text cross-check (comparing extracted values against original PDF text to flag potential errors), and targeted re-extraction of flagged observations. This produced nine cross-check batch reports and nine re-extractions. Hui and Li datasets used single-pass extraction. The three-pass approach represents a pragmatic choice for the most complex dataset (46 papers with multi-factorial designs across 23 elements), not a fundamental architectural requirement.

**Cost.** Approximately $0.15/paper (March 2026 API pricing), compared to an estimated $120–240/paper for manual extraction at 2–8 hours × $30/hour (Schmidt et al., 2025).

## 3.2 Validation Datasets

No agent parameters were tuned against any reference standard. Hui and Li are fully independent holdouts for both the agent and the validation infrastructure. Loladze is independent for the agent (no agent prompts, parameters, or extraction logic were developed using Loladze reference data), but the matching protocol and validation scripts were developed iteratively with Loladze data during earlier pipeline work, making Loladze a development-adjacent dataset for the infrastructure though not for the agent itself.

**Loladze 2014** (Loladze, 2014). A comprehensive meta-analysis of elevated CO₂ effects on plant mineral concentrations, published in *eLife* with open supplementary data. We processed 46 papers, yielding 655 matched observations across 23 elements. This dataset is the most demanding: complex factorial designs (CO₂ × cultivar × ozone × nitrogen), diverse exposure systems (free-air CO₂ enrichment [FACE], open-top chambers [OTC], growth chambers), and multiple tissues and developmental stages. Effect sizes are typically small (mean −6.3%), requiring high extraction precision to detect.

**Hui et al. 2023** (Hui et al., 2023). A meta-analysis of zinc biofortification in wheat, published in *Journal of Soil Science and Plant Nutrition*. We processed 37 papers, of which 25 matched ground-truth entries, yielding 461 matched observations. This dataset features standardized single-element tabular data with relatively consistent reporting formats. Effect sizes are large (mean ~53%), reflecting substantial zinc concentration increases from biofortification treatments.

**Li et al. 2022** (Li et al., 2022). A meta-analysis of non-microbial biostimulant effects on agronomic outcomes, published in *Frontiers in Plant Science* with open data. We processed 50 papers, of which 31 matched ground truth. After programmatic scale harmonization (accounting for unit conversions such as t/ha ↔ kg/ha), 68 high-confidence observations from 16 papers were identified. The raw extraction (150 obs, r = 0.460) suffered from unit-scale mismatches identical to those observed in independent pipeline validation of the same dataset, confirming that the matching challenge is a property of the data, not of the extraction method.

Together, the three datasets span a range of extraction difficulty (simple tabular to complex factorial), effect-size magnitude (1–200+ pp), and reporting conventions, providing a diverse testbed for evaluating agent performance.

## 3.3 Matching Protocol

Extracted observations were matched to published reference standards using metadata-constrained matching: paper ID and element/tissue (Loladze), paper ID and tissue/application method (Hui), and paper ID with crop species and scale harmonization (Li). Matching proceeded hierarchically: first, candidate observations were filtered by metadata fields (paper, element or tissue, treatment type). Only when multiple extracted candidates matched a single reference row on all metadata fields was minimum-error selection applied as a tiebreaker — analogous to how human dual-extractors resolve discrepancies by checking source data.

Effect sizes are percentage change: (treatment − control) / |control| × 100. All "pp" values throughout this paper refer to differences on this percentage-point scale.

To test sensitivity to the matching algorithm, we ran a bootstrap analysis on Loladze (the dataset most affected by contested matching): of 655 matched observations, 326 (50%) involved the min-error tiebreaker (mean 5.0 candidates per contested match). Random selection among tied candidates over 1,000 bootstrap trials yielded MAE = 9.3% (95% CI: 8.5–10.1%) and r = 0.595 (95% CI: 0.475–0.720), compared with the min-error result of MAE = 5.4% and r = 0.848. Restricting to uncontested observations only (n = 329) yielded MAE = 7.6% and r = 0.693. The tiebreaker thus improves accuracy for contested matches, but even the uncontested-only floor (r = 0.693) confirms meaningful extraction quality independent of the matching protocol. For Hui and Li, matching used raw means (within 20–30% tolerance) while validation used derived effect sizes, so the matching criterion and validation metric were not the same quantity.

## 3.4 Statistical Analysis

**Primary metric.** ICC(3,1) (two-way mixed, consistency) following Koo and Li (2016), who classify 0.50–0.75 as "moderate," 0.75–0.90 as "good," and >0.90 as "excellent." We report ICC(3,1) because the agent represents a single fixed rater applied systematically, not a random sample of raters (Shrout & Fleiss, 1979). We also report Lin's concordance correlation coefficient (CCC; Lin, 1989), the standard metric for method-comparison studies, which simultaneously captures both precision (correlation) and accuracy (bias). Secondary metrics: Pearson r, Spearman ρ (as a robustness check against outlier influence), mean absolute error (MAE), and direction agreement (whether extracted and reference effect sizes share the same sign).

**Equivalence testing.** Two one-sided tests (TOST) with both naive and cluster-robust standard errors. Cluster-robust SEs use the CR1 sandwich estimator, clustering by paper, with t(K−1) degrees of freedom, where K is the number of papers. We note that the CR2 bias-corrected estimator with Satterthwaite degrees of freedom (Pustejovsky & Tipton, 2018) is more appropriate for datasets with few or imbalanced clusters. Supplementary Table S1 reports CR2 results for all datasets. The SE bias correction is minimal (CR2/CR1 ≈ 1.01–1.02), but Satterthwaite df are substantially lower than K−1 due to cluster imbalance (Loladze: 8.1 vs. 45; Hui: 2.6 vs. 29; Li: 3.3 vs. 15). Notably, the Hui ±3 pp equivalence result, which is marginal under CR1 (p = 0.047), does not survive CR2 correction (p = 0.099). All other decisions are unchanged. We retain CR1 as the primary analysis for comparability with prior work and report both estimators. We selected the ±3 pp margin as the primary equivalence threshold based on practical meta-analytic considerations: in inverse-variance pooled meta-analysis, a ±3 pp shift in individual effect sizes produces negligible change in the pooled estimate when averaging over dozens of observations. For context, Tendal et al. (2009) found only 53% agreement between trained human dual-extractors on continuous outcomes — suggesting that even human-human concordance for quantitative data is far from perfect. Our ±3 pp margin corresponds to approximately 25% of the mean Hui effect, 50% of the mean Li effect, and a larger fraction of the mean Loladze effect; consequently, we also report results at tighter margins (±1, ±2 pp) to enable readers to apply their own thresholds.

**Bland-Altman analysis.** Mean difference and 95% limits of agreement (Bland & Altman, 1986), with proportional bias assessed via regression of the difference on the mean.

**Paired bias tests.** Paired t-test, Wilcoxon signed-rank test, and Cohen's d for systematic bias assessment.

**Reproducibility.** A second independent agent run (Run 2) used the same model but was executed with no shared state, cached outputs, or intermediate results. Run 1 and Run 2 observations were matched by value similarity and compared.

**Cross-method agreement.** Agent output was compared against an independently developed multi-model consensus pipeline (details in Appendix A) that shared no code, prompts, or intermediate outputs with the agent. Observations were matched by value similarity (control and treatment means within 25% relative tolerance, with scale-factor harmonization). No ground truth was consulted.

---

# 4. Results

## 4.1 Extraction Accuracy Against Ground Truth

**Table 1. Agent extraction accuracy against published reference standards.**

| Dataset | Papers | Obs | r | ρ | CCC | MAE (pp) | Direction | ICC(3,1) | 95% CI |
|---------|--------|-----|---|---|-----|----------|-----------|----------|--------|
| Loladze 2014 | 46 | 655 | 0.848 | 0.815 | 0.844 | 5.4 | 90% | **0.845** | 0.822–0.866 |
| Hui 2023 | 25 | 461 | 0.942 | 0.915 | 0.942 | 7.4 | 96% | **0.942** | 0.930–0.951 |
| Li 2022 (high-conf) | 16 | 68 | 0.968 | 0.946 | 0.966 | 1.6 | 97% | **0.966** | 0.946–0.979 |
| **Total** | **87** | **1,184** | — | — | — | — | — | — | — |

*ρ = Spearman rank correlation. CCC = Lin's concordance correlation coefficient. Bootstrap 95% CIs (1,000 iterations): Loladze r [0.780, 0.906], MAE [4.8, 6.1]; Hui r [0.914, 0.960], MAE [6.1, 8.8]; Li r [0.944, 0.986], MAE [1.0, 2.4].*

Koo and Li (2016) classify ICC 0.75–0.90 as "good" and >0.90 as "excellent." CCC values closely mirror ICC(3,1), confirming that the agreement is robust to the choice of metric. Spearman ρ values are modestly lower than Pearson r (e.g., 0.815 vs. 0.848 for Loladze), consistent with rank-order perturbations from alignment ambiguity in factorial data — outlier influence does not inflate the Pearson results. The Li and Hui results are excellent; the Loladze result is good, reflecting the greater extraction difficulty of complex factorial designs.

**Loladze.** Of 46 papers, 25 achieved Excellent-tier accuracy (MAE < 5 pp), 11 Good (5–10 pp), 9 Fair (10–20 pp), and 1 Poor (MAE > 20 pp). The single Poor-tier paper (Natali 2009) reflected a matching artifact in trace-metal observations rather than a systematic extraction failure. The agent captured 97% of elements present in the reference standard (23 of 23 elements attempted, with high coverage across Ca, Fe, Zn, Mg, N, P, K, S, Mn, Cu, and others). Aggregate effect: ground truth (hereafter GT) = −6.30%, agent = −5.21%, difference = 1.09 pp.

**Hui.** Four papers achieved perfect extraction (r = 1.0): Erdal, Peck, Zou, and fpls-10-00426. The high ICC reflects the standardized reporting format of zinc biofortification studies: most papers present single-element data in well-structured tables. Aggregate effect: GT = 53.21%, agent = 53.48%, difference = 0.27 pp.

**Li.** The progression from raw matching (r = 0.446, 166 obs) to scale-harmonized high-confidence matching (r = 0.968, 68 obs) changed no extracted values — only the matching methodology. The raw-match degradation resulted from unit-scale mismatches (e.g., yield in t/ha vs. kg/ha), which produce spurious discrepancies when matching by value similarity. Aggregate effect: GT = 11.65%, agent = 11.87%, difference = 0.22 pp.

[FIGURE 1: Scatter plots of agent-extracted vs. reference-standard effect sizes. Panel A: Loladze 2014 (r = 0.848, N = 655). Panel B: Hui 2023 (r = 0.942, N = 461). Panel C: Li 2022 high-confidence (r = 0.968, N = 68). Dashed lines indicate identity (y = x).]

[FIGURE 2: Per-paper MAE distribution across all three datasets, sorted by accuracy. Color-coded by dataset (Loladze = blue, Hui = green, Li = orange). Loladze: median 3.8 pp, IQR 1.5–7.2. Hui: median 4.1 pp, IQR 1.0–9.5. Li: median 0.8 pp, IQR 0.2–2.1.]

## 4.2 Formal Agreement Statistics

### 4.2.1 Equivalence Testing (TOST)

**Table 2. Cluster-robust TOST results.**

| Dataset | N | K | Margin | p-value | Result |
|---------|---|---|--------|---------|--------|
| **Li 2022** | 68 | 16 | ±1 pp | 0.009 | PASS |
| | | | ±2 pp | <0.001 | PASS |
| **Hui 2023**‡ | 461 | 30 | ±2 pp | 0.141 | FAIL |
| | | | ±3 pp | 0.047 | PASS |
| **Loladze 2014** | 655 | 46 | ±2 pp | 0.091 | FAIL |
| | | | ±3 pp | 0.003 | PASS |

Design effects: Loladze = 2.66, Hui = 4.09, Li = 0.49. The Hui design effect reflects substantial within-paper correlation (one paper contributes 104 of 461 observations), which inflates the effective standard error. Naive TOST passes at ±2 pp for all three datasets (all p < 0.014); the cluster adjustment substantially inflates standard errors for Hui and Loladze. Under CR2 with Satterthwaite degrees of freedom (Supplementary Table S1), the Hui ±3 pp result becomes non-significant (p = 0.099); all other decisions are unchanged.

‡ The Hui TOST used K = 30 citation-level clusters (some papers appear under variant citation strings in the matching output), while 25 unique papers contributed matched observations. Using K = 25 would reduce degrees of freedom and yield slightly more conservative p-values; the ±3 pp equivalence result should therefore be interpreted with this caveat.

### 4.2.2 Bias Assessment

**Table 3. Systematic bias tests.**

| Dataset | Mean diff (pp) | Paired t | p | Cohen's d | Interpretation |
|---------|---------------|----------|---|-----------|----------------|
| Loladze 2014 | +1.09 | 2.63 | 0.009 | 0.103 | negligible |
| Hui 2023 | +0.27 | 0.35 | 0.726 | 0.016 | negligible |
| Li 2022 | +0.22 | 0.53 | 0.596 | 0.065 | negligible |

All Cohen's d < 0.20 (the conventional threshold for "small"), indicating negligible practical bias in all three datasets. The statistically significant Loladze bias (p = 0.009) is not practically meaningful given its magnitude (+1.09 pp on a −6.30% mean effect, or approximately 17% of the effect size) and its negligible Cohen's d.

### 4.2.3 Bland-Altman Agreement

| Dataset | Mean diff (pp) | 95% Limits | Prop. bias r | Prop. bias p |
|---------|---------------|------------|-------------|-------------|
| Li 2022 | +0.22 | −6.5 to +7.0 | 0.223 | 0.067 |
| Hui 2023 | +0.27 | −32.5 to +33.0 | 0.085 | 0.069 |
| Loladze 2014 | +1.09 | −19.6 to +21.8 | −0.150 | <0.001 |

The wide Hui limits (±33 pp) reflect the large effect-size range in zinc biofortification studies, where some studies report >200% increases from foliar zinc application. Loladze shows statistically significant proportional bias (r = −0.150, p < 0.001), indicating that extraction errors are somewhat larger for observations with extreme effect sizes. For aggregate meta-analytic pooling — the primary use case — small mean bias matters more than individual scatter, and the mean differences (0.22–1.09 pp) are well within practically meaningful thresholds.

[FIGURE 3: Bland-Altman plots (difference vs. mean) for all three datasets. Panel A: Loladze 2014. Panel B: Hui 2023. Panel C: Li 2022. Horizontal lines show mean difference (solid) and 95% limits of agreement (dashed). Proportional bias regression line shown for Loladze.]

## 4.3 Run-to-Run Reproducibility

**Table 4. Run-to-run reproducibility (Run 1 vs. Run 2).**

| Dataset | Papers | Matched obs | r | MAE (pp) | Effect diff (pp) |
|---------|--------|-------------|---|----------|-------------------|
| Loladze 2014 | 41 | 665 | 0.816 | 8.4 | 0.09 |
| Hui 2023 | 24 | 362 | 0.946 | 12.3 | 6.31 |
| Li 2022 | 30 | 204 | 0.849 | 5.6 | 0.23 |
| **Total** | **95** | **1,231** | — | — | — |

Aggregate effect sizes are highly stable for Loladze (0.09 pp) and Li (0.23 pp), demonstrating that stochastic variation in the agent's extraction cancels at the aggregate level. At the paper level, 27 papers achieved perfect reproducibility (r = 1.0 between runs): 8 Loladze papers, 8 Hui papers, and 11 Li papers.

The larger Hui aggregate gap (6.31 pp vs. <0.3 pp for Loladze and Li) reflects three factors: (a) Hui zinc biofortification studies report effect sizes an order of magnitude larger than the other datasets (mean ~68% vs. <20%), amplifying absolute errors; (b) greater run-to-run variability in which factorial treatment combinations were extracted (15 of 24 papers differed by >3 observations between runs); and (c) the matched-pair aggregate gap (6.31 pp) substantially exceeds the unmatched aggregate gap (0.23 pp), indicating a composition effect from differential observation matching. Five papers (Cakmak 1997, Kalayci 1999, Yilmaz 1997, Li 2013, fpls-10-00426) account for the entire discrepancy; excluding them reduces the gap to 0.31 pp.

The overall pattern — high aggregate stability with moderate per-observation noise — is consistent with alignment ambiguity (which factorial sub-condition to extract) rather than reading errors. This distinction is explored further in Section 4.5.

## 4.4 Cross-Method Agreement (Ground-Truth-Free)

The strongest epistemic warrant comes from comparing the agent against a structurally independent extraction method without any ground truth. We compared agent output against a multi-model consensus pipeline (Appendix A) that shared no code, prompts, or intermediate outputs with the agent. The pipeline uses a different primary model family (Kimi K2.5 and Gemini) and a different architectural approach (dual-model consensus with tiebreaker).

**Table 5. Agent–pipeline agreement (no ground truth used).**

| Dataset | Papers | Matched obs | r | Direction | Effect diff (pp) |
|---------|--------|-------------|---|-----------|-------------------|
| Loladze 2014 | 44 | 1,205 | **0.933** | 91% | 1.30 |
| Hui 2023 | 20 | 185 | **0.971** | 96% | 0.29 |
| Li 2022 | 36 | 499 | **0.994** | 88% | 1.89 |
| **Total** | **100** | **1,889** | — | — | — |

All three correlations exceed 0.93. Two largely independent methods — differing in primary model family, architecture, prompts, and implementation — converge on the same effect sizes for 1,889 observations across 100 papers in three agricultural domains. Overall effect sizes differ by only 0.29–1.89 pp between methods. This convergence provides a circularity-free validation: neither method was calibrated against the other, and no ground truth was consulted.

[FIGURE 4: Scatter plots of agent-extracted vs. pipeline-extracted effect sizes (no ground truth). Panel A: Loladze 2014 (r = 0.933, N = 1,205). Panel B: Hui 2023 (r = 0.971, N = 185). Panel C: Li 2022 (r = 0.994, N = 499). Dashed lines indicate identity (y = x).]

## 4.5 Error Taxonomy

### 4.5.1 Classification of Discrepancies

Among Loladze papers where specific extraction errors could be diagnosed (121 observations with sufficient documentation to trace the source of discrepancy), we classified each discrepancy into one of three categories:

**Alignment ambiguity** (93%, 113/121). The agent extracted correct values from the correct table but selected a different factorial sub-condition than the original meta-analyst. For example, in a CO₂ × cultivar × nitrogen experiment with 12 treatment combinations, both the agent and the original analyst might extract wheat grain zinc — but the agent selects the high-nitrogen treatment while the analyst selects the low-nitrogen treatment. Both are defensible analytical choices; neither is "wrong."

**True reading errors** (3%, 4/121). The agent extracted an incorrect value from the paper. Two specific cases were identified: Campbell 2002 (extracted total phosphorus instead of foliar phosphorus concentration, a column-selection error) and Niu 2013 (missed the NH₄ treatment arm, extracting only the NO₃ data). These represent genuine extraction failures.

**Undiagnosable** (3%, 4/121). The discrepancy could not be traced to a specific cause from the available documentation.

### 4.5.2 The Granularity Barrier

The dominance of alignment ambiguity over true reading errors (97% vs. 3% of diagnosable discrepancies) reveals a structural constraint we term the "Granularity Barrier." Complex factorial agricultural experiments present multiple valid extraction targets within the same table. When a meta-analyst specifies "extract wheat grain zinc under elevated CO₂," the instruction is under-determined if the paper reports data for 3 cultivars × 2 nitrogen levels × 2 harvest dates. Different extractors — human or AI — will make different analytical sub-selections.

This finding has two implications. First, improving the AI's reading ability would yield diminishing returns; the binding constraint is the analytical decision of *which* factorial sub-condition to extract. Second, per-observation accuracy metrics (MAE, Bland-Altman limits) overstate the practical extraction error, because alignment ambiguities add noise at the observation level but cancel when pooled across many observations. This explains why aggregate effect errors (0.22–1.09 pp) are substantially smaller than per-observation MAE (1.6–7.4 pp).

[FIGURE 5: Error taxonomy for Loladze dataset. Pie chart showing alignment ambiguity (93%), true reading errors (3%), and undiagnosable (3%). Inset: example of alignment ambiguity from a factorial table.]

Future benchmark datasets should document analytical sub-selections (which specific rows, columns, and treatment combinations were extracted) to enable principled decomposition of validation error into alignment vs. extraction components.

---

# 5. Discussion

## 5.1 Principal Findings

The strongest evidence comes from the two fully independent holdout datasets: Hui (ICC = 0.942, CCC = 0.942) and Li (ICC = 0.966, CCC = 0.966). Neither dataset was seen during agent development, and the validation infrastructure was built without access to their reference standards. Both achieve "excellent" agreement (ICC > 0.90) and pass cluster-robust equivalence testing at ±3 pp. These results establish that a single general-purpose AI agent can extract continuous quantitative data from scientific PDFs with sufficient accuracy for aggregate meta-analytic pooling.

The Loladze dataset (ICC = 0.845, CCC = 0.844, "good" agreement) presents a more nuanced picture. Although the agent itself was developed without access to Loladze reference data, the matching protocol and validation scripts were refined iteratively using Loladze during earlier pipeline work (Section 3.2). The Loladze accuracy should therefore be interpreted as a development-adjacent benchmark rather than a fully independent test. That said, the matching sensitivity analysis (Section 3.3) shows that even with random tiebreaker selection, the uncontested-only correlation (r = 0.693, n = 329) confirms meaningful extraction quality independent of any circularity in the matching protocol.

Across all three datasets, Cohen's d ranged from 0.016 to 0.103 (all negligible), and aggregate effect sizes were reproduced within 0.22–1.09 pp — well within the range of human dual-extractor disagreement on continuous data (Tendal et al., 2009). A structurally independent multi-model pipeline (Appendix A) converged on the same values (r > 0.93, 1,889 observations) without ground truth, and the pipeline's own formal statistics against the same reference standards are consistent with the agent's (Appendix A, Table A1).

## 5.2 Cost-Effectiveness

**Table 6. Cost comparison of extraction methods.**

| Method | Cost/paper | Time/paper | Accuracy | Source |
|--------|-----------|------------|----------|--------|
| Single human extractor | $60–240 | 2–8 hours | 82.3% field-level* | Buscemi et al., 2006; Schmidt et al., 2025 |
| Dual human extraction | $120–480 | 4–16 hours | 91.2% field-level* | Buscemi et al., 2006 |
| AI-assisted human | ~$30 + $0.15 | ~1 hour | 91.0% field-level | Helms Andersen et al., 2025 |
| AI agent (this study) | $0.15 | ~2 min | ICC 0.85–0.97† | — |

*Field-level accuracy (% of data fields without errors). †ICC is not directly comparable to field-level accuracy; it measures agreement on continuous effect sizes rather than discrete correctness.

At ~$0.15/paper and approximately two minutes of processing time, the agent enables meta-analyses that would be prohibitively expensive to conduct manually. The cost for the entire 87-paper validation (~$13) is less than the labor cost of manually extracting a single paper. For a hypothetical 500-paper agricultural meta-analysis, the agent would cost approximately $75 and complete extraction in under a day, compared to an estimated $30,000–120,000 and 3–12 months for manual dual extraction.

This cost structure also makes *living meta-analyses* economically viable for the first time. A living systematic review that updates extraction from 200 new papers annually would cost approximately $30/year with AI extraction, compared to $24,000–96,000/year for manual dual extraction — a difference that explains why living reviews remain rare despite their recognized value (Elliott et al., 2014). When the cost of re-extraction falls below the cost of a journal subscription, the economics of evidence synthesis fundamentally change: researchers can afford to re-extract entire corpora when protocols improve, rather than locking in extraction decisions made years earlier.

## 5.3 The Granularity Barrier

The Loladze dataset illustrates a challenge inherent in any extraction system applied to complex factorial data. When the agent extracts correct numbers from the correct table but selects different factorial sub-conditions than the original meta-analyst, the resulting discrepancy is a legitimate analytical disagreement, not an extraction error. This explains why per-observation MAE (5.4 pp) is substantially larger than aggregate effect error (1.09 pp): alignment ambiguities add noise at the observation level but cancel when pooled.

The Granularity Barrier is not specific to AI extraction. Human dual-extractors face the same challenge: when instructions do not fully specify which factorial combination to extract, extractors will make different choices. The difference is that human discrepancies are resolved through discussion, while AI discrepancies are revealed only through validation. Future meta-analysis protocols should provide explicit extraction rules for factorial designs (e.g., "extract the highest nitrogen treatment if multiple levels are reported") to reduce this source of variability for both human and AI extractors.

## 5.4 Comparison with Published Systems

**Table 7. Comparison with published LLM extraction systems.**

| System | Domain | Models | Quant. accuracy | Equivalence test | GT-free validation |
|--------|--------|:------:|-----------------|:----------------:|:------------------:|
| Jansen et al. 2025 | Clinical | 1 | 26–36% effect sizes | No | No |
| Kataoka et al. 2026 | Clinical | 1 | 75% (o3) | No | No |
| Khan et al. 2025 | Clinical | 2 | 0.25% halluc. (concordant) | No | No |
| Poser et al. 2026 | Clinical | 3 | 1.48% true-error | No | No |
| Gougherty & Clipp 2024 | Ecology | 1 | 23.8% | No | No |
| Cao et al. 2025 (OttoSR) | Clinical | Multiple | 93.1% | No | No |
| **This study** | **Plant sci.** | **1** | **ICC 0.845–0.966** | **TOST ≤ 0.047** | **r > 0.93 (1,889 obs)** |

*Note: Accuracy metrics are not directly comparable across studies due to differences in domains, outcome types, denominators, and evaluation methodology.*

Our results substantially exceed published benchmarks for continuous numerical extraction, though direct comparison is constrained by differences in domain, outcome types, and evaluation methodology (Table 7 note). The difference likely reflects three factors: (1) a frontier model with its full 200K-token context window, allowing the agent to read entire papers rather than isolated excerpts; (2) the structured JSON output schema, which constrains extraction format and reduces hallucination; and (3) the agricultural domain's relatively standardized numerical reporting compared to clinical trial narratives.

It is worth noting that many comparison systems were evaluated on harder tasks (e.g., extracting from clinical narratives with ambiguous outcome definitions) or used older, less capable models. As model capabilities improve rapidly, the specific accuracy numbers reported here are less important than the validation framework: ICC, CCC, TOST, and cross-method agreement provide a reusable toolkit for evaluating any future extraction system.

## 5.5 Recommendations for Practice

Based on our validation, we propose the following extraction equivalence testing (EET) protocol for researchers deploying AI extraction in meta-analyses:

1. **Pilot validation.** Extract 5–10 papers for which ground-truth values are known. Compute ICC, CCC, and MAE. If ICC < 0.75 or CCC < 0.70, revise prompts before proceeding.
2. **Cross-method agreement.** Run a second, structurally independent extraction method (different model, different prompts) on the full corpus. Compute inter-method r. If r > 0.90 on ≥50 observations, the extraction is likely reliable for aggregate pooling.
3. **Sensitivity analysis.** Re-run extraction on a random 20% subsample. If aggregate effects shift by <2 pp and per-observation MAE changes by <15%, the extraction is stable.
4. **Transparency.** Report the model name and version, exact prompts, matching protocol, and all formal agreement statistics (ICC, CCC, TOST). Deposit extraction code and outputs in a public repository.
5. **Human oversight.** Spot-check papers with MAE > 15 pp or flagged by cross-method disagreement. The AI extracts; the researcher verifies outliers and makes analytical decisions.

This protocol operationalizes the Cochrane/Campbell/JBI/CEE (2025) position statement's call for AI validation evidence while remaining practical for individual researchers without dedicated validation infrastructure.

## 5.6 Limitations

1. **Single model.** Results are specific to Claude Opus 4.6 (March 2026). Model updates may alter performance, and model deprecation renders results tied to a specific model version. Users should re-validate when updating models.

2. **Three-pass workflow.** The Loladze extraction used a three-pass approach (extract, cross-check, re-extract), which is more involved than truly zero-shot. Hui and Li used single-pass extraction.

3. **Observation-level scatter.** Bland-Altman limits span ±7–33 pp depending on the dataset. The agent is better suited for aggregate pooling than observation-level precision.

4. **Variance extraction.** We validated means and effect sizes but did not systematically validate variance (SD/SE) extraction, which is known to be challenging for LLMs.

5. **Li sample size.** Only 68 high-confidence Li observations were validated, limiting statistical power for that dataset.

6. **Hui reproducibility.** The 6.31 pp aggregate effect difference between runs reflects composition effects from differential extraction coverage, not systematic extraction drift.

7. **Domain scope.** All three datasets are from plant science. Generalization to clinical trials, social sciences, or other domains with different reporting conventions is untested.

8. **Single-author validation.** Matching protocols were developed and applied by a single author. All code and data are publicly available to enable independent replication.

9. **Proportional bias.** Loladze showed significant proportional bias (r = −0.150): errors correlated with effect magnitude. Users should apply sensitivity analyses for extreme-value subgroups.

10. **Data contamination.** Claude Opus 4.6 was trained on internet data through early 2025. All three reference datasets are publicly available: Loladze's supplementary data is hosted on eLife's website; Hui and Li published in Frontiers journals with open data. We cannot rule out that the model encountered these values during training. However, three observations argue against memorization as the primary driver of performance: (a) the agent makes errors — alignment ambiguities, wrong factorial conditions, occasional column-selection errors — that would not occur with memorized data; (b) run-to-run variation (Table 4) shows the agent is actively processing documents rather than recalling fixed values; and (c) the ground-truth-free agreement between the agent and a pipeline using different model families (Kimi and Gemini) would not be improved by Claude-specific memorization.

11. **Non-English papers.** All source papers were in English. Performance on non-English publications is untested.

12. **Figure-only data.** Papers where quantitative data appear only in figures (not tables) were not separately validated. The agent can read figures but accuracy may differ from tabular extraction.

## 5.7 Ethical Considerations

The Cochrane, Campbell, JBI, and CEE joint position statement (2025) on AI use in evidence synthesis requires disclosure of AI tool use and human oversight of AI-generated outputs. We endorse this position and note several considerations for responsible deployment of AI extraction tools.

First, this tool augments rather than replaces human judgment. Analytical decisions — inclusion criteria, outlier handling, factorial sub-condition selection, and quality assessment — remain with the researcher. The agent extracts data; the researcher decides what to do with it.

Second, the low cost and speed of AI extraction creates a risk of low-quality meta-analysis "mills" if analytical judgment is also automated. We emphasize that extraction is only one component of evidence synthesis; protocol development, quality assessment, and interpretation require domain expertise that AI currently cannot provide.

Third, all AI-assisted extraction should be disclosed per RAISE (Reporting AI use in Systematic reviews and meta-analyses in Evidence synthesis) recommendations. Researchers should report the model, version, prompts, and any post-processing applied to AI-extracted data.

## 5.8 Implications

For plant science — with decades of unprocessed literature on CO₂ effects, biofortification, biostimulants, and other topics — reliable AI extraction at this accuracy level has immediate practical significance. A single researcher with API access can now extract data from hundreds of papers in days rather than months, enabling meta-analyses that were previously infeasible due to resource constraints. If even 10% of the estimated 600,000 researcher-hours spent annually on manual data extraction in agricultural sciences (based on ~300 published agricultural meta-analyses/year × 200 papers × 5 hours/paper) could be automated at equivalent quality, the freed capacity would correspond to approximately 30 full-time equivalent researchers redirected toward analysis and interpretation.

The cross-method agreement framework (agent vs. pipeline, no ground truth) offers a scalable approach to validating AI extraction systems. Rather than requiring curated reference standards — which are expensive to create and limited in availability — researchers can establish confidence by demonstrating convergence between structurally independent extraction methods. This is analogous to inter-rater reliability between human extractors, but between AI systems operating on the same source documents. When two independent AI methods agree on an extracted value, the probability that both made the same error is low — particularly when the methods differ in model family, architecture, and prompts, and when the error is not driven by ambiguity in the source document itself.

---

# 6. Conclusion

A single AI agent reading scientific PDFs directly achieves ICC = 0.845–0.966 against published reference standards across three independent plant science meta-analysis datasets (1,184 observations, 87 papers). Cluster-robust TOST (CR1) confirms equivalence at ±3 pp for all datasets (p ≤ 0.047), though the Hui result is marginal and does not survive CR2 bias correction (p = 0.099; Supplementary Table S1). Aggregate effects are reproduced within 0.22–1.09 pp (all Cohen's d < 0.20). Run-to-run reproducibility is high at the aggregate level (0.09–0.23 pp for Loladze and Li). Ground-truth-free comparison with a structurally independent multi-model pipeline yields r > 0.93 on 1,889 observations across 100 papers. Error analysis reveals that alignment ambiguity — not reading errors — dominates extraction discrepancies, suggesting that the binding constraint on accuracy is analytical decision-making, not AI capability.

At ~$0.15/paper, the approach reduces extraction cost by approximately three orders of magnitude relative to manual methods while achieving formal statistical equivalence. The cross-method agreement framework provides a scalable, ground-truth-free validation paradigm applicable beyond the three datasets studied here.

The error taxonomy reveals a finding with implications beyond AI validation: the binding constraint on extraction accuracy is not the quality of the extractor — human or AI — but the precision of the extraction protocol. When factorial experimental designs produce multiple defensible extraction targets, no extractor can be "correct" without unambiguous instructions specifying which sub-condition to select. Future meta-analysis guidelines should address this granularity problem explicitly, for the benefit of both human and AI extractors.

For agricultural science, where the volume of unprocessed primary literature far exceeds the capacity for manual extraction, AI agents offer a path to more comprehensive, more timely, and more reproducible evidence synthesis.

Code, configuration files, and validation scripts are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture.

---

# References

- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307–310.
- Buscemi, N., Hartling, L., Vandermeer, B., Tjosvold, L., & Klassen, T. P. (2006). Single data extraction generated more errors than double data extraction in systematic reviews. *Journal of Clinical Epidemiology*, 59(7), 697–703.
- Cao, X., et al. (2025). OttoSR: Automation of systematic reviews with large language models. *medRxiv*. https://doi.org/10.1101/2025.01.15.25320588
- Cochrane, Campbell Collaboration, JBI, & CEE. (2025). Position statement on the use of artificial intelligence in the production of evidence syntheses. *Environmental Evidence*, 14(20).
- Elliott, J. H., Turner, T., Clavisi, O., et al. (2014). Living systematic reviews: An emerging opportunity to narrow the evidence-practice gap. *PLoS Medicine*, 11(2), e1001603.
- Gougherty, A. V., & Clipp, H. L. (2024). Testing the reliability of an AI-based large language model to extract ecological information from the scientific literature. *npj Biodiversity*, 3(1), 13.
- Helms Andersen, T., et al. (2025). Using AI tools as second reviewers in systematic reviews. *Cochrane Evidence Synthesis and Methods*.
- Higgins, J. P. T., Thomas, J., Chandler, J., et al. (Eds.). (2023). *Cochrane Handbook for Systematic Reviews of Interventions* (version 6.4). Cochrane.
- Hui, Y., Wang, J., Jiang, T., Li, S., Zhang, Y., & Liu, X. (2023). Zinc biofortification of wheat through soil, foliar, and combined applications: A meta-analysis. *Journal of Soil Science and Plant Nutrition*, 23, 5384–5397.
- Jansen, T., et al. (2025). Data extraction by generative artificial intelligence. *Psychological Bulletin*, 151(10), 1280–1306.
- Kataoka, Y., et al. (2026). Automating the data extraction process for systematic reviews using GPT-4o and o3. *Research Synthesis Methods*, 17, 42–62.
- Khan, M. A., Ayub, U., Naqvi, S. A. A., et al. (2025). Collaborative large language models for automated data extraction in living systematic reviews. *JAMIA*, 32(4), 638–647.
- Khraisha, Q., et al. (2024). Can large language models replace humans in systematic reviews? A study of LLM performance in screening and extracting data. *Research Synthesis Methods*, 15(4), 616–626.
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155–163.
- Li, J., Van Gerrewey, T., & Geelen, D. (2022). A meta-analysis of biostimulant yield effectiveness in field trials. *Frontiers in Plant Science*, 13, 836702.
- Li, Y., Mathrani, A., & Susnjak, T. (2025). What level of automation is "good enough"? A meta-analysis of AI-assisted systematic review accuracy. *arXiv:2507.15152*.
- Lin, L. I.-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics*, 45(1), 255–268.
- Loladze, I. (2014). Hidden shift of the ionome of plants exposed to elevated CO₂ depletes minerals at the base of human nutrition. *eLife*, 3, e02245.
- Nakagawa, S., et al. (2023). A robust and readily implementable method for the meta-analysis of response ratios with and without missing standard deviations. *Ecology Letters*, 26(2), 232–244.
- Poser, P. L., Klimas, R., Luerweg, J., et al. (2026). Improving reliability and accuracy of structured data extraction using a consensus large-language model approach. *Frontiers in Artificial Intelligence*.
- Schmidt, L., Shokraneh, F., Pieper, D., & Mathes, T. (2025). Data extraction methods for systematic review (semi)automation: Update of a living systematic review. *F1000Research*.
- Pustejovsky, J. E., & Tipton, E. (2018). Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. *Journal of Business & Economic Statistics*, 36(4), 672–683.
- Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater reliability. *Psychological Bulletin*, 86(2), 420–428.
- Tendal, B., Higgins, J. P. T., Jüni, P., Hróbjartsson, A., & Gøtzsche, P. C. (2009). Multiplicity of data in trial reports and the reliability of meta-analyses: Empirical study. *BMJ*, 339, b3128.
- Topp, C. F. E., et al. (2023). AgroEcoList: A checklist to improve reporting of ecological research in agronomy. *PLOS ONE*, 18(6), e0285478.

---

# Data Availability Statement

Agent extraction code, configuration files, validation scripts, and pre-computed outputs are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture. Reference-standard datasets are from published meta-analyses: Loladze (2014), Hui et al. (2023), and Li et al. (2022). Source PDFs cannot be redistributed due to publisher copyright.

# Author Contributions (CRediT)

All roles: Moshe Halpern.

# Conflict of Interest Statement

The author declares no conflicts of interest. The AI model used is a commercial product; the author has no financial relationship with the provider beyond standard API usage fees.

# Ethics and Funding

Not applicable (published literature only; no human subjects). No external funding was received.

---

# Figure Captions

**Figure 1.** Scatter plots of agent-extracted vs. reference-standard effect sizes (percentage change from control). Panel A: Loladze 2014 (r = 0.848, N = 655). Panel B: Hui 2023 (r = 0.942, N = 461). Panel C: Li 2022 high-confidence (r = 0.968, N = 68). Dashed lines indicate identity (y = x). Points are colored by paper.

**Figure 2.** Per-paper MAE distribution across all three datasets, sorted by accuracy. Color-coded by dataset (Loladze = blue, Hui = green, Li = orange). Loladze: median 3.8 pp, IQR 1.5–7.2. Hui: median 4.1 pp, IQR 1.0–9.5. Li: median 0.8 pp, IQR 0.2–2.1.

**Figure 3.** Bland-Altman plots showing the difference between agent-extracted and reference-standard effect sizes plotted against their mean. Panel A: Loladze 2014. Panel B: Hui 2023. Panel C: Li 2022. Horizontal solid lines show mean difference; dashed lines show 95% limits of agreement. The regression line for proportional bias is shown for Loladze (r = −0.150, p < 0.001).

**Figure 4.** Scatter plots of agent-extracted vs. pipeline-extracted effect sizes (ground-truth-free comparison). Panel A: Loladze 2014 (r = 0.933, N = 1,205). Panel B: Hui 2023 (r = 0.971, N = 185). Panel C: Li 2022 (r = 0.994, N = 499). Dashed lines indicate identity (y = x). Neither method was calibrated against the other.

**Figure 5.** Error taxonomy for the Loladze dataset (121 diagnosable discrepancies). Alignment ambiguity — the agent and original meta-analyst selected different factorial sub-conditions from the same table — accounts for 93% of discrepancies. True reading errors account for 3%.

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

Two LLMs independently extract data using identical structured prompts. Consensus categories: **High confidence** (2+ models agree within ±15% tolerance), **Medium** (single model or tiebreaker-resolved), **Low** (vision-only). Cost: ~$0.37/paper.

## A.2 Pipeline Validation Results

The pipeline was validated against the same three reference standards as the agent. Note: the Loladze dataset was used during pipeline development (not a true holdout), while Hui and Li were independent holdouts for both the pipeline and the agent.

**Table A1. Pipeline validation results (for comparison with agent Table 1).**

| Dataset | Papers | Obs | r | MAE (pp) | ICC(3,1) | TOST ±2pp | TOST ±3pp |
|---------|--------|-----|---|----------|----------|-----------|-----------|
| Hui 2023 | 19 | 308 | 0.999 | 0.43 | 0.999 | p < 0.001 PASS | p < 0.001 PASS |
| Li 2022 (all) | 27 | 200 | 0.951 | 2.30 | 0.949 | p = 0.052 FAIL | p = 0.002 PASS |
| Loladze 2014† | 46 | 413 | 0.886 | 4.36 | 0.870 | p = 0.020 PASS | p = 0.001 PASS |

† Development case study, not an independent holdout for the pipeline.

**Key comparison.** On the two datasets that are independent holdouts for both methods:
- **Hui**: Pipeline ICC = 0.999, Agent ICC = 0.942. The pipeline achieves higher per-observation accuracy on this standardized dataset, likely due to multi-model cross-validation.
- **Li**: Pipeline ICC = 0.949, Agent ICC = 0.966. The agent achieves comparable or slightly better results on the biostimulant dataset.

Both methods achieve negligible bias (all Cohen's d < 0.20) and strong aggregate effect reproduction. The pipeline's main advantage is its built-in confidence stratification, which concentrates 95% of large errors in flagged observations. The agent's advantages are simplicity and lower cost ($0.15 vs. $0.37/paper).

## A.3 Consensus Ablation

On a fixed 322-observation scope (Loladze), single-model performance varied:

| Method | MAE (pp) | r |
|--------|----------|---|
| Kimi solo | 4.10 | 0.903 |
| Consensus | 4.54 | 0.886 |
| Gemini solo | 5.53 | 0.843 |
| Claude solo | 6.29 | 0.742 |

Consensus does not improve per-observation accuracy over the best single model (identified post-hoc). Its value is threefold: (1) confidence prediction without ground truth, (2) 15% coverage gain, and (3) robustness against model-specific blind spots.
