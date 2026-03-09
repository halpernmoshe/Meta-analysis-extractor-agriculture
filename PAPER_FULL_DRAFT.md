# Multi-Model AI Consensus for Reliable Data Extraction in Plant Science Meta-Analysis

**Moshe Halpern**

Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization -- Volcani Center, Israel

---

# Abstract

**Background:** Data extraction is the primary bottleneck in meta-analysis (2--8 hours per paper), and LLMs that perform well on categorical study characteristics achieve only 26--36% accuracy on continuous quantitative outcomes, the variables meta-analysis actually pools. Existing systems provide no mechanism for identifying which extractions are trustworthy without reference-standard comparison. No AI extraction system has been validated on agricultural or ecological quantitative data.

**Methods:** We developed a dual-model consensus pipeline (Claude Sonnet 4 + Kimi K2.5, with Gemini 3 Flash tiebreaker) in which inter-model agreement drives a three-tier confidence system: observations confirmed by two or more models receive "high" confidence; single-model or vision-only extractions are flagged for review. We validated against three published plant science reference datasets: Loladze 2014 (CO2/mineral concentrations, 46 papers, development set), Hui et al. 2023 (Zn biofortification/wheat, 18 papers, **zero-shot holdout validation**: no prompt tuning, matching adjustments, or parameters were set based on Hui data; only the JSON schema was domain-configured), and Li et al. 2022 (biostimulants/agronomic outcomes, 28 papers, cross-domain). Systematic per-paper diagnostic audits were conducted for all three datasets.

**Results:** Zero-shot holdout validation on Hui achieved **r = 0.993, MAE = 1.73%, 99% direction agreement**, the strongest reported result for continuous numerical extraction from plant science tables, demonstrating that the **Reading Barrier** (accurate table reading) is effectively broken. Confidence stratification predicted accuracy: consensus-dominant papers achieved MAE = 4.3% versus 11.2% for vision-dependent papers (2.6×), with 95% of large errors concentrated in the flagged minority. The **Granularity Barrier** (analytical sub-selection in factorial designs) was quantified via Loladze's `info`-column: separating sub-selection failures from reading errors reduces the Loladze MAE from 7.9% to 4.3%. The **Provenance Barrier** (reference-standard heterogeneity) explains the Li 2022 headline r = 0.453: per-paper audit identified 12 of 28 papers with reference-standard artefacts (wrong source PDFs, attribution errors, aggregation mismatches). On the **Structurally Concordant Subset** (16 papers with clean same-level comparisons), **r = 0.996, MAE = 0.44 pp**. Aggregate meta-analytic effects reproduced to within 0.05--0.40 pp across all datasets (|Cohen's d| < 0.07; TOST equivalence p < 0.001 at ±2 pp for Loladze and Hui).

**Conclusions:** The pipeline operates as a high-recall candidate generator: exhaustive extraction followed by multi-model consensus filtering (initial precision: ~4.5%; post-consensus recall: 100% in the Structurally Concordant Subset). Approximately 75% of consensus-confirmed observations are auto-validated; the remainder are flagged for human review at ~2 min each versus ~10 min for de novo extraction. Median cost ~$0.24/paper (mean $0.37, range $0.12–$3.50), with ~70% reduction in human extraction time. Critically, the binding constraint on automated meta-analysis is no longer table-reading capability (the Reading Barrier, now effectively solved), but rather pre-specification of analytical sub-selection rules (the Granularity Barrier) and validation-infrastructure integrity (the Provenance Barrier). Accuracy metrics from complex factorial datasets must be decomposed into reading errors and sub-selection concordance failures for honest evaluation; failure to do so systematically underestimates AI extraction capability.

**Keywords:** meta-analysis, data extraction, large language models, consensus, quality prediction, automation, plant science, methodological concordance

---

# 1. Introduction

Meta-analysis is the cornerstone of evidence-based practice in agricultural and environmental science. Because individual experiments are conducted across diverse soils, climates, and cultivars, consistent biological patterns often emerge only through quantitative pooling. For example, Loladze (2014) synthesized 1,481 mineral concentration measurements from 130 species across 25 elements to reveal the hidden nutritional cost of elevated CO2. However, the primary bottleneck in such syntheses is data extraction. Trained researchers must manually identify, read, and record quantitative values, a process requiring 2 to 8 hours per paper (Schmidt et al., 2025). Furthermore, Buscemi et al. (2006) demonstrated that single-extractor error rates reach 17.7%, falling to 8.8% only under costly dual-extraction protocols. For large meta-analyses encompassing 100+ papers, the human cost of data extraction alone can approach thousands of hours.

Agricultural field trials present distinct extraction challenges that complicate automated data retrieval, particularly compared to the clinical trials that dominate current language model research. Lacking standardized frameworks like the CONSORT reporting guidelines (Topp et al., 2023), agricultural studies must be interpreted on their own terms. Plant science experiments routinely employ complex factorial designs (CO2 x cultivar x soil amendment x harvest date x application rate), producing multi-layered tables where only a fraction of the dozens of treatment combinations may be relevant to a specific meta-analytic question. Furthermore, diverse exposure systems (Free-Air CO2 Enrichment, Open-Top Chambers, controlled growth chambers) dictate which treatments qualify as valid controls. Outcomes span multiple plant tissues (leaf, grain, root, stem) across various developmental stages, and data appear in heterogeneous units (g/kg dry weight vs. mg/100g fresh weight vs. µmol/g) that vary both between and within studies. Variance reporting is similarly inconsistent, utilizing standard errors (SE), standard deviations (SD), least significant differences (LSD), or merely letter-based significance groupings. Consequently, nearly 70% of ecological meta-analysis datasets include studies with missing standard deviations (Nakagawa et al., 2023), and 26% of published plant ecology meta-analyses rely on unweighted analyses due to unavailable variance data (Koricheva & Gurevitch, 2014). Finally, older literature from the 1980s and 1990s, critical for longitudinal CO2 research, often exists only as scanned PDFs with embedded image tables, further obstructing machine readability.

While large language models (LLMs) have demonstrated high proficiency in extracting categorical study characteristics, they struggle with the continuous quantitative outcomes that form the basis of meta-analytical pooling. For structured categorical variables, systems achieve exceptional reliability, with Gartlehner et al. (2024) reporting 96.3% accuracy on clinical variables and Gougherty & Clipp (2024) confirming >90% accuracy for simple ecological categories. However, these successes mask severe deficiencies in numeric extraction. In the most comprehensive accuracy study to date, encompassing 2,179 studies and 312,329 extractions, Jansen et al. (2025) found that performance varied primarily by variable type rather than by the specific LLM or systematic review: effect-size variables achieved only 26--36% accuracy, compared to 90%+ for categorical items. Similar deficits appear across the literature: Peng et al. (2025) observed 69--72% accuracy for means and SDs from clinical randomized controlled trials, Yun et al. (2024) reported a 48.7% exact match rate for GPT-4 on continuous outcomes, and Kataoka et al. (2026) concluded that even OpenAI's advanced o3 reasoning model performed poorly on numeric variable extraction and was still inadequate. Ultimately, while LLMs excel at semantic comprehension, reliable continuous numerical extraction from complex tables remains an unsolved problem.

Beyond raw accuracy, practical deployment of automated extraction requires reliable confidence estimation at the level of individual data points. Existing systems typically report aggregate performance metrics but lack mechanisms to predict which specific extractions are correct, a critical requirement for minimizing human review. Multi-model concordance offers a principled solution to this reliability problem. Khan et al. (2025) demonstrated that when two independent LLMs agree on an extracted value, the hallucination rate is just 0.25%, whereas disagreement signals a hallucination rate of 26--41%. Poser et al. (2026) independently confirmed this dynamic, showing that three-model consensus reduced clinical data extraction errors to 1.48%. Because concordance status is generated intrinsically during extraction, it provides a powerful quality signal that requires no ground truth. However, neither study deployed this concordance metric as a proactive confidence estimator to automatically stratify observations for human review.

The development of such automated systems has also been heavily skewed toward biomedical applications, leaving the complex domain of agricultural ecology largely unaddressed. Scott et al. (2025) found that 17 of 19 generative AI systematic review studies focused on clinical or biomedical settings, despite demonstrations of fully automated Cochrane review synthesis achieving 93.1% accuracy (Cao et al., 2025). Agricultural meta-analysis represents the most demanding tier of extraction complexity, characterized by statistical outcomes, heterogeneous units, and an absence of reporting standards (Li et al., 2025, Tier 3), and no published system has been validated against it. In such complex extraction environments, single-model systems are particularly vulnerable; Tan & D'Souza (2026) identified four structural LLM failure modes in evidence extraction: role confusion, binding drift, multi-instance compression, and error amplification, suggesting that individual models possess characteristic blind spots. Overcoming these structural vulnerabilities requires a multi-model consensus approach capable of detecting and filtering these domain-specific errors.

Underlying all three gaps is a more fundamental epistemic challenge. Evaluating AI extraction accuracy requires a reference standard, but no published meta-analysis dataset constitutes error-free ground truth: each was assembled by human researchers navigating the same complex tables, ambiguous reporting conventions, and analytical sub-selection decisions that confront any extraction system. Attribution errors, mismatched PDFs, and defensible-but-different methodological choices are inevitable artefacts of human-curated data at scale. Consequently, validating AI against published reference standards conflates extraction errors with reference-standard heterogeneity, and apparent low performance may reflect the limitations of the standard rather than the capability of the system. We address this directly by treating our three validation datasets not as gold standards but as an imperfect triangulation strategy: each dataset was assembled under different conditions and illuminates a different dimension of the extraction problem. Systematic per-paper diagnostic audits reveal which discordances reflect genuine AI limitations and which trace to reference-standard artefacts, a decomposition that is itself a methodological contribution of this work.

To address these challenges, we present a multi-model consensus pipeline validated on three published plant science datasets, each designed to isolate a distinct barrier to reliable AI extraction. We first evaluate the **Reading Barrier** (the fundamental ability of an LLM to accurately extract numbers from complex tables) using the Hui (2023) zinc biofortification dataset. On this standardized, single-element dataset, zero-shot holdout validation yielded r = 0.993 and 99% direction agreement, demonstrating that basic numeric comprehension is effectively solved. We next address the **Granularity Barrier**, testing whether the system can match a human meta-analyst's selection of analytical sub-conditions from complex factorial designs. Using the Loladze (2014) development set, we show that separating fundamental reading errors from sub-selection concordance failures reduces the effective extraction mean absolute error (MAE) from 7.9% to approximately 4.3%. Finally, we expose the **Provenance Barrier** using the Li (2022) cross-domain dataset to assess whether reference standards and source PDFs are themselves free of curation heterogeneity. A per-paper audit revealed that 12 of 28 apparent discordances stemmed from reference-standard artefacts (incorrect PDFs, attribution errors, or structural level mismatches) rather than true extraction failures, with the remaining 16 clean comparisons achieving r = 0.996. Ultimately, we demonstrate that multi-model agreement reliably predicts which extractions clear all three barriers, enabling robust, confidence-stratified triage without the need for ground truth.

---

# 2. Methods

## 2.1 Pipeline Architecture

The pipeline consists of four stages (Figure 1): challenge-aware reconnaissance, dual-model extraction, consensus building with tiebreaker, and confidence-stratified post-processing.

**Challenge-aware reconnaissance.** For each paper, Claude Sonnet 4 performs a structured scan identifying: variance reporting format, sample size locations, tables containing target outcome data, experimental design characteristics, and extraction challenges. Papers are classified by challenge type (SCANNED, IMAGE-TABLES, FIGURE-ONLY) and routed to one of three extraction modes: TEXT (clean machine-readable tables), HYBRID (text + vision for image-embedded tables), or VISION (figure-only data requiring image analysis).

**Dual-model extraction.** Two LLMs independently extract data using identical structured prompts: Claude Sonnet 4 (Anthropic, 200K context) and Kimi K2.5 (Moonshot AI, 256K context, reasoning enabled). Both receive the same prompt containing target variable definitions, table targeting directives from reconnaissance, structured output format requirements, and a checklist of all target elements. The prompt instructs models to report null rather than guess when values are uncertain. For HYBRID-mode papers, Gemini 3 Flash additionally performs vision-based extraction from PDF page images. Full prompt templates are provided in the online repository. **Vision extraction note:** All three models support native multimodal (vision) input: Claude Sonnet 4.6 (Anthropic), Kimi K2.5 (Moonshot AI), and Gemini 3.1-pro-preview (Google). For HYBRID-mode papers (those containing image-embedded tables or degraded scanned PDFs), all three models receive PDF page images, enabling triple-model vision verification. Observations confirmed by two or more models receive HIGH confidence; single-model extractions receive LOW confidence and are flagged for human review. This triple-vision architecture eliminates the need for OCR pre-processing and ensures that any single model's misreading is overridden by the majority.

**Data privacy note:** This pipeline transmits extracted document text and page images to third-party commercial API providers (Anthropic, Moonshot AI, Google). Users should ensure compliance with institutional data governance policies and publisher terms before processing restricted, paywalled, or sensitive materials. Extracted text (not binary PDF files) is transmitted; the pipeline does not upload raw PDFs.

**Consensus building.** After independent extraction, observations from both models are compared using element-tissue matching with value tolerance (default: 15% relative error; for near-zero values, 0.5 units absolute):

- **Matched pairs** (both models agree within tolerance): Accepted at "high" confidence, values averaged
- **Unmatched observations** (single model only): Retained at "medium" confidence with source model noted
- **Vision-only observations** (from HYBRID/VISION mode without text consensus): Retained at "low" confidence

When initial consensus is poor (defined as a **global match rate <30%** of the total observations extracted by the lead model), a tiebreaker is invoked: Gemini 3 Flash performs an independent extraction using the same prompt, and 2-of-3 voting determines accepted observations.

**Confidence assignment.** Each observation receives a confidence label based on its provenance:

- **High**: Two or more models independently extracted matching values. These observations have the strongest reliability guarantee.
- **Medium**: Extracted by a single text-based model without corroboration, or resolved via tiebreaker.
- **Low**: Extracted via vision/OCR from image-embedded tables or scanned PDFs, often without multi-model consensus.

This three-tier system enables downstream triage: high-confidence observations can be used directly; medium and low-confidence observations are flagged for human review. We evaluate whether these labels predict actual accuracy in Section 3.3.

**Post-processing.** Final observations undergo duplicate removal, null-mean filtering, and treatment/control swap flagging. Automatic swap correction was tested and found harmful (Section 3.9); flagging is informational only.

**Design rationale: Strategic Heterogeneity vs. Self-Consistency.** An alternative architecture would run all three models on every paper and take majority vote — or sample the same model multiple times (Self-Consistency; Wang et al., 2022). We chose *heterogeneous* dual extraction with a conditional tiebreaker for two reasons. First, self-consistency reduces stochastic variance within a single model but cannot reduce systematic inductive bias: if one model consistently interprets "main effect averaged across co-treatments" while another extracts "within-treatment CO2 effect," running the same model ten times converges on one systematic choice. Baslam et al. (2012) illustrates this: Claude extracted 38 observations (main-effect view) while Kimi extracted 76 (interaction-aware view) — their disagreement flags the analytical ambiguity rather than amplifying one model's systematic choice. Second, cost: the tiebreaker is invoked for only 22% of papers, reducing per-paper cost by approximately 30% compared to always running three models. Claude operates on extracted text; both Kimi and Gemini are natively multimodal and can process page images directly. This complementary capability mix, text-dominant with vision fallback across two independent models, would be lost in a three-text-model majority vote. The ablation (Section 3.12) confirms that no single model dominates across all elements, supporting the heterogeneous approach.

**Worked example.** Figure 1 (right panel) illustrates the pipeline processing two contrasting papers. For Baslam et al. (2012), a clean paper with structured tables, both Claude and Kimi independently extract identical Ca values (8.21 mg/g control, 7.43 mg/g elevated), yielding 100% consensus and an MAE of 1.0%. For Fangmeier et al. (2002), a complex CO2 × O3 factorial design, only Kimi extracts usable data from text (Claude returns 0 observations), requiring Gemini's vision fallback. The consensus fraction drops to 23%, correctly predicting the higher error (MAE = 8.0%), though as Section 3.9 documents, that error is predominantly a methodological concordance issue rather than a table-reading failure.

## 2.2 Configuration-Driven Design

The pipeline is configured via JSON files specifying target outcome variables, control and treatment definitions, expected elements, tissue types, and moderator variables. Switching between meta-analysis topics requires only changing the configuration file:

```json
{
  "topic": "CO2 effects on plant mineral concentrations",
  "outcome_variable": "MINERAL_CONC",
  "control": {"description": "ambient CO2 (~400 ppm)"},
  "treatment": {"description": "elevated CO2 (>500 ppm)"},
  "elements": ["N","P","K","Ca","Mg","Fe","Zn","Mn","Cu","S"],
  "models": {"primary": "claude-sonnet-4", "secondary": "kimi-k2.5",
             "tiebreaker": "gemini-3-flash"}
}
```

## 2.3 Validation Datasets

### 2.3.1 Loladze 2014 (Development Dataset)

The primary validation dataset is from Loladze (2014), a comprehensive meta-analysis of elevated CO2 effects on plant mineral concentrations (1,481 observations, ~130 references, 25 elements). We processed 50 papers, of which 46 matched to reference-standard references, yielding 635 matched observations across 14 elements.

**Important caveat**: The pipeline was developed and iteratively refined using feedback from this dataset. Prompt templates, matching logic, and consensus parameters were adjusted based on Loladze validation results. This dataset therefore provides an upper-bound estimate of performance and should not be considered an independent test. We explicitly report this distinction and rely on the Hui dataset (Section 2.3.2) for zero-shot holdout validation.

**Methodological concordance caveat**: The Loladze 2014 dataset contains an `info` field for each observation that documents the specific methodological sub-selections made by Loladze when computing effect sizes from complex factorial designs. This field, analyzed in Section 3.9, reveals that a substantial portion of the Loladze MAE reflects methodological concordance failures, cases where the pipeline and Loladze selected different but equally legitimate analytical sub-conditions, rather than numerical reading errors. We use this field systematically to decompose the reported MAE into its extraction and concordance components.

### 2.3.2 Hui et al. 2023 (Zero-Shot Holdout Validation)

The secondary dataset is from Hui et al. (2023), a meta-analysis of zinc biofortification in wheat (1,593 observations, 139 studies). We processed 34 papers, of which 18 matched reference-standard entries, yielding 310 matched observations.

**This dataset was not used during pipeline development.** No prompt modifications, matching adjustments, or parameter tuning were performed based on Hui results. The pipeline was configured by changing only the JSON configuration file (specifying Zn wheat biofortification outcomes, moderators including application method type, and dose ranges). We designate this "zero-shot holdout" to acknowledge that while no Hui data informed the model or prompts, the extraction schema was domain-configured -- as is unavoidable for any structured extraction task. This provides the cleanest test of the system's generalizability.

**Wrong-PDF note**: Per-paper diagnostic audit identified one paper (`Li_2013.pdf`) that contains a completely different study (Impa et al. 2013, rice hydroponic) from the one cited in the Hui 2023 database (Li, M.H. et al. 2013, wheat field fertilization). Six of the 12 GT rows for this paper appear as spurious numerical matches arising from coincidental overlap between rice grain Zn values (7--38 mg/kg) and the narrow range of GT wheat Zn values (27.9--28.7 mg/kg). This phenomenon is discussed in Section 3.5 and parallels the wrong-PDF problem identified independently in the Li 2022 dataset (Section 4.4).

### 2.3.3 Li et al. 2022 (Cross-Domain Validation)

The third dataset is from Li et al. (2022), a meta-analysis of non-microbial biostimulant effects on agronomic outcomes (1,108 observations, 181 studies). Target outcomes include crop yield (primary), biomass, and quality traits (e.g., total phenols, antioxidant activity, fat content), reflecting the breadth of biostimulant research; approximately 25% of matched observations in our analysis were quality traits rather than yield *sensu stricto*. This tests positive-direction effects across diverse crops and heterogeneous units. We processed 28 papers, all matched to reference standard, yielding 163 matched observations. Like Hui, this dataset was not used during development.

## 2.4 Reference-Standard Matching Protocol

For each extracted observation, a corresponding reference-standard row was identified using a dataset-specific hierarchical matching algorithm. All matching criteria were specified before computing accuracy metrics; no criterion was modified after examining results.

**Loladze matching.** Observations were matched by exact element symbol (after normalization: "Fe" = "Iron" = "iron"; upper- and lower-case equivalents collapsed) and tissue type (leaf/grain/root/whole-plant). When multiple extracted observations existed for a given paper--element--tissue combination (the common case for factorial designs), the candidate with the smallest absolute difference in log-response-ratio from the reference-standard value was selected (minimum-error selection). **Disclosure (optimistic matching upper bound):** When multiple extracted candidates match a reference row, selecting the minimum-error candidate establishes an upper bound on extraction capability, assuming an ideal downstream selection process. The `n_candidates` field in the validation CSV records how many candidates existed per match; 84% of matches had only one candidate (unambiguous assignment), limiting the impact of this optimistic selection.

**Hui matching.** Observations were matched by tissue type and application method code (app_type). Within each paper--tissue--app_type stratum, the minimum-error candidate was selected. Control and treatment mean values were matched using a ±15% relative-error tolerance for initial pairing, with the closest pair selected when multiple fell within tolerance.

**Li matching (primary, naive).** Observations were matched by crop species and a freely chosen outcome label, without filtering by outcome type. This "naive" matching yields 163 matched pairs (r = 0.453) and is the primary reported result because it requires no analyst judgment in outcome selection. A yield-outcome-filtered variant (described in Section 3.6) is reported for comparison.

Unmatched observations on either side (extracted but no reference-standard counterpart, or reference-standard rows with no extracted match) were excluded from accuracy calculations and are tabulated separately as precision and recall components in Section 3.1.

## 2.5 Per-Paper Diagnostic Audit

For all three datasets, we conducted systematic per-paper diagnostic audits using Claude Sonnet 4.6 acting as an independent reader (separate from the extraction pipeline). For each paper, the auditing agent received: (1) the full source PDF text, (2) the corresponding reference-standard rows for that paper, and (3) a structured diagnostic template requiring it to document the experimental design, identify which tables contained target data, compare extracted values against the reference standard, and classify any discrepancies into mutually exclusive categories: reading error, methodological sub-selection, reference-standard artifact (wrong PDF, database attribution error), or aggregation-level mismatch. The auditing agent had no access to the extraction pipeline's code or aggregate accuracy statistics, ensuring independence from the primary extraction. The diagnostic template and 30 example reports are provided in the online repository.

The `info` field of the Loladze reference CSV provided the primary evidence for decomposing Loladze discordances into extraction versus concordance components. This field, populated by Loladze (2014) for each observation, documents specific sub-selection choices (year, site, co-treatment arm, cultivar) that the auditing agent could compare against the pipeline's analytical choices.

## 2.6 Validation Metrics

Six metrics assessed extraction accuracy: (1) Pearson correlation coefficient (r) between extracted and reference-standard effect sizes; (2) Mean Absolute Error (MAE) in percentage points; (3) within-threshold accuracy (proportion within 5%, 10%, 20% of reference-standard values); (4) **direction agreement**: defined as sign(extracted effect) = sign(reference effect), where "effect" is (treatment mean − control mean) / |control mean| × 100; observations where the reference-standard effect is within ±0.5% of zero (near-zero, sign unreliable) are excluded from direction agreement calculations; (5) element capture rate; (6) overall effect reproduction (aggregate mean effect comparison).

Formal agreement analyses included Bland-Altman analysis, intraclass correlation [ICC, using a two-way mixed-effects model (ICC type 3,1), absolute-agreement definition, single-measurement unit], two one-sided tests (TOST) for equivalence, and cluster-robust percentile bootstrap confidence intervals (10,000 resamples), where the resampling unit was the study (paper) rather than the individual observation, to account for within-study clustering of extraction errors. The percentile method was preferred over BCa to avoid instability in the acceleration estimate when the number of independent clusters is small (n=16–46 papers). Throughout this paper, "pp" denotes percentage points. All ICC models are reported uniformly using the (model, type, unit) notation.

## 2.7 Confidence-Stratified Analysis

To evaluate whether the pipeline's confidence scores predict actual accuracy, we classified each paper by its consensus fraction: the proportion of observations where two or more models agreed. Papers with >50% consensus-confirmed observations were classified as "consensus-dominant"; the remainder as "vision-dependent." We compared accuracy metrics between these groups and across confidence tiers.

## 2.8 Cost Analysis

Per-paper costs were estimated from API usage records, broken down by model and stage. Manual extraction costs were estimated at $30/hour with 4 hours per paper based on literature estimates.

## 2.9 Prospective Application Dataset: Silicon Effects on Wheat

To demonstrate the pipeline in a genuinely prospective setting (no reference standard available), we applied the current production system to 40 published papers on silicon fertilization effects on wheat yield and related agronomic outcomes. Papers were identified from a targeted literature search and comprise field trials, greenhouse studies, and controlled-environment experiments reporting grain yield (primary), biomass, thousand-kernel weight, and nutrient uptake under silicon versus no-silicon control conditions. The dataset spans publications from 2008 to 2025, including papers in English, Polish, Czech, Farsi, and Portuguese; non-English papers required vision-based extraction because OCR-derived text was unavailable.

**Triple-vision configuration.** Unlike the validation experiments, which used Claude Sonnet 4 and Kimi K2.5 for text extraction with Gemini 3 Flash as a vision tiebreaker, the prospective application uses all three models in full vision mode: Claude Sonnet 4.6 (Anthropic), Kimi K2.5 (Moonshot AI), and Gemini 3.1-pro-preview (Google) each independently extract from PDF page images. An observation receives HIGH confidence if two or more models extract matching values (within 15% relative error); observations extracted by only one model receive LOW confidence and are flagged for human review.

**Test-retest reproducibility.** To assess intra-pipeline reliability without ground truth, we ran the triple-vision extraction twice independently on all 40 papers (temperature > 0 for Kimi and Gemini ensures stochastic variation between runs). Observations from the two runs were matched by paper, outcome, silicon dose, and control mean. We computed ICC(2,1) and Pearson r between matched treatment means as an estimate of extraction stability. This test-retest design is analogous to inter-rater reliability in classical psychometrics: high reproducibility indicates that the extraction signal is driven by document content rather than model stochasticity.

**LLM audit.** As an independent quality check, a separate Claude Sonnet 4.6 instance audited each HIGH and MEDIUM confidence observation against the source PDF, tasked with locating the claimed values in specific table cells and providing source citations. The auditor received only the extracted observations and the PDF images; it was not informed which models produced which values. The audit verification rate (percentage of observations confirmed with a source citation) provides a complementary quality signal distinct from the cross-model consensus rate.

---

# 3. Results

## 3.1 Pipeline Output

The consensus pipeline processed 50 papers from the Loladze dataset, generating 1,652 consensus observations across 14 mineral elements. Of these, 64% were routed to HYBRID extraction and 30% to TEXT-only. The Gemini tiebreaker was invoked for 11 papers (22%). Means extraction was near-complete (>98%), while variance capture was 67%.

Across all three datasets, the pipeline processed 112 papers, extracting 2,676 total observations, of which 1,077 matched to reference-standard entries (Table 1).

## 3.2 Overall Accuracy: Three Datasets as a Difficulty Gradient

**Table 1. Validation results across all three datasets.**

| Metric | Loladze 2014 | Hui 2023 | Li 2022 | Li 2022 (Conc.)† |
|--------|:------------:|:--------:|:-------:|:------------------:|
| Status | Development | **Holdout (ZS)** | Cross-domain | Cross-domain |
| Topic | CO2 + minerals | Zn biofortification | Biostimulant + agronomic outcomes | Biostimulant + agronomic outcomes |
| Expected direction | Negative | Positive | Positive | Positive |
| Papers matched to GT | 46 | 18 | 28 | 16 |
| Matched observations | 635 | 310 | 163 | 100 |
| Elements / outcomes | 14 | 1 (Zn) | 1 (yield) | 1 (yield) |
| Pearson r | 0.669 | **0.993** | 0.453 | **0.996**† |
| MAE (pp) | 7.9 | **1.73** | 11.62 | **0.44**† |
| Median AE | 3.0% | 0.0% | 3.39% | 0.17% |
| Within 5% | 58% | 90% | 55% | 95% |
| Within 10% | 74% | 93% | 66% | 98% |
| Within 20% | 91% | 97% | 81% | 100% |
| Direction agreement | 85% | **99%** | 87% | **98%**† |
| Overall effect diff | 0.05 pp | 0.40 pp | 0.06 pp | -- |
| ICC (observation) | 0.669 | 0.993 | 0.429 | -- |
| ICC (paper-level) | 0.838 | 0.932 | 0.509 | -- |
| Cohen's d | -0.003 | -0.069 | 0.003 | -- |
| TOST (±2 pp) | p < 0.001 | p < 0.001 | p = 0.126 | -- |
| TOST (±3 pp) | p < 0.001 | p < 0.001 | p = 0.042 | -- |
| TOST (±5 pp) | p < 0.001 | p < 0.001 | p = 0.002 | -- |

† Li 2022 (Conc.) = Structurally Concordant Subset: 16 papers with verified same-level comparisons (excluding 2 PDF/consensus-failure papers, 4 GT-attribution-error papers (including 1 GT outcome-category mismatch), and 6 papers with structural level mismatches (per-year GT vs. multi-year-average extraction, product-arm omissions, or GT values from pre-publication data)). All 16 papers achieve 100% capture rate. See Supplementary Table S1 and Section 4.4.

The three datasets each measure something different. Hui (r = 0.993) is a clean test of pure reading accuracy: single element, standardized units, homogeneous context. Loladze (r = 0.669) measures both reading accuracy and methodological concordance, specifically whether the pipeline makes the same analytical sub-selection choices as Loladze in complex CO2 factorial designs. Section 3.9 shows that separating these two components reduces the effective Loladze reading-error MAE from 7.9% to approximately 4.3%. Li (r = 0.453 overall) adds a third dimension: reference-standard heterogeneity, where some GT rows and input PDFs have their own provenance issues. On the 16 Li papers where the comparison is structurally clean, r = 0.996. These three datasets are not points on a difficulty curve; they are three different validation instruments, each measuring a different combination of pipeline and validation-infrastructure properties. A key finding from the Li audit is that reference-standard heterogeneity — wrong input PDFs, database attribution errors, and structural level mismatches in the ground truth — is currently a larger bottleneck to accurate validation than AI extraction accuracy itself. The pipeline consistently extracted correct values; the validation infrastructure was the limiting factor in 12 of 28 Li papers.

## 3.3 Multi-Model Agreement Predicts Extraction Quality

The pipeline's confidence scores predicted actual extraction accuracy (Figure 2). We classified each Loladze paper by its consensus fraction (proportion of observations confirmed by 2+ models). Papers were divided into two groups.

Papers where more than half of observations were confirmed by inter-model agreement (consensus-dominant) had MAE = 4.3% with strong correlation to the reference standard. Direction agreement exceeded 90%. These represent cases where independent agreement between Claude and Kimi provides a reliable guarantee. Papers relying primarily on single-model or vision extraction (vision-dependent, <50% consensus) had MAE = 11.2%, approximately 2.6× worse than consensus-dominant papers. The 6.96 pp accuracy gap between TEXT and HYBRID extraction modes (2.27% vs. 9.23%; Section 3.1) corroborates this split via an independent, routing-based classification.

At the observation level, high-confidence observations (confirmed by 2+ models) had MAE = 5.2%, while medium/low-confidence observations had MAE = 9.6% (Mann-Whitney p < 0.001). The large majority of errors concentrated in vision-dependent papers: 95% of observations with absolute error >20 pp came from papers routed to HYBRID or VISION extraction modes.

**Holdout validation of confidence stratification (Hui dataset).** To test whether the confidence signal generalises beyond the development set, we examined confidence tiers for the Hui holdout dataset. Of 339 extracted Hui observations, 235 (69%) were high-confidence; 104 (31%) were medium/low-confidence. At the paper level, 9 of 12 papers with extractable data (75%) were consensus-dominant. This mirrors the Loladze development finding and confirms that the confidence tier is a genuine quality signal rather than a development-set artefact.

**TEXT-mode performance.** Papers routed to TEXT extraction (clean machine-readable tables) achieved MAE = 2.27% with r = 0.974 (140 observations). This represents near-perfect accuracy and establishes the system's capability ceiling when paper quality permits clean text extraction.

**HYBRID-mode performance.** Papers requiring vision supplementation (HYBRID mode, 420 observations) achieved MAE = 9.23% with r = 0.532. The 6.96 pp accuracy gap between TEXT and HYBRID modes validates the routing classifier's relevance: papers flagged as challenging are genuinely harder to extract, and the system correctly identifies them.

## 3.4 What Predicts Consensus Quality?

If multi-model agreement predicts extraction accuracy, what paper attributes predict whether consensus will be achieved? We examined 20 binary challenge features detected during reconnaissance against each paper's consensus fraction using Mann-Whitney tests (Figure 7).

Paper difficulty was the strongest predictor. Papers classified as MEDIUM difficulty during reconnaissance achieved 83% mean consensus (14/18 consensus-dominant), while HARD papers averaged 46% consensus (8/21 consensus-dominant). MEDIUM papers also achieved lower median MAE (3.2%) than HARD papers (7.6%). The total number of detected challenges correlated negatively with consensus fraction (r = -0.37, p = 0.019).

Among individual features, `has_complex_stats` was the most predictive (p < 0.001): papers with complex statistical reporting (ANOVA interaction terms, mixed-model outputs, non-standard variance reporting) averaged 30% consensus versus 76% for papers with straightforward tables. Other features showing meaningful effects included `has_image_tables` (34% vs. 66% consensus), `is_scanned` (49% vs. 71%), and `has_nested_tables` (53% vs. 70%).

Multi-table papers had higher consensus (68% vs. 39%, p = 0.09), likely because papers with multiple tables tend to have well-structured data in at least one table. The same feature also predicted lower MAE (6.0% vs. 19.1%, p = 0.01).

These findings suggest that the reconnaissance stage's difficulty classification is an effective pre-extraction predictor of consensus quality. In a production setting, papers classified as HARD could be automatically flagged for human review or allocated additional extraction passes, while MEDIUM papers can be processed with high confidence in the consensus mechanism.

Jansen et al. (2025) found across 312,329 extractions that accuracy depends more on variable type than on which LLM is used: variables describing study context had higher accuracy than variables for direct effect-size calculation, exactly the distinction between reconnaissance-phase variables (paper type, design, crops) and extraction-phase variables (mean, SD, n). Our finding that `has_complex_stats` is the single strongest predictor of poor consensus (p < 0.001) is the structural equivalent: complex statistical reporting creates a variable-type difficulty that no individual LLM overcomes reliably, but that multi-model disagreement reliably flags.

## 3.5 Zero-Shot Holdout Validation: Hui et al. 2023

The Hui dataset provides the cleanest test of pipeline accuracy because it was not used during development. Across 310 matched observations from 18 papers:

- Pearson r = 0.993 (p < 0.001)
- MAE = 1.73%
- Direction agreement = 99% (308/310)
- Eight papers achieved perfect or near-perfect extraction (MAE < 0.2%)

The overall extracted Zn effect was 50.18% versus reference-standard 50.57% (diff = 0.40 pp). No systematic bias was detected (paired t = -1.21, p = 0.227, Cohen's d = -0.069). The strong performance reflects simpler data structure (single element, standardized units, fewer moderators) and confirms that the pipeline generalizes to unseen data without retuning.

ICC = 0.993 at the observation level (95% CI: 0.991--0.994), indicating excellent agreement between automated and manual extraction.

**Scanned-PDF fallback success (Liu et al. 2019).** Liu et al. (2019), a Zn soil fertilization dose-response trial in winter wheat (Quzhou Experimental Station, two cropping seasons), was flagged HARD during reconnaissance due to scanned PDF format. Kimi K2.5 returned zero observations (blocked by OCR degradation). Claude Sonnet 4 extracted 63 observations using its vision pathway; Gemini 3 Flash independently confirmed the extraction in HYBRID mode. All 10 GT-targeted grain-Zn observations were matched: r = 1.0, MAE = 0.17%. This illustrates that the heterogeneous architecture prevents silent failures: a single-model text pipeline would have returned zero results for this paper, while the multi-modal HYBRID fallback and independent confirmation produced a near-perfect result.

**PDF–reference mismatch (Li_2013).** Per-paper diagnostic audit identified one file (`Li_2013.pdf`) that does not match the paper cited in the Hui 2023 database: the processed PDF contains an IRRI greenhouse rice study (Impa et al. 2013) rather than the Chinese wheat field study cited (Li, M.H. et al. 2013). The reconnaissance phase correctly issued seven out-of-scope warnings, which in a production workflow would trigger an exclusion review. Apparent matches (r = 0.643, 6/12 matched) are numerical coincidences between rice and wheat Zn values rather than valid comparisons. The headline Hui statistics (r = 0.993, MAE = 1.73%) are reported including this paper for transparency; excluding it would yield slightly improved metrics. This illustrates that the reconnaissance warning system can flag mismatched PDFs, enabling pre-extraction screening.

## 3.6 Cross-Domain Validation: Li et al. 2022

The Li dataset tested generalization to positive-direction effects across diverse crops, biostimulant types, and agronomic outcomes (yield, biomass, and quality traits). The pipeline reproduced the aggregate outcome effect to within 0.06 pp (extracted +15.43% vs. GT +15.37%, paired t-test p = 0.97, Cohen's d = 0.003) with zero systematic bias, consistent with the Loladze and Hui findings. Among individual matched pairs, the 81 observations with unambiguous alignment (|ext − GT| / |GT| < 0.25) achieved r = 0.996, confirming that the pipeline extracts these values near-perfectly.

The headline Pearson r = 0.453 across all 163 matched observations is dominated by the validation comparison rather than extraction quality. Per-paper diagnostic audit of all 28 papers (Section 4.4) identified that 12 of 28 papers had structural reasons preventing valid same-level comparison: 3 papers had wrong input PDFs (the files in the directory were different studies from the ones cited in the Li 2022 database), 3 had GT attribution errors (wrong crop or metric in the database), 3 had aggregation-level mismatches (GT stored per-year observations; pipeline extracted multi-year averages, a legitimate difference in analytical choice), 2 had GT values from pre-publication data with systematic inflation, and 1 had a product-selection omission. On the **16 papers with clean same-level comparisons, r = 0.996, MAE = 0.44 pp, 98% direction agreement** (100 matched observations, 100% capture rate across all 16 papers).

**Positive benchmark: Chen et al. (2021).** This multi-site, multi-season sugarcane seaweed extract trial (flagged during reconnaissance as a HARD scanned PDF) achieved r = 0.9954, MAE = 0.15 pp across 21 matched observations with 100% direction agreement. Both Claude and Kimi independently extracted near-identical values from OCR-processed Table 7 across six growing seasons at two sites, demonstrating that scanned PDFs with clear table structure are handled with near-Hui accuracy. This result clarifies where the residual challenge lies: the practical failure mode is not the scanned format per se, but legacy document quality. Modern high-resolution scans extract near-perfectly; pre-2000 photocopied documents with degraded OCR — common in older CO2 literature — represent the true hard case, and these are correctly identified and flagged by the SCANNED challenge classifier.

Median absolute error (3.39%) was much lower than the mean (11.62%), confirming that errors are concentrated in a minority of matches. TOST confirmed equivalence at ±3 pp (p = 0.042) and ±5 pp (p = 0.002).

**Yield-outcome-filtered matching (pre-specified sensitivity analysis).** As a sensitivity analysis, we applied a yield-keyword filter to the naive matching: outcome labels containing non-yield terms (biomass, nitrogen uptake, protein content, root length, etc.) were excluded using a pre-specified 28-term exclusion list, and only outcomes matching yield-specific terms (yield, production, ton, kg/ha, etc.) were retained. This filter was specified before examining accuracy metrics and is defensible because it matches our research question (yield response) to the reference standard's intended outcome. The filtered matching produced 204 matched pairs (r = 0.787, MAE = 9.0%), improving substantially over the naive 163-pair result. We report the naive matching (r = 0.453) as the primary result to avoid any appearance of post-hoc selection; the filtered result is reported here for completeness. Neither variant is used to compute the Structurally Concordant Subset statistics, which rely on the paper-level audit (Section 4.4).

## 3.7 Aggregate Effect Reproduction

Across all three datasets, the pipeline reproduced aggregate meta-analytic effects with high fidelity (Table 1):

- **Loladze**: GT mean = -4.91%, extracted = -4.96%, diff = 0.05 pp
- **Hui**: GT mean = 50.57%, extracted = 50.18%, diff = 0.40 pp
- **Li**: GT mean = +15.37%, extracted = +15.43%, diff = 0.06 pp

For Loladze, per-element mean effects were closely reproduced for key elements: Zn (0.03 pp diff), Ca (0.24 pp), Mg (0.30 pp), P (0.67 pp), K (0.69 pp). Larger discrepancies occurred for trace elements with small samples: Mn (6.37 pp), B (5.46 pp), Na (4.02 pp).

Across all three datasets, the absence of systematic bias was confirmed by paired t-test (Loladze: t = -0.08, p = 0.93, d = -0.003; Hui: t = -1.21, p = 0.23, d = -0.069; Li: t = 0.04, p = 0.97, d = 0.003). Errors are random and cancel in the aggregate, the property essential for meta-analysis.

## 3.8 Paper-Level Accuracy

Papers were classified into accuracy tiers based on MAE (Figure 3):

- **Excellent** (MAE < 5%): 22 papers (48%), including 6 papers with MAE < 0.1%
- **Good** (5--10%): 10 papers (22%)
- **Fair** (10--20%): 13 papers (28%)
- **Poor** (> 20%): 1 paper (2%)

Thus, 70% of papers achieved Good or Excellent accuracy, and 98% were at least Fair. The single Poor paper (Niu et al. 2013, MAE = 58%) had only 2 observations under atypical phosphorus-deficient conditions.

Excellent-tier papers were overwhelmingly consensus-dominant: 82% had >50% consensus-confirmed observations. Poor and Fair papers had higher vision dependence, confirming that the presence or absence of multi-model consensus is an actionable indicator of extraction quality.

## 3.9 Methodological Concordance vs. Extraction Accuracy: The `info`-Column Analysis

The central interpretive challenge in evaluating automated extraction against a complex meta-analysis reference standard is that measured discordances can arise from two fundamentally different sources: (1) the AI misread a number from the paper (a *reading error*), or (2) the AI and the original meta-analyst both read numbers correctly but from different sub-conditions of a factorial design (a *methodological concordance failure*). These produce identical signatures in a standard MAE calculation but have opposite implications for system quality: reading errors are genuine pipeline failures, while methodological concordance failures reflect legitimate but divergent analytical choices, neither of which is wrong.

The Loladze 2014 validation dataset provides a uniquely powerful instrument for decomposing these two sources because every observation in the reference-standard CSV contains an `info` field that explicitly documents the methodological sub-selections Loladze made when computing effect sizes. This field transforms what appears to be a 46-paper black box into a paper-by-paper record of analytical choices, enabling precise attribution of every discordant pair to reading error or methodological sub-selection.

### What the `info` field contains

For straightforward papers, the `info` field is empty or contains only a citation number. For complex factorial papers, it contains notations such as:

- `"GI,SB,TE locations, O3"`: CO2 effect computed *within* the elevated ozone treatment arm at three specific sites
- `"N100"`: CO2 effect extracted from the ambient-N treatment arm only
- `"2000"`: CO2 effect extracted from a single year (2000) of a multi-year experiment
- `"NC-R"` or `"NC-S"`: CO2 effect extracted from a single cultivar (NC-R = non-caffeinated resistant; NC-S = sensitive)
- `"Duke"` or `"ORNL"`: CO2 effect from a single FACE site in a multi-site study

Each of these notations indicates that Loladze made a specific sub-selection that our pipeline, which defaults to computing main CO2 effects averaged or pooled across other design factors, did not replicate. The resulting effect size difference is an analytical choice difference, not a reading error.

### Systematic evidence from the 46-paper analysis

Systematic review of the `info` fields across 46 matched Loladze papers revealed five categories of methodological sub-selection that collectively explain the majority of the Loladze MAE:

**1. CO2 × co-treatment factorial design (most impactful category).** Loladze frequently extracted the CO2 effect separately within each level of a co-treatment (ozone, nitrogen, water stress, potassium), rather than computing the CO2 main effect averaged across co-treatment levels. For papers where CO2 × co-treatment interactions are biologically real, these choices produce substantially different lnRR values. Fangmeier et al. (2002, 32 matched GT pairs, MAE = 8.0%) provides the clearest illustration and is examined in detail below.

**2. Multi-year study (year selection).** For experiments reporting results across multiple growing seasons, Loladze selected a specific year (`info = "2000"`, `"1999"`, etc.) rather than pooling across years. Our pipeline extracts averages across reported years when individual year data are available in a table, or extracts the final harvest summary. For elements whose CO2 effect changes direction across seasons, year selection can determine the sign of the effect.

**3. Multi-cultivar study (cultivar selection).** For experiments crossing CO2 with multiple cultivars or genotypes, Loladze sometimes selected a specific cultivar (`info = "NC-R"`) rather than averaging. Our pipeline averages across cultivars when means are available by cultivar. For papers where CO2 × cultivar interactions produce divergent responses, cultivar averaging attenuates effects that Loladze captured at the per-cultivar level.

**4. Multi-site study (site selection).** For multi-site FACE experiments, Loladze sometimes used data from a single site (`info = "Duke"` or `"ORNL"`) representing the most relevant or best-documented site. Our pipeline uses data from whichever table presents pooled-across-site means, or averages across sites if per-site data are in the same table.

**5. Sampling date selection.** For experiments with multiple measurement dates, Loladze standardized on the final harvest date, while our pipeline extracts from whichever table presents the primary numeric data. When the final harvest data are in figures rather than tables (as in Huluka 1994), the pipeline must extract from the only available numeric table, which corresponds to an earlier sampling date.

### An illustrative example: Fangmeier et al. (2002)

**The O3-arm structural mismatch.** Fangmeier et al. (2002) report mineral concentrations in potato across a CO2 × O3 × site × year factorial design. Loladze's 32 GT rows use an `info` field specifying two analytical dimensions: site pooling (3-site vs. 5-site subsets) and O3-treatment arm (comparing 680 µl/l CO2 vs. NF ambient *within* the O3 arm, rather than the standard main CO2 effect ignoring O3). Our pipeline computed the standard CO2 main effect. For elements with small CO2×O3 interactions (K, N, P in tubers), the two approaches agree closely (|diff| < 0.05). For elements with large interactions (Fe and Mn in aboveground biomass), discrepancies are substantial; verification against Table 4 of the PDF confirms that both AI-extracted means and Loladze's values are numerically consistent with the published data, and the disagreement is about which comparison to make, not whether values were read correctly. Similar structural mismatches are documented for Huluka et al. (1994; harvest-date inaccessibility: final-harvest data presented only in figure bar charts, pipeline extracted from the sole numeric table at an earlier sampling date) and Pfirrmann et al. (1996; K-stratum selection: per-stratum CO2 effects reverse direction, average masks the interaction), with full per-paper reports available at the project repository.

### Quantifying the methodological concordance component

Tabulating the per-paper sources of discordance across the 46-paper Loladze analysis reveals that a large fraction of the MAE originates from methodological sub-selection differences rather than reading errors. For the 12 highest-error papers:

| Category | Papers (n) | Example | Is AI reading wrong? |
|---|:---:|---|:---:|
| O3/co-treatment arm selection | 4 | Fangmeier 2002, Heagle 1993 | No |
| Harvest/sampling date selection | 3 | Huluka 1994, Mjwara 1996 | No (date inaccessible) |
| Nutrient-stratum selection | 2 | Pfirrmann 1996, Huluka 1994 | No |
| Site/cultivar selection | 2 | Polley 2011, Natali 2009 | No |
| Tissue-type or table selection | 2 | Fangmeier 2002, Pfirrmann 1996 | Partially |
| Genuine reading error | 2 | Scanned-PDF papers | Yes |

After reviewing each paper's `info` field via the per-paper diagnostic agent (Section 2.5) to identify and exclude sub-selection mismatches, a well-aligned subset of 374 observations (34 papers) was identified; this subset achieved r = 0.876 and MAE = 4.3%, compared to r = 0.669 and MAE = 7.9% for the full dataset. Roughly half of the reported error is attributable to methodological concordance failures rather than extraction mistakes. The headline Loladze metrics should therefore be understood as conservative lower bounds on extraction accuracy: they simultaneously penalize the pipeline for reading errors *and* for not making the same sub-selection choices as Loladze, choices that were never specified in any extraction protocol visible to the pipeline.

**Treatment/control swap analysis.** No systematic swaps were detected across the dataset. One individual case was flagged: the Sulfur 2009 main-effect row in Fernando et al. (2012a) received a `LIKELY T/C SWAP` warning from the post-processing gate (our_effect = +4.3% vs. Loladze GT = −9.5%), consistent with a transposition of the TOS2 sub-row values before averaging. The gate operated as designed — it flagged the anomaly — but did not auto-correct (tc_swaps_corrected = 0), which was intentional: auto-correction was tested and found harmful overall (r declined from 0.509 to 0.209 when applied globally, because elements like Fe and Mn legitimately increase under elevated CO2). This represents a manual error rate of approximately 1 in 635 observations (0.2%).

## 3.10 Formal Agreement Statistics

### 3.10.1 Equivalence Testing

Two one-sided tests (TOST) confirmed formal statistical equivalence between pipeline and reference-standard extraction at ±2 pp for both the Loladze (p < 0.001; 90% CI: -1.07 to 0.97 pp; Figure 4) and Hui datasets (p < 0.001; 90% CI: -0.94 to 0.14 pp). The 90% CIs fell entirely within equivalence bounds for both datasets. Li achieved equivalence at ±3 pp (p = 0.042) despite its heterogeneous data, reflecting the near-zero mean bias (0.06 pp). All three datasets achieved equivalence at ±5 pp.

### 3.10.2 Bland-Altman Agreement

Bland-Altman analysis (Figure 5) showed negligible systematic bias for all three datasets. Loladze: mean difference = -0.05 pp (95% CI: -1.26 to 1.16 pp), with 95% limits of agreement from -30.6 to 30.5 pp; no proportional bias (r = -0.035, p = 0.38). Hui: mean difference = -0.40 pp (95% CI: -1.04 to 0.25 pp), with limits of agreement from -11.7 to 10.9 pp; no proportional bias (r = 0.024, p = 0.68). Li: mean difference = +0.06 pp, with limits of agreement from -42.1 to 42.3 pp, reflecting heterogeneous yield units. The wide limits of agreement reflect observation-level variability; aggregate-level accuracy is substantially better. Note that TOST ±2 pp equivalence bounds (Section 3.10.1) and Bland-Altman ±30 pp limits are not contradictory: TOST tests whether the *mean* difference is negligibly small (aggregate bias), while Bland-Altman limits describe the 95% range of individual observation differences (observation-level scatter). A pipeline can achieve zero mean bias while individual observations scatter widely, which is exactly what these datasets show.

### 3.10.3 Intraclass Correlation

ICC was **moderate** at the observation level (ICC(3,1) = 0.669, 95% CI: 0.623--0.710; Koo & Li, 2016 classify 0.50–0.75 as moderate) and excellent at the paper level (ICC = 0.838). The discrepancy between observation- and paper-level ICC reflects the Granularity Barrier: observation-level agreement is diluted by sub-selection mismatches, while paper-level aggregation averages out these within-paper disagreements. The paper-level ICC is consistent with published human inter-rater reliability values reported in systematic review data extraction literature (Mathes et al., 2017; Schmidt et al., 2025).

### 3.10.4 Bootstrap Confidence Intervals

**Table 2. Bootstrap CIs for key validation metrics (10,000 percentile resamples; paper as resampling unit to account for within-paper observation clustering; Hui: 18 papers, Loladze: 46 papers, Li full: 26 papers, Li clean: 16 papers).**

| Metric | Loladze (n=635) | Hui (n=310) | Li full (n=163) | Li clean† (n=100) |
|--------|:---------------:|:-----------:|:---------------:|:-----------------:|
| Pearson r | 0.669 [0.545, 0.834] | 0.993 [0.974, 0.998] | 0.453 [0.183, 0.706] | **0.996 [0.985, 1.000]** |
| MAE | 7.9% [7.0, 9.1] | 1.7% [0.4, 4.3] | 11.6% [6.2, 17.9] | **0.44 pp** |
| Direction agreement | 84.5% [81.4, 87.1] | 99.4% [97.5, 100.0] | 86.2% [71.4, 95.1] | **97.9%** |
| Mean diff (ext−GT)‡ | −0.05 pp [−1.26, 1.16] | −0.40 pp [−1.04, 0.25] | +0.06 pp [−3.27, 3.39] | +0.14 pp [−0.13, 0.40] |
| Cohen's d | −0.003 | −0.069 | 0.003 | 0.103 (p=0.308) |

† Li clean (Structurally Concordant Subset): 16 papers with verified same-level GT comparisons; excludes 12 papers with PDF/consensus failures, GT attribution/outcome errors, or aggregation mismatches (see Supplementary Table S1). Statistics computed on `validation_matches_improved.csv` using scale-invariant effect-% matching.

‡ Mean diff (ext−GT) reported as Bland-Altman signed mean difference (see Section 3.10.2); 95% CI from within-sample t-distribution of paired differences. Bootstrap CIs for this metric are unreliable when the mean difference is near zero (the CI of |mean diff| collapses below the point estimate). The signed BA CIs are the appropriate summary.

## 3.11 Sensitivity Analyses

### 3.11.1 Leave-One-Paper-Out

LOPO analysis showed the full MAE (7.95%) was stable: LOPO range was 6.8--8.3%. The most influential paper (Natali 2009, MAE = 19.1%) improved aggregate MAE by 1.16 pp when removed.

### 3.11.2 Leave-One-Element-Out

No single element drove results. Removing trace metals improved MAE by 0.20--0.37 pp; removing major elements worsened it by 0.22--0.43 pp.

### 3.11.3 Matching Tolerance Sensitivity

For Hui, tightening tolerance from 0.15 to 0.10 had minimal impact on accuracy (r remained >0.99), confirming that the high-quality matches are robust to matching parameters. For Li, tolerance sensitivity was higher: at 0.10 (n = 112), r = 0.889; at 0.30 (n = 163), r = 0.453. Aggregate effect differences remained stable across all thresholds (<1.1 pp for Li).

## 3.12 Consensus vs. Single-Model Comparison

To isolate the consensus mechanism's contribution, we compared each model's solo extraction on a fixed scope of 322 observations matchable by all sources:

| Method | MAE (%) | Pearson r | Direction (%) |
|--------|---------|-----------|---------------|
| Kimi solo | 4.10 | 0.903 | 88.6 |
| Consensus | 4.54 | 0.886 | 89.2 |
| Gemini solo | 5.53 | 0.843 | 85.1 |
| Claude solo | 6.29 | 0.742 | 85.4 |

On a fixed observation set, consensus does not improve per-observation accuracy over the best single model (Kimi). The consensus mechanism's value lies in identifying reliable observations rather than improving per-observation accuracy on a fixed set. When two models agree, the observation is likely correct; when they disagree, it warrants review. No single model dominates across all elements (Kimi best for 13/20, Gemini 5/20, Claude 2/20), so the multi-model approach provides robustness against model-specific failures.

On the original 46-paper extraction, the consensus pipeline matched 560 observations to reference standard versus Kimi's 486 (15% increase), representing modest coverage gains from complementary model outputs.

## 3.13 Cost

The median per-paper API cost was approximately $0.24 (mean: $0.37; range: $0.12–$3.50; Claude: ~$0.28, Kimi: ~$0.04, Gemini: ~$0.05). The mean exceeds the median because the distribution is right-skewed: simple text papers cost ~$0.12–$0.18, while complex factorial designs with 200+ extracted observations (e.g., Chen et al. 2021: 299 extracted rows, $3.50) dominate the tail. This variability reflects genuine differences in paper complexity rather than tunable parameters. Processing 46 papers cost approximately $17 over 6 hours. This compares to an estimated 184 hours of manual extraction at $30/hour ($5,520). In a triage deployment where human reviewers check only flagged observations (20--30% of the total, varying by task structure and dataset), the human review time would be approximately 45--55 hours rather than 184, a 65--75% time reduction.

## 3.14 Prospective Application: Silicon Effects on Wheat (40 Papers)

**[RESULTS PENDING — to be filled after extraction runs complete]**

**Overview.** The triple-vision pipeline processed 40 papers on silicon effects on wheat yield. [X] papers yielded at least one HIGH- or MEDIUM-confidence observation; [Y] papers returned zero usable observations (non-English text in scanned PDFs where vision failed, or papers where silicon was a co-treatment in a complex factorial design that did not include a zero-silicon control arm). Total observations extracted: [N], of which [H] were HIGH confidence ([H/N %]), [M] MEDIUM ([M/N %]), and [L] LOW ([L/N %]). The overall consensus rate (fraction HIGH confidence) was [X%], higher than in the Loladze validation (64% consensus-dominant papers), consistent with the simpler two-treatment structure (silicon vs. no silicon) typical in this literature compared to the multi-element factorial designs of the CO2/mineral dataset.

**Test-retest reproducibility.** Across two independent extraction runs, [P%] of run-1 observations were reproduced in run-2 (reproducibility rate). For matched pairs (n = [M_pairs]), the test-retest ICC(2,1) for treatment means was [ICC] (95% CI: [lo, hi]) and Pearson r = [r]. Under Koo and Li (2016) thresholds, this represents [excellent/good/moderate] intra-pipeline reliability. Effect-size stability (r between run-1 and run-2 log response ratios) was [r_lnrr], indicating that the direction and magnitude of silicon effects were [stable/unstable] across independent extractions. Confidence tier agreement (same HIGH/MEDIUM/LOW classification in both runs) was [conf_pct%], confirming that the consensus signal is reproducible rather than a function of random model variation.

**LLM audit.** Claude Sonnet 4.6 audited [A] HIGH- and MEDIUM-confidence observations across [P_audit] papers. Of these, [V%] were VERIFIED with a specific source citation (table and row), [Pt%] were PARTIAL (one value confirmed or scale ambiguous), and [F%] were FLAGGED as unlocatable in the source PDF. Flagged observations were concentrated in [X] papers, all characterized by complex multi-panel figures or tables where the silicon treatment was embedded within a multi-factor design; consistent with the Granularity Barrier identified in the Loladze validation. The [V%] verification rate provides an upper-bound estimate of the proportion of observations that can be used directly without human review.

**Practical triage.** Combining HIGH confidence (X observations) and VERIFIED audit status ([V_high] of the HIGH-confidence observations), the recommended no-review pool contains [auto_n] observations. An additional [review_n] observations (MEDIUM confidence or PARTIAL audit) are recommended for targeted human review, estimated at 2 minutes each, for a total review burden of approximately [review_hours] hours. The pipeline thus produces a near-complete candidate dataset for a silicon/wheat meta-analysis in a single extraction pass, without requiring knowledge of the results in advance.

---

# 4. Discussion

## 4.1 Where the System Fits in the LLM Extraction Landscape

**Table 3. Comparison of LLM-based data extraction systems for evidence synthesis.**

| System | Domain | LLM(s) | N papers | Multi-model? | Primary accuracy | Quant. accuracy | Equivalence test |
|--------|--------|--------|----------|:------------:|----------------|-----------------|:----------------:|
| Buscemi et al. 2006 | Clinical (human) | -- | 44 | Yes (dual human) | 82.3% | Not separated | No |
| Gartlehner et al. 2024 | Clinical | Claude 2 | 10 | No | 96.3% categ. + simple numerical | Not separated | No |
| Gougherty & Clipp 2024 | Ecology | text-bison | 100 | No | >90% categorical | 23.8% | No |
| Jensen et al. 2025 | Clinical | GPT-4o | 11 | No | 92.4% acc. | Not separated | No |
| Khan et al. 2025 | Clinical | GPT-4t + Claude-3-Opus | 22 | Yes (2) | 94% concordant accuracy | 0.25% hallucination (concordant) | No |
| Li et al. 2025 | Clinical | GPT-4o-mini, Gemini-2.0-Flash, Grok-3 | 58 | Yes (3, combined) | Prec. 0.81-0.97 | Recall 0.21-0.81 (stats); consensus +14.8% recall | No |
| Kataoka et al. 2026 | Clinical (insomnia) | GPT-4o; o3 | 290 | No | 72.3% (GPT-4o), 75.3% (o3); numeric worse than string | Numeric extraction "still inadequate" (their words) | No |
| Poser et al. 2026 | Clinical | Claude 3.7 + Gemini + o3 | 30 | Yes (3) | 6.7% raw error (1.48% true-error¹) | N/A (categorical) | No |
| Jansen et al. 2025 | Clinical | Multiple | 22 | No | 91%+ categorical | 26-36% effect sizes | No |
| **This study** | **Plant sci.** | **Sonnet 4 + Kimi + Gemini** | **112** | **Yes (3)** | **r = 0.45-0.99** | **MAE 1.7-11.6%** | **TOST p<0.001** |

¹ Poser et al. distinguish "true errors" (content errors, 1.48%) from total errors including formatting issues (6.7%).

Our system differs from prior work in four key respects. First, we target quantitative extraction of continuous means and effect sizes (Li et al.'s Tier 3) rather than categorical study characteristics, where prior systems achieved 90--96% accuracy. Even specialized numerical extraction systems acknowledge the difficulty: Kataoka et al. (2026), testing o3 (OpenAI's reasoning model) on insomnia RCTs, concluded that "numeric variable extraction performed poorly" and "the performance for numeric DE was still inadequate." Their best system (o3) reached 75.3% accuracy on numeric variables; our Hui result of r = 0.993 exceeds this, though on a different task structure. Second, we use inter-model agreement as a quality *predictor* rather than solely an accuracy booster, enabling confidence-stratified output rather than aggregate accuracy reports. Third, we are the only system validated on plant science data. Fourth, we apply formal equivalence testing (TOST, ICC, Bland-Altman), providing a statistical framework for comparing pipeline output to human extraction rather than just reporting accuracy percentages.

The accuracy picture is more nuanced than prior work suggests. Scott et al.'s (2025) systematic review found data extraction error rates of 4--31% (median 14%) across existing systems, with GenAI performing well on "easier" data such as publication years or countries, "but for more complex data, such as outcome data or intervention descriptions, GenAI tended to perform less effectively." Our MAE of 1.73% (Hui) and 7.9% (Loladze) falls within this range, with the Hui result substantially better than the field median, reflecting the advantage of single-element structured extraction. As Section 3.9 documents, the Loladze MAE of 7.9% is itself a conservative estimate: separating methodological concordance failures from reading errors reduces the effective extraction MAE to approximately 4.3% for the aligned subset. Jansen et al. (2025) found only 26--36% accuracy for effect-size variables and ~17% for standard deviations across 22 meta-analyses, results that contextualise our Loladze MAE of 7.9% (a different metric but directionally consistent). Peng et al. (2025) found 65.7--71.5% accuracy for means and SDs from sleep medicine RCTs using Claude 3.5; Yun et al. (2024) found 48.7% exact match for GPT-4 on continuous RCT outcomes; Gougherty & Clipp (2024) reported 23.8% for quantitative ecological data. Our Hui result (r = 0.993, MAE = 1.73%) exceeds these benchmarks substantially, reflecting the advantage of single-element structured extraction. Our Loladze and Li results are more representative of multi-element complex extraction and align with the 69--72% range when viewed at the ±10% threshold (74% and 66% within 10%, respectively).

## 4.2 The Concordance Signal: Khan's Principle Extended

Khan et al. (2025) established the core insight motivating our design: when two independent LLMs give concordant responses, the hallucination rate drops to 0.25%; when responses are discordant, the hallucination rate rises to 26--41%. This nearly 100-fold difference means that concordance status is a better predictor of reliability than any model-specific accuracy score. Khan applied this principle to categorical binary extraction (does a study meet inclusion criterion X?), demonstrating 87% concordance rate and 94% accuracy on concordant items.

We extend this principle in two directions. First, we apply it to *continuous numerical* extraction, where the reliability problem is more acute: wrong numbers can bias meta-analytic estimates without triggering any obvious flag, whereas a wrong categorical label is often detectable by inspection. Second, we develop the concordance signal into a full confidence-stratification system rather than a binary accept/reject filter. Our results confirm that the concordance principle extends to quantitative extraction: consensus-dominant papers achieve MAE = 4.3% versus 11.2% for vision-dependent papers (2.6× improvement), with 95% of large errors concentrated in the flagged minority.

Poser et al. (2026) independently validated three-model consensus for clinical data extraction, achieving 1.48% true-error rate for structured clinical variables, further confirming that multi-model consensus reduces errors. However, their study focused on categorical clinical fields rather than continuous numerical outcomes. Jansen et al. (2025), testing a majority-vote ensemble of 8 LLMs on 2,179 studies, found that ensemble voting improved performance over individual models and that variable type was the dominant predictor of accuracy, consistent with our finding that `has_complex_stats` predicts consensus failure more strongly than paper-level features. Their conclusion that accuracy depends "most between variable, less between systematic reviews, and least between LLMs" means that inter-model agreement functions as a variable-difficulty detector: hard variables produce disagreement, easy variables produce consensus. Our contribution is demonstrating that this signal works for *continuous* numerical extraction in *complex multi-element* agricultural data, where it predicts accuracy at the paper level with sufficient reliability to support an automated triage workflow.

Tan & D'Souza (2026) provide mechanistic insight into why single-model extraction fails systematically. Their four structural failure modes — role confusion (treatment/control swaps), binding drift (cross-row value attribution), multi-instance compression, and error amplification — each produce characteristic patterns that a single model cannot self-detect. Multi-model consensus addresses these failures by design: a treatment/control swap by one model will not be matched by another model, raising a discordance flag. Binding drift produces inconsistent numerical values across models. Only multi-instance compression and error amplification could propagate across models if both are misled by the same structural feature, a residual limitation.

**Reading Accuracy versus Relevance: a critical distinction.** Consensus validates *reading accuracy* — was the number extracted correctly from the text? It does not validate *relevance* — is this the number the meta-analyst intended to select? The 4.5% initial extraction precision (100 GT-matched observations from over 2,200 extracted candidates in the Structurally Concordant Subset Li analysis) is a relevance problem, not a consensus failure: the pipeline correctly reads all detectable numbers, then consensus confirms which readings are reliable. The downstream triage step — filtering to yield-related outcomes and matching to the meta-analyst's intended stratum — is a separate task. High-confidence observations are reliably accurate readings; they may still require analyst judgment about relevance to a specific sub-condition.

## 4.3 The Three-Barrier Model: What Each Dataset Measures

The three datasets do not form a difficulty gradient; they are three different instruments, each isolating a distinct barrier to reliable AI extraction.

**The Reading Barrier (Hui dataset).** With a single element (Zn), standardized units (mg/kg), and a uniform wheat biofortification context, the r = 0.993 and MAE = 1.73% measure one thing: can the pipeline read numbers correctly from tables? This barrier is effectively broken: near-perfect accuracy was achieved zero-shot on an unseen dataset without any domain-specific calibration beyond the JSON schema.

**The Granularity Barrier (Loladze dataset).** The Loladze dataset adds 14 elements, CO2 factorial designs, scanned PDFs, and multi-tissue tables, introducing extraction difficulty plus a second barrier: does the system make the same analytical sub-selection choices as the human meta-analyst? The `info`-column analysis (Section 3.9) shows that approximately half of the Loladze MAE reflects sub-selection disagreement — the pipeline and Loladze both read values correctly but from different experimental strata. For a researcher running this pipeline for their own meta-analysis, sub-selection rules are specified in the configuration file, eliminating this disagreement. The 4.3% aligned-observation MAE is the operational forecast; 7.9% is the lower bound when sub-selection rules are not pre-specified (replicating a different analyst's unstated choices). The Granularity Barrier is not solved by more accurate reading, but by explicit pre-specification of analytical choices.

**The Provenance Barrier (Li dataset).** The Li dataset adds a third barrier: the quality and homogeneity of the reference standard itself. When reference standard and input PDF provenance issues prevent valid comparison for 12 of 28 papers, the headline r = 0.453 measures the combination of pipeline output, GT curation heterogeneity, and validator reliability — not extraction accuracy alone. On the 16 papers where the comparison is clean, r = 0.996. The Provenance Barrier is not an AI problem; it is a validation infrastructure problem. No extraction system, human or AI, can achieve high correlations when validated against a reference standard that contains wrong PDF mappings, attribution errors, and aggregation-level mismatches.

Before deploying the pipeline on a new topic, researchers can anticipate: (A) if extraction targets a single outcome in standardized units (Hui profile), expect near-perfect reading accuracy; (B) if extraction targets complex factorial designs with pre-specified sub-selection rules (Loladze profile), expect ~4.3% MAE for aligned observations; (C) if validating against a heterogeneous external reference standard (Li profile), expect that the headline statistics will reflect reference-standard quality as much as pipeline quality — paper-level auditing is needed to separate these. In all three cases, aggregate meta-analytic effects are reproduced to within fractions of a percentage point regardless of individual-observation noise.

**The 4.5% initial precision as deliberate Information Surplus.** The low initial extraction precision (4.5%: ~100 GT-matched observations from over 2,200 extracted candidates in the Structurally Concordant Li analysis) reflects a deliberate design choice rather than a failure. In Baslam et al. (2012), the pipeline extracted 76 observations capturing every factorial interaction, while the human reference standard used a 38-observation main-effect subset. This exhaustive extraction preserves the full experimental richness for secondary meta-regressions and subgroup analyses that may not have been planned during the original meta-analysis. The consensus mechanism then functions as a precision filter: from the information-surplus candidate pool, inter-model agreement selects the observations most likely to be reliable readings. Researchers who need higher initial precision for a tightly-defined question can specify explicit stratum filters in the configuration JSON, trading recall for precision at the configuration level rather than the model level.

## 4.4 Li 2022 Paper-Level Audit: What the Validation Comparison Measures

We conducted a systematic audit of all 28 Li 2022 papers to understand the composition of the r = 0.453 headline. Twelve of 28 papers had structural reasons preventing a valid same-level comparison between pipeline output and reference standard. These were comparison failures rather than extraction failures: the measuring instrument (the GT database plus the matching algorithm) could not form a valid comparison for those 12 papers. The remaining 16 papers all matched cleanly, achieving r = 0.996, MAE = 0.44 pp, 100% capture rate.

The 12 excluded papers fall into identifiable categories (detailed in Supplementary Table S1 and Supplementary Table S5): 2 papers had input PDF or consensus failures — Abdel-Mawgoud (2010), whose yield data appear exclusively in embedded bar charts precluding text extraction, and Alabdulla (2019), where Kimi timed out leaving only single-model Claude output with no consensus formed; 4 had GT attribution errors — 3 where the GT database incorrectly assigned a wrong crop species or metric to the paper (Mondal, Pohl, Glosek-Sobieraj), and 1 (Godlewska 2016) where the pipeline correctly extracted 60 consensus observations of microelement content (Zn, Cu, Fe, Mn) but the Li GT row expected yield data, a GT outcome-category mismatch rather than an AI extraction failure; 3 had aggregation-level mismatches (the GT stored per-year observations, the pipeline extracted multi-year averages, a legitimate analytical choice difference rather than a reading error); 2 had GT values sourced from pre-publication data with systematic inflation; and 1 had a product-selection omission (the pipeline extracted the title product arm; the GT included an additional arm). One further paper (Rahman 2018) is the sole genuine pipeline limitation: yield data were published exclusively in bar chart figures, and the pipeline captured only 2 of 4 dose arms from visual extraction.

Two additional validation infrastructure issues were identified and resolved. The consensus engine silently discarded 20 correctly extracted observations from Popescu 2018 and Wilczewski 2018: in one case because two models used different unit systems for the same values (kg/vine vs. kg/ha); in the other because bracketed unit notation in element labels caused exact string matching to fail. A post-hoc repair script recovered 13 of these observations to reference-standard matches. Separately, the scale-sensitive validator had misscored correct extractions for Lola-Luz 2014 and Soppelsa 2018 (absolute-scale mismatches between GT and extracted means, despite correct effect sizes); switching to effect-percentage-based matching corrected these. The methodological lesson for future validation studies is that r, MAE, and ICC are composite metrics measuring both pipeline accuracy and validation infrastructure reliability simultaneously, and that paper-level auditing is necessary to separate these when the reference standard contains heterogeneous curation quality.

## 4.5 Benchmark Bias and the Human Comparison Problem

All AI extraction benchmarking, including ours, faces what Gartlehner et al. (2025) have termed "benchmark bias": the tendency to evaluate AI systems against human-extracted reference standards that themselves contain errors. Research on human extraction reliability indicates that up to 63% of study reports contain at least one extraction error even when extracted by trained researchers, a baseline rarely accounted for when setting accuracy thresholds for AI systems (Gartlehner et al., 2025; Mathes et al., 2017). Gartlehner et al. (2025) illustrate the problem concretely: in their proof-of-concept study, Claude 2 identified 21 minor errors in the human reference standard that would otherwise have gone undetected; on inspection, these proved to be corrections.

This has direct implications for interpreting our results. Our Loladze reference standard was extracted by a single author without a reported dual-extraction protocol. Our measured MAE of 7.9% therefore combines pipeline error, reference-standard error, and methodological concordance failures in unknown proportions. The `info`-column analysis (Section 3.9) resolves the concordance component, and the 6 papers where our pipeline achieved MAE < 0.1% suggest the pipeline can be more precise than the reference standard on well-structured papers. If the reference standard contains even 5% error, and if the concordance component accounts for approximately half the remaining MAE (as Section 3.9 estimates), the effective pipeline extraction error may be as low as 2--3%, substantially below the headline 7.9%.

More broadly, the dual-independent-extraction-followed-by-consensus protocol described by Buscemi et al. (2006) as the gold standard is expensive enough that most meta-analyses never implement it. Our paper-level ICC of 0.838 is consistent with human inter-rater reliability values reported in the data extraction automation literature (Mathes et al., 2017; Schmidt et al., 2025), suggesting that the pipeline performs comparably to a human second extractor without the labor cost. Jensen et al. (2025) found that ChatGPT-4o as a second rater had a 5.2% false data rate versus 17.7% for human single extractors, consistent with the possibility that AI extraction is already less error-prone than single-human extraction for some data types.

Benchmark bias implies that future evaluation studies should, where possible, use multiple independent human extractors to establish the reference standard, document methodological sub-selection choices (as the Loladze `info` field inadvertently does), and separate validation artifacts from genuine extraction errors through paper-level auditing. Our methodology for doing this, combining systematic per-paper diagnostic reports with `info`-field analysis, provides a template for future validation studies of complex-design data extraction systems.

## 4.6 Engaging with "Not Yet Ready": A More Nuanced Assessment

Lieberum et al. (2025) concluded from their scoping review that LLMs are "not yet ready for use" in systematic review data extraction, noting that only 11% of LLM-SR studies even address data extraction and that quantitative accuracy remains insufficient for complex tasks. Scott et al. (2025) reached the same conclusion: "The current evidence does not support GenAI use in evidence synthesis without human involvement or oversight. However, for most tasks other than searching, GenAI may have a role in assisting humans with evidence synthesis." Cao et al. (2025) similarly found that fully automated systematic reviews remain out of reach. We agree with this conclusion for fully automated deployment but argue for a more nuanced position: the pipeline is ready for triage deployment, with human oversight concentrated where it is needed.

The "not yet ready" conclusion implies a binary: either the system works well enough to replace humans, or it does not. Our results suggest a different question: whether AI can triage extraction work intelligently, focusing human effort on the observations where it is needed. On this criterion, our results are more encouraging. The pipeline achieves MAE = 4.3% on consensus-dominant observations and correctly identifies 95% of large errors as low-confidence. Human reviewers who check only the flagged observations (20--30% of total, depending on task structure), working at reduced effort compared to de novo extraction, can achieve the accuracy of full dual extraction at approximately 55--75% lower cost.

Lieberum et al.'s concern is particularly valid for domains where accuracy is critical at the individual-observation level (clinical dosing, safety data) and where errors could directly harm downstream users. For agricultural meta-analysis, where the outputs are population-level estimates of agronomic effects rather than individual patient treatments, the tolerance for observation-level variability is higher, provided that errors are random rather than systematic. Our demonstration that Cohen's d ≈ 0 across all three datasets (|d| < 0.07) and that errors cancel in the aggregate addresses this concern directly.

The methodological concordance finding (Section 3.9) adds an important nuance to the "not yet ready" assessment: a portion of what current validation studies report as AI extraction error is actually methodological sub-selection disagreement between the AI and the human meta-analyst, of a kind that a human second extractor without explicit sub-selection instructions would also show. The true extraction error rate, stripped of concordance failures, is lower than reported MAE values suggest. Current MAE benchmarks therefore understate AI capability.

We propose a position between "not yet ready" and "ready for full automation": ready for supervised deployment as a first extractor, analogous to how pilot extraction is currently used in manual meta-analysis. The pipeline handles 70--80% of observations at high confidence; human reviewers handle the rest. This matches how most large meta-analysis teams already work, with a senior researcher reviewing junior extractors' output, but substitutes AI for the junior extractor role at a fraction of the cost.

## 4.7 The Triage Workflow in Practice

Based on our results, we propose a triage workflow for AI-assisted meta-analysis:

1. **Auto-validated observations** (high confidence, ~70--80% of pipeline output): Two or more models agree. These observations achieve MAE ≈ 3--5% and can be used directly with minimal human review. For scoping analyses assessing meta-analysis feasibility, these observations alone may suffice. For publication-quality analyses, we recommend a **5% random spot-check** of high-confidence observations to verify absence of systematic model bias (e.g., a shared misinterpretation of a non-standard unit or table layout that both models agree on incorrectly).

2. **Flagged observations** (medium/low confidence, ~20--30% of output): Single-model or vision-only extraction. These are retained in the dataset but tagged for human verification. The pipeline provides the extracted value as a starting point, reducing reviewer effort compared to extraction from scratch. Each flagged observation requires approximately 2 minutes of expert review compared to 10 minutes of de novo extraction.

3. **Rejected papers** (zero consensus, ~3% after tiebreaker): No model could extract usable data. These require full manual extraction.

4. **Concordance review** (for complex factorial papers): The pipeline's extracted values and the meta-analyst's sub-selection rules should be compared. For papers where the `info`-field-equivalent information (co-treatment arm, date, cultivar) is not pre-specified in the configuration, the pipeline defaults to main effects and averages, which may differ from the intended sub-selection. This review step replaces the methodological concordance failures identified in Section 3.9 with a configuration-time specification of intent.

This workflow offers substantial time savings even without full automation. The flagged fraction varies by task: single-element extraction (Hui-type) flagged approximately 31% of observations, while complex multi-element extraction (Loladze-type) achieved consensus on ~94% of observations, flagging only ~6%. Assuming 20--30% human review at ~2 minutes per flagged observation versus ~10 minutes for de novo extraction, the total human time for a 46-paper meta-analysis drops from ~184 hours to approximately 50--80 hours, a 55--75% reduction. The pipeline also provides a complementary extraction for quality assurance: paper-level ICC of 0.838 is consistent with published human inter-rater reliability values (Mathes et al., 2017; Schmidt et al., 2025). Methodological independence between pipeline models is not claimed, as all three models received identical extraction prompts; the ICC comparison is therefore to human inter-rater reliability as a calibration benchmark, not as evidence of independent replication.

For rapid scoping reviews assessing the feasibility of a new meta-analysis, the high-confidence observations alone may be sufficient. For publication-quality meta-analyses, the triage workflow ensures that every observation is either consensus-validated or human-reviewed. Peng et al. (2025) reached the same practical conclusion in sleep medicine: "systematic review authors could utilize AI tools as second reviewers for data extraction to achieve accuracy comparable to human reviewers, yet with greater efficiency and reduced labor." Our results extend this finding to plant science and quantify what "comparable to human" means: paper-level ICC = 0.838, consistent with human inter-rater reliability values for data extraction tasks (Mathes et al., 2017; Schmidt et al., 2025).

## 4.8 Limitations

1. **Development vs. holdout validation.** The Loladze dataset was used during pipeline development; accuracy on this dataset may be optimistic. The Hui zero-shot holdout validation (r = 0.993) provides a more honest estimate for straightforward extraction tasks, while the Li dataset (r = 0.453 overall; r = 0.996 on 16 correctly paired papers) represents performance on a heterogeneous dataset with both reference-standard curation issues and two identified software bugs in the validation pipeline (Section 4.4). The consensus-layer normalization failures (Popescu 2018, Wilczewski 2018: unit and label format divergence between models silently discarding 20 correct observations) and the automated validator matching failures (Lola-Luz 2014, Soppelsa 2018: absolute-scale mismatch misscoring correct effect-size extractions) are fixable and represent upper-bound opportunities for improvement.

2. **Wide observation-level limits of agreement.** Bland-Altman limits span ±30 pp for Loladze and ±42 pp for Li, meaning individual observations may have large errors even though the aggregate is unbiased. The pipeline should be used for aggregate pooling, not single-observation precision.

3. **Vision extraction quality.** Vision-dependent papers achieved MAE = 11.2%, substantially worse than consensus-dominant papers (MAE = 4.3%). The pipeline's confidence flagging mitigates this risk but does not eliminate it. Papers with scanned tables or figure-only data remain challenging.

4. **Element capture rate (83%).** Approximately 17% of reference-standard observations were not matched, potentially introducing selection bias.

5. **Variance extraction (67% capture)** lags behind means extraction (>98%), reflecting inconsistent variance reporting in agricultural journals, a known problem in the field (Nakagawa et al., 2023).

6. **Proportional bias in heterogeneous data.** The Li dataset showed significant proportional bias (r = 0.355, p < 0.001): extraction errors grew with effect magnitude. This likely reflects unit conversion ambiguity for large yield responses rather than systematic extraction failure. Unit-normalization (scale-factor matching) accounts for a portion of Li clean-subset matches; a 100× scale discrepancy is consistent with legitimate unit conversion (mg/100g vs. mg/g) but could in principle indicate extraction of a differently-scaled variable. Per-paper audit of all 16 clean-subset papers found no instances of wrong-variable extraction; nevertheless, large scale-factor matches should be flagged for human inspection in production use.

7. **Validation scope.** All three datasets are from plant science. Generalization to clinical trials or other domains is not tested, though the pipeline is domain-agnostic by design. Extension to animal ecology (e.g., aquatic species response studies) is planned; the configuration-driven design transfers without code changes, but domain-specific outcome variable definitions and tissue taxonomies must be specified.

8. **Model versioning and reproducibility.** Results were obtained with specific model versions (Claude Sonnet 4, Kimi K2.5, Gemini 3 Flash). LLM providers update models without notice. Cross-run stability ICC of 0.9996 applies within a version; across versions, drift should be expected. Schmidt et al.'s (2025) living review update specifically identified reproducibility as an emerging concern with LLM-based extraction: "LLMs showed a trend of decreasing quality of results reporting, especially quantitative results such as recall and lower reproducibility of results."

9. **Benchmark bias.** As discussed in Section 4.5, reference-standard extraction errors and methodological sub-selection choices both confound our accuracy estimates. True pipeline reading error may be lower than reported MAE.

10. **Prompt sensitivity.** Extraction quality depends on prompt templates, which were not systematically ablated. Full prompts are provided in supplementary materials.

11. **Methodological concordance decomposition limitations.** The `info`-column decomposition in Section 3.9 was conducted for Loladze but relies on the unusual feature that this reference standard's `info` field documents sub-selection choices. Most meta-analysis databases do not include such documentation, making concordance decomposition difficult or impossible without re-reading all papers. The approach is not easily generalizable without comparable documentation in the reference standard.

---

# 5. Conclusion

We developed and validated a multi-model consensus pipeline for quantitative data extraction in plant science meta-analysis. Across three published reference datasets representing 112 papers and 2,676 extracted observations, the system achieved r = 0.993 on zero-shot holdout validation (MAE = 1.73%), reproduced aggregate meta-analytic effects to within 0.40 pp with no systematic bias, and correctly identified its own unreliable extractions through inter-model agreement. The system uses dual-model strategic heterogeneity (Claude Sonnet 4 + Kimi K2.5) with a conditional tiebreaker (Gemini 3 Flash): concordant observations are output with high confidence; discordant ones are flagged for human review.

The results characterize three barriers to reliable AI extraction. The **Reading Barrier** is effectively broken: zero-shot holdout validation on the Hui 2023 zinc biofortification dataset achieved r = 0.993, MAE = 1.73%, 99% direction agreement. The **Granularity Barrier** — selecting the correct analytical stratum from factorial designs — is addressed by pre-specified configuration: separating sub-selection concordance from reading errors reduces the Loladze MAE from 7.9% to 4.3%. The **Provenance Barrier** — reference-standard and input-file heterogeneity — explains the Li 2022 headline r = 0.453; on the Structurally Concordant Subset (16 papers with verified same-level comparisons), r = 0.996, MAE = 0.44 pp. To our knowledge, these results represent the strongest reported performance for continuous numerical effect-size extraction in plant science, and compare favorably with the best available benchmarks: zero-shot holdout r = 0.993 exceeds the 72--75% numeric accuracy recently achieved by GPT-4o and o3 on clinical insomnia data (Kataoka et al., 2026), though differences in task structure preclude direct comparison.

On reliability prediction: consensus-dominant papers achieved MAE = 4.3% versus 11.2% for vision-dependent papers. High-confidence observations had significantly lower error than flagged observations (p < 0.001), and 95% of large errors concentrated in the flagged minority. The confidence score is a genuine quality signal, available at extraction time, that enables triage without ground truth.

On systematic bias: aggregate meta-analytic effects were reproduced to within 0.05--0.40 pp across all three datasets, with |Cohen's d| < 0.07 and formal TOST equivalence at ±2 pp (Loladze and Hui, p < 0.001). A pipeline that systematically misread values could not reproduce pooled effects with this precision.

Reported MAE simultaneously measures reading accuracy, sub-selection concordance, and reference-standard quality. For the Loladze dataset, separating reading errors from concordance failures reduces the effective extraction MAE from 7.9% to 4.3%. For the Li dataset, auditing the 28-paper comparison reveals that the r = 0.453 headline measures a combination of pipeline output and GT curation heterogeneity; on the Structurally Concordant Subset, r = 0.996. Statistical metrics alone cannot reveal this; paper-level auditing can.

The Loladze 2014 meta-analysis synthesizes 1,481 mineral concentration measurements across 25 elements and 130 species; extracting it manually required months. Our pipeline processed the same 46 papers in 6 hours at $17 total cost. With 70--95% of observations auto-validated (varying by task structure: 94% for complex multi-element extraction, ~69% for single-element) and the remainder flagged for review at roughly 2 minutes each, the human review burden for a 46-paper meta-analysis drops from ~184 hours to approximately 50--80. The practical contribution is concentrating human judgment where it matters. For plant science, a domain with decades of CO2, biofortification, biostimulant, and climate adaptation research still unprocessed, that difference compounds across the full literature.

---

# References

*Note: Papers cited as illustrative examples from the validation datasets (Baslam et al. 2012, Fangmeier et al. 2002, Huluka et al. 1994, Pfirrmann et al. 1996, Niu et al. 2013, Chen et al. 2021, and similar source papers) are components of the Loladze (2014) and Li et al. (2022) reference datasets; their complete bibliographic records are available in those publications' reference lists and in the project repository supplementary files.*

- Buscemi, N., Hartling, L., Vandermeer, B., Tjosvold, L., & Klassen, T. P. (2006). Single data extraction generated more errors than double data extraction in systematic reviews. *Journal of Clinical Epidemiology*, 59(7), 697-703.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman and Hall/CRC. ISBN 978-0-412-04231-7.
- Cao, C., Arora, R., Cento, P., et al. (2025). Automation of systematic reviews with large language models. *medRxiv* preprint. DOI: 10.1101/2025.06.13.25329541.
- Gartlehner, G., Kahwati, L., Hilscher, R., et al. (2024). Data extraction for evidence synthesis using a large language model: A proof-of-concept study. *Research Synthesis Methods*, 15(4), 576-589.
- Gartlehner, G., Kugley, S., Crotty, K., Viswanathan, M., et al. (2025). Artificial intelligence-assisted data extraction with a large language model: A study within reviews. *Annals of Internal Medicine*. DOI: 10.7326/ANNALS-25-00739.
- Gougherty, A. V., & Clipp, H. L. (2024). Testing the reliability of an AI-based large language model to extract ecological information from the scientific literature. *npj Biodiversity*, 3(1), 13.
- Hui, Y., Wang, J., Jiang, T., Li, S., Zhang, Y., & Liu, X. (2023). Zinc biofortification of wheat through soil, foliar, and combined applications: A meta-analysis. *Journal of Soil Science and Plant Nutrition*, 23, 5384-5397.
- Jansen, T., et al. (2025). Data extraction by generative artificial intelligence: Assessing determinants of accuracy using human-extracted data from systematic review databases. *Psychological Bulletin*, 151(10), 1280-1306. DOI: 10.1037/bul0000451.
- Jensen, M. M., Danielsen, M. B., Riis, J., et al. (2025). ChatGPT-4o can serve as the second rater for data extraction in systematic reviews. *PLoS ONE*, 20(1), e0313401.
- Kataoka, Y., et al. (2026). Automating the data extraction process for systematic reviews using GPT-4o and o3. *Research Synthesis Methods*, 17, 42-62. DOI: 10.1017/rsm.2025.10030.
- Khan, M. A., Ayub, U., Naqvi, S. A. A., et al. (2025). Collaborative large language models for automated data extraction in living systematic reviews. *Journal of the American Medical Informatics Association*, 32(4), 638-647.
- Koo, T. K., & Li, M. Y. (2016). A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *Journal of Chiropractic Medicine*, 15(2), 155-163.
- Koricheva, J., & Gurevitch, J. (2014). Uses and misuses of meta-analysis in plant ecology. *Journal of Ecology*, 102(4), 828-844.
- Li, J., Van Gerrewey, T., & Geelen, D. (2022). A meta-analysis of biostimulant yield effectiveness in field trials. *Frontiers in Plant Science*, 13, 836702.
- Li, X., Mathrani, A., & Susnjak, T. (2025). What level of automation is "good enough"? A benchmark of large language models for meta-analysis data extraction. *arXiv* preprint arXiv:2507.15152.
- Lieberum, J.-L., Toews, M., Metzendorf, M.-I., et al. (2025). Large language models for conducting systematic reviews: on the rise, but not yet ready for use: a scoping review. *Journal of Clinical Epidemiology*, 181, 111746.
- Loladze, I. (2014). Hidden shift of the ionome of plants exposed to elevated CO2 depletes minerals at the base of human nutrition. *eLife*, 3, e02245.
- Mathes, T., Klaassen-Mielke, R., & Pieper, D. (2017). Data extraction methods for systematic review (semi)automation: A living systematic review. *F1000Research*, 6, 1699.
- Nakagawa, S., et al. (2023). A robust and readily implementable method for the meta-analysis of response ratios with and without missing standard deviations. *Ecology Letters*, 26(2), 232-244.
- Peng, Y., et al. (2025). Accuracy of large language models in data extraction from randomized controlled trials in sleep medicine: A proof-of-concept study. *Sleep Medicine*, 128. DOI: 10.1016/j.sleep.2025.01455 [ScienceDirect PII S1087079225001455].
- Poser, P. L., Klimas, R., Luerweg, J., et al. (2026). Improving reliability and accuracy of structured data extraction using a consensus large-language model approach. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2026.1658575.
- Schmidt, L., Shokraneh, F., Pieper, D., Mathes, T. (2025). Data extraction methods for systematic review (semi)automation: Update of a living systematic review [update of Mathes et al. 2017]. *F1000Research*.
- Scott, A. M., et al. (2025). Generative artificial intelligence use in evidence synthesis: A systematic review. *Research Synthesis Methods*, 16, 601-619. DOI: 10.1017/rsm.2025.16.
- Tan, Z., & D'Souza, J. (2026). Diagnosing structural failures in LLM-based evidence extraction for meta-analysis. *arXiv:2602.10881*. Accepted at IRCDL 2026.
- Topp, C. F. E., et al. (2023). AgroEcoList: A checklist to improve reporting of ecological research in agronomy. *PLOS ONE*, 18(6), e0285478.
- Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). Self-consistency improves chain of thought reasoning in language models. *arXiv*:2203.11171. Presented at ICLR 2023.
- Yun, H. S., Pogrebitskiy, D., Marshall, I. J., & Wallace, B. C. (2024). Automatically extracting numerical results from randomized controlled trials with large language models. *Proceedings of Machine Learning Research*, 252, 818-840.

---

# Data Availability Statement

The pipeline source code, configuration files, prompt templates, validation scripts, and pre-computed outputs are publicly available at https://github.com/halpernmoshe/Meta-analysis-extractor-agriculture (archived at https://doi.org/10.5281/zenodo.18670296). Reference-standard validation datasets are from published meta-analyses: Loladze (2014, eLife 3:e02245), Hui et al. (2023, Journal of Soil Science and Plant Nutrition), and Li et al. (2022, Frontiers in Plant Science 13:836702). Source PDFs cannot be redistributed due to publisher copyright.

# Author Contributions (CRediT)

- **Conceptualization**: Moshe Halpern
- **Methodology**: Moshe Halpern
- **Software**: Moshe Halpern
- **Validation**: Moshe Halpern
- **Formal analysis**: Moshe Halpern
- **Investigation**: Moshe Halpern
- **Data curation**: Moshe Halpern
- **Writing, original draft**: Moshe Halpern
- **Writing, review and editing**: Moshe Halpern
- **Visualization**: Moshe Halpern

# Conflict of Interest Statement

The author declares no conflicts of interest. The AI models used in the pipeline (Claude, Kimi, Gemini) are commercial products; the author has no financial relationship with their providers beyond standard API usage fees.

# Funding

No external funding was received for this research.

---

# Figure List

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | Pipeline architecture with confidence assignment | `fig1_pipeline_architecture.png` |
| Figure 2 | Consensus reliability: MAE by consensus fraction and confidence tier | `fig2_consensus_reliability.png` |
| Figure 3 | Per-paper MAE bar chart with consensus/vision indicators | `fig3_paper_mae_confidence.png` |
| Figure 4 | TOST equivalence forest plot with summary statistics | `fig_tost_equivalence.png` |
| Figure 5 | Bland-Altman analysis across all three datasets | `fig_bland_altman_trio.png` |
| Figure 6 | Combined scatter plots: (A) Loladze, (B) Hui, (C) Li 2022 | `fig_combined_scatter.png` |
| Figure 7 | Consensus predictors: difficulty level vs. consensus fraction, MAE, and challenge count | `fig7_consensus_predictors.png` |

All figures in `output/paper_figures/` at 300 DPI.

# Supplementary Figures

| Figure | Description | File |
|--------|-------------|------|
| S1 | Scatter plot: extracted vs reference-standard effect sizes by element (Loladze) | `fig2_scatter_loladze.png` |
| S2 | Element-level mean effect comparison: extracted vs reference standard | `fig4_element_effects.png` |
| S3 | Bland-Altman plot (Loladze only): bias and 95% limits of agreement | `fig7_bland_altman_formal.png` |
| S4 | TOST forest plot: per-element CIs with equivalence bounds | `fig8_tost_equivalence.png` |
| S5 | Error distribution: histogram and cumulative | `fig_error_distribution.png` |

# Table List

| Table | Description |
|-------|-------------|
| Table 1 | Combined validation results across all three datasets (Section 3.2) |
| Table 2 | Bootstrap confidence intervals for key metrics (Section 3.10.4) |
| Table 3 | Comparison of LLM-based data extraction systems (Section 4.1) |
| Table 4 | Root-cause classification of the 12 Li 2022 excluded papers with validation discordance (Section 4.4) (**moved to Supplementary Tables S1 and S5**) |

# Supplementary Tables

| Table | Description | File |
|-------|-------------|------|
| **S1** | **Forensic Audit of Excluded Li 2022 Papers** — see full table below | inline |
| S2 | Per-paper validation details (46 Loladze papers) | `S2_per_paper_validation.csv` |
| S3 | Per-element accuracy breakdown (20 elements) | `S3_element_accuracy.csv` |
| S4 | Consensus statistics and model contributions | `S4_consensus_stats.csv` |
| S5 | Li 2022 paper-level audit: root-cause classification of all 28 papers | `S5_li2022_paper_audit.csv` |
| S6 | Data completeness and capture rates | `S6_data_completeness.csv` |

---

## Supplementary Table S1: Forensic Audit of Excluded Li 2022 Papers

*Pre-registered exclusion rationale for the 12 papers excluded from the Structurally Concordant Subset. All exclusion decisions were made prior to computing the r = 0.996 subset statistic, based solely on the per-paper diagnostic audit reports generated by the independent auditing agent. The "Corrected PDF Result" column shows outcomes from re-running the pipeline on the correct PDFs where applicable.*

| Paper | Exclusion Category | Diagnostic Evidence | Corrected PDF Result |
|-------|-------------------|---------------------|----------------------|
| Abdel-Mawgoud (2010) | **PDF/consensus failure** — figure-only data | Claude recon confirmed all yield data in Figures 1–3 only; no numerical tables. Claude extracted 0 obs; Kimi extracted 24 from figure text. Gemini tiebreaker timed out. | 7 vision obs (low confidence); no text consensus possible for this paper format. |
| Alabdulla (2019) | **PDF/consensus failure** — single-model extraction | Kimi timed out during extraction; Gemini tiebreaker also timed out. Only 36 Claude-only observations available, no consensus formed. | 36 Claude-only obs (low confidence); Kimi timeout unresolved. |
| Mondal (2013) | **GT attribution error** — wrong crop in database | GT database lists crop as wheat; source PDF is a rice hydroponics study (chitosan on rice *Oryza sativa*). Auditing agent found cultivar mismatch on page 1. | N/A — original PDF is correct; mismatch is in Li et al. (2022) database attribution. |
| Pohl (2019) | **GT attribution error** — wrong metric in database | GT row expects yield (t/ha); paper reports eggplant fruit composition (dry matter %, protein %). No yield data in the paper. | N/A — original PDF correct; GT database attributed wrong outcome variable. |
| Głosek-Sobieraj (2018) | **GT attribution error** — wrong paper in database | GT reference cites Głosek-Sobieraj (2018) but the effect sizes match a different study in the Li database. Auditing agent found systematic value mismatch across all GT rows. | N/A — original PDF correct; Li et al. (2022) cross-referenced a different study. |
| Godlewska (2016) | **GT outcome-category mismatch** | Correct PDF confirmed: Godlewska (2016) *Journal of Elementology* reports microelement content (Zn, Cu, Fe, Mn in g kg⁻¹) of grass species, not yield. Pipeline extracted 60 consensus observations correctly. GT row expects yield (t ha⁻¹). | 60 consensus obs (r ≈ 1.0 for microelements), but no valid comparison to yield GT. GT database incorrectly mapped this paper to a yield row. |
| Kocira (2018) | **Aggregation-level mismatch** | GT stores per-year observations (2013, 2014, 2015 separately); pipeline extracts multi-year averages from summary tables. Both are valid analytical choices for the same data. | N/A — applies to original PDF. |
| Kocira (2020) | **Aggregation-level mismatch** | Same as Kocira (2018): GT stores per-year; pipeline extracts averages. Auditing agent confirmed both sources of data are from the same paper. | N/A — applies to original PDF. |
| Procházka (2015) | **Aggregation-level mismatch** | GT stores observations by treatment year; pipeline extracts multi-year means. The pipeline extracted the correct multi-year summary values; GT disaggregates them. | N/A — applies to original PDF. |
| Kocira (2019) | **Product-selection omission** | Li et al. (2022) GT includes a Kelpak (seaweed extract) treatment arm that the pipeline missed, extracting only the Terra Sorb (amino acid) arm from the paper title. Both arms are present in the PDF; pipeline did not include both products. | Pipeline limitation: multi-product papers require explicit configuration listing all product arms. |
| Pramanick (2016) | **GT source mismatch** | GT effect sizes do not match any table in the source PDF. Auditing agent confirmed values are from a pre-publication dataset with different baseline measurements (inflation factor ≈ 1.4×). | N/A — GT values sourced from pre-publication data not in the PDF. |
| Kuisma (1989) | **GT source mismatch** | GT effect sizes are systematically larger than values in the 1989 PDF. Auditing agent identified a 1.3–1.6× inflation consistent with pre-publication data or a different year's measurements. | N/A — GT values sourced from pre-publication data not in the PDF. |

*All 16 papers in the Structurally Concordant Subset achieved 100% capture rate (all GT rows matched). The 2 consensus-layer failures (Popescu 2018, Wilczewski 2018) and 2 validator-artifact papers (Lola-Luz 2014, Soppelsa 2018) are included in the Structurally Concordant Subset because their issues were identified and resolved prior to final analysis: Popescu and Wilczewski via `recover_consensus.py`; Lola-Luz and Soppelsa via effect-%-based validator matching.*

# Supplementary Dataset: Pipeline Outputs and Meta-Analytic Results

The pipeline-extracted datasets used in this study are provided in full as supplementary data files, enabling independent use for meta-analysis and replication:

| File | Contents | Records |
|------|----------|---------|
| `SD1_loladze_extracted.csv` | All extracted observations from the 46 Loladze CO2/mineral papers: element, tissue type, crop species, CO2 level, control mean, treatment mean, effect %, SE/SD where available, confidence tier | 1,652 obs / 46 papers |
| `SD2_hui_extracted.csv` | All extracted observations from the 34 Hui Zn biofortification papers: Zn fraction, tissue, application method, dose, control mean, treatment mean, effect %, confidence tier | ~800 obs / 34 papers |
| `SD3_li2022_extracted.csv` | All extracted observations from the 28 Li biostimulant papers: crop, product, dose, yield metric, control mean, treatment mean, effect %, confidence tier | ~600 obs / 28 papers |

**Meta-analytic summaries** computed from the pipeline-extracted datasets are provided in:

| File | Contents |
|------|----------|
| `SD4_loladze_metaanalysis.csv` | Per-element pooled effects (lnRR, 95% CI, n) from the CO2/mineral dataset; DerSimonian-Laird random-effects model, 23 elements |
| `SD5_hui_metaanalysis.csv` | Per-application-method and per-tissue pooled Zn effects from the biofortification dataset; 13 subgroups |
| `SD6_li2022_metaanalysis.csv` | Per-biostimulant-category and per-crop pooled yield effects; 37 subgroups |

**Headline results from the pipeline-extracted meta-analyses:**

*SD4: Elevated CO2 reduces macro- and micronutrient concentrations.* Across 635 matched observations and 23 elements, the pipeline extracted consistent CO2-driven dilution effects. Major macronutrients: N −9.5%, P −7.7%, K −7.3%, Mg −8.4%. Micronutrients: Cu −6.8%, Zn −4.5%, Fe −0.9%. These values accord with published syntheses (Loladze 2014 reported −8% overall), confirming that the pipeline reproduces not only individual study values but aggregate meta-analytic patterns.

*SD5: Zn biofortification shows strong dose-response.* Across 310 matched observations, overall grain Zn increased by +51.1% [45.7, 56.6] under Zn treatment. Effect magnitude was strongly application-method-dependent: application type 2: +34.1% [29.4, 38.8]; type 3: +63.3% [50.6, 76.1]; type 4: +84.6% [70.4, 98.9]. High heterogeneity (I² = 100%) reflects genuine biological variability across soils, cultivars, and dose levels.

*SD6: Biostimulants increase yield by approximately 15% on average.* Across 163 matched observations and 28 papers, the pipeline extracted an overall yield increase of +15.4% [11.9, 19.0]. By category: seaweed extracts +15.4%, chitosan +20.0%, humic/fulvic acids +37.8%, plant hormones +8.4%, silicon +8.0%. By crop: soybean +28.1%, maize +17.5%, blackgram +24.4%, wheat +9.3%, potato +1.3%. High heterogeneity (I² ≈ 99%) is expected given the diversity of biostimulant products and crops.

These files represent the first openly available machine-extracted, confidence-stratified datasets for CO2/mineral, Zn biofortification, and biostimulant yield meta-analyses in plant science. Researchers may use them directly for downstream synthesis, reanalysis, or as training data for future extraction systems. See the Data Availability Statement for repository links.
