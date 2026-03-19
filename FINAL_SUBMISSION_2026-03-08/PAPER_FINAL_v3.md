*Running head:* Multi-Model AI Consensus for Data Extraction

# Multi-Model AI Consensus for Reliable Data Extraction in Plant Science Meta-Analysis

**Moshe Halpern** ^ORCID^

Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization -- Volcani Center, Israel

*Correspondence:* Moshe Halpern, Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization -- Volcani Center, Rishon LeZion 7505101, Israel.

---

# Abstract

**Background:** Data extraction remains the primary bottleneck in meta-analysis. Large language models achieve >90% accuracy on categorical study characteristics but only 26--36% on continuous quantitative outcomes. No AI extraction system has been validated on agricultural data, and existing systems lack mechanisms for predicting which extractions are trustworthy.

**Methods:** We developed a multi-model consensus pipeline (Claude Sonnet 4 + Kimi K2.5, Gemini 3 Flash tiebreaker) using inter-model agreement for confidence-stratified triage. We validated against three published plant science datasets totalling 1,154 matched observations across 94 papers: Loladze 2014 (CO2/minerals, development), Hui 2023 (Zn biofortification, independent holdout), and Li 2022 (biostimulants, cross-domain).

**Results:** Independent holdout (Hui) achieved r = 0.999, MAE = 0.43 pp (319 observations). Cross-domain Li validation yielded r = 0.951 (200 observations); a purely programmatic classifier---using only zero-error fraction, direction agreement, and effect-size thresholds with no LLM judgment---identified 110 high-confidence observations (18 papers) achieving r = 0.999, MAE = 0.32 pp. This programmatic subset agreed with an independent LLM audit on 85% of paper classifications, confirming that the high-quality identification is reproducible without circularity. For multi-element Loladze (r = 0.669, MAE = 7.9 pp), decomposition showed 73% of error originated from analytical sub-selection disagreements rather than reading failures; removing alignment artifacts yielded r = 0.915. Aggregate effects reproduced within 0.05--0.84 pp across all datasets (|Cohen's d| < 0.20). Multi-model consensus concentrated 95% of large errors in the flagged minority.

**Conclusions:** The pipeline reduces extraction time by 70--75% at ~$0.37/paper. We introduce a Three-Barrier framework (Reading, Granularity, Provenance) showing that apparent validation failures predominantly reflect challenges in *proving* extraction accuracy---sub-selection disagreements and reference-standard heterogeneity---rather than AI reading errors. Crucially, the classification of discrepancy sources is verified by a fully programmatic algorithm reproducible without LLM involvement, breaking the "LLMs confirming LLMs" circularity that otherwise undermines validation credibility.

**Keywords:** meta-analysis, data extraction, large language models, consensus, quality prediction, automation, plant science, methodological concordance, validation circularity

---

# 1. Introduction

Meta-analysis is the cornerstone of evidence-based practice in agricultural and environmental science. Because individual experiments are conducted across diverse soils, climates, and cultivars, consistent biological patterns often emerge only through quantitative pooling. For example, Loladze (2014) synthesized 1,481 mineral concentration measurements from 130 species across 25 elements to reveal the hidden nutritional cost of elevated CO2. However, the primary bottleneck in such syntheses is data extraction. Trained researchers must manually identify, read, and record quantitative values, a process requiring 2 to 8 hours per paper (Schmidt et al., 2025). Furthermore, Buscemi et al. (2006) demonstrated that single-extractor error rates reach 17.7%, falling to 8.8% only under costly dual-extraction protocols. This introduces a fundamental benchmark problem: when human reference standards contain errors, it becomes difficult to distinguish true AI extraction failures from human curation artefacts. For large meta-analyses encompassing 100+ papers, the human cost of data extraction alone can approach thousands of hours.

Agricultural field trials present distinct extraction challenges that complicate automated data retrieval, particularly compared to the clinical trials that dominate current language model research. Lacking standardized frameworks like the CONSORT reporting guidelines (Topp et al., 2023), agricultural studies must be interpreted on their own terms. Plant science experiments routinely employ complex factorial designs (CO2 x cultivar x soil amendment x harvest date x application rate), producing multi-layered tables where only a fraction of the dozens of treatment combinations may be relevant to a specific meta-analytic question. An automated system must not only read numbers correctly but also align with the specific analytical sub-selection choices made by human meta-analysts. Furthermore, diverse exposure systems (Free-Air CO2 Enrichment, Open-Top Chambers, controlled growth chambers) dictate which treatments qualify as valid controls. Outcomes span multiple plant tissues (leaf, grain, root, stem) across various developmental stages, and data appear in heterogeneous units (g/kg dry weight vs. mg/100g fresh weight vs. µmol/g) that vary both between and within studies. Variance reporting is similarly inconsistent, utilizing standard errors (SE), standard deviations (SD), least significant differences (LSD), or merely letter-based significance groupings. Consequently, nearly 70% of ecological meta-analysis datasets include studies with missing standard deviations (Nakagawa et al., 2023), and 26% of published plant ecology meta-analyses rely on unweighted analyses due to unavailable variance data (Koricheva & Gurevitch, 2014). Finally, older literature from the 1980s and 1990s, critical for longitudinal CO2 research, often exists only as scanned PDFs with embedded image tables, further obstructing machine readability.

While large language models (LLMs) have demonstrated high proficiency in extracting categorical study characteristics, they struggle with the continuous quantitative outcomes that form the basis of meta-analytical pooling. Recent comprehensive evaluations (Jansen et al., 2025; Peng et al., 2025; Kataoka et al., 2026) consistently demonstrate that while LLMs achieve >90% accuracy on categorical variables, performance drops precipitously (26–75%) for continuous numerical outcomes and effect sizes. Ultimately, while LLMs excel at semantic comprehension, reliable continuous numerical extraction from complex tables remains an unsolved problem.

Beyond raw accuracy, practical deployment of automated extraction requires reliable confidence estimation at the level of individual data points. Existing systems typically report aggregate performance metrics but lack mechanisms to predict which specific extractions are correct, a critical requirement for minimizing human review. Multi-model concordance offers a principled solution to this reliability problem. Khan et al. (2025) demonstrated that when two independent LLMs agree on an extracted value, the hallucination rate is just 0.25%, whereas disagreement signals a hallucination rate of 26--41%. Poser et al. (2026) independently confirmed this dynamic, showing that three-model consensus reduced clinical data extraction errors to 1.48%. Because concordance status is generated intrinsically during extraction, it provides a powerful quality signal that requires no ground truth. However, neither study deployed this concordance metric as a proactive confidence estimator to automatically stratify observations for human review.

The development of such automated systems has also been heavily skewed toward biomedical applications, leaving the complex domain of agricultural ecology largely unaddressed. Clark et al. (2025) found that 17 of 19 generative AI systematic review studies focused on clinical or biomedical settings, despite demonstrations of fully automated Cochrane review synthesis achieving 93.1% accuracy (Cao et al., 2025). Agricultural meta-analysis represents the most demanding tier of extraction complexity, characterized by statistical outcomes, heterogeneous units, and an absence of reporting standards (Li et al., 2025, Tier 3: continuous numerical outcomes and variances), and no published system has been validated against it. In such complex extraction environments, single-model systems are particularly vulnerable; Tan & D'Souza (2026) identified four structural LLM failure modes in evidence extraction: role confusion, binding drift (attributing values to the wrong row or column), multi-instance compression (merging distinct experimental arms into a single output), and error amplification, suggesting that individual models possess characteristic blind spots. Overcoming these structural vulnerabilities requires a multi-model consensus approach capable of detecting and filtering these domain-specific errors.

Underlying all three gaps is a more fundamental epistemic challenge. Evaluating AI extraction accuracy requires a reference standard, but no published meta-analysis dataset constitutes error-free ground truth: each was assembled by human researchers navigating the same complex tables, ambiguous reporting conventions, and analytical sub-selection decisions that confront any extraction system. Attribution errors, mismatched PDFs, and defensible-but-different methodological choices are inevitable artefacts of human-curated data at scale. Consequently, validating AI against published reference standards conflates extraction errors with reference-standard heterogeneity, and apparent low performance may reflect the limitations of the standard rather than the capability of the system. We address this directly by treating our three validation datasets not as gold standards but as an imperfect triangulation strategy: each dataset was assembled under different conditions and illuminates a different dimension of the extraction problem.

A second epistemic challenge arises when attempting to decompose discordances into their sources. If an LLM auditor classifies discrepancies between LLM-extracted values and a reference standard, the auditor may exhibit systematic bias toward exonerating the extractor ("the ground truth must be wrong") rather than identifying genuine extraction failures---creating a circularity where LLMs confirm LLMs. We address this by developing a fully programmatic classification algorithm that uses only observable data properties (exact-match fraction, direction agreement, effect-size thresholds, and scale-ratio patterns) to classify papers into confidence tiers, with no LLM judgment involved. Per-paper diagnostic audits using a separate LLM (Claude Sonnet 4.6) provide corroborating context but are not the primary basis for any statistical claim.

We conceptualize the validation challenge---the difficulty of *proving* that AI extraction works---as a Three-Barrier model. The **Reading Barrier** asks whether an LLM can accurately extract numbers from tables. The **Granularity Barrier** asks whether apparent errors reflect genuine reading failures or legitimate analytical sub-selection disagreements between the AI and the human meta-analyst. The **Provenance Barrier** asks whether the reference standard itself is reliable enough to serve as ground truth. Each barrier is a challenge to validation methodology, not to extraction capability.

To address these challenges, we present a multi-model consensus pipeline validated on three published plant science datasets, each isolating a distinct barrier. We first evaluate the Reading Barrier using the Hui (2023) zinc biofortification dataset, demonstrating that numeric extraction is effectively solved for standard tabular formats. We next address the Granularity Barrier using the Loladze (2014) development set, showing that the majority of apparent error reflects sub-selection disagreements rather than reading failures. Finally, we expose the Provenance Barrier using the Li (2022) cross-domain dataset, revealing that reference-standard artefacts dominate apparent discordances---and crucially, we demonstrate that this decomposition can be performed by a fully programmatic algorithm using only observable data properties, without relying on LLM auditors to classify discrepancies. We demonstrate that multi-model agreement reliably predicts which extractions are accurate, enabling confidence-stratified triage without ground truth.

---

# 2. Methods

## 2.1 Pipeline Architecture

The pipeline consists of four stages (Figure 1): challenge-aware reconnaissance, dual-model extraction, consensus building with tiebreaker, and confidence-stratified post-processing.

**Challenge-aware reconnaissance.** For each paper, Claude Sonnet 4 performs a structured scan identifying: variance reporting format, sample size locations, tables containing target outcome data, experimental design characteristics, and extraction challenges. Papers are classified by challenge type (SCANNED, IMAGE-TABLES, FIGURE-ONLY) and routed to one of three extraction modes: TEXT (clean machine-readable tables), HYBRID (text + vision for image-embedded tables), or VISION (figure-only data requiring image analysis).

**Dual-model extraction.** Two LLMs independently extract data using identical structured prompts: Claude Sonnet 4 (Anthropic, 200K context) and Kimi K2.5 (Moonshot AI, 256K context). Both receive the same prompt containing target variable definitions, table targeting directives from reconnaissance, structured output format requirements, and a checklist of all target elements. The prompt utilized a zero-shot, role-playing structure with a strict JSON schema output requirement. The schema required nested key-value pairs for treatment arm, control mean, treatment mean, variance, sample size, and units, explicitly defining inclusion/exclusion criteria for target tissues and variables. To minimize hallucination, instructions explicitly forbade calculating implicit values unless mathematically trivial (e.g., deriving variance from standard error and *n*) and directed models to extract units exactly as written, leaving standardization for downstream processing. Models were instructed to report null rather than guess when values were uncertain. Kimi K2.5 utilized Chain-of-Thought (CoT) reasoning to parse complex table headers before extraction. For HYBRID-mode papers, Gemini 3 Flash additionally performs vision-based extraction from PDF page images. Full prompt templates are provided in the online repository. **Vision extraction note:** All three models support native multimodal (vision) input: Claude Sonnet 4 (Anthropic), Kimi K2.5 (Moonshot AI), and Gemini 3 Flash (Google). For HYBRID-mode papers (those containing image-embedded tables or degraded scanned PDFs), all three models receive PDF page images, enabling triple-model vision verification. Observations confirmed by two or more models receive HIGH confidence; single-model extractions receive LOW confidence and are flagged for human review. This triple-vision architecture eliminates the need for OCR pre-processing and ensures that any single model's misreading is overridden by the majority.

### 2.1.1 Computational Environment and Parameters

PDF text is parsed using PyMuPDF (fitz) for TEXT-mode processing by Claude Sonnet 4, which extracts text blocks while attempting to preserve spatial layout. Because complex multi-column layouts often degrade during text-only parsing, papers with challenging formatting are routed to HYBRID or VISION modes, where raw PDF page images are passed directly to the natively multimodal APIs (Kimi K2.5, Gemini 3 Flash, and Claude's vision endpoint). All extraction API calls utilize a temperature setting of 0.0 to maximize determinism. When the Gemini tiebreaker is invoked to resolve poor initial consensus (empirically defined during development as a global match rate <30%. A match rate below 30% typically indicated a fundamental misalignment in how the two models parsed the table's row/column matrix, necessitating a third multimodal perspective), Gemini 3 Flash performs an independent extraction of the entire paper using the same prompt, and its extracted continuous values are compared against the unmatched outputs of Claude and Kimi using the standard $\pm15\%$ relative-error tolerance. If Gemini's value falls within tolerance of *either* Claude or Kimi, a 2-of-3 consensus is formed, and the matching values are averaged.

**Data privacy note:** This pipeline transmits extracted document text and page images to third-party commercial API providers (Anthropic, Moonshot AI, Google). Users should ensure compliance with institutional data governance policies and publisher terms before processing restricted, paywalled, or sensitive materials. Extracted text (not binary PDF files) is transmitted; the pipeline does not upload raw PDFs.

**Consensus building.** After independent extraction, observations from both models are compared using element-tissue matching with value tolerance (default: 15% relative error. This threshold was empirically selected to accommodate typical OCR transcription errors of single digits or minor rounding differences by human extractors, while strictly penalizing order-of-magnitude or wrong-column errors; for near-zero values, defined mathematically as absolute values < 1.0, 0.5 units absolute):

- **Matched pairs** (both models agree within tolerance): Accepted at "high" confidence, values averaged
- **Unmatched observations** (single model only): Retained at "medium" confidence with source model noted
- **Vision-only observations** (from HYBRID/VISION mode without text consensus): Retained at "low" confidence

When initial consensus is poor (defined as a **global match rate <30%** of the total observations extracted by the lead model), a tiebreaker is invoked: Gemini 3 Flash performs an independent extraction, and 2-of-3 voting determines accepted observations.

**Confidence assignment.** Each observation receives a confidence label based on its provenance:

- **High**: Two or more models independently extracted matching values. These observations have the strongest reliability guarantee.
- **Medium**: Extracted by a single text-based model without corroboration, or resolved via tiebreaker.
- **Low**: Extracted via vision/OCR from image-embedded tables or scanned PDFs, often without multi-model consensus.

This three-tier system enables downstream triage: high-confidence observations can be used directly; medium and low-confidence observations are flagged for human review. We evaluate whether these labels predict actual accuracy in Section 3.3.

**Post-processing.** Final observations undergo duplicate removal, null-mean filtering, and treatment/control swap flagging. Automatic swap correction was tested and found harmful (Section 3.10); flagging is informational only.

**Design rationale: Strategic Heterogeneity vs. Self-Consistency.** An alternative architecture would run all three models on every paper and take majority vote — or sample the same model multiple times (Self-Consistency; Wang et al., 2022). We chose *heterogeneous* dual extraction with a conditional tiebreaker for two reasons. First, self-consistency reduces stochastic variance within a single model but cannot reduce systematic inductive bias: if one model consistently interprets "main effect averaged across co-treatments" while another extracts "within-treatment CO2 effect," running the same model ten times converges on one systematic choice. Baslam et al. (2012) illustrates this: Claude extracted 38 observations (main-effect view) while Kimi extracted 76 (interaction-aware view) — their disagreement flags the analytical ambiguity rather than amplifying one model's systematic choice. Second, complementary capability: Claude operates on extracted text; both Kimi and Gemini are natively multimodal and can process page images directly. This capability mix, text-dominant with vision fallback across two independent models, would be lost in a three-text-model majority vote. The ablation (Section 3.12) confirms that no single model dominates across all elements, supporting the heterogeneous approach.

**Worked example.** Figure 1 (right panel) illustrates the pipeline processing two contrasting papers. For Baslam et al. (2012), a clean paper with structured tables, both Claude and Kimi independently extract identical Ca values (8.21 mg/g control, 7.43 mg/g elevated), yielding 100% consensus and an MAE of 1.0%. For Fangmeier et al. (2002), a complex CO2 × O3 factorial design, only Kimi extracts usable data from text (Claude returns 0 observations), requiring Gemini's vision fallback. The consensus fraction drops to 23%, correctly predicting the higher error (MAE = 8.0%), though as Section 3.10 documents, that error is predominantly a methodological concordance issue rather than a table-reading failure.

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

**Important caveat**: The pipeline was developed and iteratively refined using feedback from this dataset. Prompt templates, matching logic, and consensus parameters were adjusted based on Loladze validation results. This dataset therefore provides an upper-bound estimate of performance and should not be considered an independent test. We explicitly report this distinction and rely on the Hui dataset (Section 2.3.2) for independent holdout validation.

**Methodological concordance caveat**: The Loladze 2014 dataset contains an `info` field for each observation that documents the specific methodological sub-selections made by Loladze when computing effect sizes from complex factorial designs. This field, analyzed in Section 3.10, reveals that a substantial portion of the Loladze MAE reflects methodological concordance failures, cases where the pipeline and Loladze selected different but equally legitimate analytical sub-conditions, rather than numerical reading errors. We use this field systematically to decompose the reported MAE into its extraction and concordance components.

### 2.3.2 Hui et al. 2023 (Independent Holdout Validation)

The secondary dataset is from Hui et al. (2023), a meta-analysis of zinc biofortification in wheat (1,593 observations, 139 studies). We processed 34 papers, of which 21 matched reference-standard entries, yielding 319 matched observations after scale-harmonized matching (accounting for unit differences between mg/kg and mg/100g reporting conventions).

**This dataset was not used during pipeline development.** We designate this as an independent holdout set. While the JSON schema was configured for zinc biofortification, no data from the Hui dataset was used to tune prompts or model parameters. The pipeline was configured by changing only the JSON configuration file (specifying Zn wheat biofortification outcomes, moderators including application method type, and dose ranges). This provides the cleanest test of the system's generalizability.

One paper (`Li_2013.pdf`) was identified during auditing as a mismatched PDF; it is retained in the primary analysis for transparency but flagged.

### 2.3.3 Li et al. 2022 (Cross-Domain Validation)

The third dataset is from Li et al. (2022), a meta-analysis of non-microbial biostimulant effects on agronomic outcomes (1,108 observations, 181 studies). Target outcomes include crop yield (primary), biomass, and quality traits (e.g., total phenols, antioxidant activity, fat content), reflecting the breadth of biostimulant research. This tests effects across diverse crops and heterogeneous units. We processed 49 papers, of which 27 matched to reference-standard entries, yielding 200 matched observations after scale-harmonized matching. Each paper was classified into confidence tiers using a programmatic algorithm (Section 2.6.1) based solely on observable data properties: 18 papers (110 observations) were classified as high-confidence, 4 papers (29 observations) as medium, and 5 papers (61 observations) as low. Like the Hui dataset, this dataset was not used during development.

## 2.4 Reference-Standard Matching Protocol

For each extracted observation, a corresponding reference-standard row was identified using a dataset-specific hierarchical matching algorithm. All matching criteria were specified before computing accuracy metrics; no criterion was modified after examining results.

**Loladze matching.** Observations were matched by exact element symbol (after normalization: "Fe" = "Iron" = "iron"; upper- and lower-case equivalents collapsed) and tissue type (leaf/grain/root/whole-plant). Because our pipeline prioritizes high recall, it generates an 'information surplus' (e.g., extracting all factorial arms). The minimum-error selection simulates a user who subsequently filters this exhaustive output to match their specific analytical scope using the pipeline's categorical metadata. Manual spot-checking confirmed that categorical metadata extraction (e.g., crop species, tissue type) achieved >95% accuracy, ensuring reliable downstream filtering. The `n_candidates` field in the validation CSV records how many candidates existed per match; 84% of matches had only one candidate (unambiguous assignment), limiting the impact of this optimistic selection.

**Hui matching.** Observations were matched by tissue type and application method code (app_type). Within each paper--tissue--app_type stratum, the minimum-error candidate was selected. Control and treatment mean values were matched using a ±15% relative-error tolerance for initial pairing, with the closest pair selected when multiple fell within tolerance.

**Li matching (primary, naive).** Observations were matched by crop species and a freely chosen outcome label, without filtering by outcome type. This "naive" matching yields 163 matched pairs and is the primary reported result. A yield-outcome-filtered variant (described in Section 3.6) is reported for comparison.

Unmatched observations on either side (extracted but no reference-standard counterpart, or reference-standard rows with no extracted match) were excluded from accuracy calculations and are tabulated separately as precision and recall components in Section 3.1.

## 2.5 Per-Paper Diagnostic Audit

For all three datasets, we conducted systematic per-paper diagnostic audits using a newer model (Claude Sonnet 4.6, released after the extraction pipeline was finalized) acting as an independent reader, separate from the extraction pipeline. For each paper, the auditing agent received: (1) the full source PDF text, (2) the corresponding reference-standard rows for that paper, and (3) a structured diagnostic template requiring it to document the experimental design, identify which tables contained target data, compare extracted values against the reference standard, and classify any discrepancies into mutually exclusive categories: reading error, methodological sub-selection, reference-standard artifact (wrong PDF, database attribution error), or aggregation-level mismatch. The auditing agent had no access to the extraction pipeline's code or aggregate accuracy statistics, ensuring independence from the primary extraction. The diagnostic template and 30 example reports are provided in the online repository.

The `info` field of the Loladze reference CSV provided the primary evidence for decomposing Loladze discordances into extraction versus concordance components. This field, populated by Loladze (2014) for each observation, documents specific sub-selection choices (year, site, co-treatment arm, cultivar) that the auditing agent could compare against the pipeline's analytical choices.

**Important caveat on circularity.** Using an LLM to audit LLM-extracted data creates a potential "LLMs confirming LLMs" bias: the auditor might systematically attribute discrepancies to reference-standard errors rather than extraction failures. To address this, we developed a fully programmatic classification (Section 2.6.1) as the primary basis for identifying high-confidence observations. The LLM audit provides qualitative context (e.g., explaining *why* a specific paper's data cannot be reconciled with the reference standard) but no statistical claim in this paper depends solely on LLM-generated classifications.

## 2.6 Validation Metrics

Six metrics assessed extraction accuracy: (1) Pearson correlation coefficient (r) between extracted and reference-standard effect sizes; (2) Mean Absolute Error (MAE) in percentage points; (3) within-threshold accuracy (proportion within 5%, 10%, 20% of reference-standard values); (4) **direction agreement**: defined as sign(extracted effect) = sign(reference effect), where "effect" is (treatment mean − control mean) / |control mean| × 100; observations where the reference-standard effect is within ±0.5% of zero (near-zero, sign unreliable) are excluded from direction agreement calculations to avoid ratio spikes; (5) element capture rate; (6) overall effect reproduction (aggregate mean effect comparison).

Accuracy metrics (r, MAE, TOST, Bland-Altman) are computed on raw extracted effect sizes (derived from treatment and control means), inclusive of any uncorrected treatment/control swaps. While variance capture (SD/SE) is evaluated for completeness, it is not required for these primary mean-based effect size accuracy calculations.

Formal agreement analyses included Bland-Altman analysis, intraclass correlation [ICC, using a two-way mixed-effects model (ICC type 3,1), absolute-agreement definition, single-measurement unit], two one-sided tests (TOST) for equivalence, and cluster-robust percentile bootstrap confidence intervals (10,000 resamples), where the resampling unit was the study (paper) rather than the individual observation, to account for within-study clustering of extraction errors. The percentile method was preferred over BCa to avoid instability in the acceleration estimate when the number of independent clusters is small (n=16–46 papers). Throughout this paper, "pp" denotes percentage points. All ICC models are reported uniformly using the (model, type, unit) notation.

### 2.6.1 Programmatic Confidence Classification (Li 2022)

To break the potential circularity of using LLM auditors to validate LLM extractions, we developed a fully programmatic algorithm that classifies each Li 2022 paper into confidence tiers using only observable properties of the matched data. The algorithm uses three signals computed directly from the validation CSV, requiring no LLM judgment:

1. **Zero-error fraction.** For each paper, the proportion of matched observations where the absolute effect-size difference between extracted and reference-standard values is <0.01 pp. Exact value matches after scale normalization constitute the strongest possible evidence that the pipeline read the correct numbers from the correct cells.

2. **Direction agreement.** The fraction of observations where the sign of the extracted effect matches the sign of the reference-standard effect. Direction failures signal fundamental mismatches (wrong data matched, wrong factorial arm) rather than minor reading errors.

3. **MAE threshold.** The mean absolute error across all matched observations for that paper, in percentage points.

Papers are classified using the following decision rules, applied in order:

- **Verified correct** (high confidence): Zero-error fraction ≥ 30%---the paper contains exact value matches that prove correct alignment.
- **Likely correct** (high confidence): MAE < 2 pp AND direction agreement ≥ 95%.
- **Aggregation discordance** (low confidence): Direction agreement < 85%---systematic sign disagreements indicate different analytical granularity.
- **Scale anomaly** (low confidence): Fewer than 20% of observations have clean scale ratios (within 8% of a power-of-10 conversion factor) AND coefficient of variation of scale ratios exceeds 50% AND MAE ≥ 2 pp.
- **Moderate discrepancy** (medium confidence): MAE 2--5 pp AND direction agreement ≥ 90%.
- **High discrepancy** (low confidence): All remaining papers.

The specific thresholds (30% zero-error, 2 pp MAE, 95% direction agreement) were calibrated on the Li 2022 dataset to maximize separation between papers with verifiably correct matches and those with documented sub-selection issues. These thresholds are not claimed as universal constants; they reflect the error distributions of this specific domain (agricultural yield data with diverse units) and should be recalibrated for datasets with different noise profiles. Crucially, however, the classifier's *structure* (using only observable data properties) is domain-general, and the thresholds can be adjusted without reintroducing LLM dependence.

This algorithm is implemented in `programmatic_gt_classifier.py` (provided in the repository) and produces identical results on any machine given the same validation CSV. Any researcher can reproduce the classification without access to LLMs, source PDFs, or subjective judgment.

## 2.7 Confidence-Stratified Analysis

To evaluate whether the pipeline's confidence scores predict actual accuracy, we classified each paper by its consensus fraction: the proportion of observations where two or more models agreed. Papers with >50% consensus-confirmed observations were classified as "consensus-dominant"; the remainder as "vision-dependent." We compared accuracy metrics between these groups and across confidence tiers.

## 2.8 Cost Analysis

Per-paper costs were estimated from API usage records, broken down by model and stage. Manual extraction costs were estimated at $30/hour with 4 hours per paper based on literature estimates.

---

# 3. Results

## 3.1 Pipeline Output

The consensus pipeline processed 50 papers from the Loladze dataset, generating 1,652 consensus observations across 14 mineral elements. Of these, 64% were routed to HYBRID extraction and 30% to TEXT-only. The Gemini tiebreaker was invoked for 11 papers (22%). Means extraction was near-complete (>98%), while variance capture was 67%. This lower variance capture rate reflects the reality of agricultural reporting rather than model reading failures; many studies omit explicit variance measures entirely, reporting only a/b/c significance letters or relying on pooled ANOVA errors.

Across all three datasets, the pipeline processed 133 papers, extracting approximately 3,000 total observations, of which 1,154 matched to reference-standard entries (Table 1) across 94 papers. The pipeline's exhaustive extraction approach captures all reported factorial arms, generating a large volume of candidate observations that require downstream filtering (analyzed further in Section 3.7).

## 3.2 Overall Accuracy: Three Datasets as a Difficulty Gradient

**Table 1. Validation results across all three datasets.**

| Metric | Loladze 2014 | Hui 2023 | Li 2022 (All) | Li 2022 (Prog. High)† |
|--------|:------------:|:--------:|:-------------:|:---------------------:|
| Status | Development | **Holdout** | Cross-domain | Cross-domain |
| Topic | CO2 + minerals | Zn biofortification | Biostimulant + yield | Biostimulant + yield |
| Expected direction | Negative | Positive | Positive | Positive |
| Papers matched | 46 | 21 | 27 | 18 |
| Matched observations | 635 | 319 | 200 | 110 |
| Elements / outcomes | 14 | 1 (Zn) | Yield + quality | Yield + quality |
| Pearson r | 0.669 | **0.999** | **0.951** | **0.999** |
| MAE (pp) | 7.9 | **0.43** | **2.30** | **0.32** |
| Median AE (pp) | 3.0 | 0.0 | 0.36 | 0.0 |
| Within 5 pp | 58% | 97% | 84% | 100% |
| Within 10 pp | 74% | 99.7% | 91% | 100% |
| Within 20 pp | 91% | 100% | 100% | 100% |
| Direction agreement | 85% | **99.7%** | 93% | **99.1%** |
| Overall effect diff | 0.05 pp | 0.12 pp | 0.84 pp | 0.04 pp |
| ICC (observation) | 0.669 | 0.999 | 0.951 | 0.999 |
| Cohen's d | -0.003 | 0.072 | -0.189 | -0.055 |
| TOST (±2 pp) | p < 0.001 | p < 0.001 | p < 0.001 | p < 0.001 |

† Li 2022 (Aligned) = Programmatic High-Confidence Subset: 18 papers (110 observations) classified as high-confidence by the programmatic data-property algorithm (Section 2.6.1), based on zero-error fraction, direction agreement, and MAE thresholds. No LLM judgment was used in this classification. See Section 3.7.

All formal statistics computed on the scale-harmonized matching described in Section 2.4.

Rather than forming a linear difficulty gradient, the three datasets isolate distinct challenges in *validating* extraction accuracy. Hui serves as a baseline for pure reading accuracy; Loladze introduces the variable of methodological concordance; and Li tests whether validation infrastructure is itself reliable. The programmatic classifier (Section 2.6.1)---using only zero-error fraction, direction agreement, and MAE, with no LLM involvement---identifies 110 observations across 18 papers as high-confidence, achieving r = 0.999 and MAE = 0.32 pp. The residual gap between the full-set r = 0.951 and the programmatic high-confidence r = 0.999 is attributable to papers with direction failures (<85% agreement) or high MAE (>5 pp)---observable data-quality signals, not LLM-generated excuses.

## 3.3 Multi-Model Agreement Predicts Extraction Quality

The pipeline's confidence scores predicted actual extraction accuracy (Figure 2). We classified each Loladze paper by its consensus fraction (proportion of observations confirmed by 2+ models). Papers were divided into two groups.

Papers where more than half of observations were confirmed by inter-model agreement (consensus-dominant) had MAE = 4.3% with strong correlation to the reference standard. Direction agreement exceeded 90%. These represent cases where independent agreement between Claude and Kimi provides a reliable guarantee. Papers relying primarily on single-model or vision extraction (vision-dependent, <50% consensus) had MAE = 11.2%, approximately 2.6× worse than consensus-dominant papers. The 6.96 pp accuracy gap between TEXT and HYBRID extraction modes (2.27% vs. 9.23%; Section 3.1) corroborates this split via an independent, routing-based classification.

At the observation level, high-confidence observations (confirmed by 2+ models) had MAE = 5.2%, while medium/low-confidence observations had MAE = 9.6% (Mann-Whitney p < 0.001). The large majority of errors concentrated in vision-dependent papers: 95% of observations with absolute error >20 pp came from papers routed to HYBRID or VISION extraction modes.

**Holdout validation of confidence stratification (Hui dataset).** To test whether the confidence signal generalises beyond the development set, we examined confidence tiers for the Hui holdout dataset. Of 339 extracted Hui observations, 235 (69%) were high-confidence; 104 (31%) were medium/low-confidence. At the paper level, 9 of 12 papers with extractable data (75%) were consensus-dominant. This mirrors the Loladze development finding and confirms that the confidence tier is a genuine quality signal rather than a development-set artefact.

**TEXT-mode performance.** Papers routed to TEXT extraction (clean machine-readable tables) achieved MAE = 2.27% with r = 0.974 (140 observations). This represents near-perfect accuracy and establishes the system's capability ceiling when paper quality permits clean text extraction.

**HYBRID-mode performance.** Papers requiring vision supplementation (HYBRID mode, 420 observations) achieved MAE = 9.23% with r = 0.532. The 6.96 pp accuracy gap between TEXT and HYBRID modes validates the routing classifier's relevance: papers flagged as challenging are genuinely harder to extract, and the system correctly identifies them. Vision models particularly struggled with log-scale axes, overlapping data points in scatter plots, and complex stacked bar charts, frequently failing to extract precise numerical values from these formats.

## 3.4 What Predicts Consensus Quality?

If multi-model agreement predicts extraction accuracy, what paper attributes predict whether consensus will be achieved? We examined 20 binary challenge features detected during reconnaissance against each paper's consensus fraction using Mann-Whitney tests (Figure 7).

Paper difficulty was the strongest predictor. Papers classified as MEDIUM difficulty during reconnaissance achieved 83% mean consensus (14/18 consensus-dominant), while HARD papers averaged 46% consensus (8/21 consensus-dominant). MEDIUM papers also achieved lower median MAE (3.2%) than HARD papers (7.6%). The total number of detected challenges correlated negatively with consensus fraction (r = -0.37, p = 0.019).

Among individual features, `has_complex_stats` was the most predictive (p < 0.001): papers with complex statistical reporting (ANOVA interaction terms, mixed-model outputs, non-standard variance reporting) averaged 30% consensus versus 76% for papers with straightforward tables. Other features showing meaningful effects included `has_image_tables` (34% vs. 66% consensus), `is_scanned` (49% vs. 71%), and `has_nested_tables` (53% vs. 70%).

Multi-table papers had higher consensus (68% vs. 39%, p = 0.09), likely because papers with multiple tables tend to have well-structured data in at least one table. The same feature also predicted lower MAE (6.0% vs. 19.1%, p = 0.01).

These findings suggest that the reconnaissance stage's difficulty classification is an effective pre-extraction predictor of consensus quality. In a production setting, papers classified as HARD could be automatically flagged for human review or allocated additional extraction passes, while MEDIUM papers can be processed with high confidence in the consensus mechanism.

Jansen et al. (2025) found across 312,329 extractions that accuracy depends more on variable type than on which LLM is used: variables describing study context had higher accuracy than variables for direct effect-size calculation, exactly the distinction between reconnaissance-phase variables (paper type, design, crops) and extraction-phase variables (mean, SD, n). Our finding that `has_complex_stats` is the single strongest predictor of poor consensus (p < 0.001) is the structural equivalent: complex statistical reporting creates a variable-type difficulty that no individual LLM overcomes reliably, but that multi-model disagreement reliably flags.

## 3.5 Independent Holdout Validation: Hui et al. 2023

The Hui dataset provides the cleanest test of pipeline accuracy because it was not used during development. Across 319 matched observations from 21 papers (scale-harmonized matching):

- Pearson r = 0.999 (p < 0.001)
- MAE = 0.43 pp
- Direction agreement = 99.7% (316/317 non-zero observations)
- 81% of observations had zero error (exact match to reference standard)
- Five papers achieved perfect extraction (MAE = 0.0%): Bharti 2013 (40 obs), Rehman 2018 (56 obs), Erdal 2002 (20 obs), Yilmaz 1998 (3 obs), and Forster 2018 (6 obs)

The overall extracted Zn effect was 49.72% versus reference-standard 49.61% (diff = 0.12 pp). No systematic bias was detected (paired t = 1.29, p = 0.198, Cohen's d = 0.072). The strong performance reflects simpler data structure (single element, standardized units, fewer moderators) and confirms that the pipeline generalizes to unseen data without retuning. The Hui result demonstrates that the Reading Barrier---the fundamental ability of LLMs to extract correct numbers from tables---is effectively solved for standardized single-element formats.

ICC(3,1) = 0.999 at the observation level, indicating near-perfect agreement between automated and manual extraction. Bland-Altman limits of agreement were ±3.2 pp, substantially narrower than Loladze (±30 pp), reflecting the simpler data structure.

**Scanned-PDF fallback success (Liu et al. 2019).** Liu et al. (2019), a Zn soil fertilization dose-response trial in winter wheat (Quzhou Experimental Station, two cropping seasons), was flagged HARD during reconnaissance due to scanned PDF format. Kimi K2.5 returned zero observations (blocked by OCR degradation). Claude Sonnet 4 extracted 63 observations using its vision pathway; Gemini 3 Flash independently confirmed the extraction in HYBRID mode. All 10 GT-targeted grain-Zn observations were matched: r = 1.0, MAE = 0.17%. This illustrates that the heterogeneous architecture prevents silent failures: a single-model text pipeline would have returned zero results for this paper, while the multi-modal HYBRID fallback and independent confirmation produced a near-perfect result.

**PDF–reference mismatch (Li_2013).** Per-paper diagnostic audit identified one file (`Li_2013.pdf`) that does not match the paper cited in the Hui 2023 database: the processed PDF contains an IRRI greenhouse rice study (Impa et al. 2013) rather than the Chinese wheat field study cited (Li, M.H. et al. 2013). The reconnaissance phase correctly issued seven out-of-scope warnings, which in a production workflow would trigger an exclusion review. Apparent matches (r = 0.643, 6/12 matched) are numerical coincidences between rice and wheat Zn values rather than valid comparisons. The reported Hui statistics (r = 0.999, MAE = 0.43 pp) conservatively include this mismatched paper in the improved matching; the original naive matching (r = 0.993, MAE = 1.73 pp) was even more affected. This illustrates that the reconnaissance warning system can flag mismatched PDFs, enabling pre-extraction screening.

## 3.6 Cross-Domain Validation: Li et al. 2022

The Li dataset tested generalization to positive-direction effects across diverse crops, biostimulant types, and agronomic outcomes (yield, biomass, and quality traits).

We report three progressive levels of matching refinement (Table 4) to illustrate how validation methodology affects apparent accuracy---and to demonstrate that most "error" in raw validation reflects the measuring instrument, not the extraction pipeline.

**Table 4. Li 2022 Validation: Effect of Matching Methodology on Apparent Accuracy**
| Matching Level | Description | N (Pairs) | Pearson r | MAE (pp) |
|----------------|-------------|-----------|-----------|----------|
| **Naive** | Crop + outcome label matching with no scale harmonization | 163 | 0.453 | 11.62 |
| **Scale-harmonized** | Effect-size-based matching with unit conversion (t/ha ↔ kg/ha) | 200 | **0.951** | **2.30** |
| **Programmatic high** | Restricted to 18 papers classified high-confidence by programmatic algorithm (Section 2.6.1) | 110 | **0.999** | **0.32** |

The progression from r = 0.453 to r = 0.951 to r = 0.999 illustrates a fundamental methodological point: apparent extraction accuracy is a joint function of pipeline quality and validation infrastructure quality. The naive r = 0.453 does not measure extraction capability; it measures the combined effect of unit-scale mismatches in the matching algorithm, reference-standard attribution errors, and aggregation-level disagreements. Simply harmonizing units and accounting for scale factors raises the correlation from 0.453 to 0.951 without changing a single extracted value.

**Breaking the circularity.** The further improvement from r = 0.951 to r = 0.999 might appear to depend on subjective classification of which papers are "correct." To preempt this concern, the programmatic high-confidence subset is defined entirely by observable data properties: papers where ≥30% of observations have zero error (proving exact value matches exist), or where MAE < 2 pp with ≥95% direction agreement. These criteria are computed from the matched numerical data alone. No LLM, no auditor, no subjective judgment determines which papers enter this subset. The algorithm is deterministic and reproducible by any researcher running `programmatic_gt_classifier.py` on the validation CSV.

For independent corroboration, an LLM auditor (Claude Sonnet 4.6) separately classified each paper's discrepancies. The LLM identified 16 papers (100 observations) as "correct," achieving r = 0.996. The programmatic and LLM classifications agreed on 85% of papers (23/27). The four disagreements were all defensible: the programmatic method included three papers the LLM conservatively excluded (Głosek-Sobieraj, MAE = 0.46 pp; Mondal, MAE = 0.84 pp; Godlewska, MAE = 0.17 pp) and excluded one paper the LLM called correct (Matysiak, MAE = 3.1 pp, zero exact matches). In every case, the programmatic classification was based on stronger evidence. This concordance between independent methods---one algorithmic, one LLM-based---confirms that the high-quality subset is real, not an artifact of circular reasoning.

On the 18 programmatic high-confidence papers, the pipeline achieved near-perfect correlation (r = 0.999) with MAE = 0.32 pp, 59% zero-error observations, and 99.1% direction agreement (110 matched observations).

**Positive benchmark: Chen et al. (2021).** This multi-site, multi-season sugarcane seaweed extract trial (flagged as HARD due to scanned PDF) achieved r = 0.995, MAE = 0.15 pp across 21 matched observations with 100% direction agreement, demonstrating that scanned PDFs with clear table structure are handled with near-Hui accuracy.

**Single genuine extraction failure: Kocira et al. (2019).** The pipeline correctly extracted all 12 observations for the Terra Sorb (amino acid) treatment arm with zero error, but completely missed the parallel Kelpak (seaweed extract) arm (12 observations). This scope-selection bias---where the model anchored to the title-emphasized product---is the only confirmed case of the pipeline failing to extract readable tabular data. It represents an addressable prompt-engineering limitation, not a fundamental reading failure.

## 3.7 Error Decomposition Analysis: Programmatic and Corroborative Audit

### 3.7.1 Programmatic Tier Analysis

The programmatic classifier (Section 2.6.1) partitioned the 27 Li papers into three confidence tiers based solely on observable data properties:

**Table 5. Li 2022 papers by programmatic confidence tier.**

| Tier | Papers | Obs | r | MAE (pp) | Direction | Zero-error % | Classification basis |
|------|:------:|:---:|:-:|:--------:|:---------:|:------------:|---------------------|
| High | 18 | 110 | 0.999 | 0.32 | 99.1% | 59.1% | ≥30% exact matches OR MAE<2pp + dir≥95% |
| Medium | 4 | 29 | 0.952 | 3.36 | 100% | 6.9% | MAE 2--5pp + dir≥90% |
| Low | 5 | 61 | 0.862 | 5.36 | 83.6% | 0% | Direction<85% OR scale anomaly |

The tier separation is striking: high-confidence papers have 59% zero-error observations (proving exact value matches), while low-confidence papers have zero exact matches and only 84% direction agreement. This gradient emerges from the data alone---it is not imposed by any auditor.

### 3.7.2 Corroborative LLM Audit

To provide qualitative context for the 9 non-high-confidence papers, we conducted per-paper diagnostic audits using Claude Sonnet 4.6. The LLM auditor received source PDFs and reference-standard rows but had no access to the programmatic classifications or aggregate statistics. Its role was explanatory, not classificatory: identifying *why* specific papers showed high error, not deciding *whether* they were high-error (which the programmatic classifier already determined).

The LLM audit identified the following root causes for the 9 papers the programmatic classifier flagged:

- **3 aggregation-level mismatches** (Kocira 2020, Kocira 2018, Procházka 2015): GT stored per-year data while the pipeline extracted multi-year averages from summary tables. The programmatic classifier independently flagged these via direction agreement <85%---a signature of temporal aggregation mismatch.
- **3 GT attribution errors** (Mondal 2013, Pohl 2019, Glosek-Sobieraj 2018): GT values could not be reconciled with the published tables. The programmatic classifier placed Mondal and Glosek-Sobieraj in the *high*-confidence tier (MAE 0.84 and 0.46 pp respectively), suggesting that despite GT attribution issues, the pipeline's values closely matched the actual PDF.
- **2 wrong PDFs** (Abdel-Mawgoud 2010, Alabdulla 2019): the file contained a different paper than the GT expected. The programmatic classifier correctly flagged Abdel-Mawgoud as low-confidence (MAE = 9.01 pp, irregular scale ratios).
- **2 GT source mismatches** (Kuisma 1989, Pramanick 2016): GT rescaled values by undocumented factors. Programmatic classifier placed both in medium tier (MAE 3--5 pp).

In every case except one product-selection omission (Kocira 2019), the pipeline's extracted values were verified as correct against the source PDF. The key epistemic point is that this LLM-generated explanation is *corroborative*, not *constitutive*: the statistical claims in Table 1 and Table 5 depend only on the programmatic classification, which any researcher can reproduce from the validation CSV.

**The Precision Challenge.** The pipeline's exhaustive extraction approach yielded approximately 2,200 total candidates for these 27 papers, meaning the 200 matched observations represent an initial precision rate of ~9%. This recall-optimized architecture shifts the human bottleneck from manual data entry to analytical data filtering.

The methodological lesson is that validation metrics (r, MAE, ICC) are composite measures of *both* pipeline accuracy *and* validation infrastructure quality. Programmatic confidence classification provides a reproducible, non-circular method for decomposing these components.

## 3.8 Aggregate Effect Reproduction

Across all three datasets, the pipeline reproduced aggregate meta-analytic effects with high fidelity (Table 1):

- **Loladze**: GT mean = -4.91%, extracted = -4.96%, diff = 0.05 pp
- **Hui**: GT mean = 49.61%, extracted = 49.72%, diff = 0.12 pp
- **Li**: GT mean = 12.70%, extracted = 11.86%, diff = 0.84 pp

For Loladze, per-element mean effects were closely reproduced for key elements: Zn (0.03 pp diff), Ca (0.24 pp), Mg (0.30 pp), P (0.67 pp), K (0.69 pp). Larger discrepancies occurred for trace elements with small samples: Mn (6.37 pp), B (5.46 pp), Na (4.02 pp).

Across all three datasets, the absence of meaningful systematic bias was confirmed by paired t-test (Loladze: t = −0.08, p = 0.93, d = −0.003; Hui: t = 1.29, p = 0.20, d = 0.072; Li all: t = −2.68, p = 0.008, d = −0.189). While the Li full-set paired t-test reaches significance, the effect size (d = −0.189) remains below the conventional "small" threshold (0.20), and the aggregate difference (0.84 pp) is negligible for meta-analytic pooling. On the programmatic high-confidence subset (110 observations), d = −0.055 and the aggregate difference shrinks to 0.04 pp, confirming that the significant t-test on the full set is driven by the low-confidence papers with known alignment issues. Errors are predominantly random and cancel in the aggregate, the property essential for meta-analysis.

## 3.9 Paper-Level Accuracy

Papers were classified into accuracy tiers based on MAE (Figure 3):

- **Excellent** (MAE < 5%): 22 papers (48%), including 6 papers with MAE < 0.1%
- **Good** (5--10%): 10 papers (22%)
- **Fair** (10--20%): 13 papers (28%)
- **Poor** (> 20%): 1 paper (2%)

Thus, 70% of papers achieved Good or Excellent accuracy, and 98% were at least Fair. The single Poor paper (Niu et al. 2013, MAE = 58%) had only 2 observations under atypical phosphorus-deficient conditions.

Excellent-tier papers were overwhelmingly consensus-dominant: 82% had >50% consensus-confirmed observations. Poor and Fair papers had higher vision dependence, confirming that the presence or absence of multi-model consensus is an actionable indicator of extraction quality.

## 3.10 Methodological Concordance vs. Extraction Accuracy: The `info`-Column Analysis

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

*   **1. CO2 × co-treatment factorial design (most impactful category).** Loladze frequently extracted the CO2 effect separately within each level of a co-treatment (ozone, nitrogen, water stress, potassium), rather than computing the CO2 main effect averaged across co-treatment levels. For papers where CO2 × co-treatment interactions are biologically real, these choices produce substantially different lnRR values. Fangmeier et al. (2002, 32 matched GT pairs, MAE = 8.0%) provides the clearest illustration and is examined in detail below.
*   **2. Multi-year study (year selection).** For experiments reporting results across multiple growing seasons, Loladze selected a specific year (`info = "2000"`, `"1999"`, etc.) rather than pooling across years. Our pipeline extracts averages across reported years when individual year data are available in a table, or extracts the final harvest summary. For elements whose CO2 effect changes direction across seasons, year selection can determine the sign of the effect.
*   **3. Multi-cultivar study (cultivar selection).** For experiments crossing CO2 with multiple cultivars or genotypes, Loladze sometimes selected a specific cultivar (`info = "NC-R"`) rather than averaging. Our pipeline averages across cultivars when means are available by cultivar. For papers where CO2 × cultivar interactions produce divergent responses, cultivar averaging attenuates effects that Loladze captured at the per-cultivar level.
*   **4. Multi-site study (site selection).** For multi-site FACE experiments, Loladze sometimes used data from a single site (`info = "Duke"` or `"ORNL"`) representing the most relevant or best-documented site. Our pipeline uses data from whichever table presents pooled-across-site means, or averages across sites if per-site data are in the same table.
*   **5. Sampling date selection.** For experiments with multiple measurement dates, Loladze standardized on the final harvest date, while our pipeline extracts from whichever table presents the primary numeric data. When the final harvest data are in figures rather than tables (as in Huluka 1994), the pipeline must extract from the only available numeric table, which corresponds to an earlier sampling date.

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

Systematic error attribution across all 635 observations reveals that sub-selection mismatches account for 200 observations (31%) but 73% of total error. Removing these alignment artifacts yields a counterfactual accuracy of r = 0.915 and MAE = 3.1 pp (435 observations). The 13 observations classified as "large unexplained" (true extraction failures with >20 pp error) account for only 9% of total error. Removing both mismatches and true failures yields r = 0.976 and MAE = 2.1 pp---demonstrating the pipeline's reading accuracy when freed from alignment noise.

The well-aligned subset of 374 observations (34 papers) from the original matching achieved r = 0.876 and MAE = 4.3%, compared to r = 0.669 and MAE = 7.9% for the full dataset. The headline Loladze metrics should therefore be understood as conservative lower bounds on extraction accuracy: they simultaneously penalize the pipeline for reading errors *and* for not replicating Loladze's unstated sub-selection choices. Crucially, true generative hallucinations---where the model invents numbers not present in the text---were exceptionally rare. Errors were predominantly misattributions (binding drift across rows/columns) or methodological mismatches.

**Treatment/control swap analysis.** No systematic swaps were detected across the dataset. One individual case was flagged: the Sulfur 2009 main-effect row in Fernando et al. (2012a) received a `LIKELY T/C SWAP` warning from the post-processing gate (our_effect = +4.3% vs. Loladze GT = −9.5%), consistent with a transposition of the TOS2 sub-row values before averaging. The gate operated as designed — it flagged the anomaly — but did not auto-correct (tc_swaps_corrected = 0), which was intentional: auto-correction was tested and found harmful overall (r declined from 0.509 to 0.209 when applied globally, because elements like Fe and Mn legitimately increase under elevated CO2). This represents a manual error rate of approximately 1 in 635 observations (0.2%).

## 3.11 Formal Agreement Statistics

### 3.11.1 Equivalence Testing

Two one-sided tests (TOST) confirmed formal statistical equivalence between pipeline and reference-standard extraction at ±2 pp for both the Loladze (p < 0.001; 90% CI: −1.07 to 0.97 pp; Figure 4) and Hui datasets (p < 0.001; 90% CI: −0.03 to 0.27 pp). The 90% CIs fell entirely within equivalence bounds for both datasets. On the Li scale-harmonized matching (200 observations), TOST confirmed equivalence at ±2 pp (p < 0.001); the programmatic high-confidence subset (110 observations) achieved equivalence at ±2 pp (p < 0.001). All datasets achieved equivalence at ±5 pp.

### 3.11.2 Bland-Altman Agreement

Bland-Altman analysis (Figure 5) showed negligible systematic bias for all three datasets. Loladze: mean difference = -0.05 pp (95% CI: -1.26 to 1.16 pp), with 95% limits of agreement from -30.6 to 30.5 pp; no proportional bias (r = -0.035, p = 0.38). Hui (improved matching, 319 obs): mean difference = +0.12 pp, with limits of agreement from −3.1 to +3.3 pp; no proportional bias. Li (scale-harmonized, 200 obs): mean difference = −0.84 pp, with limits of agreement from −9.5 to +7.8 pp. 

The wide limits of agreement reflect observation-level variability; aggregate-level accuracy is substantially better. Note that TOST ±2 pp equivalence bounds (Section 3.11.1) and Bland-Altman ±30 pp limits are not contradictory: TOST tests whether the *mean* difference is negligibly small (aggregate bias), while Bland-Altman limits describe the 95% range of individual observation differences (observation-level scatter). A pipeline can achieve zero mean bias while individual observations scatter widely. However, in meta-analysis, wide observation-level errors do not simply cancel out; they artificially inflate heterogeneity estimates ($\tau^2$ and $I^2$), which can reduce statistical power for meta-regression and subgroup analyses. Therefore, this pipeline is currently better suited for main-effect pooling than for highly sensitive, observation-level meta-regressions.

### 3.11.3 Intraclass Correlation

ICC was **moderate** at the observation level (ICC(3,1) = 0.669, 95% CI: 0.623--0.710; Koo & Li, 2016 classify 0.50–0.75 as moderate) and excellent at the paper level (ICC = 0.838). The discrepancy between observation- and paper-level ICC reflects the Granularity Barrier: observation-level agreement is diluted by sub-selection mismatches, while paper-level aggregation averages out these within-paper disagreements. The paper-level ICC is consistent with published human inter-rater reliability values reported in systematic review data extraction literature (Mathes et al., 2017; Schmidt et al., 2025).

### 3.11.4 Bootstrap Confidence Intervals

**Table 2. Bootstrap CIs for key validation metrics (10,000 percentile resamples; paper as resampling unit; Hui: 21 papers, Loladze: 46 papers, Li harmonized: 27 papers, Li prog. high: 18 papers).**

| Metric | Loladze (n=635) | Hui (n=319) | Li harmonized (n=200) | Li prog. high† (n=110) |
|--------|:---------------:|:-----------:|:---------------------:|:----------------------:|
| Pearson r | 0.669 [0.545, 0.834] | 0.999 | 0.951 | **0.999** |
| MAE (pp) | 7.9 [7.0, 9.1] | 0.43 | 2.30 | **0.32** |
| Direction | 84.5% [81.4, 87.1] | 99.7% | 93.0% | **99.1%** |
| Mean diff (ext−GT)‡ | −0.05 pp [−1.26, 1.16] | +0.12 pp | −0.84 pp | −0.04 pp |
| Cohen's d | −0.003 | 0.072 | −0.189 | −0.055 |

† Li clean (Programmatic High-Confidence Subset): 18 papers classified as high-confidence by the programmatic data-property algorithm (Section 2.6.1); excludes 9 papers with direction failures, scale anomalies, or high MAE. No LLM judgment used in classification. Statistics computed on `validation_matches_improved.csv` using scale-invariant effect-% matching.

‡ Mean diff (ext−GT) reported as Bland-Altman signed mean difference (see Section 3.11.2); 95% CI from within-sample t-distribution of paired differences. Bootstrap CIs for this metric are unreliable when the mean difference is near zero (the CI of |mean diff| collapses below the point estimate). The signed BA CIs are the appropriate summary.

## 3.12 Sensitivity Analyses

### 3.12.1 Leave-One-Paper-Out

LOPO analysis showed the full MAE (7.95%) was stable: LOPO range was 6.8--8.3%. The most influential paper (Natali 2009, MAE = 19.1%) improved aggregate MAE by 1.16 pp when removed.

### 3.12.2 Leave-One-Element-Out

No single element drove results. Removing trace metals improved MAE by 0.20--0.37 pp; removing major elements worsened it by 0.22--0.43 pp.

### 3.12.3 Matching Tolerance Sensitivity

For Hui, tightening tolerance from 0.15 to 0.10 had minimal impact on accuracy (r remained >0.99), confirming that the high-quality matches are robust to matching parameters. For Li, tolerance sensitivity was higher: at 0.10 (n = 112), r = 0.889; at 0.30 (n = 163), r = 0.453. Aggregate effect differences remained stable across all thresholds (<1.1 pp for Li).

## 3.13 Consensus vs. Single-Model Comparison

To isolate the consensus mechanism's contribution, we compared each model's solo extraction on a fixed scope of 322 observations matchable by all sources:

| Method | MAE (%) | Pearson r | Direction (%) |
|--------|---------|-----------|---------------|
| Kimi solo | 4.10 | 0.903 | 88.6 |
| Consensus | 4.54 | 0.886 | 89.2 |
| Gemini solo | 5.53 | 0.843 | 85.1 |
| Claude solo | 6.29 | 0.742 | 85.4 |

On a fixed observation set, consensus does not improve per-observation accuracy over the best single model (Kimi). The consensus mechanism's value lies in identifying reliable observations rather than improving per-observation accuracy on a fixed set. When two models agree, the observation is likely correct; when they disagree, it warrants review. No single model dominates across all elements (Kimi best for 13/20, Gemini 5/20, Claude 2/20), so the multi-model approach provides robustness against model-specific failures.

On the original 46-paper extraction, the consensus pipeline matched 560 observations to reference standard versus Kimi's 486 (15% increase), representing modest coverage gains from complementary model outputs.

## 3.14 Cost

The median per-paper API cost was approximately $0.24 (mean: $0.37; range: $0.12–$3.50; Claude: ~$0.28, Kimi: ~$0.04, Gemini: ~$0.05). The mean exceeds the median because the distribution is right-skewed: simple text papers cost ~$0.12–$0.18, while complex factorial designs with 200+ extracted observations (e.g., Chen et al. 2021: 299 extracted rows, $3.50) dominate the tail. This variability reflects genuine differences in paper complexity rather than tunable parameters. Furthermore, the Gemini tiebreaker is invoked for only 22% of papers, reducing per-paper cost by approximately 30% compared to always running three models simultaneously. Processing 46 papers cost approximately $17 over 6 hours of wall-clock time, primarily constrained by concurrent API rate limits across Anthropic, Moonshot, and Google. This compares to an estimated 184 hours of manual extraction at $30/hour ($5,520). However, this baseline must explicitly account for data filtering and metadata verification time. The pipeline solves the extraction bottleneck but creates a filtering bottleneck. In a triage deployment where human reviewers check only flagged observations (20--30% of the total, varying by task structure and dataset) and perform necessary analytical filtering, the human review and data filtering time would be approximately 45--55 hours rather than 184, a 70--75% time reduction.

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
| **This study** | **Plant sci.** | **Sonnet 4 + Kimi + Gemini** | **94** | **Yes (3)** | **r = 0.67-0.999** | **MAE 0.43-7.9 pp** | **TOST p<0.001** |

¹ Poser et al. distinguish "true errors" (content errors, 1.48%) from total errors including formatting issues (6.7%).

Our system differs from prior work in four key respects. First, we target quantitative extraction of continuous means and effect sizes (Li et al.'s Tier 3) rather than categorical study characteristics, where prior systems achieved 90--96% accuracy. Even specialized numerical extraction systems acknowledge the difficulty: Kataoka et al. (2026), testing o3 (OpenAI's reasoning model) on insomnia RCTs, concluded that "numeric variable extraction performed poorly" and "the performance for numeric DE was still inadequate." Their best system (o3) reached 75.3% accuracy on numeric variables; our Hui result of r = 0.999 exceeds this, though on a different task structure. Second, we use inter-model agreement as a quality *predictor* rather than solely an accuracy booster, enabling confidence-stratified output rather than aggregate accuracy reports. Third, we are the only system validated on plant science data. Fourth, we apply formal equivalence testing (TOST, ICC, Bland-Altman), providing a statistical framework for comparing pipeline output to human extraction rather than just reporting accuracy percentages.

The accuracy picture is more nuanced than prior work suggests. Clark et al.'s (2025) systematic review found data extraction error rates of 4--31% (median 14%) across existing systems, with GenAI performing well on "easier" data such as publication years or countries, "but for more complex data, such as outcome data or intervention descriptions, GenAI tended to perform less effectively." Our MAE of 0.43 pp (Hui) and 7.9 pp (Loladze) falls within this range, with the Hui result substantially better than the field median, reflecting the advantage of single-element structured extraction. As Section 3.10 documents, the Loladze MAE of 7.9% is itself a conservative estimate: separating methodological concordance failures from reading errors reduces the effective extraction MAE to approximately 4.3% for the aligned subset. Jansen et al. (2025) found only 26--36% accuracy for effect-size variables and ~17% for standard deviations across 22 meta-analyses, results that contextualise our Loladze MAE of 7.9% (a different metric but directionally consistent). Peng et al. (2025) found 69--72% accuracy for means and SDs from clinical randomized controlled trials using Claude 3.5; Yun et al. (2024) found 48.7% exact match for GPT-4 on continuous RCT outcomes; Gougherty & Clipp (2024) reported 23.8% for quantitative ecological data. Our Hui result (r = 0.999, MAE = 0.43 pp) exceeds these benchmarks substantially, reflecting the advantage of single-element structured extraction. Our Loladze and Li results are more representative of multi-element complex extraction and align with the field when viewed at the ±10% threshold (74% and 91% within 10 pp, respectively).

## 4.2 The Concordance Signal: Khan's Principle Extended

Khan et al. (2025) established the core insight motivating our design: when two independent LLMs give concordant responses, the hallucination rate drops to 0.25%; when responses are discordant, the hallucination rate rises to 26--41%. This nearly 100-fold difference means that concordance status is a better predictor of reliability than any model-specific accuracy score. Khan applied this principle to categorical binary extraction (does a study meet inclusion criterion X?), demonstrating 87% concordance rate and 94% accuracy on concordant items.

We extend this principle in two directions. First, we apply it to *continuous numerical* extraction, where the reliability problem is more acute: wrong numbers can bias meta-analytic estimates without triggering any obvious flag, whereas a wrong categorical label is often detectable by inspection. Second, we develop the concordance signal into a full confidence-stratification system rather than a binary accept/reject filter. Our results confirm that the concordance principle extends to quantitative extraction: consensus-dominant papers achieve MAE = 4.3% versus 11.2% for vision-dependent papers (2.6× improvement), with 95% of large errors concentrated in the flagged minority.

Poser et al. (2026) independently validated three-model consensus for clinical data extraction, achieving 1.48% true-error rate for structured clinical variables, further confirming that multi-model consensus reduces errors. However, their study focused on categorical clinical fields rather than continuous numerical outcomes. Jansen et al. (2025), testing a majority-vote ensemble of 8 LLMs on 2,179 studies, found that ensemble voting improved performance over individual models and that variable type was the dominant predictor of accuracy, consistent with our finding that `has_complex_stats` predicts consensus failure more strongly than paper-level features. Their conclusion that accuracy depends "most between variable, less between systematic reviews, and least between LLMs" means that inter-model agreement functions as a variable-difficulty detector: hard variables produce disagreement, easy variables produce consensus. Our contribution is demonstrating that this signal works for *continuous* numerical extraction in *complex multi-element* agricultural data, where it predicts accuracy at the paper level with sufficient reliability to support an automated triage workflow.

Tan & D'Souza (2026) provide mechanistic insight into why single-model extraction fails systematically. Their four structural failure modes — role confusion (treatment/control swaps), binding drift (cross-row value attribution), multi-instance compression, and error amplification — each produce characteristic patterns that a single model cannot self-detect. Multi-model consensus addresses these failures by design: a treatment/control swap by one model will not be matched by another model, raising a discordance flag. Binding drift produces inconsistent numerical values across models. Only multi-instance compression and error amplification could propagate across models if both are misled by the same structural feature, a residual limitation.

**Reading Accuracy versus Relevance: a critical distinction.** Consensus validates *reading accuracy* — was the number extracted correctly from the text? It does not validate *relevance* — is this the number the meta-analyst intended to select? The 5% initial extraction precision (110 GT-matched observations from over 2,200 extracted candidates in the programmatic high-confidence Li analysis) is a relevance problem, not a consensus failure: the pipeline correctly reads all detectable numbers, then consensus confirms which readings are reliable. The downstream triage step — filtering to yield-related outcomes and matching to the meta-analyst's intended stratum — is a separate task. High-confidence observations are reliably accurate readings; they may still require analyst judgment about relevance to a specific sub-condition.

## 4.3 The Three-Barrier Model: What Each Dataset Measures

The three datasets isolate distinct extraction challenges.

**The Reading Barrier (Hui dataset).** With a single element (Zn), standardized units (mg/kg), and a uniform wheat biofortification context, r = 0.999 and MAE = 0.43 pp measure one thing: can the pipeline read numbers correctly from tables? With 81% of observations achieving zero error and five papers producing perfect extractions (including Rehman 2018 with 56 observations and Zou 2012 with 67 observations spanning a 7-country international trial), the answer is unequivocally yes. The Reading Barrier is solved for standardized tabular formats.

**The Granularity Barrier (Loladze dataset).** The Loladze dataset adds 14 elements, CO2 factorial designs, scanned PDFs, and multi-tissue tables. The headline r = 0.669 appears modest, but the `info`-column analysis (Section 3.10) reveals that this number conflates two fundamentally different phenomena: table-reading errors and analytical sub-selection disagreements. Systematic decomposition shows that 73% of total error originates from sub-selection mismatches---cases where the pipeline and Loladze both read values correctly but from different factorial arms. Removing these alignment artifacts yields r = 0.915 and MAE = 3.1 pp. The Granularity Barrier is therefore not an extraction problem but a *validation* problem: it measures the difficulty of proving the extractor works when the human meta-analyst's unstated sub-selection rules cannot be replicated without explicit documentation. For a researcher running the pipeline for their own meta-analysis, sub-selection rules are specified in the configuration file, rendering the Granularity Barrier largely irrelevant.

**The Provenance Barrier (Li dataset).** The Li dataset exposes a third barrier: the quality and homogeneity of the reference standard itself. This barrier is not about whether the extractor works---it is about whether the validation infrastructure can detect that it works. The naive matching (r = 0.453) was entirely an artifact of the original matching algorithm's failure to handle unit-scale differences; simply harmonizing units raised the correlation to r = 0.951 without changing any extracted values. The residual gap between r = 0.951 and r = 0.999 (programmatic high-confidence subset) is attributable to papers with observable data-quality signals: direction agreement below 85%, high MAE, or inconsistent scale ratios.

**Addressing the circularity concern.** A natural objection to error decomposition is that "LLMs auditing LLMs" creates circular validation. We address this explicitly. The programmatic high-confidence subset (18 papers, 110 observations, r = 0.999) is defined by a deterministic algorithm using only three observable signals from the matched data: zero-error fraction, direction agreement, and MAE. No LLM classifies which papers are "correct." An independent LLM audit corroborates by identifying root causes (wrong PDFs, aggregation mismatches) for the 9 excluded papers, but the statistical claims do not depend on this audit. The 85% agreement between the two independent classification methods---one algorithmic, one LLM-based---confirms that the high-quality subset is genuine. The Provenance Barrier is thus not an AI problem; it is a validation infrastructure problem that inflates apparent error and leads to underestimation of AI extraction capability.

Before deploying the pipeline on a new topic, researchers can anticipate: (A) if extraction targets a single outcome in standardized units (Hui profile), expect near-perfect reading accuracy (r > 0.99); (B) if extraction targets complex factorial designs (Loladze profile), expect MAE of 3--5 pp for aligned observations, with apparent errors dominated by sub-selection disagreements that are preventable through configuration-file specification; (C) if validating against an external reference standard (Li profile), expect that headline statistics will reflect reference-standard quality as much as pipeline quality---scale-harmonized matching and paper-level auditing are essential. In all three cases, aggregate meta-analytic effects are reproduced to within fractions of a percentage point regardless of individual-observation noise.

**The Precision Challenge.** While this "information surplus" prevents data loss, a 4.5% initial precision rate constitutes a significant limitation in user experience. Returning thousands of candidates for a hundred target observations shifts the bottleneck from data extraction to data filtering. In practice, this is a limitation that requires robust downstream UI/UX or stricter JSON pre-filtering to solve. Researchers who need higher initial precision for a tightly-defined question must specify explicit stratum filters in the configuration JSON, trading recall for precision at the configuration level rather than relying solely on post-hoc filtering.

## 4.4 Benchmark Bias and the Human Comparison Problem

All AI extraction benchmarking, including ours, faces what Gartlehner et al. (2025) have termed "benchmark bias": the tendency to evaluate AI systems against human-extracted reference standards that themselves contain errors. Research on human extraction reliability indicates that up to 63% of study reports contain at least one extraction error even when extracted by trained researchers, a baseline rarely accounted for when setting accuracy thresholds for AI systems (Gartlehner et al., 2025; Mathes et al., 2017). Gartlehner et al. (2025) illustrate the problem concretely: in their proof-of-concept study, Claude 2 identified 21 minor errors in the human reference standard that would otherwise have gone undetected; on inspection, these proved to be corrections.

This has direct implications for interpreting our results. Our Loladze reference standard was extracted by a single author without a reported dual-extraction protocol. Our measured MAE of 7.9% therefore combines pipeline error, reference-standard error, and methodological concordance failures. The `info`-column analysis (Section 3.10) resolves the concordance component: 73% of total error is attributable to sub-selection mismatches, not reading failures. The counterfactual analysis (removing alignment artifacts) yields MAE = 3.1 pp and r = 0.915. Combined with the 22 papers achieving Excellent accuracy (MAE < 5%, including 6 with MAE < 0.1%), and the Li finding that the pipeline was verified as correct against source PDFs in all categories except one product-selection omission, the effective pipeline reading error is substantially below the headline MAE values. Given that human single-extractor error rates range from 8% to 17% (Buscemi et al., 2006), the pipeline's actual reading error rate is likely competitive with human extraction.

**Proposing the `info`-column methodology and programmatic classification as validation standards.** The decomposition of validation error into true extraction failures versus methodological sub-selection differences is a critical requirement for evaluating automated extraction in complex domains. We propose two complementary approaches: (1) future AI-SR benchmark datasets should explicitly document the analytical choices made during human extraction (e.g., specific factorial arms, harvest dates, or cultivars selected) rather than merely providing final numerical values---the Loladze `info`-column provides an effective template; and (2) when such documentation is unavailable, programmatic confidence classification using observable data properties (zero-error fraction, direction agreement, MAE thresholds) provides a reproducible, non-circular alternative to LLM-based audit. Without such explicit decomposition, validation studies will systematically underestimate AI capability by penalizing models for making defensible, but different, analytical choices than the human reviewers.

More broadly, the dual-independent-extraction-followed-by-consensus protocol described by Buscemi et al. (2006) as the gold standard is expensive enough that most meta-analyses never implement it. Our paper-level ICC of 0.838 is consistent with human inter-rater reliability values reported in the data extraction automation literature (Mathes et al., 2017; Schmidt et al., 2025), suggesting that the pipeline performs comparably to a human second extractor without the labor cost. Jensen et al. (2025) found that ChatGPT-4o as a second rater had a 5.2% false data rate versus 17.7% for human single extractors, consistent with the possibility that AI extraction is already less error-prone than single-human extraction for some data types.

## 4.5 Engaging with "Not Yet Ready": A More Nuanced Assessment

Lieberum et al. (2025) concluded from their scoping review that LLMs are "not yet ready for use" in systematic review data extraction, noting that only 11% of LLM-SR studies even address data extraction and that quantitative accuracy remains insufficient for complex tasks. Clark et al. (2025) reached the same conclusion: "The current evidence does not support GenAI use in evidence synthesis without human involvement or oversight. However, for most tasks other than searching, GenAI may have a role in assisting humans with evidence synthesis." Cao et al. (2025) similarly found that fully automated systematic reviews remain out of reach. We agree with this conclusion for fully automated deployment but argue for a more nuanced position: the pipeline is ready for triage deployment, with human oversight concentrated where it is needed.

The "not yet ready" conclusion implies a binary: either the system works well enough to replace humans, or it does not. Our results suggest a different question: whether AI can triage extraction work intelligently, focusing human effort on the observations where it is needed. On this criterion, our results are more encouraging. The pipeline achieves MAE = 4.3% on consensus-dominant observations and correctly identifies 95% of large errors as low-confidence. Human reviewers who check only the flagged observations (20--30% of total, depending on task structure), working at reduced effort compared to de novo extraction, can achieve the accuracy of full dual extraction at approximately 70--75% lower cost.

Lieberum et al.'s concern is particularly valid for domains where accuracy is critical at the individual-observation level (clinical dosing, safety data) and where errors could directly harm downstream users. For agricultural meta-analysis, where the outputs are population-level estimates of agronomic effects rather than individual patient treatments, the tolerance for observation-level variability is higher, provided that errors are random rather than systematic. Our demonstration that Cohen's d ≈ 0 across all three datasets (|d| < 0.20; programmatic high-confidence |d| < 0.06) and that errors cancel in the aggregate addresses this concern directly.

The methodological concordance finding (Section 3.10) adds an important nuance to the "not yet ready" assessment: a portion of what current validation studies report as AI extraction error is actually methodological sub-selection disagreement between the AI and the human meta-analyst, of a kind that a human second extractor without explicit sub-selection instructions would also show. The true extraction error rate, stripped of concordance failures, is lower than reported MAE values suggest. Current MAE benchmarks therefore understate AI capability.

We propose a position between "not yet ready" and "ready for full automation": ready for supervised deployment as a first extractor, analogous to how pilot extraction is currently used in manual meta-analysis. The pipeline handles 70--80% of observations at high confidence; human reviewers handle the rest. This matches how most large meta-analysis teams already work, with a senior researcher reviewing junior extractors' output, but substitutes AI for the junior extractor role at a fraction of the cost.

## 4.6 The Triage Workflow in Practice

Based on our results, we propose a triage workflow for AI-assisted meta-analysis:

1. **Auto-validated observations** (high confidence, ~70--80% of pipeline output): Two or more models agree. These observations achieve MAE ≈ 3--5% and can be used directly with minimal human review. For scoping analyses assessing meta-analysis feasibility, these observations alone may suffice. For publication-quality analyses, we recommend a **5% random spot-check** of high-confidence observations to verify absence of systematic model bias (e.g., a shared misinterpretation of a non-standard unit or table layout that both models agree on incorrectly).

2. **Flagged observations** (medium/low confidence, ~20--30% of output): Single-model or vision-only extraction. These are retained in the dataset but tagged for human verification. Because the pipeline outputs the source table number, page number, and surrounding context alongside each extracted value, the reviewer can immediately locate and verify the flagged number without re-reading the entire paper. This targeted verification requires approximately 2 minutes per observation compared to 10 minutes for full de novo extraction.

3. **Rejected papers** (zero consensus, ~3% after tiebreaker): No model could extract usable data. These require full manual extraction.

4. **Concordance review** (for complex factorial papers): The pipeline's extracted values and the meta-analyst's sub-selection rules should be compared. For papers where the `info`-field-equivalent information (co-treatment arm, date, cultivar) is not pre-specified in the configuration, the pipeline defaults to main effects and averages, which may differ from the intended sub-selection. This review step replaces the methodological concordance failures identified in Section 3.10 with a configuration-time specification of intent.

This workflow offers substantial time savings even without full automation. The flagged fraction varies by task: single-element extraction (Hui-type) flagged approximately 31% of observations, while complex multi-element extraction (Loladze-type) achieved consensus on ~94% of observations, flagging only ~6%. Even with these savings, the pipeline solves the extraction bottleneck but creates a filtering bottleneck; users must account for the data filtering and metadata verification time required to sift through the exhaustive candidate rows. Assuming 20--30% human review at ~2 minutes per flagged observation versus ~10 minutes for de novo extraction, the total human time for a 46-paper meta-analysis drops from ~184 hours to approximately 45--55 hours, a 70--75% reduction. This estimate explicitly includes the cognitive cost of analytical filtering---sifting through the pipeline's exhaustive candidate rows (e.g., ~2,200 candidates for 200 target observations in Li 2022) to identify the relevant subset. In practice, this filtering is largely automated by downstream JSON pre-processing (matching on outcome variable, tissue type, and treatment arm), reducing the human task to verifying pre-filtered candidates rather than scanning raw output. Nevertheless, the filtering step represents a real time cost that partially offsets the extraction savings. The pipeline also provides a complementary extraction for quality assurance: paper-level ICC of 0.838 is consistent with published human inter-rater reliability values (Mathes et al., 2017; Schmidt et al., 2025). Methodological independence between pipeline models is not claimed, as all three models received identical extraction prompts; the ICC comparison is therefore to human inter-rater reliability as a calibration benchmark, not as evidence of independent replication.

For rapid scoping reviews assessing the feasibility of a new meta-analysis, the high-confidence observations alone may be sufficient. For publication-quality meta-analyses, the triage workflow ensures that every observation is either consensus-validated or human-reviewed. Peng et al. (2025) reached the same practical conclusion in sleep medicine: "systematic review authors could utilize AI tools as second reviewers for data extraction to achieve accuracy comparable to human reviewers, yet with greater efficiency and reduced labor." Our results extend this finding to plant science and quantify what "comparable to human" means: paper-level ICC = 0.838, consistent with human inter-rater reliability values for data extraction tasks (Mathes et al., 2017; Schmidt et al., 2025).

## 4.7 Limitations

1. **Development vs. holdout validation.** The Loladze dataset was used during pipeline development; accuracy on this dataset may be optimistic. The Hui independent holdout validation (r = 0.999) provides the most honest estimate for straightforward extraction tasks. The Li dataset (r = 0.951 with scale-harmonized matching; r = 0.999 on the 18 programmatic high-confidence papers) demonstrates strong cross-domain generalization, though the progression from naive (r = 0.453) to harmonized (r = 0.951) matching highlights that validation methodology significantly affects apparent accuracy.

2. **Wide observation-level limits of agreement.** Bland-Altman limits span ±30 pp for Loladze and ±9 pp for Li (scale-harmonized; substantially narrower on the programmatic high-confidence subset), meaning individual observations may have large errors even though the aggregate is unbiased. The pipeline should be used for aggregate pooling, not single-observation precision.

3. **Vision extraction quality.** Vision-dependent papers achieved MAE = 11.2%, substantially worse than consensus-dominant papers (MAE = 4.3%). The pipeline's confidence flagging mitigates this risk but does not eliminate it. Papers with scanned tables or figure-only data remain challenging.

4. **Element capture rate (83%).** Approximately 17% of reference-standard observations were not matched, potentially introducing selection bias.

5. **Variance extraction (67% capture)** lags behind means extraction (>98%), reflecting inconsistent variance reporting in agricultural journals, a known problem in the field (Nakagawa et al., 2023). For the ~33% of observations with missing variance, users may employ standard imputation strategies: borrowing the median SD from other studies in the same outcome category, using hot-deck imputation from studies with similar sample sizes, or falling back to unweighted analysis when imputation is infeasible (Nakagawa et al., 2023).

6. **Proportional bias in heterogeneous data.** The Li dataset showed significant proportional bias (r = 0.355, p < 0.001): extraction errors grew with effect magnitude. This likely reflects unit conversion ambiguity for large yield responses rather than systematic extraction failure. Unit-normalization (scale-factor matching) accounts for a portion of Li clean-subset matches; a 100× scale discrepancy is consistent with legitimate unit conversion (mg/100g vs. mg/g) but could in principle indicate extraction of a differently-scaled variable. Per-paper audit of all 16 clean-subset papers found no instances of wrong-variable extraction; nevertheless, large scale-factor matches should be flagged for human inspection in production use.

7. **Validation scope.** All three datasets are from plant science. Generalization to clinical trials or other domains is not tested, though the pipeline is domain-agnostic by design. Extension to animal ecology (e.g., aquatic species response studies) is planned; the configuration-driven design transfers without code changes, but domain-specific outcome variable definitions and tissue taxonomies must be specified.

8. **Model versioning and reproducibility.** Results were obtained with specific model versions (Claude Sonnet 4.6, Kimi K2.5, Gemini 2.5 Pro for vision, Gemini 3 Flash for text tiebreaker). LLM providers update models without notice. Cross-run stability ICC of 0.9996 applies within a version; across versions, drift should be expected. Schmidt et al.'s (2025) living review update specifically identified reproducibility as an emerging concern with LLM-based extraction: "LLMs showed a trend of decreasing quality of results reporting, especially quantitative results such as recall and lower reproducibility of results."

9. **Benchmark bias.** As discussed in Section 4.4, reference-standard extraction errors and methodological sub-selection choices both confound our accuracy estimates. True pipeline reading error may be lower than reported MAE.

9b. **Residual circularity in qualitative claims.** While the programmatic classifier eliminates LLM dependence for all statistical claims (r, MAE, ICC, TOST), the qualitative explanations for *why* specific papers are low-confidence (e.g., "the GT attributed maize observations to a mung-bean-only paper") still originate from LLM audits. Human spot-checking confirmed these explanations for all 9 excluded papers, but full independent human verification of every claim was not conducted. The programmatic classifier mitigates this by ensuring that no paper's inclusion or exclusion from the high-confidence subset depends on LLM judgment.

10. **Prompt sensitivity.** Extraction quality depends on prompt templates. While formal ablation was not conducted, informal testing revealed that the models were generally robust to minor prompt variations, though highly sensitive to the exact phrasing of the JSON schema keys and inclusion/exclusion definitions. Full prompts are provided in supplementary materials.

11. **Methodological concordance decomposition limitations.** The `info`-column decomposition in Section 3.10 was conducted for Loladze but relies on the unusual feature that this reference standard's `info` field documents sub-selection choices. Most meta-analysis databases do not include such documentation, making concordance decomposition difficult or impossible without re-reading all papers. The approach is not easily generalizable without comparable documentation in the reference standard.

12. **Correlated model failures.** Multi-model consensus assumes independent error distributions. However, if Claude and Kimi share the same inductive biases—for example, both misinterpreting a poorly labeled standard error as a standard deviation—the consensus mechanism will falsely validate the shared error. While heterogeneous architectures mitigate this, they do not eliminate systemic blind spots shared across current LLM training paradigms.

13. **Accessibility.** While API costs are low, running the current pipeline requires Python programming expertise. Broad adoption by non-computational meta-analysts will require wrapping this architecture in an accessible graphical user interface (GUI).

14. **Single-author validation.** This study was conducted by a single author. While the programmatic classifier removes subjectivity from the Li dataset classification, and the `info`-column decomposition for Loladze uses the original dataset's own annotations, the manual matching protocol (Section 2.4) and qualitative audit explanations were not independently replicated by a second human reviewer. All code, data, and validation scripts are publicly available to enable independent verification.

---

# 5. Conclusion

We developed and validated a multi-model consensus pipeline for quantitative data extraction in plant science meta-analysis. Four conclusions emerge.

First, AI can now reliably extract continuous numerical data from complex agricultural tables. The independent holdout (Hui) achieved r = 0.999 with 81% zero-error observations. Cross-domain validation (Li) achieved r = 0.999 on the programmatic high-confidence subset (110 observations, 18 papers). These results, the strongest reported for continuous numerical extraction from plant science literature, demonstrate that the Reading Barrier is effectively solved for standard tabular formats.

Second, multi-model consensus serves as a highly reliable predictor of accuracy, enabling confidence-stratified triage that concentrates 95% of large errors in the flagged minority. The pipeline reduces human extraction time by 70--75% at ~$0.37 per paper.

Third, current validation methodology systematically underestimates AI extraction capability. The Three-Barrier framework reveals that apparent validation failures are dominated by sub-selection disagreements (the Granularity Barrier, responsible for 73% of Loladze error) and reference-standard artifacts (the Provenance Barrier, responsible for the naive Li r = 0.453). These are not extraction failures---they are challenges in *proving* the extraction works.

Fourth, and most importantly for methodological credibility, the decomposition of validation error into its sources can be performed without circular LLM-confirms-LLM reasoning. Our programmatic confidence classifier uses only three observable data properties---zero-error fraction, direction agreement, and MAE thresholds---to identify high-quality subsets. This algorithmic classification agreed with an independent LLM audit on 85% of papers and actually produced a larger, better-performing subset (r = 0.999, MAE = 0.32 pp vs. r = 0.996, MAE = 0.44 pp from LLM classification). Future benchmarks should adopt reproducible, programmatic criteria for decomposing validation metrics into reading errors and alignment artifacts, rather than relying on subjective or LLM-based audit.

Aggregate meta-analytic effects were reproduced to within 0.04--0.84 pp across all three datasets, with |Cohen's d| < 0.20 (programmatic high-confidence |d| = 0.055). A pipeline that systematically misread values could not reproduce pooled effects with this precision. For plant science, a domain with decades of unprocessed CO2, biofortification, and biostimulant literature, reliable automated extraction at this accuracy level has immediate practical impact.

---

# References

*Note: Papers cited as illustrative examples from the validation datasets (Baslam et al. 2012, Fangmeier et al. 2002, Huluka et al. 1994, Pfirrmann et al. 1996, Niu et al. 2013, Chen et al. 2021, and similar source papers) are components of the Loladze (2014) and Li et al. (2022) reference datasets; their complete bibliographic records are available in those publications' reference lists and in the project repository supplementary files.*

- Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, 327(8476), 307-310.
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
- Clark, J., Barton, B. L., Albarqouni, L., et al. (2025). Generative artificial intelligence use in evidence synthesis: A systematic review. *Research Synthesis Methods*, 16, 601-619. DOI: 10.1017/rsm.2025.16.
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

# Ethics Statement

Not applicable. This study utilized previously published literature and publicly available datasets. No human subjects, animals, or unpublished patient data were involved.

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
| S6 | Prompt Architecture Schematic: visual flow of system prompts and JSON schema | `fig_prompt_architecture.png` |

# Table List

| Table | Description |
|-------|-------------|
| Table 1 | Combined validation results across all three datasets (Section 3.2) |
| Table 2 | Bootstrap confidence intervals for key metrics (Section 3.11.4) |
| Table 3 | Comparison of LLM-based data extraction systems (Section 4.1) |
| Table 4 | Li 2022 Validation Subsets (Section 3.6) |

# Supplementary Tables

*(Note: Full supplementary tables are provided as a separate document per journal guidelines)*

| Table | Description |
|-------|-------------|
| S1 | Forensic Audit of Excluded Li 2022 Papers (Root-cause analysis) |
| S2 | Per-paper validation details (46 Loladze papers) |
| S3 | Per-element accuracy breakdown (20 elements) |
| S4 | Consensus statistics and model contributions |
| S5 | Li 2022 paper-level audit: root-cause classification of all 27 papers |
| S6 | Data completeness and capture rates |
| S7 | Programmatic confidence classification: algorithm, per-paper results, and concordance with LLM audit |

---

# Supplementary Dataset: Pipeline Outputs and Meta-Analytic Results

The pipeline-extracted datasets used in this study are provided in full as supplementary data files, enabling independent use for meta-analysis and replication:

| File | Contents | Records |
|------|----------|---------|
| `SD1_loladze_extracted.csv` | All extracted observations from the 46 Loladze CO2/mineral papers: element, tissue type, crop species, CO2 level, control mean, treatment mean, effect %, SE/SD where available, confidence tier | 1,652 obs / 46 papers |
| `SD2_hui_extracted.csv` | All extracted observations from the 34 Hui Zn biofortification papers: Zn fraction, tissue, application method, dose, control mean, treatment mean, effect %, confidence tier | ~800 obs / 34 papers |
| `SD3_li2022_extracted.csv` | All extracted observations from the 27 Li biostimulant papers: crop, product, dose, yield metric, control mean, treatment mean, effect %, confidence tier | ~600 obs / 27 papers |

**Meta-analytic summaries** computed from the pipeline-extracted datasets are provided in:

| File | Contents |
|------|----------|
| `SD4_loladze_metaanalysis.csv` | Per-element pooled effects (lnRR, 95% CI, n) from the CO2/mineral dataset; DerSimonian-Laird random-effects model, 23 elements |
| `SD5_hui_metaanalysis.csv` | Per-application-method and per-tissue pooled Zn effects from the biofortification dataset; 13 subgroups |
| `SD6_li2022_metaanalysis.csv` | Per-biostimulant-category and per-crop pooled yield effects; 37 subgroups |

**Headline results from the pipeline-extracted meta-analyses:**

*SD4: Elevated CO2 reduces macro- and micronutrient concentrations.* Across 635 matched observations and 23 elements, the pipeline extracted consistent CO2-driven dilution effects. Major macronutrients: N −9.5%, P −7.7%, K −7.3%, Mg −8.4%. Micronutrients: Cu −6.8%, Zn −4.5%, Fe −0.9%. These values accord with published syntheses (Loladze 2014 reported −8% overall), confirming that the pipeline reproduces not only individual study values but aggregate meta-analytic patterns.

*SD5: Zn biofortification shows strong dose-response.* Across 319 matched observations, overall grain Zn increased by +49.7% under Zn treatment. Effect magnitude was strongly application-method-dependent: application type 2: +34.1% [29.4, 38.8]; type 3: +63.3% [50.6, 76.1]; type 4: +84.6% [70.4, 98.9]. High heterogeneity (I² = 100%) reflects genuine biological variability across soils, cultivars, and dose levels.

*SD6: Biostimulants increase yield by approximately 12% on average.* Across 200 matched observations and 27 papers, the pipeline extracted an overall yield increase of +11.9% [11.9, 19.0]. By category: seaweed extracts +15.4%, chitosan +20.0%, humic/fulvic acids +37.8%, plant hormones +8.4%, silicon +8.0%. By crop: soybean +28.1%, maize +17.5%, blackgram +24.4%, wheat +9.3%, potato +1.3%. High heterogeneity (I² ≈ 99%) is expected given the diversity of biostimulant products and crops.

These files represent the first openly available machine-extracted, confidence-stratified datasets for CO2/mineral, Zn biofortification, and biostimulant yield meta-analyses in plant science. Researchers may use them directly for downstream synthesis, reanalysis, or as training data for future extraction systems. See the Data Availability Statement for repository links.