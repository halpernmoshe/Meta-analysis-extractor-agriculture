# Supplementary Materials

**Breaking the Extraction Bottleneck: A Single AI Agent Achieves Statistical Equivalence with Human-Extracted Meta-Analysis Data Across Five Agricultural Datasets**

Moshe Halpern

Institute of Soil, Water and Environmental Sciences, Agricultural Research Organization -- Volcani Center, Israel

---

## Table S1: Per-Paper Validation Results Across All Five Datasets

Paper-level metrics: observation count (n), Pearson correlation (r), mean absolute error (MAE, in percentage points), and quality tier assignment (Excellent: MAE < 5 pp; Good: 5--10 pp; Fair: 10--20 pp; Poor: > 20 pp).

### S1a. Loladze 2014 -- CO2 Effects on Plant Mineral Concentrations

646 matched observations, 46 papers. Overall: r = 0.812, MAE = 6.16 pp.

| Paper | n_obs | r | MAE (pp) | Tier |
|-------|-------|---|----------|------|
| Fernando et al. 2012a | 8 | -- | -- | Excellent |
| Ziska 1997 | 4 | -- | -- | Excellent |
| Baslam 2012 | 20 | -- | -- | Good |
| Finzi 2001 | 6 | -- | -- | Good |
| Niinemets 1999 | 8 | -- | -- | Fair |
| Azam 2013 | 12 | -- | -- | Excellent |
| Woodin 1992 | 4 | -- | -- | Good |
| Campbell 2002 | 10 | -- | -- | Good |
| Barnes 1992 | 14 | -- | -- | Fair |
| Li 2010 (Hoeqy 2009) | 16 | -- | -- | Fair |
| Huluka 1994 | 10 | -- | -- | Excellent |
| Wu 2004 | 18 | -- | -- | Fair |
| Keutgen 2001 | 16 | -- | -- | Good |
| Lieffering 2004 | 20 | -- | -- | Excellent |
| Pleijel 2009 | 22 | -- | -- | Good |
| Fernando 2012a | 8 | -- | -- | Excellent |
| Fangmeier 2002 | 10 | -- | -- | Fair |
| Al-Rawahy 2013 | 11 | -- | -- | Excellent |
| Baxter 1994 | 18 | -- | -- | Excellent |
| Overdieck 1993 | 11 | -- | -- | Good |
| (remaining 26 papers) | ... | ... | ... | ... |
| **Tier summary** | | | | **11 Excellent, 11 Good, 8 Fair, 16 Poor** |

*Note:* Per-paper r values are not meaningful for papers with < 5 observations or near-zero variance. Full per-paper data is available in the repository at `output/loladze_v3_combined/validation_matches.csv`. Tier counts from the formal stats JSON: 11 Excellent, 11 Good, 8 Fair, 16 Poor.

### S1b. Hui 2023 -- Zinc Biofortification in Wheat

319 matched observations, 17 papers. Overall: r = 0.999, MAE = 0.43 pp.

| Paper | n_obs | r | MAE (pp) | Tier |
|-------|-------|---|----------|------|
| (All 17 papers) | 319 | 0.999 | 0.43 | -- |
| **Tier summary** | | | | **14 Excellent, 2 Good, 1 Fair, 0 Poor** |

*Note:* 81% of observations had zero error (exact match). Full per-paper data at `output/hui2023_extraction/validation_hui2023_matches.csv`.

### S1c. Li 2022 -- Biostimulant Effects on Crop Yield

125 matched observations, 20 papers. Overall: r = 0.806, MAE = 5.78 pp.

Programmatic high-confidence subset (scale-harmonized): 110 obs, 18 papers, r = 0.999, MAE = 0.32 pp.

| Paper | n_obs | r | MAE (pp) | Tier |
|-------|-------|---|----------|------|
| (All 20 papers) | 125 | 0.806 | 5.78 | -- |
| **Tier summary** | | | | **7 Excellent, 2 Good, 6 Fair, 5 Poor** |

*Note:* Many "errors" in this dataset are unit-scale mismatches (e.g., t/ha vs. kg/ha), not extraction errors. After programmatic scale harmonization, accuracy improves dramatically. Full data at `output/li2022_combined/validation_matches.csv`.

### S1d. Biochar (Li et al. 2024) -- Biochar Effects on Crop Yield

254 matched observations, 26 papers. Overall: r = 0.997, MAE = 1.20 pp.

| Paper | n_obs | r | MAE (pp) | Tier |
|-------|-------|---|----------|------|
| Adekiya 2019 | -- | -- | -- | Excellent |
| Gathorne-Hardy 2009 | -- | -- | -- | Excellent |
| Guerena 2013 | -- | -- | -- | Excellent |
| Wei 2022 | -- | -- | -- | Excellent |
| (remaining 22 papers) | ... | ... | ... | ... |
| **Tier summary** | | | | **22 Excellent, 4 Good, 0 Fair, 0 Poor** |

*Note:* Fully prospective holdout -- processed after all methods were finalized. Full data at `output/biochar_extraction/summary.csv`.

### S1e. Boldorini 2024 -- Predator Biocontrol Effects on Crop Yield

46 matched observations, 18 papers. Overall: r = 0.972, MAE = 3.06 pp (on lnRR scale: MAE = 0.026).

| Paper | n_obs | Direction | Tier |
|-------|-------|-----------|------|
| Ali 2018 | -- | -- | -- |
| Bisseleua 2017 | -- | -- | -- |
| Borkhataria 2012 | -- | -- | -- |
| Classen 2014 | -- | -- | -- |
| (14 additional papers) | ... | ... | ... |
| **Overall** | **46** | **95.7%** | -- |

*Note:* Capture rate 97.9% (46/47 GT observations matched). Full data at `output/boldorini_extraction/validation_results.json`.

---

## Table S2: Extraction Configuration Files

Each dataset used a JSON configuration file specifying PICO criteria, extraction priorities, moderators, and ground-truth locations.

| Dataset | Config File | Key Parameters |
|---------|-------------|----------------|
| Loladze 2014 | `configs/loladze_co2_minerals.json` | 25 mineral elements; FACE/OTC/greenhouse; C3/C4 plants |
| Hui 2023 | `configs/hui2023_zinc_wheat.json` | Grain Zn concentration (mg/kg); soil/foliar/combined application |
| Li 2022 | `configs/li2022_biostimulant_yield.json` | Fresh yield; 7 biostimulant categories (SWE, HFA, PHs, etc.) |
| Biochar 2024 | `configs/biochar_crop_yield.json` | Crop yield; feedstock, pyrolysis temp, application rate moderators |
| Boldorini 2024 | `configs/boldorini2024_predator_yield.json` | Crop yield; exclusion vs. addition experiments; 5 predator types |

All configuration files are available in the repository under `configs/`. Each config specifies:
- `intervention` and `control` definitions
- `primary_outcomes` with expected direction and typical effect size
- `important_moderators` for factorial design handling
- `tc_confusion_warnings` to prevent treatment/control swap errors
- `extraction_priorities` (e.g., "extract EVERY row, no pooling")
- `ground_truth` section with dataset path, column mappings, and encoding

---

## Table S3: Variance Recovery Statistics

Variance (SE/SD) extraction was assessed but not formally validated against ground truth. These statistics describe recovery rates from extracted data.

| Dataset | Total Obs | With Variance | Recovery Rate | Variance Types | Imputed Obs | Imputation Method |
|---------|-----------|---------------|---------------|----------------|-------------|-------------------|
| Loladze 2014 | 646 | ~325 | ~50% | SE, SD | -- | Not imputed |
| Hui 2023 | 319 | ~160 | ~50% | SE, SD, LSD | -- | Not imputed |
| Li 2022 | 125 | ~48 | ~38% | SE, SD, LSD | -- | Not imputed |
| Biochar 2024 | 413 | 106 | 25.7% | SE, SD | +83 (22.4%) | CV-based imputation |
| Boldorini 2024 | 46 | ~40 | ~87% | SD | -- | Not imputed |

### Biochar Variance Imputation Details

For the biochar dataset, missing variance was imputed using the coefficient of variation (CV) method:

- **106 observations** had extracted variance (25.7%)
- **Mean CV** from studies with variance: 11.94% (median 7.82%)
- **Imputation**: SD_imputed = CV_median x mean, stratified by crop type
- **83 additional observations** received imputed variance (22.4% recovery gain)
- **Imputation spread**: 0.78 pp (difference in MAE with vs. without imputed observations)

CV by crop type (from studies with variance):
- Maize: median CV 6.89% (n=66)
- Rice: median CV 8.93%
- Wheat: median CV 5.15%
- Annual crops: median CV 2.68% (n=6)

---

## Table S4: Source Type Distribution and Accuracy

Every extracted observation was labeled by data source type (table, figure, or text).

### Distribution Across Datasets

| Dataset | Table | Figure | Text | Total |
|---------|-------|--------|------|-------|
| Loladze 2014 | ~420 (65%) | ~210 (33%) | ~16 (2%) | 646 |
| Hui 2023 | ~290 (91%) | ~25 (8%) | ~4 (1%) | 319 |
| Li 2022 | ~85 (68%) | ~35 (28%) | ~5 (4%) | 125 |
| Biochar 2024 | 122 (35%) | 225 (65%) | -- | 347 |
| Boldorini 2024 | ~35 (76%) | ~11 (24%) | -- | 46 |
| **Overall** | ~952 (65%) | ~506 (35%) | ~25 (<2%) | 1,483 |

### Accuracy by Source Type (Biochar dataset -- most detailed breakdown)

| Source Type | n_obs | Median AE (pp) | MAE (pp) | Within 2pp | Within 5pp |
|-------------|-------|-----------------|----------|------------|------------|
| Table | 122 | 0.57 | 0.89 | 85% | 100% |
| Figure | 225 | 3.12 | 4.87 | 42% | 78% |

Table-sourced observations achieved **5.5x lower median error** than figure-sourced observations (0.57 vs. 3.12 pp). This difference reflects the inherent precision loss in reading values from graphical displays versus structured numeric tables.

---

## Table S5: Agent Replication Results (Run 1 vs. Run 2)

To assess extraction reproducibility, the agent was run twice on the same papers with identical prompts. Observations from both runs were matched and compared.

| Dataset | Papers Matched | Obs Matched | r | MAE (pp) | Within 5pp | Direction | Effect Diff (pp) |
|---------|---------------|-------------|---|----------|------------|-----------|------------------|
| Loladze 2014 | 44 | 698 | 0.816 | 8.22 | 63% | 87% | 0.05 |
| Hui 2023 | 24 | 362 | 0.946 | 12.34 | 56% | 96% | 6.31 |
| Li 2022 | 31 | 216 | 0.836 | 5.81 | 72% | 93% | 0.74 |
| **Total** | **99** | **1,276** | -- | -- | -- | -- | -- |

### Interpretation

Observation-level reproducibility (r = 0.82--0.95) is lower than ground-truth accuracy in some cases. This is expected: the agent may extract different subsets of factorial combinations on different runs (e.g., selecting different cultivars or time points from a complex table), producing different but individually correct observations. The critical metric is **aggregate effect agreement**, which is excellent:

- Loladze: 0.05 pp difference in aggregate effect between runs
- Li 2022: 0.74 pp difference
- Hui 2023: 6.31 pp difference (driven by variation in which treatment combinations are extracted from high-effect-size papers)

---

## Table S6: Consensus Mechanism Details

A multi-model consensus pipeline was developed as a secondary extraction method, using three LLM providers.

### Models Used

| Model | Provider | Context Window | Cost/Paper | Role |
|-------|----------|---------------|------------|------|
| Moonshot Kimi K2.5 | Moonshot AI | 256K | ~$0.03 | Primary extractor (highest observation yield) |
| Claude Sonnet 4 | Anthropic | 200K | ~$0.08 | Secondary extractor (highest per-obs accuracy) |
| Gemini 2.5 Flash | Google | 1M | ~$0.02 | Tertiary extractor |

### Consensus Logic

1. Each model independently extracts observations from the same PDF
2. Observations are matched across models using a fuzzy value-matching algorithm
3. When 2 or 3 models agree on a value (within tolerance), the consensus value is adopted
4. Observations extracted by only one model are included but flagged as single-source

### Coverage Gains

| Metric | Best Single Model (Kimi) | Consensus (3 models) | Gain |
|--------|--------------------------|----------------------|------|
| Total observations | 884 | 1,528 | +73% |
| Papers with >= 1 obs | -- | -- | -- |
| Unique observations | -- | 1,528 | -- |

Model contributions: Kimi 884 obs, Claude 841 obs, Gemini 255 obs.
Total cost for consensus extraction across all papers: approximately $17.

### Consensus vs. Ground Truth

Consensus did not improve per-observation accuracy over the best single model. Its value is **coverage**: the multi-model approach recovered 73% more observations than the best single model, primarily by capturing data from figures and tables that individual models missed.

---

## Table S7: Cross-Dataset Summary Statistics

| Metric | Loladze | Hui | Li | Biochar | Boldorini |
|--------|---------|-----|-----|---------|-----------|
| **Domain** | CO2/minerals | Zn/wheat | Biostimulants/yield | Biochar/yield | Predators/yield |
| **Papers processed** | 46 | 17 | 20 | 26 | 18 |
| **Matched observations** | 646 | 319 | 125 | 254 | 46 |
| **Pearson r** | 0.812 | 0.999 | 0.806 | 0.997 | 0.972 |
| **ICC(3,1)** | 0.807 | 0.999 | 0.806 | 0.997 | 1.000 |
| **MAE (pp)** | 6.16 | 0.43 | 5.78 | 1.20 | 3.06 |
| **Median AE (pp)** | 1.55 | 0.00 | 2.29 | 0.64 | -- |
| **Direction agreement** | 85.4% | 99.7% | 86.7% | 92.3% | 95.7% |
| **GT mean effect** | -5.92% | 49.61% | 11.30% | 12.27% | 20.43% |
| **Extracted mean effect** | -4.95% | 49.72% | 11.61% | 12.05% | 20.43% |
| **Effect diff (pp)** | 0.97 | 0.12 | 0.31 | 0.22 | 0.00 |
| **TOST +/-2pp** | p=0.013 PASS | p<0.001 PASS | p=0.027 PASS | p<0.001 PASS | p<0.001 PASS |
| **TOST +/-3pp** | p<0.001 PASS | p<0.001 PASS | p=0.001 PASS | p<0.001 PASS | p<0.001 PASS |
| **Cohen's d** | 0.082 | 0.072 | 0.032 | -0.125 | 0.217 |
| **Holdout status** | Dev-adjacent | Independent | Independent | Prospective | Independent |
| **Effect size range** | 1--30 pp | 5--200+ pp | 1--100+ pp | 1--50 pp | 5--50 pp |

---

## Appendix A: Statistical Methods Detail

### A.1 Equivalence Testing (TOST)

Two one-sided tests were conducted with cluster-robust standard errors using the CR2 bias-corrected sandwich estimator with Satterthwaite degrees of freedom (Pustejovsky & Tipton, 2018). Papers serve as clusters to account for within-paper correlation of observations.

The null hypothesis for TOST is that the true mean difference falls outside the equivalence margin [-delta, +delta]. Rejection (p < 0.05 for both one-sided tests) implies the mean difference lies within the margin.

Primary margin: +/-3 pp. Secondary margin: +/-2 pp.

### A.2 ICC Computation

ICC(3,1) was computed using a two-way mixed-effects model with consistency definition, following Koo and Li (2016). This variant treats extracted and ground-truth as two fixed "raters" measuring the same observations.

### A.3 Bland-Altman Analysis

For each matched observation, the difference (extracted - GT) was plotted against the mean of the two values. The mean bias and 95% limits of agreement (mean +/- 1.96 x SD of differences) were computed. Proportional bias was assessed by regressing the difference on the mean.

Loladze: mean bias = 0.97 pp, LoA = [-22.15, 24.09], proportional bias r = -0.185 (p < 0.001)
Hui: mean bias = 0.12 pp, LoA = [-3.06, 3.29], no proportional bias (p = 0.22)
Li: mean bias = 0.31 pp, LoA = [-18.72, 19.35], no proportional bias (p = 0.27)

### A.4 Direction Agreement

Direction agreement was computed as the fraction of observations where the extracted and ground-truth effect sizes share the same sign. Observations where either value is exactly zero were excluded from the denominator.

### A.5 Tier Classification

Papers were classified into quality tiers based on paper-level MAE:
- **Excellent**: MAE < 5 pp
- **Good**: 5 <= MAE < 10 pp
- **Fair**: 10 <= MAE < 20 pp
- **Poor**: MAE >= 20 pp

### A.6 LLM-Driven Alignment Method

The LLM alignment method operates in three steps:

1. **Schema reading**: The LLM reads column names and sample values from both the extracted data and the reference standard.
2. **Mapping proposal**: The LLM proposes (a) study-level mappings (extracted paper ID to reference study ID), (b) column-level mappings (which extracted fields correspond to which GT fields), and (c) value-level synonyms (e.g., "corn" = "Maize", "hardwood" = "Wood").
3. **Deterministic matching**: The proposed mappings are applied deterministically -- no LLM is involved in the actual matching step. Observations are joined on study ID and moderator values using the LLM-proposed synonym dictionaries.

The alignment output is cached as a human-editable JSON file, enabling manual inspection and correction.

---

## Appendix B: Agent Prompts

The agent received a natural-language instruction for each dataset specifying:
- What data to extract (intervention, control, outcomes)
- Expected direction and typical effect sizes
- Moderators to record
- Common treatment/control confusion warnings
- Extraction priorities (e.g., "extract every row, no pooling")

The full configuration files serve as the effective prompts and are provided in `configs/`. The agent model (Claude Opus 4.6) received no few-shot examples, no domain-specific templates, and no training on any reference standard.

---

## Appendix C: Ground-Truth-Free Validation (Agent vs. Pipeline Agreement)

Cross-method agreement was computed between the agent extraction and a structurally independent multi-model consensus pipeline on the same papers, with no reference to any ground truth.

| Dataset | Papers | Obs | r | MAE (pp) | Direction | Effect Diff (pp) |
|---------|--------|-----|---|----------|-----------|------------------|
| Loladze | 44 | 1,205 | 0.933 | 5.04 | 91% | 1.30 |
| Hui | 20 | 185 | 0.971 | 8.14 | 96% | 0.29 |
| Li | 36 | 499 | 0.994 | 10.29 | 88% | 1.89 |
| **Total** | **100** | **1,889** | -- | -- | -- | -- |

The high cross-method agreement (r > 0.93 across all datasets) provides evidence that extraction results are not artifacts of a particular model or pipeline. The aggregate effect differences (0.29--1.89 pp) are small relative to typical effect sizes.
