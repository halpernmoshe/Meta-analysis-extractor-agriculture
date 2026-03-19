# Extraction Quality Report: 61_Kumar_2018

**Paper:** Velu, G., Singh, R.P., Crespo-Herrera, L., Juliana, P., Dreisigacker, S., Valluru, R., Stangoulis, J., Sohu, V.S., Mavi, G.S., Mishra, V.K., Balasubramaniam, A., Chatrath, R., Gupta, V., Singh, G.P. & Joshi, A.K. (2018). Genetic dissection of grain zinc concentration in spring wheat for mainstreaming biofortification in CIMMYT wheat breeding. *Scientific Reports*, 8, 13526. DOI: 10.1038/s41598-018-31951-z

**Match summary:** 0 extracted (no_extraction), 4 GT rows missed (2 in Data 2 Soil, 0 in Data 3 Foliar [study_id=60], 2 in Data 4 Soil+Foliar)

> Note: The gt file reports 6 total GT rows (2 soil, 2 foliar, 2 soil+foliar). The task prompt states 4 GT rows. The discrepancy is because the foliar rows (Obs 602-603) are attributed to a different per-sheet study_id (60), which may not be counted in the 4 rows assigned to this paper's primary IDs for validation purposes. All 6 rows are documented here for completeness.

---

## 1. Paper Design

This is a **genetic/genomic study** (GWAS), not an agronomic fertilizer trial.

- **Authors:** Govindan Velu et al. (CIMMYT and partner institutions in India)
- **Journal:** Scientific Reports (2018), published online 10 September 2018
- **Species:** Spring bread wheat (*Triticum aestivum*)
- **Panel:** HarvestPlus Association Mapping (HPAM) panel — 330 wheat lines from CIMMYT's biofortification breeding program
- **Design:** Randomized complete block design with 2 replications; plot size 2 m²
- **Environments:** 6 environments across India (PAU-Ludhiana, BHU-Varanasi, IIWBR-Karnal) and Mexico (CENEB, Ciudad Obregon) over crop seasons 2011-12, 2012-13, and 2013-14
- **Objective:** Genome-wide association study (GWAS) using Illumina iSelect 90K SNP array to identify QTL and candidate genes for grain Zn concentration
- **Zn application:** 25 kg ha⁻¹ ZnSO₄ applied uniformly to all plots at CENEB every season since 2009-10 solely to correct soil Zn heterogeneity — **not as an experimental treatment variable**
- **Primary outcome:** Grain Zn concentration (mg kg⁻¹) measured by EDXRF (Energy-Dispersive X-ray Fluorescence Spectrometry)
- **Statistical approach:** Mixed linear model (MLM) with population structure (Q matrix) and kinship (K matrix) in TASSEL5.0; broad-sense heritability H² = 0.6 for grain Zn

**Key data reported in paper:**
- Range of grain Zn: 35.5–67.7 mg/kg (Obregon 2012); 42.5–80.3 mg/kg (Obregon 2013); 27.2–54.6 mg/kg (IIWBR); 26.7–39.4 mg/kg (BHU); 17.8–40.5 mg/kg (PAU-2013); 27.9–50.9 mg/kg (PAU-2014)
- 39 significant marker-trait associations (MTA) for grain Zn identified across chromosomes 1A, 2A, 2B, 2D, 5A, 6B, 6D, 7B, UN
- Two major QTL regions on chromosomes 2 and 7

---

## 2. Grain Zn Data in PDF?

**Yes, grain Zn concentration values are reported — but not in a treatment vs. control table format.**

The PDF contains:
- **Figure 1:** Frequency distribution histograms of grain Zn concentration across 6 environments (population-level distribution, not individual treatment means)
- **Figure 5:** Boxplots of grain Zn and TKW by SNP genotype (CC vs TT) for markers RAC875_c34757_180 and IAAV1375 — these show genotypic group means, not Zn-fertilizer treatment effects
- **Table 1:** 39 significant SNPs with F-statistics, additive effects, p-values, marker R², genetic variance, and residual variance — genomic statistics, not agronomic treatment means
- **Table 2:** Candidate genes linked to stable SNPs — molecular genetics data

There are **no tables or figures in the PDF showing a control (no Zn applied) vs. treatment (Zn applied) comparison** for grain Zn concentration. The paper presents population-level phenotypic distributions and GWAS results, not agronomic fertilizer response data.

The ZnSO₄ application at CENEB (25 kg ha⁻¹) is explicitly described as a soil correction measure applied uniformly to all plots, with no control arm without Zn application.

---

## 3. AI Extraction (Nothing — Why?)

All three AI models (Claude, Kimi, Gemini) extracted **0 observations**. The consensus JSON confirms:

**Recon phase correctly identified the problem:**
The recon module issued 8 explicit warnings including:
- "This is a genetic/breeding study, NOT a Zn fertilizer intervention study"
- "Paper focuses on genetic variation in Zn concentration using GWAS analysis"
- "No experimental Zn fertilizer treatments — examines natural genetic variation"
- "Paper is completely irrelevant to Zn fertilizer meta-analysis — focuses on genetic breeding"
- "extraction_guidance: DO NOT EXTRACT DATA FROM THIS PAPER"

**Extraction result:** `claude_obs: 0`, `kimi_obs: 0`, `gemini_obs: 0`

**Post-processing note:** The consensus JSON records `post_processing.original_count: 2` with `null_means_removed: 2`, meaning 2 candidate rows were tentatively generated but were removed because the mean values were null. This suggests at least one model briefly attempted an extraction but produced empty/null means — consistent with the paper presenting population distributions (histograms, boxplots) rather than discrete control/treatment means.

**Root cause of zero extraction:** The AI pipeline correctly determined this is a GWAS study with no experimental Zn fertilizer intervention. The recon phase's `extraction_guidance` flag directed all extraction models to skip the paper. The scanned PDF format (OCR-dependent, `is_scanned: true`) would have further complicated any attempted table extraction.

---

## 4. GT Data

The MOESM5 spreadsheet assigns data from this paper across three sheets, reflecting that the Hui 2023 meta-analysis extracted data from this paper for three different application categories:

### Sheet: Data 2 — Soil Application (study_id = 61)

| Field | Obs 564 | Obs 565 |
|-------|---------|---------|
| Country | India | India |
| Available Zn (mg kg⁻¹) | 2.91 | 2.91 |
| pH | 5.5 | 5.5 |
| Organic matter (g kg⁻¹) | 8.10 | 8.10 |
| N rate (kg N ha⁻¹) | 100 | 100 |
| P rate (kg P ha⁻¹) | 60.46 | 60.46 |
| K rate (kg K ha⁻¹) | 39.84 | 39.84 |
| Zn fertilizer type | ZnSO₄ | ZnSO₄ |
| Zn rate (kg Zn ha⁻¹) | 22.75 | 22.75 |
| Replicates (n) | 3 | 3 |
| Grain Zn conc. (mg kg⁻¹) | [not shown — col header missing] | [not shown] |
| Grain Zn accumulation (g ha⁻¹) | 149.3 | 148.69 |
| Straw Zn conc. (mg kg⁻¹) | 26.06 | 26.30 |
| Grain yield (kg ha⁻¹) | 3932 | 3758 |

### Sheet: Data 3 — Foliar Application (study_id = 60)

| Field | Obs 602 | Obs 603 |
|-------|---------|---------|
| Country | India | India |
| Spraying concentration (g Zn L⁻¹) | 0.1135 | 0.1135 |
| Spraying frequency (times) | 3 | 3 |
| Spraying timing | 10 | 10 |
| Replicates (n) | 3 | 3 |
| Grain Zn conc. (mg kg⁻¹) | 38.15 | 39.58 |
| Grain Zn accumulation (g ha⁻¹) | 149.3 | 148.69 |

### Sheet: Data 4 — Soil+Foliar Application (study_id = 13)

| Field | Obs 57 | Obs 58 |
|-------|--------|--------|
| Country | India | India |
| Replicates (n) | 3 | 3 |
| Grain Zn conc. (mg kg⁻¹) | 38.15 | 39.58 |
| Grain Zn accumulation (g ha⁻¹) | 149.3 | 148.69 |
| Straw Zn accumulation (g ha⁻¹) | 170.88 | 177.65 |
| Grain yield (kg ha⁻¹) | 3932 | 3758 |

**Interpretation of GT values:** The two grain Zn concentration values in the foliar and soil+foliar sheets (38.15 and 39.58 mg kg⁻¹) and Zn accumulation values (149.3 and 148.69 g ha⁻¹) closely match values that could be read from population-level summaries (e.g., mean grain Zn at specific environments). The Zn rate of 22.75 kg Zn ha⁻¹ in the soil sheet matches ZnSO₄ application with appropriate molecular weight conversion. The two observations likely represent data from two different environments or seasons.

---

## 5. Root Cause

**This is a PDF-label/study-type mismatch — a cataloguing error, not an extraction failure.**

The file `61_Kumar_2018.pdf` contains the paper:
> Velu et al. (2018), *Scientific Reports* 8:13526 — a GWAS study of 330 wheat lines for genetic loci controlling grain Zn concentration.

The MOESM5 spreadsheet attributes data to:
> "Kumar, A., Denre, M., Prasad, R., 2018. Agronomic biofortification of zinc in wheat (*Triticum aestivum* L.). *Current Science*, 115(5), 944-948."

**These are two completely different papers published in 2018:**

| Attribute | PDF content (Velu et al.) | GT citation (Kumar et al.) |
|-----------|--------------------------|---------------------------|
| Authors | Velu, Singh, Crespo-Herrera et al. | Kumar, Denre, Prasad |
| Journal | Scientific Reports | Current Science |
| DOI | 10.1038/s41598-018-31951-z | not in PDF |
| Study type | GWAS, 330 genetic lines | Agronomic fertilizer trial |
| Location | India + Mexico (CIMMYT) | India (field trial) |
| Design | Multi-environment genotype panel | Treatment vs. control, n=3 |
| Zn data | Population Zn distribution, QTL | Grain Zn conc. by Zn rate |

The GT data (soil Zn rate 22.75 kg ha⁻¹, soil pH 5.5, grain Zn 38.15–39.58 mg kg⁻¹, n=3) is consistent with a small-plot agronomic trial in India — characteristic of the Kumar et al. (2018) *Current Science* paper. This paper describes applied soil and foliar Zn biofortification treatments in wheat with measurement of grain Zn concentration and accumulation. That paper was **not present** in the PDF corpus; only the Velu et al. GWAS paper was downloaded under filename `61_Kumar_2018.pdf`.

**Additional contributing factor:** The recon module confirmed `is_scanned: true`, indicating the Velu et al. PDF is a scanned document with OCR text, which would make any attempted extraction harder even if it were the correct paper.

---

## 6. Assessment

**Verdict: WRONG PDF — correct AI behavior**

The AI extraction system performed correctly. All three models and the recon module unanimously identified that the PDF in the corpus is a GWAS genetic study (Velu et al. 2018, *Scientific Reports*) with no agronomic Zn fertilizer treatment-control data. The decision not to extract was appropriate and well-reasoned.

The 4–6 missed GT rows are not recoverable from the available PDF because the correct source paper (Kumar, Denre & Prasad, 2018, *Current Science* 115(5):944-948) is absent from the PDF folder. The filename `61_Kumar_2018.pdf` misleadingly suggests it is the Kumar et al. paper, but its contents are entirely different.

**To recover these observations:** Obtain the PDF for Kumar, A., Denre, M. & Prasad, R. (2018). Agronomic biofortification of zinc in wheat (*Triticum aestivum* L.). *Current Science*, 115(5), 944-948, and replace or add it to the corpus under this study ID.

**Impact on validation metrics:** The no_extraction status for this paper inflates the missed-observation count in the Hui 2023 validation but does not reflect any deficiency in the extraction pipeline. This paper should be flagged as `WRONG_PDF` in validation accounting and excluded from per-paper extraction quality statistics.
