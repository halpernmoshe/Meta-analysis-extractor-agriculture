# Extraction Quality Report: Zhang_2017

**Match summary:** 0 extracted matching GT, 8 GT rows missed (no_extraction)
**GT source:** MOESM5 "Data 2 Soil application" sheet, study_id = 35, obs IDs 189–196
**PDF:** Zhang P, Ma G, Wang C, et al. (2017) "Effect of irrigation and nitrogen application on grain amino acid composition and protein quality in winter wheat." *PLoS ONE* 12(6): e0178494. https://doi.org/10.1371/journal.pone.0178494

---

## 1. Paper Design

Zhang et al. (2017) is a field experiment conducted at Wenxian, Henan Province, North China (34°92'N, 112°99'E) from 2010, reported across three cropping seasons: 2012/2013, 2013/2014, and 2014/2015. The cultivar was Yumai 49-198 (winter wheat).

**Design:** Split-plot factorial
- **Main plots (irrigation):** I0 (no irrigation), I1 (irrigation at jointing), I2 (irrigation at jointing + anthesis), 750 m³ ha⁻¹ each time
- **Sub-plots (nitrogen):** N0, N180, N240, N300 (kg N ha⁻¹)
- **Replicates:** n = 3 per treatment per season
- **Background fertilisation:** P (150 kg P₂O₅ ha⁻¹) and K (120 kg K₂O ha⁻¹) applied uniformly to all plots

**Stated objectives:** To evaluate the effect of irrigation and nitrogen application on grain yield, protein content, and amino acid composition (TAA, EAA, NAA, EAAI, PDCAAS) in winter wheat.

**Soil characteristics (Table 1):** Total N, available N, available P, available K, organic matter, pH measured before sowing for each cropping season under each N treatment. Soil pH ≈ 8.19–8.32 (alkaline), organic matter ≈ 15–17 g kg⁻¹, available Zn reported: **not measured, not tabulated anywhere in the paper.**

**Statistical analysis:** ANOVA with LSD test at p = 0.05 probability level (SPSS, split-plot design).

---

## 2. Grain Zn Data in PDF?

**No.** A complete review of all 15 pages, all 5 tables, 3 figures, and the S1 supporting file description confirms that **grain zinc (Zn) concentration is never measured, reported, or mentioned anywhere in this paper.**

The variables reported in this study are:

| Table | Variables |
|-------|-----------|
| Table 1 | Soil: total N, available N, available P, available K, organic matter, pH |
| Table 2 | Grain yield (t ha⁻¹), grain protein content (%) |
| Table 3 | TAA, EAA, NAA, EAA/TAA ratio (mg g⁻¹) |
| Table 4 | EAAI (%), PDCAAS (%) |
| Table 5 | Correlation matrix among yield, protein, TAA, EAA, EAAI, PDCAAS |
| S1 File | 17 individual amino acids (Asp, Thr, Ser, Glu, Gly, Ala, Cys, Val, Met, Ile, Leu, Tyr, Phe, Lys, His, Arg, Pro) |
| Figs 1-3 | Climate data; EAA composition bar charts; protein vs. amino acid regressions |

The word "zinc" does not appear in the paper. The word "Zn" appears only in the fertilizer description (potassium chloride = 60% K₂O) and in the soil available K analysis method context — never in the context of grain micronutrient concentration.

**This paper contains no grain Zn concentration data whatsoever.**

---

## 3. AI Extraction (Nothing — Why?)

The recon phase of the consensus pipeline correctly identified the problem. The recon JSON contains an explicit, accurate warning:

> "MAJOR WARNING: This paper is about IRRIGATION and NITROGEN effects on wheat, NOT zinc fertilizer effects"
> "No grain Zn concentration data reported anywhere in the paper"
> "This paper is completely irrelevant to the zinc biofortification meta-analysis"
> `"tables_with_target_data": []`
> `"extraction_guidance": "DO NOT EXTRACT DATA FROM THIS PAPER"`

Despite this guidance, the consensus JSON shows `"matched_obs": 8` and `"consensus_observations"` contains 8 records. However, inspection of those 8 records reveals they contain **none of the variables the Hui 2023 meta-analysis requires**:

| Extracted element | Unit | Relevance to Hui meta-analysis |
|-------------------|------|-------------------------------|
| grain yield | t ha⁻¹ | Not a Hui outcome variable |
| protein content | % | Not a Hui outcome variable |
| TAA | mg g⁻¹ | Not a Hui outcome variable |
| EAA | mg g⁻¹ | Not a Hui outcome variable |
| EAAI | % | Not a Hui outcome variable |
| PDCAAS | % | Not a Hui outcome variable |
| total N (soil) | g kg⁻¹ | Not a Hui outcome variable |
| available N (soil) | mg kg⁻¹ | Not a Hui outcome variable |

The AI extracted data that **exists in the PDF** (grain yield, protein, amino acids) but none of it is grain Zn concentration. The pipeline's extraction stage ignored the recon-stage guidance not to extract from this paper and instead fell back to extracting whatever numerical data was present.

The result is that the validation matcher found zero observations in the consensus JSON that could be paired with the 8 GT rows (which all require `Grain Zn concentration (mg kg⁻¹)`), and correctly logged the paper as `no_extraction` from the meta-analysis perspective.

**Summary of AI behaviour:**
- Recon: CORRECT — paper flagged as irrelevant, Zn data identified as absent
- Extraction: INCORRECT — extracted irrelevant variables (yield, protein, amino acids) despite the explicit recon guidance
- Validation match: CORRECT outcome — 0/8 GT rows matched, as expected

---

## 4. GT Data (8 Rows from Soil Application Sheet)

The MOESM5 ground truth assigns this paper to the "Data 2 Soil application" sheet as study_id = 35, observations 189–196. All 8 rows share the same soil and site characteristics.

**Site / soil metadata (fixed across all 8 rows):**
- Country: China
- Available Zn: 1.09 mg kg⁻¹ (grouping: >0.5)
- pH: 8.28 (grouping: >7.0)
- Organic matter: 16.45 g kg⁻¹ (grouping: 10–20)
- P rate: 66 kg P ha⁻¹ (grouping: >44)
- K rate: 124.5 kg K ha⁻¹ (grouping: >58)
- Zn fertilizer type: ZnSO₄
- Zn rate: 2.1792 kg Zn ha⁻¹ (grouping: <8)
- Replicates (n): 5

**The 8 observations vary by N rate, and appear in two groups of 4 (likely two Zn application methods or two seasons averaged):**

| Obs ID | N rate (kg N ha⁻¹) | Grain Zn conc. (mg kg⁻¹) | Grain yield (kg ha⁻¹) |
|--------|-------------------|--------------------------|----------------------|
| 189 | 0 | 23.23 | 4761.11 |
| 190 | 180 | 27.86 | 8621.67 |
| 191 | 240 | 26.35 | 8721.11 |
| 192 | 300 | 25.56 | 8545.56 |
| 193 | 0 | 20.83 | 3932.72 |
| 194 | 180 | 23.28 | 8219.33 |
| 195 | 240 | 25.11 | 8002.18 |
| 196 | 300 | 23.88 | 9119.44 |

**Key observation:** Hui et al. (2023) extracted grain Zn concentration data from this study with n = 5 replicates and a ZnSO₄ soil application rate of 2.18 kg Zn ha⁻¹. The GT grain Zn values (20.83–27.86 mg kg⁻¹) are plausible for Chinese wheat under moderate Zn application. The grain yields in the GT (3,933–9,119 kg ha⁻¹) are consistent with what Table 2 of the paper reports for the I0–I2 × N0–N300 factorial design across seasons.

**Critical discrepancy:** The Zhang_2017 PDF in the pipeline's PDF folder (the PLoS ONE 2017 paper on irrigation and amino acid composition) **does not contain grain Zn data**. The GT entries can only originate from a **different Zhang 2017 document** — most likely the PhD thesis cited in MOESM5:

> "Zhang, P.P., 2017. *Regulating effects of N and Zn application on accumulation and distribution of mineral elements and grain quality in winter wheat.* Thesis for Doctor's Degree. Henan Agricultural University."

The thesis title explicitly mentions "Zn application" and "mineral elements" — this is the source of the GT data. The published PLoS ONE paper by the same first author in the same year focuses on amino acids and does not include Zn data. The two works share the same first author (Panpan Zhang), the same institution (Henan Agricultural University), and overlap substantially in experimental design, but the thesis contains the Zn-specific data tables that Hui et al. used.

---

## 5. Root Cause

**Primary cause: Wrong PDF — the pipeline holds the published journal article, not the PhD thesis.**

The MOESM5 ground truth references:
> Zhang, P.P., 2017. *Regulating effects of N and Zn application on accumulation and distribution of mineral elements and grain quality in winter wheat.* **Thesis** for Doctor's Degree. Henan Agricultural University.

The PDF in the pipeline's source folder is:
> Zhang P et al. (2017) "Effect of irrigation and nitrogen application on grain amino acid **composition and protein quality** in winter wheat." **PLoS ONE** 12(6): e0178494.

These are two separate publications by the same lead author (Panpan Zhang) from the same institution and approximate time, but with completely different scopes:

| Attribute | GT source (thesis) | Pipeline PDF (journal) |
|-----------|-------------------|----------------------|
| Document type | PhD thesis | Peer-reviewed journal article |
| Key treatment | Zn application (ZnSO₄) | Irrigation × nitrogen |
| Primary outcome | Grain Zn concentration + mineral elements | Amino acid composition, EAAI, PDCAAS |
| Grain Zn data | Present (the GT data) | Absent |
| DOI/access | No DOI; thesis, likely Chinese repository | doi:10.1371/journal.pone.0178494 |

**Secondary cause: The extraction pipeline did not enforce the recon-stage "DO NOT EXTRACT" guidance.** The recon correctly identified the paper as irrelevant but the extraction stage ran anyway and produced 8 spurious observations (yield, protein, amino acids). These observations are internally consistent with the journal article's Table 2 and Table 3 data, but are completely wrong for the Hui 2023 meta-analysis schema.

**There is no extraction failure in the traditional sense** — the AI correctly read the available PDF. The PDF simply does not contain the required data.

---

## 6. Assessment

| Dimension | Assessment |
|-----------|-----------|
| Is grain Zn data in the pipeline PDF? | No — confirmed across all 15 pages |
| Is the pipeline PDF the correct source? | No — wrong document; thesis vs. journal article |
| Can the pipeline extract GT data from current PDF? | No — data structurally absent |
| Was recon correct? | Yes — correctly flagged paper as irrelevant and Zn-free |
| Was extraction correct? | Partially — correctly found no Zn; incorrectly extracted irrelevant variables anyway |
| Is this a fixable extraction error? | No — fix requires obtaining the actual PhD thesis |
| GT data quality | Plausible; internally consistent with paper's yield range |

**Classification: WRONG SOURCE DOCUMENT**

This is not an AI extraction failure. The pipeline cannot extract data that does not exist in the provided PDF. The correct source — Zhang P.P. (2017) doctoral thesis, Henan Agricultural University — is a Chinese-language PhD thesis that is not publicly available via DOI and was likely accessed by Hui et al. through institutional or direct-contact channels.

**Recommended action:** Attempt to obtain the PhD thesis directly (via ResearchGate, CNKI — China National Knowledge Infrastructure, or by contacting the corresponding author Chenyang Wang at xmzxwang@163.com or co-author Tiancai Guo). The thesis title in Chinese would be approximately "氮和锌施用对冬小麦矿质元素积累分配及籽粒品质的调控效应." If the thesis cannot be obtained, these 8 GT observations (obs IDs 189–196) must be excluded from validation metrics for the Hui 2023 dataset, and the paper should be flagged as `source_unavailable` rather than `extraction_failure`.
