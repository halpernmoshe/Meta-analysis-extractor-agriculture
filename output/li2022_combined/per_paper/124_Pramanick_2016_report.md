# Extraction Quality Report: 124_Pramanick_2016

**Paper:** Pramanick B, Brahmachari K, Ghosh A (2013). "Effect of seaweed saps on growth and yield improvement of green gram." *African Journal of Agricultural Research* 8(13):1180–1186. DOI: 10.5897/AJAR12.1894

**Note on filename:** The file is labeled "2016" but the paper was published in 2013 (accepted March 2013). This is a filename mislabeling; the publication year is 2013.

**Match summary:** 8 matched pairs, r = 0.949, MAE = 11.5 pp, 0 unmatched GT rows, 2 unmatched JSON observations.

---

## 1. Paper Design

**Crop:** Green gram (*Vigna radiata* L.) — a pulse crop also known as mung bean. Grown in pre-kharif season (sown 25 February 2012, harvested 5 May 2012) on sandy clay loam inceptisol at Uttar Chandamari village, Nadia district, West Bengal, India.

**Experimental design:** Randomized block design (RBD), 3 replications, plot size 5 × 6 m.

**Products tested:** Two seaweed sap products derived from:
- *Kappaphycus* sp. (red alga; higher yield response)
- *Gracilaria* sp. (red alga; lower yield response)

Seaweed saps were prepared as liquid filtrates by blending fresh algae, with the undiluted filtrate defined as 100% concentration. Application was by foliar spray twice (at 20 DAS and 40 DAS), mixed with surfactant (Active 80 at 0.5 ml L⁻¹), at 650 L ha⁻¹ spray volume.

**Treatments (Table 1):**

| Treatment | Description |
|-----------|-------------|
| T1 | 2.5% *Kappaphycus* sap + RDF |
| T2 | 5% *Kappaphycus* sap + RDF |
| T3 | 10% *Kappaphycus* sap + RDF |
| T4 | 15% *Kappaphycus* sap + RDF |
| T5 | 2.5% *Gracilaria* sap + RDF |
| T6 | 5% *Gracilaria* sap + RDF |
| T7 | 10% *Gracilaria* sap + RDF |
| T8 | 15% *Gracilaria* sap + RDF |
| T9 | RDF + Water spray (control) |
| T10 | 7.5% *Kappaphycus* sap + 50% RDF |

**Fertilizer:** Recommended dose of fertilizer (RDF) for green gram = 20:40:40 kg ha⁻¹ N, P₂O₅, K₂O, applied basally.

**Statistical note:** ANOVA differences considered significant at p < 0.05.

---

## 2. Yield Tables in the PDF

### Table 5: Seed Yield and Stover Yield

The primary data source is Table 5 (page 4 of the PDF), titled "Effect of treatments on yield components and seed and stover yield of green gram."

**Seed yield (kg ha⁻¹):**

| Treatment | Seed Yield (kg ha⁻¹) | vs Control (T9) |
|-----------|---------------------|-----------------|
| T9 (Control) | 910.3 | — |
| T1 (Kappa 2.5%) | 1085.6 | +19.26% |
| T2 (Kappa 5.0%) | 1090.3 | +19.77% |
| T3 (Kappa 10.0%) | 1158.6 | +27.28% |
| T4 (Kappa 15.0%) | 1265.0 | +38.97% |
| T5 (Graci 2.5%) | 995.3 | +9.34% |
| T6 (Graci 5.0%) | 1036.5 | +13.86% |
| T7 (Graci 10.0%) | 1103.0 | +21.17% |
| T8 (Graci 15.0%) | 1216.1 | +33.59% |
| T10 (Kappa 7.5% + 50% RDF) | 1101.7 | +21.03% |

SEm(±) = 8.12 kg ha⁻¹; CD at 5% = 24.33 kg ha⁻¹.

**Stover yield (kg ha⁻¹):**

| Treatment | Stover Yield (kg ha⁻¹) |
|-----------|------------------------|
| T9 (Control) | 3712.7 |
| T1 | 4157.9 |
| T2 | 4199.5 |
| T3 | 4372.1 |
| T4 | 5220.3 |
| T5 | 3909.7 |
| T6 | 4125.3 |
| T7 | 4657.8 |
| T8 | 5107.2 |
| T10 | 4298.3 |

SEm(±) = 9.07; CD at 5% = 27.12.

**Internal consistency check:** The paper's own results text (page 4) states: "The treatment T4 (15% *Kappaphycus*-sap + RDF) showed the maximum increase in yield over control to the extent of 38.97% and this treatment was followed by the treatments T8 (15% *Gracilaria*-sap + RDF), T3 (10% *Kappaphycus*-sap + RDF), T7 (10% *Gracilaria*-sap + RDF), T2 (5% *Kappaphycus*-sap + RDF) and T6 (5% *Gracilaria*-sap + RDF) recording 33.58, 27.28, 21.17, 19.77 and 13.86% yield increase, respectively over control." This exactly matches the values computable from Table 5 with T9 as the denominator.

There are no additional yield tables. Table 4 contains growth parameters (plant height, dry matter, CGR, LAI) and Table 6 contains nutrient uptake data — neither is a yield outcome for this meta-analysis.

---

## 3. AI Extraction Results

The extraction (data file) retrieved 10 observations: 9 seed yield observations (one per treatment except T9) plus 1 stover yield observation (T3 only, which happens to be the numerically largest stover effect). All seed yield values and the stover yield value match Table 5 exactly to the decimal place. The extraction correctly identified:

- Crop: green gram (recorded as blackgram in GT — see note below)
- Control: T9 (RDF + water spray), control mean = 910.3 kg ha⁻¹
- Product identity: Kappaphycus vs Gracilaria correctly distinguished
- Concentration levels: 2.5%, 5%, 10%, 15% correctly read
- The reduced-fertilizer arm (T10: 7.5% Kappa + 50% RDF) correctly extracted as a separate observation
- n = 3 replications correctly identified
- Units: kg ha⁻¹

**Unmatched JSON observations:**
1. *Stover yield (json_idx 3):* The extraction also captured stover yield for T3 (Kappa 10%): control 3712.7, treatment 4372.1 kg ha⁻¹. Li 2022 did not include stover yield for this paper, so this observation is correctly identified as unmatched. The extraction of stover yield is not wrong — it reflects a legitimate reading of Table 5.
2. *T10 seed yield (json_idx 9):* Kappaphycus 7.5% at 50% RDF (control 910.3, treatment 1101.7 kg ha⁻¹). Li 2022 included only the full-RDF dose-response series (T1–T8) for this paper; T10 is a partial-fertilizer exploratory arm not represented in the 8 GT rows. Again, the extraction is correct — this treatment exists in Table 5 — but falls outside the Li 2022 inclusion scope.

---

## 4. Why Is MAE = 11.5 pp Despite Good r = 0.949?

### 4.1 The Core Issue: GT Effect Sizes Do Not Match the Published Table

The extraction reads the PDF correctly. The MAE arises entirely from a systematic discrepancy between the GT database values and the published Table 5 values — not from any extraction error.

**Per-pair comparison:**

| Treatment | GT effect (%) | Ext effect (%) | |diff| | Paper text |
|-----------|--------------|----------------|--------|------------|
| Kappa 2.5% | 24.69 | 19.26 | 5.43 pp | N/A |
| Kappa 5.0% | 26.48 | 19.77 | 6.71 pp | 19.77% |
| Kappa 10.0% | 43.69 | 27.28 | 16.41 pp | 27.28% |
| Kappa 15.0% | 51.06 | 38.97 | 12.09 pp | 38.97% |
| Graci 2.5% | 22.57 | 9.34 | 13.23 pp | N/A |
| Graci 5.0% | 24.58 | 13.86 | 10.72 pp | 13.86% |
| Graci 10.0% | 35.08 | 21.17 | 13.91 pp | 21.17% |
| Graci 15.0% | 47.15 | 33.59 | 13.56 pp | 33.58% |
| **Mean** | | | **11.51 pp** | |

The paper text explicitly states effects of 38.97%, 33.58%, 27.28%, 21.17%, 19.77%, and 13.86% — these match the extraction exactly and contradict the GT values.

### 4.2 Characteristics of the Discrepancy

**All 8 GT effect sizes are higher than the extraction** — there are no cases where GT is lower. This is a strictly one-directional bias of mean +11.51 pp (which equals the MAE exactly, confirming there is no variance in the sign of the error).

**The r = 0.949 is high** because the rank ordering is perfectly preserved: both GT and extraction agree that Kappa > Graci at each dose, and that effects increase with dose.

**The GT raw values use an unusual encoding.** The GT ctrl_mean = 0.0895 and treat means range from 0.1097 to 0.1352 — approximately 10,171× smaller than the kg ha⁻¹ values in the PDF (910.3 to 1265.0). When the GT values are scaled up by this factor, the treatment means approximate but do not match the PDF Table 5 values (differences of 3–11%). This suggests the Li 2022 coder may have extracted data from a visual source (bar chart) or from a slightly different version of the paper, yielding numerical values that differ from the published table.

**The GT SDs are not from the PDF table.** Table 5 provides a single pooled SEm = 8.12 kg ha⁻¹ for all treatments (implying SD = SEm × √n = 8.12 × √3 ≈ 14.1 kg ha⁻¹). The GT treatment SDs are all exactly 13.4% of their respective means (CV = 0.134 for all 9 values including control). This constant CV pattern indicates that the Li 2022 meta-analysis imputed treatment SDs proportionally from the control SD, rather than reading them from the published SEm. This is a plausible imputation strategy but produces values incompatible with the original data.

### 4.3 Largest Diverging Pairs

The three pairs with the largest discrepancies all involve mid-to-high doses:

- **Kappa 10%:** GT = 43.69%, ext = 27.28%, |diff| = 16.41 pp. This is the worst pair.
- **Graci 10%:** GT = 35.08%, ext = 21.17%, |diff| = 13.91 pp.
- **Graci 15%:** GT = 47.15%, ext = 33.59%, |diff| = 13.56 pp.

The smallest divergence is at Kappa 2.5% (5.43 pp). The discrepancy tends to grow with dose, suggesting that if the GT coder read from a bar chart, the higher bars may have been over-read more than the lower ones.

---

## 5. Assessment

### Is the extraction correct?

**Yes, the extraction is correct.** The AI system read Table 5 accurately:
- All 9 seed yield values (T1–T9 plus T10) match the PDF to the decimal.
- The single stover yield value (T3 = 4372.1 kg ha⁻¹, control 3712.7 kg ha⁻¹) is likewise exact.
- The computed effect sizes for T4 (38.97%) and T8 (33.59%) match the paper's explicit text verbatim.
- Treatment-to-product mapping (Kappaphycus vs Gracilaria), dose levels, n, and units are all correctly extracted.
- The extraction correctly distinguished the partial-fertilizer arm (T10, 50% RDF) from the main dose-response series and retained it as a separate observation.

### What explains the MAE?

The MAE of 11.5 pp is entirely explained by the GT database containing higher effect sizes than those published in the paper. The extraction matches the source paper; the GT does not. This is not an extraction failure.

**Probable cause of GT discrepancy:** The Li 2022 coder appears to have obtained yield values from a source (possibly a bar chart, a pre-publication version, or a data transcription) that gave systematically higher treatment means and/or a lower control mean than Table 5. The constant-CV SD imputation in the GT data (all SDs = 13.4% of mean) is a further sign that the GT encoding involved post-processing steps that introduced divergence from the original published numbers.

### Minor issues in the extraction

1. The data file records the crop as "not specified" for cultivar, which is correct (no cultivar name is mentioned in the paper).
2. The moderator field records the year as "2012" (growing season), which is appropriate. The publication year is 2013, not 2016 as the filename suggests.
3. No variance values were extracted (no SE or SD columns appear in Table 5 — only SEm and CD at 5% for the pooled ANOVA error). The extraction does not provide variance, which reflects the actual table format. The GT SDs are imputed, not read from the PDF.
4. The extraction did not capture the stover yields for T1, T2, T4, T5, T6, T7, T8, T9, and T10 (only T3 stover yield was extracted). This is not a problem for Li 2022 matching since the GT only contains seed yield, but a comprehensive extraction would include all stover yield rows from Table 5.

### Overall quality rating

**High quality** for the seed yield extraction. The AI correctly identified the primary yield outcome table, read all numerical values precisely, distinguished the two seaweed species across four dose levels, and properly separated the reduced-fertilizer exploratory arm. The match statistics (r = 0.949, MAE = 11.5 pp) reflect a GT encoding anomaly rather than any deficiency in the AI extraction.
