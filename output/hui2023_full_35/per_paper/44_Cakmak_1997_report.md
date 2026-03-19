# Extraction Quality Report: 44_Cakmak_1997

**Paper (filename label):** Cakmak, I., Ekiz, H., Yilmaz, A., Torun, B., Koleli, N., Gultekin, I., Alkan, A., Eker, S. (1997). Differential response of rye, triticale, bread and durum wheats to zinc deficiency in calcareous soils. *Plant and Soil*, 188(1), 1–10.

**Paper (actual PDF content):** Gomez-Coronado, F., Poblaciones, M.J., Almeida, A.S., Cakmak, I. (2016). Zinc (Zn) concentration of bread wheat grown under Mediterranean conditions as affected by genotype and soil/foliar Zn application. *Plant and Soil*, 401:331–346. DOI: 10.1007/s11104-015-2758-0

**Match summary:** 0/5 GT matched (zero-match)

**Root cause category:** PDF MISLABELING — the file named "44_Cakmak_1997.pdf" contains an entirely different paper (Gomez-Coronado et al. 2016) from the one cited in MOESM5 (Cakmak et al. 1997). The AI extracted correctly from the wrong paper. The 1997 paper appears to be absent from the source PDF folder.

---

## 1. Paper Design

### GT Paper (Cakmak et al. 1997 — what MOESM5 cites)
- **Title:** Differential response of rye, triticale, bread and durum wheats to zinc deficiency in calcareous soils
- **Journal:** Plant and Soil, 188(1), 1–10
- **Country:** Turkey (Central Anatolia)
- **Crop:** Rye, triticale, bread wheat, durum wheat
- **Intervention:** Soil ZnSO4 application (23 kg Zn ha⁻¹)
- **Control:** No Zn application
- **Soil:** Highly calcareous (pH 7.9, CaCO3 = 32%, available Zn = 0.09 mg kg⁻¹ — severely Zn-deficient)
- **Replicates:** n = 5
- **Design:** Presumably a field trial comparing species/cultivars × Zn treatments on calcareous Turkish soils
- **Key context:** Cakmak is one of the world's leading Zn-in-wheat researchers; this 1997 paper is a foundational study on species-level Zn response in Zn-deficient calcareous soils typical of the Anatolian Plateau

### Actual PDF (Gomez-Coronado et al. 2016 — what the AI read)
- **Title:** Zinc (Zn) concentration of bread wheat grown under Mediterranean conditions as affected by genotype and soil/foliar Zn application
- **Journal:** Plant and Soil, 401:331–346 (2016)
- **Country:** Portugal (Elvas, SE Portugal, 38°53' N, 7°2' W, 220 m asl)
- **Crop:** Bread wheat (*Triticum aestivum* L.) — 10 advanced breeding lines (INIAV-1 through INIAV-10) + 3 commercial varieties (Ardila, Roxo, Nabao)
- **Intervention:** Four Zn treatments: (i) control (no Zn), (ii) soil Zn application (50 kg ZnSO4·7H2O ha⁻¹), (iii) foliar Zn application (0.5% ZnSO4·7H2O at anthesis and milk stages), (iv) soil+foliar combined
- **Design:** Split-plot, 3 replications; main plots = Zn treatments, subplots = cultivars
- **Seasons:** 2010–2011 and 2012–2013 (2011–2012 excluded due to Tilletia caries infection)
- **Soil:** Xerofluvents, loamy, pH 7.4±0.12, SOM 8.2±0.12%, DTPA-Zn 0.30±0.02 mg kg⁻¹ (Zn-deficient, below critical threshold of 0.5 mg kg⁻¹)
- **Outcomes measured:** Total grain Zn (mg kg⁻¹), total grain Zn (g ha⁻¹), phytate concentration, phytate:Zn molar ratio, grain yield (kg ha⁻¹), thousand grain weight, test weight, grain protein (%), SDS, ash (%)
- **Primary data table:** Table 3 — mean ± SE for each cultivar × Zn treatment × year combination

---

## 2. Grain Zn Data in PDF

Table 3 of the actual PDF (Gomez-Coronado et al. 2016) contains extensive grain Zn data organized by cultivar, year, and Zn treatment. Key summary values from the text and tables are:

**Total grain Zn concentration (mg kg⁻¹) — control group (no Zn) by year:**

| Year | Grand mean (control) | Range across cultivars |
|------|---------------------|----------------------|
| 2010–2011 | ~26.2 ± 1.2 | 31–46 mg kg⁻¹ |
| 2012–2013 | ~11–20 mg kg⁻¹ | varies widely |

**Effect of Zn treatments (grand means across cultivars):**

| Treatment | 2010–2011 increase | 2012–2013 increase |
|-----------|-------------------|-------------------|
| Soil Zn | +~12% (ns) | +~12% (significant) |
| Foliar Zn | +~156% | +~260% |
| Soil+Foliar | +~156% | +~260% |

**Selected INIAV-1 control values from Table 3 (the cultivar the AI focused on):**
- 2010–2011: Control = 46±4 mg kg⁻¹
- 2012–2013: Control = 14±2 mg kg⁻¹

These values (control grain Zn of 14–46 mg kg⁻¹) come from Portuguese Mediterranean conditions with pH 7.4 soils. They bear no relationship to the GT values (9.3–12 mg kg⁻¹) from Turkish calcareous soils (pH 7.9, CaCO3 = 32%).

**Variance reporting:** Mean ± SE throughout all tables; n = 3 (3 replications per split-plot). Statistics used: Fisher's LSD (P≤0.05).

---

## 3. AI Extraction Results

The AI pipeline (Claude + Gemini consensus) extracted 9 consensus observations from the actual PDF content (Gomez-Coronado et al. 2016, Table 3), plus 15 additional Claude-only observations that were rejected by the voting system. Kimi extracted 0 observations (extraction failure for this paper).

**Consensus observations (9 total):**

| Element | Cultivar | Year | Treatment | Control mean | Treatment mean | Effect (%) |
|---------|----------|------|-----------|-------------|----------------|------------|
| Total grain Zn (mg kg⁻¹) | INIAV-1 | 2010–2011 | Soil | 46 | 33 | -28.3% |
| Total grain Zn (mg kg⁻¹) | INIAV-1 | 2010–2011 | Foliar | 46 | 57 | +23.9% |
| Total grain Zn (mg kg⁻¹) | INIAV-1 | 2010–2011 | Soil+Foliar | 46 | 58 | +26.1% |
| Total grain Zn (mg kg⁻¹) | INIAV-1 | 2012–2013 | Soil | 14 | 24 | +71.4% |
| Total grain Zn (mg kg⁻¹) | INIAV-1 | 2012–2013 | Foliar | 14 | 51 | +264.3% |
| Total grain Zn (g ha⁻¹) | INIAV-1 | 2010–2011 | Soil | 106 | 110 | +3.8% |
| Grain yield (kg ha⁻¹) | INIAV-1 | 2010–2011 | Soil | 2341 | 3370 | +44.0% |
| Phytate:Zn ratio | INIAV-1 | 2010–2011 | Soil | 17 | 23 | +35.3% |
| Phytate:Zn ratio | INIAV-1 | 2010–2011 | Foliar | 17 | 14 | -17.6% |

**Key AI extraction observations:**
- The AI correctly identified Table 3 as the primary data source
- Variance type correctly identified as SE (confirmed by "mean ± standard error" in paper text)
- n = 3 correctly identified (split-plot with 3 replications)
- The recon warned about the scanned/OCR nature of the PDF; the actual PDF is typeset (not scanned), indicating the recon was reading wrong metadata
- The recon described a "scanned PDF" with "OCR errors" — this is false for the 2016 paper, suggesting the recon AI may have been operating on some cached or corrupted state
- Recon also warned about "two growing seasons (2010-2011 and 2012-2013)" — this is correct for the 2016 paper, not what the 1997 paper would contain
- **Verification flags raised:** One T/C swap flag (Soil treatment, 2010–2011: control 46 > treatment 33 mg kg⁻¹, which is biologically plausible for soil-only Zn in drought year). Several variance_type flags (heuristic suggested SD more likely than SE given CV values).

**The AI extracted real, internally consistent data from the wrong paper.** The extraction quality relative to the actual PDF content is reasonable — the consensus values for INIAV-1 match Table 3 of the 2016 paper well. The zero match against GT is entirely due to the PDF mislabeling.

---

## 4. GT Data (5 MOESM5 rows)

Source: MOESM5_dataset.xlsx, Sheet "Data 2 Soil application", Study ID = 44

All 5 rows refer to the same experiment (Cakmak et al. 1997, Turkey), same soil conditions, same ZnSO4 treatment (23 kg Zn ha⁻¹), same n = 5:

| Obs ID | Grain Zn control (mg kg⁻¹) | Grain yield control (kg ha⁻¹) | Zn biofortification index | Species/cultivar |
|--------|--------------------------|-------------------------------|--------------------------|-----------------|
| 311 | 9.7 | 2032 | 0.735 | (bread wheat, cultivar unspecified) |
| 312 | 9.3 | 1240 | 0.239 | (cultivar 2) |
| 313 | 11.7 | 366 | 0.309 | (cultivar 3) |
| 314 | 10.5 | 316 | 0.283 | (cultivar 4) |
| 315 | 12.0 | 152 | 0.383 | (cultivar 5) |

**GT data interpretation:**
- These 5 rows likely represent 5 different cultivar/species entries (rye, triticale, bread wheat, durum wheat) tested in the Turkish calcareous soil trial
- Grain Zn concentrations are very low (9.3–12 mg kg⁻¹) — consistent with severely Zn-deficient Turkish calcareous soils (DTPA-Zn = 0.09 mg kg⁻¹)
- Grain yields are extremely variable (152–2032 kg ha⁻¹), consistent with the paper's focus on differential species response under severe Zn deficiency
- The Zn biofortification index values (0.24–0.74) indicate the ratio of treated-to-control Zn accumulation
- Available Zn grouping "≤0.5" and CaCO3 = 32% confirm this is Turkish calcareous soil
- **These values are entirely incompatible with the Portuguese 2016 paper's data** (control Zn: 14–46 mg kg⁻¹; Portuguese soils with pH 7.4 and lower calcareous content)

**What the GT expects the AI to extract:**
- Grain Zn concentrations in the range 9–12 mg kg⁻¹ for the no-Zn control
- Grain yields of 152–2032 kg ha⁻¹ (very low — drought/Zn-deficiency-stressed Turkish field)
- n = 5
- Country: Turkey
- Soil application of ZnSO4 at 23 kg Zn ha⁻¹
- The Hui meta-analysis placed this paper in the "Soil application" sheet, meaning the primary comparison of interest is control vs. soil Zn

---

## 5. Root Cause (Why No Match?)

### Primary Cause: PDF Mislabeling (Definitive)

The file `44_Cakmak_1997.pdf` does **not** contain the Cakmak et al. 1997 paper. It contains:

> Gomez-Coronado, F., Poblaciones, M.J., Almeida, A.S., **Cakmak, I.** (2016). Zinc (Zn) concentration of bread wheat grown under Mediterranean conditions as affected by genotype and soil/foliar Zn application. *Plant and Soil*, 401:331–346.

The filename error is understandable: Ismail Cakmak is the last/senior author on the 2016 paper, and the research group is the same. Someone collecting PDFs likely saved the 2016 Cakmak-group paper under a filename referencing the 1997 Cakmak first-author paper, creating a collision.

**The actual Cakmak et al. 1997 paper** (Plant and Soil 188(1):1–10) is absent from the source PDF folder.

### Evidence confirming mislabeling:

| Feature | Expected (Cakmak 1997) | Actual PDF (Gomez-Coronado 2016) |
|---------|------------------------|----------------------------------|
| Year | 1997 | 2016 |
| First author | Cakmak, I. | Gomez-Coronado, F. |
| Cakmak role | First/corresponding | Last (senior) author |
| Country | Turkey | Portugal |
| Soil pH | 7.9 | 7.4 |
| Soil CaCO3 | 32% | Low (Xerofluvents, not calcareous) |
| DTPA-Zn | 0.09 mg kg⁻¹ | 0.30 mg kg⁻¹ |
| Control grain Zn | ~9–12 mg kg⁻¹ | 11–46 mg kg⁻¹ |
| n | 5 | 3 |
| Zn rate (soil) | 23 kg Zn ha⁻¹ | ~13.1 kg Zn ha⁻¹ (50 kg ZnSO4·7H2O) |
| Species | Rye, triticale, bread + durum wheat | Bread wheat only (13 cultivars) |
| Journal volume | 188(1) | 401 |

### Secondary Causes (would not have mattered given primary cause):

1. **Scale mismatch:** Even if the recon had flagged a discrepancy, the AI had no way to know it was reading the wrong paper — the paper is internally consistent and plausible as a Cakmak-group Zn study.

2. **Recon confusion artifacts:** The recon JSON contains internally inconsistent signals — it flagged "SCANNED PDF" and "OCR errors" which do not apply to the actual typeset 2016 paper. This suggests the recon stage may have partially failed or mixed signals from a different paper's metadata.

3. **Kimi failure (0 observations):** Kimi extracted nothing, leaving only Claude and Gemini. This reduced the consensus robustness. The tiebreaker fell to Claude (24 obs) being used as primary. However this is a secondary issue — even perfect three-model consensus would have extracted data from the 2016 paper, not the 1997 paper.

4. **No paper-identity verification step:** The pipeline has no step that verifies the PDF filename matches the paper's actual bibliographic metadata (title, authors, year, journal). Such a step would have caught this error immediately.

---

## 6. Assessment

### Match failure classification: IRRECOVERABLE (PDF absent)

The zero-match outcome is **not an extraction quality failure** — it is a **data preparation failure**. The AI extracted data correctly and consistently from the PDF it was given. The problem is that the PDF is the wrong paper.

### What the AI did correctly:
- Correctly identified Table 3 as the data source
- Correctly identified grain Zn (mg kg⁻¹) as the primary outcome
- Correctly identified the split-plot design with 3 replications
- Correctly extracted mean ± SE format
- Correctly distinguished control vs. soil vs. foliar vs. combined treatments
- Consensus between Claude and Gemini was high (9 agreed observations); only Kimi failed

### What cannot be fixed without the correct PDF:
- The 5 GT observations (MOESM5 obs IDs 311–315) require the actual Cakmak et al. 1997 paper, which reports grain Zn concentrations of 9–12 mg kg⁻¹ in Turkish calcareous soils across multiple cereal species/cultivars with n=5 under soil ZnSO4 fertilization at 23 kg Zn ha⁻¹
- This paper is not in the source PDF collection under any other filename

### Recommended actions:

1. **Locate and add the correct PDF:** Cakmak, I., Ekiz, H., Yilmaz, A., Torun, B., Koleli, N., Gultekin, I., Alkan, A., Eker, S. (1997). Differential response of rye, triticale, bread and durum wheats to zinc deficiency in calcareous soils. *Plant and Soil*, 188(1), 1–10. DOI: 10.1007/BF00015299

2. **Rename or separate the 2016 paper:** The Gomez-Coronado et al. 2016 paper contains valid Zn biofortification data and could potentially be a separate study entry in the meta-analysis if it meets inclusion criteria. It should be saved under a filename reflecting its actual content (e.g., `Gomez-Coronado_2016.pdf`).

3. **Add PDF identity verification to pipeline:** Before extraction, extract the title and first author from the PDF's first page and compare against the expected paper. A simple text match on year and first author surname would catch this class of error.

4. **Check other papers in the collection for similar mislabeling:** If PDFs were bulk-collected by a research group, there may be other files where a senior author's name (rather than first author's) was used in the filename, creating cross-paper collisions.

### Quantitative impact on validation:
- 0/5 GT observations matched → contributes to depressing Hui dataset r and ICC statistics
- The 5 GT obs represent grain Zn values from severely Zn-deficient Turkish calcareous soils (9–12 mg kg⁻¹) — a unique and scientifically important part of the Hui dataset (extreme low-Zn baseline). Their absence from the validated set is a meaningful gap.
- The extracted data (Portugal 2016) cannot substitute — the soil context, Zn concentrations, and experimental scale are all different.

### Overall assessment: NOT AN AI EXTRACTION FAILURE

The extraction system performed correctly given its inputs. This is a **source data curation error** that requires human intervention to resolve. The correct Cakmak et al. 1997 paper must be obtained and the mislabeled 2016 paper must be separately registered. No changes to extraction prompts or pipeline logic would improve this outcome.
