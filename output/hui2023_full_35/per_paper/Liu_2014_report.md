# Extraction Quality Report: Liu_2014

**Paper:** Uddin MN, Kaczmarczyk A, Vincze E (2014). Effects of Zn Fertilization on Hordein Transcripts at Early Developmental Stage of Barley Grain and Correlation with Increased Zn Concentration in the Mature Grain. *PLoS ONE* 9(9): e108546. doi:10.1371/journal.pone.0108546

**Match summary:** 0/10 GT matched (zero-match, all 3 application types)

---

## 1. Paper Design

The PDF retrieved for this paper is **a completely different study from the one cited in the Hui 2023 meta-analysis ground truth**. This is a file mislabeling error, not an extraction failure.

**PDF content (what is actually in the file):**
- **Authors:** Mohammad Nasir Uddin, Agnieszka Kaczmarczyk, Eva Vincze
- **Affiliation:** Department of Molecular Biology & Genetics, Aarhus University, Slagelse, Denmark
- **Title:** "Effects of Zn Fertilization on Hordein Transcripts at Early Developmental Stage of Barley Grain and Correlation with Increased Zn Concentration in the Mature Grain"
- **Journal:** PLoS ONE 9(9), September 2014
- **Species:** *Hordeum vulgare* cv. Golden Promise (**barley**, not wheat)
- **System:** Greenhouse (individual 1-L plastic pots, 200 g PindstrupUnimud soil, Denmark)
- **Design:** Three Zn fertilization levels (Low, Medium, High) applied via combined soil + foliar method. This is NOT a field trial.
- **Country:** Denmark (Research Centre Flakkebjerg, Slagelse)
- **Primary focus:** qRT-PCR hordein transcript quantification and grain Zn concentration via ICP-OES. Primary aim is molecular biology of storage proteins, not agronomic biofortification.
- **Grain Zn values reported:** in µg/g (= mg/kg), from individual biological replicates (Table 1), n=6/9/10 individual plants.

**GT citation (what the meta-analysis expects from "Liu_2014"):**
- **Authors:** Liu, D.Y., Pang, L.L., Zhang, W., Li, Z.X., Wang, X.Z., Yu, F.T., Zou, C.Q.
- **Title:** "Effects of different zinc fertilization methods on yield and grain Zn concentration of maize and wheat"
- **Journal:** Soil and Fertilizer Sciences in China, (04), 76-80, 2014
- **Species:** Wheat (and maize)
- **Country:** China
- **Design:** Field experiment, 2 sites (pH 7.2 and pH 8.2), multiple Zn application methods (soil, foliar, soil+foliar), n=4 replicates per treatment
- **Grain Zn values reported:** 32.0 mg/kg (Site 1) and 28.1 mg/kg (Site 2)

These are entirely different papers by different research groups in different countries on different species.

---

## 2. Grain Zn Data in the PDF

**Table 1 of the PDF (Uddin et al. 2014, barley greenhouse):**

| Treatment | Plant IDs | n | Average (µg/g) | St. Error |
|-----------|-----------|---|----------------|-----------|
| Low Zn | P2, P4, P6, P7, P11, P13 | 6 | 65.0 | 4.4 |
| Medium Zn | Q1, Q2, Q4, Q5, Q9, Q10, Q11, Q12, Q13 | 9 | 151.1 | 12.4 |
| High Zn | R1, R2, R4, R6, R7, R8, R11, R13, R14, R15 | 10 | 466.4 | 44.9 |

- Units are µg/g (equivalent to mg/kg)
- Variance type is Standard Error (SE), labeled "St. Error" in table header
- Values are biologically plausible for a greenhouse experiment with no Zn in soil and very high Zn supplementation (466 µg/g is extreme, as expected with 10 mM ZnSO4 foliar spray)
- The paper's Zn values (65–466 µg/g) are approximately 2–15x higher than typical field wheat grain Zn (15–45 mg/kg), consistent with the artificial greenhouse conditions

**Soil application details in PDF:**
- Low Zn: no additional Zn added to soil; foliar = water only
- Medium Zn: soil 0.25 mM ZnSO4·7H2O + foliar 1 mM ZnSO4·7H2O
- High Zn: soil 1 mM ZnSO4·7H2O + foliar 10 mM ZnSO4·7H2O
- Foliar applications: 4 mL per plant, twice a week, from 35 to 90 days after sowing

---

## 3. AI Extraction

The AI pipeline processed the PDF correctly given what was in the file, but extracted data from the wrong paper.

**Recon phase findings:**
- Correctly identified the paper as being about **barley** (Hordeum vulgare), and flagged this in a warning: "This paper uses barley instead of wheat - verify if barley studies are included in the wheat meta-analysis"
- Correctly identified Table 1 as the target table for grain Zn concentration
- Correctly identified SE as the variance type from the "St. Error" column header
- Correctly identified the combined soil+foliar application method
- Noted the paper was a scanned PDF with potential OCR errors
- Flagged "VAR-UNCLEAR" in warnings (despite correctly identifying SE)

**Consensus extraction output (3 observations after post-processing):**

| Obs | Treatment | Treatment mean (mg/kg) | Control mean (mg/kg) | Treatment variance | Control variance | Var type | n |
|-----|-----------|------------------------|----------------------|-------------------|-----------------|---------|---|
| 1 | Medium Zn (soil+foliar) | 151.1 | 65.0 | 12.4 | 4.4 | SE | 9 |
| 2 | High Zn (soil+foliar) | 466.4 | 65.0 | 44.9 | 4.4 | SE | 10 |
| 3 | Medium Zn (duplicate via Kimi tiebreaker) | 151.1 | 65.0 | 12.4 | 4.4 | SE | 9 |

- Post-processing removed 1 null-means observation and 0 duplicates (final count: 3)
- Claude and Gemini agreed on both observations; Kimi produced matching values independently
- Tiebreaker was triggered due to low consensus (0/2 = 0%), suggesting models agreed in values but not in metadata structure
- Verification flags correctly raised "magnitude" warnings (+132% and +618% effects are large but genuine for this greenhouse design)
- GRIM test failed on treatment mean 151.1 with n=9 (151.1 × 9 = 1359.9, not an integer), which is expected because ICP-OES measurements are continuous, not integer data — this GRIM failure is a false positive from the validation pipeline applying an inappropriate test to continuous chemical measurements

**What the AI extracted vs what GT expects:**

| Dimension | AI Extracted (Uddin 2014, barley) | GT Expected (Liu 2014, wheat) |
|-----------|-----------------------------------|-------------------------------|
| Species | Barley | Wheat |
| Country | Denmark | China |
| System | Greenhouse, individual pots | Field |
| Site 1 control Zn (mg/kg) | 65.0 (Low Zn = 0 added) | 32.0 (ambient soil Zn) |
| Site 2 control Zn (mg/kg) | — (single site) | 28.1 (ambient soil Zn) |
| Treatment Zn (mg/kg) | 151.1 (medium), 466.4 (high) | Not separately extracted |
| Soil available Zn (mg/kg) | Very low (greenhouse potting soil, <0.5) | 0.45 (Site 1), 0.33 (Site 2) |
| Zn rate (kg Zn/ha) | N/A (mM concentration) | 3.405 or 6.810 (soil); 2.179 (foliar) |
| n replicates | 6 (control), 9–10 (treatment) | 4 |
| Application sheets matched | None | Soil (obs 108–111), Foliar (obs 1–2), Soil+Foliar (obs 3–6) |

The numeric values share no overlap. The control means of 65.0 µg/g (barley, no-Zn greenhouse) vs 32.0 and 28.1 mg/kg (wheat, field) differ by approximately 2-fold in the same units. No matching threshold in the validation pipeline would connect these values.

---

## 4. GT Data (all 3 sheets)

The GT covers 10 observations from Liu D.Y. et al. 2014 (Chinese journal, Soil and Fertilizer Sciences in China):

**Sheet: Data 2 — Soil application (study_id = 16, obs IDs 108–111)**

| Obs ID | Site | Available soil Zn (mg/kg) | pH | Zn rate (kg/ha) | n | Grain yield (kg/ha) | Grain Zn (mg/kg) |
|--------|------|---------------------------|-----|-----------------|---|---------------------|------------------|
| 108 | Site 1 | 0.45 | 7.2 | 3.405 (ZnSO4) | 4 | 6914 | 32.0 |
| 109 | Site 1 | 0.45 | 7.2 | 6.810 (ZnSO4) | 4 | 6914 | 32.0 |
| 110 | Site 2 | 0.33 | 8.2 | 3.405 (ZnSO4) | 4 | 8050 | 28.1 |
| 111 | Site 2 | 0.33 | 8.2 | 6.810 (ZnSO4) | 4 | 8050 | 28.1 |

Note: Observations 108/109 share the same grain Zn value (32.0) and 110/111 share 28.1 — these are the treatment group means; the control values are not separately listed in this sheet's visible columns (the Hui meta-analysis uses a within-paper control reference for effect size calculation).

**Sheet: Data 3 — Foliar application (study_id = 1, obs IDs 1–2)**

| Obs ID | Site | Spraying concentration (g Zn/L) | Frequency | Timing | n | Grain Zn (mg/kg) |
|--------|------|----------------------------------|-----------|--------|---|------------------|
| 1 | Site 1 (pH 7.2) | 0.0908 | 3× | Timing code 9 | 4 | 32.0 |
| 2 | Site 2 (pH 8.2) | 0.0908 | 3× | Timing code 9 | 4 | 28.1 |

**Sheet: Data 4 — Soil+Foliar application (study_id = 2, obs IDs 3–6)**

| Obs ID | Site | n | Grain yield (kg/ha) | Grain Zn (mg/kg) | Grain Zn accum. (g/kg) |
|--------|------|---|---------------------|------------------|------------------------|
| 3 | Site 1 | 4 | 6914 | 32.0 | 221.248 |
| 4 | Site 1 | 4 | 6914 | 32.0 | 221.248 |
| 5 | Site 2 | 4 | 8050 | 28.1 | 226.205 |
| 6 | Site 2 | 4 | 8050 | 28.1 | 226.205 |

**GT grain Zn summary:** Two distinct field-realistic values: 32.0 mg/kg (Site 1, pH 7.2, available Zn 0.45 mg/kg) and 28.1 mg/kg (Site 2, pH 8.2, available Zn 0.33 mg/kg). These are consistent with typical wheat grain Zn concentrations in Chinese field soils with moderate Zn deficiency.

---

## 5. Root Cause

**Primary cause: File mislabeling — the PDF file named "Liu_2014.pdf" does not contain the Liu et al. 2014 Chinese-language paper cited in the Hui 2023 meta-analysis.**

The PDF contains Uddin MN, Kaczmarczyk A, Vincze E (2014) published in PLoS ONE, a completely unrelated English-language barley molecular biology paper. The filename coincidence is purely by year (both 2014) and first-author initial similarity (Liu vs Uddin).

The correct paper — Liu DY, Pang LL, Zhang W, Li ZX, Wang XZ, Yu FT, Zou CQ (2014), published in the Chinese journal *Soil and Fertilizer Sciences in China* (土壤肥料) — is likely inaccessible through standard open-access PDF searches because:
1. It is published in a Chinese-language journal with limited English-language indexing
2. The journal title "Soil and Fertilizer Sciences in China" may not have a straightforward DOI-based PDF retrieval path
3. The paper was likely downloaded as a name-matched false positive from an English-language database

**Secondary causes (all flow from the primary):**

1. **Species mismatch**: The retrieved PDF covers barley (Hordeum vulgare), while the GT expects wheat (Triticum aestivum). The recon phase correctly flagged this but the pipeline continued to extract regardless.

2. **Scale mismatch**: The Uddin et al. greenhouse barley values (65–466 µg/g) are 2–15x higher than the GT wheat field values (28–32 mg/kg). Even if the matching algorithm had no species filter, no tolerance threshold would bridge a 2-fold difference on the control mean alone.

3. **Experimental system mismatch**: Greenhouse pot experiment (Denmark) vs. field trial (China). These produce fundamentally different Zn concentration ranges because greenhouse soils can be Zn-depleted to near zero, amplifying treatment effects.

4. **No GT control values available for comparison**: The MOESM5 sheets for this paper record only treatment grain Zn values (32.0 and 28.1 mg/kg); the control (unfertilized) values for Liu et al. 2014 are not shown in the GT data extracted, making any partial matching impossible even in principle.

5. **Language barrier for source paper**: The true Liu et al. 2014 is a Chinese-language journal article. Even if the correct PDF were retrieved, OCR on Chinese-language text would require specialized handling not present in the current pipeline.

---

## 6. Assessment

**Classification: Wrong Paper — File Mislabeling**

This is not an AI extraction failure. The AI correctly processed the content of the PDF it was given:
- It accurately read Table 1 values (65.0, 151.1, 466.4 µg/g)
- It correctly identified the variance type (SE)
- It correctly structured two treatment observations (Medium Zn and High Zn vs Low Zn control)
- It flagged the species mismatch (barley, not wheat) as a warning in the recon phase

The zero-match outcome is 100% attributable to the wrong PDF being in the source data folder. The Hui 2023 meta-analysis cites a Chinese-language wheat field study; the file contains an English-language barley greenhouse molecular biology paper.

**Correctability:** High, but requires obtaining the actual PDF. The true paper (Liu DY et al., Soil and Fertilizer Sciences in China, 2014) must be sourced from a Chinese agricultural database (e.g., CNKI — China National Knowledge Infrastructure, or WanFang Data). Once the correct PDF is obtained, extraction should be straightforward: the paper reports simple grain Zn concentration data for two field sites with clear Zn fertilizer treatments.

**Impact on validation statistics:**
- This paper contributes 10 GT observations across all three application sheets (4 soil, 2 foliar, 4 soil+foliar)
- All 10 are unmatched, artificially depressing the overall match rate
- These 10 observations should be excluded from the validation denominator when reporting extraction accuracy, or flagged as "source unavailable" rather than "extraction failure"

**Recommended action:**
1. Source the correct PDF from CNKI or WanFang using the full citation: Liu DY, Pang LL, Zhang W, Li ZX, Wang XZ, Yu FT, Zou CQ (2014). Effects of different zinc fertilization methods on yield and grain Zn concentration of maize and wheat. *Soil and Fertilizer Sciences in China* (04): 76–80.
2. Replace the current `Liu_2014.pdf` in the source data folder with the correct file
3. Re-run extraction for this paper only
4. Expected result: 10/10 GT match at grain Zn values of 32.0 mg/kg (Site 1) and 28.1 mg/kg (Site 2)
