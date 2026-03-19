# Extraction Quality Report: Dong_2018

**Paper (PDF):** Xia H, Xue Y, Liu D, Kong W, Xue Y, Tang Y, Li J, Li D and Mei P (2018). Rational Application of Fertilizer Nitrogen to Soil in Combination With Foliar Zn Spraying Improved Zn Nutritional Quality of Wheat Grains. *Frontiers in Plant Science* 9:677. doi: 10.3389/fpls.2018.00677

**Paper (GT / MOESM5):** Dong M., 2017. Effects of foliar zinc application on different layers of grain quality and physiological mechanisms in wheat. Thesis for Master's Degree. Nanjing Agricultural University. [Study ID 31 in Data 3 Foliar application sheet]

**Match summary:** 0/5 GT obs matched (zero-match) — 32 obs extracted, 3 GT obs (study_id=31), 0 match

---

## 1. Paper Design

**PDF content (Xia et al. 2018 — what the AI actually read):**

- Field experiment, 2013-2014 growing season, Yinmaquan Experimental Station, Shandong, China (36.4°N, 117.5°E)
- Split-split-plot factorial design: 3 factors × 4 replicates
  - Main plot: N application rate (75, 200, 275 kg N ha−1)
  - Subplot: Cultivar (4 wheat cultivars: Jinan 17, Jimai 20, Jimai 22, Luyuan 502)
  - Sub-subplot: Foliar spray treatment (3 levels: deionized water control; ZnSO4·7H2O 0.4% w/v; ZnSO4·7H2O 0.4% + sucrose 3.0% w/v)
- ALL plots received 30 kg ZnSO4·7H2O ha−1 soil application at planting (so no zero-Zn soil control)
- Foliar sprays applied 4 times at 5-day intervals starting 5 days after wheat flowering
- Primary outcome: Grain Zn concentration (mg kg−1) in whole flour
- Tables 2 and 3 contain the Zn concentration data

**GT source (Dong M. 2017 thesis — what the GT represents):**

- A Master's thesis from Nanjing Agricultural University (not a journal article)
- Three observations in MOESM5 (obs IDs 341, 342, 343; all study_id=31)
- Single N rate (240 kg N ha−1), single Zn rate (2 kg Zn ha−1 foliar, concentration 0.081 g Zn L−1), spraying frequency = 2 times, n = 3 replicates
- Grain Zn control (no foliar Zn) = 18.15 mg kg−1 for all three observations
- Three treatment rows represent different foliar Zn conditions/timings (treat values: 27.43, 37.4, 27.67 mg kg−1)
- Effects: +51.1%, +106.1%, +52.5%

**Key structural difference:** Xia et al. 2018 has 3 N-rates × 4 cultivars × 2 Zn treatments = 24 granular grain Zn observations (Table 3) plus 2 main-effect observations (Table 2). Dong 2017 thesis has a simpler design with a control grain Zn concentration of 18.15 mg kg−1 — a value that does not appear anywhere in the Xia 2018 PDF.

---

## 2. Grain Zn Data in PDF

The Xia et al. 2018 PDF (pages 4-5) contains clear grain Zn concentration data:

**Table 2 — Main effects (marginal means across all cultivars and N rates):**

| Foliar Treatment | Zn conc. (mg kg−1) | Zn yield (g ha−1) | TAZ (mg d−1) | PA/Zn | PA×Ca/Zn |
|---|---|---|---|---|---|
| Deionized water (control) | 40.9c | 303.7c | 1.4c | 20.8a | 234.2a |
| ZnSO4·7H2O | 52.0b | 382.0b | 1.8b | 16.0b | 178.1b |
| ZnSO4·7H2O + Sucrose | 56.5a | 423.7a | 2.0a | 15.1b | 168.6b |
| LSD0.05 | 1.6 | 19.0 | 0.1 | 0.8 | 11.4 |

**Table 3 — Interaction effects (grain Zn concentration, mg kg−1 — selected values):**

Control values by cultivar × N rate combination range from 34.9 to 47.7 mg kg−1. Treatment values range from 48.3 to 60.9 mg kg−1. All values are substantially higher than 18.15 mg kg−1 (the GT control value). Variance reported as LSD0.05. n=4 replicates.

Statistical analysis: SAS ANOVA, Fisher's LSD at P ≤ 0.05, 0.01, or 0.001.

---

## 3. AI Extraction Results

The AI (Claude only — Kimi extracted 0 observations, Gemini also 0) extracted 34 total consensus observations, all flagged as low confidence (single-model fallback). Of these, 26 are grain Zn concentration observations and 8 are secondary outcomes (Zn yields, TAZ, PA/Zn, PA×Ca/Zn).

**Grain Zn concentration obs summary (26 obs from Tables 2 and 3):**

From Table 2 (main effects, 2 obs):
- ZnSO4 only: ctrl=40.9, treat=52.0, effect=+27.1%
- ZnSO4 + Sucrose: ctrl=40.9, treat=56.5, effect=+38.1%

From Table 3 (3 N-rates × 4 cultivars × 2 treatments = 24 obs, selected examples):
- Jinan 17 / 75 kg N / ZnSO4: ctrl=39.3, treat=49.2, effect=+25.2%
- Jinan 17 / 75 kg N / ZnSO4+Sucrose: ctrl=39.3, treat=56.7, effect=+44.3%
- Luyuan 502 / 275 kg N / ZnSO4+Sucrose: ctrl=38.6, treat=58.7, effect=+52.1%
- (... 20 additional rows covering all cultivar × N-rate × treatment combinations)

**AI extraction quality assessment:** The AI correctly read Tables 2 and 3 from the Xia et al. 2018 paper. Values align with what is visible in the PDF scan. LSD variance extracted for Table 2 (LSD=1.6 for Zn conc), but not for Table 3 individual cells. The AI correctly identified n=4 replicates. The extraction is internally consistent and represents the Xia 2018 paper faithfully.

**Critical finding:** No extracted observation has ctrl=18.15 mg kg−1. All extracted control values are 34.9–47.7 mg kg−1. This is the fundamental reason for zero match.

---

## 4. GT Data from MOESM5 (Foliar sheet, study_ids 8 and 31)

### Study ID 31 (Dong M. 2017 Master's thesis)

Three observations present in Data 3 Foliar application sheet (Obs IDs 341–343):

| Obs ID | ctrl Zn (mg kg−1) | treat Zn (mg kg−1) | Effect (%) | n | Zn rate (kg ha−1) | Spray conc. (g Zn L−1) | Spray freq. |
|---|---|---|---|---|---|---|---|
| 341 | 18.15 | 27.43 | +51.1% | 3 | 2.0 | 0.081 | 2 |
| 342 | 18.15 | 37.40 | +106.1% | 3 | 2.0 | 0.081 | 2 |
| 343 | 18.15 | 27.67 | +52.5% | 3 | 2.0 | 0.081 | 2 |

Additional moderators: Country=China, Available Zn=3.46 mg kg−1, N rate=240 kg N ha−1, P=52.8, K=124.5 kg ha−1, Grain yield=6022 kg ha−1.

The three observations share the same control (18.15 mg kg−1) and the same Zn rate. The three treatment values (27.43, 37.4, 27.67) likely represent different spraying timings (column 29 = 5) or different cultivars within the thesis experiment. The treatment value range spans +51% to +106%, which is substantially higher than the +27% to +52% range in Xia 2018.

### Study ID 8 (in PAPER_TO_SHEET_IDS mapping for Dong_2018)

Study ID 8 does **not** exist in the Data 3 Foliar application sheet. It appears only in Data 2 Soil application, where it corresponds to Lu et al. 2010 (20 soil-application observations). This is a mapping error in the validation script: `"Dong_2018": {"Data 3 Foliar application": [8, 31]}` — study_id=8 in the foliar sheet returns zero rows, contributing 0 to the 5 GT rows count.

**Resolution of n_gt=5 discrepancy:** The validation report shows n_gt=5 for Dong_2018. Inspection shows only 3 rows exist (obs 341–343, all study_id=31). The count of 5 likely arises from how the gt_txt file was pre-processed or from a now-obsolete version of the GT loading code. The actual GT size is 3 observations.

---

## 5. Root Cause Analysis

### Primary cause: Wrong paper in PDF slot

The file `Dong_2018.pdf` contains **Xia et al. 2018** (Frontiers in Plant Science), not the Dong M. 2017 Master's thesis. The MOESM5 GT (study_id=31) references the Dong 2017 thesis. These are two entirely different papers:

| Dimension | Dong 2017 thesis (GT) | Xia et al. 2018 (PDF extracted) |
|---|---|---|
| Authors | Dong M. (single author) | Xia H., Xue Y., Liu D., et al. |
| Venue | Master's thesis, Nanjing Agric. Univ. | Frontiers in Plant Science 9:677 |
| Control grain Zn | 18.15 mg kg−1 | 34.9–47.7 mg kg−1 |
| Design | Simpler (1 N rate) | 3 N rates × 4 cultivars |
| n | 3 | 4 |
| Foliar Zn effects | +51% to +106% | +19% to +52% |

The overlap in author name ("Dong") and year proximity (thesis 2017, paper 2018) likely caused mislabeling of the file in the input PDF set.

### Secondary cause: Study ID 8 mapping error

The validation script maps `"Dong_2018"` to study_ids `[8, 31]` in the Foliar sheet. Study_id=8 in the Foliar sheet does not exist (it belongs to Data 2 Soil sheet, corresponding to Lu et al. 2010). This stale mapping inflates the n_gt count and adds unnecessary confusion.

### Why 0 matches despite the AI extracting 32 observations

The matching algorithm compares ctrl and treat means with 15% combined tolerance. The GT control is 18.15 mg kg−1; the closest AI-extracted control is 34.9 mg kg−1 (92% higher). No extracted value is within any reasonable tolerance of the GT values:

- GT ctrl range: 18.15 mg kg−1 (all 3 obs identical)
- AI extracted ctrl range: 34.9–47.7 mg kg−1
- Minimum ctrl discrepancy: (34.9 - 18.15) / 18.15 = 92% — far beyond the 15% tolerance

The matching algorithm correctly finds zero matches: the extracted data and GT data are from different experiments.

---

## 6. Assessment

**Extraction quality (for Xia et al. 2018 content):** GOOD

The AI extracted the correct paper content accurately. The 26 grain Zn concentration observations from Tables 2 and 3 of Xia et al. 2018 appear to reflect the PDF values faithfully. The factorial structure (N rate × cultivar × foliar treatment) was correctly decomposed. LSD variance was correctly identified from the paper's statistical analysis section. The extraction is a reasonable representation of the Xia 2018 data.

**Validation outcome:** ZERO MATCH — caused entirely by PDF mislabeling, not extraction error

**What needs to be fixed:**

1. **PDF mislabeling (critical):** `Dong_2018.pdf` contains Xia et al. 2018, not Dong 2017. The correct PDF for the Dong 2017 Master's thesis needs to be located (if publicly available) or the paper should be excluded from the validation set. The Xia 2018 paper may itself be a valid entry in the meta-analysis, but it needs to be assigned the correct study_id in MOESM5 (if it appears there under a different entry).

2. **Study ID mapping error (minor):** Remove `8` from the `"Dong_2018"` mapping in `validate_hui2023_full.py`. Study_id=8 in the Foliar sheet has no rows; it belongs to Lu et al. 2010 in the Soil sheet. Corrected mapping should be: `"Dong_2018": {"Data 3 Foliar application": [31]}`.

3. **Note on GT effects magnitude:** The Dong 2017 thesis reports control grain Zn of 18.15 mg kg−1, which is well below the HarvestPlus biofortification target of 38 mg kg−1. The three treatment values (27.43, 37.4, 27.67) also largely fail to reach this target. By contrast, Xia et al. 2018 reports baseline controls of 35–48 mg kg−1 (already above or near target) with treatments reaching 49–61 mg kg−1. The two papers represent quite different agronomic contexts, reinforcing that this is indeed a different paper in the PDF slot.

**Classification:** Zero-match due to PDF mislabeling. The AI extraction faithfully captured the content of the wrong PDF. No re-extraction needed; the root fix is at the source PDF level.
