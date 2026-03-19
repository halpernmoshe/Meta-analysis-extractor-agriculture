# Extraction Quality Report: 131_Rahman_2018

**Paper:** Rahman M, Mukta JA, Sabir AA, et al. "Chitosan biopolymer promotes yield and stimulates accumulation of antioxidants in strawberry fruit." *PLoS ONE* 13(9): e0203769. https://doi.org/10.1371/journal.pone.0203769

**Report generated for:** Li (2022) meta-analysis validation

---

## 1. Paper Design Summary

### Study overview

This is a single-site, single-season, randomized complete block design experiment conducted at Bangabandhu Sheikh Mujibur Rahman Agricultural University (BSMRAU), Gazipur, Bangladesh (24.0379 N, 90.3996 E) from November 2014 to March 2015. The crop is strawberry cv. Festival (*Fragaria* × *annanasa*). The intervention is foliar spray application of a chitosan biopolymer (poly beta-1,4-D-glucosamine, Sigma-Aldrich, CAS 9012-76-4) at four concentrations compared to an untreated water control.

### Experimental design

| Parameter | Value |
|-----------|-------|
| Crop | Strawberry cv. Festival |
| Design | Randomized complete block design (RCBD) |
| Replicates | 3 |
| Plants per plot | 8 |
| Application method | Foliar spray (run-off), 5 times at 10-day intervals |
| Application period | December 20, 2014 to February 22, 2015 |
| Harvest period | January 23 to March 17, 2015 |
| Country | Bangladesh |

### Chitosan dose arms (GT pairs 422-425)

| GT pair | Dose (%) | Dose (ppm) | Treat mean (g/plant) | GT effect |
|---------|----------|------------|---------------------|-----------|
| 422 | 0.125% | 125 ppm | 339.1 | +28.1% |
| 423 | 0.25% | 250 ppm | 349.5 | +32.0% |
| 424 | 0.5% | 500 ppm | 378.4 | +43.0% |
| 425 | 1.0% | 1000 ppm | 361.9 | +36.7% |

*Control mean: 264.7 g/plant across all four GT pairs. Note: GT values are in units of 100 g/plant (i.e., GT ctrl_mean = 2.647 = 264.7 g/plant).*

This is flagged as a dose-response study (`ifdoseresponse = 1`) in the Li 2022 ground truth dataset.

---

## 2. Data Presentation in the PDF

### Tables

The paper contains two numeric tables:

**Table 1** (page 5) — Vegetative canopy parameters:
- Columns: Leaf length (cm), Leaf width (cm), Leaf number/plant, Canopy diameter (cm)
- Rows: Control, Ch 125, Ch 250, Ch 500, Ch 1000
- All four dose arms present with mean ± SE values (n = 3)
- AI extraction: Fully and correctly extracted (json_idx 0-15)

**Table 2** (page 7) — Shoot and root biomass:
- Columns: Shoot fresh weight (g), Shoot dry weight (g), Root fresh weight (g), Root dry weight (g)
- Rows: Control, Ch 125, Ch 250, Ch 500, Ch 1000
- All four dose arms present with mean ± SE values (n = 24)
- AI extraction: Fully and correctly extracted (json_idx 16-31)

### Figures

**Figure 1** (page 6) — Bar charts for plant height, root length, individual fruit weight, total fruit weight per plant, and percent increase in fruit yield:
- This figure contains the primary yield outcome (total fruit weight per plant in g) for **all four dose arms** plus control
- Data are presented as bar charts with vertical error bars (SE, n = 24)
- Values readable from bars: Control ~246.6 g, Ch 125 ~318 g, Ch 250 ~330 g, Ch 500 ~351 g, Ch 1000 ~330 g
- The percent increase panel explicitly labels: Ch 125 = 29%, Ch 250 = 32%, Ch 500 = 42%, Ch 1000 = 40%
- **There is no separate numeric table for total fruit yield.** Figure 1 is the sole source.

**Figure 3** (page 9) — Bar charts for antioxidant biochemical content (anthocyanins, carotenoids, flavonoids, phenolics, antioxidant activity) for all four doses. AI did not extract these (outside Li 2022 GT scope).

### Statistical methods

The paper states: "Values are means ± standard errors of three independent replications (n = 3)" for Table 1, and "n = 24" for Table 2 and Figure 1. Treatment means were separated using Fisher's protected LSD test (p ≤ 0.05). Variance type is SE throughout; this is explicitly declared in the table footnotes and figure captions.

---

## 3. What the AI Extracted

The AI produced 38 observations across the following outcome categories:

| Outcome | Doses covered | Source |
|---------|---------------|--------|
| Leaf length (cm) | 125, 250, 500, 1000 ppm | Table 1 |
| Leaf width (cm) | 125, 250, 500, 1000 ppm | Table 1 |
| Leaf number/plant | 125, 250, 500, 1000 ppm | Table 1 (+ duplicate) |
| Canopy diameter (cm) | 125, 250, 500, 1000 ppm | Table 1 |
| Shoot fresh weight (g) | 125, 250, 500, 1000 ppm | Table 2 |
| Shoot dry weight (g) | 125, 250, 500, 1000 ppm | Table 2 |
| Root fresh weight (g) | 125, 250, 500, 1000 ppm | Table 2 |
| Root dry weight (g) | 125, 250, 500, 1000 ppm | Table 2 |
| Individual fruit weight (g) | 125 ppm only | Figure 1 |
| Total fruit weight/plant (g/plant) | 125 ppm and 250 ppm only | Figure 1 |
| Percent increase in fruit yield (%) | 125 ppm only | Figure 1 (text) |

### Notable extraction issues

1. **Coverage failure for total fruit yield at 500 ppm and 1000 ppm.** The AI extracted total fruit weight/plant for only two of the four dose arms (125 ppm and 250 ppm). The 500 ppm and 1000 ppm arms are absent from the extracted data (json_obs has no entry for "total fruit weight per plant" at 500 or 1000 ppm).

2. **Inconsistent treatment of Figure 1.** The AI partially read Figure 1 — it captured some figure values for the 125 ppm and 250 ppm arms but did not systematically extract all four arms. This is a bar-chart reading problem rather than a missing-data problem.

3. **Duplicate observations.** json_idx 8 and json_idx 34 both represent "leaf number/plant" at 125 ppm with the same values but slightly different metadata (moderator site field: "Bangladesh" vs. "Gazipur, Bangladesh"). These are duplicate entries from the same table cell.

4. **Mixed tissue labels for yield.** json_idx 33 labels tissue as "grain" and json_idx 35 labels tissue as "fruit" for the same outcome (total fruit weight per plant). This inconsistency is not a data error but indicates the AI processed Figure 1 twice in different passes, producing partially redundant entries with slightly different values (246.6 → 300.0 g at 250 ppm vs 246.6 → 318.0 g at 125 ppm).

5. **Effect size discrepancy at 250 ppm.** GT pair 423 gives a treatment effect of +32.0% (treat_mean = 349.5 g/plant), but the extracted value for total fruit weight at 250 ppm is 300.0 g/plant (+21.7%). The GT value of ~330-350 g corresponds more closely to Figure 1's visual bar heights (which the percent-increase panel confirms at +32%), while the AI read the 250 ppm bar as 300.0 g. This is a figure digitization error — the AI's bar reading was ~10% below the ground truth.

6. **No variance values extracted for any yield observation.** None of the json_obs for total fruit weight include SE or SD values, despite the paper explicitly providing SE bars in Figure 1. This is a known limitation: bar chart error bar extraction requires figure digitization that the system does not perform.

---

## 4. Why the AI Missed the 0.5% and 1.0% Dose Arms

### Root cause: bar chart extraction is incomplete and inconsistent

The total fruit yield data in this paper exists **only in Figure 1**, presented as bar charts. There is no tabular summary of total fruit yield values anywhere in the paper. This is a structural feature of the PDF that creates extraction difficulty:

- The AI correctly extracted all data from Tables 1 and 2, which are machine-readable text tables.
- For Figure 1, the AI attempted to read bar heights visually but did so incompletely. It captured values for the first two treatment arms (125 ppm and 250 ppm) but stopped before reading the 500 ppm and 1000 ppm bars.
- This is likely because the AI prioritized text-table extraction and treated figure data as supplementary, processing it less systematically.

### The data was accessible but required figure digitization

The 500 ppm and 1000 ppm fruit yield data is **genuinely present and accessible** in the PDF. From Figure 1:

- Total fruit weight/plant at 500 ppm: visually ~351 g/plant (consistent with GT: 378.4 g, +43% effect)
- Total fruit weight/plant at 1000 ppm: visually ~330 g/plant (consistent with GT: 361.9 g, +37% effect)
- The percent-increase panel labels these as 42% and 40% respectively

The ground truth values in the Li (2022) dataset are likely derived from the same Figure 1 using digital image analysis (e.g., WebPlotDigitizer) or from direct data shared by the authors. The data was not hidden or ambiguous; it was simply in a figure rather than a table.

### Compounding factor: the AI focused extraction on tabular outcomes

The AI extracted 32 high-quality, variance-complete observations from Tables 1 and 2 (vegetative morphology and biomass), which are fully outside the scope of what Li (2022) coded as the primary outcome. Having found abundant tabular data, the AI may have deprioritized the more effort-intensive figure-reading task and extracted only the first few bars of Figure 1 before stopping.

---

## 5. Match Summary

| GT pair | Dose | Status | Extracted element | Effect GT | Effect extracted | Notes |
|---------|------|--------|-------------------|-----------|-----------------|-------|
| 422 | 0.125% (125 ppm) | MATCHED (high conf.) | total fruit weight/plant | +28.1% | +28.9% | Excellent agreement |
| 423 | 0.25% (250 ppm) | MATCHED (medium conf.) | total fruit weight/plant | +32.0% | +21.7% | Figure digitization error (~10% underread) |
| 424 | 0.5% (500 ppm) | UNMATCHED | — | +43.0% | not extracted | Data in Figure 1, not extracted |
| 425 | 1.0% (1000 ppm) | UNMATCHED | — | +36.7% | not extracted | Data in Figure 1, not extracted |

- **Matched pairs:** 2 / 4 (50% of GT dose arms)
- **Unmatched GT:** 2 (500 ppm and 1000 ppm arms entirely absent from extracted yield data)
- **Unmatched JSON:** 36 (all vegetative and biomass outcomes outside Li 2022 scope)

---

## 6. Assessment of Extraction Quality

### Overall verdict: PARTIAL — Structural Figure Extraction Failure

| Dimension | Assessment |
|-----------|------------|
| Tabular data extraction | EXCELLENT — Tables 1 and 2 fully extracted with correct values |
| Figure data extraction | POOR — Only 2 of 4 dose arms captured from Figure 1 |
| Dose-response coverage | 50% (2/4 arms matched) |
| Effect size accuracy (matched arms) | GOOD for 125 ppm (+0.8 pp error); MODERATE for 250 ppm (-10.3 pp error) |
| Variance extraction | ABSENT for all yield observations (figure error bars not digitized) |
| Metadata accuracy | Good — correct crop, method, country, n=3, cultivar |

### Strengths

1. The AI correctly identified the study as a dose-response design and extracted all four doses for vegetative and biomass outcomes from Tables 1 and 2.
2. The 125 ppm yield match (+28.1% GT vs +28.9% extracted) is excellent and confirms the AI correctly identified the relevant yield metric from Figure 1.
3. Moderator metadata is accurate: crop = strawberry, method = foliar, country = Bangladesh, cultivar = Festival, n = 3.
4. The variance type (SE) declared in the paper was identified by the AI even if numeric SE values were not extracted for figure-based observations.

### Weaknesses

1. **Systematic under-coverage of figure-based data.** The primary outcome (total fruit yield) is presented only in Figure 1. The AI extracted this outcome for only 2 of 4 doses. This is a 50% miss rate for the primary meta-analysis outcome.
2. **Effect size underestimation at 250 ppm.** The extracted value (300.0 g vs GT 349.5 g) suggests the bar height for the 250 ppm arm was underread by approximately 14%. Accurate bar chart digitization would require either dedicated figure-reading tools or manual verification.
3. **No SE/SD for yield observations.** All four GT yield pairs have associated SD values (derived from SE × sqrt(n)), but the AI extracted no variance values for any figure-based observation. This means all four GT observations lack extractable variance, reducing their usability for inverse-variance weighted meta-analysis without imputation.
4. **Duplicate entries** (leaf number/plant at 125 ppm appears twice with slightly different moderator metadata), indicating the AI processed some sections more than once.

### Implication for meta-analysis use

For the Li (2022) validation dataset, this paper contributes 4 dose-response pairs (pairs 422-425) for total fruit yield in strawberry. The AI's extraction recovers 2 of these pairs with partial accuracy. The missing 500 ppm and 1000 ppm arms represent the two highest-effect observations in this paper (43% and 37% yield increase), which are biologically and statistically important for characterizing the dose-response curve. Their absence would bias any dose-response modelling for chitosan on strawberry yield toward underestimating the effect at the optimal dose.

### Recommended remediation

1. **Re-run extraction with explicit figure digitization instruction** directing the AI to read all bars in Figure 1 for each of the five treatment groups (Control, Ch 125, Ch 250, Ch 500, Ch 1000) and report the numeric bar heights.
2. **Alternatively**, manually digitize Figure 1 total fruit weight/plant bars using WebPlotDigitizer or equivalent, which would recover all four dose arms with high accuracy.
3. **SE recovery**: Figure 1 caption states n = 24 and error bars = SE. The error bar heights for total fruit weight are visually readable and could be manually extracted to provide variance values for all four dose arms.
