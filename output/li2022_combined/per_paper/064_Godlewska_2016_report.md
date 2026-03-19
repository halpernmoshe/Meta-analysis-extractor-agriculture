# Extraction Quality Report: 064_Godlewska_2016

**Paper:** Godlewska A & Ciepiela GA (2016). "The effect of growth regulator on dry matter yield and some chemical components in selected grass species and cultivars." *Soil Science and Plant Nutrition*, 62(3), 297–302. DOI: 10.1080/00380768.2016.1185741

**Report generated:** 2026-02-18
**Match result:** 3 matched pairs, r = -0.673, MAE = 3.6 pp, 0 unmatched GT rows, 35 unmatched JSON obs

---

## 1. Paper Design

### Experimental Setup
- **Location:** Siedlce Experimental Unit, University of Natural Sciences and Humanities, Poland
- **Duration:** 2009–2012 (establishment year 2009; data collection years 2010–2012)
- **Design:** Split-split-plot (randomized sub-block), **3 replicates**, plot area 10 m²
- **Treatment factor:** Kelpak SL seaweed extract (biostimulant) at 2 dm³ ha⁻¹ vs. untreated control (0 dm³ ha⁻¹)
- **Plant factor:** 4 cultivars of 2 grass species grown as **pure stands (monocultures)**:
  - *Dactylis glomerata* L. cv. Amila
  - *Dactylis glomerata* L. cv. Tukan
  - *Festulolium braunii* cv. Felopa
  - *Festulolium braunii* cv. Agula
- **Harvest structure:** 3 cuts per year × 3 years (2010, 2011, 2012) = 9 harvest events per plot

### Outcome Variables Reported
The paper contains five tables:
| Table | Content | Units |
|-------|---------|-------|
| Table 1 | Meteorological data | Temperature (°C), precipitation (mm) |
| Table 2 | Dry matter yield (DMY) | t ha⁻¹ |
| Table 3 | Total nitrogen content | g kg⁻¹ DM |
| Table 4 | NDF (Neutral Detergent Fiber) | g kg⁻¹ DM |
| Table 5 | ADF (Acid Detergent Fiber) | g kg⁻¹ DM |

---

## 2. Table 2 Structure: The Critical Table for DMY

Table 2 is the primary source of discrepancy. It reports dry matter yield (t ha⁻¹) in a two-panel format with both cut-level and year-level aggregations:

### Left Panel: "Cut (mean from years 2010–2012)"
Values are **means across the 3 study years** for each of the 3 cuts:

| Grass species | Cultivar | Kelpak dose | Cut 1 | Cut 2 | Cut 3 | Mean (of 3 cuts) |
|---|---|---|---|---|---|---|
| *D. glomerata* | Amila | 0 | **5.0** | **4.1** | **2.8** | 4.0 |
| | | 2 | 5.2 | 4.2 | 3.0 | 4.1 |
| | Tukan | 0 | **5.3** | **4.0** | **3.0** | 4.1 |
| | | 2 | 5.7 | 4.3 | 3.0 | 4.3 |
| *F. braunii* | Felopa | 0 | **4.3** | **3.2** | **1.9** | 3.1 |
| | | 2 | 4.7 | 3.6 | 2.1 | 3.5 |
| | Agula | 0 | **4.7** | **2.9** | **1.7** | 3.1 |
| | | 2 | 5.1 | 3.4 | 1.8 | 3.4 |
| **Mean (both doses)** | | | **5.0 A** | **3.7 B** | **2.4 C** | 3.7 |
| **Control mean only** | | | **4.8** | **3.6** | **2.4** | 3.6 |

### Right Panel: "Year (mean from three cuts)"
Values are **sums of 3 cuts per year** (i.e., annual totals):

| Grass species | Cultivar | Dose | 2010 | 2011 | 2012 | Mean |
|---|---|---|---|---|---|---|
| *D. glomerata* | Amila | 0 | 14.5 | 12.5 | 8.7 | 11.9 |
| | | 2 | 14.8 | 13.3 | 9.1 | 12.4 |
| | Tukan | 0 | 13.3 | 14.1 | 9.5 | 12.3 |
| | | 2 | 14.6 | 14.5 | 10.0 | 13.0 |
| *F. braunii* | Felopa | 0 | 13.4 | 7.6 | 7.2 | 9.4 |
| | | 2 | 14.0 | 9.2 | 8.1 | 10.4 |
| | Agula | 0 | 12.1 | 8.1 | 7.7 | 9.3 |
| | | 2 | 13.4 | 9.6 | 8.2 | 10.4 |

**The right panel confirms these are annual yield totals** (sum of 3 cuts). The text confirms: "The highest yields were recorded in 2010 (on average 13.8 t ha⁻¹) and the lowest in 2012 (on average 8.6 t ha⁻¹)."

---

## 3. What Our AI Extracted (JSON Observations)

The AI correctly read Table 2 and extracted **two sets of observations**:

### Set A: Per-cultivar, per-cut (left panel of Table 2)
- 12 observations: 4 cultivars × 3 cuts
- **Amila cut 1 ctrl = 5.0, cut 2 ctrl = 4.1, cut 3 ctrl = 2.8** t ha⁻¹
- These are per-cut means averaged across 2010–2012
- Correctly labeled as "mean of 2010–2012"

### Set B: Per-cultivar, per-year (right panel of Table 2)
- 12 observations: 4 cultivars × 3 years
- **Amila 2010 ctrl = 14.5, 2011 ctrl = 12.5, 2012 ctrl = 8.7** t ha⁻¹
- These are annual totals (sum of 3 cuts per year)
- The AI correctly flagged these as "sum of three cuts"

### Summary of AI extraction
The AI extracted **the correct values from Table 2** for all cultivars × cut combinations and all cultivars × year combinations. The extraction is factually accurate relative to the PDF.

The AI also extracted total nitrogen (Table 3), NDF (Table 4), and ADF (Table 5) data for Amila across cuts and years (24 additional observations), all correctly labeled.

**Total JSON obs: 38** (12 per-cut + 12 per-year + 4 N-content + 4 NDF + 4 ADF + 2 duplicates)

---

## 4. What the Li 2022 Ground Truth Contains (3 GT Rows)

The Li 2022 meta-analysis database contains only **3 rows** for this paper (GT pairs 514–516):

| GT pair | Cut | ctrl_mean (t ha⁻¹) | treat_mean (t ha⁻¹) | Effect (%) |
|---|---|---|---|---|
| 514 | Cut 1 | **1.33** | 1.42 | +6.77% |
| 515 | Cut 2 | **1.06** | 1.17 | +10.38% |
| 516 | Cut 3 | **0.83** | 0.89 | +7.23% |

**Key GT metadata:**
- crop = "grass mixture" (not "pure stands")
- dose = 1 (dm³ ha⁻¹) — but the paper applies 2 dm³ ha⁻¹
- Frequency = 3
- replicates = 3
- No cultivar specified

The GT describes the crop as **"grass mixture"**, whereas this paper grows pure monocultures of individual cultivars.

---

## 5. Root Cause Analysis: Why GT ctrl Values (1.33, 1.06, 0.83) Do Not Match JSON Values (5.0, 4.1, 2.8)

### 5.1 Scale Comparison

| Cut | GT ctrl | AI ctrl (Amila) | AI ctrl (grand mean) | Ratio |
|-----|---------|-----------------|----------------------|-------|
| 1 | 1.33 | 5.0 | 4.8 | 3.6–3.8× |
| 2 | 1.06 | 4.1 | 3.6 | 3.4–3.9× |
| 3 | 0.83 | 2.8 | 2.4 | 2.9–3.4× |

The GT values are approximately **3.5–4× lower** than any single-cultivar or grand-mean value in Table 2 of this paper. This ratio is not consistent with a simple unit conversion error:

- 1 t ha⁻¹ = 1 Mg ha⁻¹ (same unit, different notation) → ratio would be 1.0
- 1 t ha⁻¹ = 10 dt ha⁻¹ → would produce ratios of 0.1 (opposite direction)
- Plot-level conversion (10 m² plot): 5.0 t/ha × 10/10000 = 0.005 t/plot → far smaller, not 1.33

No arithmetic transformation of the values in Table 2 of this PDF produces the GT values.

### 5.2 Effect Direction Analysis

Critically, the GT values and the AI-extracted values **disagree in the ordering of effect sizes across cuts**:

| Cut | GT effect (%) | AI Amila effect (%) | AI grand mean effect (%) |
|-----|--------------|---------------------|--------------------------|
| 1 | +6.77 | +4.0 | +8.3 |
| 2 | **+10.38** | +2.4 | **+8.3** |
| 3 | +7.23 | +7.1 | +4.2 |

The GT shows Cut 2 as the largest positive effect (+10.38%), while the AI extracts Cut 1 or Cut 3 as largest depending on cultivar. This ordering mismatch directly produces the **negative correlation r = -0.673**.

### 5.3 The "Grass Mixture" Discrepancy

The most telling clue is the GT crop descriptor: **"grass mixture"**. This paper explicitly grows pure stands in monoculture. The authors note: "pure sown grass species and cultivars grown in monoculture." The Li 2022 meta-analysis expected data from a mixed-species grassland sward, which would naturally produce lower per-species yields.

In a grass mixture, total sward yield might be ~5 t ha⁻¹ per cut, but any one component species within that mixture would yield only a fraction (~1–2 t ha⁻¹). The GT values (1.33, 1.06, 0.83) are consistent with per-species contribution within a mixed sward rather than total monoculture yield.

### 5.4 Most Probable Explanation: Li 2022 Cited a Different Companion Paper

The authors Godlewska and Ciepiela published multiple papers on Kelpak applications to grasses. Their 2013 paper (Ciepiela GA, Godlewska A, Jankowska J 2013, *Fresen. Environ. Bull.*, 22(12b)) studied **grass/red clover mixed stands** and reported seaweed extract increased yields by 10.6% — which more closely matches the GT effect patterns.

The PDF filed as `064_Godlewska_2016` (the Soil Science and Plant Nutrition paper on pure stands) is likely **not the paper Li 2022 actually extracted data from**. Li 2022 probably extracted from a companion paper by the same research group that:
1. Used a **grass mixture** (not pure stands)
2. Had lower per-cut yields consistent with mixed-sward measurements (~1.0–1.4 t ha⁻¹ per component per cut)
3. Reported Kelpak effects on a mixed cultivar basis (not split by individual cultivar)

### 5.5 Alternative Explanation: Cross-Cultivar Averaging with Year Subdivision

A secondary hypothesis is that Li 2022 extracted from Table 2 but applied a different aggregation:

**Could Li 2022 have divided annual totals by number of cultivars (4) AND by number of cuts (3)?**
- Amila+Tukan+Felopa+Agula ctrl annual mean ÷ 4 cultivars ÷ 3 cuts:
  - 2010: (14.5+13.3+13.4+12.1) / 4 / 3 = 53.3/12 = **4.44** → not 1.33

**Could Li 2022 have used per-year per-cut divided by number of species (2)?**
- Overall mean ctrl cut 1 (4.8) / 2 species = 2.4 → not 1.33

**Could the values be from only the 2012 data, divided by 4 cultivars?**
- 2012: Amila 8.7 + Tukan 9.5 + Felopa 7.2 + Agula 7.7 = 33.1 / 4 cultivars / 3 cuts = **2.76** → not 1.33

No plausible averaging scheme applied to the numbers in Table 2 of the PDF produces 1.33, 1.06, 0.83. This strongly supports the interpretation that the GT and this PDF derive from **different underlying papers**.

### 5.6 The Negative Correlation Explained

The r = -0.673 arises directly from the mismatch between which cut has the largest effect:

- In the GT (presumably from a different, grass-mixture paper): Cut 2 shows the largest effect (+10.38%) > Cut 3 (+7.23%) > Cut 1 (+6.77%)
- In this PDF for Amila: Cut 3 ≈ Cut 1 > Cut 2 in effect magnitude; for other cultivars the pattern varies
- When the matching algorithm pairs GT Cut 1 → JSON Cut 1 (Amila), GT Cut 2 → JSON Cut 2 (Amila), GT Cut 3 → JSON Cut 3 (Amila), the effect size ordering is inverted: GT says Cut 2 is the biggest responder, JSON says Cut 2 has the smallest effect (+2.4% for Amila, +7.5% for Tukan)

This inversion produces a strong negative correlation across the 3 matched pairs.

---

## 6. Verification Against Table 2 Bottom Row

The paper text explicitly states: "The greatest plant dry matter yields were at the first cut (on average **5.0 t ha⁻¹**) and the lowest in the third cut (on average **2.4 t ha⁻¹**)."

These are the grand means (all cultivars, both doses): Cut 1 = 5.0, Cut 2 = 3.7, Cut 3 = 2.4 t ha⁻¹.

The control-only grand means are: Cut 1 = 4.8, Cut 2 = 3.6, Cut 3 = 2.4 t ha⁻¹.

**Conclusion:** There is no table, sub-table, footnote, or textual value in this PDF that contains control means of approximately 1.33, 1.06, or 0.83 t ha⁻¹. These values simply do not exist in this paper.

---

## 7. Assessment of AI Extraction Quality

| Dimension | Assessment |
|-----------|-----------|
| **Values read from PDF** | Correct — AI faithfully read Table 2 per-cultivar, per-cut values |
| **Units identified** | Correct — t ha⁻¹ throughout |
| **Treatment/control identification** | Correct — dose 0 = control, dose 2 = Kelpak treatment |
| **Aggregation level** | Correct — properly separated per-cut (3-year mean) vs per-year (3-cut sum) |
| **Cultivar-level extraction** | Appropriate — extracted all 4 cultivars separately |
| **Match to GT** | Impossible — GT values do not exist in this PDF |
| **Effect direction** | AI effects are internally consistent with the paper; GT effects derive from a different dataset |

**The AI extraction is of high quality and accurately reflects the content of the PDF.** The matching failure is not due to an extraction error but due to a **source paper mismatch** between the paper filed under ID 064 and the paper Li 2022 actually coded as pairs 514–516.

---

## 8. Unmatched JSON Observations (35 of 38)

The 35 unmatched observations fall into four categories:

| Category | Count | Reason |
|----------|-------|--------|
| Per-cultivar per-cut yield (non-Amila) | 9 | GT has only 3 rows (cut-level averages, not per-cultivar) |
| Per-cultivar per-year yield (annual totals) | 12 | Different aggregation: GT is per-cut, not per-year |
| Total nitrogen content (Table 3) | 4 | Forage quality outcome not in GT scope for this paper |
| NDF content (Table 4) | 4 | Forage quality outcome not in GT scope |
| ADF content (Table 5) | 4 | Forage quality outcome not in GT scope |
| Duplicates | 2 | Re-extracted entries already covered |

All 35 unmatched observations are **correctly extracted from the paper** — they are unmatched because Li 2022 chose a more aggregated representation (one data point per cut across all cultivars combined) and only coded yield, not forage quality variables.

---

## 9. Recommendations

### For the Validation Analysis
1. **Flag pairs 514–516 as "source mismatch"**: The GT values cannot be reconciled with any values in this PDF. The matching with Amila cut 1–3 values is a forced match with low confidence and produces a misleading negative correlation.

2. **Exclude from correlation calculation**: Since the source paper for GT pairs 514–516 is almost certainly not this PDF, including these 3 pairs in the per-paper r calculation is inappropriate. The paper should be coded as "unverifiable" rather than r = -0.673.

3. **Investigate Li 2022 reference list**: Confirm which Godlewska/Ciepiela paper Li 2022 actually cited for pairs 514–516. The most likely candidate is a grass-mixture companion paper with lower per-component yields (~1.0–1.4 t ha⁻¹ per cut), possibly:
   - Ciepiela et al. 2013 (grass/clover mixtures, *Fresen. Environ. Bull.*)
   - Another Godlewska & Ciepiela paper on mixed grassland swards

### For the AI Extraction Pipeline
- No changes needed: the AI extraction is accurate
- The matching algorithm correctly assigned "low confidence" to cuts 1–2 and "medium" to cut 3, reflecting the true uncertainty
- Consider adding a "source mismatch" flag when GT crop descriptor ("grass mixture") conflicts with paper description (pure monocultures) and absolute values differ by >3×

---

## 10. Summary

| Issue | Finding |
|-------|---------|
| GT ctrl values (1.33, 1.06, 0.83 t ha⁻¹) | **Not present anywhere in this PDF** |
| AI ctrl values (5.0, 4.1, 2.8 t ha⁻¹) | **Correct** — directly from Table 2, Amila cultivar, cut means |
| Scale difference (~3.7×) | Not explainable by unit conversion, averaging, or year selection |
| GT crop = "grass mixture" | **Contradicts paper** — this paper grows pure monocultures |
| Root cause of r = -0.673 | Forced matching between two different papers; effect size ordering is inverted when GT cut 2 > GT cut 3 > GT cut 1, but JSON Amila cut 2 < cut 1 ≈ cut 3 |
| Most likely explanation | **Li 2022 coded a different Godlewska/Ciepiela paper** (grass mixture study) but the same reference ID was assigned to this pure-stand monoculture paper in the PDF dataset |
| AI extraction quality | **High** — accurate, complete, appropriately structured |
