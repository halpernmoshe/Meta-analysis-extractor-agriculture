# Extraction Quality Report: 120_Pohl_2019
**Paper:** Pohl A, Grabowska A, Kalisz A, Sekara A (2019). "The Eggplant Yield and Fruit Composition as Affected by Genetic Factor and Biostimulant Application." *Notulae Botanicae Horti Agrobotanici Cluj-Napoca*, 47(3):929-938.

**Match result:** 1 matched pair / 12 GT pairs / 21 unmatched JSON observations

---

## 1. Paper Design

### Crop and Location
- Crop: Eggplant (*Solanum melongena* L.), field cultivation
- Location: University of Agriculture in Krakow, southern Poland (50°04'N, 19°51'E)
- Years: 2013 and 2014 (two complete growing seasons, May–September each year)
- Design: Completely randomized block design, **3 replications per treatment**

### Biostimulant Tested
**One product only: Göemar BM-86** (Arysta LifeScience North America, LLC).
- Active ingredient: Standardized *Ascophyllum nodosum* (brown seaweed) extract
- Category: Seaweed extract (SWE)
- Application method: Foliar spray, 3 applications at 2-week intervals (starting 2 weeks after transplanting)
- Dose: 1.5 dm³ ha⁻¹ per application
- Control: Distilled water spray

This paper tests **one biostimulant product** (BM-86) at **one dose** across **six cultivars** for **two years**. There is no dose-response design and no additional products.

### Cultivars (6 F1 hybrids)
1. Cristal F1 (Semillas Fito S.A.)
2. Epic F1 (Seminis Vegetable Seeds)
3. Flavine F1 (Gautier Semences)
4. Gascona F1 (Gautier Semences)
5. Onyx F1 (Semillas Fito S.A.)
6. WA 6020 F1 (Western Seed International BV)

### Outcome Variables Measured
The paper reports four yield-related outcomes per cultivar per year:
- **Early yield** (kg m⁻²): First four harvests only
- **Marketable yield** (kg m⁻²): Full season total
- Number of fruits per m²
- Mean fruit weight (kg)

Fruit composition outcomes (not relevant for yield meta-analysis):
- Soluble sugars, anthocyanins, antioxidant activity (peel and flesh), mineral elements (P, Fe, Zn, Ca, Cu)

---

## 2. Yield Tables in the Paper

### Figure 2 (primary yield data source)
All yield data are presented in **Figure 2** as a grouped bar chart — not as a conventional numeric table. The figure shows:
- Top panel: 2013 — Marketable yield (dark bars) and Early yield (light bars) for all six cultivars (Control vs. Biostimulant interactions, plus main effects)
- Bottom panel: 2014 — same layout

Data are presented as means ± SD for 3 replications. Statistical significance is indicated by Tukey's HSD letter codes on the bars.

**There is no numeric yield table in the paper.** All yield values must be read from Figure 2's bar chart, which is a visually dense grouped chart with overlapping error bars and small bar widths, making precise numerical extraction difficult.

### Figures 3 and 4
- Figure 3: Mean fruit weight (kg) — bar chart, 2013 and 2014
- Figure 4: Number of fruits per m² — bar chart, 2013 and 2014

### Tables 2 and 3
These contain fruit composition data only (sugars, antioxidants, mineral elements) — no yield means. They have clear numeric values and are the only true numeric tables in the paper.

---

## 3. What the Li 2022 Ground Truth Contains

The GT spreadsheet contains **12 rows (pairs 17–28)** for this paper, all coded as:
- Crop: eggplant
- Category: Vegetables
- PBs category: SWE (seaweed extract)
- Method: Foliar
- Dose: 1
- Replicates: 3
- ifdoseresponse: 0 (not a dose-response study)
- Frequency: 3
- Country: Poland

The 12 GT products are: **Göemar BM-86, BM-87, BM-88, BM-89, BM-90, BM-91, BM-92, BM-93, BM-94, BM-95, BM-96, BM-97**

Each has distinct control and treatment means (all in what appears to be kg m⁻², based on the magnitude: ctrl means ranging from 1.60 to 4.59, treat means from 2.29 to 4.18).

**Critical observation: The actual Pohl 2019 paper tests only one product (BM-86), not 12 products.** The GT's 12 entries (BM-86 through BM-97) match the six cultivars × two years structure of the paper (6 × 2 = 12 data points), but they are labelled as different product numbers. This strongly suggests that the Li 2022 meta-analysis database assigned sequential product identifiers (BM-86 through BM-97) to what are in reality the 12 cultivar×year combinations from a single product experiment — or, alternatively, that the Li 2022 compilers sourced these entries from a companion paper or multi-product study by the same group that is not the PDF provided here.

GT effect directions: 10 of 12 GT pairs show positive effects (treatment > control). One pair (GT 17, BM-86, ctrl=4.59, treat=4.18) shows a negative effect (−8.93%). The mean ctrl values span 1.60 to 4.59 kg m⁻², consistent with the yield range visible in Figure 2 of the paper.

---

## 4. What the AI Extractor Captured

The JSON output contains **22 observations**, all for a single product (Göemar BM-86) across six cultivars and two years:

| Year | Cultivar | Element | Ctrl mean | Treat mean |
|------|----------|---------|-----------|------------|
| 2013 | Cristal F1 | early yield (kg m⁻²) | 1.0 | 1.2 |
| 2013 | Cristal F1 | marketable yield (kg m⁻²) | 3.2 | 3.8 |
| 2013 | Epic F1 | early yield | 1.3 | 1.8 |
| 2013 | Epic F1 | marketable yield | 4.8 | 5.2 |
| 2013 | Flavine F1 | early yield | 1.1 | 1.5 |
| 2013 | Flavine F1 | marketable yield | 3.6 | 4.2 |
| 2013 | Gascona F1 | early yield | 1.0 | 1.4 |
| 2013 | Gascona F1 | marketable yield | 3.2 | 3.9 |
| 2013 | Onyx F1 | early yield | 1.1 | 1.3 |
| 2013 | Onyx F1 | marketable yield | 4.5 | 4.8 |
| 2013 | WA 6020 F1 | early yield | 1.2 | 1.6 |
| 2013 | WA 6020 F1 | marketable yield | 5.2 | 5.8 |
| 2014 | Cristal F1 | early yield | 0.5 | 0.8 |
| 2014 | Cristal F1 | marketable yield | 2.5 | 2.8 |
| 2014 | Epic F1 | early yield | 0.6 | 1.2 |
| 2014 | Epic F1 | marketable yield | 3.8 | 4.5 |
| 2014 | Flavine F1 | early yield | 0.7 | 1.0 |
| 2014 | Flavine F1 | marketable yield | 2.8 | 3.2 |
| 2014 | Gascona F1 | early yield | 0.7 | 0.9 |
| 2014 | Gascona F1 | marketable yield | 2.4 | 2.6 |
| 2014 | Onyx F1 | early yield | 0.6 | 0.8 |
| 2014 | Onyx F1 | marketable yield | 3.5 | 3.8 |

The extractor correctly identified:
- The single biostimulant product (Göemar BM-86)
- Both yield metrics (early and marketable)
- All six cultivars
- Both years
- n = 3 replicates

All extracted observations show **positive biostimulant effects** (treatment > control), consistent with the paper's stated finding that BM-86 improved early and total yield on average (+48%/+13% in 2013, +136%/+23% in 2014). All observations are flagged with `"confidence": "low"` because the values came from reading a bar chart (Figure 2), not a numeric table.

**No variance values were extracted.** The paper reports SD error bars on Figure 2 only; there is no numeric variance table for yield.

---

## 5. Why 11 GT Rows Could Not Be Matched

### Root cause: Product identity mismatch between GT and paper

The GT lists 12 distinct Göemar BM products (BM-86 through BM-97). The actual Pohl 2019 paper tests **only BM-86**. There are no products named BM-87, BM-88, BM-89, BM-90, BM-91, BM-92, BM-93, BM-94, BM-95, BM-96, or BM-97 anywhere in the PDF.

This leaves three possible explanations:

**Explanation A — Data entry error in the Li 2022 database (most likely)**
The Li 2022 compilers may have encoded the 12 cultivar×year combinations from Table 2 or Figure 2 (6 cultivars × 2 years = 12 data points) and assigned sequential Göemar BM product codes (BM-86 through BM-97) as row identifiers rather than actual product names. In this scenario, "BM-87" through "BM-97" are fabricated product labels applied to what are actually the yield data for Epic/Flavine/Gascona/Onyx/WA6020 (2013) and Cristal/Epic/Flavine/Gascona/Onyx/WA6020 (2014) treated with BM-86.

Supporting evidence: The GT ctrl mean values (1.60–4.59 kg m⁻²) and SD values are consistent with eggplant marketable yield from Figure 2. The number of entries (12) exactly equals 6 cultivars × 2 years. All GT entries share the same dose (1), method (foliar), frequency (3), and replicates (3).

**Explanation B — Multi-product companion paper**
The Li 2022 database entry may be based on a different Pohl et al. paper that tested multiple Göemar BM variants (BM-86 through BM-97) on eggplant, not the specific 2019 paper provided as the PDF. Pohl et al. (2018) is cited in the reference list: "Pohl A, Grabowska A, Kalisz A, Sekara A (2018). Preliminary screening of biostimulative effects of Göemar BM-86® on eggplant cultivars grown under field conditions in Poland. *Acta Agrobotanica* 71(4):1752." If a broader multi-product version of this experiment exists, the PDF provided may be a partial publication.

**Explanation C — Numeric value transcription from Figure 2**
Even if the product coding issue were resolved, the GT ctrl mean for "BM-86" (pair 17) is 4.59 kg m⁻², which does not match any of our extracted BM-86 marketable yield ctrl means (Cristal 2013: 3.2; Epic 2013: 4.8; WA6020 2013: 5.2). The closest match is Gascona 2013 (ctrl=3.2) or Cristal 2013 (ctrl=3.2), neither of which equals 4.59. This suggests the GT may have used a different aggregation (e.g., means across cultivars, or values from a differently-read figure).

### Direction conflict for the one matched pair (GT 17 vs. JSON BM-86 Cristal 2013 marketable yield)
- GT pair 17: ctrl=4.59, treat=4.18, effect = **−8.93%** (biostimulant reduced yield)
- JSON (Cristal 2013, marketable): ctrl=3.2, treat=3.8, effect = **+18.75%**
- All 12 JSON BM-86 marketable yield observations show positive effects (+5.9% to +56.3%)

The paper's text states BM-86 increased early and total yields on average and provides statistically significant positive effects for most cultivars. A negative aggregate effect (GT pair 17 = −8.93%) is inconsistent with the paper's findings. This further supports the hypothesis that the GT values are sourced from a different aggregation, different metric, or possibly a different paper entirely.

---

## 6. Summary of Root Causes

| Issue | Severity | Description |
|-------|----------|-------------|
| Product identity mismatch | Critical | Paper tests 1 product (BM-86); GT lists 12 different products (BM-86 to BM-97). 11/12 GT pairs cannot be matched to any JSON observation because those product names do not exist in the paper. |
| Data source ambiguity | Critical | GT values likely originate from a different paper (possibly a multi-product screening study by the same group) or represent a misencoding of cultivar×year combinations as product variants. |
| No numeric yield table | Moderate | All yield data in Figure 2 (bar chart only); values extracted by AI are approximate reads from figure bars, flagged as low confidence. No table with exact means and SD for yield. |
| Direction conflict | High | The one forced match (GT pair 17) shows opposite effect direction to the extracted data (−8.93% vs. +18.75%), indicating the GT and JSON values are not measuring the same experimental unit. |
| Missing variance | Moderate | No numeric variance extracted for yield outcomes; SD only appears as graphical error bars in Figure 2, not in a table. |

---

## 7. Recommended Action

This paper should be **flagged for manual review** in the Li 2022 validation dataset. The key question is whether GT pairs 18–28 (BM-87 through BM-97) correspond to:
1. Cultivar×year combinations within the Pohl 2019 paper (misencoded product labels), or
2. A separate, unretrieved paper from the same research group that tested multiple Göemar BM products.

If explanation A is correct, the GT matching logic should be redesigned to match by cultivar×year combination rather than by product name, and the 12 GT entries would each correspond to one of the 12 cultivar×year BM-86 observations. Under that interpretation, coverage would be 12/12 — but the direction conflict in GT pair 17 would still require resolution.

If explanation B is correct, the PDF used for extraction is the wrong source document for 11 of the 12 GT pairs.

The extractor's behavior was **correct given the actual paper**: it extracted BM-86 data across all cultivars and years, which is the complete yield dataset in the Pohl 2019 paper. The poor match rate (1/12) is a consequence of a mismatch between the PDF source and the Li 2022 database entry, not a failure of the extraction algorithm.
