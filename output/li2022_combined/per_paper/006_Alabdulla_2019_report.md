# Extraction Quality Report: 006_Alabdulla_2019

**Paper:** Al-Freeh, Alabdulla & Huthily (2019). "Effect of Mineral-Biofertilizer on Physiological Parameters and Yield of Three Varieties of Oat (*Avena sativa* L.)." *Basrah Journal of Agricultural Sciences*, 32(Special Issue): 8–25.

**Match result:** 0 matched pairs | 6 unmatched GT rows | 33 unmatched JSON observations

---

## 1. What the Paper Measures

This is a two-season field experiment (2016–2017 and 2017–2018) conducted at Al-Zubair district, Basrah, Iraq, on sandy loam soil. The study examines the effect of **microbial biofertilizers** (not humic acid) on physiological growth parameters and grain yield of three oat (*Avena sativa*) varieties: Genzania, Shaffaa, and Carloup.

**Experimental design:** Split-plot RCBD with 3 replicates. Main plots = 3 oat varieties; sub-plots = 7 biofertilizer treatments:

| Code | Treatment |
|------|-----------|
| B0 | No addition (control) |
| B1 | NPK mineral fertilizer (as recommended) |
| B2 | Bio-Fertilizer NPK (microbial inoculant) |
| B3 | Mineral PK + Bio-N |
| B4 | Mineral NP + Bio-K |
| B5 | Mineral P + Bio-NK |
| B6 | Mineral N + Bio-PK |

The microbial biofertilizers used are: *Azotobacter chroococcum* (N-fixer), *Pseudomonas putida* and *Pantoea agglomerans* (P-solubilizers), *Bacillus subtilis* and *Bacillus mucilaginosus* (K-solubilizers). These are applied as seed inoculants, not as foliar sprays.

**Outcomes measured (Table 2, both seasons):**
- Flag Leaf Area (FLA, cm²)
- Leaf Area Duration (LAD, days)
- Leaf Area Index (LAI)
- Crop Growth Rate (CGR, gm day⁻¹ m⁻²)
- Net Assimilation Rate (NAR, gm m⁻² day⁻¹)
- Relative Growth Rate (RGR, mg gm⁻¹ day⁻¹)
- Plant Height (cm)
- Number of Tillers (m⁻²)
- **Grain Yield (ton ha⁻¹)**

**There is no humic acid, fulvic acid, or any foliar-applied organic acid in this paper.** The paper is exclusively about soil-applied and seed-applied microbial (bacterial) biofertilizer combinations.

---

## 2. What the Li 2022 Ground Truth (GT) Expected

The GT dataset contains 6 rows for this paper (pairs 241–246), all classified as **HFA (humic/fulvic acid)** biostimulant applied via **foliar** method. The GT rows represent a **dose-response experiment** with three humic acid application rates across two seasons:

| GT Pair | Dose (L/ha) | Season | ctrl_mean (ton/ha) | treat_mean (ton/ha) | Effect size |
|---------|-------------|--------|--------------------|---------------------|-------------|
| 241 | 0.33 | 1 | 0.3117 | 0.5115 | +64.1% |
| 242 | 0.67 | 1 | 0.3117 | 0.6194 | +98.7% |
| 243 | 1.00 | 1 | 0.3117 | 0.5952 | +91.0% |
| 244 | 0.33 | 2 | 0.2459 | 0.4199 | +70.8% |
| 245 | 0.67 | 2 | 0.2459 | 0.5967 | +142.7% |
| 246 | 1.00 | 2 | 0.2459 | 0.5872 | +138.8% |

The GT control means (0.3117 and 0.2459 ton ha⁻¹) are an order of magnitude lower than the B0 control values in Table 2 of the actual PDF (2.739 and 3.889 ton ha⁻¹). This is a critical diagnostic: the numbers are wholly incompatible with the paper in the PDF file.

**These GT rows describe a completely different study** — one involving foliar humic acid applied at graduated doses to oat, with very low absolute grain yields. That experiment does not appear anywhere in the PDF.

---

## 3. What Our AI Extractor Captured (JSON Observations)

The AI extractor produced 33 observations, none of which are grain yield from the HFA dose-response arms, because those arms do not exist in the PDF. The extractor extracted what is actually present in the paper:

**Grain yield observations (2 of 33):**
- Season 1, B2 vs B0: ctrl = 2.739, treat = 7.942 ton ha⁻¹ (+190%)
- Season 2, B2 vs B0: ctrl = 3.889, treat = 11.562 ton ha⁻¹ (+197%)

These match the B2 row in Table 2 of the PDF precisely. The extractor correctly read the biofertilizer yield data.

**Non-yield observations (31 of 33):**
The remaining 31 observations span FLA, LAD, LAI, CGR, NAR, RGR, Plant Height, and Tiller Count — all correctly extracted from Table 2, but outside the scope of the Li 2022 yield-focused meta-analysis.

**Assessment of AI extraction quality:** The AI extracted the correct data for the paper that exists in the PDF. The values match Table 2 precisely (e.g., B2 Season 1 CGR = 11.19, NAR = 1.926, LAD = 138.00, LAI = 5.82 — all confirmed in the PDF). The extraction is accurate and competent.

---

## 4. Why There Are 0 Matches: Root Cause Analysis

The failure is not an extraction failure. It is a **ground truth assignment error** — the Li 2022 meta-analysis has attributed HFA (humic/fulvic acid) dose-response yield data to this paper's filename/ID, but the PDF at that path contains an entirely different study.

**Evidence for wrong-paper assignment:**

1. **Intervention mismatch:** The GT specifies PBsCategory = "HFA" (humic/fulvic acid) with foliar application. The PDF describes only soil/seed-applied microbial biofertilizers (*Azotobacter*, *Pseudomonas*, *Bacillus*). There is no humic acid anywhere in the paper, not in title, abstract, methods, results, or discussion.

2. **Quantitative magnitude mismatch:** The GT control means (0.3117 and 0.2459 ton ha⁻¹) differ from the PDF's B0 control means (2.739 and 3.889 ton ha⁻¹) by a factor of ~9–16x. This is not a unit conversion discrepancy — it is a categorical difference in experimental context.

3. **Treatment structure mismatch:** The GT expects three graded doses of a single foliar product (0.33, 0.67, 1.00 L/ha) — a classic dose-response structure. The PDF has seven biofertilizer combination treatments (B0–B6) with no dose gradient.

4. **Effect size pattern mismatch:** The GT effects for season 1 are +64%, +99%, +91% — a non-linear dose response typical of humic acid trials. The PDF's B2 vs B0 effect is +190% — consistent with a microbial inoculant on impoverished soil in a semi-arid environment, and structurally incomparable.

5. **Title evidence:** The paper is titled "Effect of Mineral-**Biofertilizer**..." and the author list includes "Alabdulla" — which appears to be the basis for the filename "006_Alabdulla_2019." However, the paper that would contain HFA dose-response data for oat at these control yield levels (0.31 ton/ha) must be a separate, unlocated publication.

**Most likely explanation:** The Li 2022 dataset cites a paper by Alabdulla or a co-author from 2019 that involved humic acid foliar application to oat. A second, distinct paper by overlapping authors on biofertilizer (this PDF) was assigned the same filename by the curator or downloaded in its place. The two papers share a year, institution, and crop, but test different interventions.

---

## 5. Could Better Extraction Fix This?

**No.** Better extraction cannot fix this problem because:

- The HFA dose-response grain yield data (ctrl ~0.31 ton/ha, three dose levels) is **physically absent** from the PDF.
- The AI extractor correctly read the paper that exists at this filepath.
- The mismatch is upstream of extraction: the wrong PDF was assigned to this paper's ID in the dataset.

**What would fix it:**
1. Locate the correct PDF — the actual Alabdulla (or co-author) 2019 paper on humic acid foliar application to oat in Iraq. This may be a separate conference paper, thesis chapter, or journal article from the University of Basrah with similar authorship but a different experimental intervention.
2. Replace the PDF at `006_Alabdulla_2019_Effect of foliar application of humic ac.pdf` with the correct file.
3. Re-run extraction on the correct paper.

---

## 6. Summary

| Dimension | Finding |
|-----------|---------|
| Paper in PDF | Al-Freeh et al. 2019, biofertilizer (microbial NPK) on oat — correctly extracted |
| GT expectation | HFA (humic acid) foliar dose-response on oat, ctrl ~0.31 ton/ha |
| Data present in PDF? | No — wrong paper loaded |
| AI extraction quality | Correct for the paper present; 33 obs accurately reflect Table 2 |
| Root cause | Wrong PDF assigned to this paper ID in the Li 2022 ground truth dataset |
| Verdict | Coverage failure due to file mismatch — not fixable by improved extraction |
| Recommended action | Locate and substitute the correct PDF for this paper ID |
