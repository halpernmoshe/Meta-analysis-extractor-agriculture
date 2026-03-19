# Extraction Quality Report: 002_Abdel-Mawgoud_2010

**Paper ID:** `002_Abdel-Mawgoud_2010_Growth and yield responses of strawberry`
**Match result:** 0 matched pairs | 8 unmatched GT rows | 20 unmatched JSON observations
**Report generated:** 2026-02-18

---

## 1. What the PDF Actually Contains

The PDF at this file path is **NOT the Abdel-Mawgoud 2010 strawberry paper**. It is a completely different article:

> **Al-Tawaha et al. (2020). "Growth, Yield and Biochemical Responses in Barley to DAP and Chitosan Application Under Water Stress."** *Journal of Ecological Engineering*, 21(6): 86–93. DOI: 10.12911/22998993/123251.

Key facts about the actual PDF:
- **Crop:** Barley (*Hordeum vulgare* L.)
- **Biostimulant:** Chitosan (0, 5, 10 g L⁻¹ foliar), combined with DAP fertilizer (0 or 100 kg ha⁻¹)
- **Study type:** Split-plot field experiment, rainfed, Jordan (Irbid governorate), 2014/15 and 2015/16 growing seasons
- **Replications:** 3
- **Outcomes measured:** Grain yield per plant (g), number of spikes per plant, number of grains per spike, 1000-grain weight (g), spike length (cm), plant height (cm), days to heading, protein content (%), starch content (%)
- **Key tables:**
  - Table 2: Yield and yield components (grain yield, spikes/plant, grains/spike, 1000-grain weight) by growing season, fertilizer, and chitosan treatment
  - Table 3: Phenological traits (spike length, plant height, days to heading) by growing season, fertilizer, and chitosan treatment
  - Table 4: Protein and starch content (%) by fertilizer × chitosan × growing season (2×3×2 factorial)

The paper mentions Abdel-Mawgoud (2010) only once — as a citation in the results section ("Abdel-Mawgoud, (2010) reported the enhanced plant height, the number of leaves and the yield in strawberry plants by the foliar application of chitosan"). The actual Abdel-Mawgoud 2010 paper (the one the GT refers to) is cited as reference #2 in the reference list: *"Growth and yield responses of strawberry plants to chitosan application. Eurasian Journal of Scientific Research, 39: 170–177."*

---

## 2. What the Ground Truth (Li 2022) Expects

The GT contains **8 observations** (pairs 655–662), all with:
- **Crop:** Strawberry
- **Product:** Chito-care (chitosan formulation)
- **Method:** Foliar application
- **Country:** Egypt
- **Replicates:** 3
- **Outcome:** Implied yield metric (fresh fruit weight or equivalent), units consistent with kg/plant or similar (ctrl_mean values of ~2.93 and ~4.85, which are plausible for strawberry yield in kg/plant across seasons)

The 8 GT rows break into two groups:
- **Pairs 655–658 (dose-response arm, ifdoseresponse=1):** Four chitosan dose levels (0.25, 0.5, 0.75, 1 g/L?) with a shared control mean of ~2.9259, representing a dose-response experiment from one season or condition
- **Pairs 659–662 (ifdoseresponse=0):** Four observations with a shared control mean of ~4.8519, representing a second season or cultivar comparison at dose=1

These values (ctrl_mean ~2.93–4.85, treat_mean up to ~8.78) are consistent with strawberry fruit yield data expressed in units such as kg per plant, total fruit weight per plant (g × 100), or similar. The GT is clearly drawing on Abdel-Mawgoud, Tantawy, El-Nemret, Sassine, and Y.N. (2010), "Growth and yield responses of strawberry plants to chitosan application," *Eurasian Journal of Scientific Research*, 39: 170–177 — an Egypt-based greenhouse or field experiment on strawberry with multiple chitosan dose levels.

---

## 3. What the AI Extractor Captured (JSON Observations)

The extractor produced **20 observations**, all from the barley paper that was physically present in the PDF file:

| # | Element | Control mean | Treatment mean | Treatment |
|---|---------|-------------|----------------|-----------|
| 1–2 | Grain yield per plant (g) | 5.3 | 5.5 / 5.7 | 5 / 10 g/L chitosan |
| 3–4 | Number of spikes per plant | 1.5 | 1.5 / 2.0 | 5 / 10 g/L chitosan |
| 5–6 | Number of grains per spike | 24.0 | 25.5 / 26.5 | 5 / 10 g/L chitosan |
| 7–8 | 1000 grain weight (g) | 46.0 | 49.0 / 51.0 | 5 / 10 g/L chitosan |
| 9 | Spike length (cm) | 6.5 | 6.5 | 5 g/L chitosan |
| 10–11 | Plant height (cm) | 70.0 | 71.0 / 74.0 | 5 / 10 g/L chitosan |
| 12 | Days to heading | 87.5 | 87.5 | 5 g/L chitosan |
| 13–16 | Protein content (%) | 10.9–11.9 | 11.05–12.35 | 5 / 10 g/L chitosan |
| 17–20 | Starch content (%) | 60.2–60.55 | 60.3–62.2 | 5 / 10 g/L chitosan |

Verification against the PDF confirms these values are **correctly extracted from the barley paper**. For example:
- Table 2 shows Control grain yield = 5.3 g/plant, 5 g/L = 5.5, 10 g/L = 5.7 — matches JSON exactly
- Table 3 shows Control plant height = 70.0 cm, 5 g/L = 71.0, 10 g/L = 74.0 cm — matches JSON exactly
- Table 4 protein/starch values match the JSON observations

The AI extraction was **internally accurate** for the file it received. It faithfully read and extracted the data from the barley paper. The problem is not extraction quality — it is that the extractor processed the wrong PDF.

---

## 4. Root Cause: Wrong PDF at This File Path

The failure is a **file assignment error (mislabeling)**, not an extraction error. The file:

```
Li 2022/downloaded_papers/002_Abdel-Mawgoud_2010_Growth and yield responses of strawberry.pdf
```

contains the **Al-Tawaha et al. 2020 barley paper**, not the Abdel-Mawgoud et al. 2010 strawberry paper. The filename was assigned to the wrong downloaded PDF during the dataset construction phase.

Evidence:
1. The PDF title page reads: "Growth, Yield and Biochemical Responses in **Barley** to DAP and Chitosan Application Under Water Stress" — not strawberry, not Abdel-Mawgoud as first author
2. The actual author is Abdel Rahman M. Al-Tawaha et al. (2020), *Journal of Ecological Engineering*
3. The year is 2020, not 2010
4. The crop is barley (*Hordeum vulgare*), not strawberry (*Fragaria* spp.)
5. The study was conducted in Jordan, not Egypt (GT country = Egypt)
6. Abdel-Mawgoud 2010 appears only as a cited reference within this barley paper (#2 in the reference list)

The barley paper likely entered the dataset because it cites the Abdel-Mawgoud 2010 strawberry paper and discusses chitosan effects on yield, causing a bibliographic database or download script to retrieve this barley paper instead of the intended Egyptian strawberry paper.

---

## 5. Why There Are 0 Matches

There are zero matches because the GT rows and the JSON observations describe entirely different experiments:

| Dimension | GT (Li 2022 expects) | JSON (what was extracted) |
|-----------|---------------------|--------------------------|
| Crop | Strawberry | Barley |
| Country | Egypt | Jordan |
| Year | 2010 | 2020 |
| Outcome units | ~2.9–8.8 (kg/plant or similar) | 5.3–5.8 g/plant grain yield |
| Dose levels | 0.25, 0.5, 0.75, 1 (g/L or %) | 0, 5, 10 g/L |
| Metric | Fruit yield (implied) | Grain yield + morphological |

No numerical overlap exists, and no re-scoring of the matching algorithm could bridge this gap. The two datasets are from different plants, different countries, different decades, and different journals.

---

## 6. Could Better Extraction Fix This?

**No.** This failure cannot be resolved by improving the extraction algorithm, prompts, or post-processing logic. The correct data (strawberry fruit yield from Abdel-Mawgoud et al. 2010, Egypt) does not exist anywhere in the PDF at this file path.

The only remediation paths are:

1. **Locate and download the correct PDF:** Abdel-Mawgoud A., Tantawy M.R., El-Nemret A.S., Sassine M.A., Y.N. (2010). "Growth and yield responses of strawberry plants to chitosan application." *Eurasian Journal of Scientific Research*, 39: 170–177. This is the paper cited in the Al-Tawaha et al. 2020 reference list as #2.

2. **Replace the file:** Obtain the correct PDF and replace `002_Abdel-Mawgoud_2010_Growth and yield responses of strawberry.pdf` with the actual Abdel-Mawgoud 2010 content, then re-run extraction.

3. **Flag as unrecoverable:** If the Abdel-Mawgoud 2010 paper cannot be obtained, the 8 GT rows (pairs 655–662) should be flagged as unrecoverable (PDF unavailable) rather than as an extraction failure.

---

## 7. Summary

| Category | Detail |
|----------|--------|
| **Root cause** | Wrong PDF downloaded — file contains Al-Tawaha et al. 2020 barley paper instead of Abdel-Mawgoud et al. 2010 strawberry paper |
| **Failure type** | File assignment / coverage failure (PDF mislabeling) |
| **AI extraction quality** | Correct for the file it received — all 20 barley observations accurately reflect Tables 2, 3, and 4 of the PDF |
| **GT data present in PDF** | No — strawberry fruit yield data is absent from this file |
| **Fixable by better extraction?** | No |
| **Recommended action** | Locate and download the correct Abdel-Mawgoud 2010 *Eurasian Journal of Scientific Research* paper (Egypt, strawberry, chitosan dose-response) |
| **GT rows affected** | 8 (pairs 655–662, all unrecoverable from current file) |
| **JSON obs status** | 20 valid barley observations, not attributable to the Li 2022 dataset entry for this paper ID |
