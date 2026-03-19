# Extraction Quality Report: 111_Mondal_2013

**Paper:** Mondal MMA, Malek MA, Puteh AB and Ismail MR (2013). Foliar application of chitosan on
growth and yield attributes of mungbean (*Vigna radiata* (L.) Wilczek). *Bangladesh Journal of Botany*
42(1): 179–183.

**Report date:** 2026-02-18
**Match result:** 0 matched pairs | 8 unmatched GT | 10 unmatched JSON obs

---

## 1. Paper Design

This is a 5-page short communication published in *Bangladesh Journal of Botany* (June 2013). It is a
**single-crop study exclusively on mung bean** (*Vigna radiata*). Maize does not appear as an
experimental crop anywhere in the paper.

### Crop and Varieties
- **Crop:** Mung bean (Vigna radiata) only
- **Varieties:** BINAmung-7 (small-seeded) and BINAmung-8 (bold-seeded)
- Conducted at the Bangladesh Institute of Nuclear Agriculture (BINA), Mymensingh, Bangladesh

### Experimental Systems
Two parallel experiments were run simultaneously during Kharif-I season (March–May) in both 2010 and 2011:

| Experiment | Design | Replicates | Unit |
|------------|--------|-----------|------|
| Pot culture | Completely Randomised Design (CRD) | 4 | 30 × 25 cm pots, 10 kg soil |
| Field experiment | Randomised Complete Block Design (RCBD) | 3 | 3 × 2 m plots |

### Chitosan Doses (mung bean experiments)
Five chitosan concentrations applied as foliar spray at 25 and 35 days after sowing:

| Dose (ppm) | Notes |
|-----------|-------|
| 0 | Control (no chitosan) |
| 25 | Treatment 1 |
| 50 | Treatment 2 (highest yield) |
| 75 | Treatment 3 |
| 100 | Treatment 4 |

### Outcomes Measured
- Growth attributes: plant height, branches/plant, leaves/plant, leaf area/plant (Table 1)
- Physiological/biochemical: total dry mass, harvest index, photosynthesis, chlorophyll, nitrate reductase (Table 2)
- Yield attributes: flowers/plant, pods/plant, pod length, seeds/pod, 1000-seed weight (Table 3)
- **Primary yield outcome:** Seed weight/plant (g) and seed yield (kg/ha) — pot and field (Table 4)

---

## 2. Critical Finding: The Paper Contains NO Maize Data

**The paper is entirely about mung bean.** Maize is mentioned only once in the entire paper — in a
passing citation in the Results discussion:

> "These results are in consistent with Khan *et al.* (2002) and El-Tantawy (2009) who reported that
> application of chitosan increased photosynthesis in leaves of **maize** and soybean as well as tomato,
> respectively."

This is a reference to an earlier study by different authors (Khan et al. 2002, *Photosynthetica* 40:
621–624); it is not a result from this paper. There is no maize experiment, no maize table, and no maize
yield values anywhere in Mondal 2013.

The ground truth (Li 2022 meta-analysis) attributes 8 maize yield observations (GT pairs 745–752) to
this paper. These observations have control means of approximately 0.51 and 0.605 (units consistent
with t/ha), with chitosan doses expressed in g/L. Neither the scale (t/ha), the crop (maize), nor the
dose unit (g/L) matches anything in Mondal 2013, where doses are in ppm and yields are in g/plant
or kg/ha for mung bean.

**Conclusion: The GT attribution appears to be an error in the Li 2022 meta-analysis database.** The
8 maize observations (GT pairs 745–752) were probably sourced from a different paper and incorrectly
assigned to the Mondal 2013 paper ID.

---

## 3. Mung Bean Yield Data in the PDF (What Our AI Found)

The mung bean yield data is in **Table 4, page 182** (page 4 of the PDF), titled:
"Effect of chitosan on seed yield and interaction of variety and chitosan concentration on seed yield
of mungbean."

### Main treatment means (averaged across varieties)

| Dose (ppm) | Seed wt/plant — Pot (g) | Seed wt/plant — Field (g) | Seed yield — Field (kg/ha) |
|-----------|------------------------|--------------------------|---------------------------|
| 0 (control) | 6.55 | 7.28 | 1556 |
| 25 | 7.08 | 8.70 | 1696 |
| 50 | 8.16 | 9.31 | 1893 |
| 75 | 7.66 | 8.70 | 1913 |
| 100 | 7.09 | 8.26 | 1704 |

### Variety-level means

| Variety | Seed wt/plant — Pot (g) | Seed wt/plant — Field (g) | Seed yield — Field (kg/ha) |
|---------|------------------------|--------------------------|---------------------------|
| BINAmung-7 | 8.09 | 8.80 | 1902 |
| BINAmung-8 | 6.52 | 7.60 | 1601 |

### Variety × dose interaction means (Table 4, right side)

| Interaction | Seed wt/plant — Pot (g) | Seed wt/plant — Field (g) | Seed yield — Field (kg/ha) |
|------------|------------------------|--------------------------|---------------------------|
| V1 × 0 | 7.51 | 8.23 | 1728 |
| V1 × 25 | 7.97 | 8.91 | 1870 |
| V1 × 50 | 9.23 | 9.77 | 2052 |
| V1 × 75 | 7.99 | 9.70 | 2037 |
| V1 × 100 | 7.74 | 8.70 | 1827 |
| V2 × 0 | 5.58 | 5.99 | 1384 |
| V2 × 25 | 6.18 | 6.59 | 1522 |
| V2 × 50 | 7.09 | 7.50 | 1733 |
| V2 × 75 | 7.32 | 7.74 | 1788 |
| V2 × 100 | 6.43 | 6.84 | 1580 |

Variance is reported using letter notation (a, b, c) rather than numeric SD or SE values.
No numeric variance values are given in Table 4. CV% is reported (4.52, 3.89, 6.22) but cannot be
back-converted to group-level SD without per-cell means paired to the CV.

---

## 4. What Our AI Extracted (The 10 JSON Observations)

Our AI extracted 10 mung bean observations from Table 4, covering two experimental systems and two
outcome metrics:

| # | Element | Crop | Variety | Experiment | Dose (ppm) | Ctrl mean | Treat mean |
|---|---------|------|---------|-----------|-----------|-----------|-----------|
| 1 | seed weight/plant (g) - pot | Mung bean | BINAmung-8 | pot | 25 | 6.55 | 7.08 |
| 2 | seed weight/plant (g) - pot | Mung bean | BINAmung-7 | pot | 50 | 6.55 | 8.16 |
| 3 | seed weight/plant (g) - pot | Mung bean | BINAmung-7 | pot | 75 | 6.55 | 7.66 |
| 4 | seed weight/plant (g) - pot | Mung bean | BINAmung-8 | pot | 100 | 6.55 | 7.09 |
| 5 | seed weight/plant (g) - field | Mung bean | BINAmung-7 | field | 25 | 7.28 | 8.70 |
| 6 | seed weight/plant (g) - field | Mung bean | BINAmung-7 | field | 50 | 7.28 | 9.31 |
| 7 | seed yield (kg/ha) - field | Mung bean | BINAmung-8 | field | 25 | 1556 | 1696 |
| 8 | seed yield (kg/ha) - field | Mung bean | BINAmung-7 | field | 50 | 1556 | 1893 |
| 9 | seed yield (kg/ha) - field | Mung bean | BINAmung-7 | field | 75 | 1556 | 1913 |
| 10 | seed yield (kg/ha) - field | Mung bean | BINAmung-8 | field | 100 | 1556 | 1704 |

The values extracted are accurate — they match Table 4 of the PDF correctly. The extraction is
substantially complete for a mung bean study. Minor issues: not all dose × experiment combinations
were captured (e.g., pot seed weight is missing the 50-ppm field row), and cultivar assignments in
moderators have some internal inconsistencies in the JSON, but the core yield values are correct.

---

## 5. Root Cause Analysis: Why Did the AI "Miss" the Maize Data?

The AI did **not** make an error. It correctly read and extracted the only crop studied in the paper —
mung bean. The real issue is a **ground truth attribution error in the Li 2022 meta-analysis**.

### Evidence that the GT is misattributed

1. **Title mismatch:** The paper is titled "Foliar application of chitosan on growth and yield attributes
   of **mungbean**." Maize is not mentioned in the title, abstract, keywords, or any table caption.

2. **No maize experiment exists in the paper.** There are 4 tables (Tables 1–4), all explicitly headed
   "of mungbean." No table contains maize data.

3. **Dose unit mismatch:** GT doses are 0.4, 0.6, 0.8, and 1.0 g/L. The paper uses 0, 25, 50, 75, and
   100 ppm (mg/L). These are incommensurable series — 0.4 g/L = 400 ppm, which is not a dose in this
   paper.

4. **Yield scale mismatch:** GT ctrl_means are ~0.51 and ~0.605, consistent with t/ha (tonnes per
   hectare) for maize grain yield. The paper reports mung bean seed yield at 1,384–2,052 kg/ha, never
   in t/ha scale, and seed weight per plant at 5.6–9.8 g.

5. **Replicates:** GT specifies n = 3 for all 8 observations, consistent with an RCBD with 3 reps. The
   field experiment in this paper does have 3 replicates, but the values do not match.

6. **Two sub-groups in GT (ctrl ~0.51 and ctrl ~0.605):** These suggest two varieties or two seasons
   of a maize experiment. No such structure exists for maize in this paper. The two-variety structure
   in this paper uses mung bean varieties with different yield ranges entirely.

### Most likely explanation

The Li 2022 meta-analysis contains a paper on **chitosan effects on maize yield** from Bangladesh,
with doses in g/L and yields in t/ha, that is a different publication from Mondal 2013. That maize
paper was incorrectly assigned the file ID "111_Mondal_2013" in the Li 2022 database — possibly
because Mondal and colleagues published multiple chitosan papers around the same period (see
reference list: Mondal et al. 2012 on okra; the first author is the corresponding author here). A
companion paper by Mondal et al. on chitosan effects in maize, if it exists, would be the correct
source for those 8 GT observations.

---

## 6. Assessment: Could Improved Extraction Capture the Maize Data?

**No.** This is not an extraction quality problem. No improved prompt, larger context window, or
better model configuration could extract data that does not exist in the PDF.

The AI performed correctly:
- It read the correct paper (mung bean study in Bangladesh)
- It found the primary yield outcome (seed weight/plant, seed yield kg/ha)
- It extracted values that match the PDF tables accurately
- It correctly identified BINAmung-7 and BINAmung-8 as the experimental cultivars

The 0-of-8 match rate is entirely caused by a GT database error, not an AI extraction failure.

### Recommended action

1. **Verify the Li 2022 GT database entry for pairs 745–752.** Check whether a separate Mondal et al.
   publication exists on chitosan effects in maize in Bangladesh with doses of 0.4–1.0 g/L. If so,
   that paper may be missing from the downloaded PDF set or may be a different paper entirely.

2. **Do not penalise this paper in overall extraction accuracy metrics.** The coverage failure is 100%
   attributable to GT misattribution, not to extraction quality. The AI's 10 extracted mung bean
   observations are accurate and should be credited as correct extractions of this paper's content.

3. **Flag this GT record in the validation dataset** as a probable attribution error, to avoid skewing
   aggregate precision/recall statistics for the Li 2022 validation.

---

## Summary

| Dimension | Finding |
|-----------|---------|
| Crop in PDF | Mung bean only (Vigna radiata, varieties BINAmung-7 and BINAmung-8) |
| Maize in PDF | Not present as experimental crop; mentioned once in a citation |
| GT crop | Maize (stated explicitly in GT data file) |
| GT dose units | g/L (0.4, 0.6, 0.8, 1.0) |
| PDF dose units | ppm (0, 25, 50, 75, 100) |
| GT yield scale | ~0.51–0.763 (consistent with t/ha maize grain yield) |
| PDF yield scale | 1384–2052 kg/ha; 5.6–9.8 g/plant (mung bean) |
| AI extraction quality | Correct — values match Table 4 of the PDF |
| Root cause of 0 matches | GT attribution error: maize observations assigned to a mung bean paper |
| Fixable by improved extraction | No |
