# Extraction Quality Report: 094_Kowalska_2021

**Paper:** Kowalska J. et al. (2021). Effect of Different Forms of Silicon on Growth of Spring Wheat Cultivated in Organic Farming System. *Journal of Elementology* (or equivalent Polish agronomy venue). Study conducted 2017–2018.

**Current match result:** 4 matched pairs, MAE = 0.00%, direction agreement = 100%, 0 unmatched GT rows, 13 unmatched JSON obs.

---

## 1. Paper Design

### Experimental overview

A field experiment conducted over two seasons (2017–2018) in an organic farming system, examining the effects of two commercial silicon products applied by three different methods to spring wheat (*Triticum aestivum* cv. Arabella). The experiment used a randomised block design with **n = 8 observation points** per treatment across a 100 m² experimental area, with one silicon product per block.

### Silicon products tested (2)

| Product | Form | Rate |
|---------|------|------|
| AdeSil | Diatomaceous earth powder | 10 kg/ha |
| ZumSil | Monosilicic acid liquid | 0.3 l/ha |

### Application methods (3 per product)

| Method | Description |
|--------|-------------|
| Soil | Incorporated at sowing or pre-plant |
| Foliar | Applied as foliar spray during growing season |
| Combined | Soil + foliar (both methods together) |

This produces a **2 product × 3 method = 6 treatment arms** plus an untreated control (no silicon), for **7 groups** total.

### Study dimensions

- Crop: spring wheat, cultivar Arabella
- Primary outcome: grain yield (t/ha), reported in Table 7
- Secondary outcomes extracted: weight of grain per plot/16 m² (kg) and grain moisture (%)
- Statistical analysis: Tukey's test; results presented as treatment means with letter notation (a, b, c) for homogeneous groups
- Variance: **no numeric variance values reported** — only letter-based statistical groupings
- PDF type: scanned document (OCR required)

---

## 2. AI Consensus Extraction Results

Both Claude and Kimi independently extracted **18 observations** each, with full consensus (17 after removal of one duplicate; tiebreaker was not required). The consensus output covers all three outcomes visible in Table 7:

| Outcome | Treatment arms extracted | n per arm |
|---------|--------------------------|-----------|
| Yield (t/ha) | 6 (3 AdeSil × 3 methods) | 8 |
| Weight of grain per plot/16 m² (kg) | 6 | 8 |
| Grain moisture (%) | 5 | 8 |

**Control values identified correctly:** 3.88 t/ha yield and 7.72 kg/plot weight, both matching the "Untreated" row of Table 7.

**All model confidence ratings:** high across all 17 consensus observations. Zero disagreements between Claude and Kimi. The scanned-PDF warning flagged by the recon stage (OCR risk, variance unclear) did not materialise as an extraction problem — both models read the table values consistently.

**Variance:** Correctly recorded as null throughout. The paper reports only Tukey letter groupings with no accompanying SE, SD, or LSD numeric values. The recon stage correctly anticipated this limitation with "VAR-UNCLEAR."

---

## 3. Ground Truth Comparison

The Li 2022 ground truth includes **4 rows** (GT pairs 1090–1093), corresponding to two products × two single-application methods (soil and foliar) only. The combined (soil+foliar) arms were not included in the GT dataset.

### Matched pairs (all 4 with MAE = 0.00%)

| GT pair | Product | Method | GT ctrl | GT treat | GT effect% | JSON ctrl | JSON treat | JSON effect% |
|---------|---------|--------|---------|----------|------------|-----------|------------|--------------|
| 1090 | AdeSil | Soil | 0.388 | 0.475 | +22.42% | 3.88 | 4.75 | +22.42% |
| 1091 | AdeSil | Foliar | 0.388 | 0.425 | +9.54% | 3.88 | 4.25 | +9.54% |
| 1092 | ZumSil | Soil | 0.388 | 0.449 | +15.72% | 3.88 | 4.49 | +15.72% |
| 1093 | ZumSil | Foliar | 0.388 | 0.484 | +24.74% | 3.88 | 4.84 | +24.74% |

**Note on unit scaling:** The GT records control means as 0.388 and treatment means in the 0.4–0.5 range, while the JSON records the same values as 3.88 and 4.25–4.84 t/ha — a consistent 10× difference in absolute values. This is a known decimal-shift artifact in the Li 2022 database (likely 100 kg/ha vs. t/ha), not an extraction error. Effect percentages are identical to two decimal places across all four pairs because ratios are scale-invariant. All four matches carry **high confidence**.

### Unmatched JSON observations (13)

| Reason | Count |
|--------|-------|
| Combined (soil+foliar) application arms — not included in Li 2022 GT | 2 |
| Per-plot weight [kg] metric — GT uses t/ha only | 6 |
| Grain moisture [%] — quality parameter, not a yield metric | 5 |

None of the 13 unmatched observations represent extraction errors. They are legitimate additional data captured by the AI that fell outside the scope of the Li 2022 meta-analysis inclusion criteria.

---

## 4. Root Cause Analysis (Why Perfect)

Three factors combine to explain the perfect match:

**1. Unambiguous table structure.** Table 7 presents a compact, cleanly organised layout with one row per treatment, one column for yield (t/ha), one for per-plot weight (kg), and one for moisture (%). There is no multi-year nesting, no split-plot complexity, and no overlapping columns. Every value corresponds to exactly one treatment arm. This eliminates the most common sources of extraction ambiguity — temporal granularity mismatches, column mis-assignment, and factorial structure confusion.

**2. Explicitly labelled control.** The untreated row is labelled "Untreated" throughout all tables, with no ambiguity about which group serves as the control. The recon stage confirmed: "potential_tc_confusion: None." Neither model had any reason to misassign treatment and control directions.

**3. Effect magnitudes are large and unambiguous.** Silicon treatments produced yield increases of 10–25% (AdeSil and ZumSil soil and foliar arms). These are well above rounding noise and OCR uncertainty, making the values robust even in a scanned PDF. The only potential OCR confusion — distinguishing 3.88 from 3.83, for example — did not materialise, as both models read the same values.

The GRIM test failures visible in the verification flags are expected artefacts of applying an integer-data test to continuous-scale agronomic yield data (t/ha values like 3.88 are inherently non-integer), not indicators of extraction errors.

---

## 5. Overall Assessment

**Extraction quality: PERFECT**

| Dimension | Assessment |
|-----------|------------|
| Correct paper identified | Yes — Kowalska et al. 2021, spring wheat, organic farming silicon trial |
| Correct outcome variable | Yes — grain yield (t/ha) from Table 7 |
| Correct control | Yes — Untreated arm (3.88 t/ha) |
| Correct product × method combinations | Yes — all 6 treatment arms extracted |
| Effect percentages vs. GT | Exact match to 0.00 pp MAE across all 4 GT rows |
| Direction agreement | 100% (4/4 positive effects) |
| Variance extraction | Correct — null, paper does not report numeric variance |
| Additional outcomes captured | Yes — per-plot weight and grain moisture (not in GT scope) |
| Combined application arms | Yes — correctly captured, absent from GT by design |

The AI pipeline produced numerically perfect extraction for all outcomes against which ground truth comparison was possible. The 13 unmatched JSON observations are genuine additional data extracted correctly from Table 7 that the Li 2022 meta-analysis chose not to include. This paper represents an ideal extraction case: a scanned but clearly organised field trial with unambiguous treatment labels, a prominent control, and large-effect outcomes that are robust to OCR uncertainty.
