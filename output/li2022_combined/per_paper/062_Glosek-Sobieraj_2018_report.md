# Extraction Quality Report: 062_Głosek-Sobieraj_2018

**Paper:** Głosek-Sobieraj M, Cwalina-Ambroziak B, Hamouz K (2018). "The Effect of Growth Regulators and a Biostimulator on the Health Status, Yield and Yield Components of Potatoes (*Solanum tuberosum* L.)." *Gesunde Pflanzen* 70(1):1–11. https://doi.org/10.1007/s10343-017-0407-7

**Match result:** 0 matched pairs | 6 unmatched GT rows | 45 unmatched JSON observations

---

## 1. What the Paper Measures

This is a Polish field experiment conducted at Tomaszkowo near Olsztyn (53°41'N, 20°24'E) across three growing seasons (2013, 2014, 2015) using a randomised sub-block design with **three replications** (n = 3).

**Crop:** Potato (*Solanum tuberosum* L.), five cultivars differing in flesh colour:
- Irga and Satina (cream/yellow flesh)
- Valfi and Blaue St. Galler (purple flesh)
- Highland Burgundy Red / HB Red (red flesh)

**Treatments (foliar applied, four times per season):**
1. Control (no growth regulators)
2. 0.1% Asahi SL biostimulator (nitrophenols)
3. 1.0% Bio-Algeen S-90 (brown seaweed extract, *Ascophyllum nodosum*)
4. 0.2% Kelpak SL (brown algae extract, *Ecklonia maxima*)
5. Trifender WP (fungal biocontrol, *Trichoderma asperellum*)

**Primary outcomes measured:**
- Disease severity (late blight, early blight) — infection index %
- **Potato tuber total yield** (dt ha⁻¹) — reported in **Table 4** for each cultivar × year × treatment combination
- **Tuber size fraction distribution** (% of total yield in three fractions: >50 mm, 35–50 mm, <35 mm diameter) — reported in **Table 5**

**Design dimensions:** 5 cultivars × 4 treatments + 1 control × 3 years = 75 treatment means, each from n = 3 replications.

---

## 2. What the GT (Li 2022) Included

The Li 2022 ground truth database contains **6 rows** (pairs 484–489) for this paper. All 6 involve only two treatments:
- Bio-Algeen S-90 (SWE category)
- Kelpak SL (SWE category)

Asahi SL (a non-seaweed biostimulator) and Trifender WP (fungal biocontrol) are **absent** from the GT — consistent with Li 2022 being a meta-analysis of seaweed extract (SWE) biostimulants only.

The 6 GT rows form three control-mean groups (each with one Bio-Algeen and one Kelpak arm):

| Pair | Product | ctrl_mean (t/ha) | treat_mean (t/ha) | Effect % |
|------|---------|-----------------|------------------|---------|
| 484 | Bio-Algeen S-90 | 2.5379 | 2.7652 | +8.95% |
| 485 | Kelpak SL | 2.5379 | 2.9924 | +17.91% |
| 486 | Bio-Algeen S-90 | 1.2500 | 1.3258 | +6.06% |
| 487 | Kelpak SL | 1.2500 | 1.4773 | +18.18% |
| 488 | Bio-Algeen S-90 | 1.7803 | 1.5909 | −10.64% |
| 489 | Kelpak SL | 1.7803 | 1.7045 | −4.26% |

**Critical observation:** The three distinct control means (2.5379, 1.25, 1.7803 t/ha) represent three separate data series extracted from the paper. The standard deviations are also provided (ctrl_sd ≈ 0.606, 0.379, 0.227 t/ha respectively), indicating these are real measured variance values, not artefacts.

---

## 3. What Our AI Extractor Captured (JSON Observations)

The AI extractor produced **45 observations** covering:
- All 5 cultivars (Irga, Satina, Valfi, Blaue St. Galler, Highland Burgundy Red)
- All 3 years (2013, 2014, 2015)
- All 3 treatments per year-cultivar pair (Asahi SL, Bio-Algeen S-90, Kelpak SL)
- Each with a shared control mean for the same cultivar × year group

**Unit:** kg/ha (correct unit after conversion from dt/ha in Table 4; 1 dt = 100 kg)

**Example JSON observations (control means):**

| Cultivar | Year | Control mean (kg/ha) | Equivalent (t/ha) |
|---------|------|---------------------|------------------|
| Irga | 2013 | 40,041 | 40.04 |
| Satina | 2013 | 38,436 | 38.44 |
| Valfi | 2013 | 19,053 | 19.05 |
| Blaue St. Galler | 2013 | 8,561 | 8.56 |
| HB Red | 2013 | 19,099 | 19.10 |
| Irga | 2014 | 20,210 | 20.21 |
| Valfi | 2014 | 6,272 | 6.27 |
| HB Red | 2014 | 5,630 | 5.63 |
| Irga | 2015 | 21,506 | 21.51 |

These values match **Table 4** of the paper (potato total yield in dt ha⁻¹) with high fidelity. For example, Table 4 reports Irga 2013 control = 400.41 dt/ha = 40,041 kg/ha — exactly what the AI extracted.

**What the AI did NOT extract:**
- Variance (SD/SE) for any observation — all variance fields are null
- Trifender WP treatment (not included in JSON — partial coverage)
- Any breakdown by tuber size fraction

---

## 4. Why There Are 0 Matches: Root Cause Analysis

### 4.1 Scale Mismatch — The Primary Barrier

The fundamental reason for zero matches is a **scale discrepancy of approximately 16–40×** between the GT control means and the JSON control means:

| Source | Irga 2013 control | Valfi 2013 control |
|--------|-----------------|-----------------|
| Li 2022 GT | 2.5379 t/ha | — |
| AI extractor (JSON) | 40.041 t/ha | 19.053 t/ha |
| Paper Table 4 | 400.41 dt/ha (= 40.041 t/ha) | 190.53 dt/ha (= 19.053 t/ha) |

The AI extractor is correct: it faithfully reproduced Table 4 values. The GT values of ~1.25–2.54 t/ha are incompatible with any single cultivar × year combination shown in Table 4, whose control means range from 56 to 524 dt/ha (5.6–52.4 t/ha).

### 4.2 Identifying the GT Data Source

The three GT control means (2.5379, 1.25, 1.7803 t/ha) do not correspond to any row, column, or simple aggregation in Table 4 of the paper. Several hypotheses were evaluated:

**Hypothesis A: Per-size-fraction absolute yield (t/ha)**
Table 5 gives the percentage share of tuber fractions (>50 mm, 35–50 mm, <35 mm). Computing absolute yield per fraction for Irga 2013 control:
- Total yield = 400.41 dt/ha = 40.041 t/ha
- >50 mm: 32.7% × 40.041 = 13.09 t/ha
- 35–50 mm: 49.6% × 40.041 = 19.86 t/ha
- <35 mm: 17.7% × 40.041 = 7.09 t/ha

None of these match 2.5379 t/ha.

**Hypothesis B: Yield averaged across cultivars within a year**
Mean total yield across all 5 cultivars for 2013 control = (400.41 + 384.36 + 190.53 + 85.61 + 190.99) / 5 = 250.38 dt/ha = 25.04 t/ha. Does not match.

**Hypothesis C: Yield from a specific cultivar subset or subgroup**
No single cultivar × year × treatment combination in Table 4 yields a control mean near 1.25 or 1.78 or 2.54 t/ha, as all Table 4 values exceed 56 dt/ha (5.6 t/ha).

**Hypothesis D: The GT extracted data from a different table, figure, or publication**
This is the most plausible explanation. The GT values are in a range consistent with **per-plant yield (kg/plant)** or **plot yield (kg/plot)** rather than field-scale yield (t/ha). If each micro-plot held approximately 16 plants (common for randomised sub-block micro-plot designs at 40 cm spacing and ~0.25 m² plots), then 2.54 t/ha × plot_area would give per-plot values. Alternatively, Li 2022 may have extracted from a **supplementary table, figure data, or a related conference paper** that reported data for a subset of the experiment.

**Hypothesis E: Unit interpretation error in Li 2022**
The GT may have recorded the values in t/ha but the original extraction read them as dt/ha (decitonnes), then recorded without converting — e.g., reading "25.38 dt/ha" as "2.538 t/ha" through a decimal shift. Table 4 values near 25 dt/ha correspond to lower-yielding cultivar-year combinations (e.g., Blaue St. Galler 2015 = 125.31 dt/ha control; Valfi 2014 = 62.72 dt/ha). No Table 4 value is close to 25 dt/ha.

**Most likely root cause:** Li 2022 extracted data from the paper using a **different yield metric** than total tuber yield — most likely the **marketable (medium-sized, 35–50 mm) tuber yield expressed as an absolute mass per unit area** (computed from Table 4 × Table 5 combined), or from an undigitised figure. The three distinct control means and their associated SDs are too precise to be errors, but they are irreconcilable with any directly readable table in the PDF.

### 4.3 Treatment Scope Mismatch

Even if the scale issue were resolved, the AI extractor captured Asahi SL observations (30 of 45 JSON obs) which are not present in the Li 2022 GT for this paper. The GT includes only Bio-Algeen S-90 and Kelpak SL. This is expected — Asahi SL is a synthetic growth regulator, not a seaweed extract, and falls outside the Li 2022 meta-analysis scope. This is not an extraction error but a scoping difference.

### 4.4 Missing Variance in JSON

All 45 JSON observations lack variance values. Table 4 reports means only; variance is implicitly represented through the Duncan letter codes. No numeric SD, SE, or LSD values are printed in Table 4 or Table 5. This is a genuine data availability limitation in the paper, not an AI extraction error. The GT SDs (0.227–0.606 t/ha) presumably were obtained by Li 2022 through a method not directly apparent from the published tables — possibly calculation from the letter-coded ANOVA results, communication with authors, or from a separate data source.

---

## 5. Summary of Failure Taxonomy

| Issue | Classification | Correctable? |
|-------|---------------|-------------|
| GT ctrl means (1.25–2.54 t/ha) do not match Table 4 values (5.6–52.4 t/ha) | **Coverage failure / different data source** | No — data source unknown |
| AI extracted total yield (Table 4) rather than GT's yield metric | **Wrong metric extracted** | Only if GT source is identified |
| Asahi SL observations present in JSON but absent from GT | **Scope difference** (correct AI behaviour) | N/A |
| No variance values in JSON | **True data limitation** (paper uses letter codes) | No — paper does not report numeric SDs |

---

## 6. Verdict

**Could better extraction fix this?**

**No — not without identifying the exact yield metric that Li 2022 used.**

The AI extractor performed correctly: it read Table 4 (total potato tuber yield, dt ha⁻¹) with high accuracy, correctly converted units to kg/ha, correctly identified all 5 cultivars, 3 years, and 3 seaweed treatments. The 45 JSON observations are internally consistent with the paper's Table 4 and are scientifically valid extractions.

The 0-match outcome is driven by the GT using a yield metric or data source that produces control means in the 1.25–2.54 t/ha range, which are **16–40 times smaller** than any directly published total yield value in the paper. The most plausible explanation is that Li 2022 computed an absolute per-fraction yield (marketable fraction only) by multiplying Table 4 totals by the Table 5 fraction percentages, then averaged across cultivars or years in a way that produces the observed GT values — but this calculation pathway could not be confirmed from the published tables alone.

**Recommended action:** Manual inspection by a human reviewer to determine which specific combination of Table 4 × Table 5 × cultivar/year averaging produces the three GT control means (2.5379, 1.25, 1.7803 t/ha), or to identify whether Li 2022 used data from a source external to this PDF (e.g., a supplementary file, the authors' dataset, or a related publication). Until this source is identified, this paper cannot be matched automatically and should be flagged as requiring expert adjudication in the validation pipeline.

---

## 7. Appendix: Key Table Values for Cross-Reference

**Table 4 — Total potato yield (dt ha⁻¹) for control plots:**

| Year | Irga | Satina | Valfi | Blaue | HB Red |
|------|------|--------|-------|-------|--------|
| 2013 | 400.41 | 384.36 | 190.53 | 85.61 | 190.99 |
| 2014 | 202.10 | 209.63 | 62.72 | 82.60 | 56.30 |
| 2015 | 215.06 | 246.17 | 126.67 | 125.31 | 161.11 |

*Note: 1 dt = 100 kg; values in t/ha = values above ÷ 10.*

**Li 2022 GT control means vs. closest Table 4 values:**

| GT ctrl (t/ha) | Closest Table 4 value (t/ha) | Cultivar / Year | Ratio |
|----------------|------------------------------|-----------------|-------|
| 2.5379 | 5.630 (HB Red 2014) | — | 0.45× |
| 1.2500 | 5.630 (HB Red 2014) | — | 0.22× |
| 1.7803 | 5.630 (HB Red 2014) | — | 0.32× |

No Table 4 value is within 2× of any GT control mean. The mismatch is not explainable by a simple unit conversion factor or rounding.
