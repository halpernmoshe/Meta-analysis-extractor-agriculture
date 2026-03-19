# Extraction Quality Report: 095_Kuisma_1989

**Paper:** Kuisma, P. (1989). The effect of foliar application of seaweed extract on potato. *Maataloustieteellinen Aikakauskirja (Journal of Agricultural Science in Finland)*, Vol. 61: 371–377.

**Outcome:** 0 matched pairs | 3 unmatched GT rows | 4 unmatched AI observations

---

## 1. Paper Design

- **Crop:** Potato (*Solanum tuberosum* cv. Record)
- **Treatment:** Foliar application of commercial seaweed extract SM3 (Chase Organics, GB)
- **Location:** Viikki Experimental Farm, Helsinki, Finland (60°10'N 25°00'E)
- **Year:** 1979
- **Design:** Split-plot with 3 replications. Spraying date was the main plot factor (3 levels); dose was the subplot factor (4 levels).
- **Plot size:** Two rows, 70 cm apart × 10 m long (approximately 14 m² per plot)
- **Primary outcome:** Ware potato yield (tubers >35 mm), reported in t/ha

---

## 2. Doses and Table Structure in the PDF

### 2a. Actual doses tested (from Methods and all tables)

The paper explicitly states four dose levels for SM3 seaweed extract:

| Dose (l/ha) | Dilution ratio | Description |
|-------------|---------------|-------------|
| 0 | 0:100 | Water control (untreated) |
| 5 | 1:100 | Lowest seaweed dose |
| 10 | 1:50 | Mid dose |
| 20 | 1:25 | High dose (near double recommended 11 l/ha) |

All doses were applied as foliar spray at a **total spray volume of 500 l/ha**. There are no doses of 0.25, 0.5, or 1.0 l/ha anywhere in the paper.

### 2b. Table 5 — Ware yield (t/ha): the primary outcome table

| Doses l/ha | July 10 | July 31 | August 15 | Mean |
|------------|---------|---------|-----------|------|
| 5 | 39.60 | 36.60 | 39.00 | 38.40 |
| 10 | 39.33 | 34.90 | 37.57 | 37.27 |
| 20 | 39.21 | 39.30 | 39.36 | 39.29 |
| Mean | 39.38 | 36.93 | 38.64 | 38.32 |
| **Untreated** | — | — | — | **35.63** |

The untreated control mean is 35.63 t/ha (a single row mean across all spraying dates, since the control receives no seaweed treatment). This is the value the AI correctly extracted.

### 2c. Other tables

- **Table 4** — Tuber size (g): Untreated = 107.2 g; doses 5, 10, 20 l/ha.
- **Table 6** — DM content (%): Untreated = 21.0%; doses 5, 10, 20 l/ha.
- **Table 3** — Haulm senescence score: same dose structure.

In all tables, the dose levels are consistently 5, 10, 20 l/ha. The dose axis in Figure 2 (bar chart) also shows "5 10 20 l/ha Dose" on the x-axis.

---

## 3. What the AI Extracted

The AI produced 4 observations, all attributed to the July 10 application date (24 days after emergence):

| # | Outcome | Control mean | Treatment mean | Dose in description | Effect |
|---|---------|-------------|----------------|---------------------|--------|
| 1 | Ware yield t/ha | 35.63 | 39.60 | "5 l/ha" | +11.1% |
| 2 | Tuber size g | 107.2 | 96.9 | "5 l/ha" | -9.6% |
| 3 | DM % | 21.0 | 20.4 | "5 l/ha" | -2.9% |
| 4 | Tuber yield t/ha | 36.5 | 40.5 | "5 l/ha" | +10.9% |

**Assessment of AI extraction accuracy against the PDF:**

- The AI correctly read the untreated control mean (35.63 t/ha) from Table 5.
- The AI correctly extracted the July 10 treatment mean for 5 l/ha (39.60 t/ha) from Table 5.
- The dose label "5 l/ha" is factually correct — it is the smallest active dose in the paper.
- The AI also extracted tuber size (107.2 → 96.9 g) and DM% (21.0 → 20.4%) from Tables 4 and 6, both accurate.
- Observations 1 and 4 appear to represent the same outcome (ware yield) from two slightly different framings (Table 5 vs. Figure 2 bar chart estimate). The small discrepancy (35.63 vs. 36.5 for control) suggests the AI read from two sources.

**The AI extraction is internally consistent and factually correct for a single dose arm (5 l/ha, July 10 application).** The AI did not extract the full factorial structure (3 application dates × 3 doses = 9 data points per outcome variable).

---

## 4. Why the GT Dose Labels Do Not Match the PDF

### 4a. GT doses: 0.25, 0.5, 1.0 l/ha

The ground truth (Li 2022 database, pairs 683–685) records three dose arms:

| GT pair | GT dose | GT ctrl_mean | GT treat_mean | Effect |
|---------|---------|-------------|---------------|--------|
| 683 | 0.25 l/ha | 3.563 | 3.840 | +7.8% |
| 684 | 0.50 l/ha | 3.563 | 3.727 | +4.6% |
| 685 | 1.00 l/ha | 3.563 | 3.832 | +7.5% |

**None of these doses (0.25, 0.5, 1.0 l/ha) appear anywhere in the Kuisma 1989 paper.** The paper's dose series is 0, 5, 10, 20 l/ha.

### 4b. Unit discrepancy: GT ctrl_mean = 3.563 vs PDF = 35.63 t/ha

The GT control mean of 3.563 is exactly 1/10 of the PDF value of 35.63 t/ha. Two hypotheses:

1. **Unit conversion error in GT data entry:** The PDF reports ware yield in t/ha (Table 5: Untreated = 35.63 t/ha). If the GT entry divided by 10 to convert to some other unit (e.g., t/1000 m² or kg/plot), this would produce 3.563. This appears the most likely explanation.

2. **Per-plot units:** Each plot was two rows × 10 m = approximately 14 m². If yields were recorded per plot and converted to t/ha via a different plot area calculation, the factor could shift. However, 35.63 / 10 = 3.563 is a suspiciously clean factor-of-10, consistent with a unit error rather than a plot-area calculation.

### 4c. Dose mapping hypothesis

The GT doses of 0.25, 0.5, and 1.0 l/ha could represent the SM3 extract **concentrate volume** when the dilution ratios in the paper are converted differently:

- At 500 l/ha spray volume with 1:100 dilution → 500/100 = 5 l/ha of SM3 concentrate
- If someone divided by 20 (perhaps assuming 100 l/ha spray volume): 5/20 = 0.25 l/ha
- Similarly: 10/20 = 0.5 l/ha; 20/20 = 1.0 l/ha

This arithmetic matches exactly: the GT doses of 0.25, 0.5, and 1.0 correspond to the PDF doses of 5, 10, and 20 l/ha divided by 20. This suggests the Li 2022 database **recalculated doses using a different assumed spray volume** (perhaps 100 l/ha instead of the actual 500 l/ha stated in the Methods), resulting in a 5-fold dose rescaling.

---

## 5. Root Cause Analysis

### Primary cause: Ground truth dose rescaling (not an AI extraction error)

The Li 2022 ground truth database recorded doses of 0.25, 0.5, 1.0 l/ha for this paper. These values do not appear in the Kuisma 1989 paper. The paper explicitly states doses of 5, 10, and 20 l/ha of SM3, diluted in 500 l water/ha. The GT doses are exactly 1/20 of the PDF doses, consistent with a recalculation using a different assumed spray volume (100 l/ha instead of the stated 500 l/ha).

The GT control means (3.563) are also exactly 1/10 of the PDF values (35.63 t/ha), pointing to a separate unit error or a different yield unit (possibly reported per 100 m² plot rather than per hectare).

### Secondary cause: AI extracted only one dose arm from a factorial design

The AI extracted data for the 5 l/ha dose applied on July 10 only, rather than generating separate observations for all three seaweed doses (5, 10, 20 l/ha) at all three application dates. A complete extraction of Table 5 would yield 9 treatment observations (3 doses × 3 dates) plus the untreated control. The AI collapsed the dose-response structure into a single "best" observation.

Even if the AI had extracted all three dose arms, the doses (5, 10, 20 l/ha) would still not match the GT dose labels (0.25, 0.5, 1.0 l/ha) because of the dose rescaling described above.

### Tertiary cause: Effect size mismatch

The GT reports effects of +7.8%, +4.6%, +7.5% for the three dose arms. The AI extracted an effect of +11.1% for the 5 l/ha dose on July 10. This discrepancy is due to the AI extracting from the July 10 application only, whereas the GT appears to use means averaged across all three application dates (Table 5 row means: 38.40/35.63 = +7.8% for 5 l/ha — exactly matching GT pair 683).

---

## 6. Summary

| Issue | Description |
|-------|-------------|
| GT dose labels | 0.25, 0.5, 1.0 l/ha — not present in PDF; appear to be PDF doses (5, 10, 20 l/ha) ÷ 20 |
| GT ctrl_mean | 3.563 = PDF value of 35.63 t/ha ÷ 10; likely unit error in GT |
| AI dose label | "5 l/ha" — factually correct per PDF, but 20x the GT dose label |
| AI yield values | Correct per PDF (35.63 ctrl, 39.60 treat); 10x the GT values |
| AI coverage | Only 1 of 3 dose arms extracted; collapsed across application date |
| Matching outcome | Complete coverage failure: 0/3 GT pairs matched |
| Responsible party | Primarily GT data entry (dose rescaling + unit error); secondarily AI (partial extraction) |

**Conclusion:** The zero-match outcome for this paper is primarily attributable to errors or unconventional rescaling in the Li 2022 ground truth database rather than to AI extraction failure. The AI correctly read the data as presented in the paper (doses in l/ha, yield in t/ha). To match the GT, both the dose values and yield units would need to be rescaled by factors of 20 and 10 respectively — transformations that are not documented in the paper or traceable to the paper's methods. The AI should also be improved to extract all factorial dose arms rather than a single arm, but that improvement would not resolve the dose-label mismatch with the GT.
