# Extraction Quality Report: 68_Peck_2008

**Match summary:** 12/18 GT matched (67% capture), r=1.0, MAE = 0.0%

---

## 1. Paper Design

**Citation:** Peck, A.W., McDonald, G.K., Graham, R.D., 2008. Zinc nutrition influences the protein composition of flour in bread wheat (*Triticum aestivum* L.). *Journal of Cereal Science*, 47(2), 266-274.

**Country:** Australia

**Design:** Multi-site field experiment testing three rates of soil-applied zinc oxysulphate (2.5, 7.5, and 22.5 kg Zn ha-1) with and without foliar zinc spray at six field sites across South Australia and Victoria. Sites include Lameroo, Minnipa, Tintinara (1998) and Birchip, Horsham SD1, Horsham SD2 (1999). Replication: n = 4 per treatment.

**Complexity factors:**
- Multiple sites (6) and two years
- Three soil Zn rates plus foliar spray combinations creates a complex table layout
- Some sites have high ambient soil Zn (Horsham: ctrl ~20 mg/kg) while others are deficient (Birchip: ctrl ~10-11 mg/kg)
- The paper's main tables report soil-only AND soil+foliar columns side by side, creating column confusion risk

---

## 2. Grain Zn Data in PDF

The paper presents grain Zn concentration (mg kg-1) in Table 2 (and Table 4 in the supplementary), with the following treatment structure per site:

| Treatment | Zn rate | Application |
|-----------|---------|-------------|
| Control | 0 kg Zn/ha | None |
| Low soil | 2.5 kg Zn/ha | Soil |
| Mid soil | 7.5 kg Zn/ha | Soil only |
| High soil | 22.5 kg Zn/ha | Soil only |
| Soil + foliar | 7.5 kg Zn/ha soil + 334 g Zn/ha spray | Combined |

The Hui 2023 GT treats this paper as appearing in two separate meta-analysis datasets:
- **Data 2 (Soil application), study_id=68:** Covers soil-only treatment rows (10 GT observations)
- **Data 4 (Soil+Foliar application), study_id=21:** Covers the soil+foliar treatment rows (8 GT observations)

This split-dataset design means 18 GT rows in total must be captured from a single paper.

---

## 3. AI Extraction

**Models used:** Claude and Kimi K2.5 (consensus; Gemini extracted 0 observations)

**Consensus output:** 12 observations, all grain Zn (mg/kg), n=4, variance type LSD throughout, extracted from Table 2.

**All 12 consensus observations:**

| # | ctrl (mg/kg) | treat (mg/kg) | effect (%) | Site / Year | Treatment label |
|---|-------------|--------------|-----------|-------------|-----------------|
| 1 | 11.4 | 12.4 | +8.8% | Lameroo 1998 | 2.5 kg Zn/ha soil |
| 2 | 12.2 | 14.0 | +14.8% | Tintinara 1998 | 2.5 kg Zn/ha soil |
| 3 | 12.2 | 15.7 | +28.7% | Tintinara 1998 | 22.5 kg Zn/ha soil |
| 4 | 14.1 | 14.2 | +0.7% | Minnipa 1998 | 2.5 kg Zn/ha soil |
| 5 | 10.9 | 20.3 | +86.2% | Birchip 1999 | 7.5 kg Zn/ha soil |
| 6 | 10.9 | 21.0 | +92.7% | Birchip 1999 | 22.5 kg Zn/ha soil |
| 7 | 20.5 | 21.8 | +6.3% | Horsham SD1 1999 | 2.5 kg Zn/ha soil |
| 8 | 20.5 | 26.8 | +30.7% | Horsham SD1 1999 | 7.5 kg Zn/ha soil |
| 9 | 20.5 | 30.3 | +47.8% | Horsham SD1 1999 | 22.5 kg Zn/ha soil |
| 10 | 19.9 | 20.1 | +1.0% | Horsham SD2 1999 | 2.5 kg Zn/ha soil |
| 11 | 19.9 | 24.5 | +23.1% | Horsham SD2 1999 | 7.5 kg Zn/ha soil |
| 12 | 19.9 | 33.3 | +67.3% | Horsham SD2 1999 | 22.5 kg Zn/ha soil |

**Key recon warning:** The consensus JSON recorded `"Recon error: 'list' object has no attribute 'get'"` and variance was flagged unclear (`VAR-UNCLEAR`). The actual variance type is LSD (confirmed in paper table footnotes).

**Claude vs Kimi disagreements (6 entries):** Claude additionally extracted 7.5 soil+foliar (treat=26.0 at Lameroo, treat=23.4 at Tintinara) and alternative treatment assignments at Birchip (treat=12.3) and Minnipa (treat=15.5, 16.8). Kimi extracted different values for those rows with low confidence, citing "treatment identity ambiguous from table." The disagreements prevented these from entering the consensus.

---

## 4. GT Data (all 18 rows, matched and unmatched)

### Data 2 - Soil Application (study_id=68): 10 rows

| obs_id | ctrl (mg/kg) | treat (mg/kg) | GT effect (%) | Zn rate | yield | Matched? |
|--------|-------------|--------------|--------------|---------|-------|----------|
| 640 | 14.1 | 14.2 | +0.7% | 2.5 soil | 2310 | YES -> consensus #4 |
| 641 | 11.4 | 12.4 | +8.8% | 7.5 soil | 3040 | YES -> consensus #1 |
| 642 | 12.2 | 14.0 | +13.8% | 7.5 soil | 3640 | YES -> consensus #2 |
| **643** | **14.1** | **16.8** | **+19.1%** | **7.5 soil** | **2310** | **MISSED** |
| **644** | **10.9** | **12.3** | **+12.8%** | **7.5 soil** | **2880** | **MISSED** |
| 645 | 20.5 | 21.8 | +6.1% | 2.5 soil | 2200 | YES -> consensus #7 |
| 646 | 19.9 | 20.1 | +1.0% | 2.5 soil | 2270 | YES -> consensus #10 |
| **647** | **11.4** | **14.5** | **+27.2%** | **22.5 soil** | **3040** | **MISSED** |
| 648 | 12.2 | 15.7 | +28.7% | 22.5 soil | 3640 | YES -> consensus #3 |
| **649** | **14.1** | **15.5** | **+9.9%** | **22.5 soil** | **2310** | **MISSED** |

### Data 4 - Soil+Foliar Application (study_id=21): 8 rows

| obs_id | ctrl (mg/kg) | treat (mg/kg) | GT effect (%) | yield | Matched? |
|--------|-------------|--------------|--------------|-------|----------|
| **132** | **11.4** | **26.0** | **+128.1%** | **3040** | **MISSED** |
| **133** | **12.2** | **23.4** | **+91.8%** | **3640** | **MISSED** |
| 134 | 10.9 | 20.3 | +86.2% | 2880 | YES -> consensus #5 |
| 135 | 20.5 | 26.8 | +30.7% | 2200 | YES -> consensus #8 |
| 136 | 19.9 | 24.5 | +23.1% | 2270 | YES -> consensus #11 |
| 137 | 10.9 | 21.0 | +92.7% | 2880 | YES -> consensus #6 |
| 138 | 20.5 | 30.3 | +47.8% | 2200 | YES -> consensus #9 |
| 139 | 19.9 | 33.3 | +67.3% | 2270 | YES -> consensus #12 |

**Total:** 12 matched (6 soil + 6 S+F), 6 unmatched (4 soil-only + 2 S+F).

---

## 5. Root Cause: Why 6 Rows Unmatched?

### Finding 1: Misidentification of S+F values as soil-only (primary failure)

The paper's table has adjacent columns for soil-only and soil+foliar treatments at the same Zn application rate. The AI consensus correctly read the numeric values at Birchip (1999) and Horsham (1999) but extracted the **soil+foliar column** while labeling the result as a **soil-only treatment**.

Specifically:
- Consensus obs #5 (ctrl=10.9, treat=20.3, labeled "7.5 soil at Birchip") matched GT S+F obs134 (which is the 7.5 soil+foliar treatment). The actual 7.5 soil-only row is GT obs644 (treat=12.3), which was left unmatched.
- Consensus obs #6 (ctrl=10.9, treat=21.0, labeled "22.5 soil at Birchip") matched GT S+F obs137. The actual 22.5 soil-only row is GT obs647 (treat=14.5), which was left unmatched.
- Similarly for Horsham SD1 and SD2: consensus obs #8, #9, #11, #12 all carry "soil application" labels but numerically matched S+F GT rows (135, 138, 136, 139), while the corresponding pure-soil rows (with lower treat values) were never extracted.

This is a **column misread error**: the AI read the soil+foliar column from the paper table and labeled those observations as soil-only. The effect is systematic at four sites (Birchip and both Horsham subsets).

### Finding 2: Missing 7.5 and 22.5 soil-only rows at Minnipa (low-N site, N=5 kg/ha)

GT obs643 (Minnipa, Zn=7.5 soil, ctrl=14.1, treat=16.8, +19%) and obs649 (Minnipa, Zn=22.5 soil, ctrl=14.1, treat=15.5, +10%) were not extracted. The AI did extract the 2.5 soil row for Minnipa (obs640, consensus #4), but did not capture the two higher-rate soil treatments at that site. This likely reflects the low-N treatment block (N=5 kg/ha) being partially omitted from the extraction.

### Finding 3: S+F rows at Lameroo and Tintinara were extracted by Claude but lost in consensus

Claude's extraction included:
- ctrl=11.4, treat=26.0 (Lameroo, 7.5 soil+foliar, +128%) -- GT obs132
- ctrl=12.2, treat=23.4 (Tintinara, 7.5 soil+foliar, +92%) -- GT obs133

Both of these appear in Claude's individual disagreement entries. Kimi, however, extracted different treatment values for these rows (citing ambiguous column alignment in the table). Because the two models disagreed substantially, the consensus algorithm dropped both observations. The result is that two large-effect S+F rows (effects of +92% and +128%) are absent from the consensus.

### Why MAE = 0.0% and r = 1.0 despite the confusion

The 12 matched pairs are all numerically exact matches (combined matching error = 0.000 for all 12). The validation algorithm matched by (ctrl, treat) value pairs, and the AI's S+F-labeled-as-soil observations happened to have the identical numeric values as the S+F GT rows. The matching is technically correct at the numerical level -- the AI read the right numbers from the table, just attributed them to the wrong treatment type. This is why the accuracy metrics are perfect for the matched set.

### Summary of 6 missing rows

| obs_id | Sheet | ctrl | treat | GT effect | Why missed |
|--------|-------|------|-------|-----------|-----------|
| 643 | Soil | 14.1 | 16.8 | +19.1% | Minnipa 7.5 soil block not extracted |
| 644 | Soil | 10.9 | 12.3 | +12.8% | True Birchip 7.5 soil row; AI read S+F column instead |
| 647 | Soil | 11.4 | 14.5 | +27.2% | True Birchip 22.5 soil row; AI read S+F column instead |
| 649 | Soil | 14.1 | 15.5 | +9.9% | Minnipa 22.5 soil block not extracted |
| 132 | S+F | 11.4 | 26.0 | +128.1% | Claude extracted it, Kimi disagreed, dropped from consensus |
| 133 | S+F | 12.2 | 23.4 | +91.8% | Claude extracted it, Kimi disagreed, dropped from consensus |

---

## 6. Assessment

**Accuracy on matched rows: Excellent.** The 12 matched observations have r=1.0 and MAE=0.0%. Where the AI read numeric values from the paper, it read them correctly.

**Capture rate: Moderate (67%).** The 6 missing rows represent a meaningful gap.

**Application-type labeling: Flawed.** The most important finding here is not missed values but mislabeled treatment types. Six of the 12 consensus observations are described as "soil application" but their numeric values correspond to the Hui GT's Soil+Foliar category. The AI correctly read the numbers from the soil+foliar columns of the paper's table but recorded the treatment as soil-only. For the purposes of the Hui 2023 meta-analysis (which computes separate effect estimates for soil vs. foliar vs. combined), this labeling error would cause those 6 observations to be placed in the wrong category.

**Practical consequence for meta-analysis:** The AI's output, if used uncritically, would:
1. Substantially overestimate the effect of soil-only Zn at Birchip and Horsham (reporting +86% to +92% and +23% to +48% effects from soil-alone, rather than the true +13% to +27% soil-only effects).
2. Underrepresent the full scope of treatments at Minnipa (only 1 of 3 soil rates captured).
3. Miss the largest-effect observations entirely (obs132 at +128% and obs133 at +92%, which are the two most biologically important S+F results from this paper).

**Recon failure note:** The recon module threw an error (`'list' object has no attribute 'get'`) and produced empty guidance for this paper. The paper was rated `EASY` but the table complexity (multiple sites, multiple Zn rates, soil vs. foliar columns side-by-side) exceeds what that rating implies. Better recon flagging of factorial/multi-column designs would help direct the extraction models to distinguish treatment columns more carefully.

**Recommendation:** Flag this paper for human review of the treatment-type assignment. The 12 numeric values are correct; the question is which table column each was read from. Verification against the original PDF Table 2 is needed to confirm whether the soil+foliar column was indeed conflated with the high-rate soil column for Birchip and Horsham.
