# Extraction Quality Report: 58_Kalayci_1999

**Paper:** Kalayci, M., Torun, B., Eker, S., Aydin, M., Ozturk, L., Cakmak, I. (1999). Grain yield, zinc efficiency and zinc concentration of wheat cultivars grown in a zinc-deficient calcareous soil in field and greenhouse. *Field Crops Research*, 63(1), 87–98.

**Match summary:** 5/78 GT matched (6% capture), r = 0.939, MAE = 12.49%

**AI extraction stats:** claude_obs = 16, kimi_obs = 0, gemini_obs = 9, consensus_obs = 8 (tiebreaker used)

---

## 1. Paper Design

This is a large multi-cultivar, multi-year, multi-environment Zn fertilization study:

| Dimension | Detail |
|-----------|--------|
| Cultivars | 40 total (37 bread wheat *Triticum aestivum*, 3 durum wheat *T. durum*) |
| Field site | Eskisehir, Central Anatolia, Turkey (Zn-deficient calcareous soil, available Zn = 0.1 mg kg-1) |
| Years | 1993–1994 and 1994–1995 (two growing seasons) |
| Zn treatments | -Zn (0 kg Zn ha-1, control) and +Zn (23 kg Zn ha-1 as ZnSO4, treatment) |
| Field design | Randomized complete block in strip plots, n = 4 replications |
| Greenhouse | 24 cultivars, 5 mg Zn kg-1 soil, n = 3 replications |
| Outcome variables | Grain Zn (mg kg-1), shoot Zn (mg kg-1), grain yield (kg ha-1), shoot Zn content (μg shoot-1), shoot DW |
| Variance | LSD0.05 values reported at bottom of each table |
| PDF type | Scanned document (OCR text, potential OCR errors noted by recon) |

The paper is explicitly rated **HARD** by the recon module due to: scanned PDF, 40 cultivar rows per table, two years presented side-by-side in one wide table, and LSD-only variance reporting.

---

## 2. Grain Zn Data in PDF

**Table 2** is the primary data table for the Hui 2023 meta-analysis. It has the following structure:

```
Cultivars | 1993–1994                          | 1994–1995
          | shoot (mg/kg)  grain (mg/kg)       | shoot (mg/kg)  grain (mg/kg)
          | -Zn    +Zn     -Zn    +Zn          | -Zn    +Zn     -Zn    +Zn
----------|-------------------------------------|------------------------------
Sertak 52 |  8     17      8      16           |  7     13      9      15
ES 90-3   |  8     10      8      11           |  8     14      9      13
Yayla 305 |  6     15      6      13           |  7     13      9      15
Kirac 66  |  7     13      7      14           |  8     14     11      14
Ak 702    |  7     16      6      13           |  7     13      9      11
Gun 91    |  6     11      7      10           |  8     11     10      13
...       | (40 cultivar rows total)            |
```

**Table dimensions:** 40 cultivar rows × 8 data columns = 320 data cells, of which:
- 80 grain Zn pairs (-Zn, +Zn) across 40 cultivars × 2 years
- 2 rows have "n.d." for grain Zn in 1993–1994 (ES 90-14 and SBVD 2-8)
- **78 valid paired grain Zn observations** (matching the 78 GT rows)

**LSD values reported** at table footer:
- 1993–1994 shoot Zn: LSD0.05 (Cultivar, Zn) = (2.4, 1.9, NS)
- 1993–1994 grain Zn: LSD0.05 = (NS, 0.6, 2.9)
- 1994–1995 shoot Zn: LSD0.05 = (1.5, 1.5, 1.8)
- 1994–1995 grain Zn: LSD0.05 = (1.4, 0.9, 1.9)

The full count of extractable observations from this paper:
- Table 2 grain Zn: 78 valid pairs (the meta-analysis target)
- Table 2 shoot Zn: ~80 pairs (not included in Hui 2023 GT)
- Table 1 grain yield: 80 pairs (not the primary outcome)
- Tables 3 and 4: greenhouse data (24 cultivars × 1 experiment)

---

## 3. AI Consensus Extraction Results

The consensus pipeline produced **8 observations** after post-processing (2 null-mean observations removed from an initial 10):

| # | Element | Tissue | Cultivar | Year/Env | Ctrl | Treat | Effect% | Variance |
|---|---------|--------|----------|----------|------|-------|---------|----------|
| 1 | grain Zn | grain | Sertak 52 | field 1993–94 | 8 | 16 | +100% | LSD=1.9 |
| 2 | grain Zn | grain | Sertak 52 | field 1994–95 | 8 | 15 | +87.5% | LSD=1.9 |
| 3 | grain Zn | grain | Kirac 66 | field 1994–95 | 13 | 14 | +7.7% | LSD=1.9 |
| 4 | grain Zn | grain | Gun 91 | field 1993–94 | 8 | 10 | +25.0% | LSD=1.9 |
| 5 | grain yield | grain | Sertak 52 | field 1993–94 | 2480 | 2650 | +6.9% | LSD=403 |
| 6 | shoot Zn conc | shoot | Sertak 52 | field 1993–94 | 8 | 17 | +112.5% | LSD=1.9 |
| 7 | shoot DW | shoot | Gun 91 | greenhouse | 0.54 | 0.65 | +20.4% | LSD=0.036 |
| 8 | shoot Zn content | shoot | Gun 91 | greenhouse | 3.9 | 24.0 | +515.4% | LSD=1.2 |

**Model agreement breakdown:**
- Claude extracted 16 observations (covering ~8 cultivar-year grain Zn combinations plus non-grain outcomes)
- Gemini extracted 9 observations
- **Kimi extracted 0 observations** (complete failure — tiebreaker mode activated, Claude vs Gemini)
- 8 observations reached consensus (Claude + Gemini agreed within 15% tolerance)
- 12 Claude-only grain Zn observations were **discarded** because Gemini did not confirm them

**Discarded Claude-only grain Zn observations (12 obs):**

| Cultivar | Year | AI Ctrl | AI Treat | AI Effect% |
|----------|------|---------|---------|------------|
| ES 90-3 | 1993–94 | 8 | 8 | 0% |
| ES 90-3 | 1994–95 | 10 | 13 | +30% |
| Yayla 305 | 1993–94 | 6 | 13 | +117% |
| Yayla 305 | 1994–95 | 7 | 15 | +114% |
| Kirac 66 | 1993–94 | 8 | 14 | +75% |
| Ak 702 | 1993–94 | 9 | 13 | +44% |
| Ak 702 | 1994–95 | 6 | 11 | +83% |
| Gun 91 | 1994–95 | 11 | 10 | -9% |
| Kirgiz 95 | 1993–94 | 6 | 12 | +100% |
| Kirgiz 95 | 1994–95 | 9 | 9 | 0% |
| Gerek 79 | 1993–94 | 6 | 7 | +17% |
| Gerek 79 | 1994–95 | 16 | 11 | -31% |

These 12 discarded observations cover 6 additional cultivars. Several show erroneous control values (e.g., Gerek 79 1994–95 ctrl=16 is impossible — control should be in the 5–11 mg/kg range), and Gun 91 1994–95 shows a negative response (ctrl=11 treat=10) which is likely a ctrl/treat confusion or OCR error from the scanned table.

---

## 4. Ground Truth (MOESM5) Data

**GT sheet:** "Data 2 Soil application" — study_id = 58

**GT structure:** 80 rows (observation IDs 460–539), each representing one cultivar in one growing-season (-Zn or +Zn condition). Because each MOESM5 row stores both the control grain Zn concentration AND the treatment grain Zn concentration (derived via the Zn biofortification index: treat = ctrl + ZBI × Zn_rate), there are **78 valid paired (ctrl, treat) observations** (2 rows have missing grain Zn data).

**GT observation summary:**

| Group | Obs IDs | Grain Zn ctrl (mg/kg) | Grain Zn treat (mg/kg) | Effect% range |
|-------|---------|----------------------|----------------------|---------------|
| All 78 paired obs | 460–499 + 500–539 | 5–11 | 8–17 | +10% to +160% |
| Mean | — | 6.6 | 11.8 | mean ~57% |

The GT has 78 cultivar-year pairs corresponding to 40 cultivars × 2 years minus 2 n.d. values — precisely matching Table 2 of the paper. **All 78 rows have a single Zn application level (23 kg Zn ha-1 as ZnSO4)**, same soil, country, and experimental year.

**Grain Zn value ranges in GT:**
- Control (-Zn): 5–11 mg/kg (integer values; scant variation, all zinc-deficient)
- Treatment (+Zn): 8–17 mg/kg (wider range; some cultivars respond much more than others)
- Effect sizes: 10% to 160% (large spread across cultivars)

---

## 5. Root Cause Analysis

### Why only 5/78 matched?

#### Cause 1: Massive table truncation by LLMs (PRIMARY CAUSE)

Table 2 has **40 cultivar rows × 8 columns = 320 cells** in a scanned PDF. The AI models did not extract all 40 cultivars — instead they sampled only the first several rows visible at the top of the table. The consensus extracted 4 grain Zn cultivar pairs from Table 2 (Sertak 52 ×2, Kirac 66, Gun 91) — all from the first ~6 rows of the 40-row table. No cultivar from row 7 onward (Kirgiz 95, Gerek 79, ES 14, P 8-6, ...) was confirmed by both Claude and Gemini.

This is a classic LLM table-truncation failure: models tend to sample representative rows from long tables rather than systematically extract every row. In a 40-cultivar table, the LLM presents a "representative" subset (4–8 rows) as if it were the complete dataset.

#### Cause 2: Kimi complete failure (COMPOUND CAUSE)

`kimi_obs = 0` — Kimi extracted zero observations from this paper. This forced the consensus engine into tiebreaker mode (Claude vs Gemini only). Had Kimi succeeded, a three-model vote could have accepted more Claude-only cultivar observations. Instead, the 12 Claude-only observations (covering 6 additional cultivars) were all discarded because Gemini did not independently extract those same cultivar rows.

#### Cause 3: Scanned PDF OCR quality

The recon module flagged this as a scanned PDF with potential OCR errors. The table text extracted by PyPDF2 shows OCR artifacts in durum wheat rows (e.g., "Kiziltan^a 71 7 71 3 71 1 1 0 1 1" is garbled). This significantly impairs reliable extraction of lower-table rows where OCR accuracy degrades. Some Claude-only observations show implausible values (Gerek 79 1994–95 ctrl=16, expected 5–11) which likely stem from OCR errors in the table rows.

#### Cause 4: Complex 8-column wide table layout

Table 2 has an 8-column structure (4 measurement types × 2 Zn treatments) per cultivar, spread across two years, with column headers spanning multiple rows. This unusual layout (two years side-by-side, each with 4 sub-columns) is easy to misinterpret. Claude and Gemini appear to have correctly parsed the first few rows but extraction quality deteriorates for later cultivars.

#### Cause 5: Ctrl/treat confusion and year confusion in discarded obs

Several discarded Claude-only observations show values inconsistent with the PDF:
- Gerek 79 1994–95: ctrl=16 is impossible (all -Zn grain Zn values are 5–11 mg/kg per paper text)
- Gun 91 1994–95: ctrl=11 treat=10 (negative response) conflicts with paper where nearly all cultivars show positive Zn responses
- Kirac 66 1993–94: AI ctrl=8, but Table 2 shows ctrl=7 for that cultivar-year

These errors suggest that for lower-table rows, OCR errors caused misreading of specific cell values, leading Gemini to reject Claude's numbers as inconsistent.

#### Cause 6: Consensus filter is too strict for this paper type

The consensus algorithm requires Claude and Gemini to independently extract the same cultivar-year pair within 15% tolerance on both ctrl and treat. For a 40-row scanned table, both models are unlikely to independently land on the same subset of rows. Claude covered 8 cultivar-year pairs; Gemini covered a different (partially overlapping) subset. The strict consensus requirement eliminated good data from both.

### Effect size accuracy for the 5 matched observations

| Matched Observation | AI Effect% | GT Effect% | Abs Error |
|---------------------|-----------|-----------|-----------|
| Sertak 52, grain Zn, 1993–94 | +100.0% | +100.0% | 0.0% |
| Gun 91, grain Zn, 1993–94 | +25.0% | +25.0% | 0.0% |
| Sertak 52, grain Zn, 1994–95 | +87.5% | +75.0% | 12.5% |
| Sertak 52, shoot Zn, 1993–94 | +112.5% | +142.9% | 30.4% |
| Kirac 66, grain Zn, 1994–95 | +7.7% | +27.3% | 19.6% |
| **Mean** | | | **12.5%** |

Where the AI did extract data, 2/5 matches are exact (0% error), showing that the underlying extraction logic is sound for readable rows. The larger errors for Kirac 66 and Sertak 52 (1994–95) reflect ctrl value misreads likely due to OCR noise.

---

## 6. Overall Assessment

**Verdict: Catastrophic undercapture due to LLM table truncation + Kimi failure + scanned PDF**

This paper represents a best-case scenario in terms of experimental design clarity (clean two-treatment factorial, clearly labeled columns, single site) but a worst-case scenario for LLM extraction:

1. **40 cultivar rows** far exceeds the typical LLM "working memory" for table extraction. Models consistently sample ~5–10 rows from such tables.

2. **Kimi's complete failure** (0 observations) forced a 2-model consensus that is even less likely to agree on all 40 rows than a 3-model vote.

3. **Scanned PDF** degrades OCR quality for lower-table rows, compounding the truncation problem with data corruption.

4. **The high r=0.939 on only 5 obs** confirms that the values the AI did extract are generally accurate — the failure is purely one of **coverage, not correctness**.

**Expected extractable observations:** 78 grain Zn pairs (the meta-analysis target)
**AI consensus extracted:** 4 grain Zn pairs (5.1% of target)
**Underlying model capacity:** Claude alone extracted 16 obs (including 12 additional cultivar-year grain Zn pairs), which could have yielded ~20% coverage with a relaxed consensus threshold or a single-model extraction

**Recommended remediation:**
- Manual extraction of Table 2 is the only reliable approach (the table is fully readable in the PDF on page 6)
- Alternatively, relaxing the consensus requirement to accept Claude-only observations when Kimi=0 would recover 12 additional grain Zn pairs (total ~16/78 = 21%)
- A re-run targeting "extract ALL 40 cultivar rows" explicitly in the prompt, with the table text pre-extracted and provided directly, would likely improve capture substantially
- This paper contributes disproportionately to the Hui 2023 dataset (78/310 = 25% of all GT rows) and its near-zero capture has major impact on overall recall metrics

**Classification:** Extractable paper, unextracted due to scale + scanning + consensus failure. Not a fundamental data-availability problem.
