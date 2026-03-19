# Extraction Quality Report: 11_Zhao_2020

**Paper:** Mirbolook, A., Lakzian, A., Rasouli Sadaghiani, M., Sepehr, E., & Hakimi, M. (2020). Fortification of Bread Wheat Using Synthesized Zn-Glycine and Zn-Alanine Chelates in Comparison with ZnSO4 in a Calcareous Soil. *Communications in Soil Science and Plant Analysis*, 51(8), 1048–1064. https://doi.org/10.1080/00103624.2020.1744635

**Match summary:** 0 GT rows matched out of 6 GT rows across 3 MOESM5 sheets (zero-match)

**Root cause category:** Wrong control group + Figure misread + Incompatible comparator framework

---

## 1. Paper Design

**Species:** *Triticum aestivum* cv. N91-8 (bread wheat)

**System:** Greenhouse, pot experiment (4 kg calcareous soil per pot), Ferdowsi University of Mashhad, Iran

**Experimental design:** Completely randomised design, 2×3 factorial

- Factor 1 — Zinc source (3 levels): Zn-Glycine chelate (Zn-Gly), Zn-Alanine chelate (Zn-Ala), ZnSO4
- Factor 2 — Application method (3 levels): soil (8 mg Zn kg⁻¹ soil), foliar (0.25% w/v), fertigation (same concentration, 3 timings)
- Replications: n = 3

**Key soil properties (Table 1):** pH 7.8, CaCO3 14%, DTPA-Zn 0.4 mg kg⁻¹ (Zn-deficient), organic carbon 2.1 g kg⁻¹

**Primary research question:** Do synthesised Zn-amino acid chelates (Zn-Gly, Zn-Ala) deliver superior grain Zn biofortification compared to conventional ZnSO4?

**CRITICAL DESIGN NOTE:** The paper contains **no zero-Zn control treatment** in its results tables or figures. Every data point in the paper represents a Zn-fertilised treatment. The experiment directly compares Zn sources to each other, with ZnSO4 serving as the internal reference/comparator.

**How Hui 2023 included this paper:** The Hui 2023 meta-analysis coded ZnSO4 and ZnEDTA as "conventional" fertilisers and treated them as treatments relative to a zero-Zn control baseline. The MOESM5 dataset records the zero-Zn control grain Zn (33.3 mg kg⁻¹) as an external baseline drawn from Methods or other sources in the paper, not from a directly reported zero-Zn result table.

---

## 2. Grain Zn Data in PDF

### Figure 4 (primary source for grain Zn)

Figure 4 is a bar chart titled "Interaction of application methods with different sources of Zn (Zn-Gly, Zn-Ala and ZnSO4) on grain Zn concentration of wheat under greenhouse conditions." The y-axis label reads "Grain Zn Concentration (mg 100⁻¹)" — an ambiguous notation that could mean mg per 100 g (= g kg⁻¹ × 10) or could be a typographic rendering of mg kg⁻¹.

**Approximate values readable from Figure 4 bars (soil application method):**

| Treatment | Apparent bar height | If y-axis = mg/100g → mg/kg |
|-----------|--------------------|-----------------------------|
| Zn-Gly soil | ~4.5–5.0 | ~45–50 mg/kg |
| Zn-Ala soil | ~3.5–4.0 | ~35–40 mg/kg |
| ZnSO4 soil | ~1.2–1.5 | ~12–15 mg/kg |

The paper text confirms: "grain Zn concentration in Zn-Gly and Zn-Ala were 3.67 and 2.77 times more than ZnSO4 treatment" (soil application). If ZnSO4-soil = 12 mg/kg, then Zn-Gly-soil = 44 mg/kg (3.67×) — internally consistent.

**However, ZnSO4-soil yielding only 12 mg/kg grain Zn is biologically implausible for wheat.** Typical wheat grain Zn in Zn-deficient soils without any fertiliser is 15–25 mg/kg, and ZnSO4 application almost always increases above that. The zero-Zn control value recorded by Hui 2023 (33.3 mg/kg) is biologically realistic and almost certainly represents the baseline from the paper's Methods description or a separate data source, not from the Figure 4 bars.

### Table 5 (growth parameters, no grain Zn column)

Table 5 shows shoot length, shoot dry weight, spike length, spikelet number, and 1000-grain weight by treatment. **Grain Zn concentration is not in Table 5.**

### Table 6 (nutritional quality ANOVA table only)

Table 6 is an ANOVA summary with mean squares, not cell means. It confirms that grain Zn concentration differences between chelate types, application methods, and their interaction are all highly significant (p < 0.01), but supplies no extractable treatment means.

### Conclusion on PDF data availability

The only source of grain Zn means in the PDF is Figure 4, a bar chart. The figure shows a comparison among three Zn sources across three application methods — no zero-Zn control bar is visible. The absolute scale of Figure 4 is ambiguous.

---

## 3. AI Extraction Results

### Model votes

| Model | Observations extracted |
|-------|----------------------|
| Claude | 0 |
| Kimi | 30 |
| Gemini | 14 |
| Consensus (matched) | 9 |

Claude extracted nothing (0 observations), triggering a tiebreaker that favoured Kimi+Gemini agreement. The consensus pipeline selected 9 observations.

### Consensus observations (all 9)

| # | Element | Treatment | Control | Treat mean | Ctrl mean | Unit | Effect | Source |
|---|---------|-----------|---------|-----------|----------|------|--------|--------|
| 1 | shoot length | Zn-Gly soil | ZnSO4 soil | 50.45 cm | 49.31 cm | cm | +2.3% | Table 5 |
| 2 | **grain Zn concentration** | Zn-Gly soil | ZnSO4 soil | **44.0** | **12.0** | **mg/kg** | **+266.7%** | **Figure 4** |
| 3 | grain yield | Zn-Gly soil | ZnSO4 soil | 2.7 g/plant | 0.6 g/plant | g/plant | +350.0% | Figure 3 |
| 4 | 1000-grain weight | Zn-Gly soil | ZnSO4 soil | 37.83 g | 28.81 g | g | +31.3% | Table 5 |
| 5 | shoot dry weight | Zn-Gly soil | ZnSO4 soil | 1.89 g/plant | 0.98 g/plant | g/plant | +92.9% | Table 5 |
| 6 | spike length | Zn-Gly soil | ZnSO4 soil | 9.41 cm | 7.87 cm | cm | +19.6% | Table 5 |
| 7 | spikelet number | Zn-Gly soil | ZnSO4 soil | 15.0 spike⁻¹ | 11.03 spike⁻¹ | spike⁻¹ | +36.0% | Table 5 |
| 8 | grain protein | Zn-Gly soil | ZnSO4 soil | 14.5 | 10.5 | mg 100g⁻¹ | +38.1% | Figure 5 |
| 9 | grain phytic acid | Zn-Gly soil | ZnSO4 soil | 0.28 | 0.36 | mg 100g⁻¹ | −22.2% | Figure 6 |

**Only observation 2 (grain Zn) is relevant to the Hui 2023 meta-analysis outcome.** All other observations concern agronomic/quality endpoints not tracked by MOESM5.

### AI notes on grain Zn extraction

The AI notes for observation 2: *"Values converted from mg/100g to mg/kg. Means estimated from Figure 4 and text (3.67x increase). SE derived from EMS in Table 6. [from vision]"*

This confirms the AI:
1. Read Figure 4 bar heights visually (vision API)
2. Applied a ×10 unit conversion (mg/100g → mg/kg)
3. Used Table 6 EMS to derive variance

### Verification flags

All 9 consensus observations failed at least one verification check. Most failed the GRIM test (means not consistent with integer data at n=3). The grain Zn observation additionally failed magnitude (>100% effect flagged as extreme) and variance_type (CV heuristic suggested SD not SE). Observation 9 (grain phytic acid) was flagged as a likely T/C swap.

### Kimi-only disagreements

29 additional observations were extracted by Kimi but rejected by consensus vote (Gemini and Claude did not agree). These cover shoot length, spike length, spikelet number, shoot dry weight, and 1000-grain weight for all 9 treatment combinations (3 Zn sources × 3 application methods), using ZnSO4 within each application method as the comparator. None include grain Zn concentration.

---

## 4. GT Data from MOESM5 (all 3 sheets)

The GT text file (`gt_11_Zhao_2020.txt`) contains 6 rows across 3 sheets. In all cases the same zero-Zn baseline grain Zn (33.3 mg kg⁻¹) is recorded.

### Sheet: Data 2 — Soil application (study_id = 11)

| Obs ID | Zn fertiliser | Zn rate (kg/ha) | n | Ctrl grain Zn (mg/kg) | Unnamed col (lnRR) | Biofortif. index | Implied treatment Zn | Implied effect |
|--------|--------------|----------------|---|----------------------|-------------------|-----------------|----------------------|----------------|
| 68 | ZnSO4 | 4.52 | 4 | 33.3 | 0.15880 | 0.664 | 39.0 mg/kg | +17.2% |
| 69 | ZnEDTA | 0.56 | 4 | 33.3 | 0.14440 | 2.679 | 38.5 mg/kg | +15.5% |

Additional moderators: Available Zn 0.82 mg/kg, pH 8.2, OM 13.8 g/kg, N 100 kg/ha, P 52.8 kg/ha, K 0 kg/ha.

Note: n=4 in GT (Hui 2023 counted replicates differently) vs n=3 reported by the paper.

### Sheet: Data 3 — Foliar application (study_id = 18)

| Obs ID | Zn fertiliser | Zn rate (kg/ha) | Spray conc. (g Zn/L) | Spray frequency | n | Ctrl grain Zn (mg/kg) | Biofortif. index | Implied treatment Zn | Implied effect |
|--------|--------------|----------------|----------------------|-----------------|---|----------------------|-----------------|----------------------|----------------|
| 102 | ZnSO4 | 1.50 | 0.0682 | 3 times, timing 5 | 4 | 33.3 | −0.667 | 32.3 mg/kg | −3.0% |
| 103 | ZnEDTA | 0.30 | 0.0489 | 3 times, timing 5 | 4 | 33.3 | −14.333 | 29.0 mg/kg | −12.9% |

### Sheet: Data 4 — Soil+Foliar application (study_id = 8)

| Obs ID | n | Grain yield (kg/ha) | Ctrl grain Zn (mg/kg) | Straw Zn (mg/kg) | Grain Zn accum. (g/kg) |
|--------|---|--------------------|-----------------------|-----------------|----------------------|
| 22 | 4 | 3860 | 33.3 | — | 129.1 |
| 23 | 4 | 3860 | 33.3 | 19.1 | 91.7 |

Data 4 contains no lnRR column and no treatment grain Zn column. Effect size cannot be computed from this sheet alone.

### GT summary

- All 6 GT rows record control grain Zn = **33.3 mg/kg**
- The two computable effect sizes (Data 2) are: **+17.2%** (ZnSO4 soil) and **+15.5%** (ZnEDTA soil)
- The two foliar effects (Data 3): **−3.0%** (ZnSO4 foliar) and **−12.9%** (ZnEDTA foliar)
- Zn-Gly and Zn-Ala chelates are **not represented** in the GT dataset
- The "control" in GT is always zero-Zn (no fertiliser), baseline = 33.3 mg/kg

---

## 5. Root Cause Analysis

### 5a. Fundamental comparator mismatch (primary cause)

The paper's design compares novel Zn chelates to ZnSO4. The Hui 2023 meta-analysis re-coded the same paper to compare ZnSO4 (and ZnEDTA) against an external zero-Zn control baseline. These are two completely different comparisons from the same experimental data:

| Framework | Treatment | Control | Grain Zn effect |
|-----------|-----------|---------|----------------|
| This paper (AI's view) | Zn-Gly soil | ZnSO4 soil | +266.7% |
| Hui 2023 MOESM5 (GT) | ZnSO4 soil | Zero-Zn baseline | +17.2% |
| Hui 2023 MOESM5 (GT) | ZnEDTA soil | Zero-Zn baseline | +15.5% |

The AI correctly identified what the paper reports (Zn-Gly vs ZnSO4), but this comparison is structurally different from what Hui 2023 coded in MOESM5.

### 5b. Zero-Zn control not in the paper (contributing cause)

The paper presents no zero-Zn control in its results. The 33.3 mg/kg baseline used in MOESM5 cannot be read from Figure 4 or Table 5. Hui 2023 likely sourced this value from:
- An external reference for typical wheat grain Zn in Iranian calcareous soils
- A baseline measurement described in the Methods section not tabulated in results
- An inference from the experimental soil properties (DTPA-Zn = 0.4 mg/kg, a severely Zn-deficient soil)

The AI had no way to extract a zero-Zn control value from this paper because it is not reported in the results.

### 5c. AI grain Zn values are almost certainly wrong in absolute terms

The AI extracted ZnSO4-soil grain Zn = 12.0 mg/kg (after converting from mg/100g), which is biologically implausible. Wheat grain Zn in a ZnSO4-fertilised calcareous soil would normally be 25–50 mg/kg, not 12 mg/kg. The GT value of 33.3 mg/kg as the zero-Zn baseline is more biologically plausible than 12.0 mg/kg as the ZnSO4 value.

The most likely explanation: Figure 4's y-axis runs to approximately 6 units, but the actual scale appears to be in mg/100g. Reading from Figure 4 at the ZnSO4-soil bar:
- If the bar is at ~1.2 mg/100g × 10 = **12 mg/kg** (AI's reading after conversion) — too low
- If the bar is at ~3.3 mg/100g × 10 = **33 mg/kg** — matches GT

The ratio of GT-to-AI control values = 33.3 / 12.0 = **2.77**, which happens to equal the stated Zn-Ala/ZnSO4 ratio ("2.77 times more than ZnSO4"). This suggests the AI may have misread the ZnSO4 bar for the Zn-Ala bar, or misread the absolute scale by a factor of ~2.8. The figure reading is inherently unreliable because it is an image-only bar chart from a scanned/compressed PDF.

### 5d. Wrong Zn source extracted

Even if the absolute values were correct, the AI extracted only the **Zn-Gly vs ZnSO4** comparison. Hui 2023 coded only **ZnSO4** and **ZnEDTA** as the relevant treatments (both conventional forms). Zn-Gly and Zn-Ala are novel experimental chelates not tracked in MOESM5. The AI extracted the paper's primary finding but not the comparisons the meta-analysis cares about.

### 5e. Non-grain-Zn observations not matchable

The remaining 8 consensus observations (shoot length, grain yield, 1000-grain weight, etc.) are not outcomes tracked by MOESM5, which focuses exclusively on grain Zn concentration as the primary outcome variable.

### 5f. Claude produced zero observations

The recon correctly flagged this paper as "HARD" (scanned PDF, figure-only data for the key outcome, no clear zero-Zn control). Claude, the most conservative model, extracted nothing, which on reflection reflects the correct reading: this paper cannot be used for the Hui 2023 meta-analysis as it contains no zero-Zn control data. The tiebreaker forced Kimi+Gemini consensus, producing 9 technically extracted but meta-analytically incompatible observations.

---

## 6. Assessment

**Overall verdict: Structurally incompatible paper — zero match is correct**

This is not an extraction failure in the conventional sense. The AI correctly identified the paper's experimental structure and extracted data that the paper actually reports. The zero-match outcome arises because:

1. **The paper does not report the comparison that Hui 2023 requires.** MOESM5 needs Zn-fertiliser vs zero-Zn control. This paper only reports Zn-source A vs Zn-source B.

2. **The zero-Zn baseline (33.3 mg/kg) does not appear in the paper's results.** It was apparently supplied by Hui 2023 authors independently. The AI cannot reconstruct this from the PDF.

3. **The AI extracted the wrong Zn treatments** (Zn-Gly vs ZnSO4) instead of the treatments the meta-analysis coded (ZnSO4 vs zero-Zn; ZnEDTA vs zero-Zn).

4. **Figure 4 reading errors** compounded the problem: the AI likely misread bar heights, yielding ZnSO4-soil = 12 mg/kg instead of ~33 mg/kg, and a spurious +266.7% effect instead of realistic ~+17% and ~+16%.

**Could the correct data have been extracted?** Possibly, but only with Hui 2023-specific instructions:
- The prompt would need to specify that the control is zero-Zn application, not ZnSO4
- The prompt would need to tell the AI to look for a background grain Zn value in the Methods/soil description section
- The prompt would need to specify that only ZnSO4 and ZnEDTA are the relevant treatments

**Impact on validation statistics:** The zero-match correctly prevents this paper's spurious +266.7% grain Zn effect from contaminating the validation dataset. Including it would introduce a large outlier. The matching pipeline's failure here is the correct outcome.

**Recommendation:** Flag this paper as "incompatible design" in the validation log. It belongs in a separate category from papers where the AI extracted the right comparison but with inaccurate values. The recon warning ("This paper does not contain suitable data for the CO2 meta-analysis" — this warning was generated for the CO2 context but analogously applies to the Hui Zn meta-analysis context) correctly identified the structural problem.

| Dimension | Assessment |
|-----------|-----------|
| GT rows in MOESM5 | 6 (across 3 sheets) |
| AI grain Zn obs | 1 |
| Correct comparator | No — AI used ZnSO4 as control; GT uses zero-Zn as control |
| Correct Zn source | No — AI extracted Zn-Gly; GT tracks ZnSO4 and ZnEDTA |
| AI effect | +266.7% (ZnGly vs ZnSO4) |
| GT effect | +17.2% (ZnSO4 vs zero-Zn), +15.5% (ZnEDTA vs zero-Zn) |
| Figure reading accuracy | Likely wrong by ~2.8× in absolute scale |
| Match achievable? | No — requires external zero-Zn baseline not in PDF results |
| Zero match verdict | Correct outcome |
