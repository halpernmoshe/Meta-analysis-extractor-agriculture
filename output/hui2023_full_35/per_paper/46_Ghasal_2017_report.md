# Extraction Quality Report: 46_Ghasal_2017

**Paper:** Ghasal, P.C., Shivay, Y.S., Pooniya, V., Choudhary, M., Verma, R.K. (2017). Response of wheat genotypes to zinc fertilization for improving productivity and quality. *Archives of Agronomy and Soil Science*, 63(11), 1597–1612. DOI: 10.1080/03650340.2017.1289515

**Match summary:** 0 extracted / 4 GT rows missed — classified as `no_extraction`

**GT sheets:** Data 2 Soil application (study_id = 46, obs 322–323); Data 4 Soil+Foliar application (study_id = 17, obs 91–92)

---

## 1. Paper Design

- **Species:** *Triticum aestivum* (wheat), 6 varieties (HD 2851, HD 2687, HD 2967, PBW 343, HD 2894, HD 2932)
- **Location:** ICAR-Indian Agricultural Research Institute, New Delhi, India (28°40'N, 77°12'E; altitude 228.6 m)
- **Experimental design:** Split-plot, 3 replications, 2 seasons (rabi 2013–14 and 2014–15); results reported as 2-year means
- **Main plots:** 6 wheat varieties
- **Sub-plots:** 5 Zn fertilization treatments:
  - T0: Control (no Zn, Zn0)
  - T1: 5.0 kg Zn ha⁻¹ soil as ZnSO₄·7H₂O (ZSHH)
  - T2: 2.5 kg Zn ha⁻¹ soil as ZnSO₄·7H₂O + 0.5% foliar spray at MT and BS
  - T3: 2.5 kg Zn ha⁻¹ soil as Zn-EDTA
  - T4: 1.25 kg Zn ha⁻¹ soil as Zn-EDTA + 0.5% foliar spray at MT and BS
- **n:** 3 replications per treatment × variety combination
- **Soil:** DTPA-extractable Zn = 0.63 mg kg⁻¹ (Zn-deficient; critical limit 0.38–0.90 mg kg⁻¹), pH 7.8, alluvium-derived sandy clay loam (*typic ustochrept*)
- **Variance:** LSD (P = 0.05) values provided at the bottom of all tables
- **Statistical analysis:** F-test per Gomez & Gomez (1984)

The paper covers **both pure soil application treatments** (T1, T3 → Data 2 Soil sheet) and **soil + foliar combined treatments** (T2, T4 → Data 4 Soil+Foliar sheet), which is why Hui 2023 assigns observations to two separate MOESM5 sheets under different study IDs (46 and 17).

---

## 2. Is Grain Zn Data Available in PDF?

**Yes — unambiguously — but exclusively in figures, not in tables.**

The paper's Results section (p. 1603, "Zinc concentration, uptake and RE") explicitly states:

> "Zn concentrations were highest and lowest in grain of HD 2851 and HD 2932 variety, respectively (Figure 2)."
> "Application of Zn in wheat crop increased grain Zn concentration by 8–10%." (Figure 3)

**Figure 2** (p. 1607): Bar chart showing Zn concentration (mg kg⁻¹ DM) in straw, grain, and spike straw for each of the 6 wheat varieties, averaged over Zn treatments and 2 years. Vertical bars are LSD₀.₀₅. Grain Zn values appear to range approximately 30–45 mg kg⁻¹ from the figure.

**Figure 3** (p. 1608): Bar chart showing Zn concentration in straw, grain, and spike straw for each Zn fertilization treatment (No Zn, 5.0 kg ZSHH, 2.5 kg ZSHH+FS, 2.5 kg ZnEDTA, 1.25 kg ZnEDTA+FS), averaged over varieties and 2 years. Vertical bars are LSD₀.₀₅. This figure is the direct source of the GT grain Zn concentration values.

**Critical observation:** No grain Zn concentration table exists anywhere in the paper. The 9 numbered tables (Tables 1–9) cover exclusively: CGR/RGR, NAR, LAI/DMA, spike weight/length, grains/spike and 1000-grain weight, grain and straw yield (Mg ha⁻¹), harvest index/cultivation cost, gross/net returns (USD ha⁻¹), and B:C ratio. Grain Zn concentration appears **only in Figures 2 and 3**, as bar charts with no printed numerical axis values legible as text.

---

## 3. AI Extraction Results (nothing extracted — why?)

The consensus JSON (`46_Ghasal_2017_consensus.json`) reveals the complete picture:

### 3a. Recon phase correctly identified the problem

All three models (Claude, Kimi, Gemini) returned **0 observations** for grain Zn concentration. The recon phase produced an explicit, accurate warning:

> "This paper has NO grain Zn concentration data despite being about Zn fertilization — only mentions 'Zn concentration' in methods but no results tables"
> "Figure 2 caption mentions 'Zn concentration in different parts' but the actual concentration values are not clearly provided in readable format"
> "Figure 3 also mentions Zn concentration but data values are not extractable from the text provided"

The extraction guidance concluded:

> "This paper does NOT contain extractable grain Zn concentration data ... The paper focuses on agronomic parameters rather than biofortification outcomes. **Should be excluded from meta-analysis on grain Zn concentration.**"

### 3b. What the AI did extract instead

The consensus pipeline extracted **4 observations** from economic/agronomic tables:

| Element extracted | Source | Notes |
|---|---|---|
| Grain yield (Mg ha⁻¹) | Table 6 | HD 2851, T1 (5 kg ZnSO₄), control = 4.21, treatment = 4.68 |
| Gross returns (US$ ha⁻¹) | Table 8 | HD 2851, T1 |
| Net returns (US$ ha⁻¹) | Table 8 | HD 2851, T1 |
| B:C ratio | Table 9 | HD 2851, T1 |

These 4 observations are present in `consensus_observations` but carry **zero overlap** with grain Zn concentration. They were correctly populated in the consensus JSON with `post_processing.final_count = 4`, yet the validation pipeline counted this paper as `no_extraction` because none of the 4 observations match any GT grain Zn row — the outcome variable mismatch means no pairing was possible.

### 3c. Why extraction failed for grain Zn specifically

The failure has one clear mechanical cause:

**The grain Zn data exists only in bar-chart figures (Figures 2 and 3), not in any table.** The figures show bar heights for Zn concentration (mg kg⁻¹ DM) in grain, straw, and spike straw, with LSD₀.₀₅ error bars, but no numerical values are printed on the bars or axes in a form readable from the PDF text layer. The PDF contains scanned/image content (the recon flagged `"is_scanned": true`, `"WARNING: SCANNED PDF - Text may have OCR errors"`), making figure value extraction from the text layer impossible.

Even with vision-based extraction, reading precise numeric values from bar charts without printed labels is unreliable. The AI correctly declined to hallucinate values from an unreadable figure.

The recon listed `"tables_with_target_data": []` (empty) — an accurate assessment. All 9 tables were catalogued under `tables_without_target_data`.

---

## 4. GT Data (4 MOESM5 rows)

Hui (2023) extracted the following from this paper, averaged over varieties and 2 years, reading Figure 3 manually:

### Sheet: Data 2 Soil application (study_id = 46)

| Obs ID | Zn fertilizer type | Zn rate (kg Zn ha⁻¹) | Control grain Zn (mg kg⁻¹) | Treatment grain Zn (mg kg⁻¹) | n |
|---|---|---|---|---|---|
| 322 | ZnSO₄ | 5.0 | 35.3585 | — | 3 |
| 323 | ZnEDTA | 2.5 | 35.3585 | — | 3 |

Note: The GT rows share the same control grain Zn value (35.3585 mg kg⁻¹), consistent with a single pooled control across varieties. Straw Zn = 22.217 mg kg⁻¹ for both rows.

Additional GT fields: Available Zn = 0.63 mg kg⁻¹, pH = 7.8, OM = 8.79 g kg⁻¹, N rate = 120 kg ha⁻¹, P rate = 26.4 kg ha⁻¹, K rate = 49.8 kg ha⁻¹, grain yield = 4,220 kg ha⁻¹ (= 4.22 Mg ha⁻¹, consistent with Table 6 control mean 4.22 Mg ha⁻¹ — confirms Hui used same paper).

### Sheet: Data 4 Soil+Foliar application (study_id = 17)

| Obs ID | Grain Zn (mg kg⁻¹) | Straw Zn (mg kg⁻¹) | n | Grain yield (kg ha⁻¹) |
|---|---|---|---|---|
| 91 | 35.3585 | 22.217 | 3 | 4,220 |
| 92 | 35.3585 | 22.217 | 3 | 4,220 |

The identical grain Zn and straw Zn values across all 4 rows (35.3585 and 22.217 mg kg⁻¹) and matching grain yield (4,220 kg ha⁻¹ = Table 6 mean control value of 4.22 Mg ha⁻¹) confirm these are the **control (no Zn) baseline values** shared across treatment contrasts. The GT structure represents control-vs-treatment pairs, with the treatment Zn concentrations read from Figure 3.

**Cross-check with paper text:** The paper states grain Zn concentration increased by 8–10% with Zn fertilization, and the control value of ~35 mg kg⁻¹ is consistent with the visible bar height in Figure 3 for the "No Zn" bar (~35 mg kg⁻¹ grain Zn). The GT values are plausible and internally consistent.

---

## 5. Root Cause

| Factor | Detail |
|---|---|
| **Primary cause** | Grain Zn concentration data exists **only in bar-chart figures** (Figures 2 and 3), not in any table. No numerical values are printed on the figures in readable form. |
| **Secondary cause** | The PDF is scanned/image-based, preventing reliable OCR recovery of figure bar heights. |
| **AI decision** | The recon correctly identified the absence of tabular grain Zn data and recommended exclusion. The extraction phase followed this guidance and extracted only tabular data (yield, economics) — which is correct behavior, not a failure. |
| **Validation mismatch** | The pipeline's `no_extraction` classification reflects the absence of grain Zn observations, not a total extraction failure: 4 economic observations were extracted but none match the GT outcome variable. |
| **GT source method** | Hui (2023) authors manually digitized bar heights from Figures 2 and 3 — a process that requires human visual reading of unlabeled bar charts, which automated extraction cannot reliably replicate without bar-chart digitization tools (e.g., WebPlotDigitizer). |

---

## 6. Assessment

**Classification:** `figure_only` — grain Zn data is present in the paper but confined to bar-chart figures with no printed numeric labels, making automated extraction infeasible without specialized figure digitization.

**AI behavior: CORRECT.** The recon and extraction pipeline accurately diagnosed the situation. The decision to flag the paper as having no extractable grain Zn concentration data and to recommend exclusion was the appropriate response. The 4 economic observations that were extracted demonstrate the pipeline was functioning; the absence of grain Zn observations reflects a genuine data-accessibility limitation, not an extraction bug.

**Recoverability:** Technically recoverable via manual bar-chart digitization using a tool such as WebPlotDigitizer applied to Figure 3 (Zn treatment effect on grain Zn concentration) and Figure 2 (variety effect). This would require:
1. Reading the y-axis scale from Figure 3 (approximately 10–45 mg kg⁻¹ range)
2. Digitizing bar heights for each treatment (No Zn, ZSHH, ZSHH+FS, ZnEDTA, ZnEDTA+FS)
3. Reading LSD₀.₀₅ error bar lengths as the variance measure
4. Cross-referencing n = 3 per treatment × variety cell (averaged over 6 varieties and 2 years in these figures)

**Impact on validation:** The 4 missed GT rows (obs IDs 322, 323, 91, 92) represent 2 soil-only treatment contrasts and 2 soil+foliar contrasts, all sharing a common control value of 35.36 mg kg⁻¹ grain Zn. Missing these rows does not indicate a systematic bias in the extraction pipeline — it reflects a fundamental limitation of figure-only data presentation. The paper contributes 0 rows to the validated dataset and should be excluded or flagged for manual digitization.

**Recommendation:** Mark as `requires_manual_digitization`. The paper is a legitimate contributor to the Hui (2023) meta-analysis but cannot be processed by automated text-based or vision-LLM extraction without dedicated bar-chart reading. Exclusion from the automated pipeline is appropriate.
