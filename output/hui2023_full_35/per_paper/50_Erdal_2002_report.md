# Extraction Quality Report: 50_Erdal_2002

**Paper:** Erdal, I., Yilmaz, A., Taban, S., Eker, S., Torun, B., Cakmak, I. (2002). Phytic acid and phosphorus concentrations in seeds of wheat cultivars grown with and without zinc fertilization. *Journal of Plant Nutrition*, 25(1), 113–127.

**Match summary:** 20/20 GT matched | r = 1.0 | MAE = 0.0% — PERFECT

**GT sheet:** Data 2 Soil application (study_id = 50)

**Consensus JSON:** `output/hui2023_full_35/50_Erdal_2002_consensus.json`

---

## 1. Paper Design

| Feature | Detail |
|---------|--------|
| Country | Turkey |
| Species | *Triticum aestivum* (18 cultivars), *T. durum* (2 cultivars) — 20 total |
| System | Field experiment (randomized complete block, strip-plot, 4 replications) |
| Intervention | Soil Zn fertilization: +Zn = 23 kg Zn ha⁻¹ as ZnSO₄·7H₂O |
| Control | No Zn fertilization (−Zn = 0 kg Zn ha⁻¹) |
| Soil Zn (available) | 0.1 mg kg⁻¹ (severely Zn-deficient) |
| Target table | Table 1 only (effect of Zn on P and Zn concentrations in 20 cultivars) |
| Excluded tables | Table 2 (phytic acid / PA:Zn ratio), Table 3 (phytase activity), Table 4 (observational survey, 55 locations — not experimental) |
| Variance type | LSD (single table-level values: LSD = 0.8 mg kg⁻¹ for Zn, 0.14 mg g⁻¹ for P) |
| PDF condition | Scanned — OCR-dependent; extraction method flagged as HARD |
| Recon difficulty | HARD (scanned, OCR risk, two separate datasets to distinguish) |

The paper's primary scientific focus is phytic acid chemistry, not grain Zn per se. Zinc concentration data (Table 1) is a secondary outcome in the original publication but the primary target outcome in the Hui 2023 meta-analysis. The 55-location observational study (Table 4) was correctly excluded by the extractor.

---

## 2. Highlights

### 2.1 Perfect numeric recovery across 20 cultivars

The extractor recovered control Zn concentrations for all 20 cultivars with zero error. Treatment Zn concentrations were also matched exactly, yielding a perfect r = 1.0 and MAE = 0.0% against the Hui 2023 ground truth.

| Statistic | Value |
|-----------|-------|
| GT observations (Zn grain) | 20 |
| Extracted Zn observations | 20 |
| Matched | 20 / 20 (100%) |
| Pearson r | 1.0 |
| MAE | 0.0% |
| Control Zn range (−Zn) | 7.0 – 10.5 mg kg⁻¹ (mean 8.8) |
| Treatment Zn range (+Zn) | 13.8 – 22.8 mg kg⁻¹ (mean 16.8) |
| Effect size range | +48.0% to +132.7% (mean +92.2%) |
| lnRR range | 0.392 – 0.844 (mean 0.647) |

Grain Zn effects are large and uniformly positive, which is expected for soil application on severely Zn-deficient soil (available Zn = 0.1 mg kg⁻¹). The cultivar with the strongest response is Kirac 66 (+132.7%); the weakest is Partizanka (+48.0%).

### 2.2 Correct table scope and treatment identification

The recon correctly identified Table 1 as the sole source of meta-analytically usable data and explicitly excluded Table 4. Treatment (−Zn vs +Zn) columns were correctly assigned with no treatment/control swap.

### 2.3 Model agreement: Claude + Gemini unanimous

All 37 consensus observations (20 Zn + 17 P) carry the note "Models agree (diff=0.0%) [Claude+Gemini agree]". Kimi extracted 0 observations (likely failed on the scanned OCR content) and was excluded via the tiebreaker rule. The Claude + Gemini agreement therefore functions as a two-model consensus rather than a three-model consensus, but the result is reliable given the exact numeric match to GT.

### 2.4 Bonus extraction of P concentration (17 observations)

The extractor also captured grain P concentration data for 17 of the 20 cultivars from the same Table 1. These P observations are not part of the Hui 2023 GT (which tracks only Zn) and do not affect validation metrics. However, they represent additional correctly extracted agronomic data. Three cultivars appear to be missing P entries — likely a minor OCR issue on the scanned table.

### 2.5 Variance: LSD correctly assigned, one table-level value used for all

Both Zn LSD (0.8 mg kg⁻¹) and P LSD (0.14 mg g⁻¹) were correctly extracted from the table footnotes. Because the paper provides only a single pooled LSD per variable (not cultivar-specific SEs), the same variance value is assigned to all rows. This is the correct handling for this data structure; it does not constitute an extraction error.

### 2.6 Verification flags (non-critical)

The consensus JSON records verification flags for the P observations:

- **Direction flag:** Zn fertilization reduced grain P concentration (−5% to −27%), which the extractor's generic direction check flags as "expected positive." This is scientifically correct behavior — soil Zn suppressing grain P is a known interaction — but the generic flag treats P in the same direction as Zn. Not an error.
- **TC-swap suggestion:** Triggered by the direction flag; not a real swap. P concentrations are correctly assigned (treatment < control).
- **GRIM flag:** Triggered because the P means (e.g., 3.6, 4.2 mg g⁻¹) are not constrained to integer granularity as assumed by the GRIM test. These are ratio measurements on a continuous scale and the GRIM assumption does not apply. Not an error.

None of these flags affect the 20 Zn observations that constitute the validation match.

---

## 3. Assessment: Perfect

This is an ideal extraction result. Despite the added complexity of a scanned PDF, a 20-cultivar factorial layout, a secondary dataset (Table 4) to exclude, and the paper's focus on phytic acid rather than Zn per se, the extractor recovered every single observation precisely.

The key decisions made correctly by the pipeline were:

1. **Table scope:** Only Table 1 used; Table 4 (observational, 55 locations) correctly excluded.
2. **Treatment identification:** −Zn and +Zn columns correctly assigned; no T/C swap.
3. **Cultivar-as-observation:** Each of 20 cultivars treated as an independent observation with n = 4 replications.
4. **Variance:** Table-level LSD extracted correctly for both Zn and P.
5. **OCR robustness:** Despite scanned PDF status, all 20 numeric values read correctly.
6. **Dataset disambiguation:** Field experiment (Tables 1–3) correctly distinguished from observational survey (Table 4).

**Suitability for meta-analysis:** High. All 20 Zn observations have control mean, treatment mean, n = 4, and LSD variance. Effect sizes span a biologically plausible range (+48% to +133%) consistent with severe Zn deficiency conditions. No data cleaning required before inclusion in Hui 2023 meta-analysis.
