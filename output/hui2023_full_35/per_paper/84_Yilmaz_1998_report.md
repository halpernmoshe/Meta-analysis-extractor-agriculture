# Extraction Quality Report: 84_Yilmaz_1998

**Match summary:** 3/3 GT matched, r = 1.0, MAE = 0.0% — PERFECT

---

## 1. Paper Design

**Full citation:** Yilmaz, A., Ekiz, H., Gultekin, I., Torun, B., Barut, H., Karanlik, S., Cakmak, I. (1998). Effect of seed zinc content on grain yield and zinc concentration of wheat grown in zinc-deficient calcareous soils. *Journal of Plant Nutrition*, 21(10), 2257–2264.

**Country:** Turkey

**Crop:** Bread wheat (*Triticum aestivum*, cv. Atay 85)

**Experimental system:** Field trial, two growing seasons (1994–95 and 1995–96)

**Design:** Two-factor split-plot randomized complete block, 6 replications

**Factorial structure:**
- Primary factor: seed Zn content (3 levels: 355, 800, 1465 ng Zn seed⁻¹)
- Secondary factor: soil Zn application (2 levels: 0 vs. 23 kg Zn ha⁻¹ as ZnSO4)
- Third factor: irrigation regime (rainfed vs. irrigated)
- Two years reported separately

**Soil conditions:** Severely Zn-deficient calcareous soil — available Zn = 0.09 mg kg⁻¹ (well below 0.5 mg kg⁻¹ critical threshold), pH 7.8, CaCO3 30%, organic matter 15 g kg⁻¹.

**Outcome variables extracted:**
- Table 1: Grain yield (kg ha⁻¹) — yield response to soil Zn application and seed Zn content
- Table 2: Shoot Zn concentration at shooting stage (mg Zn kg⁻¹ DW) and grain Zn concentration at harvest (mg Zn kg⁻¹ DW) — rainfed conditions only

**Variance reporting:** LSD(5%) values provided in Table 1 for yield data. Zn concentration data in Table 2 shares a common LSD = 1.3 mg kg⁻¹ (grain) and 1.4 mg kg⁻¹ (shoot).

**Important design note:** The primary research question is the effect of seed Zn content (a biofortification mechanism), not soil Zn fertilization per se. Soil Zn application is a secondary factor. The meta-analysis (Hui 2023) uses this paper for its soil Zn application treatment arm (0 vs. 23 kg Zn ha⁻¹), treating each seed Zn content level as a separate observation. This is a legitimate use of the data but requires awareness of the confounded design.

**PDF status:** Scanned document — OCR text with potential character-level errors; estimated difficulty HARD.

---

## 2. Ground Truth Structure

The Hui 2023 MOESM5 dataset (Data 2 Soil sheet) contains 9 rows for study ID 84, spanning observation IDs 829–837.

Of these 9 rows:
- **Rows 829–831** (obs IDs 829, 830, 831): Contain grain Zn concentration values (9.8, 10.1, 9.5 mg kg⁻¹) — these are the **control-arm** (0 kg Zn ha⁻¹) grain Zn values for each of the three seed Zn content levels (355, 800, 1465 ng seed⁻¹).
- **Rows 832–837** (obs IDs 832–837): Contain grain yield data only (1,485–6,190 kg ha⁻¹ across seed Zn levels and irrigation regimes) — no grain Zn concentration reported.

The 3 GT rows that participate in Zn concentration validation are therefore the control-arm means from Table 2, rainfed conditions, averaged across two years (as the GT does not record year).

---

## 3. Consensus Extraction Summary

The pipeline produced 7 consensus observations total:

| Element | Tissue | Control mean | Treatment mean | Effect (%) | Confidence |
|---------|--------|-------------|----------------|-----------|------------|
| Grain Zn conc. | grain | 9.8 mg/kg (seed 355) | 13.2 mg/kg | +34.7% | high |
| Grain Zn conc. | grain | 10.1 mg/kg (seed 800) | 13.0 mg/kg | +28.7% | high |
| Grain Zn conc. | grain | 9.5 mg/kg (seed 1465) | 13.3 mg/kg | +40.0% | high |
| Shoot Zn conc. | shoot | 7.6 mg/kg (seed 355) | 16.2 mg/kg | +113.2% | high |
| Shoot Zn conc. | shoot | 7.8 mg/kg (seed 800) | 16.1 mg/kg | +106.4% | high |
| Shoot Zn conc. | shoot | 7.2 mg/kg (seed 1465) | 16.2 mg/kg | +125.0% | high |
| Grain yield | grain | 480 kg/ha (seed 800 vs 355) | 920 kg/ha | +91.7% | high |

Both Claude (6 obs) and Kimi (18 obs) contributed; Gemini extracted 0 obs (likely OCR failure on the scanned PDF). No tiebreaker was needed. The 3 grain Zn concentration consensus observations match the GT control means of 9.8, 10.1, and 9.5 mg kg⁻¹ exactly.

---

## 4. Validation Match Detail

The 3 GT control-arm grain Zn concentration values were matched with zero error:

| GT Obs ID | GT Grain Zn (mg/kg) | Extracted Control Mean | Error |
|-----------|---------------------|----------------------|-------|
| 829 | 9.8 | 9.8 | 0.0% |
| 830 | 10.1 | 10.1 | 0.0% |
| 831 | 9.5 | 9.5 | 0.0% |

The perfect match reflects that Table 2 in the original paper reports these values unambiguously as single numbers per cell (no rounding choices, no unit conversions needed). The LSD values (1.3 mg kg⁻¹ for grain Zn) were also correctly extracted from the shared table footnote.

---

## 5. Assessment: Perfect

**Why this paper matched perfectly:**

1. **Table 2 values are simple and unambiguous.** The grain Zn concentration values (9.5, 9.8, 10.1 mg kg⁻¹) are low-precision integers with one decimal place. Even OCR on a scanned PDF can read these without error.

2. **Both models agreed exactly.** The consensus JSON records "Models agree (diff=0.0%)" for all three grain Zn observations, indicating Claude and Kimi independently produced identical values — strong confirmation of correct extraction.

3. **The LSD was stated once for the whole table**, making variance assignment unambiguous (LSD = 1.3 mg kg⁻¹ for grain Zn, LSD = 1.4 mg kg⁻¹ for shoot Zn).

4. **The GT uses only the control arm for validation.** Since the GT records the baseline (0 kg Zn ha⁻¹) means, the extraction only needed to correctly identify which column was "no soil Zn" — a straightforward label in the original table.

**Verification flags (non-critical):**

All three grain Zn obs triggered `grim` and `variance_type` check failures. These are expected artifacts of the paper's data type and reporting convention:

- **GRIM failures:** The GRIM test assumes integer-scale underlying data. Grain Zn concentration in mg kg⁻¹ is continuous, so the GRIM test does not apply. These failures are false positives and can be ignored.
- **Variance type ambiguity:** The pipeline flagged the variance as possibly "SD" (CV heuristic) rather than "LSD" as reported. This is a known limitation of the CV-based heuristic applied to LSD values, which are experiment-level statistics rather than group-level dispersions. The paper explicitly states LSD(5%) in the table footnote; the reported type is correct.

The three shoot Zn observations additionally triggered `magnitude` flags (effects of +106% to +125%). These large effects are ecologically plausible: the soil was severely Zn-deficient (available Zn = 0.09 mg kg⁻¹), and shoot tissue at shooting stage accumulates Zn rapidly when soil supply is adequate. The flag reflects a conservative threshold, not an extraction error.

**Overall quality rating: EXCELLENT**

The extraction correctly recovered all 3 GT grain Zn concentration values with zero error. The paper is a difficult scanned PDF with a complex factorial design, yet the key outcome table (Table 2) was read accurately by both models. The 6 unmatched GT rows (grain yield only, obs 832–837) are outside the grain Zn concentration outcome used for validation and do not represent extraction failures.

**Caveats for meta-analysis use:**

- The paper's primary intervention is seed Zn content, not soil Zn fertilization. Observations represent a partially confounded design.
- Grain Zn concentration data in Table 2 is only available for rainfed conditions; irrigated data are not reported for Zn concentrations.
- No year breakdown is given for the Zn concentration data in Table 2 (values appear averaged or from a single year not specified), while yield data in Table 1 is year-specific.
- The LSD is a shared table-level value, not group-specific; effect size weighting should account for this.
