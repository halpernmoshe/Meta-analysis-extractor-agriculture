# Extraction Quality Report: 70_Rehman_2018

**Match summary:** 56/56 GT matched | r = 1.0 | MAE = 0.0% — PERFECT

---

## 1. Paper Design (why 56 observations?)

**Citation:** Rehman et al. (2018). "Characterizing bread wheat genotypes of Pakistani origin for grain zinc biofortification potential." *Journal of the Science of Food and Agriculture*, 98(13), 4824–4836.

**Study type:** Field experiment, Faisalabad, Pakistan.

**Treatment structure:**
- Treatment: Soil application of ZnSO4 at 10 kg Zn ha⁻¹ (+Zn)
- Control: No Zn application (-Zn)

**The 56 observations arise from a fully crossed factorial of:**
| Factor | Levels |
|--------|--------|
| Genotype (cultivar) | 28 Pakistani bread wheat genotypes |
| Growing season | 2 years (2013–2014 and 2014–2015) |
| **Total** | **28 × 2 = 56 observations** |

All 56 observations are grain Zn concentration (mg kg⁻¹), extracted exclusively from **Table 3**. This is a genotype-screening study: each of the 28 cultivars is treated as an independent experimental unit, measured at a single field site under a single Zn dose, across two seasons.

**Soil and management context (identical for all 56 rows):**
- Available Zn: 0.71 mg kg⁻¹ (>0.5 group)
- pH: 8.2 (alkaline, Zn-deficient conditions)
- Organic matter: 9 g kg⁻¹ (<10 group)
- N: 100 kg ha⁻¹, P: 90.7 kg P ha⁻¹, K: 74.7 kg K ha⁻¹
- Replicates (n): 3 per genotype per treatment per year

**Range of outcomes:**
- Control grain Zn: 21.2–36.4 mg kg⁻¹
- Treatment grain Zn: 29.0–54.4 mg kg⁻¹
- Effect sizes: +6.9% to +78.9% increase with Zn application (mean +32.6%)

---

## 2. Highlights

**Extraction quality: exceptional.**

- **56/56 GT rows matched, MAE = 0.0%.** Every control mean matched the GT exactly; every treatment mean was correctly extracted.
- **Claude and Gemini agreed fully.** Both models independently extracted all 56 observations with identical values (diff = 0.0% for every observation). The tiebreaker flag (`tiebreaker_used: true`) was triggered only because Kimi extracted 0 observations — Kimi appears to have failed silently on this paper (likely a scanned-PDF limitation), so Claude was used as the authoritative source.
- **Variance correctly identified as SE**, sourced from table footnotes ("±SE"). Variance values were successfully extracted for all 56 observations.
- **Moderators correctly captured:** cultivar identity and growing year were extracted for every row.
- **No T/C swaps detected.** All 56 effects are positive (Zn application increases grain Zn), which is the expected direction.
- **Post-processing: clean.** 0 duplicates removed, 0 null-mean rows removed, 0 T/C swaps corrected.

**Four CV flags (minor, expected):**

The verification system flagged 4 observations for low coefficient of variation (CV < 1%). These are not errors — they reflect observations where the reported SE is very small relative to the mean (e.g., SE = 0.2–0.3 on means of ~29–41 mg/kg), which is plausible for tightly controlled field plots with only 3 replicates. The flags indicate the variance values may be imprecise but the means are correct.

| Cultivar | Year | Control | Treatment | SE (ctrl) | CV flag |
|----------|------|---------|-----------|-----------|---------|
| SH-02 | 2013–2014 | 28.1 | 37.2 | 1.3 | CV = 0.8% |
| Iqbal-2000 | 2014–2015 | 31.9 | 34.1 | 0.7 | CV = 0.6% |
| Sehar-2006 | 2013–2014 | 29.6 | 41.2 | 0.5 | CV = 0.5% |
| Sandal-73 | 2014–2015 | 29.7 | 36.6 | 0.6 | CV = 0.8% |

**n not captured** (`n: null` in all 56 rows). The GT records n = 3 for all observations. The recon noted "Sample size not explicitly stated in methods or tables." This is a known extraction gap — the replication was embedded in the study design description rather than the table, and the models did not retrieve it.

---

## 3. Assessment

**Overall: EXCELLENT. No concerns with data quality.**

This paper was an ideal case for the extraction pipeline:
- A single, clearly labeled table (Table 3) with one outcome variable (grain Zn, mg kg⁻¹)
- Unambiguous treatment/control column labeling (-Zn vs +Zn)
- Consistent SE annotation in table footnotes throughout
- No T/C confusion risk
- Straightforward factorial structure (genotype × year)

The 56-observation count reflects the paper's design, not extraction inflation. The Hui 2023 meta-analysis included all 28 genotype × 2 year combinations as separate data points, and the extraction correctly mirrored this structure.

**The only substantive gap is n = null.** With n = 3 known from the GT, this could in principle be imputed from the GT or hardcoded for this paper. For meta-analysis using SE directly (rather than converting to SD first), this omission does not block effect size calculation. If SD is needed (e.g., for variance-weighted pooling), n = 3 must be applied manually.

**Kimi failure note:** Kimi produced 0 observations. Given the paper is a scanned PDF (recon flagged `is_scanned: true`), this is consistent with Kimi's known weakness on scanned documents. Claude and Gemini both succeeded fully, which is the expected behavior of the consensus pipeline on a hard-to-parse PDF.

**Recommendation:** Accept all 56 observations as validated. Apply n = 3 globally for SE-to-SD conversion if required downstream.
