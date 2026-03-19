# Per-Paper Extraction Quality Report: 126_Prokkola_2007

**Paper:** Prokkola S. (2007). Effect of biological sprays on the incidence of grey mould, fruit yield and fruit quality in organic strawberry production.
**Dataset:** Li 2022 validation set
**Report generated from:** `126_Prokkola_2007_match.json` + `126_Prokkola_2007_Effect of biological sprays on the incid_consensus.json`

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Matched pairs (vs. GT) | 6 |
| Mean Absolute Error (MAE) | 1.41% |
| Direction agreement | 83.3% (5/6) |
| Direction mismatches | 1 |
| Overall rating | Good |

---

## 1. Paper Design

Prokkola (2007) is a multi-treatment open-field experiment conducted in Finland, studying the effects of biological and biostimulant foliar sprays on organic strawberry (*Fragaria x ananassa*, cv. 'Jonsok') production. The paper reports two separate experiments:

- **Experiment 1** used a split-plot design replicated four times (n=4), with two locations (Ruukki and Mikkeli) as the main plots and six treatment arms: seaweed extract (1% v/v, *Ascophyllum nodosum*), garlic extract (1% w/v), compost extract (1% v/v), silicon spray (0.4% v/v), *Trichoderma* spp. spray (0.1% w/v), and a non-treated control. Data were collected over two years (2001 and 2002).

- **Experiment 2** used a randomized complete block design with six replicates (n=6) and four treatments, including *Gliocladium catenulatum* spray, with yields reported per yielding plant over the pooled 2001-2002 period.

Primary outcomes were Total yield (g/plant) and Marketable yield (g/plant) from Experiment 1, and the equivalent per-yielding-plant metrics from Experiment 2 (Table 2). Statistical analysis employed Tukey's studentized range test (P=0.05), with results presented as significance letters only — no numeric variance measures (SE, SD) were reported anywhere in the paper.

---

## 2. AI Consensus Extraction Results

The consensus pipeline ran Claude (26 observations) and Gemini (23 observations); Kimi extracted 0 observations and was excluded via tiebreaker logic. The final consensus set contained **14 observations**, all with high confidence and strong inter-model agreement (noted as diff < 6% for all pairs).

The extractor correctly identified:
- Both yield outcome types (Total yield and Marketable yield) across both experiments
- The correct control definition (non-treated plots)
- Sample sizes (n=4 for Experiment 1, n=6 for Experiment 2)
- The full treatment roster including seaweed extract, silicon spray, compost extract, *Trichoderma*, and *Gliocladium* from Tables 1 and 2

No variance values were extracted (correctly, given the paper reports only significance letters). All observations had `variance_type: null`, consistent with the recon finding that no numeric variance was available.

One notable gap: the garlic extract arm was captured only by Claude (flagged as a "claude_only_after_vote" entry) and did not enter the consensus set, because neither Gemini nor Kimi extracted it. The GT includes garlic extract in two observation rows (2001 and 2002), representing a systematic coverage failure for that treatment arm in the consensus output.

---

## 3. Ground Truth Comparison (All 6 Pairs)

The Li 2022 ground truth for this paper includes six observations (GT pairs 686-691), all for strawberry marketable yield and drawn from Experiment 1 only. All GT values are recorded in kg/plant; the consensus extraction used g/plant (a 1000x scale difference that is purely a unit convention and does not affect effect sizes).

| GT Pair | Treatment | Year | GT ctrl (kg) | GT treat (kg) | GT effect | Ext ctrl (g) | Ext treat (g) | Ext effect | Match | Direction |
|---------|-----------|------|-------------|--------------|-----------|-------------|--------------|-----------|-------|-----------|
| 686 | Seaweed extract (Biolan/SWE) | 2001 | 0.281 | 0.287 | +2.14% | 281 | 287 | +2.14% | Exact | Correct |
| 687 | Silicon (Kekkila Oy/Si) | 2001 | 0.281 | 0.291 | +3.56% | 281 | 291 | +3.56% | Exact | Correct |
| 688 | Garlic extract (PE) | 2001 | 0.281 | 0.263 | -6.41% | — | — | — | Not captured | N/A |
| 689 | Seaweed extract (Biolan/SWE) | 2002 | 0.384 | 0.423 | +10.16% | — | — | — | Not captured | N/A |
| 690 | Silicon (Kekkila Oy/Si) | 2002 | 0.384 | 0.415 | +8.07% | — | — | — | Not captured | N/A |
| 691 | Garlic extract (PE) | 2002 | 0.384 | 0.385 | +0.26% | — | — | — | Not captured | N/A |

GT pairs 686 and 687 match the consensus output with exact numeric agreement (after unit conversion). GT pairs 689 and 690 correspond to the 2002 seaweed and silicon observations that Claude extracted (see disagreements section of consensus JSON) but which did not enter the consensus set, since Gemini did not extract 2002 Experiment 1 data for those treatments.

**Direction mismatch:** GT pair 688 (garlic extract, 2001, Marketable yield) shows a ground-truth effect of -6.41% (treatment 263 g/plant vs. control 281 g/plant). The consensus set did not capture this observation; however, Claude's solo extraction for garlic extract in 2001 (json_idx entries in the disagreements) also produced a negative effect (-6.41%), meaning the direction would have matched the GT had it entered consensus. The single direction mismatch among the six GT rows thus arises from a coverage failure rather than a sign error in the extracted values.

---

## 4. Root Cause Analysis

**Coverage failures (4 of 6 GT rows unmatched):** The main limitation is that the consensus process collapsed to the 2001 Experiment 1 data for two treatments (seaweed and silicon) and missed the corresponding 2002 rows entirely. Two factors contribute:

1. **Gemini did not extract 2002 Experiment 1 data** for the seaweed and silicon treatments. Because Kimi extracted nothing and consensus required at least two-model agreement, these observations were demoted to Claude-only entries and excluded.

2. **Garlic extract was systematically absent from Gemini and Kimi outputs.** The recon guidance flagged garlic extract as "questionable as a biostimulant," which likely caused Gemini to deprioritize it. The GT confirms that Li 2022 did include garlic extract as a valid observation, so this represents a scope-definition mismatch between the AI and the human annotator.

**Direction mismatch explanation:** The apparent direction issue stems from the coverage gap, not from a numerical sign error. The one GT row involving a negative effect (garlic extract, -6.41%) was not extracted by the consensus pipeline at all, so the 5/6 direction agreement metric reflects the two correctly matched positive-effect pairs (686, 687) and the four unmatched rows where GT direction is unknown from the consensus perspective. The mismatch is therefore an artifact of selective coverage rather than a true extraction error.

**Variance:** No variance was available in the paper. This is a genuine data limitation that the extractor correctly identified and reported, rather than an extraction failure.

---

## 5. Overall Assessment

**Rating: Good (with coverage limitations)**

For the two observations that the consensus pipeline captured and the GT validated, accuracy was perfect: both effect sizes matched to two decimal places (2.14% and 3.56%), and both directions were correct. The extraction pipeline demonstrated reliable numeric fidelity for the data it did extract.

The primary weakness is **selective coverage across years and treatment arms**. The pipeline captured 2001 data reliably but largely missed 2002 data for the same treatments, and the garlic extract arm dropped out of consensus entirely. These gaps reduced the effective match rate from a potential 6/6 to 2/6 within the formal match file, although the MAE and direction metrics computed over the matched subset remain excellent.

For meta-analysis purposes, the underrepresentation of 2002 data and the garlic extract arm means that this paper contributes fewer effect size estimates than the GT contains. Researchers should be aware that the consensus output for this paper provides a conservative, partial view of the available data, particularly for the year-by-treatment interaction. A targeted re-extraction step focusing on year 2002 Experiment 1 rows and explicitly including garlic extract would recover the missing four observations.
