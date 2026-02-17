# Table 1. Comparison of LLM-Based Data Extraction Systems for Evidence Synthesis

This table compares published LLM-based data extraction systems for systematic reviews and meta-analysis. Systems are ordered chronologically. "Quantitative accuracy" refers specifically to extraction of continuous numerical values (means, SDs, effect sizes) rather than categorical study characteristics.

| Feature | Gartlehner et al. (2024) | Gougherty & Clipp (2024) | Gartlehner et al. (2025) | Jensen et al. (2025) | Khan et al. (2025) | Li et al. (2025) | Sallam et al. (2025) | Poser et al. (2026) | **This study** |
|---|---|---|---|---|---|---|---|---|---|
| **Journal** | Res. Synth. Methods | npj Biodiversity | Ann. Intern. Med. | PLoS ONE | JAMIA | arXiv preprint | Res. Synth. Methods | Front. Artif. Intell. | -- |
| **Domain** | Clinical (RCTs) | Ecology | Clinical (6 SRs) | Clinical (exercise) | Clinical (LSR) | Clinical (3 domains) | Clinical (CBT/insomnia) | Clinical (MS neurology) | **Plant science** |
| **LLM(s)** | Claude 2 | text-bison-001 | Claude 2.1/3.0/3.5 | ChatGPT-4o | GPT-4-turbo + Claude-3-Opus | GPT-4o-mini, Gemini-2.0-flash, Grok-3 | GPT-4o; o3 | Claude 3.7 Sonnet + Gemini 2.0 Pro + o3-mini | **Claude Sonnet 4 + Kimi K2.5 + Gemini 2.5 Flash** |
| **N papers** | 10 | 100 | 63 | 11 | 22 pubs (10 trials) | 58 | 290 (dev) + 15 (val) | 30 reports | **104 (42 + 34 + 28)** |
| **Data type** | Categorical | Both | Both | Both | Both | Both | Both | Categorical | **Quantitative** |
| **Primary metric** | 96.3% accuracy | >90% categorical | 91.0% accuracy | 92.4% accuracy | 94% concordant acc. | Precision 0.75--0.95 | 72.6--75.3% accuracy | 1.48% error rate | **r = 0.453--0.950** |
| **Quant. accuracy** | Not tested | 23.8% | Not separated | Not separated | Incl. in 94% (exact match) | Recall 0.21--0.76 | Incl. in 72--75% | N/A (categorical) | **MAE 4.95--11.62%; 85--99% direction** |
| **Multi-model?** | No | No | No* | No | Yes (2-model) | Tested (ensemble) | No | Yes (3-model) | **Yes (3-model consensus)** |
| **Consensus gain** | -- | -- | -- | -- | Concordant: 94% vs discord: 41--50% | +5.9% recall | -- | 30% error reduction | **73% more observations** |
| **Cost/paper** | NR | NR | NR | NR | NR | NR | NR | NR | **~$0.37** |
| **Equivalence test** | No | No | No | No | No | No | No | No | **TOST p = 0.004 (+-2 pp)** |
| **Key limitation** | 10 papers; categorical only | 23.8% quant. accuracy | 77.2% concordance | 11 papers; single domain | Small sample; discord acc. low | Low recall (21--76%) | Moderate acc. (72--75%) | 30 reports; categorical only | Plant science; +-30 pp LOA |

\* Gartlehner et al. (2025) tested multiple Claude versions sequentially but not as a true multi-model consensus.

NR = Not reported; SR = systematic review; LSR = living systematic review; MS = multiple sclerosis; CBT = cognitive behavioral therapy; LOA = limits of agreement.

---

## Key Observations

1. **Categorical vs. quantitative gap.** Most existing systems report high accuracy (90--96%) on categorical data (study characteristics, PICO elements, risk-of-bias), but quantitative extraction accuracy is substantially lower where tested: 23.8% for ecological data (Gougherty & Clipp 2024), recall of 21--76% for statistical results (Li et al. 2025), and 72--75% overall when both types are combined (Sallam et al. 2025). Our system targets the harder quantitative extraction task exclusively.

2. **Multi-model consensus.** Only three systems employ true multi-model consensus: Khan et al. (2025) with 2 models, Poser et al. (2026) with 3 models for categorical data, and our system with 3 models for quantitative data. All three demonstrate that concordance between independent models substantially improves reliability -- from error reduction of 30% (Poser) to accuracy gains from 41--50% to 94% for concordant pairs (Khan) to 73% more observations captured (this study).

3. **Formal equivalence testing.** Our system is the only one to perform formal statistical equivalence testing (TOST), demonstrating that extracted data are statistically indistinguishable from human extraction at a +-2 percentage point margin. Other systems report accuracy percentages but do not perform equivalence analyses.

4. **Cost transparency.** Our system is the only one to report per-paper extraction costs (~$0.37/paper, $17 total for 46 papers), enabling practical cost-benefit assessment for researchers planning large-scale meta-analyses.

5. **Scale of quantitative validation.** With 560 + 279 + 163 = 1,002 matched quantitative observations across 91 GT-matched papers and 3 independent datasets, our validation is substantially larger than prior quantitative extraction benchmarks. The cross-dataset validation (Loladze 2014 plant minerals; Hui 2023 zinc biofortification; Li 2022 biostimulant/crop yield) demonstrates domain-transferable capability via configuration changes alone, spanning both positive and negative effect directions.

6. **End-to-end automation.** Cao et al. (2025) demonstrated with otto-SR (o3-mini-high) that end-to-end LLM SR automation can achieve 93.1% data extraction accuracy compared to 79.7% for dual-human workflows, reproducing 12 Cochrane reviews in two days. While otto-SR targets structured clinical trial data, our system addresses the harder problem of heterogeneous quantitative data from plant science papers. The two approaches are complementary: otto-SR automates the entire SR pipeline while our system focuses specifically on high-fidelity quantitative extraction with formal equivalence validation.

---

## References

- Gartlehner, G. et al. (2024). Data extraction for evidence synthesis using a large language model: A proof-of-concept study. *Research Synthesis Methods*, 15(4), 576--589.
- Gougherty, A. V. & Clipp, H. L. (2024). Testing the reliability of an AI-based large language model to extract ecological information from the scientific literature. *npj Biodiversity*, 3(1), 13.
- Gartlehner, G. et al. (2025). Artificial intelligence-assisted data extraction with a large language model: A study within reviews. *Annals of Internal Medicine*. DOI: 10.7326/ANNALS-25-00739.
- Jensen, M. M. et al. (2025). ChatGPT-4o can serve as the second rater for data extraction in systematic reviews. *PLoS ONE*, 20(1), e0313401.
- Khan, M. A. et al. (2025). Collaborative large language models for automated data extraction in living systematic reviews. *JAMIA*, 32(4), 638--647.
- Li, X., Mathrani, A. & Susnjak, T. (2025). What level of automation is "good enough"? A benchmark of large language models for meta-analysis data extraction. arXiv:2507.15152.
- Sallam, M. et al. (2025). Automating the data extraction process for systematic reviews using GPT-4o and o3. *Research Synthesis Methods*. DOI: 10.1017/rsm.2025.10030.
- Poser, P. L. et al. (2026). Improving reliability and accuracy of structured data extraction using a consensus large-language model approach. *Frontiers in Artificial Intelligence*. DOI: 10.3389/frai.2026.1658575.
