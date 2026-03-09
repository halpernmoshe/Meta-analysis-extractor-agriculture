# Submission Checklist - PAPER_FINAL_v3.md

## Status: Ready for user review before submission

### Completed (v2 → v3 changes)
- [x] All statistics verified against source CSV and JSON data files
- [x] Cohen's d values corrected throughout (Loladze -0.003, Hui 0.072, Li all -0.189)
- [x] Hui improved matching stats used consistently (r=0.999, MAE=0.43pp, direction=99.7%)
- [x] Li improved matching stats used consistently (r=0.951, MAE=2.30pp)
- [x] All cross-references verified
- [x] All 15+ citations verified as real published papers
- [x] Time reduction consistently 70-75%
- [x] Model version inconsistency resolved (Sonnet 4 extraction, Sonnet 4.6 auditing)
- [x] Three-Barrier framework reframed as barriers to PROVING extraction works
- [x] Li paired t-test significance (p=0.008) honestly reported with context
- [x] Bland-Altman LOA updated to improved values (Hui ±3.2pp, Li ±9pp)
- [x] Cover letter updated with correct numbers

### v3 additions (circularity-breaking rewrite)
- [x] **Programmatic GT classifier** created (`programmatic_gt_classifier.py`)
- [x] **Section 2.6.1** added: Programmatic Confidence Classification methodology
- [x] **Abstract** updated: mentions programmatic classifier and circularity-breaking
- [x] **Introduction** updated: second epistemic challenge (circularity) foreshadowed
- [x] **Section 2.5** updated: LLM audit reframed as corroborative, not constitutive
- [x] **Table 1** updated: Li Aligned → Li Prog. High (18 papers, 110 obs, r=0.999, MAE=0.32)
- [x] **Section 3.6** rewritten: programmatic classification primary, LLM corroborative
- [x] **Section 3.7** rewritten: Table 5 (programmatic tiers), LLM audit as 3.7.2
- [x] **Section 3.8** updated: programmatic high d=-0.055, diff=0.04pp
- [x] **Table 2** updated: Li prog. high numbers
- [x] **Section 4.3** updated: Provenance Barrier addresses circularity explicitly
- [x] **Section 4.4** updated: proposes programmatic classification as validation standard
- [x] **Section 5 (Conclusion)** rewritten: four conclusions, circularity-breaking is #4
- [x] **Limitation 9b** added: residual circularity in qualitative claims
- [x] **Cover letter** updated with circularity-breaking contribution
- [x] **All programmatic stats verified**: r=0.999, MAE=0.32, direction=99.1%, d=-0.055, zero-error=59.1%

### User action required before submission
- [ ] Fill in ORCID (line 5: `^ORCID^`)
- [ ] Verify Hui et al. citation year (2023 vs potentially 2025 - Nature Comms?)
- [ ] Create/update GitHub repository URL (currently placeholder)
- [ ] Create/update Zenodo DOI (currently placeholder)
- [ ] Convert MD to DOCX (pandoc not installed on this machine)
- [ ] Review word count (~16,650 words including tables/refs; RSM may want ~10,000)
- [ ] Consider generating actual figure files from data

### Key numbers for quick reference
| Dataset | N obs | N papers | r | MAE (pp) | Direction | Effect diff | Cohen's d |
|---------|-------|----------|---|----------|-----------|-------------|-----------|
| Loladze | 635 | 46 | 0.669 | 7.9 | 85% | 0.05 | -0.003 |
| Hui (improved) | 319 | 21 | 0.999 | 0.43 | 99.7% | 0.12 | 0.072 |
| Li (harmonized) | 200 | 27 | 0.951 | 2.30 | 93% | 0.84 | -0.189 |
| **Li (prog. high)** | **110** | **18** | **0.999** | **0.32** | **99.1%** | **0.04** | **-0.055** |
| **Total** | **1,154** | **94** | | | | | |

### Programmatic classifier concordance with LLM audit
- Agreement: 23/27 papers (85%)
- Programmatic more inclusive: 3 papers (all MAE < 1pp, defensibly correct)
- Programmatic more conservative: 1 paper (MAE = 3.1pp, zero exact matches)
