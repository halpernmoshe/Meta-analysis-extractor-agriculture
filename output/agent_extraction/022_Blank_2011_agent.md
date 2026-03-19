# 022_Blank_2011 - Agent Extraction

## Paper Details
- **Title**: Effect of Atmospheric CO2 Levels on Nutrients in Cheatgrass Tissue
- **Authors**: Blank, Morgan, Ziska, White (2011)
- **Species**: Bromus tectorum (cheatgrass)
- **Experiment**: Growth chamber, Beltsville MD (USDA-ARS)
- **CO2 levels**: 270, 320, 370, 420 ppmv
- **Ecotypes**: Low (1120 m, salt desert), Mid (1585 m, sagebrush steppe), High (2170 m, mountain brush)
- **Harvest times**: Day 42, 57, 75, 87
- **Tissue**: Aboveground
- **Variance**: None reported (Tukey HSD significance letters only)
- **n**: 5 replications per treatment combination (4 CO2 x 2 bays x 3 ecotypes x 5 reps = 120 total, but CO2 x Age data pooled over ecotype)

## Ground Truth Match (Loladze 2014)
Loladze uses 420 vs 270 ppmv, Day 87, pooled over ecotype. 6 elements:

| Element | GT Effect | Extracted Effect | Match |
|---------|-----------|-----------------|-------|
| Ca | -8.13% | -8.13% | YES |
| K | -16.03% | -16.03% | YES |
| Mg | -30.43% | -30.43% | YES |
| Mn | +5.05% | +5.05% | YES |
| N | -12.5% | -12.5% | YES |
| P | -29.17% | -29.17% | YES |

## Extracted Data (420 vs 270 ppmv, all harvest times)

### Table 2 - Tissue N (%)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 5.0 | 5.4 | +8.0 |
| Day 57 | 4.7 | 4.2 | -10.64 |
| Day 75 | 4.3 | 4.3 | 0.0 |
| **Day 87** | **4.8** | **4.2** | **-12.5** |

### Table 2 - Tissue C (%)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 34.6 | 34.4 | -0.58 |
| Day 57 | 34.7 | 34.0 | -2.02 |
| Day 75 | 34.4 | 35.2 | +2.33 |
| **Day 87** | **31.2** | **33.6** | **+7.69** |

### Table 3 - Tissue P (mol/kg)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 0.19 | 0.17 | -10.53 |
| Day 57 | 0.17 | 0.16 | -11.76 (approx) |
| Day 75 | 0.18 | 0.16 | -22.22 (approx) |
| **Day 87** | **0.24** | **0.17** | **-29.17** |

### Table 3 - Tissue K (mol/kg)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 1.10 | 1.13 | +2.73 |
| Day 57 | 1.16 | 1.15 | -0.86 |
| Day 75 | 1.28 | 1.21 | -5.47 |
| **Day 87** | **1.56** | **1.31** | **-16.03** |

### Table 3 - Tissue Mg (mol/kg)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 0.17 | 0.17 | 0.0 |
| Day 57 | 0.20 | 0.15 | -25.0 |
| Day 75 | 0.20 | 0.16 | -20.0 |
| **Day 87** | **0.23** | **0.16** | **-30.43** |

### Table 3 - Tissue Ca (mmol/kg)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 50.9 | 36.7 | -27.90 |
| Day 57 | 76.9 | 49.3 | -35.89 |
| Day 75 | 68.2 | 54.4 | -20.23 |
| **Day 87** | **56.6** | **52.0** | **-8.13** |

### Table 3 - Tissue Mn (mmol/kg)
| Harvest | Control (270) | Treatment (420) | Effect (%) |
|---------|---------------|-----------------|------------|
| Day 42 | 1.93 | 2.39 | +23.83 |
| Day 57 | 2.20 | 2.22 | +0.91 |
| Day 75 | 2.57 | 3.04 | +18.29 |
| **Day 87** | **3.17** | **3.33** | **+5.05** |

## Elements Not in Tables
- **Na**: Measured (mentioned in Methods) but not reported in any table. Abstract mentions Na increases with plant age.
- **Fe**: Measured via atomic absorption spectrophotometry (Methods) but no concentration data in tables.

## Notes
- Bold rows are the Day 87 values that match Loladze ground truth.
- Data in CO2 x Age interaction panels are pooled over the 3 ecotypes.
- Table 4 contains hemicellulose, ADF, K-lignin, glucan, mannan for Day 87 only - these are biochemical composition data, not mineral elements.
- Plants were supplied luxury N levels (14.5 mM), which authors note may have masked CO2 dilution effects on N.
- The paper also has Ecotype x Age interaction data (pooled over CO2) which is not extracted here since it cannot be used for CO2 effect calculation.
- P values at Days 42, 57 read from OCR'd text may have minor rounding variations.
