# 048_Khan_2013 - Agent Extraction

## Paper Details
- **Title**: The impact of enhanced atmospheric carbon dioxide on yield, proximate composition, elemental concentration, fatty acid and vitamin C contents of tomato (Lycopersicon esculentum)
- **Authors**: Khan, Azam, Mahmood
- **Year**: 2013
- **Journal**: Environmental Monitoring and Assessment, 185:205-214
- **Species**: Lycopersicon esculentum (tomato)
- **Varieties**: Astra, Eureka
- **CO2 levels**: 400 (ambient) vs 1000 (elevated) umol/mol
- **Replicates**: n=3 (triplicate)
- **Tissue**: Fruit (dried, mature stage unless noted)
- **Growth**: Greenhouse, pots, Chakwal Pakistan

## Data Sources
- **Table 4**: Elemental composition (macro-elements in %, trace elements in ug/g) - Astra and Eureka, mature fruit
- **Table 3**: Proximate composition including protein and vitamin C - Astra mature, Eureka mature, Eureka premature

## Notes on Ca/Mg Values
- Ca values in Table 4 show 0.13 vs 0.14 for Astra (paper reports 3.85% change). The exact values to 3+ decimals would be needed for exact replication. Using 2-decimal values: (0.14-0.13)/0.13*100 = 7.69%. The paper's 3.85% suggests finer precision (e.g., 0.1300 vs 0.1350).
- Mg for Astra shows 0.17 vs 0.17 but paper reports -5.48%. Finer precision exists (e.g., 0.1736 vs 0.1641).
- I use the paper's reported % change values as the authoritative effect_pct where rounding artifacts exist.

## Extracted Observations (30 total - mineral/elemental only)

### Table 4 - Macro-elements (%)

| Element | Variety | Control (400) | Treatment (1000) | Effect % | P-value |
|---------|---------|--------------|------------------|----------|---------|
| C | Astra | 31.66 | 44.41 | +40.27 | 0.002 |
| C | Eureka | 32.71 | 43.55 | +33.14 | 0.000 |
| N | Astra | 2.46 | 2.01 | -18.29 | 0.026 |
| N | Eureka | 1.96 | 1.69 | -13.78 | 0.004 |
| H | Astra | 4.32 | 5.52 | +27.78 | 0.002 |
| H | Eureka | 4.77 | 5.74 | +20.33 | 0.000 |
| S | Astra | 0.37 | 0.39 | +5.41 | ns |
| S | Eureka | 0.53 | 0.57 | +7.55 | ns |
| Ca | Astra | 0.13 | 0.14 | +3.85* | 0.000 |
| Ca | Eureka | 0.14 | 0.15 | +4.81* | 0.000 |
| Mg | Astra | 0.17 | 0.17 | -5.48* | 0.014 |
| Mg | Eureka | 0.18 | 0.14 | -22.22 | 0.001 |
| K | Astra | 0.46 | 0.46 | -0.43 | ns |
| K | Eureka | 0.46 | 0.46 | -0.22 | ns |

*Paper reports different % due to higher precision values than displayed in table

### Table 4 - Trace elements (ug/g)

| Element | Variety | Control (400) | Treatment (1000) | Effect % | P-value |
|---------|---------|--------------|------------------|----------|---------|
| Zn | Astra | 196.27 | 140.60 | -28.36 | 0.000 |
| Zn | Eureka | 154.07 | 132.47 | -14.02 | 0.000 |
| Mn | Astra | 431.53 | 398.00 | -7.77 | 0.000 |
| Mn | Eureka | 430.00 | 411.80 | -4.23 | 0.000 |
| Fe | Astra | 373.60 | 384.93 | +3.03 | 0.000 |
| Fe | Eureka | 343.40 | 388.60 | +13.16 | 0.000 |
| Cu | Astra | 30.33 | 35.73 | +17.80 | 0.000 |
| Cu | Eureka | 27.80 | 35.07 | +26.15 | 0.000 |
| Pb | Astra | 61.60 | 42.33 | -31.28 | 0.018 |
| Pb | Eureka | 59.27 | 46.27 | -21.93 | 0.029 |
| Ni | Astra | 50.40 | 19.73 | -60.85 | 0.000 |
| Ni | Eureka | 50.27 | 24.60 | -51.06 | 0.000 |
| Cr | Astra | 22.73 | 17.87 | -21.38 | 0.031 |
| Cr | Eureka | 25.53 | 18.20 | -28.71 | 0.001 |
| Cd | Astra | 27.00 | 24.47 | -9.37 | 0.010 |
| Cd | Eureka | 26.73 | 25.13 | -5.99 | 0.001 |

## Summary
- **30 elemental observations** extracted from Table 4 (2 varieties x 15 elements)
- Elements measured: C, N, H, S, Ca, Mg, K, Zn, Mn, Fe, Pb, Ni, Cu, Cr, Cd
- Most elements decreased under elevated CO2 (N, Mg, K, Zn, Mn, Pb, Ni, Cr, Cd)
- Some elements increased: C, H, S, Ca, Fe, Cu
- Eureka premature data available in Table 3 (vitamin C, protein) but not in Table 4 (no elemental data for premature stage)
- Fatty acid data in Table 5 not extracted (not mineral/element concentration)
- Heavy metals (Pb, Ni, Cr, Cd) included as they are reported in the elemental analysis
