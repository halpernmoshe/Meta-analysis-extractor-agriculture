# Table S1: CR1 vs CR2 Cluster-Robust TOST Equivalence Tests

Generated: 2026-03-16 20:03

CR1 = standard sandwich estimator with df = K - 1.
CR2 = bias-corrected sandwich estimator (Pustejovsky & Tipton, 2018) with Satterthwaite degrees of freedom.
Equivalence declared at alpha = 0.05.

## Panel A: Estimator Comparison

| Dataset | N | K | Mean diff (pp) | SE(naive) | SE(CR1) | SE(CR2) | CR2/CR1 | df(CR1) | df(CR2) | DEFF(CR1) | DEFF(CR2) |
|---------|---|---|----------------|-----------|---------|---------|---------|---------|---------|-----------|-----------|
| Loladze 2014 | 655 | 46 | 1.088 | 0.4127 | 0.6726 | 0.6813 | 1.013 | 45 | 8.1 | 2.66 | 2.73 |
| Hui 2023 | 461 | 30 | 0.273 | 0.7786 | 1.5748 | 1.5952 | 1.013 | 29 | 2.6 | 4.09 | 4.20 |
| Li 2022 | 68 | 16 | 0.222 | 0.4171 | 0.2910 | 0.2974 | 1.022 | 15 | 3.3 | 0.49 | 0.51 |

## Panel B: TOST Results by Margin

### Margin = +/-1 pp

| Dataset | CR1 p | CR1 decision | CR2 p | CR2 decision | 90% CI (CR1) | 90% CI (CR2) |
|---------|-------|--------------|-------|--------------|--------------|--------------|
| Loladze 2014 | 0.5515 | Not equiv. | 0.5496 | Not equiv. | [-0.04, 2.22] | [-0.18, 2.35] |
| Hui 2023 | 0.3239 | Not equiv. | 0.3418 | Not equiv. | [-2.40, 2.95] | [-3.70, 4.25] |
| Li 2022 | 0.0087 | Equivalent | 0.0363 | Equivalent | [-0.29, 0.73] | [-0.45, 0.90] |

### Margin = +/-2 pp

| Dataset | CR1 p | CR1 decision | CR2 p | CR2 decision | 90% CI (CR1) | 90% CI (CR2) |
|---------|-------|--------------|-------|--------------|--------------|--------------|
| Loladze 2014 | 0.0908 | Not equiv. | 0.1083 | Not equiv. | [-0.04, 2.22] | [-0.18, 2.35] |
| Hui 2023 | 0.1409 | Not equiv. | 0.1840 | Not equiv. | [-2.40, 2.95] | [-3.70, 4.25] |
| Li 2022 | 0.0000 | Equivalent | 0.0036 | Equivalent | [-0.29, 0.73] | [-0.45, 0.90] |

### Margin = +/-3 pp

| Dataset | CR1 p | CR1 decision | CR2 p | CR2 decision | 90% CI (CR1) | 90% CI (CR2) |
|---------|-------|--------------|-------|--------------|--------------|--------------|
| Loladze 2014 | 0.0033 | Equivalent | 0.0113 | Equivalent | [-0.04, 2.22] | [-0.18, 2.35] |
| Hui 2023 | 0.0470 | Equivalent | 0.0992 | Not equiv. | [-2.40, 2.95] | [-3.70, 4.25] |
| Li 2022 | 0.0000 | Equivalent | 0.0009 | Equivalent | [-0.29, 0.73] | [-0.45, 0.90] |

## Panel C: Cluster Balance Diagnostics

| Dataset | K | Min n_j | Max n_j | Median n_j | Mean n_j | CV(n_j) | Imbalance ratio |
|---------|---|---------|---------|------------|----------|---------|-----------------|
| Loladze 2014 | 46 | 1 | 60 | 10 | 14.2 | 0.83 | 60.0 |
| Hui 2023 | 30 | 1 | 104 | 7 | 15.4 | 1.44 | 104.0 |
| Li 2022 | 16 | 1 | 9 | 4 | 4.2 | 0.59 | 9.0 |

## Notes

1. CR2 corrects the downward bias of CR1 in small-sample settings (K < 40). The correction inflates the SE, producing more conservative p-values and wider confidence intervals.
2. Satterthwaite degrees of freedom account for unequal cluster sizes, unlike the fixed df = K - 1 used by CR1. When clusters are balanced, df(Satt) approaches K - 1.
3. The CR2/CR1 SE ratio indicates the magnitude of the bias correction. Values near 1.0 indicate minimal correction; values >> 1.0 indicate CR1 was substantially biased.
4. For Li 2022 (K = 16), the CR2 correction is most consequential because small-sample bias in CR1 is proportional to 1/K.
5. Imbalance ratio = max(n_j)/min(n_j). Higher values indicate more heterogeneous cluster sizes, which increases the importance of using CR2 + Satterthwaite df.
6. DEFF (design effect) = (SE_robust / SE_naive)^2. Values > 1 indicate positive intracluster correlation; observations within the same paper are not independent.

## Reference

Pustejovsky, J.E. & Tipton, E. (2018). Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. *Journal of Business & Economic Statistics*, 36(4), 672-683. https://doi.org/10.1080/07350015.2016.1247004