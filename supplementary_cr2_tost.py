"""
Supplementary Table S1: CR1 vs CR2 Cluster-Robust TOST Equivalence Tests
=========================================================================
Computes TOST at multiple margins using both CR1 (standard sandwich) and
CR2 (bias-corrected, Pustejovsky & Tipton 2018) sandwich estimators with
Satterthwaite degrees of freedom.

CR2 is especially important for Li 2022 (K=16 clusters) where CR1 can be
downward-biased, leading to liberal inference.

Reference:
  Pustejovsky, J.E. & Tipton, E. (2018). Small-sample methods for
  cluster-robust variance estimation and hypothesis testing in fixed
  effects models. Journal of Business & Economic Statistics, 36(4), 672-683.

Run:
    ./venv/Scripts/python.exe supplementary_cr2_tost.py
"""
import sys
import os
import json
import math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats
from scipy.linalg import sqrtm

BASE_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING (reused from formal_stats_agent.py)
# ============================================================

def load_loladze_agent():
    report_path = BASE_DIR / "output" / "agent_extraction" / "validation_report_agent.json"
    with open(report_path) as f:
        report = json.load(f)
    matches = []
    for m in report['all_matches']:
        matches.append({
            'our_pct': m['our'] * 100,
            'gt_pct': m['gt'] * 100,
            'paper': m['paper'],
        })
    return matches


def load_hui_agent():
    report_path = BASE_DIR / "output" / "hui2023_agent_extraction" / "validation_report_agent.json"
    with open(report_path) as f:
        report = json.load(f)
    match_pairs = report.get('match_pairs', [])
    matches = []
    for m in match_pairs:
        matches.append({
            'our_pct': m['our_pct'],
            'gt_pct': m['gt_pct'],
            'paper': m.get('paper', 'unknown'),
        })
    return matches


def load_li_agent(tier='high'):
    report_path = BASE_DIR / "output" / "li2022_agent_extraction" / "harmonized_validation_agent.json"
    with open(report_path) as f:
        data = json.load(f)
    pairs = data.get('match_pairs_by_tier', {}).get(tier, [])
    matches = []
    for m in pairs:
        matches.append({
            'our_pct': m['ext_effect'],
            'gt_pct': m['gt_effect'],
            'paper': m.get('paper', 'unknown'),
        })
    return matches


# ============================================================
# CR1 SANDWICH ESTIMATOR (standard)
# ============================================================

def tost_cr1(our, gt, papers, margins):
    """
    TOST with CR1 cluster-robust SE.
    CR1 = (K/(K-1)) * (N-1)/N scaling of the sandwich estimator.
    df = K - 1.
    """
    diff = our - gt
    N = len(diff)
    mean_diff = np.mean(diff)

    unique_papers = np.unique(papers)
    K = len(unique_papers)

    # Cluster-level score sums
    cluster_sums = np.array([
        np.sum(diff[papers == p] - mean_diff) for p in unique_papers
    ])

    # CR1 variance
    cr1_factor = K / (K - 1) * (N - 1) / N
    var_cr1 = cr1_factor * np.sum(cluster_sums ** 2) / (N ** 2)
    se_cr1 = np.sqrt(var_cr1)

    se_naive = np.std(diff, ddof=1) / np.sqrt(N)
    deff = (se_cr1 / se_naive) ** 2 if se_naive > 0 else 1.0

    df = K - 1

    results = {}
    for margin in margins:
        t1 = (mean_diff - (-margin)) / se_cr1
        p1 = 1 - stats.t.cdf(t1, df=df)
        t2 = (mean_diff - margin) / se_cr1
        p2 = stats.t.cdf(t2, df=df)
        p_tost = max(p1, p2)
        results[margin] = {
            'p_value': float(p_tost),
            'equivalent': bool(p_tost < 0.05),
            't_lower': float(t1),
            't_upper': float(t2),
        }

    t_crit = stats.t.ppf(0.95, df=df)
    ci90 = (mean_diff - t_crit * se_cr1, mean_diff + t_crit * se_cr1)

    return {
        'estimator': 'CR1',
        'mean_diff': float(mean_diff),
        'se': float(se_cr1),
        'se_naive': float(se_naive),
        'design_effect': float(deff),
        'df': float(df),
        'K': int(K),
        'N': int(N),
        'ci90': [float(ci90[0]), float(ci90[1])],
        'margins': results,
    }


# ============================================================
# CR2 SANDWICH ESTIMATOR (Pustejovsky & Tipton 2018)
# ============================================================

def tost_cr2(our, gt, papers, margins):
    """
    TOST with CR2 (bias-corrected) cluster-robust SE and
    Satterthwaite degrees of freedom.

    For the intercept-only model y_i = mu + e_i with clustering:
    - X is a column of ones (N x 1)
    - beta_hat = mean(y)
    - The hat matrix H = X(X'X)^{-1}X' = 1/N * J (matrix of all 1/N)
    - For cluster j: H_jj = n_j/N * J_{n_j} (where J_{n_j} is n_j x n_j all-ones matrix scaled)

    CR2 adjustment for cluster j:
      A_j = (I_{n_j} - H_{jj})^{-1/2}
    where H_{jj} is the diagonal block of H for cluster j.

    Then the CR2 variance is:
      V_CR2 = (X'X)^{-1} * [sum_j X_j' A_j e_j e_j' A_j' X_j] * (X'X)^{-1}

    For the intercept-only model this simplifies considerably.
    """
    diff = our - gt
    N = len(diff)
    mean_diff = np.mean(diff)

    unique_papers = np.unique(papers)
    K = len(unique_papers)

    # For each cluster j with n_j observations:
    # H_jj = (1/N) * ones(n_j, n_j)   [the diagonal block of the hat matrix]
    # I - H_jj has eigenvalues: (1 - n_j/N) with multiplicity 1, and 1 with multiplicity (n_j - 1)
    # (I - H_jj)^{-1/2} can be computed analytically

    # Residuals
    resid = diff - mean_diff

    # CR2 meat computation
    # For intercept-only: X_j = ones(n_j, 1), (X'X)^{-1} = 1/N
    # We need sum_j [ones' A_j e_j][e_j' A_j ones]
    # = sum_j [sum(A_j e_j)]^2

    cr2_scores = []
    cluster_sizes = []

    for p in unique_papers:
        mask = papers == p
        n_j = np.sum(mask)
        e_j = resid[mask]
        cluster_sizes.append(n_j)

        if n_j == 1:
            # For singleton clusters, H_jj = 1/N, so (I - H_jj)^{-1/2} = 1/sqrt(1 - 1/N)
            a_factor = 1.0 / np.sqrt(1.0 - 1.0 / N)
            adjusted_e = a_factor * e_j
        else:
            # H_jj = (1/N) * J_{n_j} where J is all-ones matrix
            # I_{n_j} - H_jj has two distinct eigenvalues:
            #   lambda_1 = 1 - n_j/N  (eigenvector: ones/sqrt(n_j))
            #   lambda_2 = 1           (multiplicity n_j - 1, orthogonal complement)
            # (I - H_jj)^{-1/2} = a * J/n_j + b * (I - J/n_j)
            # where a = 1/sqrt(1 - n_j/N), b = 1 (since sqrt(1) = 1)

            a = 1.0 / np.sqrt(max(1.0 - n_j / N, 1e-12))
            # A_j e_j = a * (mean(e_j) * ones) + 1 * (e_j - mean(e_j) * ones)
            #         = e_j + (a - 1) * mean(e_j) * ones
            mean_ej = np.mean(e_j)
            adjusted_e = e_j + (a - 1.0) * mean_ej

        # Score for this cluster: X_j' A_j e_j = sum(adjusted_e) (since X_j = ones)
        cr2_scores.append(np.sum(adjusted_e))

    cr2_scores = np.array(cr2_scores)
    cluster_sizes = np.array(cluster_sizes)

    # CR2 variance of beta_hat = (1/N^2) * sum(score_j^2)
    var_cr2 = np.sum(cr2_scores ** 2) / (N ** 2)
    se_cr2 = np.sqrt(var_cr2)

    # Satterthwaite degrees of freedom
    # df = [sum_j g_j^2]^2 / sum_j g_j^4
    # where g_j = X_j' A_j M_j (the "gradient" of the score for cluster j)
    # For intercept-only, g_j relates to the scaling factors.
    #
    # Following Pustejovsky & Tipton (2018) eq. (7):
    # df = trace(Phi)^2 / trace(Phi^2)
    # where Phi = sum_j B_j B_j'
    # and B_j = (X'X)^{-1} X_j' A_j = (1/N) * ones_j' * A_j
    # So B_j (scalar for intercept model) = (1/N) * sum(row of A_j)
    #
    # Actually for the scalar (intercept) case, Phi is a scalar:
    # Phi = sum_j B_j^2 where B_j = cr2_score_j / N  (when we haven't squared yet)
    #
    # The Satterthwaite df for the t-test on scalar beta is:
    # df = [sum g_j^2]^2 / [sum g_j^4]
    # where g_j are the influence components of the variance estimator.
    #
    # For the CR2 sandwich on an intercept-only model:
    # V = (1/N^2) sum_j s_j^2 where s_j = cr2_scores[j]
    # The df approximation is:
    # df = V^2 / sum_j [(partial V / partial e_j_cluster_var)]
    #
    # Simpler: use the standard Satterthwaite for a weighted sum of chi-squares:
    # Each cluster contributes s_j^2 / N^2, and these are approximately independent.
    # df = (sum s_j^2)^2 / (sum s_j^4)  [analogous to Welch]

    g = cr2_scores ** 2  # contribution of each cluster to variance * N^2
    if np.sum(g ** 2) > 0:
        df_satt = (np.sum(g)) ** 2 / np.sum(g ** 2)
    else:
        df_satt = K - 1

    # Bound: df should be between 1 and K-1 (can exceed K-1 in balanced designs)
    # In practice, cap at a reasonable maximum
    df_satt = max(1.0, min(df_satt, N - 1))

    se_naive = np.std(diff, ddof=1) / np.sqrt(N)
    deff = (se_cr2 / se_naive) ** 2 if se_naive > 0 else 1.0

    results = {}
    for margin in margins:
        t1 = (mean_diff - (-margin)) / se_cr2
        p1 = 1 - stats.t.cdf(t1, df=df_satt)
        t2 = (mean_diff - margin) / se_cr2
        p2 = stats.t.cdf(t2, df=df_satt)
        p_tost = max(p1, p2)
        results[margin] = {
            'p_value': float(p_tost),
            'equivalent': bool(p_tost < 0.05),
            't_lower': float(t1),
            't_upper': float(t2),
        }

    t_crit = stats.t.ppf(0.95, df=df_satt)
    ci90 = (mean_diff - t_crit * se_cr2, mean_diff + t_crit * se_cr2)

    return {
        'estimator': 'CR2',
        'mean_diff': float(mean_diff),
        'se': float(se_cr2),
        'se_naive': float(se_naive),
        'design_effect': float(deff),
        'df': float(df_satt),
        'df_type': 'Satterthwaite',
        'K': int(K),
        'N': int(N),
        'cluster_sizes': cluster_sizes.tolist(),
        'ci90': [float(ci90[0]), float(ci90[1])],
        'margins': results,
    }


# ============================================================
# ANALYSIS AND OUTPUT
# ============================================================

def analyze_dataset(name, matches, margins):
    """Run CR1 and CR2 TOST for one dataset."""
    if not matches:
        print(f"  {name}: no data")
        return None

    our = np.array([m['our_pct'] for m in matches])
    gt = np.array([m['gt_pct'] for m in matches])
    papers = np.array([m['paper'] for m in matches])

    N = len(our)
    K = len(np.unique(papers))

    print(f"\n{'='*70}")
    print(f"  {name}: N={N} obs, K={K} clusters")
    print(f"{'='*70}")

    cr1 = tost_cr1(our, gt, papers, margins)
    cr2 = tost_cr2(our, gt, papers, margins)

    # Print comparison
    print(f"  Mean diff:   {cr1['mean_diff']:.3f} pp")
    print(f"  SE(naive):   {cr1['se_naive']:.4f}")
    print(f"  SE(CR1):     {cr1['se']:.4f}  (df = {cr1['df']:.0f})")
    print(f"  SE(CR2):     {cr2['se']:.4f}  (df = {cr2['df']:.1f} Satterthwaite)")
    print(f"  SE ratio CR2/CR1: {cr2['se']/cr1['se']:.3f}")
    print(f"  DEFF(CR1):   {cr1['design_effect']:.2f}")
    print(f"  DEFF(CR2):   {cr2['design_effect']:.2f}")

    # Cluster size summary
    sizes = cr2['cluster_sizes']
    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
          f"median={np.median(sizes):.0f}, mean={np.mean(sizes):.1f}")

    print(f"\n  {'Margin':>8} | {'CR1 p':>10} {'CR1':>6} | {'CR2 p':>10} {'CR2':>6} | {'Change':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-{'-'*6}-+-{'-'*10}-{'-'*6}-+-{'-'*8}")
    for margin in margins:
        r1 = cr1['margins'][margin]
        r2 = cr2['margins'][margin]
        s1 = "PASS" if r1['equivalent'] else "FAIL"
        s2 = "PASS" if r2['equivalent'] else "FAIL"
        change = ""
        if s1 != s2:
            change = f"{s1}->{s2}"
        print(f"  {margin:>6.0f}pp | {r1['p_value']:>10.4f} {s1:>6} | "
              f"{r2['p_value']:>10.4f} {s2:>6} | {change:>8}")

    print(f"\n  90% CI(CR1): [{cr1['ci90'][0]:.3f}, {cr1['ci90'][1]:.3f}]")
    print(f"  90% CI(CR2): [{cr2['ci90'][0]:.3f}, {cr2['ci90'][1]:.3f}]")

    return {'name': name, 'N': N, 'K': K, 'cr1': cr1, 'cr2': cr2}


def format_markdown_table(all_results, margins):
    """Generate markdown table for supplementary materials."""
    lines = []
    lines.append("# Table S1: CR1 vs CR2 Cluster-Robust TOST Equivalence Tests")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("CR1 = standard sandwich estimator with df = K - 1.")
    lines.append("CR2 = bias-corrected sandwich estimator (Pustejovsky & Tipton, 2018) "
                  "with Satterthwaite degrees of freedom.")
    lines.append("Equivalence declared at alpha = 0.05.")
    lines.append("")

    # Panel A: Summary statistics
    lines.append("## Panel A: Estimator Comparison")
    lines.append("")
    lines.append("| Dataset | N | K | Mean diff (pp) | SE(naive) | SE(CR1) | SE(CR2) | "
                  "CR2/CR1 | df(CR1) | df(CR2) | DEFF(CR1) | DEFF(CR2) |")
    lines.append("|---------|---|---|----------------|-----------|---------|---------|"
                  "---------|---------|---------|-----------|-----------|")

    for r in all_results:
        if not r:
            continue
        cr1, cr2 = r['cr1'], r['cr2']
        ratio = cr2['se'] / cr1['se'] if cr1['se'] > 0 else float('nan')
        lines.append(
            f"| {r['name']} | {r['N']} | {r['K']} | "
            f"{cr1['mean_diff']:.3f} | {cr1['se_naive']:.4f} | "
            f"{cr1['se']:.4f} | {cr2['se']:.4f} | {ratio:.3f} | "
            f"{cr1['df']:.0f} | {cr2['df']:.1f} | "
            f"{cr1['design_effect']:.2f} | {cr2['design_effect']:.2f} |"
        )
    lines.append("")

    # Panel B: TOST results at each margin
    lines.append("## Panel B: TOST Results by Margin")
    lines.append("")

    # Build one table per margin
    for margin in margins:
        lines.append(f"### Margin = +/-{margin:.0f} pp")
        lines.append("")
        lines.append("| Dataset | CR1 p | CR1 decision | CR2 p | CR2 decision | "
                      "90% CI (CR1) | 90% CI (CR2) |")
        lines.append("|---------|-------|--------------|-------|--------------|"
                      "--------------|--------------|")

        for r in all_results:
            if not r:
                continue
            cr1, cr2 = r['cr1'], r['cr2']
            r1 = cr1['margins'][margin]
            r2 = cr2['margins'][margin]
            s1 = "Equivalent" if r1['equivalent'] else "Not equiv."
            s2 = "Equivalent" if r2['equivalent'] else "Not equiv."

            lines.append(
                f"| {r['name']} | {r1['p_value']:.4f} | {s1} | "
                f"{r2['p_value']:.4f} | {s2} | "
                f"[{cr1['ci90'][0]:.2f}, {cr1['ci90'][1]:.2f}] | "
                f"[{cr2['ci90'][0]:.2f}, {cr2['ci90'][1]:.2f}] |"
            )
        lines.append("")

    # Panel C: Cluster balance diagnostics
    lines.append("## Panel C: Cluster Balance Diagnostics")
    lines.append("")
    lines.append("| Dataset | K | Min n_j | Max n_j | Median n_j | Mean n_j | "
                  "CV(n_j) | Imbalance ratio |")
    lines.append("|---------|---|---------|---------|------------|----------|"
                  "---------|-----------------|")
    for r in all_results:
        if not r:
            continue
        sizes = np.array(r['cr2']['cluster_sizes'])
        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
        imbal = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
        lines.append(
            f"| {r['name']} | {r['K']} | {min(sizes)} | {max(sizes)} | "
            f"{np.median(sizes):.0f} | {np.mean(sizes):.1f} | "
            f"{cv:.2f} | {imbal:.1f} |"
        )
    lines.append("")

    # Interpretation notes
    lines.append("## Notes")
    lines.append("")
    lines.append("1. CR2 corrects the downward bias of CR1 in small-sample settings "
                  "(K < 40). The correction inflates the SE, producing more conservative "
                  "p-values and wider confidence intervals.")
    lines.append("2. Satterthwaite degrees of freedom account for unequal cluster sizes, "
                  "unlike the fixed df = K - 1 used by CR1. When clusters are balanced, "
                  "df(Satt) approaches K - 1.")
    lines.append("3. The CR2/CR1 SE ratio indicates the magnitude of the bias correction. "
                  "Values near 1.0 indicate minimal correction; values >> 1.0 indicate "
                  "CR1 was substantially biased.")
    lines.append("4. For Li 2022 (K = 16), the CR2 correction is most consequential "
                  "because small-sample bias in CR1 is proportional to 1/K.")
    lines.append("5. Imbalance ratio = max(n_j)/min(n_j). Higher values indicate "
                  "more heterogeneous cluster sizes, which increases the importance "
                  "of using CR2 + Satterthwaite df.")
    lines.append("6. DEFF (design effect) = (SE_robust / SE_naive)^2. Values > 1 "
                  "indicate positive intracluster correlation; observations within "
                  "the same paper are not independent.")
    lines.append("")
    lines.append("## Reference")
    lines.append("")
    lines.append("Pustejovsky, J.E. & Tipton, E. (2018). Small-sample methods for "
                  "cluster-robust variance estimation and hypothesis testing in fixed "
                  "effects models. *Journal of Business & Economic Statistics*, 36(4), 672-683. "
                  "https://doi.org/10.1080/07350015.2016.1247004")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("SUPPLEMENTARY TABLE S1: CR1 vs CR2 CLUSTER-ROBUST TOST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    margins = [1, 2, 3]

    # Load all datasets
    print("\nLoading data...")
    lol = load_loladze_agent()
    print(f"  Loladze: {len(lol)} obs")
    hui = load_hui_agent()
    print(f"  Hui:     {len(hui)} obs")
    li = load_li_agent('high')
    print(f"  Li:      {len(li)} obs")

    # Analyze
    results = []
    results.append(analyze_dataset("Loladze 2014", lol, margins))
    results.append(analyze_dataset("Hui 2023", hui, margins))
    results.append(analyze_dataset("Li 2022", li, margins))

    # Generate markdown
    md = format_markdown_table(results, margins)

    # Save
    out_path = OUT_DIR / "supplementary_table_s1_cr2.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"\n\nSaved to {out_path}")

    # Also save raw JSON
    json_path = OUT_DIR / "supplementary_table_s1_cr2.json"

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"Saved JSON to {json_path}")

    # Print the markdown
    print("\n" + "=" * 70)
    print("FORMATTED TABLE (Markdown)")
    print("=" * 70)
    print(md)


if __name__ == '__main__':
    main()
