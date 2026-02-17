"""
Formal Statistical Analyses for Hui 2023 Dataset
=================================================
Computes Bland-Altman, TOST, ICC, bootstrap CIs for the Hui 2023 zinc/wheat validation.

Hui uses ln(response ratio) as effect size, so we compute percent change
from control/treatment means for consistency with the other datasets.

Run:
    python formal_stats_hui2023.py
"""
import sys, os, json, math, csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
MATCHES_CSV = BASE_DIR / "output" / "hui2023_v2" / "validation_hui2023_matches.csv"
OUT_DIR = BASE_DIR / "output" / "hui2023_formal_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_matches():
    """Load matched observations from Hui validation CSV.

    CSV columns: our_ctrl, our_treat, our_lnrr, gt_ctrl, gt_treat, gt_lnrr, gt_pub, match_qual
    We compute percent change from means for consistency.
    """
    rows = []
    with open(MATCHES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                our_ctrl = float(row['our_ctrl'])
                our_treat = float(row['our_treat'])
                gt_ctrl = float(row['gt_ctrl'])
                gt_treat = float(row['gt_treat'])

                if gt_ctrl <= 0 or our_ctrl <= 0:
                    continue

                gt_pct = (gt_treat - gt_ctrl) / gt_ctrl * 100
                our_pct = (our_treat - our_ctrl) / our_ctrl * 100

                rows.append({
                    'gt_effect_pct': gt_pct,
                    'ext_effect_pct': our_pct,
                    'gt_pub': row.get('gt_pub', ''),
                    'our_lnrr': float(row['our_lnrr']) if row.get('our_lnrr') else None,
                    'gt_lnrr': float(row['gt_lnrr']) if row.get('gt_lnrr') else None,
                })
            except (ValueError, KeyError):
                continue

    df = pd.DataFrame(rows)
    # Create paper_id from gt_pub (first author)
    df['paper_id'] = df['gt_pub'].apply(lambda x: x.split(',')[0].strip()[:30] if x else 'unknown')
    print(f"Loaded {len(df)} matched observations from {df['paper_id'].nunique()} papers")
    return df


def bland_altman(gt, ext):
    """Compute Bland-Altman limits of agreement."""
    diff = ext - gt
    mean_pair = (gt + ext) / 2

    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)
    n = len(diff)
    se_mean = sd_diff / np.sqrt(n)

    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff

    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = mean_diff - t_crit * se_mean
    ci_upper = mean_diff + t_crit * se_mean

    r_prop, p_prop = stats.pearsonr(mean_pair, diff)

    result = {
        "n": int(n),
        "mean_difference": round(float(mean_diff), 4),
        "sd_difference": round(float(sd_diff), 4),
        "ci_95_lower": round(float(ci_lower), 4),
        "ci_95_upper": round(float(ci_upper), 4),
        "loa_lower": round(float(loa_lower), 4),
        "loa_upper": round(float(loa_upper), 4),
        "proportional_bias_r": round(float(r_prop), 4),
        "proportional_bias_p": round(float(p_prop), 4),
    }

    print(f"\n=== Bland-Altman Analysis ===")
    print(f"  N: {n}")
    print(f"  Mean difference: {mean_diff:.2f} pp (95% CI: {ci_lower:.2f} to {ci_upper:.2f})")
    print(f"  95% LoA: {loa_lower:.2f} to {loa_upper:.2f}")
    print(f"  Proportional bias: r={r_prop:.3f}, p={p_prop:.4f}")

    return result


def tost_equivalence(gt, ext, margin=2.0):
    """Two One-Sided Tests for equivalence."""
    diff = ext - gt
    n = len(diff)
    mean_diff = np.mean(diff)
    se = np.std(diff, ddof=1) / np.sqrt(n)
    df = n - 1

    t_upper = (mean_diff - margin) / se
    p_upper = stats.t.cdf(t_upper, df)

    t_lower = (mean_diff + margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)

    t_90 = stats.t.ppf(0.95, df)
    ci90_lower = mean_diff - t_90 * se
    ci90_upper = mean_diff + t_90 * se

    equivalent = bool(p_tost < 0.05)

    result = {
        "margin_pp": margin,
        "mean_difference": round(float(mean_diff), 4),
        "se": round(float(se), 4),
        "t_upper": round(float(t_upper), 4),
        "t_lower": round(float(t_lower), 4),
        "p_upper": round(float(p_upper), 6),
        "p_lower": round(float(p_lower), 6),
        "p_tost": round(float(p_tost), 6),
        "ci90_lower": round(float(ci90_lower), 4),
        "ci90_upper": round(float(ci90_upper), 4),
        "equivalent": equivalent,
    }

    print(f"\n=== TOST Equivalence (margin=+/-{margin} pp) ===")
    print(f"  Mean diff: {mean_diff:.2f} pp, SE: {se:.2f}")
    print(f"  p(TOST) = {p_tost:.6f} {'*** EQUIVALENT ***' if equivalent else '(not equivalent)'}")
    print(f"  90% CI: ({ci90_lower:.2f}, {ci90_upper:.2f})")

    return result


def compute_icc(gt, ext):
    """Compute ICC(3,1) - two-way mixed, single measure, consistency."""
    n = len(gt)
    k = 2

    data = np.column_stack([gt, ext])
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    ss_rows = k * np.sum((row_means - grand_mean)**2)
    ss_cols = n * np.sum((col_means - grand_mean)**2)
    ss_total = np.sum((data - grand_mean)**2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)

    f_value = ms_rows / ms_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_value = 1 - stats.f.cdf(f_value, df1, df2)

    f_lower = f_value / stats.f.ppf(0.975, df1, df2)
    f_upper = f_value / stats.f.ppf(0.025, df1, df2)
    ci_lower = (f_lower - 1) / (f_lower + k - 1)
    ci_upper = (f_upper - 1) / (f_upper + k - 1)

    result = {
        "icc_31": round(float(icc), 4),
        "ci_95_lower": round(float(ci_lower), 4),
        "ci_95_upper": round(float(ci_upper), 4),
        "f_value": round(float(f_value), 4),
        "p_value": round(float(p_value), 6),
        "n": int(n),
    }

    print(f"\n=== ICC(3,1) ===")
    print(f"  ICC = {icc:.4f} (95% CI: {ci_lower:.4f} to {ci_upper:.4f})")
    print(f"  F = {f_value:.2f}, p = {p_value:.6f}")

    return result


def bootstrap_ci(gt, ext, n_boot=10000, seed=42):
    """Bootstrap BCa confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(gt)

    r_point = float(np.corrcoef(gt, ext)[0, 1])
    diff = np.abs(ext - gt)
    mae_point = float(np.mean(diff))

    dir_match = np.sum(np.sign(gt) == np.sign(ext))
    nonzero = np.sum(np.abs(gt) > 0.5)
    dir_point = float(dir_match / max(nonzero, 1))

    effect_diff_point = float(abs(np.mean(ext) - np.mean(gt)))
    within5_point = float(np.mean(diff <= 5))
    within10_point = float(np.mean(diff <= 10))

    boot_r, boot_mae, boot_dir = [], [], []
    boot_effect_diff, boot_within5, boot_within10 = [], [], []

    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        bg, be = gt[idx], ext[idx]

        try:
            boot_r.append(float(np.corrcoef(bg, be)[0, 1]))
        except:
            boot_r.append(np.nan)

        bd = np.abs(be - bg)
        boot_mae.append(float(np.mean(bd)))

        nonz = np.sum(np.abs(bg) > 0.5)
        dm = np.sum(np.sign(bg) == np.sign(be))
        boot_dir.append(float(dm / max(nonz, 1)))

        boot_effect_diff.append(float(abs(np.mean(be) - np.mean(bg))))
        boot_within5.append(float(np.mean(bd <= 5)))
        boot_within10.append(float(np.mean(bd <= 10)))

    def percentile_ci(boot_vals, alpha=0.05):
        boot_arr = np.array([v for v in boot_vals if not np.isnan(v)])
        if len(boot_arr) < 100:
            return (np.nan, np.nan)
        return (float(np.percentile(boot_arr, alpha/2 * 100)),
                float(np.percentile(boot_arr, (1 - alpha/2) * 100)))

    results = {}
    print(f"\n=== Bootstrap CIs (10,000 resamples) ===")

    for name, boot_vals, point in [
        ("pearson_r", boot_r, r_point),
        ("mae_pct", boot_mae, mae_point),
        ("direction_agreement", boot_dir, dir_point),
        ("overall_effect_diff_pp", boot_effect_diff, effect_diff_point),
        ("within_5pp", boot_within5, within5_point),
        ("within_10pp", boot_within10, within10_point),
    ]:
        ci = percentile_ci(boot_vals)
        results[name] = {
            "point_estimate": round(point, 4),
            "ci_95_lower": round(ci[0], 4),
            "ci_95_upper": round(ci[1], 4),
        }
        print(f"  {name}: {point:.4f} (95% CI: {ci[0]:.4f} to {ci[1]:.4f})")

    return results


def systematic_bias(gt, ext):
    """Test for systematic bias."""
    diff = ext - gt

    t_stat, p_ttest = stats.ttest_rel(ext, gt)

    try:
        w_stat, p_wilcox = stats.wilcoxon(diff)
    except:
        w_stat, p_wilcox = np.nan, np.nan

    d = float(np.mean(diff) / np.std(diff, ddof=1))

    result = {
        "paired_t": round(float(t_stat), 4),
        "p_ttest": round(float(p_ttest), 6),
        "wilcoxon_w": round(float(w_stat), 4) if not np.isnan(w_stat) else None,
        "p_wilcoxon": round(float(p_wilcox), 6) if not np.isnan(p_wilcox) else None,
        "cohens_d": round(float(d), 4),
    }

    print(f"\n=== Systematic Bias ===")
    print(f"  Paired t-test: t={t_stat:.3f}, p={p_ttest:.4f}")
    print(f"  Cohen's d: {d:.4f}")

    return result


def main():
    print("=" * 60)
    print("FORMAL STATISTICS - Hui 2023 (8 papers, zinc/wheat)")
    print("=" * 60)

    df = load_matches()

    gt_effects = df['gt_effect_pct'].values
    ext_effects = df['ext_effect_pct'].values

    print(f"\nBasic stats:")
    print(f"  N observations: {len(gt_effects)}")
    print(f"  N papers: {df['paper_id'].nunique()}")
    print(f"  GT mean effect: {np.mean(gt_effects):.2f}%")
    print(f"  Extracted mean effect: {np.mean(ext_effects):.2f}%")
    print(f"  Difference: {abs(np.mean(ext_effects) - np.mean(gt_effects)):.2f} pp")
    print(f"  Pearson r: {np.corrcoef(gt_effects, ext_effects)[0,1]:.3f}")
    print(f"  MAE: {np.mean(np.abs(ext_effects - gt_effects)):.2f}%")
    print(f"  Median AE: {np.median(np.abs(ext_effects - gt_effects)):.2f}%")

    # 1. Bland-Altman
    ba = bland_altman(gt_effects, ext_effects)
    with open(OUT_DIR / "bland_altman_results.json", 'w') as f:
        json.dump(ba, f, indent=2)

    # 2. TOST at multiple margins
    tost_results = {}
    for margin in [2, 3, 5, 10]:
        tost_results[f"margin_{margin}pp"] = tost_equivalence(gt_effects, ext_effects, margin=margin)
    with open(OUT_DIR / "tost_results.json", 'w') as f:
        json.dump(tost_results, f, indent=2)

    # 3. ICC
    icc = compute_icc(gt_effects, ext_effects)
    with open(OUT_DIR / "icc_results.json", 'w') as f:
        json.dump(icc, f, indent=2)

    # 4. Bootstrap CIs
    boot = bootstrap_ci(gt_effects, ext_effects)
    with open(OUT_DIR / "bootstrap_ci.json", 'w') as f:
        json.dump(boot, f, indent=2)

    # 5. Systematic bias
    bias = systematic_bias(gt_effects, ext_effects)
    with open(OUT_DIR / "systematic_bias.json", 'w') as f:
        json.dump(bias, f, indent=2)

    # 6. Paper-level ICC
    paper_gt = df.groupby('paper_id')['gt_effect_pct'].mean()
    paper_ext = df.groupby('paper_id')['ext_effect_pct'].mean()
    common = paper_gt.index.intersection(paper_ext.index)
    if len(common) >= 5:
        print(f"\n=== Paper-Level ICC ({len(common)} papers) ===")
        paper_icc = compute_icc(paper_gt[common].values, paper_ext[common].values)
        with open(OUT_DIR / "paper_level_icc.json", 'w') as f:
            json.dump(paper_icc, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"N observations: {len(gt_effects)}")
    print(f"N papers: {df['paper_id'].nunique()}")
    print(f"Pearson r: {np.corrcoef(gt_effects, ext_effects)[0,1]:.3f}")
    print(f"MAE: {np.mean(np.abs(ext_effects - gt_effects)):.2f}%")
    print(f"Median AE: {np.median(np.abs(ext_effects - gt_effects)):.2f}%")
    dir_match = np.mean(np.sign(gt_effects) == np.sign(ext_effects))
    print(f"Direction agreement: {dir_match*100:.1f}%")
    print(f"Overall effect diff: {abs(np.mean(ext_effects) - np.mean(gt_effects)):.2f} pp")
    print(f"Bland-Altman bias: {ba['mean_difference']:.2f} pp")
    for margin in [2, 3, 5]:
        key = f"margin_{margin}pp"
        eq = 'EQUIVALENT' if tost_results[key]['equivalent'] else 'NOT EQUIVALENT'
        print(f"TOST (+/-{margin}pp): p={tost_results[key]['p_tost']:.6f} {eq}")
    print(f"ICC(3,1): {icc['icc_31']:.3f}")
    print(f"Cohen's d: {bias['cohens_d']:.4f}")

    print(f"\nAll results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
