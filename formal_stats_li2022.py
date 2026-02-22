"""
Formal Statistical Analyses for Li 2022 Dataset
=================================================
Computes Bland-Altman, TOST, ICC, bootstrap CIs for the Li 2022 validation.

Run:
    python formal_stats_li2022.py
"""
import sys, os, json, math
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
# Full-dataset stats (n=163, naive matching) — used for Table 1, Section 3.10 TOST/BA text
MATCHES_CSV_FULL = BASE_DIR / "output" / "li2022_consensus" / "validation_matches.csv"
# Clean-16 subset stats (n=200 pool, scale-invariant matching, filtered to 16 papers)
MATCHES_CSV_CLEAN = BASE_DIR / "output" / "li2022_combined" / "validation_matches_improved.csv"
OUT_DIR = BASE_DIR / "output" / "li2022_formal_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_matches():
    """Load full-dataset matched observations (naive matching, n=163)."""
    df = pd.read_csv(MATCHES_CSV_FULL)
    print(f"Loaded {len(df)} matched observations from {df['paper_id'].nunique()} papers")
    return df


def load_matches_clean():
    """Load improved matched observations for clean-16 subset (n=200 pool)."""
    df = pd.read_csv(MATCHES_CSV_CLEAN)
    print(f"Loaded {len(df)} improved observations from {df['paper_id'].nunique()} papers (clean-16 source)")
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

    # CI for mean difference
    t_crit = stats.t.ppf(0.975, n - 1)
    ci_lower = mean_diff - t_crit * se_mean
    ci_upper = mean_diff + t_crit * se_mean

    # Proportional bias (correlation between mean and difference)
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

    # Upper test: H0: diff >= margin
    t_upper = (mean_diff - margin) / se
    p_upper = stats.t.cdf(t_upper, df)

    # Lower test: H0: diff <= -margin
    t_lower = (mean_diff + margin) / se
    p_lower = 1 - stats.t.cdf(t_lower, df)

    p_tost = max(p_upper, p_lower)

    # 90% CI (the TOST CI)
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
    k = 2  # two raters

    data = np.column_stack([gt, ext])
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    grand_mean = np.mean(data)

    # Sum of squares
    ss_rows = k * np.sum((row_means - grand_mean)**2)
    ss_cols = n * np.sum((col_means - grand_mean)**2)
    ss_total = np.sum((data - grand_mean)**2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # ICC(3,1) - consistency
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)

    # F-test
    f_value = ms_rows / ms_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_value = 1 - stats.f.cdf(f_value, df1, df2)

    # 95% CI for ICC
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


def bootstrap_ci(df_or_gt, ext=None, n_boot=10000, seed=42):
    """Cluster-robust percentile bootstrap CIs, resampling papers not observations.

    BCa was replaced with the percentile method because the BCa implementation
    used a single jackknife acceleration constant computed from MAE and applied
    it to all statistics (Pearson r, direction agreement, effect diff), which
    produces invalid intervals when the number of clusters is small (n=16–46).
    The percentile method gives valid coverage without this assumption.

    Accepts either:
      bootstrap_ci(df)         where df has paper_id, gt_effect_pct, ext_effect_pct
      bootstrap_ci(gt, ext)    legacy array interface (observation-level)
    """
    rng = np.random.RandomState(seed)

    if isinstance(df_or_gt, pd.DataFrame):
        df_in = df_or_gt
        papers = df_in['paper_id'].unique()
        n_papers = len(papers)
        gt = df_in['gt_effect_pct'].values
        ext = df_in['ext_effect_pct'].values
        n = n_papers  # retained for legacy; jackknife no longer used

        def resample():
            chosen = rng.choice(n_papers, n_papers, replace=True)
            parts = [df_in[df_in['paper_id'] == papers[i]] for i in chosen]
            sdf = pd.concat(parts)
            return sdf['gt_effect_pct'].values, sdf['ext_effect_pct'].values

        def jackknife_resample(leave_out_idx):
            keep = np.delete(np.arange(n_papers), leave_out_idx)
            parts = [df_in[df_in['paper_id'] == papers[i]] for i in keep]
            sdf = pd.concat(parts)
            return sdf['gt_effect_pct'].values, sdf['ext_effect_pct'].values

        resampling_unit = f"paper ({n_papers} papers)"
    else:
        gt = np.asarray(df_or_gt)
        ext = np.asarray(ext)
        n = len(gt)
        def resample():
            idx = rng.choice(n, n, replace=True)
            return gt[idx], ext[idx]
        def jackknife_resample(leave_out_idx):
            mask = np.arange(n) != leave_out_idx
            return gt[mask], ext[mask]
        resampling_unit = f"observation ({n} obs)"

    # Point estimates
    r_point = float(np.corrcoef(gt, ext)[0, 1])
    diff = np.abs(ext - gt)
    mae_point = float(np.mean(diff))

    # Direction (for positive effects in Li 2022, both should be positive)
    nonzero_mask = np.abs(gt) > 0.5  # meaningful effects only
    dir_match = np.sum(np.sign(gt[nonzero_mask]) == np.sign(ext[nonzero_mask]))
    nonzero = np.sum(nonzero_mask)
    dir_point = float(dir_match / max(nonzero, 1))

    effect_diff_point = float(np.mean(ext) - np.mean(gt))  # signed: ext minus gt
    within10_point = float(np.mean(diff <= 10))

    # Bootstrap
    boot_r = []
    boot_mae = []
    boot_dir = []
    boot_effect_diff = []
    boot_within10 = []

    for _ in range(n_boot):
        bg, be = resample()

        try:
            boot_r.append(float(np.corrcoef(bg, be)[0, 1]))
        except:
            boot_r.append(np.nan)

        bd = np.abs(be - bg)
        boot_mae.append(float(np.mean(bd)))

        nonz_mask = np.abs(bg) > 0.5
        dm = np.sum(np.sign(bg[nonz_mask]) == np.sign(be[nonz_mask]))
        boot_dir.append(float(dm / max(np.sum(nonz_mask), 1)))

        boot_effect_diff.append(float(np.mean(be) - np.mean(bg)))  # signed
        boot_within10.append(float(np.mean(bd <= 10)))

    def percentile_ci(boot_vals, alpha=0.05):
        """Cluster-robust percentile bootstrap CI.

        BCa was avoided because the standard BCa implementation computes the
        jackknife acceleration using one statistic (MAE) and applies the same
        acceleration constant to all statistics — a known instability when the
        number of clusters is small (n=16–46 papers).  Percentile bootstrap
        provides valid coverage without this assumption.
        """
        boot_arr = np.array([v for v in boot_vals if not np.isnan(v)])
        if len(boot_arr) < 100:
            return (np.nan, np.nan)
        return (float(np.percentile(boot_arr, alpha / 2 * 100)),
                float(np.percentile(boot_arr, (1 - alpha / 2) * 100)))

    results = {}

    for name, boot_vals, point in [
        ("pearson_r", boot_r, r_point),
        ("mae_pct", boot_mae, mae_point),
        ("direction_agreement", boot_dir, dir_point),
        ("overall_effect_diff_pp", boot_effect_diff, effect_diff_point),
        ("within_10pp", boot_within10, within10_point),
    ]:
        ci = percentile_ci(boot_vals)
        if np.isnan(ci[0]):
            ci = (np.nan, np.nan)

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

    # Paired t-test
    t_stat, p_ttest = stats.ttest_rel(ext, gt)

    # Wilcoxon signed-rank
    try:
        w_stat, p_wilcox = stats.wilcoxon(diff)
    except:
        w_stat, p_wilcox = np.nan, np.nan

    # Cohen's d
    d = np.mean(diff) / np.std(diff, ddof=1)

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


# Papers excluded from the Structurally Concordant Subset (Clean-16)
# These 12 paper IDs are excluded due to PDF/consensus failures, GT attribution
# errors, outcome-category mismatches, aggregation-level mismatches, GT source
# mismatches, or product-selection omissions. See Section 4.4 and Supplementary
# Table S4 for detailed rationale for each exclusion.
EXCLUDED_PAPER_IDS = [
    # PDF/consensus failures (2)
    "002_Abdel-Mawgoud_2010_Growth and yield responses of strawberry",
    "006_Alabdulla_2019_Effect of foliar application of humic ac",
    # GT attribution / outcome-category errors (4)
    "062_Głosek-Sobieraj_2018_The Effect of Growth Regulators and a Bi",
    "064_Godlewska_2016_The effect of growth regulator on dry ma",
    "111_Mondal_2013_Foliar application of chitosan improves",
    "120_Pohl_2019_The Eggplant Yield and Fruit Composition",
    # Aggregation-level mismatches (3)
    "091_Kocira_2018_Modeling biometric traits",
    "090_Kocira_2020_Biochemical and economical effect of app",
    "125_Procházka_2015_The possibilities of increasing the prod",
    # Product-selection omission (1)
    "088_Kocira_2019_Effect of amino acid biostimulant on the",
    # GT source mismatches (2)
    "095_Kuisma_1989_The effect of foliar application of seaw",
    "124_Pramanick_2016_Effect of seaweed saps derived from two",
]


def run_analysis(df, label):
    """Run full formal stats on a given DataFrame slice."""
    gt = df['gt_effect_pct'].values
    ext = df['ext_effect_pct'].values
    n_papers = df['paper_id'].nunique()

    print(f"\nBasic stats ({label}):")
    print(f"  N obs: {len(gt)}, N papers: {n_papers}")
    print(f"  GT mean effect: {np.mean(gt):.2f}%")
    print(f"  Extracted mean effect: {np.mean(ext):.2f}%")
    print(f"  Difference: {abs(np.mean(ext) - np.mean(gt)):.2f} pp")

    ba = bland_altman(gt, ext)
    for margin in [2, 3, 5, 10]:
        tost_equivalence(gt, ext, margin=margin)
    compute_icc(gt, ext)
    print(f"\n=== Bootstrap CIs (10,000 percentile resamples, paper as resampling unit) ===")
    boot = bootstrap_ci(df)
    systematic_bias(gt, ext)

    paper_gt = df.groupby('paper_id')['gt_effect_pct'].mean()
    paper_ext = df.groupby('paper_id')['ext_effect_pct'].mean()
    common = paper_gt.index.intersection(paper_ext.index)
    if len(common) >= 5:
        print(f"\n=== Paper-Level ICC ({len(common)} papers) ===")
        compute_icc(paper_gt[common].values, paper_ext[common].values)

    return ba, boot


def main():
    print("=" * 60)
    print("FORMAL STATISTICS - Li 2022 (28 papers, biostimulant/yield)")
    print("=" * 60)

    df = load_matches()

    gt_effects = df['gt_effect_pct'].values
    ext_effects = df['ext_effect_pct'].values

    print(f"\nBasic stats:")
    print(f"  GT mean effect: {np.mean(gt_effects):.2f}%")
    print(f"  Extracted mean effect: {np.mean(ext_effects):.2f}%")
    print(f"  Difference: {abs(np.mean(ext_effects) - np.mean(gt_effects)):.2f} pp")

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

    # 4. Bootstrap CIs (cluster-robust, paper as resampling unit)
    print(f"\n=== Bootstrap CIs (10,000 percentile resamples, paper as resampling unit) ===")
    boot = bootstrap_ci(df)
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

    # Summary (full dataset)
    print("\n" + "=" * 60)
    print("SUMMARY — Full Dataset")
    print("=" * 60)
    print(f"N observations: {len(gt_effects)}")
    print(f"N papers: {df['paper_id'].nunique()}")
    print(f"Pearson r: {np.corrcoef(gt_effects, ext_effects)[0,1]:.3f}")
    print(f"MAE: {np.mean(np.abs(ext_effects - gt_effects)):.2f}%")
    print(f"Median AE: {np.median(np.abs(ext_effects - gt_effects)):.2f}%")
    print(f"Direction agreement: {np.mean(np.sign(gt_effects) == np.sign(ext_effects))*100:.1f}%")
    print(f"Overall effect diff: {abs(np.mean(ext_effects) - np.mean(gt_effects)):.2f} pp")
    print(f"Bland-Altman bias: {ba['mean_difference']:.2f} pp")
    print(f"TOST (±2pp): p={tost_results['margin_2pp']['p_tost']:.6f} {'EQUIVALENT' if tost_results['margin_2pp']['equivalent'] else 'NOT EQUIVALENT'}")
    print(f"TOST (±5pp): p={tost_results['margin_5pp']['p_tost']:.6f} {'EQUIVALENT' if tost_results['margin_5pp']['equivalent'] else 'NOT EQUIVALENT'}")
    print(f"ICC(3,1): {icc['icc_31']:.3f}")
    print(f"Cohen's d: {bias['cohens_d']:.4f}")

    # ─── STRUCTURALLY CONCORDANT SUBSET (Clean papers only) ──────────────────────
    print("\n" + "=" * 60)
    print("FORMAL STATISTICS — Structurally Concordant Subset")
    print("(Excluding 12 papers with GT/PDF provenance issues)")
    print("=" * 60)

    # Load the improved CSV (scale-invariant matching) for the clean-16 subset
    df_c_pool = load_matches_clean()

    # Fuzzy match exclusion IDs (paper_ids may be truncated in CSV)
    def is_excluded(pid):
        for excl in EXCLUDED_PAPER_IDS:
            # Match by prefix (first 30 chars covers paper number + author + year)
            if pid.startswith(excl[:30]) or excl.startswith(pid[:30]):
                return True
        return False

    df_clean = df_c_pool[~df_c_pool['paper_id'].apply(is_excluded)].copy()
    n_clean_papers = df_clean['paper_id'].nunique()
    n_clean_obs = len(df_clean)

    print(f"\nClean subset: {n_clean_obs} observations from {n_clean_papers} papers")
    print(f"Excluded papers: {df_c_pool['paper_id'].nunique() - n_clean_papers}")

    if n_clean_obs < 10:
        print("Too few observations for formal stats — check EXCLUDED_PAPER_IDS list")
    else:
        gt_c = df_clean['gt_effect_pct'].values
        ext_c = df_clean['ext_effect_pct'].values

        # Bland-Altman
        ba_c = bland_altman(gt_c, ext_c)
        with open(OUT_DIR / "bland_altman_clean.json", 'w') as f:
            json.dump(ba_c, f, indent=2)

        # TOST
        tost_c = {}
        for margin in [1, 2, 3, 5]:
            tost_c[f"margin_{margin}pp"] = tost_equivalence(gt_c, ext_c, margin=margin)
        with open(OUT_DIR / "tost_clean.json", 'w') as f:
            json.dump(tost_c, f, indent=2)

        # ICC
        icc_c = compute_icc(gt_c, ext_c)
        with open(OUT_DIR / "icc_clean.json", 'w') as f:
            json.dump(icc_c, f, indent=2)

        # Bootstrap
        print(f"\n=== Bootstrap CIs (10,000 percentile resamples, paper as resampling unit) ===")
        boot_c = bootstrap_ci(df_clean)
        with open(OUT_DIR / "bootstrap_clean.json", 'w') as f:
            json.dump(boot_c, f, indent=2)

        # Systematic bias
        bias_c = systematic_bias(gt_c, ext_c)
        with open(OUT_DIR / "systematic_bias_clean.json", 'w') as f:
            json.dump(bias_c, f, indent=2)

        # Paper-level ICC for clean subset
        pg_c = df_clean.groupby('paper_id')['gt_effect_pct'].mean()
        pe_c = df_clean.groupby('paper_id')['ext_effect_pct'].mean()
        comm_c = pg_c.index.intersection(pe_c.index)
        if len(comm_c) >= 5:
            print(f"\n=== Paper-Level ICC ({len(comm_c)} papers, clean subset) ===")
            compute_icc(pg_c[comm_c].values, pe_c[comm_c].values)

        # Direction agreement (|gt| > 0.5 filter)
        nonzero = np.abs(gt_c) > 0.5
        dir_agree = np.mean(np.sign(gt_c[nonzero]) == np.sign(ext_c[nonzero])) * 100

        print("\n" + "=" * 60)
        print("SUMMARY — Structurally Concordant Subset")
        print("=" * 60)
        print(f"N observations: {n_clean_obs}")
        print(f"N papers: {n_clean_papers}")
        print(f"Pearson r: {np.corrcoef(gt_c, ext_c)[0,1]:.4f}")
        print(f"MAE: {np.mean(np.abs(ext_c - gt_c)):.4f} pp")
        print(f"Median AE: {np.median(np.abs(ext_c - gt_c)):.4f} pp")
        print(f"Direction agreement (|gt|>0.5): {dir_agree:.1f}%")
        print(f"Overall effect diff: {abs(np.mean(ext_c) - np.mean(gt_c)):.4f} pp")
        print(f"ICC(3,1): {icc_c['icc_31']:.4f}")
        print(f"Cohen's d: {bias_c['cohens_d']:.4f}")

    print(f"\nAll results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
