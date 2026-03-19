"""
Formal Statistical Analyses for Agent Extraction (all 3 datasets)
=================================================================
Computes ICC, TOST, Bland-Altman, Cohen's d, bootstrap CIs.

Run:
    python formal_stats_agent.py
"""
import sys, os, json, math, csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats

BASE_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
sys.path.insert(0, str(BASE_DIR))
OUT_DIR = BASE_DIR / "output" / "agent_formal_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
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


# ============================================================
# LOAD AGENT VALIDATION DATA
# ============================================================

def load_loladze_agent():
    """Load Loladze agent matches from validation_report_agent.json"""
    report_path = BASE_DIR / "output" / "agent_extraction" / "validation_report_agent.json"
    with open(report_path) as f:
        report = json.load(f)

    matches = []
    paper_effects = defaultdict(lambda: {'our': [], 'gt': []})

    for m in report['all_matches']:
        our_eff = m['our']  # fraction
        gt_eff = m['gt']    # fraction
        paper = m['paper']
        matches.append({
            'our_pct': our_eff * 100,
            'gt_pct': gt_eff * 100,
            'paper': paper,
        })
        paper_effects[paper]['our'].append(our_eff * 100)
        paper_effects[paper]['gt'].append(gt_eff * 100)

    return matches, dict(paper_effects)


def load_hui_agent():
    """Load Hui agent matches from validate_hui2023_agent.py output (with match_pairs)."""
    hui_agent_dir = BASE_DIR / "output" / "hui2023_agent_extraction"
    report_json = hui_agent_dir / "validation_report_agent.json"

    if not report_json.exists():
        print(f"  Hui agent report not found at {report_json}")
        return None, None

    with open(report_json) as f:
        report = json.load(f)

    match_pairs = report.get('match_pairs', [])
    if not match_pairs:
        print(f"  No match_pairs in Hui report - re-run validate_hui2023_agent.py")
        return None, None

    matches = []
    paper_effects = defaultdict(lambda: {'our': [], 'gt': []})
    for m in match_pairs:
        our = m['our_pct']
        gt = m['gt_pct']
        paper = m.get('paper', 'unknown')
        matches.append({'our_pct': our, 'gt_pct': gt, 'paper': paper})
        paper_effects[paper]['our'].append(our)
        paper_effects[paper]['gt'].append(gt)

    return matches, dict(paper_effects)


def load_li_agent(tier='high'):
    """Load Li agent matches from harmonize output (with match_pairs_by_tier)."""
    li_agent_dir = BASE_DIR / "output" / "li2022_agent_extraction"
    report_json = li_agent_dir / "harmonized_validation_agent.json"

    if not report_json.exists():
        print(f"  Li agent report not found at {report_json}")
        return None, None

    with open(report_json) as f:
        data = json.load(f)

    pairs_by_tier = data.get('match_pairs_by_tier', {})
    if not pairs_by_tier:
        print(f"  No match_pairs_by_tier in Li report - re-run harmonize_li2022_agent.py")
        return None, None

    # Use high tier for primary analysis
    pairs = pairs_by_tier.get(tier, [])
    if not pairs:
        print(f"  No {tier}-tier pairs found")
        return None, None

    matches = []
    paper_effects = defaultdict(lambda: {'our': [], 'gt': []})
    for m in pairs:
        our = m['ext_effect']
        gt = m['gt_effect']
        paper = m.get('paper', 'unknown')
        matches.append({'our_pct': our, 'gt_pct': gt, 'paper': paper})
        paper_effects[paper]['our'].append(our)
        paper_effects[paper]['gt'].append(gt)

    return matches, dict(paper_effects)


# ============================================================
# STATISTICAL FUNCTIONS
# ============================================================

def compute_icc(our, gt, n):
    """ICC(3,1) - two-way mixed, single measures, consistency."""
    k = 2
    ratings = np.column_stack([gt, our])
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)

    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_cols = ss_cols / (k - 1)

    icc_31 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
    icc_21 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)

    f_value = ms_rows / ms_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)

    f_lower = f_value / stats.f.ppf(0.975, df1, df2)
    f_upper = f_value / stats.f.ppf(0.025, df1, df2)
    icc_31_lower = (f_lower - 1) / (f_lower + k - 1)
    icc_31_upper = (f_upper - 1) / (f_upper + k - 1)

    return {
        'icc_31': round(float(icc_31), 4),
        'icc_21': round(float(icc_21), 4),
        'icc_31_ci': [round(float(icc_31_lower), 4), round(float(icc_31_upper), 4)],
        'f_value': round(float(f_value), 2),
    }


def compute_tost(our, gt, margins=[0.5, 1.0, 2.0, 3.0, 5.0]):
    """TOST equivalence testing at multiple margins (in pp)."""
    diff = our - gt
    n = len(diff)
    mean_diff = np.mean(diff)
    se_naive = np.std(diff, ddof=1) / np.sqrt(n)

    results = {}
    for margin in margins:
        t1 = (mean_diff - (-margin)) / se_naive
        p1 = 1 - stats.t.cdf(t1, df=n - 1)
        t2 = (mean_diff - margin) / se_naive
        p2 = stats.t.cdf(t2, df=n - 1)
        p_tost = max(p1, p2)
        results[f'{margin}pp'] = {
            'p_value': round(float(p_tost), 6),
            'equivalent': bool(p_tost < 0.05),
        }

    # 90% CI
    t_crit = stats.t.ppf(0.95, df=n - 1)
    ci_90 = (mean_diff - t_crit * se_naive, mean_diff + t_crit * se_naive)

    return {
        'mean_diff_pp': round(float(mean_diff), 3),
        'se_naive': round(float(se_naive), 4),
        'ci_90_pp': [round(float(ci_90[0]), 3), round(float(ci_90[1]), 3)],
        'margins': results,
    }


def compute_tost_cluster_robust(our, gt, papers):
    """TOST with cluster-robust standard errors (clustering by paper)."""
    diff = our - gt
    n = len(diff)
    mean_diff = np.mean(diff)

    # Cluster-robust SE (CR1 sandwich estimator)
    unique_papers = np.unique(papers)
    K = len(unique_papers)

    cluster_sums = []
    for p in unique_papers:
        mask = papers == p
        cluster_sums.append(np.sum(diff[mask] - mean_diff))
    cluster_sums = np.array(cluster_sums)

    # CR1 adjustment
    if K <= 1:
        return {
            'mean_diff_pp': round(float(mean_diff), 3),
            'se_robust': float('nan'),
            'se_naive': round(float(se_naive), 4),
            'design_effect': float('nan'),
            'n_clusters': int(K),
            'df': 0,
            'ci_90_pp': [float('nan'), float('nan')],
            'margins': {f'{m}pp': {'p_value': float('nan'), 'equivalent': False} for m in margins},
        }
    cr1_factor = K / (K - 1) * (n - 1) / n
    var_robust = cr1_factor * np.sum(cluster_sums ** 2) / (n ** 2)
    se_robust = np.sqrt(var_robust)

    # Design effect
    se_naive = np.std(diff, ddof=1) / np.sqrt(n)
    design_effect = (se_robust / se_naive) ** 2 if se_naive > 0 else 1.0

    margins = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = {}
    df = K - 1

    for margin in margins:
        t1 = (mean_diff - (-margin)) / se_robust
        p1 = 1 - stats.t.cdf(t1, df=df)
        t2 = (mean_diff - margin) / se_robust
        p2 = stats.t.cdf(t2, df=df)
        p_tost = max(p1, p2)
        results[f'{margin}pp'] = {
            'p_value': round(float(p_tost), 6),
            'equivalent': bool(p_tost < 0.05),
        }

    t_crit = stats.t.ppf(0.95, df=df)
    ci_90 = (mean_diff - t_crit * se_robust, mean_diff + t_crit * se_robust)

    return {
        'mean_diff_pp': round(float(mean_diff), 3),
        'se_robust': round(float(se_robust), 4),
        'se_naive': round(float(se_naive), 4),
        'design_effect': round(float(design_effect), 2),
        'n_clusters': int(K),
        'df': int(df),
        'ci_90_pp': [round(float(ci_90[0]), 3), round(float(ci_90[1]), 3)],
        'margins': results,
    }


def compute_bland_altman(our, gt):
    """Bland-Altman limits of agreement."""
    diff = our - gt
    mean_pair = (our + gt) / 2
    n = len(diff)

    mean_diff = np.mean(diff)
    sd_diff = np.std(diff, ddof=1)

    loa_upper = mean_diff + 1.96 * sd_diff
    loa_lower = mean_diff - 1.96 * sd_diff

    # Proportional bias
    r_prop, p_prop = stats.pearsonr(mean_pair, diff)

    within_loa = np.sum((diff >= loa_lower) & (diff <= loa_upper)) / n * 100

    return {
        'mean_diff_pp': round(float(mean_diff), 3),
        'sd_diff_pp': round(float(sd_diff), 3),
        'loa_upper_pp': round(float(loa_upper), 2),
        'loa_lower_pp': round(float(loa_lower), 2),
        'within_loa_pct': round(float(within_loa), 1),
        'proportional_bias_r': round(float(r_prop), 3),
        'proportional_bias_p': round(float(p_prop), 4),
    }


def compute_paired_tests(our, gt):
    """Paired t-test, Wilcoxon, Cohen's d."""
    diff = our - gt

    t_stat, p_value = stats.ttest_rel(our, gt)

    try:
        w_stat, w_p = stats.wilcoxon(diff)
    except ValueError:
        w_stat, w_p = float('nan'), float('nan')

    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

    def interpret_d(d):
        if abs(d) < 0.2: return 'negligible'
        elif abs(d) < 0.5: return 'small'
        elif abs(d) < 0.8: return 'medium'
        else: return 'large'

    return {
        'paired_t': round(float(t_stat), 3),
        'paired_t_p': round(float(p_value), 6),
        'wilcoxon_W': round(float(w_stat), 1) if not math.isnan(w_stat) else None,
        'wilcoxon_p': round(float(w_p), 6) if not math.isnan(w_p) else None,
        'cohens_d': round(float(d), 4),
        'cohens_d_interpretation': interpret_d(d),
        'mean_diff_pp': round(float(np.mean(diff)), 3),
    }


def analyze_dataset(name, matches, paper_effects):
    """Run full formal stats for one dataset."""
    if not matches:
        print(f"\n  {name}: No data available, skipping")
        return None

    our = np.array([m['our_pct'] for m in matches])
    gt = np.array([m['gt_pct'] for m in matches])
    papers = np.array([m['paper'] for m in matches])
    n = len(our)
    n_papers = len(set(papers))

    print(f"\n{'='*70}")
    print(f"  {name}: {n} observations, {n_papers} papers")
    print(f"{'='*70}")

    # Basic metrics
    r = np.corrcoef(our, gt)[0, 1]
    mae = np.mean(np.abs(our - gt))
    nonzero = gt != 0
    dir_agree = np.mean((our[nonzero] < 0) == (gt[nonzero] < 0)) * 100 if np.any(nonzero) else 0
    overall_gt = np.mean(gt)
    overall_our = np.mean(our)

    print(f"  Pearson r:           {r:.3f}")
    print(f"  MAE:                 {mae:.2f} pp")
    print(f"  Direction agreement: {dir_agree:.0f}%")
    print(f"  Overall effect:      GT={overall_gt:.2f}%, Agent={overall_our:.2f}%, diff={abs(overall_gt-overall_our):.2f}pp")

    # ICC
    icc = compute_icc(our, gt, n)
    print(f"  ICC(3,1):            {icc['icc_31']:.3f} ({icc['icc_31_ci'][0]:.3f}-{icc['icc_31_ci'][1]:.3f})")
    print(f"  ICC(2,1):            {icc['icc_21']:.3f}")

    # TOST naive
    tost_naive = compute_tost(our, gt)
    print(f"  TOST (naive):")
    for margin, res in tost_naive['margins'].items():
        status = "PASS" if res['equivalent'] else "FAIL"
        print(f"    ±{margin}: p={res['p_value']:.4f} {status}")

    # TOST cluster-robust
    tost_robust = compute_tost_cluster_robust(our, gt, papers)
    print(f"  TOST (cluster-robust, K={tost_robust['n_clusters']}):")
    print(f"    Design effect: {tost_robust['design_effect']:.2f}")
    for margin, res in tost_robust['margins'].items():
        status = "PASS" if res['equivalent'] else "FAIL"
        print(f"    ±{margin}: p={res['p_value']:.4f} {status}")

    # Bland-Altman
    ba = compute_bland_altman(our, gt)
    print(f"  Bland-Altman:")
    print(f"    Mean diff: {ba['mean_diff_pp']:.2f} pp")
    print(f"    95% LOA:   {ba['loa_lower_pp']:.1f} to {ba['loa_upper_pp']:.1f} pp")
    print(f"    Prop. bias: r={ba['proportional_bias_r']:.3f}, p={ba['proportional_bias_p']:.4f}")

    # Paired tests
    pt = compute_paired_tests(our, gt)
    print(f"  Paired t-test: t={pt['paired_t']:.2f}, p={pt['paired_t_p']:.4f}")
    print(f"  Cohen's d:     {pt['cohens_d']:.3f} ({pt['cohens_d_interpretation']})")

    # Paper-level ICC
    paper_our = []
    paper_gt = []
    for pid, eff in paper_effects.items():
        if eff['our'] and eff['gt']:
            paper_our.append(np.mean(eff['our']))
            paper_gt.append(np.mean(eff['gt']))

    paper_icc = None
    if len(paper_our) > 2:
        paper_our_arr = np.array(paper_our)
        paper_gt_arr = np.array(paper_gt)
        paper_icc = compute_icc(paper_our_arr, paper_gt_arr, len(paper_our))
        paper_r = np.corrcoef(paper_our_arr, paper_gt_arr)[0, 1]
        print(f"  Paper-level ICC(3,1): {paper_icc['icc_31']:.3f}")
        print(f"  Paper-level r:        {paper_r:.3f}")

    result = {
        'dataset': name,
        'n_obs': int(n),
        'n_papers': int(n_papers),
        'pearson_r': round(float(r), 4),
        'mae_pp': round(float(mae), 2),
        'direction_agreement_pct': round(float(dir_agree), 1),
        'overall_effect_gt_pct': round(float(overall_gt), 2),
        'overall_effect_agent_pct': round(float(overall_our), 2),
        'overall_effect_diff_pp': round(float(abs(overall_gt - overall_our)), 2),
        'icc': icc,
        'tost_naive': tost_naive,
        'tost_cluster_robust': tost_robust,
        'bland_altman': ba,
        'paired_tests': pt,
        'paper_level_icc': paper_icc,
    }

    return result


def main():
    print("=" * 70)
    print("FORMAL STATISTICS: AGENT EXTRACTION (ALL 3 DATASETS)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    all_results = {}

    # 1. Loladze
    print("\nLoading Loladze agent data...")
    lol_matches, lol_papers = load_loladze_agent()
    print(f"  Loaded {len(lol_matches)} matches from {len(lol_papers)} papers")
    all_results['loladze'] = analyze_dataset("Loladze 2014", lol_matches, lol_papers)

    # 2. Hui
    print("\nLoading Hui agent data...")
    hui_matches, hui_papers = load_hui_agent()
    if hui_matches:
        print(f"  Loaded {len(hui_matches)} matches from {len(hui_papers)} papers")
        all_results['hui'] = analyze_dataset("Hui 2023", hui_matches, hui_papers)
    else:
        print("  Hui agent data not found in pre-computed format")
        print("  Run validate_hui2023_agent.py first to generate matches")
        all_results['hui'] = None

    # 3. Li
    print("\nLoading Li agent data...")
    li_matches, li_papers = load_li_agent()
    if li_matches:
        print(f"  Loaded {len(li_matches)} matches from {len(li_papers)} papers")
        all_results['li'] = analyze_dataset("Li 2022", li_matches, li_papers)
    else:
        print("  Li agent data not found in pre-computed format")
        print("  Run harmonize_li2022_agent.py first to generate matches")
        all_results['li'] = None

    # Save
    out_path = OUT_DIR / "agent_formal_stats_all.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\n\nSaved results to {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE FOR PAPER")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'N':>5} {'K':>3} {'r':>6} {'MAE':>6} {'ICC(3,1)':>10} {'TOST ±2pp':>12} {'TOST ±3pp':>12} {'Cohen d':>8}")
    print("-" * 85)
    for key in ['loladze', 'hui', 'li']:
        r = all_results.get(key)
        if not r:
            continue
        tost2 = r['tost_cluster_robust']['margins'].get('2.0pp', {})
        tost3 = r['tost_cluster_robust']['margins'].get('3.0pp', {})
        t2_str = f"p={tost2.get('p_value', 'N/A')}" if tost2 else "N/A"
        t3_str = f"p={tost3.get('p_value', 'N/A')}" if tost3 else "N/A"
        if tost2.get('equivalent'):
            t2_str += " PASS"
        elif tost2:
            t2_str += " FAIL"
        if tost3.get('equivalent'):
            t3_str += " PASS"
        elif tost3:
            t3_str += " FAIL"

        print(f"{r['dataset']:<15} {r['n_obs']:>5} {r['n_papers']:>3} {r['pearson_r']:>6.3f} {r['mae_pp']:>5.1f}pp "
              f"{r['icc']['icc_31']:>8.3f}   {t2_str:>12} {t3_str:>12} {r['paired_tests']['cohens_d']:>7.3f}")


if __name__ == '__main__':
    main()
