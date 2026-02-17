"""
Formal Statistical Analyses for Paper
======================================
Computes Bland-Altman limits of agreement, ICC, TOST equivalence testing,
and bootstrap CIs for all key metrics. Uses existing validation data.

Run:
    python formal_statistics.py

Outputs:
    output/formal_stats/
        bland_altman_results.json
        icc_results.json
        tost_results.json
        bootstrap_ci.json
        formal_stats_summary.txt
        fig_bland_altman_formal.png
        fig_tost_equivalence.png
"""
import sys, os, json, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats

BASE_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
RESULTS_DIR = BASE_DIR / "output" / "loladze_combined_51"
OUT_DIR = BASE_DIR / "output" / "formal_stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Import the validation logic
sys.path.insert(0, str(BASE_DIR))
from validate_full_46 import (
    load_gt, PAPER_TO_LOLADZE_REF, MISLABELED_PDFS,
    normalize_element, filter_obs_for_gt_row, deduplicate_vision_text,
    is_concentration_unit, compute_effect, detect_tc_swap, get_mods
)

import re


def load_all_matches():
    """Load and compute all matched observation pairs (our effect vs GT effect)."""
    gt = load_gt()
    results_files = sorted(RESULTS_DIR.glob("*_consensus.json"))

    all_matches = []
    paper_effects = defaultdict(lambda: {'our': [], 'gt': []})

    for rf in results_files:
        paper_id = rf.stem.replace("_consensus", "")
        loladze_ref = PAPER_TO_LOLADZE_REF.get(paper_id)

        if not loladze_ref or loladze_ref not in gt:
            for ref in gt:
                surname = paper_id.split('_')[1] if '_' in paper_id else paper_id
                if surname.lower() in ref.lower():
                    loladze_ref = ref
                    break
            if not loladze_ref or loladze_ref not in gt:
                continue

        with open(rf) as f:
            data = json.load(f)

        gt_rows = gt[loladze_ref]
        obs_list = data.get('consensus_observations', [])
        obs_list = deduplicate_vision_text(obs_list)

        has_conc = any(is_concentration_unit(o.get('unit', '')) for o in obs_list)
        has_total = any(not is_concentration_unit(o.get('unit', ''))
                        for o in obs_list if o.get('unit', ''))
        if not has_conc and has_total:
            continue

        # Filter sub-ambient CO2
        elevated_obs = []
        for o in obs_list:
            desc = str(o.get('treatment_description', '')).lower()
            co2_match = re.search(r'(\d{2,4})\s*(?:ppm|µmol|umol|μmol)', desc)
            if co2_match:
                co2_val = float(co2_match.group(1))
                if co2_val < 300:
                    continue
            if 'low co2' in desc or 'sub-ambient' in desc:
                continue
            elevated_obs.append(o)
        if elevated_obs:
            obs_list = elevated_obs

        swap_tc = detect_tc_swap(obs_list, gt_rows)

        used_obs_ids = set()
        gt_by_el_info = defaultdict(list)
        for gt_row in gt_rows:
            key = (gt_row['element'], gt_row['info'])
            gt_by_el_info[key].append(gt_row)

        for gt_row in gt_rows:
            candidates = filter_obs_for_gt_row(obs_list, gt_row, paper_id)
            if not candidates:
                continue

            cand_effects = []
            for i, c in enumerate(candidates):
                obs_id = id(c)
                eff = compute_effect(c, swap_tc=swap_tc)
                if eff is not None and abs(eff) <= 5.0:
                    cand_effects.append((obs_id, eff, c))

            if not cand_effects:
                continue

            gt_effect = gt_row['effect']
            key = (gt_row['element'], gt_row['info'])
            n_gt_same = len(gt_by_el_info[key])

            if n_gt_same > 1 and len(cand_effects) > 1:
                unused = [(oid, eff, c) for oid, eff, c in cand_effects
                          if oid not in used_obs_ids]
                if not unused:
                    unused = cand_effects
                best = min(unused, key=lambda x: abs(x[1] - gt_effect))
                our_effect = best[1]
                used_obs_ids.add(best[0])
            else:
                effects = [eff for _, eff, _ in cand_effects]
                our_effect = sum(effects) / len(effects)

            all_matches.append({
                'paper': paper_id,
                'ref': loladze_ref,
                'el': gt_row['element'],
                'our': our_effect,
                'gt': gt_effect,
                'err': abs(our_effect - gt_effect),
            })
            paper_effects[paper_id]['our'].append(our_effect)
            paper_effects[paper_id]['gt'].append(gt_effect)

    return all_matches, dict(paper_effects)


# ============================================================
# 1. BLAND-ALTMAN ANALYSIS
# ============================================================

def bland_altman_analysis(matches):
    """
    Bland-Altman analysis: assess agreement between pipeline and GT.
    Plots difference vs mean; computes limits of agreement.
    """
    our = np.array([m['our'] * 100 for m in matches])
    gt = np.array([m['gt'] * 100 for m in matches])

    diff = our - gt  # pipeline - GT
    mean = (our + gt) / 2

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)

    # 95% limits of agreement
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    # CIs for mean diff and LOAs (Bland & Altman 1999)
    se_mean = std_diff / np.sqrt(n)
    se_loa = np.sqrt(3 * std_diff**2 / n)

    mean_diff_ci = (mean_diff - 1.96 * se_mean, mean_diff + 1.96 * se_mean)
    loa_upper_ci = (loa_upper - 1.96 * se_loa, loa_upper + 1.96 * se_loa)
    loa_lower_ci = (loa_lower - 1.96 * se_loa, loa_lower + 1.96 * se_loa)

    # Proportional bias test (correlation between diff and mean)
    r_prop, p_prop = stats.pearsonr(mean, diff)

    # What % of observations fall within LOAs
    within_loa = np.sum((diff >= loa_lower) & (diff <= loa_upper)) / n * 100

    results = {
        'n': int(n),
        'mean_difference_pp': round(float(mean_diff), 3),
        'sd_difference_pp': round(float(std_diff), 3),
        'loa_upper_pp': round(float(loa_upper), 2),
        'loa_lower_pp': round(float(loa_lower), 2),
        'mean_diff_95ci': [round(float(mean_diff_ci[0]), 3), round(float(mean_diff_ci[1]), 3)],
        'loa_upper_95ci': [round(float(loa_upper_ci[0]), 2), round(float(loa_upper_ci[1]), 2)],
        'loa_lower_95ci': [round(float(loa_lower_ci[0]), 2), round(float(loa_lower_ci[1]), 2)],
        'within_loa_pct': round(float(within_loa), 1),
        'proportional_bias_r': round(float(r_prop), 3),
        'proportional_bias_p': round(float(p_prop), 4),
        'interpretation': (
            f"Mean difference (bias) = {mean_diff:.2f} pp (95% CI: {mean_diff_ci[0]:.2f} to {mean_diff_ci[1]:.2f}). "
            f"95% limits of agreement: {loa_lower:.1f} to {loa_upper:.1f} pp. "
            f"{within_loa:.0f}% of observations fall within LOAs. "
            f"Proportional bias {'detected' if p_prop < 0.05 else 'not detected'} (r={r_prop:.3f}, p={p_prop:.4f})."
        )
    }

    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.scatter(mean, diff, alpha=0.3, s=20, color='steelblue')
        ax.axhline(mean_diff, color='red', linestyle='-', linewidth=1.5, label=f'Mean diff = {mean_diff:.2f} pp')
        ax.axhline(loa_upper, color='gray', linestyle='--', linewidth=1, label=f'+1.96 SD = {loa_upper:.1f} pp')
        ax.axhline(loa_lower, color='gray', linestyle='--', linewidth=1, label=f'-1.96 SD = {loa_lower:.1f} pp')
        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)

        # Shade LOA CIs
        ax.axhspan(loa_upper_ci[0], loa_upper_ci[1], alpha=0.1, color='gray')
        ax.axhspan(loa_lower_ci[0], loa_lower_ci[1], alpha=0.1, color='gray')

        ax.set_xlabel('Mean of Pipeline and GT Effect (%)', fontsize=12)
        ax.set_ylabel('Pipeline - GT Effect (percentage points)', fontsize=12)
        ax.set_title('Bland-Altman Plot: Pipeline vs Ground Truth\n(Loladze 2014, 46 papers)', fontsize=13)
        ax.legend(loc='upper left', fontsize=10)
        ax.set_xlim(-60, 40)
        ax.set_ylim(-80, 80)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_bland_altman_formal.png', dpi=150)
        plt.close()
        print("  Saved fig_bland_altman_formal.png")
    except ImportError:
        print("  matplotlib not available, skipping plot")

    return results


# ============================================================
# 2. INTRACLASS CORRELATION COEFFICIENT (ICC)
# ============================================================

def icc_analysis(matches, paper_effects):
    """
    Compute ICC(3,1) - two-way mixed, single measures, consistency.
    This is the appropriate ICC for comparing a new method (pipeline) to a
    reference standard (GT) when we want to assess consistency/agreement.

    Also computes ICC at paper level (mean effect per paper).
    """
    our = np.array([m['our'] * 100 for m in matches])
    gt = np.array([m['gt'] * 100 for m in matches])
    n = len(our)

    # ICC(3,1) - two-way mixed, single measures, consistency
    # Using the formula from Shrout & Fleiss (1979)
    k = 2  # number of raters (pipeline + GT)

    # Build the rating matrix: n observations x 2 raters
    ratings = np.column_stack([gt, our])
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)

    # Sum of squares
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    ms_cols = ss_cols / (k - 1)

    # ICC(3,1) - consistency
    icc_31 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)

    # ICC(2,1) - agreement (accounts for systematic bias)
    icc_21 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)

    # F-test for ICC significance
    f_value = ms_rows / ms_error
    df1 = n - 1
    df2 = (n - 1) * (k - 1)
    p_value = 1 - stats.f.cdf(f_value, df1, df2)

    # 95% CI for ICC(3,1) using F-distribution
    f_lower = f_value / stats.f.ppf(0.975, df1, df2)
    f_upper = f_value / stats.f.ppf(0.025, df1, df2)
    icc_31_lower = (f_lower - 1) / (f_lower + k - 1)
    icc_31_upper = (f_upper - 1) / (f_upper + k - 1)

    # Paper-level ICC (collapse to mean effect per paper)
    paper_our = []
    paper_gt = []
    for pid, eff in paper_effects.items():
        if eff['our'] and eff['gt']:
            paper_our.append(np.mean(eff['our']) * 100)
            paper_gt.append(np.mean(eff['gt']) * 100)

    paper_our = np.array(paper_our)
    paper_gt = np.array(paper_gt)
    n_papers = len(paper_our)

    if n_papers > 2:
        ratings_paper = np.column_stack([paper_gt, paper_our])
        gm_p = np.mean(ratings_paper)
        rm_p = np.mean(ratings_paper, axis=1)
        cm_p = np.mean(ratings_paper, axis=0)
        ss_r_p = 2 * np.sum((rm_p - gm_p) ** 2)
        ss_c_p = n_papers * np.sum((cm_p - gm_p) ** 2)
        ss_t_p = np.sum((ratings_paper - gm_p) ** 2)
        ss_e_p = ss_t_p - ss_r_p - ss_c_p
        ms_r_p = ss_r_p / (n_papers - 1)
        ms_e_p = ss_e_p / (n_papers - 1) if (n_papers - 1) > 0 else 1e-10
        icc_paper = (ms_r_p - ms_e_p) / (ms_r_p + ms_e_p)
    else:
        icc_paper = float('nan')

    # Interpretation guidelines (Cicchetti 1994)
    def interpret_icc(val):
        if val < 0.40:
            return "poor"
        elif val < 0.60:
            return "fair"
        elif val < 0.75:
            return "good"
        else:
            return "excellent"

    results = {
        'observation_level': {
            'icc_31_consistency': round(float(icc_31), 4),
            'icc_21_agreement': round(float(icc_21), 4),
            'icc_31_95ci': [round(float(icc_31_lower), 4), round(float(icc_31_upper), 4)],
            'f_value': round(float(f_value), 2),
            'p_value': float(p_value),
            'n_observations': int(n),
            'interpretation': interpret_icc(icc_31),
        },
        'paper_level': {
            'icc_consistency': round(float(icc_paper), 4) if not np.isnan(icc_paper) else None,
            'n_papers': int(n_papers),
            'interpretation': interpret_icc(icc_paper) if not np.isnan(icc_paper) else "N/A",
        },
        'pearson_r_obs': round(float(np.corrcoef(our, gt)[0, 1]), 4),
        'pearson_r_paper': round(float(np.corrcoef(paper_our, paper_gt)[0, 1]), 4) if n_papers > 2 else None,
        'summary': (
            f"Observation-level ICC(3,1) = {icc_31:.3f} (95% CI: {icc_31_lower:.3f}-{icc_31_upper:.3f}), "
            f"interpreted as {interpret_icc(icc_31)}. "
            f"Paper-level ICC = {icc_paper:.3f} ({interpret_icc(icc_paper)}). "
            f"For reference, human inter-rater ICC in meta-analysis is typically 0.70-0.90."
        )
    }

    return results


# ============================================================
# 3. TOST EQUIVALENCE TESTING
# ============================================================

def tost_equivalence(matches, margin_pp=2.0):
    """
    Two One-Sided Tests (TOST) for equivalence.
    Tests H0: |mean_diff| >= margin vs H1: |mean_diff| < margin

    If both one-sided tests reject at alpha=0.05, we conclude equivalence.
    """
    our = np.array([m['our'] * 100 for m in matches])
    gt = np.array([m['gt'] * 100 for m in matches])
    diff = our - gt
    n = len(diff)

    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(n)

    # TOST: Two one-sided t-tests
    # Test 1: H0: mean_diff <= -margin vs H1: mean_diff > -margin
    t1 = (mean_diff - (-margin_pp)) / se_diff
    p1 = 1 - stats.t.cdf(t1, df=n - 1)

    # Test 2: H0: mean_diff >= +margin vs H1: mean_diff < +margin
    t2 = (mean_diff - margin_pp) / se_diff
    p2 = stats.t.cdf(t2, df=n - 1)

    # Equivalence is concluded if both p < alpha
    p_tost = max(p1, p2)
    equivalent = p_tost < 0.05

    # 90% CI for mean difference (corresponds to alpha=0.05 for TOST)
    t_crit = stats.t.ppf(0.95, df=n - 1)
    ci_90 = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    # Also test at different margins
    margins = [1.0, 2.0, 3.0, 5.0]
    margin_results = {}
    for m in margins:
        t1m = (mean_diff - (-m)) / se_diff
        p1m = 1 - stats.t.cdf(t1m, df=n - 1)
        t2m = (mean_diff - m) / se_diff
        p2m = stats.t.cdf(t2m, df=n - 1)
        pm = max(p1m, p2m)
        margin_results[f'{m}pp'] = {
            'p_value': round(float(pm), 6),
            'equivalent': pm < 0.05,
        }

    # Per-element TOST
    element_tost = {}
    el_data = defaultdict(list)
    for m in matches:
        el_data[m['el']].append(m)

    for el, el_matches in el_data.items():
        if len(el_matches) < 5:
            continue
        el_our = np.array([m['our'] * 100 for m in el_matches])
        el_gt = np.array([m['gt'] * 100 for m in el_matches])
        el_diff = el_our - el_gt
        el_n = len(el_diff)
        el_mean = np.mean(el_diff)
        el_se = np.std(el_diff, ddof=1) / np.sqrt(el_n)

        t1e = (el_mean - (-margin_pp)) / el_se
        p1e = 1 - stats.t.cdf(t1e, df=el_n - 1)
        t2e = (el_mean - margin_pp) / el_se
        p2e = stats.t.cdf(t2e, df=el_n - 1)
        pe = max(p1e, p2e)

        element_tost[el] = {
            'n': int(el_n),
            'mean_diff_pp': round(float(el_mean), 2),
            'p_tost': round(float(pe), 4),
            'equivalent': pe < 0.05,
        }

    results = {
        'margin_pp': margin_pp,
        'n': int(n),
        'mean_diff_pp': round(float(mean_diff), 3),
        'se_diff': round(float(se_diff), 4),
        't1': round(float(t1), 3),
        'p1': round(float(p1), 6),
        't2': round(float(t2), 3),
        'p2': round(float(p2), 6),
        'p_tost': round(float(p_tost), 6),
        'equivalent_at_alpha_05': equivalent,
        'ci_90_pp': [round(float(ci_90[0]), 3), round(float(ci_90[1]), 3)],
        'multiple_margins': margin_results,
        'per_element': element_tost,
        'interpretation': (
            f"TOST with ±{margin_pp}pp equivalence margin: "
            f"mean difference = {mean_diff:.2f} pp, p = {p_tost:.4f}. "
            f"{'Equivalence CONFIRMED' if equivalent else 'Equivalence NOT confirmed'} at alpha=0.05. "
            f"90% CI for mean difference: ({ci_90[0]:.2f}, {ci_90[1]:.2f}) pp. "
            f"{'The 90% CI falls entirely within the ±' + str(margin_pp) + 'pp margin.' if equivalent else ''}"
        ),
    }

    # Generate equivalence plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

        # Plot margin
        ax.axvspan(-margin_pp, margin_pp, alpha=0.15, color='green', label=f'±{margin_pp}pp equivalence zone')
        ax.axvline(0, color='black', linestyle=':', linewidth=0.5)

        # Plot mean diff with 90% CI
        ax.plot(mean_diff, 0.5, 'ro', markersize=10, zorder=5)
        ax.plot([ci_90[0], ci_90[1]], [0.5, 0.5], 'r-', linewidth=2, zorder=4)

        # Per-element results
        y_pos = 1
        for el in sorted(element_tost.keys(), key=lambda e: len(el_data[e]), reverse=True):
            et = element_tost[el]
            color = 'green' if et['equivalent'] else 'orange'
            el_our = np.array([m['our'] * 100 for m in el_data[el]])
            el_gt = np.array([m['gt'] * 100 for m in el_data[el]])
            el_diff = el_our - el_gt
            el_mean = np.mean(el_diff)
            el_se = np.std(el_diff, ddof=1) / np.sqrt(len(el_diff))
            t_crit_el = stats.t.ppf(0.95, df=len(el_diff) - 1)
            ci90_el = (el_mean - t_crit_el * el_se, el_mean + t_crit_el * el_se)

            ax.plot(el_mean, y_pos, 'o', color=color, markersize=6, zorder=5)
            ax.plot([ci90_el[0], ci90_el[1]], [y_pos, y_pos], '-', color=color, linewidth=1.5, zorder=4)
            ax.text(max(ci90_el[1], margin_pp) + 0.3, y_pos, f'{el} (n={et["n"]})',
                    fontsize=8, va='center')
            y_pos += 1

        ax.set_xlabel('Difference: Pipeline - GT (percentage points)', fontsize=11)
        ax.set_title(f'TOST Equivalence Test (±{margin_pp}pp margin)\nLoladze 2014 Dataset', fontsize=12)
        ax.set_yticks([0.5])
        ax.set_yticklabels(['Overall'])
        ax.set_ylim(-0.5, y_pos + 0.5)

        # Add result text
        result_text = f"Overall: {'EQUIVALENT' if equivalent else 'NOT equivalent'} (p={p_tost:.4f})"
        ax.text(0.02, 0.98, result_text, transform=ax.transAxes, fontsize=10,
                va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if equivalent else 'lightyellow', alpha=0.8))

        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_tost_equivalence.png', dpi=150)
        plt.close()
        print("  Saved fig_tost_equivalence.png")
    except ImportError:
        print("  matplotlib not available, skipping plot")

    return results


# ============================================================
# 4. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

def bootstrap_ci(matches, n_boot=10000, seed=42):
    """
    Bootstrap 95% CIs for key metrics: r, MAE, direction agreement.
    Uses BCa (bias-corrected and accelerated) bootstrap.
    """
    rng = np.random.RandomState(seed)
    n = len(matches)

    our = np.array([m['our'] for m in matches])
    gt = np.array([m['gt'] for m in matches])
    errors = np.array([m['err'] for m in matches])

    def compute_r(idx):
        o, g = our[idx], gt[idx]
        if np.std(o) == 0 or np.std(g) == 0:
            return 0
        return np.corrcoef(o, g)[0, 1]

    def compute_mae(idx):
        return np.mean(errors[idx]) * 100

    def compute_dir(idx):
        o, g = our[idx], gt[idx]
        nonzero = g != 0
        if not np.any(nonzero):
            return 0
        return np.mean((o[nonzero] < 0) == (g[nonzero] < 0)) * 100

    def compute_overall_effect_diff(idx):
        return abs(np.mean(our[idx]) - np.mean(gt[idx])) * 100

    def compute_within_10(idx):
        return np.mean(errors[idx] <= 0.10) * 100

    metrics = {
        'pearson_r': compute_r,
        'mae_pct': compute_mae,
        'direction_agreement_pct': compute_dir,
        'overall_effect_diff_pp': compute_overall_effect_diff,
        'within_10pct': compute_within_10,
    }

    results = {}
    for name, func in metrics.items():
        # Observed statistic
        all_idx = np.arange(n)
        observed = func(all_idx)

        # Bootstrap
        boot_stats = []
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            boot_stats.append(func(idx))
        boot_stats = np.array(boot_stats)

        # Percentile CI
        ci_lower = float(np.percentile(boot_stats, 2.5))
        ci_upper = float(np.percentile(boot_stats, 97.5))

        # BCa CI (bias-corrected and accelerated)
        # Bias correction
        z0 = stats.norm.ppf(np.mean(boot_stats < observed))

        # Acceleration (jackknife)
        jackknife_stats = []
        for i in range(n):
            jack_idx = np.concatenate([np.arange(i), np.arange(i + 1, n)])
            jackknife_stats.append(func(jack_idx))
        jackknife_stats = np.array(jackknife_stats)
        jack_mean = np.mean(jackknife_stats)
        a_num = np.sum((jack_mean - jackknife_stats) ** 3)
        a_den = 6 * (np.sum((jack_mean - jackknife_stats) ** 2)) ** 1.5
        a = a_num / a_den if a_den != 0 else 0

        # BCa percentiles
        alpha1 = stats.norm.cdf(z0 + (z0 + stats.norm.ppf(0.025)) / (1 - a * (z0 + stats.norm.ppf(0.025))))
        alpha2 = stats.norm.cdf(z0 + (z0 + stats.norm.ppf(0.975)) / (1 - a * (z0 + stats.norm.ppf(0.975))))
        bca_lower = float(np.percentile(boot_stats, max(0, alpha1 * 100)))
        bca_upper = float(np.percentile(boot_stats, min(100, alpha2 * 100)))

        results[name] = {
            'observed': round(float(observed), 4),
            'bootstrap_mean': round(float(np.mean(boot_stats)), 4),
            'bootstrap_se': round(float(np.std(boot_stats)), 4),
            'percentile_ci_95': [round(ci_lower, 4), round(ci_upper, 4)],
            'bca_ci_95': [round(bca_lower, 4), round(bca_upper, 4)],
            'n_bootstrap': n_boot,
        }

    return results


# ============================================================
# 5. PAIRED T-TEST (systematic bias)
# ============================================================

def paired_ttest(matches):
    """Test for systematic bias between pipeline and GT."""
    our = np.array([m['our'] * 100 for m in matches])
    gt = np.array([m['gt'] * 100 for m in matches])
    diff = our - gt

    t_stat, p_value = stats.ttest_rel(our, gt)

    # Also Wilcoxon signed-rank (non-parametric)
    w_stat, w_p = stats.wilcoxon(diff)

    # Effect size (Cohen's d for paired data)
    d = np.mean(diff) / np.std(diff, ddof=1)

    return {
        'paired_t_stat': round(float(t_stat), 3),
        'paired_t_p': round(float(p_value), 6),
        'wilcoxon_stat': round(float(w_stat), 1),
        'wilcoxon_p': round(float(w_p), 6),
        'cohens_d': round(float(d), 4),
        'mean_difference_pp': round(float(np.mean(diff)), 3),
        'systematic_bias': p_value < 0.05,
        'interpretation': (
            f"Paired t-test: t={t_stat:.2f}, p={p_value:.4f}. "
            f"{'Systematic bias detected' if p_value < 0.05 else 'No systematic bias'}. "
            f"Cohen's d = {d:.3f} ({'negligible' if abs(d) < 0.2 else 'small' if abs(d) < 0.5 else 'medium' if abs(d) < 0.8 else 'large'}). "
            f"Wilcoxon: W={w_stat:.0f}, p={w_p:.4f}."
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FORMAL STATISTICAL ANALYSES FOR PAPER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print("\nLoading validation data...")
    matches, paper_effects = load_all_matches()
    print(f"  {len(matches)} matched observations from {len(paper_effects)} papers")

    # JSON encoder for numpy types
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

    # 1. Bland-Altman
    print("\n1. BLAND-ALTMAN ANALYSIS")
    print("-" * 40)
    ba_results = bland_altman_analysis(matches)
    print(f"  {ba_results['interpretation']}")
    with open(OUT_DIR / 'bland_altman_results.json', 'w') as f:
        json.dump(ba_results, f, indent=2, cls=NumpyEncoder)

    # 2. ICC
    print("\n2. INTRACLASS CORRELATION COEFFICIENT")
    print("-" * 40)
    icc_results = icc_analysis(matches, paper_effects)
    print(f"  {icc_results['summary']}")
    with open(OUT_DIR / 'icc_results.json', 'w') as f:
        json.dump(icc_results, f, indent=2, cls=NumpyEncoder)

    # 3. TOST Equivalence
    print("\n3. TOST EQUIVALENCE TESTING")
    print("-" * 40)
    tost_results = tost_equivalence(matches, margin_pp=2.0)
    print(f"  {tost_results['interpretation']}")

    # Also test with 5pp margin
    tost_5pp = tost_equivalence(matches, margin_pp=5.0)
    tost_results['also_5pp_margin'] = {
        'p_tost': tost_5pp['p_tost'],
        'equivalent': tost_5pp['equivalent_at_alpha_05'],
    }
    with open(OUT_DIR / 'tost_results.json', 'w') as f:
        json.dump(tost_results, f, indent=2, cls=NumpyEncoder)

    # 4. Bootstrap CIs
    print("\n4. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
    print("-" * 40)
    boot_results = bootstrap_ci(matches)
    for name, vals in boot_results.items():
        print(f"  {name}: {vals['observed']} (95% BCa CI: {vals['bca_ci_95'][0]}-{vals['bca_ci_95'][1]})")
    with open(OUT_DIR / 'bootstrap_ci.json', 'w') as f:
        json.dump(boot_results, f, indent=2, cls=NumpyEncoder)

    # 5. Paired t-test
    print("\n5. PAIRED T-TEST FOR SYSTEMATIC BIAS")
    print("-" * 40)
    ttest_results = paired_ttest(matches)
    print(f"  {ttest_results['interpretation']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)
    summary = f"""
Formal Statistical Analysis - Loladze 2014 Validation
=====================================================
N = {len(matches)} matched observations, {len(paper_effects)} papers

AGREEMENT:
  Bland-Altman mean bias: {ba_results['mean_difference_pp']:.2f} pp (95% CI: {ba_results['mean_diff_95ci'][0]:.2f} to {ba_results['mean_diff_95ci'][1]:.2f})
  95% Limits of Agreement: {ba_results['loa_lower_pp']:.1f} to {ba_results['loa_upper_pp']:.1f} pp
  {ba_results['within_loa_pct']:.0f}% within LOAs

RELIABILITY:
  ICC(3,1) observation-level: {icc_results['observation_level']['icc_31_consistency']:.3f} (95% CI: {icc_results['observation_level']['icc_31_95ci'][0]:.3f}-{icc_results['observation_level']['icc_31_95ci'][1]:.3f}) [{icc_results['observation_level']['interpretation']}]
  ICC paper-level: {icc_results['paper_level']['icc_consistency']}  [{icc_results['paper_level']['interpretation']}]
  Pearson r (obs): {icc_results['pearson_r_obs']:.3f}
  Pearson r (paper): {icc_results['pearson_r_paper']}

EQUIVALENCE:
  TOST ±2pp: p={tost_results['p_tost']:.4f} ({'EQUIVALENT' if tost_results['equivalent_at_alpha_05'] else 'NOT equivalent'})
  TOST ±5pp: p={tost_results['also_5pp_margin']['p_tost']:.4f} ({'EQUIVALENT' if tost_results['also_5pp_margin']['equivalent'] else 'NOT equivalent'})
  90% CI for mean diff: ({tost_results['ci_90_pp'][0]:.2f}, {tost_results['ci_90_pp'][1]:.2f}) pp

BIAS:
  Paired t-test: p={ttest_results['paired_t_p']:.4f} ({'bias detected' if ttest_results['systematic_bias'] else 'no bias'})
  Cohen's d: {ttest_results['cohens_d']:.3f}

BOOTSTRAP 95% CIs:
  Pearson r: {boot_results['pearson_r']['observed']:.3f} ({boot_results['pearson_r']['bca_ci_95'][0]:.3f}-{boot_results['pearson_r']['bca_ci_95'][1]:.3f})
  MAE: {boot_results['mae_pct']['observed']:.1f}% ({boot_results['mae_pct']['bca_ci_95'][0]:.1f}-{boot_results['mae_pct']['bca_ci_95'][1]:.1f})
  Direction: {boot_results['direction_agreement_pct']['observed']:.1f}% ({boot_results['direction_agreement_pct']['bca_ci_95'][0]:.1f}-{boot_results['direction_agreement_pct']['bca_ci_95'][1]:.1f})
  Overall effect diff: {boot_results['overall_effect_diff_pp']['observed']:.2f}pp ({boot_results['overall_effect_diff_pp']['bca_ci_95'][0]:.2f}-{boot_results['overall_effect_diff_pp']['bca_ci_95'][1]:.2f})
"""
    print(summary)

    with open(OUT_DIR / 'formal_stats_summary.txt', 'w') as f:
        f.write(summary)
    print(f"\nAll results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
