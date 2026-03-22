#!/usr/bin/env python
"""
reproduce_all.py - Master reproducibility script for meta-analysis extractor.

Recomputes EVERY statistic reported in the paper from raw matched-pair data files.
No API keys or PDFs required. Operates entirely on pre-aligned output files.

Usage:
    ./venv/Scripts/python.exe reproduce_all.py
    python reproduce_all.py

Output:
    - Formatted tables to stdout
    - output/reproduction_results.json (full machine-readable results)

Expected runtime: <10 seconds (pure computation, no network).
"""
import sys
import os
import json
import csv
import math
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# ---------------------------------------------------------------------------
# Windows encoding fix
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')

import numpy as np
from scipy import stats as sp_stats

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# SECTION 1: Statistical Functions
# ============================================================

def pearson_r(x, y):
    """Pearson correlation with p-value and Fisher z-transform 95% CI."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return {'r': None, 'p': None, 'ci95_low': None, 'ci95_high': None, 'n': n}
    r_val, p_val = sp_stats.pearsonr(x, y)
    if abs(r_val) >= 0.9999:
        ci_low, ci_high = float(r_val), float(r_val)
    else:
        z = np.arctanh(r_val)
        se_z = 1.0 / np.sqrt(n - 3)
        ci_low = round(float(np.tanh(z - 1.96 * se_z)), 4)
        ci_high = round(float(np.tanh(z + 1.96 * se_z)), 4)
    return {
        'r': round(float(r_val), 4),
        'p': float(p_val),
        'ci95_low': ci_low,
        'ci95_high': ci_high,
        'n': n
    }


def icc_21(x, y):
    """ICC(2,1) two-way random, single measures, absolute agreement (Shrout & Fleiss 1979)."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return {'icc': None, 'ci95_low': None, 'ci95_high': None}
    k = 2
    data = np.column_stack([x, y])
    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    icc_val = (ms_rows - ms_error) / denom if denom > 0 else 0
    f_val = ms_rows / ms_error if ms_error > 0 else float('inf')
    try:
        f_l = f_val / sp_stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1))
        f_u = f_val / sp_stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1))
        ci_low = (f_l - 1) / (f_l + k - 1)
        ci_high = (f_u - 1) / (f_u + k - 1)
    except Exception:
        ci_low, ci_high = None, None
    return {
        'icc': round(float(icc_val), 4),
        'ci95_low': round(float(ci_low), 4) if ci_low is not None else None,
        'ci95_high': round(float(ci_high), 4) if ci_high is not None else None,
    }


def tost_test(diffs, margin):
    """Two one-sided tests for equivalence at +/- margin (in same units as diffs)."""
    diffs = np.array(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    df = n - 1
    if se == 0:
        return {'margin': margin, 'mean_diff': round(mean_d, 4), 'se': 0.0,
                'p_tost': 0.0, 'ci90_low': round(mean_d, 4), 'ci90_high': round(mean_d, 4),
                'equivalent': True}
    t_upper = (mean_d - margin) / se
    p_upper = float(sp_stats.t.cdf(t_upper, df))
    t_lower = (mean_d + margin) / se
    p_lower = float(1 - sp_stats.t.cdf(t_lower, df))
    p_tost = max(p_upper, p_lower)
    t_crit = float(sp_stats.t.ppf(0.95, df))
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se
    return {
        'margin': round(margin, 4),
        'mean_diff': round(mean_d, 4),
        'se': round(se, 4),
        'p_tost': round(p_tost, 6),
        'ci90_low': round(ci_low, 4),
        'ci90_high': round(ci_high, 4),
        'equivalent': bool(p_tost < 0.05)
    }


def cohens_d(diffs):
    """Cohen's d for paired differences."""
    diffs = np.array(diffs, dtype=float)
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0
    return round(float(np.mean(diffs)) / sd, 4)


def within_thresholds(abs_errors, thresholds=(1, 3, 5, 10)):
    """Percentage of observations within each threshold (pp)."""
    abs_errors = np.array(abs_errors, dtype=float)
    n = len(abs_errors)
    return {t: round(float(np.sum(abs_errors <= t) / n * 100), 1) if n > 0 else 0 for t in thresholds}


def per_paper_tiers(paper_maes):
    """Classify papers by MAE tier: Excellent (<2), Good (2-5), Fair (5-10), Poor (>10)."""
    tiers = {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
    for mae in paper_maes:
        if mae < 2:
            tiers['Excellent'] += 1
        elif mae < 5:
            tiers['Good'] += 1
        elif mae < 10:
            tiers['Fair'] += 1
        else:
            tiers['Poor'] += 1
    return tiers


def compute_full_stats(ext_effects, gt_effects, paper_ids, dataset_name, dir_threshold=0.5):
    """Compute all formal statistics for a dataset.

    All effect sizes must be in percentage points.
    Returns a dict with all statistics.
    """
    ext = np.array(ext_effects, dtype=float)
    gt = np.array(gt_effects, dtype=float)
    n_obs = len(ext)
    diffs = ext - gt
    abs_errors = np.abs(diffs)

    # Per-paper MAEs
    paper_obs = defaultdict(list)
    for i, pid in enumerate(paper_ids):
        paper_obs[pid].append(float(abs_errors[i]))
    paper_maes = {pid: float(np.mean(errs)) for pid, errs in paper_obs.items()}
    n_papers = len(paper_maes)

    # Direction agreement (skip near-zero GT)
    dir_total = sum(1 for g in gt if abs(g) > dir_threshold)
    dir_correct = sum(1 for e, g in zip(ext, gt)
                      if abs(g) > dir_threshold and
                      ((e > 0 and g > 0) or (e < 0 and g < 0) or (e == 0 and g == 0)))

    gt_mean = float(np.mean(gt))
    ext_mean = float(np.mean(ext))
    mean_abs_effect = float(np.mean(np.abs(gt)))

    # TOST battery
    prop_20 = round(mean_abs_effect * 0.20, 4)
    prop_10 = round(mean_abs_effect * 0.10, 4)
    tost_battery = {}
    for margin in [2.0, 3.0]:
        label = f'fixed_{margin:.0f}pp'
        r = tost_test(diffs, margin)
        r['margin_type'] = 'fixed'
        tost_battery[label] = r
    t20 = tost_test(diffs, prop_20)
    t20['margin_type'] = 'proportional'
    t20['margin_pct'] = 20
    tost_battery['proportional_20pct'] = t20
    t10 = tost_test(diffs, prop_10)
    t10['margin_type'] = 'proportional'
    t10['margin_pct'] = 10
    tost_battery['proportional_10pct'] = t10

    return {
        'dataset': dataset_name,
        'n_obs': n_obs,
        'n_papers': n_papers,
        'pearson': pearson_r(ext, gt),
        'icc': icc_21(ext, gt),
        'mae': round(float(np.mean(abs_errors)), 4),
        'median_ae': round(float(np.median(abs_errors)), 4),
        'direction': {
            'correct': dir_correct,
            'total': dir_total,
            'pct': round(dir_correct / dir_total * 100, 1) if dir_total > 0 else None
        },
        'overall_effect': {
            'gt_mean': round(gt_mean, 4),
            'ext_mean': round(ext_mean, 4),
            'diff_pp': round(abs(ext_mean - gt_mean), 4),
        },
        'tost_battery': tost_battery,
        'mean_abs_effect': round(mean_abs_effect, 4),
        'cohens_d': cohens_d(diffs),
        'thresholds': within_thresholds(abs_errors),
        'tiers': per_paper_tiers(list(paper_maes.values())),
        'rmse': round(float(np.sqrt(np.mean(diffs ** 2))), 4),
        'max_ae': round(float(np.max(abs_errors)), 2),
        'sd_error': round(float(np.std(diffs, ddof=1)), 4),
    }


# ============================================================
# SECTION 2: Dataset Loaders
#   Each loader reads the raw match files and returns
#   (ext_effects_pp, gt_effects_pp, paper_ids) or None on failure.
# ============================================================

def load_loladze():
    """Load Loladze 2014 CO2/minerals from validation_llm_10pp.json.

    JSON has 'matched_observations' list with fields:
      paper_id, gt_effect_pct, ext_effect_pct (already in percentage points).
    """
    json_path = BASE_DIR / 'data' / 'loladze_agent_replication' / 'validation_llm_10pp.json'
    if not json_path.exists():
        return None, f"File not found: {json_path}"

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matched_obs = data.get('matched_observations', [])
    if not matched_obs:
        return None, "No matched_observations in Loladze validation JSON"

    ext, gt, pids = [], [], []
    for obs in matched_obs:
        try:
            ext.append(float(obs['ext_effect_pct']))
            gt.append(float(obs['gt_effect_pct']))
            pids.append(obs.get('paper_id', 'unknown'))
        except (ValueError, KeyError, TypeError):
            continue

    return (ext, gt, pids), None


def load_hui():
    """Load Hui 2023 Zn/wheat from validation_matches_improved.csv (or original).

    CSV columns: ext_ctrl,ext_treat,gt_ctrl,gt_treat,ext_effect,gt_effect,...
    Effects are already in percentage points.
    """
    improved = BASE_DIR / 'data' / 'hui2023_full_35' / 'validation_matches_improved.csv'
    original = BASE_DIR / 'data' / 'hui2023_full_35' / 'validation_matches.csv'

    if improved.exists():
        use_path = improved
        has_paper_id = False
    elif original.exists():
        use_path = original
        has_paper_id = True
    else:
        return None, "Hui CSV not found in data/hui2023_full_35/"

    # If using improved (no paper_id), build mapping from original
    paper_id_map = {}
    if not has_paper_id and original.exists():
        with open(original, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (row.get('ext_ctrl', ''), row.get('ext_treat', ''),
                       row.get('gt_ctrl', ''), row.get('gt_treat', ''))
                paper_id_map[key] = row.get('paper_id', 'unknown')

    ext, gt, pids = [], [], []
    with open(use_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                ext.append(float(row['ext_effect']))
                gt.append(float(row['gt_effect']))
                if has_paper_id:
                    pids.append(row.get('paper_id', 'unknown'))
                else:
                    key = (row.get('ext_ctrl', ''), row.get('ext_treat', ''),
                           row.get('gt_ctrl', ''), row.get('gt_treat', ''))
                    pids.append(paper_id_map.get(key, 'unknown'))
            except (ValueError, KeyError):
                continue

    return (ext, gt, pids), None


def load_li2022():
    """Load Li 2022 biostimulant/yield from validation_matches_effect_first.csv.

    CSV columns: paper_id,gt_study,gt_effect_pct,ext_effect_pct,abs_error,match_quality,scale_factor
    All rows are used (no programmatic classification filter needed).
    """
    csv_path = BASE_DIR / 'data' / 'li2022_combined' / 'validation_matches_effect_first.csv'

    if not csv_path.exists():
        return None, f"File not found: {csv_path}"

    ext, gt, pids = [], [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                ext.append(float(row['ext_effect_pct']))
                gt.append(float(row['gt_effect_pct']))
                pids.append(row['paper_id'])
            except (ValueError, KeyError):
                continue

    return (ext, gt, pids), None


def load_biochar():
    """Load Biochar (Li 2024) from validation_results.json matched_observations.

    Each obs has gt_effect_pct and ext_effect_pct already in percentage points.
    """
    val_path = BASE_DIR / 'data' / 'biochar_extraction' / 'validation_results.json'
    if not val_path.exists():
        return None, f"File not found: {val_path}"

    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    matched_obs = val_data.get('matched_observations', [])
    if not matched_obs:
        return None, "No matched_observations in biochar validation_results.json"

    ext, gt, pids = [], [], []
    for obs in matched_obs:
        try:
            gt_eff = float(obs['gt_effect_pct'])
            ext_eff = float(obs['ext_effect_pct'])
            gt.append(gt_eff)
            ext.append(ext_eff)
            pids.append(obs.get('paper_id', obs.get('gt_study', 'unknown')))
        except (ValueError, KeyError, TypeError):
            continue

    return (ext, gt, pids), None


def load_boldorini():
    """Load Boldorini 2024 predator/yield by re-running the matching logic.

    Reads GT from data/boldorini_gt.csv and extraction JSONs
    from output/boldorini_extraction/, matches by author+year then lnRR similarity.

    GT convention: yi = ln(control_mean / treatment_mean)
    Effects are converted to percentage points: (exp(lnRR) - 1) * 100.
    """
    gt_path = BASE_DIR / 'data' / 'boldorini_gt' / 'boldorini_gt.csv'
    ext_dir = BASE_DIR / 'data' / 'boldorini_extraction'

    if not gt_path.exists():
        return None, f"Boldorini GT not found: {gt_path}"
    if not ext_dir.exists():
        return None, f"Boldorini extraction dir not found: {ext_dir}"

    AUTHOR_MAP = {
        "B01_Ali_2018": ("Ali", 2018),
        "B02_Bisseleua_2017": ("Bisseleua", 2017),
        "B03_Borkhataria_2012": ("Borkhataria", 2012),
        "B04_Classen_2014": ("Classen", 2014),
        "B05_Garfinkel_2015": ("Garfinkel", 2015),
        "B06_Garfinkel_2020": ("Garfinkel", 2020),
        "B07_Gras_2016": ("Gras", 2016),
        "B08_Hooks_2003": ("Hooks_et_al", 2003),
        "B09_Ismoilov_2020": ("Ismoilov", 2020),
        "B10_Lang_2003": ("Lang", 2003),
        "B11_Libran-Embid_2017": ("Libran-Embid", 2017),
        "B13_Maas_2013": ("Maas", 2013),
        "B14_Martin_2013": ("Martin", 2013),
        "B15_Mols_2002": ("Mols", 2002),
        "B16_Saunders_2016": ("Saunders", 2016),
        "B17_Snyder_2001": ("Snyder_Wise", 2001),
        "B18_Suenaga_2015": ("Suenaga_Hamamura", 2015),
        "B19_Vichitbandha_2002": ("Vichitbandha_Wise", 2002),
    }

    # Load GT
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_rows = list(csv.DictReader(f, delimiter=';'))

    # Load extractions
    extractions = {}
    for fname in sorted(os.listdir(ext_dir)):
        if fname.endswith('.json') and not fname.startswith('validation'):
            fpath = ext_dir / fname
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pid = data.get('paper_id', fname.replace('.json', ''))
            extractions[pid] = data

    # Match GT rows to extracted observations
    matched_pairs = []
    used_obs_keys = set()

    def compute_lnRR(treat_mean, control_mean):
        if treat_mean is None or control_mean is None:
            return None
        if treat_mean <= 0 or control_mean <= 0:
            if treat_mean < 0 and control_mean < 0:
                return control_mean - treat_mean
            return None
        return math.log(control_mean / treat_mean)

    for gt_row in gt_rows:
        gt_author = gt_row['author']
        gt_year = int(gt_row['year'])
        gt_yi = float(gt_row['yi'])

        candidates = []
        for pid, pdata in extractions.items():
            if pid not in AUTHOR_MAP:
                continue
            mapped_author, mapped_year = AUTHOR_MAP[pid]
            if mapped_author == gt_author and mapped_year == gt_year:
                for i, obs in enumerate(pdata.get('observations', [])):
                    obs_key = f"{pid}_obs{i}"
                    if obs_key not in used_obs_keys:
                        candidates.append((obs_key, obs))

        if not candidates:
            continue

        best_key, best_obs, best_score = None, None, float('inf')
        for obs_key, obs in candidates:
            ext_lnRR = compute_lnRR(obs.get('treatment_mean'), obs.get('control_mean'))
            if ext_lnRR is None:
                continue
            score = abs(ext_lnRR - gt_yi)
            if score < best_score:
                best_score = score
                best_key = obs_key
                best_obs = obs

        if best_obs is not None and best_score < 3.0:
            ext_lnRR = compute_lnRR(best_obs['treatment_mean'], best_obs['control_mean'])
            used_obs_keys.add(best_key)
            # Convert lnRR to percentage points
            gt_pct = (math.exp(gt_yi) - 1) * 100
            ext_pct = (math.exp(ext_lnRR) - 1) * 100
            matched_pairs.append((ext_pct, gt_pct, gt_author, ext_lnRR, gt_yi))

    if len(matched_pairs) < 3:
        return None, f"Boldorini: only {len(matched_pairs)} matched pairs (need >= 3)"

    ext = [p[0] for p in matched_pairs]
    gt = [p[1] for p in matched_pairs]
    pids = [p[2] for p in matched_pairs]
    # Also return raw lnRR arrays for correlation on native scale
    ext_lnRR = [p[3] for p in matched_pairs]
    gt_lnRR = [p[4] for p in matched_pairs]
    return (ext, gt, pids, ext_lnRR, gt_lnRR), None


# ============================================================
# SECTION 3: Paper-Claimed Reference Values
# ============================================================

# These are the numbers reported in the paper.
# The script checks computed values against these and flags discrepancies.
PAPER_CLAIMS = {
    'Loladze': {
        'n_papers': 45, 'n_obs': 413,
        'r': 0.984, 'mae': 1.36, 'icc': 0.984,
        'direction_pct': 95.0,
        'gt_mean': -7.83, 'ext_mean': -7.82, 'effect_diff_pp': 0.01,
        'cohens_d': 0.004,
        'tost_2pp_pass': True,
    },
    'Hui': {
        'n_papers': 17, 'n_obs': 319,
        'r': 0.999, 'mae': 0.43, 'icc': 0.999,
        'direction_pct': 99.7,
        'gt_mean': 49.61, 'ext_mean': 49.72, 'effect_diff_pp': 0.12,
        'cohens_d': 0.072,
        'tost_2pp_pass': True,
    },
    'Li 2022': {
        # Uses validation_matches_effect_first.csv (effect-first matching, all rows).
        'n_papers': 31, 'n_obs': 117,
        'r': 0.994, 'mae': 1.01,
    },
    'Biochar': {
        'n_papers': 26, 'n_obs': 254,
        'r': 0.997, 'mae': 1.20, 'icc': 0.997,
        'direction_pct': 92.3,
        'gt_mean': 12.27, 'ext_mean': 12.05, 'effect_diff_pp': 0.22,
        'cohens_d': -0.125,
        'tost_2pp_pass': True,
    },
    'Boldorini': {
        'n_papers': 18, 'n_obs': 46,
        'r_lnRR': 0.972,  # correlation computed on lnRR scale
        'mae': 3.06,
        'direction_pct': 95.7,
        'cohens_d_lnRR': 0.217,  # Cohen's d on lnRR diffs
        'tost_pass': True,
    },
}

# Tolerance for numeric comparisons (absolute)
TOL = {
    'r': 0.005,         # e.g. 0.812 vs 0.810
    'mae': 0.1,         # 6.16 vs 6.20
    'icc': 0.005,
    'direction_pct': 1.0,
    'effect_diff_pp': 0.1,
    'mean': 0.1,        # for gt_mean, ext_mean
    'cohens_d': 0.01,
    'n_obs': 2,         # allow +-2 obs difference
    'n_papers': 1,      # allow +-1 paper difference
}


def check_claim(computed, claimed, tolerance):
    """Check if computed value is within tolerance of claimed value."""
    if computed is None or claimed is None:
        return None  # cannot check
    return abs(computed - claimed) <= tolerance


# ============================================================
# SECTION 4: Output Formatting
# ============================================================

def fmt(val, decimals=2):
    if val is None:
        return '-'
    if isinstance(val, bool):
        return 'PASS' if val else 'FAIL'
    return f"{val:.{decimals}f}"


def print_dataset_report(stats, claims=None):
    """Print detailed stats for one dataset."""
    s = stats
    print(f"\n{'='*74}")
    print(f"  {s['dataset']}")
    print(f"  {s['n_obs']} observations, {s['n_papers']} papers")
    print(f"{'='*74}")

    def report_line(label, computed, claim_key=None, claim_tol_key=None):
        line = f"  {label:<24} {fmt(computed, 4)}"
        if claims and claim_key and claim_key in claims:
            expected = claims[claim_key]
            tol_key = claim_tol_key or claim_key
            tol = TOL.get(tol_key, 0.01)
            ok = check_claim(computed, expected, tol)
            status = 'OK' if ok else 'MISMATCH'
            line += f"   (expected: {fmt(expected, 4)}, {status})"
        print(line)

    if 'r_lnRR' in (claims or {}):
        report_line('Pearson r (pp)', s['pearson']['r'])
        report_line('Pearson r (lnRR)', s.get('pearson_lnRR', {}).get('r'), 'r_lnRR', 'r')
    else:
        report_line('Pearson r', s['pearson']['r'], 'r', 'r')
    report_line('ICC(2,1)', s['icc']['icc'], 'icc', 'icc')
    report_line('MAE (pp)', s['mae'], 'mae', 'mae')
    report_line('Median AE (pp)', s['median_ae'])
    report_line('RMSE (pp)', s['rmse'])

    d = s['direction']
    if d.get('correct') is not None:
        print(f"  {'Direction':<24} {d['correct']}/{d['total']} ({fmt(d['pct'], 1)}%)", end='')
    else:
        print(f"  {'Direction':<24} {fmt(d['pct'], 1)}%", end='')
    if claims and 'direction_pct' in claims:
        ok = check_claim(d['pct'], claims['direction_pct'], TOL['direction_pct'])
        print(f"   (expected: {fmt(claims['direction_pct'], 1)}%, {'OK' if ok else 'MISMATCH'})")
    else:
        print()

    o = s['overall_effect']
    report_line('GT mean effect (pp)', o['gt_mean'], 'gt_mean', 'mean')
    report_line('Ext mean effect (pp)', o['ext_mean'], 'ext_mean', 'mean')
    report_line('Effect diff (pp)', o['diff_pp'], 'effect_diff_pp', 'effect_diff_pp')

    # TOST battery
    print(f"\n  TOST Battery (mean |GT effect| = {fmt(s.get('mean_abs_effect'), 2)}pp):")
    for key, tb in s['tost_battery'].items():
        eq = 'PASS' if tb['equivalent'] else 'FAIL'
        label = f"    +/-{fmt(tb['margin'], 2)}pp ({tb['margin_type']})"
        print(f"  {label:<40} p={fmt(tb['p_tost'], 4)}, {eq}")

    if 'cohens_d_lnRR' in (claims or {}):
        report_line("Cohen's d (pp)", s['cohens_d'])
        report_line("Cohen's d (lnRR)", s.get('cohens_d_lnRR'), 'cohens_d_lnRR', 'cohens_d')
    else:
        report_line("Cohen's d", s['cohens_d'], 'cohens_d', 'cohens_d')

    th = s['thresholds']
    print(f"  {'Within 1pp':<24} {fmt(th.get(1), 1)}%")
    print(f"  {'Within 3pp':<24} {fmt(th.get(3), 1)}%")
    print(f"  {'Within 5pp':<24} {fmt(th.get(5), 1)}%")
    print(f"  {'Within 10pp':<24} {fmt(th.get(10), 1)}%")

    ti = s['tiers']
    total_t = sum(ti.values())
    print(f"  Tiers: Excellent={ti['Excellent']}, Good={ti['Good']}, "
          f"Fair={ti['Fair']}, Poor={ti['Poor']} (n={total_t})")


def print_comparison_table(all_stats):
    """Cross-dataset comparison table."""
    print(f"\n{'='*104}")
    print("  CROSS-DATASET COMPARISON TABLE")
    print(f"{'='*104}")

    names = []
    for s in all_stats:
        d = s['dataset']
        for short in ['Loladze', 'Hui', 'Li 2022', 'Biochar', 'Boldorini']:
            if short.lower() in d.lower():
                names.append(short)
                break
        else:
            names.append(d[:12])

    col_w = 14
    header = f"{'Metric':<30}" + ''.join(f"{n:>{col_w}}" for n in names)
    print(header)
    print('-' * len(header))

    def row(label, getter):
        vals = []
        for s in all_stats:
            try:
                vals.append(str(getter(s)))
            except (KeyError, TypeError):
                vals.append('-')
        print(f"{label:<30}" + ''.join(f"{v:>{col_w}}" for v in vals))

    row('Obs (n)',              lambda s: s['n_obs'])
    row('Papers',               lambda s: s['n_papers'])
    row('Pearson r',            lambda s: fmt(s['pearson']['r'], 3))
    row('ICC(2,1)',             lambda s: fmt(s['icc']['icc'], 3))
    row('MAE (pp)',             lambda s: fmt(s['mae'], 2))
    row('Median AE (pp)',       lambda s: fmt(s['median_ae'], 2))
    row('RMSE (pp)',            lambda s: fmt(s['rmse'], 2))
    row('Direction (%)',        lambda s: fmt(s['direction']['pct'], 1))
    row('GT mean effect',       lambda s: fmt(s['overall_effect']['gt_mean'], 2))
    row('Ext mean effect',      lambda s: fmt(s['overall_effect']['ext_mean'], 2))
    row('Effect diff (pp)',     lambda s: fmt(s['overall_effect']['diff_pp'], 2))
    row("Cohen's d",           lambda s: fmt(s['cohens_d'], 3))
    row('TOST +/-2pp pass',    lambda s: fmt(s['tost_battery'].get('fixed_2pp', {}).get('equivalent')))
    row('TOST +/-3pp pass',    lambda s: fmt(s['tost_battery'].get('fixed_3pp', {}).get('equivalent')))
    row('TOST +/-20% pass',    lambda s: fmt(s['tost_battery'].get('proportional_20pct', {}).get('equivalent')))
    row('TOST +/-10% pass',    lambda s: fmt(s['tost_battery'].get('proportional_10pct', {}).get('equivalent')))
    row('Within 1pp (%)',       lambda s: fmt(s['thresholds'].get(1), 1))
    row('Within 3pp (%)',       lambda s: fmt(s['thresholds'].get(3), 1))
    row('Within 5pp (%)',       lambda s: fmt(s['thresholds'].get(5), 1))
    row('Within 10pp (%)',      lambda s: fmt(s['thresholds'].get(10), 1))
    row('Tier: Excellent',      lambda s: s['tiers'].get('Excellent', 0))
    row('Tier: Good',           lambda s: s['tiers'].get('Good', 0))
    row('Tier: Fair',           lambda s: s['tiers'].get('Fair', 0))
    row('Tier: Poor',           lambda s: s['tiers'].get('Poor', 0))
    print()


# ============================================================
# SECTION 5: Verification Against Paper Claims
# ============================================================

def verify_dataset(stats, claims):
    """Check all claims for one dataset. Returns list of (metric, status, detail) tuples."""
    checks = []
    name = stats['dataset']

    def chk(metric, computed, claim_key, tol_key=None):
        if claim_key not in claims:
            return
        expected = claims[claim_key]
        tol = TOL.get(tol_key or claim_key, 0.01)
        ok = check_claim(computed, expected, tol)
        status = 'PASS' if ok else 'FAIL'
        detail = f"computed={fmt(computed, 4)}, expected={fmt(expected, 4)}, tol={tol}"
        checks.append((metric, status, detail))

    chk('n_obs', stats['n_obs'], 'n_obs', 'n_obs')
    chk('n_papers', stats['n_papers'], 'n_papers', 'n_papers')

    # Pearson r: use lnRR-scale r for Boldorini if available
    if 'r_lnRR' in claims and 'pearson_lnRR' in stats:
        chk('Pearson r (lnRR)', stats['pearson_lnRR']['r'], 'r_lnRR', 'r')
    else:
        chk('Pearson r', stats['pearson']['r'], 'r', 'r')

    chk('MAE', stats['mae'], 'mae', 'mae')
    chk('ICC', stats['icc']['icc'], 'icc', 'icc')
    chk('Direction %', stats['direction']['pct'], 'direction_pct', 'direction_pct')
    chk('Effect diff pp', stats['overall_effect']['diff_pp'], 'effect_diff_pp', 'effect_diff_pp')
    chk('GT mean', stats['overall_effect']['gt_mean'], 'gt_mean', 'mean')
    chk('Ext mean', stats['overall_effect']['ext_mean'], 'ext_mean', 'mean')

    # Cohen's d: use lnRR-scale for Boldorini if available
    if 'cohens_d_lnRR' in claims and 'cohens_d_lnRR' in stats:
        chk("Cohen's d (lnRR)", stats['cohens_d_lnRR'], 'cohens_d_lnRR', 'cohens_d')
    else:
        chk("Cohen's d", stats['cohens_d'], 'cohens_d', 'cohens_d')

    # TOST +/-2pp pass
    if 'tost_2pp_pass' in claims:
        tost_2 = stats['tost_battery'].get('fixed_2pp', {})
        computed = tost_2.get('equivalent')
        expected = claims['tost_2pp_pass']
        ok = computed == expected
        checks.append(('TOST +/-2pp', 'PASS' if ok else 'FAIL',
                       f"computed={computed}, expected={expected}"))

    return checks


# ============================================================
# SECTION 6: Main
# ============================================================

def main():
    print("=" * 74)
    print("  REPRODUCE ALL: Meta-Analysis Extractor Validation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Recomputing all statistics from raw matched-pair data files.")
    print("=" * 74)

    all_stats = []
    all_checks = {}
    errors = []

    # Dataset definitions: (name, loader_func, claims_key, dir_threshold)
    datasets = [
        ('Loladze 2014 (CO2/minerals)', load_loladze, 'Loladze', 0.5),
        ('Hui 2023 (Zn/wheat)', load_hui, 'Hui', 0.5),
        ('Li 2022 (biostimulant/yield)', load_li2022, 'Li 2022', 0.5),
        ('Biochar (Li 2024)', load_biochar, 'Biochar', 0.5),
        ('Boldorini 2024 (predator/yield)', load_boldorini, 'Boldorini', 0.5),
    ]

    for ds_name, loader, claims_key, dir_thresh in datasets:
        print(f"\n--- Loading {ds_name} ---")
        result, err_msg = loader()

        if result is None:
            print(f"  [SKIP] {err_msg}")
            errors.append(f"{ds_name}: {err_msg}")
            continue

        # Boldorini returns 5 items (includes raw lnRR); others return 3
        if len(result) == 5:
            ext, gt, pids, ext_lnRR, gt_lnRR = result
        else:
            ext, gt, pids = result
            ext_lnRR, gt_lnRR = None, None

        print(f"  Loaded: {len(ext)} observations, {len(set(pids))} papers")

        if len(ext) < 3:
            print(f"  [SKIP] Too few observations ({len(ext)})")
            errors.append(f"{ds_name}: only {len(ext)} observations")
            continue

        stats = compute_full_stats(ext, gt, pids, ds_name, dir_threshold=dir_thresh)

        # For Boldorini: also compute r on the native lnRR scale (as in the paper)
        if ext_lnRR is not None:
            stats['pearson_lnRR'] = pearson_r(ext_lnRR, gt_lnRR)
            # Also compute Cohen's d on lnRR diffs (matching validate_boldorini.py)
            lnRR_diffs = np.array(ext_lnRR) - np.array(gt_lnRR)
            stats['cohens_d_lnRR'] = cohens_d(lnRR_diffs)

        all_stats.append(stats)

        claims = PAPER_CLAIMS.get(claims_key, {})
        print_dataset_report(stats, claims)

        # Verify
        checks = verify_dataset(stats, claims)
        all_checks[claims_key] = checks

    # --- Cross-dataset table ---
    if len(all_stats) >= 2:
        print_comparison_table(all_stats)

    # --- Aggregate summary ---
    total_obs = sum(s['n_obs'] for s in all_stats)
    total_papers = sum(s['n_papers'] for s in all_stats)

    print(f"\n{'='*74}")
    print(f"  AGGREGATE SUMMARY")
    print(f"{'='*74}")
    print(f"  Datasets loaded:    {len(all_stats)} / 5")
    print(f"  Total observations: {total_obs}")
    print(f"  Total papers:       {total_papers}")

    # Weighted averages
    r_vals = [(s['pearson']['r'], s['n_obs']) for s in all_stats if s['pearson']['r'] is not None]
    if r_vals:
        weighted_r = sum(r * n for r, n in r_vals) / sum(n for _, n in r_vals)
        print(f"  Weighted mean r:    {weighted_r:.3f}")

    mae_vals = [(s['mae'], s['n_obs']) for s in all_stats if s['mae'] is not None]
    if mae_vals:
        weighted_mae = sum(m * n for m, n in mae_vals) / sum(n for _, n in mae_vals)
        print(f"  Weighted mean MAE:  {weighted_mae:.2f} pp")

    # TOST summary
    all_tost_pass = True
    for s in all_stats:
        tost_2 = s['tost_battery'].get('fixed_2pp', {})
        if not tost_2.get('equivalent', False):
            all_tost_pass = False
    print(f"  All TOST +/-2pp:    {'PASS' if all_tost_pass else 'FAIL'}")

    # --- Verification report ---
    print(f"\n{'='*74}")
    print(f"  VERIFICATION AGAINST PAPER CLAIMS")
    print(f"{'='*74}")

    total_checks = 0
    total_pass = 0
    total_fail = 0
    fail_details = []

    for ds_key, checks in all_checks.items():
        n_pass = sum(1 for _, s, _ in checks if s == 'PASS')
        n_fail = sum(1 for _, s, _ in checks if s == 'FAIL')
        total_checks += len(checks)
        total_pass += n_pass
        total_fail += n_fail

        status = 'ALL PASS' if n_fail == 0 else f'{n_fail} FAIL'
        print(f"\n  {ds_key}: {status} ({n_pass}/{len(checks)} checks passed)")

        for metric, status, detail in checks:
            marker = '  ' if status == 'PASS' else '>>'
            print(f"    {marker} {metric:<20} {status}  {detail}")
            if status == 'FAIL':
                fail_details.append(f"{ds_key}/{metric}: {detail}")

    # --- Final verdict ---
    print(f"\n{'='*74}")
    if errors:
        print(f"  WARNINGS: {len(errors)} dataset(s) could not be loaded:")
        for e in errors:
            print(f"    - {e}")

    if total_fail == 0 and len(all_stats) == 5:
        print(f"\n  PAPER VERIFIED: All {total_checks} checks pass across {len(all_stats)} datasets.")
        print(f"  All numbers match within stated tolerances.")
    elif total_fail == 0:
        print(f"\n  PARTIAL VERIFICATION: All {total_checks} checks pass for "
              f"{len(all_stats)}/{5} loaded datasets.")
    else:
        print(f"\n  DISCREPANCIES FOUND: {total_fail}/{total_checks} checks failed:")
        for fd in fail_details:
            print(f"    - {fd}")

    print(f"{'='*74}")

    # --- Save JSON results ---
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(v) for v in obj]
        return obj

    output = {
        'timestamp': datetime.now().isoformat(),
        'n_datasets_loaded': len(all_stats),
        'n_datasets_expected': 5,
        'total_obs': total_obs,
        'total_papers': total_papers,
        'all_checks_pass': total_fail == 0,
        'checks_summary': {
            'total': total_checks,
            'passed': total_pass,
            'failed': total_fail,
        },
        'errors': errors,
        'datasets': all_stats,
        'verification': {
            ds_key: [{'metric': m, 'status': s, 'detail': d} for m, s, d in checks]
            for ds_key, checks in all_checks.items()
        },
    }

    output = convert_numpy(output)

    out_path = BASE_DIR / 'output' / 'reproduction_results.json'
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Full results saved to: {out_path}")
    print(f"  Done.")

    # Return exit code
    return 0 if total_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
