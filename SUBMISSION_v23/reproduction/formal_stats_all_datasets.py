"""
Comprehensive formal statistics across ALL validated datasets.
Computes TOST, ICC, Cohen's d, Bland-Altman, etc. consistently for publication.

Usage:
    ./venv/Scripts/python.exe formal_stats_all_datasets.py

Outputs:
    output/formal_stats_all_datasets.json
    Prints cross-dataset comparison table
"""
import sys, os, json, csv, math
import numpy as np
from scipy import stats as sp_stats
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Statistical Functions (consistent across all datasets)
# ============================================================

def pearson_r(x, y):
    """Pearson correlation with p-value and 95% CI."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return {'r': None, 'p': None, 'ci95_low': None, 'ci95_high': None, 'n': n}
    r_val, p_val = sp_stats.pearsonr(x, y)
    # Fisher z-transform for CI
    if abs(r_val) >= 0.9999:
        # Perfect or near-perfect correlation: CI is trivially [1,1]
        ci_low, ci_high = float(r_val), float(r_val)
    else:
        z = np.arctanh(r_val)
        se_z = 1.0 / np.sqrt(n - 3)
        z_low = z - 1.96 * se_z
        z_high = z + 1.96 * se_z
        ci_low = round(float(np.tanh(z_low)), 4)
        ci_high = round(float(np.tanh(z_high)), 4)
    return {
        'r': round(float(r_val), 4),
        'p': float(p_val),
        'ci95_low': ci_low,
        'ci95_high': ci_high,
        'n': n
    }


def icc_21(x, y):
    """ICC(2,1) - two-way random, single measures, absolute agreement.
    Shrout & Fleiss (1979) formulation."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return {'icc': None, 'ci95_low': None, 'ci95_high': None}

    k = 2  # number of raters
    # Stack into n x 2 matrix
    data = np.column_stack([x, y])
    grand_mean = np.mean(data)

    # Row means (subjects)
    row_means = np.mean(data, axis=1)
    # Column means (raters)
    col_means = np.mean(data, axis=0)

    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # BMS
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # JMS
    ss_error = ss_total - ss_rows - ss_cols              # EMS

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 1e-10

    # ICC(2,1) = (MSR - MSE) / (MSR + (k-1)*MSE + k*(MSC-MSE)/n)
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    icc_val = (ms_rows - ms_error) / denom if denom > 0 else 0

    # F-test
    f_val = ms_rows / ms_error if ms_error > 0 else float('inf')

    # Confidence intervals (Shrout & Fleiss)
    try:
        f_l = f_val / sp_stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1))
        f_u = f_val / sp_stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1))
        # Approximate CI for ICC(2,1)
        ci_low = (f_l - 1) / (f_l + k - 1)
        ci_high = (f_u - 1) / (f_u + k - 1)
    except Exception:
        ci_low, ci_high = None, None

    return {
        'icc': round(float(icc_val), 4),
        'ci95_low': round(float(ci_low), 4) if ci_low is not None else None,
        'ci95_high': round(float(ci_high), 4) if ci_high is not None else None,
        'f': round(float(f_val), 2)
    }


def tost_test(diffs, margin):
    """Two one-sided tests for equivalence at +/- margin."""
    diffs = np.array(diffs, dtype=float)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))
    df = n - 1

    if se == 0:
        # Perfect agreement
        return {
            'margin': margin,
            'mean_diff': round(mean_d, 4),
            'se': 0.0,
            'p_tost': 0.0,
            'ci90_low': round(mean_d, 4),
            'ci90_high': round(mean_d, 4),
            'equivalent': True
        }

    # Upper test: H0 mean >= +margin
    t_upper = (mean_d - margin) / se
    p_upper = float(sp_stats.t.cdf(t_upper, df))

    # Lower test: H0 mean <= -margin
    t_lower = (mean_d + margin) / se
    p_lower = float(1 - sp_stats.t.cdf(t_lower, df))

    p_tost = max(p_upper, p_lower)

    # 90% CI
    t_crit = float(sp_stats.t.ppf(0.95, df))
    ci_low = mean_d - t_crit * se
    ci_high = mean_d + t_crit * se

    return {
        'margin': margin,
        'mean_diff': round(mean_d, 4),
        'se': round(se, 4),
        'p_tost': round(p_tost, 6),
        'ci90_low': round(ci_low, 4),
        'ci90_high': round(ci_high, 4),
        'equivalent': bool(p_tost < 0.05)
    }


def cohens_d(diffs):
    """Cohen's d for paired differences (mean / SD)."""
    diffs = np.array(diffs, dtype=float)
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0
    return round(float(np.mean(diffs)) / sd, 4)


def bland_altman(ext, gt):
    """Bland-Altman analysis: bias and limits of agreement."""
    ext, gt = np.array(ext, dtype=float), np.array(gt, dtype=float)
    diffs = ext - gt
    means = (ext + gt) / 2

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    loa_lower = mean_diff - 1.96 * sd_diff
    loa_upper = mean_diff + 1.96 * sd_diff

    # Proportional bias
    if len(diffs) >= 3 and np.std(diffs) > 0 and np.std(means) > 0:
        r_prop, p_prop = sp_stats.pearsonr(means, diffs)
    else:
        r_prop, p_prop = 0.0, 1.0

    # 95% CI for mean difference
    se = sd_diff / np.sqrt(len(diffs))
    ci_low = mean_diff - 1.96 * se
    ci_high = mean_diff + 1.96 * se

    return {
        'mean_bias': round(mean_diff, 4),
        'sd_diff': round(sd_diff, 4),
        'loa_lower': round(float(loa_lower), 2),
        'loa_upper': round(float(loa_upper), 2),
        'ci95_bias_low': round(float(ci_low), 4),
        'ci95_bias_high': round(float(ci_high), 4),
        'proportional_bias_r': round(float(r_prop), 4),
        'proportional_bias_p': round(float(p_prop), 6)
    }


def within_thresholds(abs_errors, thresholds=[1, 3, 5, 10]):
    """Percentage of observations within threshold (pp)."""
    abs_errors = np.array(abs_errors, dtype=float)
    n = len(abs_errors)
    result = {}
    for t in thresholds:
        pct = float(np.sum(abs_errors <= t) / n * 100) if n > 0 else 0
        result[f'within_{t}pp'] = round(pct, 1)
    return result


def per_paper_tiers(paper_maes):
    """Count papers by MAE tier: Excellent (<2), Good (2-5), Fair (5-10), Poor (>10)."""
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


def compute_all_stats(ext_effects, gt_effects, paper_ids, tost_margin, dataset_name, effect_unit='pp'):
    """Compute all formal statistics for a dataset.

    ext_effects, gt_effects: lists of effect sizes in percentage points
    paper_ids: list of paper identifiers (same length)
    tost_margin: equivalence margin in pp (legacy single margin)
    effect_unit: 'pp' for percentage points, 'lnRR' for log response ratio
    """
    ext = np.array(ext_effects, dtype=float)
    gt = np.array(gt_effects, dtype=float)
    n_obs = len(ext)

    diffs = ext - gt
    abs_errors = np.abs(diffs)

    # Per-paper MAEs
    paper_obs = defaultdict(lambda: {'ext': [], 'gt': [], 'abs_err': []})
    for i, pid in enumerate(paper_ids):
        paper_obs[pid]['ext'].append(float(ext[i]))
        paper_obs[pid]['gt'].append(float(gt[i]))
        paper_obs[pid]['abs_err'].append(float(abs_errors[i]))

    paper_maes = {}
    for pid, data in paper_obs.items():
        paper_maes[pid] = float(np.mean(data['abs_err']))

    n_papers = len(paper_maes)

    # Direction agreement (skip near-zero GT effects)
    if effect_unit == 'lnRR':
        dir_threshold = 0.01  # For lnRR
    else:
        dir_threshold = 0.5  # For percentage points

    dir_total = sum(1 for g in gt if abs(g) > dir_threshold)
    dir_correct = sum(1 for e, g in zip(ext, gt)
                      if abs(g) > dir_threshold and
                      ((e > 0 and g > 0) or (e < 0 and g < 0) or (e == 0 and g == 0)))

    # Overall effect
    gt_mean = float(np.mean(gt))
    ext_mean = float(np.mean(ext))

    # Mean absolute effect size (basis for proportional margins)
    mean_abs_effect = float(np.mean(np.abs(gt)))

    # --- TOST battery: fixed + proportional margins ---
    # Fixed margins: +/-2pp and +/-3pp
    # Proportional margins: +/-20% and +/-10% of mean |GT effect|
    prop_20pct_margin = round(mean_abs_effect * 0.20, 4)
    prop_10pct_margin = round(mean_abs_effect * 0.10, 4)

    tost_battery = {}

    for margin in [2.0, 3.0]:
        label = f'fixed_{margin:.0f}pp'
        result = tost_test(diffs, margin)
        result['margin_type'] = 'fixed'
        result['margin_label'] = f'+/-{margin:.0f}pp'
        tost_battery[label] = result

    result_20 = tost_test(diffs, prop_20pct_margin)
    result_20['margin_type'] = 'proportional'
    result_20['margin_pct'] = 20
    result_20['margin_label'] = f'+/-20% of |effect| = +/-{prop_20pct_margin:.2f}pp'
    tost_battery['proportional_20pct'] = result_20

    result_10 = tost_test(diffs, prop_10pct_margin)
    result_10['margin_type'] = 'proportional'
    result_10['margin_pct'] = 10
    result_10['margin_label'] = f'+/-10% of |effect| = +/-{prop_10pct_margin:.2f}pp'
    tost_battery['proportional_10pct'] = result_10

    # Build results dict
    results = {
        'dataset': dataset_name,
        'n_obs': n_obs,
        'n_papers': n_papers,
        'effect_unit': effect_unit,

        # 1. Pearson r
        'pearson': pearson_r(ext, gt),

        # 2. ICC(2,1)
        'icc': icc_21(ext, gt),

        # 3. MAE
        'mae': round(float(np.mean(abs_errors)), 4),

        # 4. Median AE
        'median_ae': round(float(np.median(abs_errors)), 4),

        # 5. Direction agreement
        'direction': {
            'correct': dir_correct,
            'total': dir_total,
            'pct': round(dir_correct / dir_total * 100, 1) if dir_total > 0 else None
        },

        # 6. Overall effect comparison
        'overall_effect': {
            'gt_mean': round(gt_mean, 4),
            'ext_mean': round(ext_mean, 4),
            'diff_pp': round(abs(ext_mean - gt_mean), 4),
            'signed_diff': round(ext_mean - gt_mean, 4)
        },

        # 7a. TOST equivalence (legacy single margin, backward compat)
        'tost': tost_test(diffs, tost_margin),

        # 7b. TOST battery (fixed + proportional margins)
        'tost_battery': tost_battery,

        # 7c. Mean absolute effect size (used for proportional margins)
        'mean_abs_effect': round(mean_abs_effect, 4),

        # 8. Cohen's d
        'cohens_d': cohens_d(diffs),

        # 9. Bland-Altman
        'bland_altman': bland_altman(ext, gt),

        # 10. Within-threshold rates
        'thresholds': within_thresholds(abs_errors),

        # 11. Per-paper tiers
        'tiers': per_paper_tiers(list(paper_maes.values())),

        # Additional useful stats
        'max_ae': round(float(np.max(abs_errors)), 2),
        'sd_error': round(float(np.std(diffs, ddof=1)), 4),
        'rmse': round(float(np.sqrt(np.mean(diffs ** 2))), 4),
    }

    return results


# ============================================================
# Dataset Loaders
# ============================================================

def load_loladze():
    """Load Loladze 2014 matched pairs from validation_llm_10pp.json.

    JSON has 'matched_observations' list with fields:
      paper_id, gt_effect_pct, ext_effect_pct (already in percentage points).
    """
    json_path = BASE_DIR / 'data' / 'loladze_agent_replication' / 'validation_llm_10pp.json'
    if not json_path.exists():
        print(f"  [SKIP] Loladze JSON not found: {json_path}")
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matched_obs = data.get('matched_observations', [])
    if not matched_obs:
        print("  [SKIP] No matched_observations in Loladze validation JSON")
        return None

    ext_effects = []
    gt_effects = []
    paper_ids = []

    for obs in matched_obs:
        try:
            ext_effects.append(float(obs['ext_effect_pct']))
            gt_effects.append(float(obs['gt_effect_pct']))
            paper_ids.append(obs.get('paper_id', 'unknown'))
        except (ValueError, KeyError, TypeError):
            continue

    print(f"  Loladze: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
    return ext_effects, gt_effects, paper_ids


def load_hui():
    """Load Hui 2023 matched pairs from validation CSV.
    Use the original CSV (with paper_id column) for per-paper stats.
    If improved CSV exists, also load it for the extra obs (but it lacks paper_id)."""
    # Prefer the original CSV which has paper_id column
    csv_path = BASE_DIR / 'data' / 'hui2023_full_35' / 'validation_matches.csv'
    improved_path = BASE_DIR / 'data' / 'hui2023_full_35' / 'validation_matches_improved.csv'

    # Use improved if it exists (more obs), but reconstruct paper_id
    if improved_path.exists():
        use_path = improved_path
        has_paper_id = False
    elif csv_path.exists():
        use_path = csv_path
        has_paper_id = True
    else:
        print(f"  [SKIP] Hui CSV not found")
        return None

    # If using improved (no paper_id), load original to get paper_id mapping
    paper_id_map = {}
    if not has_paper_id and csv_path.exists():
        # Build mapping: (ext_ctrl, ext_treat, gt_ctrl, gt_treat) -> paper_id
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (row.get('ext_ctrl',''), row.get('ext_treat',''),
                           row.get('gt_ctrl',''), row.get('gt_treat',''))
                    paper_id_map[key] = row.get('paper_id', 'unknown')
                except Exception:
                    pass

    ext_effects = []
    gt_effects = []
    paper_ids = []

    with open(use_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ext_eff = float(row['ext_effect'])
                gt_eff = float(row['gt_effect'])
                ext_effects.append(ext_eff)
                gt_effects.append(gt_eff)

                if has_paper_id:
                    paper_ids.append(row.get('paper_id', 'unknown'))
                else:
                    # Try lookup from original CSV
                    key = (row.get('ext_ctrl',''), row.get('ext_treat',''),
                           row.get('gt_ctrl',''), row.get('gt_treat',''))
                    paper_ids.append(paper_id_map.get(key, 'unknown'))
            except (ValueError, KeyError):
                continue

    print(f"  Hui: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
    return ext_effects, gt_effects, paper_ids


def load_li2022():
    """Load Li 2022 effect-first matched pairs (scale-invariant matching).

    Uses the effect-first matching strategy from gt_matcher.py which matches
    by percent change (effect_pct) and verifies via back-computed scale factors.
    This solves the unit heterogeneity problem (g/plant vs kg/ha vs t/ha).

    Falls back to old metadata CSV + programmatic classification if
    effect-first results are not available.
    """
    # Prefer effect-first matched results (r=0.994 vs r=0.806 with old method)
    ef_csv_path = BASE_DIR / 'data' / 'li2022_combined' / 'validation_matches_effect_first.csv'

    if ef_csv_path.exists():
        ext_effects = []
        gt_effects = []
        paper_ids = []

        with open(ef_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ext_eff = float(row['ext_effect_pct'])
                    gt_eff = float(row['gt_effect_pct'])
                    ext_effects.append(ext_eff)
                    gt_effects.append(gt_eff)
                    paper_ids.append(row['paper_id'])
                except (ValueError, KeyError):
                    continue

        print(f"  Li 2022: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
        print(f"    (effect-first matching with scale verification)")
        return ext_effects, gt_effects, paper_ids

    # Fallback: old method with metadata CSV + programmatic classification
    csv_path = BASE_DIR / 'data' / 'li2022_combined' / 'validation_matches_metadata.csv'
    class_path = BASE_DIR / 'data' / 'li2022_combined' / 'programmatic_classification.json'

    if not csv_path.exists():
        print(f"  [SKIP] Li 2022 CSV not found: {csv_path}")
        return None

    # Load programmatic classification for filtering
    high_papers = set()
    if class_path.exists():
        with open(class_path, 'r', encoding='utf-8') as f:
            classification = json.load(f)
        for paper_id, info in classification.get('papers', {}).items():
            tier = info.get('tier', '')
            if tier in ('high', 'medium'):  # Use high + medium for decent coverage
                high_papers.add(paper_id)

    ext_effects = []
    gt_effects = []
    paper_ids = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                paper_id = row['paper_id']
                # Filter to high-confidence papers if classification available
                if high_papers and paper_id not in high_papers:
                    continue

                ext_eff = float(row['ext_effect_pct'])
                gt_eff = float(row['gt_effect_pct'])
                ext_effects.append(ext_eff)
                gt_effects.append(gt_eff)
                paper_ids.append(paper_id)
            except (ValueError, KeyError):
                continue

    print(f"  Li 2022: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
    print(f"    (fallback: metadata CSV filtered to high/medium papers)")
    if high_papers:
        print(f"    (filtered to {len(high_papers)} high/medium-confidence papers)")
    return ext_effects, gt_effects, paper_ids


def load_biochar():
    """Load Biochar matched pairs from validation_results.json matched_observations."""
    val_path = BASE_DIR / 'data' / 'biochar_extraction' / 'validation_results.json'
    if not val_path.exists():
        print(f"  [SKIP] Biochar validation not found: {val_path}")
        return None

    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Use matched_observations which has per-obs gt/ext effect pairs
    matched_obs = val_data.get('matched_observations', [])
    if matched_obs:
        ext_effects = []
        gt_effects = []
        paper_ids = []
        for obs in matched_obs:
            try:
                gt_eff = float(obs['gt_effect_pct'])
                ext_eff = float(obs['ext_effect_pct'])
                gt_effects.append(gt_eff)
                ext_effects.append(ext_eff)
                paper_ids.append(obs.get('paper_id', obs.get('gt_study', 'unknown')))
            except (ValueError, KeyError, TypeError):
                continue
        if ext_effects:
            print(f"  Biochar: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
            return ext_effects, gt_effects, paper_ids

    # Fallback: try CSV
    csv_path = BASE_DIR / 'data' / 'biochar_extraction' / 'validation_matches.csv'
    if csv_path.exists():
        ext_effects = []
        gt_effects = []
        paper_ids = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ext_eff = float(row.get('ext_effect_pct', row.get('ext_effect', 0)))
                    gt_eff = float(row.get('gt_effect_pct', row.get('gt_effect', 0)))
                    ext_effects.append(ext_eff)
                    gt_effects.append(gt_eff)
                    paper_ids.append(row.get('paper_id', row.get('paper', 'unknown')))
                except (ValueError, KeyError):
                    continue
        if ext_effects:
            print(f"  Biochar: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
            return ext_effects, gt_effects, paper_ids

    print("  [SKIP] Biochar: no per-observation match data available")
    return 'precomputed'


def load_boldorini():
    """Load Boldorini 2024 matched pairs from validation results.
    Uses 'matched_observations' which contains per-obs gt/ext pairs
    including both lnRR and percentage-change formats."""
    val_path = BASE_DIR / 'data' / 'boldorini_extraction' / 'validation_results.json'
    if not val_path.exists():
        print(f"  [SKIP] Boldorini validation not found: {val_path}")
        return None

    with open(val_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # Try multiple keys for per-observation data
    per_obs = val_data.get('matched_observations', [])
    if not per_obs:
        per_obs = val_data.get('per_observation', [])
    if not per_obs:
        per_obs = val_data.get('per_observation_independent', [])

    if not per_obs:
        # Only aggregate stats available - build a precomputed result
        print(f"  Boldorini: using aggregate stats (no per-obs data)")
        print(f"    Keys available: {list(val_data.keys())}")
        return 'precomputed_boldorini'

    ext_effects = []
    gt_effects = []
    paper_ids = []

    for obs in per_obs:
        # Try pre-computed percentage format first (from updated validate_boldorini.py)
        ext_pct = obs.get('ext_effect_pct')
        gt_pct = obs.get('gt_effect_pct')
        if ext_pct is not None and gt_pct is not None:
            ext_effects.append(float(ext_pct))
            gt_effects.append(float(gt_pct))
            paper_ids.append(obs.get('author', obs.get('paper_id', 'unknown')))
            continue

        # Fallback: try lnRR format and convert
        ext_lnRR = obs.get('ext_lnRR')
        gt_yi = obs.get('gt_yi')
        if ext_lnRR is not None and gt_yi is not None:
            # Convert lnRR to percentage change
            ext_pct = (math.exp(float(ext_lnRR)) - 1) * 100
            gt_pct = (math.exp(float(gt_yi)) - 1) * 100
            ext_effects.append(ext_pct)
            gt_effects.append(gt_pct)
            paper_ids.append(obs.get('author', obs.get('paper_id', 'unknown')))

    if len(ext_effects) < 3:
        print(f"  [WARN] Boldorini: only {len(ext_effects)} obs (minimum 3 needed for stats)")

    if not ext_effects:
        print(f"  [SKIP] Boldorini: could not extract numeric pairs from per-obs data")
        return 'precomputed_boldorini'

    print(f"  Boldorini: {len(ext_effects)} matched observations from {len(set(paper_ids))} papers")
    return ext_effects, gt_effects, paper_ids


def load_biochar_precomputed():
    """Build stats dict from pre-computed biochar validation_results.json."""
    val_path = BASE_DIR / 'data' / 'biochar_extraction' / 'validation_results.json'
    with open(val_path, 'r', encoding='utf-8') as f:
        v = json.load(f)

    per_paper = v.get('per_paper', [])
    paper_maes = [p['mae_pp'] for p in per_paper if p.get('mae_pp') is not None]

    return {
        'dataset': 'Biochar (Li 2024)',
        'n_obs': v['n_matched'],
        'n_papers': v['n_papers'],
        'effect_unit': 'pp',
        'pearson': {
            'r': v['pearson_r'],
            'p': None,
            'ci95_low': None,
            'ci95_high': None,
            'n': v['n_matched']
        },
        'icc': {
            'icc': v['icc'],
            'ci95_low': None,
            'ci95_high': None,
            'f': None
        },
        'mae': v['mae_pp'],
        'median_ae': v['median_ae_pp'],
        'direction': {
            'correct': None,
            'total': None,
            'pct': v['direction_agreement_pct']
        },
        'overall_effect': {
            'gt_mean': v['overall_effect_gt'],
            'ext_mean': v['overall_effect_ext'],
            'diff_pp': v['effect_diff_pp'],
            'signed_diff': v['overall_effect_ext'] - v['overall_effect_gt']
        },
        'tost': {
            'margin': 2.0,
            'mean_diff': None,
            'se': None,
            'p_tost': v['tost_p'],
            'ci90_low': None,
            'ci90_high': None,
            'equivalent': v['tost_pass']
        },
        'cohens_d': v['cohens_d'],
        'bland_altman': {
            'mean_bias': None,
            'sd_diff': None,
            'loa_lower': None,
            'loa_upper': None,
            'ci95_bias_low': None,
            'ci95_bias_high': None,
            'proportional_bias_r': None,
            'proportional_bias_p': None
        },
        'thresholds': {
            'within_1pp': None,
            'within_3pp': None,
            'within_5pp': v.get('within_5pp'),
            'within_10pp': v.get('within_10pp'),
        },
        'tiers': {
            k.capitalize() if k[0].islower() else k: c
            for k, c in v['tiers'].items()
        },
        'mean_abs_effect': None,
        'tost_battery': {},
        'max_ae': None,
        'sd_error': None,
        'rmse': None,
        '_precomputed': True
    }


def load_boldorini_precomputed():
    """Build stats dict from pre-computed Boldorini validation_results.json."""
    val_path = BASE_DIR / 'data' / 'boldorini_extraction' / 'validation_results.json'
    with open(val_path, 'r', encoding='utf-8') as f:
        v = json.load(f)

    return {
        'dataset': 'Boldorini 2024 (predator/yield)',
        'n_obs': v.get('n_matched', 0),
        'n_papers': 0,
        'effect_unit': 'pp',
        'pearson': {
            'r': v.get('pearson_r'),
            'p': None,
            'ci95_low': None,
            'ci95_high': None,
            'n': v.get('n_matched', 0)
        },
        'icc': {'icc': None, 'ci95_low': None, 'ci95_high': None, 'f': None},
        'mae': v.get('mae_pp'),
        'median_ae': None,
        'direction': {
            'correct': None,
            'total': None,
            'pct': v.get('direction_pct')
        },
        'overall_effect': {
            'gt_mean': None,
            'ext_mean': None,
            'diff_pp': v.get('overall_diff_pp'),
            'signed_diff': None
        },
        'tost': {
            'margin': 2.0,
            'mean_diff': None,
            'se': None,
            'p_tost': v.get('tost_p'),
            'ci90_low': None,
            'ci90_high': None,
            'equivalent': v.get('tost_p', 1) < 0.05 if v.get('tost_p') is not None else None
        },
        'cohens_d': v.get('cohens_d'),
        'bland_altman': {
            'mean_bias': None, 'sd_diff': None,
            'loa_lower': None, 'loa_upper': None,
            'ci95_bias_low': None, 'ci95_bias_high': None,
            'proportional_bias_r': None, 'proportional_bias_p': None
        },
        'thresholds': {
            'within_1pp': None, 'within_3pp': None,
            'within_5pp': None, 'within_10pp': None,
        },
        'tiers': {'Excellent': 0, 'Good': 0, 'Fair': 0, 'Poor': 0},
        'mean_abs_effect': None,
        'tost_battery': {},
        'max_ae': None,
        'sd_error': None,
        'rmse': None,
        '_precomputed': True
    }


# ============================================================
# Formatted Output
# ============================================================

def fmt(val, decimals=2, pct=False):
    """Format a value for display."""
    if val is None:
        return '-'
    if isinstance(val, bool):
        return 'Yes' if val else 'No'
    if pct:
        return f"{val:.{decimals}f}%"
    return f"{val:.{decimals}f}"


def print_dataset_report(stats):
    """Print detailed stats for one dataset."""
    s = stats
    print(f"\n{'='*70}")
    print(f"  {s['dataset']}")
    print(f"  {s['n_obs']} observations, {s['n_papers']} papers")
    print(f"{'='*70}")

    print(f"\n  Pearson r:        {fmt(s['pearson']['r'], 4)}", end='')
    if s['pearson'].get('ci95_low') is not None:
        print(f"  [95% CI: {fmt(s['pearson']['ci95_low'], 3)}, {fmt(s['pearson']['ci95_high'], 3)}]")
    else:
        print()

    print(f"  ICC(2,1):         {fmt(s['icc']['icc'], 4)}", end='')
    if s['icc'].get('ci95_low') is not None:
        print(f"  [95% CI: {fmt(s['icc']['ci95_low'], 3)}, {fmt(s['icc']['ci95_high'], 3)}]")
    else:
        print()

    print(f"  MAE:              {fmt(s['mae'], 2)} {s['effect_unit']}")
    print(f"  Median AE:        {fmt(s['median_ae'], 2)} {s['effect_unit']}")
    print(f"  RMSE:             {fmt(s.get('rmse'), 2)} {s['effect_unit']}")

    d = s['direction']
    if d.get('correct') is not None:
        print(f"  Direction:        {d['correct']}/{d['total']} ({fmt(d['pct'], 1)}%)")
    else:
        print(f"  Direction:        {fmt(d['pct'], 1)}%")

    o = s['overall_effect']
    print(f"  Overall effect:   GT={fmt(o['gt_mean'], 2)}, Ext={fmt(o['ext_mean'], 2)}, diff={fmt(o['diff_pp'], 2)} {s['effect_unit']}")

    t = s['tost']
    equiv_str = 'PASS' if t['equivalent'] else 'FAIL'
    print(f"  TOST (+/-{t['margin']}):  p={fmt(t['p_tost'], 4)}, {equiv_str}", end='')
    if t.get('ci90_low') is not None:
        print(f"  [90% CI: {fmt(t['ci90_low'], 3)}, {fmt(t['ci90_high'], 3)}]")
    else:
        print()

    # TOST battery (fixed + proportional margins)
    if 'tost_battery' in s:
        print(f"\n  TOST Battery (mean |GT effect| = {fmt(s.get('mean_abs_effect'), 2)}pp):")
        for key, tb in s['tost_battery'].items():
            eq = 'PASS' if tb['equivalent'] else 'FAIL'
            label = tb.get('margin_label', f'+/-{tb["margin"]}pp')
            print(f"    {label:<42s}  p={fmt(tb['p_tost'], 4)}, {eq}")

    print(f"  Cohen's d:        {fmt(s['cohens_d'], 4)}")

    ba = s['bland_altman']
    if ba.get('mean_bias') is not None:
        print(f"  Bland-Altman:     bias={fmt(ba['mean_bias'], 3)}, LOA=[{fmt(ba['loa_lower'], 1)}, {fmt(ba['loa_upper'], 1)}]")
        print(f"                    prop. bias r={fmt(ba['proportional_bias_r'], 3)}, p={fmt(ba['proportional_bias_p'], 4)}")
    else:
        print(f"  Bland-Altman:     (not available - precomputed)")

    th = s['thresholds']
    print(f"  Within 1pp:       {fmt(th.get('within_1pp'), 1)}%")
    print(f"  Within 3pp:       {fmt(th.get('within_3pp'), 1)}%")
    print(f"  Within 5pp:       {fmt(th.get('within_5pp'), 1)}%")
    print(f"  Within 10pp:      {fmt(th.get('within_10pp'), 1)}%")

    ti = s['tiers']
    total_papers = sum(ti.values())
    print(f"  Tiers: Excellent={ti.get('Excellent',0)}, Good={ti.get('Good',0)}, "
          f"Fair={ti.get('Fair',0)}, Poor={ti.get('Poor',0)} (n={total_papers})")


def print_comparison_table(all_stats):
    """Print cross-dataset comparison table."""
    print(f"\n{'='*100}")
    print("  CROSS-DATASET COMPARISON TABLE")
    print(f"{'='*100}")

    # Header
    datasets = [s['dataset'] for s in all_stats]
    # Abbreviate names for table fit
    short_names = []
    for d in datasets:
        if 'Loladze' in d:
            short_names.append('Loladze')
        elif 'Hui' in d:
            short_names.append('Hui')
        elif 'Biochar' in d or 'biochar' in d.lower():
            short_names.append('Biochar')
        elif 'Boldorini' in d:
            short_names.append('Boldorini')
        elif 'Li 2022' in d or 'biostimulant' in d.lower():
            short_names.append('Li 2022')
        else:
            short_names.append(d[:12])

    col_w = 14
    header = f"{'Metric':<30}" + ''.join(f"{n:>{col_w}}" for n in short_names)
    print(header)
    print('-' * len(header))

    def row(label, getter):
        vals = []
        for s in all_stats:
            try:
                v = getter(s)
                vals.append(v)
            except (KeyError, TypeError):
                vals.append('-')
        line = f"{label:<30}" + ''.join(f"{str(v):>{col_w}}" for v in vals)
        print(line)

    row('Obs (n)',                  lambda s: s['n_obs'])
    row('Papers',                   lambda s: s['n_papers'])
    row('Pearson r',                lambda s: fmt(s['pearson']['r'], 3))
    row('ICC(2,1)',                  lambda s: fmt(s['icc']['icc'], 3))
    row('MAE (pp)',                  lambda s: fmt(s['mae'], 2))
    row('Median AE (pp)',            lambda s: fmt(s['median_ae'], 2))
    row('RMSE (pp)',                 lambda s: fmt(s.get('rmse'), 2))
    row('Direction (%)',             lambda s: fmt(s['direction']['pct'], 1))
    row('GT mean effect',            lambda s: fmt(s['overall_effect']['gt_mean'], 2))
    row('Ext mean effect',           lambda s: fmt(s['overall_effect']['ext_mean'], 2))
    row('Effect diff (pp)',          lambda s: fmt(s['overall_effect']['diff_pp'], 2))
    row('TOST margin',               lambda s: s['tost']['margin'])
    row('TOST p',                    lambda s: fmt(s['tost']['p_tost'], 4))
    row('TOST pass',                 lambda s: 'Yes' if s['tost']['equivalent'] else 'No')
    row("Cohen's d",                 lambda s: fmt(s['cohens_d'], 3))
    row('B-A bias (pp)',             lambda s: fmt(s['bland_altman'].get('mean_bias'), 3))
    row('B-A LOA lower',             lambda s: fmt(s['bland_altman'].get('loa_lower'), 1))
    row('B-A LOA upper',             lambda s: fmt(s['bland_altman'].get('loa_upper'), 1))
    row('Within 1pp (%)',            lambda s: fmt(s['thresholds'].get('within_1pp'), 1))
    row('Within 3pp (%)',            lambda s: fmt(s['thresholds'].get('within_3pp'), 1))
    row('Within 5pp (%)',            lambda s: fmt(s['thresholds'].get('within_5pp'), 1))
    row('Within 10pp (%)',           lambda s: fmt(s['thresholds'].get('within_10pp'), 1))
    row('Tier: Excellent',           lambda s: s['tiers'].get('Excellent', 0))
    row('Tier: Good',                lambda s: s['tiers'].get('Good', 0))
    row('Tier: Fair',                lambda s: s['tiers'].get('Fair', 0))
    row('Tier: Poor',                lambda s: s['tiers'].get('Poor', 0))

    print()


def print_tost_battery_table(all_stats):
    """Print cross-dataset TOST equivalence table with fixed and proportional margins."""
    print(f"\n{'='*110}")
    print("  TOST EQUIVALENCE BATTERY: FIXED + PROPORTIONAL MARGINS")
    print(f"{'='*110}")

    # Abbreviate names
    short_names = []
    for s in all_stats:
        d = s['dataset']
        if 'Loladze' in d:
            short_names.append('Loladze')
        elif 'Hui' in d:
            short_names.append('Hui')
        elif 'Li 2022' in d or 'biostimulant' in d.lower():
            short_names.append('Li 2022')
        elif 'Biochar' in d or 'biochar' in d.lower():
            short_names.append('Biochar')
        elif 'Boldorini' in d:
            short_names.append('Boldorini')
        else:
            short_names.append(d[:12])

    col_w = 18

    # Row: mean |effect| for each dataset
    header = f"{'':.<30}" + ''.join(f"{n:>{col_w}}" for n in short_names)
    print(header)
    print('-' * len(header))

    # Mean absolute effect size row
    line = f"{'Mean |GT effect| (pp)':<30}"
    for s in all_stats:
        val = s.get('mean_abs_effect')
        if val is not None:
            line += f"{fmt(val, 2):>{col_w}}"
        else:
            line += f"{'-':>{col_w}}"
    print(line)
    print()

    # For each margin type, show margin value, p-value, pass/fail
    margin_keys = ['fixed_2pp', 'fixed_3pp', 'proportional_20pct', 'proportional_10pct']
    margin_labels = ['+/-2pp (fixed)', '+/-3pp (fixed)', '+/-20% of |effect|', '+/-10% of |effect|']

    for mkey, mlabel in zip(margin_keys, margin_labels):
        # Margin row
        line_margin = f"  {mlabel:<28}"
        line_p = f"    {'p-value':<26}"
        line_result = f"    {'Result':<26}"

        for s in all_stats:
            battery = s.get('tost_battery', {})
            if mkey in battery:
                tb = battery[mkey]
                margin_val = f"{tb['margin']:.2f}pp"
                p_val = fmt(tb['p_tost'], 4)
                result = 'PASS' if tb['equivalent'] else 'FAIL'
                line_margin += f"{margin_val:>{col_w}}"
                line_p += f"{p_val:>{col_w}}"
                line_result += f"{result:>{col_w}}"
            else:
                line_margin += f"{'-':>{col_w}}"
                line_p += f"{'-':>{col_w}}"
                line_result += f"{'-':>{col_w}}"

        print(line_margin)
        print(line_p)
        print(line_result)
        print()

    # Summary interpretation
    print("  Interpretation:")
    print("    Fixed margins (+/-2pp, +/-3pp) apply equally regardless of effect magnitude.")
    print("    Proportional margins scale with effect size, so +/-3pp is ~6% of Hui's ~53pp")
    print("    effect but ~48% of Loladze's ~6pp effect. Proportional margins correct for this.")
    print()


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  FORMAL STATISTICS: ALL VALIDATED DATASETS")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    all_stats = []

    # --- 1. Loladze 2014 ---
    print("\nLoading Loladze 2014 (CO2/minerals)...")
    loladze_data = load_loladze()
    if loladze_data:
        ext, gt, pids = loladze_data
        stats = compute_all_stats(ext, gt, pids, tost_margin=2.0,
                                   dataset_name='Loladze 2014 (CO2/minerals)')
        all_stats.append(stats)
        print_dataset_report(stats)

    # --- 2. Hui 2023 ---
    print("\nLoading Hui 2023 (Zn/wheat)...")
    hui_data = load_hui()
    if hui_data:
        ext, gt, pids = hui_data
        stats = compute_all_stats(ext, gt, pids, tost_margin=2.0,
                                   dataset_name='Hui 2023 (Zn/wheat)')
        all_stats.append(stats)
        print_dataset_report(stats)

    # --- 3. Li 2022 ---
    print("\nLoading Li 2022 (biostimulant/yield)...")
    li_data = load_li2022()
    if li_data:
        ext, gt, pids = li_data
        stats = compute_all_stats(ext, gt, pids, tost_margin=2.0,
                                   dataset_name='Li 2022 (biostimulant/yield)')
        all_stats.append(stats)
        print_dataset_report(stats)

    # --- 4. Biochar ---
    print("\nLoading Biochar (Li 2024)...")
    biochar_data = load_biochar()
    if biochar_data == 'precomputed':
        stats = load_biochar_precomputed()
        all_stats.append(stats)
        print_dataset_report(stats)
    elif biochar_data:
        ext, gt, pids = biochar_data
        stats = compute_all_stats(ext, gt, pids, tost_margin=2.0,
                                   dataset_name='Biochar (Li 2024)')
        all_stats.append(stats)
        print_dataset_report(stats)

    # --- 5. Boldorini 2024 ---
    print("\nLoading Boldorini 2024 (predator/yield)...")
    bold_data = load_boldorini()
    if bold_data == 'precomputed_boldorini':
        stats = load_boldorini_precomputed()
        all_stats.append(stats)
        print_dataset_report(stats)
    elif bold_data:
        ext, gt, pids = bold_data
        stats = compute_all_stats(ext, gt, pids, tost_margin=2.0,
                                   dataset_name='Boldorini 2024 (predator/yield)')
        all_stats.append(stats)
        print_dataset_report(stats)

    # --- Cross-dataset comparison ---
    if len(all_stats) >= 2:
        print_comparison_table(all_stats)
        print_tost_battery_table(all_stats)

    # --- Aggregate summary ---
    total_obs = sum(s['n_obs'] for s in all_stats)
    total_papers = sum(s['n_papers'] for s in all_stats)
    print(f"\n  AGGREGATE: {total_obs} observations across {total_papers} papers, {len(all_stats)} datasets")

    # Weighted average r (by obs count)
    r_vals = [(s['pearson']['r'], s['n_obs']) for s in all_stats if s['pearson']['r'] is not None]
    if r_vals:
        weighted_r = sum(r * n for r, n in r_vals) / sum(n for _, n in r_vals)
        print(f"  Weighted mean r: {weighted_r:.3f}")

    # Weighted average MAE
    mae_vals = [(s['mae'], s['n_obs']) for s in all_stats if s['mae'] is not None]
    if mae_vals:
        weighted_mae = sum(m * n for m, n in mae_vals) / sum(n for _, n in mae_vals)
        print(f"  Weighted mean MAE: {weighted_mae:.2f} pp")

    # --- Save JSON ---
    out_path = BASE_DIR / 'output' / 'formal_stats_all_datasets.json'
    output = {
        'timestamp': datetime.now().isoformat(),
        'n_datasets': len(all_stats),
        'total_obs': total_obs,
        'total_papers': total_papers,
        'datasets': all_stats
    }

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    output = convert_types(output)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Results saved to: {out_path}")
    print("  Done.")


if __name__ == '__main__':
    main()
