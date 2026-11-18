#!/usr/bin/env python3
"""
Validate meta-analysis extraction against ground truth.

Computes: Pearson r, ICC(3,1), Concordance Correlation Coefficient (CCC),
MAE, direction agreement, Cohen's d, and TOST equivalence testing.

Requirements: numpy, scipy (pip install numpy scipy)

Usage:
    python validate.py --extracted <dir_or_file> --ground-truth <csv_or_json> [--margin 2.0] [--output report.json]

    If --ground-truth is not provided, reports descriptive statistics only.

Ground truth format (CSV or JSON):
    CSV columns: paper_id, outcome, effect_pct  (minimum required)
    Optional columns: treatment_mean, control_mean, n, variance_type, variance_value

    JSON format: list of objects with same fields, or dict keyed by paper_id.
"""

import sys
import os
import json
import csv
import argparse
import math
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import numpy as np
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: numpy/scipy not installed. Install with: pip install numpy scipy")
    print("         Running in limited mode (no ICC, TOST, or bootstrap).")


def load_extracted(path):
    """Load extracted observations from a directory of JSONs or a single JSON file."""
    path = Path(path)
    all_obs = []

    if path.is_dir():
        for f in sorted(path.glob("*.json")):
            with open(f, encoding='utf-8') as fh:
                data = json.load(fh)
            paper_id = data.get('paper_id', f.stem)
            obs_list = data.get('observations', data.get('consensus_observations', []))
            for obs in obs_list:
                obs['_paper_id'] = paper_id
                # Calculate effect if not present
                if obs.get('effect_pct') is None:
                    tm = obs.get('treatment_mean')
                    cm = obs.get('control_mean')
                    if tm is not None and cm is not None and cm != 0:
                        obs['effect_pct'] = (tm - cm) / abs(cm) * 100
                all_obs.append(obs)
    elif path.is_file():
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, list):
            all_obs = data
        elif isinstance(data, dict):
            paper_id = data.get('paper_id', path.stem)
            obs_list = data.get('observations', data.get('consensus_observations', []))
            for obs in obs_list:
                obs['_paper_id'] = paper_id
                all_obs.append(obs)

    return all_obs


def load_ground_truth(path):
    """Load ground truth from CSV or JSON."""
    path = Path(path)
    gt_obs = []

    if path.suffix.lower() == '.csv':
        with open(path, encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                obs = {}
                obs['paper_id'] = row.get('paper_id', '')
                obs['outcome'] = row.get('outcome', row.get('element', ''))
                for field in ['effect_pct', 'treatment_mean', 'control_mean', 'n', 'variance_value']:
                    val = row.get(field)
                    if val is not None and val != '':
                        try:
                            obs[field] = float(val)
                        except ValueError:
                            obs[field] = None
                    else:
                        obs[field] = None
                obs['variance_type'] = row.get('variance_type')
                gt_obs.append(obs)
    elif path.suffix.lower() == '.json':
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, list):
            gt_obs = data
        elif isinstance(data, dict):
            for paper_id, paper_data in data.items():
                if isinstance(paper_data, list):
                    for obs in paper_data:
                        obs['paper_id'] = paper_id
                        gt_obs.append(obs)

    return gt_obs


def match_observations(extracted, ground_truth):
    """Match extracted observations to ground truth by paper_id + outcome + effect proximity."""
    # Group GT by paper
    gt_by_paper = defaultdict(list)
    for obs in ground_truth:
        pid = obs.get('paper_id', '').lower().strip()
        gt_by_paper[pid].append(obs)

    matches = []
    unmatched_ext = []
    used_gt = set()

    for ext_obs in extracted:
        ext_pid = ext_obs.get('_paper_id', '').lower().strip()
        ext_outcome = (ext_obs.get('outcome', '') or ext_obs.get('element', '')).lower().strip()
        ext_effect = ext_obs.get('effect_pct')

        if ext_effect is None:
            unmatched_ext.append(ext_obs)
            continue

        # Find matching GT observations
        best_match = None
        best_dist = float('inf')

        candidates = gt_by_paper.get(ext_pid, [])
        for i, gt_obs in enumerate(candidates):
            gt_key = (ext_pid, i)
            if gt_key in used_gt:
                continue

            gt_outcome = (gt_obs.get('outcome', '') or gt_obs.get('element', '')).lower().strip()
            gt_effect = gt_obs.get('effect_pct')

            if gt_effect is None:
                continue

            # Match on outcome name
            if ext_outcome and gt_outcome and ext_outcome != gt_outcome:
                continue

            dist = abs(ext_effect - gt_effect)
            if dist < best_dist:
                best_dist = dist
                best_match = (gt_key, gt_obs)

        if best_match and best_dist < 50:  # 50 pp tolerance for matching
            used_gt.add(best_match[0])
            matches.append({
                'ext_effect': ext_effect,
                'gt_effect': best_match[1]['effect_pct'],
                'outcome': ext_outcome or best_match[1].get('outcome', ''),
                'paper': ext_pid,
                'error': abs(ext_effect - best_match[1]['effect_pct']),
            })
        else:
            unmatched_ext.append(ext_obs)

    unmatched_gt = sum(1 for pid, obs_list in gt_by_paper.items()
                       for i, _ in enumerate(obs_list)
                       if (pid, i) not in used_gt)

    return matches, len(unmatched_ext), unmatched_gt


def compute_pearson_r(matches):
    """Pearson correlation between extracted and GT effects."""
    if len(matches) < 3:
        return None
    ext = [m['ext_effect'] for m in matches]
    gt = [m['gt_effect'] for m in matches]
    r, p = stats.pearsonr(ext, gt)
    return {'r': round(float(r), 4), 'p': float(p), 'n': len(matches)}


def compute_icc(matches):
    """ICC(3,1) -- two-way mixed, single measures, consistency."""
    if len(matches) < 3:
        return None

    ext = np.array([m['ext_effect'] for m in matches])
    gt = np.array([m['gt_effect'] for m in matches])
    n = len(ext)
    k = 2

    ratings = np.column_stack([gt, ext])
    grand_mean = np.mean(ratings)
    row_means = np.mean(ratings, axis=1)
    col_means = np.mean(ratings, axis=0)

    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if ((n - 1) * (k - 1)) > 0 else 1e-10

    icc_31 = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)

    # 95% CI via F-distribution
    f_value = ms_rows / ms_error if ms_error > 0 else float('inf')
    df1, df2 = n - 1, (n - 1) * (k - 1)
    f_lower = f_value / stats.f.ppf(0.975, df1, df2) if df2 > 0 else 0
    f_upper = f_value / stats.f.ppf(0.025, df1, df2) if df2 > 0 else 0
    ci_lower = (f_lower - 1) / (f_lower + k - 1) if (f_lower + k - 1) != 0 else 0
    ci_upper = (f_upper - 1) / (f_upper + k - 1) if (f_upper + k - 1) != 0 else 0

    def interpret(val):
        if val < 0.40: return "poor"
        elif val < 0.60: return "fair"
        elif val < 0.75: return "good"
        else: return "excellent"

    return {
        'icc_31': round(float(icc_31), 4),
        'ci_95': [round(float(ci_lower), 4), round(float(ci_upper), 4)],
        'interpretation': interpret(icc_31),
        'n': len(matches),
    }


def compute_ccc(matches):
    """Lin's Concordance Correlation Coefficient."""
    if len(matches) < 3:
        return None

    ext = np.array([m['ext_effect'] for m in matches])
    gt = np.array([m['gt_effect'] for m in matches])

    mean_ext = np.mean(ext)
    mean_gt = np.mean(gt)
    var_ext = np.var(ext, ddof=1)
    var_gt = np.var(gt, ddof=1)
    cov = np.cov(ext, gt)[0, 1]

    ccc = (2 * cov) / (var_ext + var_gt + (mean_ext - mean_gt) ** 2)

    return {
        'ccc': round(float(ccc), 4),
        'n': len(matches),
    }


def compute_mae(matches):
    """Mean Absolute Error in percentage points."""
    errors = [m['error'] for m in matches]
    return {
        'mae_pp': round(float(np.mean(errors)), 2),
        'median_ae_pp': round(float(np.median(errors)), 2),
        'max_ae_pp': round(float(np.max(errors)), 2),
        'within_5pp': round(float(np.mean(np.array(errors) <= 5) * 100), 1),
        'within_10pp': round(float(np.mean(np.array(errors) <= 10) * 100), 1),
        'n': len(matches),
    }


def compute_direction(matches):
    """Direction agreement: do extracted and GT agree on sign of effect?"""
    agree = 0
    total = 0
    for m in matches:
        if m['gt_effect'] == 0:
            continue
        total += 1
        if (m['ext_effect'] < 0) == (m['gt_effect'] < 0):
            agree += 1
    return {
        'direction_agreement_pct': round(agree / total * 100, 1) if total > 0 else None,
        'agree': agree,
        'total': total,
    }


def compute_cohens_d(matches):
    """Cohen's d for paired differences."""
    ext = np.array([m['ext_effect'] for m in matches])
    gt = np.array([m['gt_effect'] for m in matches])
    diff = ext - gt
    d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0

    def interpret(val):
        val = abs(val)
        if val < 0.2: return "negligible"
        elif val < 0.5: return "small"
        elif val < 0.8: return "medium"
        else: return "large"

    return {
        'cohens_d': round(d, 4),
        'interpretation': interpret(d),
        'mean_diff_pp': round(float(np.mean(diff)), 3),
    }


def compute_tost(matches, margin_pp=2.0):
    """TOST equivalence test at the given margin (in percentage points)."""
    ext = np.array([m['ext_effect'] for m in matches])
    gt = np.array([m['gt_effect'] for m in matches])
    diff = ext - gt
    n = len(diff)

    mean_diff = np.mean(diff)
    se_diff = np.std(diff, ddof=1) / np.sqrt(n)

    # One-sided test 1: H0: mean_diff <= -margin
    t1 = (mean_diff - (-margin_pp)) / se_diff
    p1 = 1 - stats.t.cdf(t1, df=n - 1)

    # One-sided test 2: H0: mean_diff >= +margin
    t2 = (mean_diff - margin_pp) / se_diff
    p2 = stats.t.cdf(t2, df=n - 1)

    p_tost = max(p1, p2)
    equivalent = bool(p_tost < 0.05)

    # 90% CI
    t_crit = stats.t.ppf(0.95, df=n - 1)
    ci_90 = (mean_diff - t_crit * se_diff, mean_diff + t_crit * se_diff)

    return {
        'margin_pp': margin_pp,
        'p_tost': round(float(p_tost), 6),
        'equivalent_at_alpha_05': equivalent,
        'mean_diff_pp': round(float(mean_diff), 3),
        'ci_90_pp': [round(float(ci_90[0]), 3), round(float(ci_90[1]), 3)],
        'n': n,
    }


def compute_per_paper_stats(matches):
    """Compute per-paper summary statistics."""
    paper_data = defaultdict(lambda: {'ext': [], 'gt': []})
    for m in matches:
        paper_data[m['paper']]['ext'].append(m['ext_effect'])
        paper_data[m['paper']]['gt'].append(m['gt_effect'])

    tiers = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
    paper_results = {}

    for paper, data in paper_data.items():
        ext_arr = np.array(data['ext'])
        gt_arr = np.array(data['gt'])
        mae = float(np.mean(np.abs(ext_arr - gt_arr)))

        if mae <= 3:
            tier = 'excellent'
        elif mae <= 5:
            tier = 'good'
        elif mae <= 10:
            tier = 'fair'
        else:
            tier = 'poor'
        tiers[tier] += 1

        r = None
        if len(ext_arr) >= 3:
            r_val, _ = stats.pearsonr(ext_arr, gt_arr)
            r = round(float(r_val), 3)

        paper_results[paper] = {
            'n_obs': len(ext_arr),
            'mae_pp': round(mae, 2),
            'r': r,
            'tier': tier,
        }

    return {
        'per_paper': paper_results,
        'tier_summary': tiers,
        'n_papers': len(paper_data),
    }


def descriptive_stats(extracted):
    """Compute descriptive statistics when no ground truth is available."""
    effects = []
    outcomes = defaultdict(int)
    has_variance = 0
    has_n = 0
    papers = set()
    paper_obs_counts = defaultdict(int)

    for obs in extracted:
        paper = obs.get('_paper_id', 'unknown')
        papers.add(paper)
        paper_obs_counts[paper] += 1

        outcome = obs.get('outcome', obs.get('element', 'unknown'))
        outcomes[outcome] += 1

        eff = obs.get('effect_pct')
        if eff is not None:
            effects.append(eff)

        if obs.get('variance_value') is not None:
            has_variance += 1
        if obs.get('n') is not None:
            has_n += 1

    total = len(extracted)
    effects_arr = np.array(effects) if effects else np.array([0])

    # Flag potential issues
    flags = []
    large_effects = [e for e in effects if abs(e) > 100]
    if large_effects:
        flags.append(f"{len(large_effects)} observations with |effect| > 100%")
    if has_variance / max(total, 1) < 0.5:
        flags.append(f"Low variance coverage: {has_variance}/{total} ({has_variance/max(total,1)*100:.0f}%)")
    if has_n / max(total, 1) < 0.5:
        flags.append(f"Low sample size coverage: {has_n}/{total} ({has_n/max(total,1)*100:.0f}%)")
    papers_with_few_obs = [p for p, c in paper_obs_counts.items() if c <= 1]
    if papers_with_few_obs:
        flags.append(f"{len(papers_with_few_obs)} papers with only 1 observation (check extraction completeness)")

    return {
        'total_papers': len(papers),
        'total_observations': total,
        'observations_with_effect': len(effects),
        'outcome_distribution': dict(sorted(outcomes.items(), key=lambda x: -x[1])),
        'effect_size_stats': {
            'mean': round(float(np.mean(effects_arr)), 2),
            'median': round(float(np.median(effects_arr)), 2),
            'sd': round(float(np.std(effects_arr, ddof=1)), 2) if len(effects) > 1 else None,
            'min': round(float(np.min(effects_arr)), 2),
            'max': round(float(np.max(effects_arr)), 2),
            'q25': round(float(np.percentile(effects_arr, 25)), 2),
            'q75': round(float(np.percentile(effects_arr, 75)), 2),
        },
        'variance_coverage': round(has_variance / max(total, 1) * 100, 1),
        'sample_size_coverage': round(has_n / max(total, 1) * 100, 1),
        'per_paper_obs': dict(sorted(paper_obs_counts.items())),
        'flags': flags,
    }


def run_validation(extracted, ground_truth, margin_pp=2.0):
    """Run full validation suite."""
    matches, unmatched_ext, unmatched_gt = match_observations(extracted, ground_truth)

    if len(matches) < 3:
        return {
            'error': f'Only {len(matches)} observations matched. Need at least 3 for statistics.',
            'matched': len(matches),
            'unmatched_extracted': unmatched_ext,
            'unmatched_gt': unmatched_gt,
        }

    report = {
        'matching': {
            'matched_observations': len(matches),
            'unmatched_extracted': unmatched_ext,
            'unmatched_gt': unmatched_gt,
            'capture_rate_pct': round(len(matches) / (len(matches) + unmatched_gt) * 100, 1),
        },
        'pearson': compute_pearson_r(matches),
        'mae': compute_mae(matches),
        'direction': compute_direction(matches),
        'cohens_d': compute_cohens_d(matches),
        'descriptive': descriptive_stats(extracted),
    }

    if HAS_SCIPY:
        report['icc'] = compute_icc(matches)
        report['ccc'] = compute_ccc(matches)
        report['tost'] = compute_tost(matches, margin_pp=margin_pp)
        report['per_paper'] = compute_per_paper_stats(matches)

        # Overall effect comparison
        ext_effects = [m['ext_effect'] for m in matches]
        gt_effects = [m['gt_effect'] for m in matches]
        report['overall_effect'] = {
            'extracted_mean': round(float(np.mean(ext_effects)), 3),
            'gt_mean': round(float(np.mean(gt_effects)), 3),
            'difference_pp': round(float(np.mean(ext_effects) - np.mean(gt_effects)), 3),
        }

    return report


def print_report(report, output_path=None):
    """Print a human-readable validation report."""
    lines = []
    lines.append("=" * 70)
    lines.append("META-ANALYSIS EXTRACTION VALIDATION REPORT")
    lines.append("=" * 70)

    if 'error' in report:
        lines.append(f"\nERROR: {report['error']}")
        lines.append(f"Matched: {report['matched']}, Unmatched ext: {report['unmatched_extracted']}, Unmatched GT: {report['unmatched_gt']}")
        text = "\n".join(lines)
        print(text)
        return

    m = report['matching']
    lines.append(f"\nMATCHING: {m['matched_observations']} observations matched")
    lines.append(f"  Capture rate: {m['capture_rate_pct']}%")
    lines.append(f"  Unmatched (extracted): {m['unmatched_extracted']}")
    lines.append(f"  Unmatched (ground truth): {m['unmatched_gt']}")

    if report.get('pearson'):
        p = report['pearson']
        lines.append(f"\nCORRELATION: r = {p['r']} (p = {p['p']:.2e}, n = {p['n']})")

    if report.get('icc'):
        icc = report['icc']
        lines.append(f"ICC(3,1): {icc['icc_31']} (95% CI: {icc['ci_95'][0]}-{icc['ci_95'][1]}) [{icc['interpretation']}]")

    if report.get('ccc'):
        lines.append(f"CCC: {report['ccc']['ccc']}")

    if report.get('mae'):
        mae = report['mae']
        lines.append(f"\nACCURACY:")
        lines.append(f"  MAE: {mae['mae_pp']} pp")
        lines.append(f"  Median AE: {mae['median_ae_pp']} pp")
        lines.append(f"  Within 5pp: {mae['within_5pp']}%")
        lines.append(f"  Within 10pp: {mae['within_10pp']}%")

    if report.get('direction'):
        d = report['direction']
        lines.append(f"\nDIRECTION AGREEMENT: {d['direction_agreement_pct']}% ({d['agree']}/{d['total']})")

    if report.get('cohens_d'):
        cd = report['cohens_d']
        lines.append(f"Cohen's d: {cd['cohens_d']} ({cd['interpretation']}), mean diff: {cd['mean_diff_pp']} pp")

    if report.get('tost'):
        t = report['tost']
        result = "EQUIVALENT" if t['equivalent_at_alpha_05'] else "NOT equivalent"
        lines.append(f"\nTOST (+/-{t['margin_pp']}pp): p = {t['p_tost']:.4f} -> {result}")
        lines.append(f"  90% CI: ({t['ci_90_pp'][0]}, {t['ci_90_pp'][1]}) pp")

    if report.get('overall_effect'):
        oe = report['overall_effect']
        lines.append(f"\nOVERALL EFFECT:")
        lines.append(f"  Extracted: {oe['extracted_mean']}%")
        lines.append(f"  Ground truth: {oe['gt_mean']}%")
        lines.append(f"  Difference: {oe['difference_pp']} pp")

    if report.get('per_paper'):
        pp = report['per_paper']
        lines.append(f"\nPER-PAPER TIERS ({pp['n_papers']} papers):")
        ts = pp['tier_summary']
        for tier in ['excellent', 'good', 'fair', 'poor']:
            pct = ts[tier] / pp['n_papers'] * 100 if pp['n_papers'] > 0 else 0
            lines.append(f"  {tier.capitalize()}: {ts[tier]} ({pct:.0f}%)")

    if report.get('descriptive', {}).get('flags'):
        lines.append(f"\nFLAGS:")
        for flag in report['descriptive']['flags']:
            lines.append(f"  - {flag}")

    lines.append("\n" + "=" * 70)

    text = "\n".join(lines)
    print(text)

    if output_path:
        # Save JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nFull report saved to: {output_path}")

        # Save text report alongside
        txt_path = Path(output_path).with_suffix('.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text report saved to: {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate meta-analysis extraction against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate.py --extracted output/ --ground-truth gt.csv
  python validate.py --extracted output/ --ground-truth gt.json --margin 3.0
  python validate.py --extracted output/  (descriptive stats only, no GT)
        """
    )
    parser.add_argument('--extracted', required=True, help='Directory of extracted JSONs or single JSON file')
    parser.add_argument('--ground-truth', '--gt', default=None, help='Ground truth CSV or JSON file')
    parser.add_argument('--margin', type=float, default=2.0, help='TOST equivalence margin in percentage points (default: 2.0)')
    parser.add_argument('--output', '-o', default=None, help='Output path for JSON report')

    args = parser.parse_args()

    # Load extracted data
    print(f"Loading extracted data from: {args.extracted}")
    extracted = load_extracted(args.extracted)
    print(f"  Loaded {len(extracted)} observations")

    if not extracted:
        print("ERROR: No observations found in extracted data.")
        sys.exit(1)

    if args.ground_truth:
        # Full validation
        print(f"Loading ground truth from: {args.ground_truth}")
        gt = load_ground_truth(args.ground_truth)
        print(f"  Loaded {len(gt)} ground truth observations")

        if not gt:
            print("ERROR: No observations found in ground truth file.")
            sys.exit(1)

        report = run_validation(extracted, gt, margin_pp=args.margin)
    else:
        # Descriptive stats only
        print("No ground truth provided. Computing descriptive statistics only.")
        report = {'descriptive': descriptive_stats(extracted)}

        # Print descriptive report
        desc = report['descriptive']
        print(f"\n{'='*70}")
        print("EXTRACTION DESCRIPTIVE REPORT")
        print(f"{'='*70}")
        print(f"\nPapers: {desc['total_papers']}")
        print(f"Observations: {desc['total_observations']}")
        print(f"With effect size: {desc['observations_with_effect']}")
        print(f"\nOutcome distribution:")
        for outcome, count in desc['outcome_distribution'].items():
            print(f"  {outcome}: {count}")
        es = desc['effect_size_stats']
        print(f"\nEffect size (%):")
        print(f"  Mean: {es['mean']}, Median: {es['median']}")
        print(f"  Range: [{es['min']}, {es['max']}]")
        print(f"  IQR: [{es['q25']}, {es['q75']}]")
        print(f"\nVariance coverage: {desc['variance_coverage']}%")
        print(f"Sample size coverage: {desc['sample_size_coverage']}%")
        if desc['flags']:
            print(f"\nFlags:")
            for flag in desc['flags']:
                print(f"  - {flag}")

    print_report(report, args.output) if args.ground_truth else None

    # Save report if output specified
    if args.output and not args.ground_truth:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
