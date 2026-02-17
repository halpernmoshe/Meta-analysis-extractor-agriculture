"""
Tolerance sensitivity analysis for validation matching.

Uses existing validation match data to show how results change
when the matching tolerance threshold is varied.

For Hui 2023: computes match_error from raw values
For Li 2022: uses the match_error column from validation output
For Loladze: element-based matching (no tolerance to vary), but tests
  an effect-size error cutoff to show robustness of aggregate metrics.

Output: table for paper + CSV
"""
import sys, csv, math
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")

# ============================
# LOAD DATA
# ============================

def load_hui_matches():
    """Load Hui validation matches and compute match_error."""
    path = BASE / "output" / "hui2023_full_35" / "validation_matches.csv"
    if not path.exists():
        # Try alternate
        path = BASE / "output" / "hui2023_v2" / "validation_hui2023_matches.csv"
    if not path.exists():
        print(f"Hui matches not found at {path}")
        return []

    matches = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ext_ctrl = float(row['ext_ctrl'])
                ext_treat = float(row['ext_treat'])
                gt_ctrl = float(row['gt_ctrl'])
                gt_treat = float(row['gt_treat'])
                ext_effect = float(row['ext_effect'])
                gt_effect = float(row['gt_effect'])
                abs_error = float(row['abs_error'])
            except (ValueError, KeyError):
                continue

            # Compute match quality (same as validate_hui2023.py)
            c_err = abs(ext_ctrl - gt_ctrl) / max(abs(gt_ctrl), 0.1)
            t_err = abs(ext_treat - gt_treat) / max(abs(gt_treat), 0.1)
            match_error = (c_err + t_err) / 2

            matches.append({
                'ext_effect': ext_effect,
                'gt_effect': gt_effect,
                'abs_error': abs_error,
                'match_error': match_error,
            })
    return matches


def load_li_matches():
    """Load Li validation matches with match_error."""
    # Try consensus first, then combined
    for subdir in ['li2022_consensus', 'li2022_combined']:
        path = BASE / "output" / subdir / "validation_matches.csv"
        if path.exists():
            break
    else:
        print("Li matches not found")
        return []

    matches = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ext_effect = float(row['ext_effect_pct'])
                gt_effect = float(row['gt_effect_pct'])
                effect_diff = float(row['effect_diff_pp'])
                match_error = float(row['match_error'])
                direction = row['direction_match'] == 'True'
            except (ValueError, KeyError):
                continue

            matches.append({
                'ext_effect': ext_effect,
                'gt_effect': gt_effect,
                'abs_error': abs(effect_diff),
                'match_error': match_error,
                'direction': direction,
            })
    return matches


def load_loladze_matches():
    """Load Loladze validation matches.
    Loladze uses element-based matching without a value tolerance,
    but we can test robustness by filtering on effect-size error.
    """
    # Look for the validation output CSV
    for fname in ['validation_matches.csv', 'validation_full_46_matches.csv']:
        path = BASE / "output" / "loladze_full_46_v2" / fname
        if path.exists():
            break

    if not path.exists():
        # Try alignment integrated
        path = BASE / "output" / "alignment_integrated" / "all_matches.csv"

    if not path.exists():
        print(f"Loladze matches not found, trying to generate from validate_full_46.py output")
        return []

    matches = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                our = float(row.get('our_effect', row.get('our', 0)))
                gt = float(row.get('gt_effect', row.get('gt', 0)))
            except (ValueError, KeyError):
                continue
            matches.append({
                'ext_effect': our * 100 if abs(our) < 5 else our,  # Normalize to percent
                'gt_effect': gt * 100 if abs(gt) < 5 else gt,
                'abs_error': abs((our - gt) * 100) if abs(our) < 5 else abs(our - gt),
            })
    return matches


# ============================
# METRICS
# ============================

def compute_metrics(matches, effect_key_ext='ext_effect', effect_key_gt='gt_effect'):
    """Compute standard metrics."""
    if len(matches) < 3:
        return {'n': len(matches), 'r': None, 'mae': None, 'direction': None,
                'overall_diff': None, 'w5': None, 'w10': None, 'w20': None}

    our = [m[effect_key_ext] for m in matches]
    gt = [m[effect_key_gt] for m in matches]
    errors = [m['abs_error'] for m in matches]

    r, _ = sp_stats.pearsonr(our, gt)
    mae = np.mean(errors)

    dir_total = sum(1 for o, g in zip(our, gt) if abs(g) > 0.5)
    dir_ok = sum(1 for o, g in zip(our, gt) if abs(g) > 0.5 and (o > 0) == (g > 0))
    direction = dir_ok / dir_total * 100 if dir_total > 0 else None

    overall_diff = abs(np.mean(our) - np.mean(gt))

    w5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
    w10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
    w20 = sum(1 for e in errors if e <= 20) / len(errors) * 100

    return {
        'n': len(matches), 'r': round(r, 3), 'mae': round(mae, 2),
        'direction': round(direction, 1) if direction else None,
        'overall_diff': round(overall_diff, 2),
        'w5': round(w5, 1), 'w10': round(w10, 1), 'w20': round(w20, 1),
    }


def print_table(results, label):
    """Print a formatted results table."""
    print(f"\n--- {label} ---\n")
    print(f"{'Threshold':>10} {'N':>6} {'r':>8} {'MAE%':>8} {'Dir%':>8} {'EffDiff':>8} {'W5%':>6} {'W10%':>6} {'W20%':>6}")
    print("-" * 78)
    for row in results:
        r_str = f"{row['r']:.3f}" if row['r'] is not None else "N/A"
        mae_str = f"{row['mae']:.2f}" if row['mae'] is not None else "N/A"
        dir_str = f"{row['direction']:.1f}" if row['direction'] is not None else "N/A"
        diff_str = f"{row['overall_diff']:.2f}" if row['overall_diff'] is not None else "N/A"
        w5_str = f"{row['w5']:.1f}" if row['w5'] is not None else ""
        w10_str = f"{row['w10']:.1f}" if row['w10'] is not None else ""
        w20_str = f"{row['w20']:.1f}" if row['w20'] is not None else ""
        print(f"{row['threshold']:>10} {row['n']:>6} {r_str:>8} {mae_str:>8} {dir_str:>8} {diff_str:>8} {w5_str:>6} {w10_str:>6} {w20_str:>6}")


# ============================
# MAIN
# ============================

def main():
    print("=" * 80)
    print("MATCHING TOLERANCE SENSITIVITY ANALYSIS")
    print("=" * 80)

    all_results = []

    # --- Hui 2023 ---
    hui_matches = load_hui_matches()
    print(f"\nHui 2023: {len(hui_matches)} total matched observations loaded")

    if hui_matches:
        # Distribution of match errors
        me_vals = [m['match_error'] for m in hui_matches]
        print(f"  Match error distribution: min={min(me_vals):.4f}, median={np.median(me_vals):.4f}, "
              f"max={max(me_vals):.4f}, mean={np.mean(me_vals):.4f}")
        print(f"  Match error percentiles: 50th={np.percentile(me_vals, 50):.4f}, "
              f"75th={np.percentile(me_vals, 75):.4f}, 90th={np.percentile(me_vals, 90):.4f}, "
              f"95th={np.percentile(me_vals, 95):.4f}")

        thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1.00]
        hui_results = []
        for thr in thresholds:
            filtered = [m for m in hui_matches if m['match_error'] <= thr]
            metrics = compute_metrics(filtered)
            row = {'dataset': 'Hui 2023', 'threshold': thr, **metrics}
            hui_results.append(row)
            all_results.append(row)

        print_table(hui_results, "HUI 2023 (Zn/Wheat) - Varying match tolerance")

    # --- Li 2022 ---
    li_matches = load_li_matches()
    print(f"\nLi 2022: {len(li_matches)} total matched observations loaded")

    if li_matches:
        me_vals = [m['match_error'] for m in li_matches]
        print(f"  Match error distribution: min={min(me_vals):.4f}, median={np.median(me_vals):.4f}, "
              f"max={max(me_vals):.4f}, mean={np.mean(me_vals):.4f}")

        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 1.00]
        li_results = []
        for thr in thresholds:
            filtered = [m for m in li_matches if m['match_error'] <= thr]
            metrics = compute_metrics(filtered)
            row = {'dataset': 'Li 2022', 'threshold': thr, **metrics}
            li_results.append(row)
            all_results.append(row)

        print_table(li_results, "LI 2022 (Biostimulant/Yield) - Varying match tolerance")

    # --- Loladze 2014 ---
    lol_matches = load_loladze_matches()
    print(f"\nLoladze 2014: {len(lol_matches)} total matched observations loaded")

    if lol_matches:
        # For Loladze, matching is element-based (no value tolerance).
        # Instead, test robustness by filtering on abs effect-size error.
        err_vals = [m['abs_error'] for m in lol_matches]
        print(f"  Error distribution: min={min(err_vals):.2f}, median={np.median(err_vals):.2f}, "
              f"max={max(err_vals):.2f}, mean={np.mean(err_vals):.2f}")

        # Show: "if we exclude observations with > X% absolute error, how do aggregate metrics change?"
        # This tests whether outliers drive the results.
        print(f"\n--- LOLADZE 2014 - Outlier exclusion sensitivity ---")
        print(f"  (Loladze uses element-based matching without value tolerance.")
        print(f"   Instead, we test sensitivity to excluding high-error observations.)\n")
        print(f"{'Max error':>10} {'N':>6} {'r':>8} {'MAE%':>8} {'Dir%':>8} {'EffDiff':>8} {'W5%':>6} {'W10%':>6} {'W20%':>6}")
        print("-" * 78)

        for max_err in [10, 20, 30, 50, 100, 999]:
            filtered = [m for m in lol_matches if m['abs_error'] <= max_err]
            metrics = compute_metrics(filtered)
            label = f"<={max_err}%" if max_err < 999 else "All"
            r_str = f"{metrics['r']:.3f}" if metrics['r'] else "N/A"
            mae_str = f"{metrics['mae']:.2f}" if metrics['mae'] else "N/A"
            dir_str = f"{metrics['direction']:.1f}" if metrics['direction'] else "N/A"
            diff_str = f"{metrics['overall_diff']:.2f}" if metrics['overall_diff'] else "N/A"
            w5_str = f"{metrics['w5']:.1f}" if metrics['w5'] else ""
            w10_str = f"{metrics['w10']:.1f}" if metrics['w10'] else ""
            w20_str = f"{metrics['w20']:.1f}" if metrics['w20'] else ""
            print(f"{label:>10} {metrics['n']:>6} {r_str:>8} {mae_str:>8} {dir_str:>8} {diff_str:>8} {w5_str:>6} {w10_str:>6} {w20_str:>6}")

    # Save all results
    out_path = BASE / "output" / "tolerance_sensitivity.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if all_results:
        with open(out_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['dataset', 'threshold', 'n', 'r', 'mae', 'direction', 'overall_diff', 'w5', 'w10', 'w20'])
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved to {out_path}")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY FOR PAPER")
    print("=" * 80)

    if hui_matches:
        # Show that within a reasonable range, results are stable
        stable_range = [r for r in hui_results if r['threshold'] >= 0.10 and r['threshold'] <= 0.30 and r['r'] is not None]
        if stable_range:
            ns = [r['n'] for r in stable_range]
            rs = [r['r'] for r in stable_range]
            maes = [r['mae'] for r in stable_range]
            print(f"\nHui 2023 (tolerance 0.10-0.30):")
            print(f"  N: {min(ns)}-{max(ns)}, r: {min(rs):.3f}-{max(rs):.3f}, MAE: {min(maes):.1f}-{max(maes):.1f}%")

    if li_matches:
        stable_range = [r for r in li_results if r['threshold'] >= 0.10 and r['threshold'] <= 0.30 and r['r'] is not None]
        if stable_range:
            ns = [r['n'] for r in stable_range]
            rs = [r['r'] for r in stable_range]
            maes = [r['mae'] for r in stable_range]
            print(f"\nLi 2022 (tolerance 0.10-0.30):")
            print(f"  N: {min(ns)}-{max(ns)}, r: {min(rs):.3f}-{max(rs):.3f}, MAE: {min(maes):.1f}-{max(maes):.1f}%")


if __name__ == '__main__':
    main()
