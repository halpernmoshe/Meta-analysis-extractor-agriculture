"""
Leave-One-Out Sensitivity Analysis + Difficulty Stratification
==============================================================
Computes LOO-paper, LOO-element, cumulative, and stratified results.

Run:
    python sensitivity_loo.py

Outputs:
    output/sensitivity/
"""
import sys, os, json, math, re
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
OUT_DIR = BASE_DIR / "output" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from validate_full_46 import (
    load_gt, PAPER_TO_LOLADZE_REF, normalize_element,
    filter_obs_for_gt_row, deduplicate_vision_text,
    is_concentration_unit, compute_effect, detect_tc_swap, get_mods
)


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


def load_all_matches():
    """Load all matched observation pairs."""
    gt = load_gt()
    results_files = sorted(RESULTS_DIR.glob("*_consensus.json"))
    all_matches = []

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

        elevated_obs = []
        for o in obs_list:
            desc = str(o.get('treatment_description', '')).lower()
            co2_match = re.search(r'(\d{2,4})\s*(?:ppm|µmol|umol|μmol)', desc)
            if co2_match and float(co2_match.group(1)) < 300:
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
            for c in candidates:
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

            mode = data.get('extraction_mode', 'unknown')
            all_matches.append({
                'paper': paper_id, 'ref': loladze_ref, 'el': gt_row['element'],
                'our': our_effect, 'gt': gt_effect,
                'err': abs(our_effect - gt_effect), 'mode': mode,
            })

    return all_matches


def compute_metrics(matches):
    if not matches:
        return {'n': 0, 'r': None, 'mae': None, 'dir_pct': None, 'effect_diff': None}
    our = np.array([m['our'] for m in matches])
    gt = np.array([m['gt'] for m in matches])
    n = len(matches)
    if n < 3 or np.std(our) == 0 or np.std(gt) == 0:
        r = None
    else:
        r = round(float(np.corrcoef(our, gt)[0, 1]), 4)
    mae = round(float(np.mean([m['err'] for m in matches])) * 100, 2)
    nonzero = gt != 0
    dir_pct = round(float(np.mean((our[nonzero] < 0) == (gt[nonzero] < 0)) * 100), 1) if np.any(nonzero) else None
    effect_diff = round(abs(float(np.mean(our)) - float(np.mean(gt))) * 100, 3)
    return {'n': n, 'r': r, 'mae': mae, 'dir_pct': dir_pct, 'effect_diff_pp': effect_diff}


def leave_one_paper_out(matches):
    papers = sorted(set(m['paper'] for m in matches))
    full = compute_metrics(matches)
    results = []
    for paper in papers:
        reduced = [m for m in matches if m['paper'] != paper]
        paper_m = [m for m in matches if m['paper'] == paper]
        red = compute_metrics(reduced)
        results.append({
            'paper': paper,
            'n_removed': len(paper_m),
            'paper_mae': round(np.mean([m['err'] for m in paper_m]) * 100, 2),
            'loo_r': red['r'],
            'loo_mae': red['mae'],
            'delta_r': round((red['r'] or 0) - (full['r'] or 0), 4) if red['r'] and full['r'] else None,
            'delta_mae': round(red['mae'] - full['mae'], 3),
        })
    results.sort(key=lambda x: x['delta_mae'])
    return {'full': full, 'n_papers': len(papers), 'per_paper': results,
            'mae_range': [min(r['loo_mae'] for r in results), max(r['loo_mae'] for r in results)]}


def leave_one_element_out(matches):
    elements = sorted(set(m['el'] for m in matches))
    full = compute_metrics(matches)
    results = []
    for el in elements:
        reduced = [m for m in matches if m['el'] != el]
        el_m = [m for m in matches if m['el'] == el]
        red = compute_metrics(reduced)
        results.append({
            'element': el, 'n_removed': len(el_m),
            'element_mae': round(np.mean([m['err'] for m in el_m]) * 100, 2),
            'loo_mae': red['mae'],
            'delta_mae': round(red['mae'] - full['mae'], 3),
        })
    results.sort(key=lambda x: x['delta_mae'])
    return {'full': full, 'n_elements': len(elements), 'per_element': results}


def difficulty_stratification(matches):
    paper_maes = defaultdict(list)
    for m in matches:
        paper_maes[m['paper']].append(m['err'])
    paper_avg_mae = {p: np.mean(errs) for p, errs in paper_maes.items()}

    tier_results = {}
    tiers = [('Excellent (<5%)', 0, 0.05), ('Good (5-10%)', 0.05, 0.10),
             ('Fair (10-20%)', 0.10, 0.20), ('Poor (>20%)', 0.20, 100)]
    for name, lo, hi in tiers:
        ps = [p for p, mae in paper_avg_mae.items() if lo <= mae < hi]
        ms = [m for m in matches if m['paper'] in ps]
        tier_results[name] = {'n_papers': len(ps), **compute_metrics(ms)}

    mag_results = {}
    for name, lo, hi in [('Small |eff|<5%', 0, 0.05), ('Medium 5-15%', 0.05, 0.15),
                          ('Large 15-30%', 0.15, 0.30), ('Very large >30%', 0.30, 10)]:
        ms = [m for m in matches if lo <= abs(m['gt']) < hi]
        mag_results[name] = compute_metrics(ms)

    return {'by_tier': tier_results, 'by_magnitude': mag_results}


def main():
    print("=" * 70)
    print("SENSITIVITY ANALYSES")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    matches = load_all_matches()
    print(f"  {len(matches)} matches from {len(set(m['paper'] for m in matches))} papers\n")

    # 1. LOO Paper
    print("1. LEAVE-ONE-PAPER-OUT")
    print("-" * 40)
    loo_p = leave_one_paper_out(matches)
    print(f"  Full MAE: {loo_p['full']['mae']}%")
    print(f"  MAE range without any single paper: {loo_p['mae_range'][0]:.1f}-{loo_p['mae_range'][1]:.1f}%")
    top3_harmful = sorted(loo_p['per_paper'], key=lambda x: x['delta_mae'])[:3]
    top3_helpful = sorted(loo_p['per_paper'], key=lambda x: x['delta_mae'], reverse=True)[:3]
    print(f"  Top 3 papers that HURT accuracy (removing improves MAE):")
    for r in top3_helpful:
        print(f"    {r['paper']}: removing → MAE {r['delta_mae']:+.2f}pp (paper MAE={r['paper_mae']:.1f}%)")
    print(f"  Top 3 papers that HELP accuracy (removing hurts MAE):")
    for r in top3_harmful:
        print(f"    {r['paper']}: removing → MAE {r['delta_mae']:+.2f}pp (paper MAE={r['paper_mae']:.1f}%)")

    with open(OUT_DIR / 'leave_one_paper_out.json', 'w') as f:
        json.dump(loo_p, f, indent=2, cls=NumpyEncoder)

    # 2. LOO Element
    print("\n2. LEAVE-ONE-ELEMENT-OUT")
    print("-" * 40)
    loo_e = leave_one_element_out(matches)
    for r in loo_e['per_element']:
        print(f"  {r['element']:3s} (n={r['n_removed']:>3}): element MAE={r['element_mae']:>5.1f}% | LOO delta={r['delta_mae']:+.2f}pp")
    with open(OUT_DIR / 'leave_one_element_out.json', 'w') as f:
        json.dump(loo_e, f, indent=2, cls=NumpyEncoder)

    # 3. Stratification
    print("\n3. DIFFICULTY STRATIFICATION")
    print("-" * 40)
    strat = difficulty_stratification(matches)
    print("  By paper quality tier:")
    for tier, m in strat['by_tier'].items():
        print(f"    {tier:20s}: {m['n_papers']} papers, n={m['n']:>3}, MAE={m['mae'] or 'N/A'}%")
    print("  By GT effect magnitude:")
    for bucket, m in strat['by_magnitude'].items():
        print(f"    {bucket:25s}: n={m['n']:>3}, MAE={m['mae'] or 'N/A'}%")
    with open(OUT_DIR / 'difficulty_stratification.json', 'w') as f:
        json.dump(strat, f, indent=2, cls=NumpyEncoder)

    # Generate LOO figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Paper LOO
        sorted_p = sorted(loo_p['per_paper'], key=lambda x: x['delta_mae'], reverse=True)
        names = [r['paper'].split('_')[1][:12] for r in sorted_p]
        deltas = [r['delta_mae'] for r in sorted_p]
        colors = ['#d32f2f' if d < -0.15 else '#388e3c' if d > 0.15 else '#757575' for d in deltas]
        ax1.barh(range(len(names)), deltas, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=6)
        ax1.axvline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Change in MAE (pp)')
        ax1.set_title('Leave-One-Paper-Out')

        # Element LOO
        sorted_e = sorted(loo_e['per_element'], key=lambda x: x['delta_mae'], reverse=True)
        el_names = [f"{r['element']} (n={r['n_removed']})" for r in sorted_e]
        el_deltas = [r['delta_mae'] for r in sorted_e]
        el_colors = ['#d32f2f' if d < -0.15 else '#388e3c' if d > 0.15 else '#757575' for d in el_deltas]
        ax2.barh(range(len(el_names)), el_deltas, color=el_colors, alpha=0.8)
        ax2.set_yticks(range(len(el_names)))
        ax2.set_yticklabels(el_names, fontsize=9)
        ax2.axvline(0, color='black', linewidth=0.5)
        ax2.set_xlabel('Change in MAE (pp)')
        ax2.set_title('Leave-One-Element-Out')

        plt.suptitle('Sensitivity Analysis: Loladze 2014 Validation', fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(OUT_DIR / 'fig_loo_combined.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n  Saved fig_loo_combined.png")
    except ImportError:
        pass

    print(f"\nAll saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
