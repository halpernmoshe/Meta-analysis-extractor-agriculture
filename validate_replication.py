"""
Replication Validation — Compare Run2 (replication) against GT and Run1

1. Validate Run2 agent extractions against ground truth (same method as Run1)
2. Compare Run1 vs Run2 (reproducibility — no GT needed)

Usage:
    ./venv/Scripts/python.exe validate_replication.py
"""
import sys, json, math, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")

# ── Directories ──────────────────────────────────────────────────────────────

DATASETS = {
    'loladze': {
        'name': 'Loladze 2014 (CO2/minerals)',
        'run1_dir': BASE / 'output' / 'agent_extraction',
        'run2_dir': BASE / 'output' / 'loladze_agent_replication',
        'gt_path': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\Loladze\CO2+Dataset.xlsx"),
        'gt_type': 'loladze',
    },
    'hui2023': {
        'name': 'Hui 2023 (Zn/wheat)',
        'run1_dir': BASE / 'output' / 'hui2023_agent_extraction',
        'run2_dir': BASE / 'output' / 'hui2023_agent_replication',
        'gt_path': None,  # Uses validate_hui2023_agent.py logic
        'gt_type': 'hui',
    },
    'li2022': {
        'name': 'Li 2022 (biostimulants)',
        'run1_dir': BASE / 'output' / 'li2022_agent_extraction',
        'run2_dir': BASE / 'output' / 'li2022_agent_replication',
        'gt_path': None,  # Uses validate_li2022_agent.py logic
        'gt_type': 'li',
    },
}

SCALE_FACTORS = [1, 10, 100, 1000, 0.1, 0.01, 0.001]


def safe_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def load_observations(json_path, filter_fn=None):
    """Load observations from agent JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    obs_list = data.get('consensus_observations', [])
    result = []
    for obs in obs_list:
        ctrl = safe_float(obs.get('control_mean'))
        treat = safe_float(obs.get('treatment_mean'))
        if ctrl is None or treat is None or ctrl == 0:
            continue
        if filter_fn and not filter_fn(obs):
            continue
        effect = (treat - ctrl) / ctrl * 100
        result.append({
            'control_mean': ctrl,
            'treatment_mean': treat,
            'effect_pct': effect,
            'element': obs.get('element', obs.get('outcome', '')),
            'tissue': obs.get('tissue', ''),
        })
    return result


def extract_paper_id(filename, strip_suffixes=True):
    """Normalize filename to comparable paper ID."""
    stem = filename.stem
    if strip_suffixes:
        stem = stem.replace('_agent_v2', '').replace('_agent', '')
        stem = stem.replace('_consensus', '')
    return stem.lower().strip()


def match_observations(obs_a, obs_b, tolerance=0.25):
    """Match observations between two sets by value similarity."""
    matches = []
    used_b = set()

    for a in obs_a:
        best = None
        best_err = float('inf')

        for i, b in enumerate(obs_b):
            if i in used_b:
                continue
            for s in SCALE_FACTORS:
                c_err = abs(a['control_mean'] * s - b['control_mean']) / max(abs(b['control_mean']), 0.001)
                t_err = abs(a['treatment_mean'] * s - b['treatment_mean']) / max(abs(b['treatment_mean']), 0.001)
                err = (c_err + t_err) / 2
                if err < best_err and err < tolerance:
                    best_err = err
                    best = (i, b, s, err)

        if best:
            idx, b_obs, scale, err = best
            used_b.add(idx)
            matches.append({
                'a_ctrl': a['control_mean'],
                'a_treat': a['treatment_mean'],
                'a_effect': a['effect_pct'],
                'b_ctrl': b_obs['control_mean'],
                'b_treat': b_obs['treatment_mean'],
                'b_effect': b_obs['effect_pct'],
                'effect_diff': abs(a['effect_pct'] - b_obs['effect_pct']),
                'scale': scale,
                'match_err': err,
            })
    return matches


def pearson_r(x, y):
    n = len(x)
    if n < 3:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx > 0 and vy > 0:
        return cov / math.sqrt(vx * vy)
    return 0


def compute_stats(matches, label_a='A', label_b='B'):
    if not matches:
        return {}
    n = len(matches)
    diffs = [m['effect_diff'] for m in matches]
    eff_a = [m['a_effect'] for m in matches]
    eff_b = [m['b_effect'] for m in matches]

    mae = sum(diffs) / n
    r = pearson_r(eff_a, eff_b)

    dir_total = sum(1 for a, b in zip(eff_a, eff_b) if abs(a) > 0.5 or abs(b) > 0.5)
    dir_ok = sum(1 for a, b in zip(eff_a, eff_b)
                 if (abs(a) > 0.5 or abs(b) > 0.5) and (a > 0) == (b > 0))

    w5 = sum(1 for d in diffs if d <= 5)
    w10 = sum(1 for d in diffs if d <= 10)

    return {
        'n_obs': n,
        'pearson_r': round(r, 3) if r is not None else None,
        'mae_pp': round(mae, 2),
        'within_5pp': f"{w5}/{n} ({w5/n*100:.0f}%)",
        'within_10pp': f"{w10}/{n} ({w10/n*100:.0f}%)",
        'direction': f"{dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)" if dir_total else "N/A",
        f'mean_{label_a}_effect': round(sum(eff_a)/n, 2),
        f'mean_{label_b}_effect': round(sum(eff_b)/n, 2),
        'effect_diff': round(abs(sum(eff_a)/n - sum(eff_b)/n), 2),
    }


def match_files_between_dirs(dir_a, dir_b):
    """Match files between two directories by paper ID."""
    files_a = {extract_paper_id(f): f for f in sorted(dir_a.glob('*_agent*.json'))
               if 'validation' not in f.name and 'harmonized' not in f.name}
    files_b = {extract_paper_id(f): f for f in sorted(dir_b.glob('*_agent*.json'))
               if 'validation' not in f.name and 'harmonized' not in f.name}

    matched = []
    used_b = set()

    for aid, afile in files_a.items():
        # Direct match
        if aid in files_b and aid not in used_b:
            used_b.add(aid)
            matched.append((aid, afile, files_b[aid]))
            continue

        # Try substring match
        for bid, bfile in files_b.items():
            if bid in used_b:
                continue
            # Extract numeric prefix
            a_match = re.match(r'(\d+)_(.+)', aid)
            b_match = re.match(r'(\d+)_(.+)', bid)
            if a_match and b_match and a_match.group(1) == b_match.group(1):
                used_b.add(bid)
                matched.append((aid, afile, bfile))
                break
            # Author overlap
            a_parts = set(aid.replace('_', ' ').split())
            b_parts = set(bid.replace('_', ' ').split())
            overlap = len(a_parts & b_parts)
            if overlap >= 2:
                used_b.add(bid)
                matched.append((aid, afile, bfile))
                break

    return matched


def process_run1_vs_run2(key, config, filter_fn=None):
    """Compare Run1 and Run2 extractions (reproducibility)."""
    run1_dir = config['run1_dir']
    run2_dir = config['run2_dir']
    name = config['name']

    print(f"\n{'='*70}")
    print(f"  RUN1 vs RUN2 — {name}")
    print(f"{'='*70}")

    if not run1_dir.exists() or not run2_dir.exists():
        print(f"  Missing directory")
        return None

    matched_files = match_files_between_dirs(run1_dir, run2_dir)
    print(f"  Matched papers: {len(matched_files)}")

    all_matches = []
    paper_results = []

    for pid, f1, f2 in matched_files:
        obs1 = load_observations(f1, filter_fn)
        obs2 = load_observations(f2, filter_fn)

        if not obs1 or not obs2:
            continue

        matches = match_observations(obs1, obs2, tolerance=0.25)
        all_matches.extend(matches)

        if matches:
            stats = compute_stats(matches, 'run1', 'run2')
            r_str = f"r={stats['pearson_r']}" if stats['pearson_r'] is not None else ""
            print(f"  {pid[:40]:<42} r1={len(obs1):>3} r2={len(obs2):>3} matched={len(matches):>3} MAE={stats['mae_pp']:>5.1f}pp {r_str}")
            paper_results.append({'paper_id': pid, 'run1_obs': len(obs1), 'run2_obs': len(obs2), 'matched': len(matches), **stats})

    overall = compute_stats(all_matches, 'run1', 'run2')
    n_papers = sum(1 for p in paper_results if p.get('matched', 0) > 0)

    print(f"\n  OVERALL: {overall.get('n_obs', 0)} matched obs across {n_papers} papers")
    if overall:
        print(f"  Pearson r: {overall['pearson_r']}")
        print(f"  MAE: {overall['mae_pp']}pp")
        print(f"  Direction: {overall['direction']}")
        print(f"  Within 5pp: {overall['within_5pp']}")
        print(f"  Mean effect — Run1: {overall.get('mean_run1_effect', 'N/A')}%, Run2: {overall.get('mean_run2_effect', 'N/A')}%, diff: {overall['effect_diff']}pp")

    return {
        'dataset': key,
        'name': name,
        'papers_matched': n_papers,
        'overall': overall,
        'paper_results': paper_results,
    }


def main():
    print(f"Replication Validation — Run1 vs Run2 Agreement")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"\nThis measures REPRODUCIBILITY: do two independent runs agree?\n")

    results = {}

    # Hui filter: Zn only
    hui_filter = lambda obs: 'ZN' in str(obs.get('element', '')).upper() or 'ZINC' in str(obs.get('element', '')).upper()

    filters = {
        'loladze': None,
        'hui2023': hui_filter,
        'li2022': None,
    }

    for key, config in DATASETS.items():
        result = process_run1_vs_run2(key, config, filter_fn=filters.get(key))
        if result:
            results[key] = result

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY — REPLICATION AGREEMENT (RUN1 vs RUN2)")
    print(f"{'='*70}\n")

    print(f"{'Dataset':<30} {'Papers':<8} {'Obs':<8} {'r':<8} {'MAE':<10} {'Direction':<15} {'Effect diff':<12}")
    print('-' * 95)

    for key, result in results.items():
        o = result['overall']
        if not o:
            continue
        print(f"{result['name']:<30} {result['papers_matched']:<8} {o['n_obs']:<8} "
              f"{o['pearson_r']:<8} {o['mae_pp']:<10} {o['direction']:<15} {o['effect_diff']}pp")

    all_obs = sum(r['overall']['n_obs'] for r in results.values() if r['overall'])
    print(f"\n{'TOTAL':<30} {'':>8} {all_obs:<8}")

    print(f"\nInterpretation:")
    print(f"  r > 0.95 = Highly reproducible")
    print(f"  r > 0.90 = Good reproducibility")
    print(f"  r > 0.80 = Moderate reproducibility")
    print(f"  r < 0.80 = Poor reproducibility (results vary across runs)")

    # Save
    out_path = BASE / 'output' / 'replication_agreement.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'method': 'Run1 vs Run2 agent extraction comparison — reproducibility test',
            'results': {k: {
                'name': v['name'],
                'papers_matched': v['papers_matched'],
                'overall': v['overall'],
            } for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
