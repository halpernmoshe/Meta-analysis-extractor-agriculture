"""
Agent-Pipeline Agreement — GT-Free Cross-Validation

For each paper that both agent AND pipeline extracted, compare their
observations directly. This comparison needs NO ground truth, making it
the strongest epistemic warrant for both methods.

If agent and pipeline independently produce similar effect sizes,
this validates both methods without any circularity.

Usage:
    ./venv/Scripts/python.exe agent_pipeline_agreement.py
"""
import sys, json, math, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ── Configuration ──────────────────────────────────────────────────────────

DATASETS = {
    'loladze': {
        'name': 'Loladze 2014 (CO2/minerals)',
        'pipeline_dir': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\loladze_v3_combined"),
        'agent_dir': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\agent_extraction"),
        'pipeline_glob': '*_consensus.json',
        'agent_glob': '*_agent*.json',
        'value_key': 'element',  # What obs are about
        'filter_fn': None,  # No filtering
    },
    'hui2023': {
        'name': 'Hui 2023 (Zn/wheat)',
        'pipeline_dir': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\hui2023_full_35"),
        'agent_dir': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\hui2023_agent_extraction"),
        'pipeline_glob': '*_consensus.json',
        'agent_glob': '*_agent*.json',
        'value_key': 'element',
        'filter_fn': lambda obs: 'ZN' in str(obs.get('element', '')).upper() or 'ZINC' in str(obs.get('element', '')).upper(),
    },
    'li2022': {
        'name': 'Li 2022 (biostimulants/yield)',
        'pipeline_dir': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\li2022_combined"),
        'agent_dir': Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\li2022_agent_extraction"),
        'pipeline_glob': '*_consensus.json',
        'agent_glob': '*_agent*.json',
        'value_key': 'outcome',
        'filter_fn': None,
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


def extract_paper_id(filename, is_agent=False):
    """Normalize filename to a comparable paper ID."""
    stem = filename.stem
    if is_agent:
        stem = stem.replace('_agent_v2', '').replace('_agent', '')
    else:
        stem = stem.replace('_consensus', '')
    return stem.lower().strip()


def load_observations(json_path, filter_fn=None):
    """Load consensus_observations from a JSON file."""
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


def match_paper_ids(pipeline_files, agent_files):
    """Match pipeline and agent files by paper ID."""
    # Build lookup from normalized ID to file
    pipe_map = {}
    for f in pipeline_files:
        pid = extract_paper_id(f, is_agent=False)
        pipe_map[pid] = f

    agent_map = {}
    for f in agent_files:
        aid = extract_paper_id(f, is_agent=True)
        # Prefer v2 files
        if '_v2' in f.stem or aid not in agent_map:
            agent_map[aid] = f

    # Match by substring overlap
    matched = []
    used_agent = set()

    for pid, pfile in pipe_map.items():
        best_match = None
        best_score = 0

        for aid, afile in agent_map.items():
            if aid in used_agent:
                continue

            # Extract author+year patterns for matching
            # Try direct match first
            if pid == aid:
                best_match = (aid, afile)
                best_score = 1000
                break

            # Try substring match
            # Extract numeric prefix and author from both
            p_match = re.match(r'(\d+)_(.+)', pid)
            a_match = re.match(r'(\d+)_(.+)', aid)

            if p_match and a_match:
                p_num, p_rest = p_match.groups()
                a_num, a_rest = a_match.groups()
                if p_num == a_num:
                    # Same numeric prefix
                    score = 100
                    if best_score < score:
                        best_score = score
                        best_match = (aid, afile)
                    continue

            # Try author name overlap
            p_parts = set(pid.replace('_', ' ').split())
            a_parts = set(aid.replace('_', ' ').split())
            overlap = len(p_parts & a_parts)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_match = (aid, afile)

        if best_match:
            used_agent.add(best_match[0])
            matched.append((pid, pfile, best_match[0], best_match[1]))

    return matched


def match_observations_between(pipe_obs, agent_obs, tolerance=0.30):
    """Match pipeline obs to agent obs by value similarity."""
    matches = []
    used_agent = set()

    for p in pipe_obs:
        best = None
        best_err = float('inf')

        for i, a in enumerate(agent_obs):
            if i in used_agent:
                continue

            # Try matching with scale factors
            for s in SCALE_FACTORS:
                c_err = abs(p['control_mean'] * s - a['control_mean']) / max(abs(a['control_mean']), 0.001)
                t_err = abs(p['treatment_mean'] * s - a['treatment_mean']) / max(abs(a['treatment_mean']), 0.001)
                err = (c_err + t_err) / 2
                if err < best_err and err < tolerance:
                    best_err = err
                    best = (i, a, s, err)

        if best:
            idx, a_obs, scale, err = best
            used_agent.add(idx)
            matches.append({
                'pipe_ctrl': p['control_mean'],
                'pipe_treat': p['treatment_mean'],
                'pipe_effect': p['effect_pct'],
                'agent_ctrl': a_obs['control_mean'],
                'agent_treat': a_obs['treatment_mean'],
                'agent_effect': a_obs['effect_pct'],
                'effect_diff': abs(p['effect_pct'] - a_obs['effect_pct']),
                'scale': scale,
                'match_err': err,
                'element': p.get('element', a_obs.get('element', '')),
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


def compute_stats(matches):
    """Compute agreement statistics from matched observations."""
    if not matches:
        return {}
    n = len(matches)
    diffs = [m['effect_diff'] for m in matches]
    pipe_eff = [m['pipe_effect'] for m in matches]
    agent_eff = [m['agent_effect'] for m in matches]

    mae = sum(diffs) / n
    r = pearson_r(pipe_eff, agent_eff)

    # Direction agreement
    dir_total = sum(1 for p, a in zip(pipe_eff, agent_eff) if abs(p) > 0.5 or abs(a) > 0.5)
    dir_ok = sum(1 for p, a in zip(pipe_eff, agent_eff)
                 if (abs(p) > 0.5 or abs(a) > 0.5) and (p > 0) == (a > 0))

    w5 = sum(1 for d in diffs if d <= 5)
    w10 = sum(1 for d in diffs if d <= 10)

    return {
        'n_obs': n,
        'pearson_r': round(r, 3) if r is not None else None,
        'mae_pp': round(mae, 2),
        'within_5pp': f"{w5}/{n} ({w5/n*100:.0f}%)",
        'within_10pp': f"{w10}/{n} ({w10/n*100:.0f}%)",
        'direction': f"{dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)" if dir_total else "N/A",
        'mean_pipe_effect': round(sum(pipe_eff)/n, 2),
        'mean_agent_effect': round(sum(agent_eff)/n, 2),
        'effect_diff': round(abs(sum(pipe_eff)/n - sum(agent_eff)/n), 2),
    }


def process_dataset(key, config):
    """Process one dataset for agent-pipeline agreement."""
    name = config['name']
    pipe_dir = config['pipeline_dir']
    agent_dir = config['agent_dir']

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    if not pipe_dir.exists():
        print(f"  Pipeline dir not found: {pipe_dir}")
        return None
    if not agent_dir.exists():
        print(f"  Agent dir not found: {agent_dir}")
        return None

    pipe_files = sorted(pipe_dir.glob(config['pipeline_glob']))
    agent_files = sorted(agent_dir.glob(config['agent_glob']))

    # Filter out validation reports
    agent_files = [f for f in agent_files if 'validation_report' not in f.name and 'harmonized' not in f.name]

    print(f"  Pipeline files: {len(pipe_files)}")
    print(f"  Agent files: {len(agent_files)}")

    # Match papers
    matched_papers = match_paper_ids(pipe_files, agent_files)
    print(f"  Matched papers: {len(matched_papers)}\n")

    filter_fn = config.get('filter_fn')
    all_matches = []
    paper_results = []

    for pid, pfile, aid, afile in matched_papers:
        pipe_obs = load_observations(pfile, filter_fn)
        agent_obs = load_observations(afile, filter_fn)

        if not pipe_obs or not agent_obs:
            continue

        matches = match_observations_between(pipe_obs, agent_obs, tolerance=0.25)
        all_matches.extend(matches)

        if matches:
            stats = compute_stats(matches)
            r_str = f"r={stats['pearson_r']}" if stats['pearson_r'] is not None else ""
            print(f"  {pid[:40]:<42} pipe={len(pipe_obs):>3} agent={len(agent_obs):>3} matched={len(matches):>3} MAE={stats['mae_pp']:>5.1f}pp {r_str}")
            paper_results.append({
                'paper_id': pid,
                'pipe_obs': len(pipe_obs),
                'agent_obs': len(agent_obs),
                'matched': len(matches),
                **stats,
            })
        else:
            print(f"  {pid[:40]:<42} pipe={len(pipe_obs):>3} agent={len(agent_obs):>3} matched=  0")
            paper_results.append({
                'paper_id': pid,
                'pipe_obs': len(pipe_obs),
                'agent_obs': len(agent_obs),
                'matched': 0,
            })

    # Overall stats
    overall = compute_stats(all_matches)
    n_papers_matched = sum(1 for p in paper_results if p.get('matched', 0) > 0)

    print(f"\n  OVERALL: {overall.get('n_obs', 0)} matched obs across {n_papers_matched} papers")
    if overall:
        print(f"  Pearson r: {overall['pearson_r']}")
        print(f"  MAE: {overall['mae_pp']}pp")
        print(f"  Direction: {overall['direction']}")
        print(f"  Within 5pp: {overall['within_5pp']}")
        print(f"  Within 10pp: {overall['within_10pp']}")
        print(f"  Mean effect — Pipeline: {overall['mean_pipe_effect']}%, Agent: {overall['mean_agent_effect']}%, diff: {overall['effect_diff']}pp")

    return {
        'dataset': key,
        'name': name,
        'papers_matched': n_papers_matched,
        'overall': overall,
        'paper_results': paper_results,
    }


def main():
    print(f"Agent-Pipeline Agreement — GT-Free Cross-Validation")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"\nThis comparison uses NO ground truth.")
    print(f"If both methods agree, it independently validates both.\n")

    results = {}
    for key, config in DATASETS.items():
        result = process_dataset(key, config)
        if result:
            results[key] = result

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY — AGENT-PIPELINE AGREEMENT (GT-FREE)")
    print(f"{'='*70}\n")

    print(f"{'Dataset':<30} {'Papers':<8} {'Obs':<8} {'r':<8} {'MAE':<10} {'Direction':<15} {'Effect diff':<12}")
    print('-' * 95)

    for key, result in results.items():
        o = result['overall']
        if not o:
            continue
        print(f"{result['name']:<30} {result['papers_matched']:<8} {o['n_obs']:<8} "
              f"{o['pearson_r']:<8} {o['mae_pp']:<10} {o['direction']:<15} {o['effect_diff']}pp")

    # Grand total
    all_obs = sum(r['overall']['n_obs'] for r in results.values() if r['overall'])
    print(f"\n{'TOTAL':<30} {'':>8} {all_obs:<8}")

    print(f"\nInterpretation:")
    print(f"  r > 0.9  = Excellent agreement (independently validates both methods)")
    print(f"  r > 0.8  = Good agreement (methods largely concur)")
    print(f"  r > 0.7  = Moderate agreement (reasonable concordance)")
    print(f"  r < 0.7  = Poor agreement (methods disagree on many papers)")

    # Save
    out_path = Path(__file__).parent / 'output' / 'agent_pipeline_agreement.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'method': 'Direct comparison of agent vs pipeline extraction — NO ground truth used',
            'results': {k: {
                'name': v['name'],
                'papers_matched': v['papers_matched'],
                'overall': v['overall'],
            } for k, v in results.items()},
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
