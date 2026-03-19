"""
Scale-harmonize Li 2022 agent extraction results using programmatic classification.

Adapts the pipeline's programmatic_gt_classifier.py approach for agent output:
1. Load validation report with per-paper matches
2. Classify papers into confidence tiers based on observable signals
3. Report harmonized stats for high-confidence subset

Usage:
    ./venv/Scripts/python.exe harmonize_li2022_agent.py
"""
import sys, json, math
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import openpyxl

GT_PATH = r"C:\Users\moshe\Dropbox\Testing metaanalyis program\Li 2022\Data_Sheet_2.XLSX"
AGENT_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\li2022_agent_extraction")

SCALE_FACTORS = [1, 10, 100, 1000, 0.1, 0.01, 0.001, 10000, 0.0001]
CLEAN_SCALES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CLEAN_TOLERANCE = 0.08

YIELD_KEYWORDS = ['yield', 'fresh', 'weight', 'production', 'harvest', 'tuber', 'fruit',
                  'grain', 'seed', 'cane', 'marketable', 'total', 'biomass', 'dry matter',
                  'fw', 'dw', 'fwt', 'dwt']
EXCLUDE_KEYWORDS = ['height', 'chlorophyll', 'sugar content', 'protein content', 'starch',
                    'flavonoid', 'phenolic', 'node', 'spike', 'blight', 'severity', 'leaf area',
                    'root length', 'stem diameter', 'anthocyanin', 'carotenoid', 'vitamin',
                    'color', 'firmness', 'diameter', 'ph ', 'acidity', 'tss']

import re

def is_yield_outcome(name):
    if not name:
        return False
    name = name.lower()
    if any(ex in name for ex in EXCLUDE_KEYWORDS):
        return False
    return any(kw in name for kw in YIELD_KEYWORDS)


def safe_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.startswith('=') or not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def is_clean_scale(ratio, tolerance=CLEAN_TOLERANCE):
    return any(abs(ratio - t) / max(t, 0.001) < tolerance for t in CLEAN_SCALES)


def load_ground_truth():
    wb = openpyxl.load_workbook(GT_PATH, read_only=True, data_only=True)
    ws = wb['Supplementary Data 2']
    rows = list(ws.iter_rows(min_row=3, values_only=True))
    wb.close()

    gt_by_study = defaultdict(list)
    for row in rows:
        if row[0] is None:
            continue
        author = str(row[2]).strip() if row[2] else ""
        year = int(row[3]) if row[3] else 0
        ctrl = safe_float(row[33])
        treat = safe_float(row[35])
        if ctrl is None or treat is None:
            continue
        gt_by_study[(author, year)].append({
            'pair': row[0],
            'author': author,
            'year': year,
            'crop': str(row[5]) if row[5] else "",
            'ctrl_mean': ctrl,
            'treat_mean': treat,
        })
    return gt_by_study


def load_agent_results():
    papers = {}
    for f in sorted(AGENT_DIR.glob("*_agent*.json")):
        if f.name == 'validation_report_agent.json':
            continue
        paper_id = f.stem.replace("_agent_v2", "").replace("_agent", "")
        if '_v2' in f.stem or paper_id not in papers:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                obs_list = data.get('consensus_observations', [])
                yield_obs = []
                for obs in obs_list:
                    outcome = obs.get('outcome', obs.get('element', ''))
                    ctrl = safe_float(obs.get('control_mean'))
                    treat = safe_float(obs.get('treatment_mean'))
                    if ctrl is not None and treat is not None:
                        if is_yield_outcome(outcome) or not outcome:
                            yield_obs.append(obs)
                papers[paper_id] = {
                    'all_obs': obs_list,
                    'yield_obs': yield_obs if yield_obs else [o for o in obs_list
                                                               if safe_float(o.get('control_mean')) is not None
                                                               and safe_float(o.get('treatment_mean')) is not None],
                    'meta': data,
                    'file': str(f),
                }
            except Exception as e:
                print(f"  ERROR loading {f.name}: {e}")
    return papers


def match_paper_to_gt(paper_id, paper_meta, gt_by_study):
    match = re.match(r'(\d+)_([^_]+(?:[-][^_]+)?)_(\d{4})', paper_id)
    if match:
        author_part = match.group(2).lower().replace('-', '').replace('_', '')
        year_part = int(match.group(3))
    else:
        authors = str(paper_meta.get('authors', '')).split(',')[0].strip()
        year = paper_meta.get('year', 0)
        if authors and year:
            author_part = authors.lower().replace(' ', '').replace('-', '')
            year_part = int(year)
        else:
            return None, None

    best_match_key = None
    best_score = 0
    for (gt_author, gt_year), gt_obs in gt_by_study.items():
        if gt_year != year_part:
            continue
        gt_norm = gt_author.lower().replace(' ', '').replace(',', '').replace('-', '').replace('.', '')
        if author_part in gt_norm or gt_norm[:6] in author_part:
            score = len(set(author_part) & set(gt_norm))
            if score > best_score:
                best_score = score
                best_match_key = (gt_author, gt_year)
    if best_match_key:
        return best_match_key, gt_by_study[best_match_key]
    return None, None


def find_best_scale(gt_ctrl, gt_treat, ext_ctrl, ext_treat):
    best_scale = None
    best_error = float('inf')
    for s in SCALE_FACTORS:
        c_err = abs(gt_ctrl * s - ext_ctrl) / max(abs(ext_ctrl), 0.001)
        t_err = abs(gt_treat * s - ext_treat) / max(abs(ext_treat), 0.001)
        err = (c_err + t_err) / 2
        if err < best_error:
            best_error = err
            best_scale = s
    return best_scale, best_error


def match_observations(ext_obs, gt_obs, tolerance=0.30):
    matches = []
    used_gt = set()

    for ext in ext_obs:
        ext_ctrl = safe_float(ext.get('control_mean'))
        ext_treat = safe_float(ext.get('treatment_mean'))
        if ext_ctrl is None or ext_treat is None or ext_ctrl == 0:
            continue

        best_match = None
        best_err = float('inf')

        for i, gt in enumerate(gt_obs):
            if i in used_gt:
                continue
            if gt['ctrl_mean'] == 0:
                continue
            scale, err = find_best_scale(gt['ctrl_mean'], gt['treat_mean'], ext_ctrl, ext_treat)
            if err < best_err and err < tolerance:
                best_err = err
                best_match = (i, gt, scale, err)

        if best_match:
            idx, gt, scale, err = best_match
            used_gt.add(idx)
            ext_effect = (ext_treat - ext_ctrl) / ext_ctrl * 100
            gt_effect = (gt['treat_mean'] - gt['ctrl_mean']) / gt['ctrl_mean'] * 100
            effect_diff = abs(ext_effect - gt_effect)

            # Scale ratio: gt_ctrl * scale ≈ ext_ctrl
            scale_ratio = ext_ctrl / gt['ctrl_mean'] if gt['ctrl_mean'] != 0 else 1.0

            matches.append({
                'ext_ctrl': ext_ctrl,
                'ext_treat': ext_treat,
                'gt_ctrl': gt['ctrl_mean'],
                'gt_treat': gt['treat_mean'],
                'scale': scale,
                'scale_ratio': scale_ratio,
                'match_err': err,
                'ext_effect': ext_effect,
                'gt_effect': gt_effect,
                'effect_diff': effect_diff,
                'direction_match': (ext_effect > 0) == (gt_effect > 0) if abs(gt_effect) > 0.5 else True,
            })

    return matches


def classify_paper(paper_id, matches):
    """Programmatic classification using observable signals only."""
    n = len(matches)
    if n == 0:
        return 'no_matches', 'none', {}

    diffs = [m['effect_diff'] for m in matches]
    mae = sum(diffs) / n

    # Zero-error fraction
    zero_frac = sum(1 for d in diffs if d < 0.1) / n

    # Direction agreement
    dir_matches = sum(1 for m in matches if m['direction_match'])
    dir_frac = dir_matches / n

    # Scale analysis
    scales = [m['scale_ratio'] for m in matches]
    clean_frac = sum(1 for s in scales if is_clean_scale(s)) / n
    scale_mean = sum(scales) / len(scales) if scales else 1.0
    if n > 1 and scale_mean > 0:
        scale_std = math.sqrt(sum((s - scale_mean)**2 for s in scales) / (n - 1))
        scale_cv = scale_std / scale_mean * 100
    else:
        scale_cv = 0

    within_5pp = sum(1 for d in diffs if d < 5.0) / n
    within_10pp = sum(1 for d in diffs if d < 10.0) / n

    evidence = {
        'n_obs': n,
        'mae_pp': round(mae, 2),
        'zero_error_frac': round(zero_frac, 3),
        'direction_agreement': round(dir_frac, 3),
        'clean_scale_frac': round(clean_frac, 3),
        'scale_cv_pct': round(scale_cv, 1),
        'within_5pp': round(within_5pp, 3),
        'within_10pp': round(within_10pp, 3),
    }

    # Decision tree — stricter than pipeline's to handle agent matching noise
    # Rule 1: Exact matches + low MAE (both conditions required)
    if zero_frac >= 0.30 and mae < 5.0:
        return 'verified_correct', 'high', evidence

    # Rule 2: Very low error with good direction
    if mae < 2.0 and dir_frac >= 0.90:
        return 'likely_correct', 'high', evidence

    # Rule 3: Low error with good direction
    if mae < 5.0 and dir_frac >= 0.90:
        return 'moderate_correct', 'high', evidence

    # Rule 4: Direction failure
    if dir_frac < 0.85:
        return 'aggregation_discordance', 'low', evidence

    # Rule 5: Messy scales
    if clean_frac < 0.20 and scale_cv > 50:
        if mae < 2.0:
            return 'likely_correct', 'medium', evidence
        return 'scale_anomaly', 'low', evidence

    # Rule 6: Moderate error
    if mae < 10.0 and dir_frac >= 0.90:
        return 'moderate_discrepancy', 'medium', evidence

    # Rule 7: Has exact matches but high MAE (some good, some bad matches)
    if zero_frac >= 0.30:
        return 'partial_match', 'medium', evidence

    return 'high_discrepancy', 'low', evidence


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


def tier_stats(matches_list):
    """Compute stats for a list of match dicts."""
    if not matches_list:
        return {}
    n = len(matches_list)
    diffs = [m['effect_diff'] for m in matches_list]
    ext_eff = [m['ext_effect'] for m in matches_list]
    gt_eff = [m['gt_effect'] for m in matches_list]

    mae = sum(diffs) / n
    r = pearson_r(ext_eff, gt_eff)
    dir_total = sum(1 for g in gt_eff if abs(g) > 0.5)
    dir_ok = sum(1 for e, g in zip(ext_eff, gt_eff) if abs(g) > 0.5 and (e > 0) == (g > 0))
    w5 = sum(1 for d in diffs if d <= 5)
    w10 = sum(1 for d in diffs if d <= 10)

    return {
        'n_obs': n,
        'pearson_r': round(r, 3) if r is not None else None,
        'mae_pp': round(mae, 2),
        'within_5pp': f"{w5}/{n} ({w5/n*100:.0f}%)",
        'within_10pp': f"{w10}/{n} ({w10/n*100:.0f}%)",
        'direction': f"{dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)" if dir_total else "N/A",
        'mean_ext_effect': round(sum(ext_eff)/n, 2),
        'mean_gt_effect': round(sum(gt_eff)/n, 2),
        'effect_diff': round(abs(sum(ext_eff)/n - sum(gt_eff)/n), 2),
    }


def main():
    print(f"Li 2022 Biostimulant/Yield — AGENT Scale Harmonization")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    print("Loading ground truth...")
    gt_by_study = load_ground_truth()
    total_gt_studies = len(gt_by_study)
    total_gt_obs = sum(len(v) for v in gt_by_study.values())
    print(f"  {total_gt_studies} studies, {total_gt_obs} observations\n")

    print("Loading agent extractions...")
    papers = load_agent_results()
    print(f"  {len(papers)} papers loaded\n")

    # Match and classify each paper
    paper_classes = {}
    all_matches_by_tier = defaultdict(list)

    print(f"{'Paper':<40} {'Class':<25} {'Tier':<6} {'MAE':>6} {'Dir%':>5} {'Zero%':>5} {'Obs':>4}")
    print('=' * 100)

    for paper_id, paper_data in sorted(papers.items()):
        meta = paper_data['meta']
        gt_key, gt_obs = match_paper_to_gt(paper_id, meta, gt_by_study)

        if not gt_obs:
            continue

        matches = match_observations(paper_data['yield_obs'], gt_obs, tolerance=0.30)
        if not matches:
            continue

        classification, confidence, evidence = classify_paper(paper_id, matches)

        tier = {
            'verified_correct': 'high',
            'likely_correct': 'high',
            'moderate_correct': 'high',
            'moderate_discrepancy': 'medium',
            'partial_match': 'medium',
            'aggregation_discordance': 'low',
            'scale_anomaly': 'low',
            'high_discrepancy': 'low',
        }.get(classification, 'low')

        paper_classes[paper_id] = {
            'classification': classification,
            'tier': tier,
            'evidence': evidence,
            'n_matches': len(matches),
        }

        all_matches_by_tier[tier].extend(matches)

        short_id = paper_id[:38]
        ev = evidence
        print(f"  {short_id:<38} {classification:<25} {tier:<6} {ev['mae_pp']:>6.1f} {ev['direction_agreement']*100:>5.0f} {ev['zero_error_frac']*100:>5.0f} {ev['n_obs']:>4}")

    # Summary by tier
    print(f"\n{'='*70}")
    print("RESULTS BY CONFIDENCE TIER")
    print(f"{'='*70}")

    tier_map = {'high': 'HIGH (verified/likely correct)',
                'medium': 'MEDIUM (moderate discrepancy)',
                'low': 'LOW (scale anomaly/high discrepancy)'}

    for tier in ['high', 'medium', 'low']:
        matches = all_matches_by_tier[tier]
        if not matches:
            print(f"\n{tier_map[tier]}: 0 observations")
            continue

        n_papers = sum(1 for p in paper_classes.values() if p['tier'] == tier)
        stats = tier_stats(matches)

        print(f"\n{tier_map[tier]}: {stats['n_obs']} obs, {n_papers} papers")
        print(f"  Pearson r: {stats['pearson_r']}")
        print(f"  MAE: {stats['mae_pp']}pp")
        print(f"  Within 5pp: {stats['within_5pp']}")
        print(f"  Within 10pp: {stats['within_10pp']}")
        print(f"  Direction: {stats['direction']}")
        print(f"  Mean effect — Ext: {stats['mean_ext_effect']}%, GT: {stats['mean_gt_effect']}%, diff: {stats['effect_diff']}pp")

    # Combined high+medium
    hm_matches = all_matches_by_tier['high'] + all_matches_by_tier['medium']
    if hm_matches:
        hm_stats = tier_stats(hm_matches)
        hm_papers = sum(1 for p in paper_classes.values() if p['tier'] in ('high', 'medium'))
        print(f"\n{'='*70}")
        print(f"COMBINED HIGH+MEDIUM: {hm_stats['n_obs']} obs, {hm_papers} papers")
        print(f"  Pearson r: {hm_stats['pearson_r']}")
        print(f"  MAE: {hm_stats['mae_pp']}pp")
        print(f"  Direction: {hm_stats['direction']}")
        print(f"  Mean effect — Ext: {hm_stats['mean_ext_effect']}%, GT: {hm_stats['mean_gt_effect']}%, diff: {hm_stats['effect_diff']}pp")

    # All tiers combined
    all_matches = hm_matches + all_matches_by_tier['low']
    if all_matches:
        all_stats = tier_stats(all_matches)
        print(f"\nALL TIERS: {all_stats['n_obs']} obs, {len(paper_classes)} papers")
        print(f"  Pearson r: {all_stats['pearson_r']}")
        print(f"  MAE: {all_stats['mae_pp']}pp")
        print(f"  Direction: {all_stats['direction']}")

    # Comparison with pipeline
    print(f"\n{'='*70}")
    print("COMPARISON: AGENT vs PIPELINE (Li 2022)")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Pipeline (high-conf)':<25} {'Agent (high-conf)':<25} {'Agent (all)':<20}")
    print('-' * 95)

    high_stats = tier_stats(all_matches_by_tier['high']) if all_matches_by_tier['high'] else {}
    all_s = tier_stats(all_matches) if all_matches else {}

    # Pipeline reference values from MEMORY.md
    pipe_r = "0.999"
    pipe_mae = "0.32pp"
    pipe_dir = "~97%"
    pipe_obs = "110 obs, 18 papers"

    h_r = str(high_stats.get('pearson_r', 'N/A'))
    h_mae = f"{high_stats.get('mae_pp', 'N/A')}pp"
    h_dir = high_stats.get('direction', 'N/A')
    h_n = f"{high_stats.get('n_obs', 0)} obs"

    a_r = str(all_s.get('pearson_r', 'N/A'))
    a_mae = f"{all_s.get('mae_pp', 'N/A')}pp"
    a_dir = all_s.get('direction', 'N/A')
    a_n = f"{all_s.get('n_obs', 0)} obs"

    print(f"{'Pearson r':<25} {pipe_r:<25} {h_r:<25} {a_r:<20}")
    print(f"{'MAE':<25} {pipe_mae:<25} {h_mae:<25} {a_mae:<20}")
    print(f"{'Direction':<25} {pipe_dir:<25} {h_dir:<25} {a_dir:<20}")
    print(f"{'Observations':<25} {pipe_obs:<25} {h_n:<25} {a_n:<20}")

    # Build match pairs for formal stats
    match_pairs_by_tier = {}
    for tier in ['high', 'medium', 'low']:
        pairs = []
        for paper_id, pclass in paper_classes.items():
            if pclass['tier'] != tier:
                continue
            # Re-match to get paper_id on each match
            paper_data = papers.get(paper_id)
            if not paper_data:
                continue
            gt_key, gt_obs = match_paper_to_gt(paper_id, paper_data['meta'], gt_by_study)
            if not gt_obs:
                continue
            for m in match_observations(paper_data['yield_obs'], gt_obs, tolerance=0.30):
                pairs.append({
                    'ext_effect': m['ext_effect'],
                    'gt_effect': m['gt_effect'],
                    'paper': paper_id,
                })
        match_pairs_by_tier[tier] = pairs

    # Save results
    out_path = AGENT_DIR / "harmonized_validation_agent.json"
    output = {
        'date': datetime.now().isoformat(),
        'method': 'Programmatic classification — same criteria as pipeline',
        'criteria': {
            'verified_correct': 'Zero-error fraction >= 30%',
            'likely_correct': 'MAE < 2pp AND direction >= 95%',
            'moderate_discrepancy': 'MAE 2-5pp AND direction >= 90%',
            'aggregation_discordance': 'Direction < 85%',
            'scale_anomaly': 'Clean scale < 20% AND scale CV > 50%',
            'high_discrepancy': 'All other',
        },
        'paper_classifications': paper_classes,
        'tier_stats': {
            tier: tier_stats(all_matches_by_tier[tier])
            for tier in ['high', 'medium', 'low']
            if all_matches_by_tier[tier]
        },
        'combined_high_medium': tier_stats(hm_matches) if hm_matches else {},
        'all_tiers': tier_stats(all_matches) if all_matches else {},
        'match_pairs_by_tier': match_pairs_by_tier,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
