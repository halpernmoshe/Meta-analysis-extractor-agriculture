"""
Validate Li 2022 biostimulant/yield extraction against ground truth.
Works with consensus pipeline output format.

Usage:
    python validate_li2022.py [--results-dir path/to/results]
"""
import sys, os, json, math, csv, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import openpyxl
import numpy as np

GT_PATH = r"C:\Users\moshe\Dropbox\Testing metaanalyis program\Li 2022\Data_Sheet_2.XLSX"
DEFAULT_RESULTS_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\li2022_consensus")

# Scale factors for unit conversion (g vs kg vs t/ha etc.)
SCALE_FACTORS = [1, 10, 100, 1000, 0.1, 0.01, 0.001, 10000, 0.0001]

# Yield-related keywords
YIELD_KEYWORDS = ['yield', 'fresh', 'weight', 'production', 'harvest', 'tuber', 'fruit',
                  'grain', 'seed', 'cane', 'marketable', 'total', 'biomass', 'dry matter',
                  'fw', 'dw', 'fwt', 'dwt']
EXCLUDE_KEYWORDS = ['height', 'chlorophyll', 'sugar content', 'protein content', 'starch',
                    'flavonoid', 'phenolic', 'node', 'spike', 'blight', 'severity', 'leaf area',
                    'root length', 'stem diameter', 'anthocyanin', 'carotenoid', 'vitamin',
                    'color', 'firmness', 'diameter', 'ph ', 'acidity', 'tss']


def is_yield_outcome(element_name):
    """Check if an observation element/outcome is yield-related."""
    if not element_name:
        return False
    name = element_name.lower()
    if any(ex in name for ex in EXCLUDE_KEYWORDS):
        return False
    return any(kw in name for kw in YIELD_KEYWORDS)


def find_best_scale(gt_ctrl, gt_treat, ext_ctrl, ext_treat):
    """Find scale factor that best aligns GT to extracted values."""
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


def safe_float(val):
    """Convert a value to float, returning None for formulas or invalid values."""
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


def load_ground_truth():
    """Load Li 2022 ground truth from Excel."""
    wb = openpyxl.load_workbook(GT_PATH, read_only=True, data_only=True)
    ws = wb['Supplementary Data 2']

    rows = list(ws.iter_rows(min_row=3, values_only=True))  # Skip title + header
    wb.close()

    gt_by_study = defaultdict(list)
    for row in rows:
        if row[0] is None:
            continue
        author = str(row[2]).strip() if row[2] else ""
        year = int(row[3]) if row[3] else 0
        ctrl = safe_float(row[33])  # CtrlFreshYield
        treat = safe_float(row[35])  # TreatmentFreshYield
        ctrl_sd = safe_float(row[34])  # CtrlFreshYieldSD
        treat_sd = safe_float(row[36])  # TreatmentFreshYieldSD

        if ctrl is None or treat is None:
            continue

        gt_by_study[(author, year)].append({
            'pair': row[0],
            'study': row[1],
            'author': author,
            'year': year,
            'crop': str(row[5]) if row[5] else "",
            'product': str(row[26]) if row[26] else "",
            'category': str(row[27]) if row[27] else "",
            'n': int(row[14]) if row[14] else None,
            'ctrl_mean': ctrl,
            'treat_mean': treat,
            'ctrl_sd': ctrl_sd,
            'treat_sd': treat_sd,
        })

    return gt_by_study


def load_consensus_results(results_dir):
    """Load consensus pipeline output files."""
    results_dir = Path(results_dir)
    papers = {}

    for f in sorted(results_dir.glob("*_consensus.json")):
        with open(f, 'r', encoding='utf-8') as fh:
            data = json.load(fh)

        paper_id = data.get('paper_id', f.stem.replace('_consensus', ''))
        obs_list = data.get('consensus_observations', [])

        # Filter to yield-related observations
        yield_obs = []
        for obs in obs_list:
            element = obs.get('element', '')
            if is_yield_outcome(element):
                yield_obs.append(obs)

        # If no yield-specific obs found, use all obs (the config might use generic element names)
        if not yield_obs:
            yield_obs = [o for o in obs_list if o.get('control_mean') is not None and o.get('treatment_mean') is not None]

        papers[paper_id] = {
            'all_obs': obs_list,
            'yield_obs': yield_obs,
            'meta': data
        }

    return papers


def match_paper_to_gt(paper_id, gt_by_study):
    """Try to match a paper_id (from PDF filename) to a GT study."""
    # Extract author and year from paper_id
    # Format: "002_Abdel-Mawgoud_2010_Growth..." or similar
    match = re.match(r'(\d+)_([^_]+(?:_[^_]+)?)_(\d{4})', paper_id)
    if not match:
        # Try simpler patterns
        match = re.match(r'([A-Za-z-]+).*?(\d{4})', paper_id)
        if not match:
            return None

    if len(match.groups()) == 3:
        author_part = match.group(2).lower().replace('-', '').replace('_', '')
        year_part = int(match.group(3))
    else:
        author_part = match.group(1).lower().replace('-', '').replace('_', '')
        year_part = int(match.group(2))

    # Try to find matching GT study
    best_match = None
    best_score = 0

    for (gt_author, gt_year), gt_obs in gt_by_study.items():
        if gt_year != year_part:
            continue

        gt_author_norm = gt_author.lower().replace(' ', '').replace(',', '').replace('-', '').replace('.', '')

        # Check if author names overlap
        if author_part in gt_author_norm or gt_author_norm.startswith(author_part[:5]):
            score = len(author_part)
            if score > best_score:
                best_score = score
                best_match = (gt_author, gt_year)

    return best_match


def validate(results_dir):
    """Main validation logic."""
    print("=" * 70)
    print("Li 2022 VALIDATION: Biostimulant/Yield Consensus Pipeline")
    print("=" * 70)

    # Load data
    gt_by_study = load_ground_truth()
    total_gt = sum(len(v) for v in gt_by_study.values())
    print(f"Ground truth: {total_gt} observations from {len(gt_by_study)} studies")

    papers = load_consensus_results(results_dir)
    total_extracted = sum(len(p['yield_obs']) for p in papers.values())
    print(f"Extracted: {total_extracted} yield observations from {len(papers)} papers")
    print()

    # Match and validate
    all_matches = []
    paper_stats = []
    matched_studies = set()

    for paper_id, paper_data in papers.items():
        gt_key = match_paper_to_gt(paper_id, gt_by_study)
        if gt_key is None:
            paper_stats.append({
                'paper_id': paper_id,
                'status': 'no_gt_match',
                'n_extracted': len(paper_data['yield_obs']),
                'n_matched': 0
            })
            continue

        gt_obs = gt_by_study[gt_key]
        ext_obs = paper_data['yield_obs']
        matched_studies.add(gt_key)

        paper_matches = 0
        used_ext = set()

        for gt in gt_obs:
            best_match = None
            best_idx = None
            best_err = float('inf')
            best_scale = None

            for i, ext in enumerate(ext_obs):
                if i in used_ext:
                    continue

                ext_ctrl = ext.get('control_mean')
                ext_treat = ext.get('treatment_mean')
                if ext_ctrl is None or ext_treat is None:
                    continue
                try:
                    ext_ctrl = float(ext_ctrl)
                    ext_treat = float(ext_treat)
                except (ValueError, TypeError):
                    continue

                scale, err = find_best_scale(gt['ctrl_mean'], gt['treat_mean'], ext_ctrl, ext_treat)
                if err < best_err and err < 0.30:
                    best_err = err
                    best_match = ext
                    best_idx = i
                    best_scale = scale

            if best_match is not None:
                paper_matches += 1
                used_ext.add(best_idx)

                ext_ctrl = float(best_match['control_mean'])
                ext_treat = float(best_match['treatment_mean'])

                # Calculate effect sizes
                gt_effect = ((gt['treat_mean'] - gt['ctrl_mean']) / gt['ctrl_mean']) * 100 if gt['ctrl_mean'] != 0 else 0
                ext_effect = ((ext_treat - ext_ctrl) / ext_ctrl) * 100 if ext_ctrl != 0 else 0

                effect_diff = abs(ext_effect - gt_effect)
                direction_match = (gt_effect > 0) == (ext_effect > 0) or abs(gt_effect) < 1

                all_matches.append({
                    'paper_id': paper_id,
                    'gt_author': gt['author'],
                    'gt_year': gt['year'],
                    'crop': gt['crop'],
                    'product': gt['product'],
                    'category': gt['category'],
                    'gt_ctrl': gt['ctrl_mean'],
                    'gt_treat': gt['treat_mean'],
                    'ext_ctrl': ext_ctrl,
                    'ext_treat': ext_treat,
                    'scale': best_scale,
                    'gt_effect_pct': round(gt_effect, 2),
                    'ext_effect_pct': round(ext_effect, 2),
                    'effect_diff_pp': round(effect_diff, 2),
                    'direction_match': direction_match,
                    'match_error': round(best_err, 4),
                })

        paper_stats.append({
            'paper_id': paper_id,
            'status': 'matched',
            'gt_author': gt_key[0],
            'gt_year': gt_key[1],
            'n_gt': len(gt_obs),
            'n_extracted': len(ext_obs),
            'n_matched': paper_matches,
            'capture_rate': round(paper_matches / len(gt_obs) * 100, 1) if gt_obs else 0
        })

    # Compute summary statistics
    if not all_matches:
        print("NO MATCHES FOUND - check paper naming and ground truth mapping")
        return

    effect_diffs = [m['effect_diff_pp'] for m in all_matches]
    gt_effects = [m['gt_effect_pct'] for m in all_matches]
    ext_effects = [m['ext_effect_pct'] for m in all_matches]
    directions = [m['direction_match'] for m in all_matches]

    mae = np.mean(effect_diffs)
    median_ae = np.median(effect_diffs)

    # Pearson r
    if len(gt_effects) > 2:
        r = np.corrcoef(gt_effects, ext_effects)[0, 1]
    else:
        r = float('nan')

    within_5 = sum(1 for d in effect_diffs if d <= 5) / len(effect_diffs)
    within_10 = sum(1 for d in effect_diffs if d <= 10) / len(effect_diffs)
    within_20 = sum(1 for d in effect_diffs if d <= 20) / len(effect_diffs)
    direction_acc = sum(directions) / len(directions)

    papers_matched = sum(1 for s in paper_stats if s['status'] == 'matched')

    print(f"Papers matched to GT: {papers_matched}/{len(papers)}")
    print(f"GT studies covered: {len(matched_studies)}/{len(gt_by_study)}")
    print(f"Observations matched: {len(all_matches)}")
    print()
    print("ACCURACY METRICS:")
    print(f"  Pearson r:       {r:.3f}")
    print(f"  MAE:             {mae:.2f}%")
    print(f"  Median AE:       {median_ae:.2f}%")
    print(f"  Direction:       {direction_acc:.1%}")
    print(f"  Within 5pp:      {within_5:.1%}")
    print(f"  Within 10pp:     {within_10:.1%}")
    print(f"  Within 20pp:     {within_20:.1%}")
    print()

    # Overall effect comparison
    gt_overall = np.mean(gt_effects)
    ext_overall = np.mean(ext_effects)
    print(f"OVERALL EFFECT:")
    print(f"  GT mean:         {gt_overall:.2f}%")
    print(f"  Extracted mean:  {ext_overall:.2f}%")
    print(f"  Difference:      {abs(gt_overall - ext_overall):.2f}pp")
    print()

    # Per-paper summary
    print("PER-PAPER RESULTS:")
    print(f"{'Paper':<35} {'GT obs':>6} {'Ext':>5} {'Match':>5} {'Rate':>6}")
    print("-" * 60)
    for s in sorted(paper_stats, key=lambda x: -x.get('n_matched', 0)):
        if s['status'] == 'matched':
            print(f"  {s['paper_id'][:33]:<33} {s['n_gt']:>6} {s['n_extracted']:>5} {s['n_matched']:>5} {s['capture_rate']:>5.1f}%")
        else:
            print(f"  {s['paper_id'][:33]:<33}     - {s['n_extracted']:>5}     - no GT")

    # Save results
    output_dir = results_dir

    # Save matches CSV
    matches_path = output_dir / "validation_matches.csv"
    if all_matches:
        keys = list(all_matches[0].keys())
        with open(matches_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_matches)
        print(f"\nSaved {len(all_matches)} matches to {matches_path}")

    # Save report JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'Li et al. 2022 - Biostimulant/Yield',
        'papers_processed': len(papers),
        'papers_matched': papers_matched,
        'gt_studies': len(gt_by_study),
        'gt_observations': total_gt,
        'matched_observations': len(all_matches),
        'metrics': {
            'pearson_r': round(r, 3) if not math.isnan(r) else None,
            'mae_pct': round(mae, 2),
            'median_ae_pct': round(median_ae, 2),
            'direction_accuracy': round(direction_acc, 3),
            'within_5pp': round(within_5, 3),
            'within_10pp': round(within_10, 3),
            'within_20pp': round(within_20, 3),
        },
        'overall_effect': {
            'gt_mean_pct': round(gt_overall, 2),
            'extracted_mean_pct': round(ext_overall, 2),
            'difference_pp': round(abs(gt_overall - ext_overall), 2),
        },
        'paper_stats': paper_stats,
    }

    report_path = output_dir / "validation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = parser.parse_args()
    validate(Path(args.results_dir))
