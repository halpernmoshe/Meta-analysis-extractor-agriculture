"""Deep validation analysis for paper Results section.

Analyzes error patterns, outlier impact, extraction method effectiveness,
and generates publication-ready summary statistics.
"""
import sys
import json
import csv
import math
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats as sp_stats

LOL_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\loladze_full_46_v2")
HUI_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\hui2023_v2")
HUI_MATCHES_CSV = HUI_DIR / "validation_hui2023_matches.csv"


def load_matches(csv_path):
    """Load validation matches CSV."""
    matches = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                matches.append({
                    'paper': row['paper'],
                    'ref': row.get('ref', ''),
                    'element': row['el'].upper(),
                    'our': float(row['our']),
                    'gt': float(row['gt']),
                    'err': float(row['err']),
                    'n_candidates': int(row.get('n_candidates', 1)),
                    'info': row.get('info', ''),
                })
            except (ValueError, KeyError):
                continue
    return matches


def load_paper_metadata(output_dir):
    """Load extraction metadata from consensus JSONs."""
    meta = {}
    for f in output_dir.glob("*_consensus.json"):
        pid = f.stem.replace("_consensus", "")
        with open(f) as fh:
            data = json.load(fh)
        recon = data.get('recon', {})
        meta[pid] = {
            'method': (recon.get('extraction_method') or 'TEXT').upper(),
            'difficulty': recon.get('estimated_difficulty', 'UNKNOWN'),
            'is_scanned': recon.get('is_scanned', False),
            'is_fig_only': recon.get('is_fig_only', False),
            'variance_type': recon.get('variance_type', ''),
            'claude_obs': data.get('claude_obs', 0),
            'kimi_obs': data.get('kimi_obs', 0),
            'gemini_obs': data.get('gemini_obs', 0),
            'consensus_obs': len(data.get('consensus_observations', [])),
            'tiebreaker_used': data.get('tiebreaker_used', False),
        }
    return meta


def analyze_error_patterns(matches, meta):
    """Analyze where and why errors occur."""
    print("=" * 70)
    print("DEEP VALIDATION ANALYSIS")
    print("=" * 70)

    errors = [m['err'] * 100 for m in matches]
    our_effects = [m['our'] for m in matches]
    gt_effects = [m['gt'] for m in matches]

    # Overall stats
    print(f"\n--- Overall Statistics (n={len(matches)}) ---")
    print(f"  Pearson r:     {sp_stats.pearsonr(our_effects, gt_effects)[0]:.3f}")
    print(f"  MAE:           {np.mean(errors):.1f}%")
    print(f"  Median AE:     {np.median(errors):.1f}%")
    print(f"  Within 5%:     {sum(1 for e in errors if e <= 5)/len(errors)*100:.0f}%")
    print(f"  Within 10%:    {sum(1 for e in errors if e <= 10)/len(errors)*100:.0f}%")
    print(f"  Within 20%:    {sum(1 for e in errors if e <= 20)/len(errors)*100:.0f}%")

    # Direction agreement
    dir_ok = sum(1 for m in matches if (m['our'] > 0) == (m['gt'] > 0) or abs(m['gt']) < 0.005)
    dir_total = sum(1 for m in matches if abs(m['gt']) >= 0.005)
    print(f"  Direction:     {dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)")

    # --- OUTLIER ANALYSIS ---
    print(f"\n--- Outlier Impact Analysis ---")

    # Identify outlier papers (MAE > 20%)
    paper_errors = defaultdict(list)
    for m in matches:
        paper_errors[m['paper']].append(m['err'] * 100)

    outlier_papers = {p for p, errs in paper_errors.items() if np.mean(errs) > 20}
    print(f"  Outlier papers (MAE > 20%): {len(outlier_papers)}")
    for p in sorted(outlier_papers):
        errs = paper_errors[p]
        print(f"    {p}: MAE={np.mean(errs):.1f}%, n={len(errs)}")

    # Without outliers
    clean = [m for m in matches if m['paper'] not in outlier_papers]
    clean_errors = [m['err'] * 100 for m in clean]
    clean_our = [m['our'] for m in clean]
    clean_gt = [m['gt'] for m in clean]

    if len(clean) > 2:
        print(f"\n  WITHOUT outlier papers (n={len(clean)}):")
        r_clean = sp_stats.pearsonr(clean_our, clean_gt)[0]
        print(f"    Pearson r:   {r_clean:.3f}")
        print(f"    MAE:         {np.mean(clean_errors):.1f}%")
        print(f"    Median AE:   {np.median(clean_errors):.1f}%")
        print(f"    Within 5%:   {sum(1 for e in clean_errors if e <= 5)/len(clean_errors)*100:.0f}%")
        print(f"    Within 10%:  {sum(1 for e in clean_errors if e <= 10)/len(clean_errors)*100:.0f}%")
        print(f"    Within 20%:  {sum(1 for e in clean_errors if e <= 20)/len(clean_errors)*100:.0f}%")

    # --- EFFECT SIZE MAGNITUDE ---
    print(f"\n--- Error by Effect Size Magnitude ---")
    bins = [(0, 0.05, "0-5%"), (0.05, 0.10, "5-10%"), (0.10, 0.20, "10-20%"), (0.20, 1.0, ">20%")]
    for lo, hi, label in bins:
        subset = [m for m in matches if lo <= abs(m['gt']) < hi]
        if subset:
            sub_err = [m['err'] * 100 for m in subset]
            sub_our = [m['our'] for m in subset]
            sub_gt = [m['gt'] for m in subset]
            r = sp_stats.pearsonr(sub_our, sub_gt)[0] if len(subset) > 2 else float('nan')
            print(f"  |GT effect| {label} (n={len(subset)}): MAE={np.mean(sub_err):.1f}%, r={r:.3f}")

    # --- BY ELEMENT ---
    print(f"\n--- Error by Element ---")
    el_data = defaultdict(list)
    for m in matches:
        el_data[m['element']].append(m)

    el_results = []
    for el, data in sorted(el_data.items()):
        errs = [d['err'] * 100 for d in data]
        our = [d['our'] for d in data]
        gt = [d['gt'] for d in data]
        r = sp_stats.pearsonr(our, gt)[0] if len(data) > 2 else float('nan')
        dir_ok = sum(1 for d in data if (d['our'] > 0) == (d['gt'] > 0) or abs(d['gt']) < 0.005)
        dir_tot = sum(1 for d in data if abs(d['gt']) >= 0.005)
        el_results.append((el, len(data), np.mean(errs), r, dir_ok, dir_tot))
        if len(data) >= 5:
            print(f"  {el:4s} (n={len(data):3d}): MAE={np.mean(errs):5.1f}%, r={r:6.3f}, "
                  f"dir={dir_ok}/{dir_tot} ({dir_ok/dir_tot*100:.0f}%)" if dir_tot > 0 else
                  f"  {el:4s} (n={len(data):3d}): MAE={np.mean(errs):5.1f}%, r={r:6.3f}")

    # --- BY EXTRACTION METHOD ---
    print(f"\n--- Error by Extraction Method ---")
    method_data = defaultdict(list)
    for m in matches:
        pid = m['paper']
        if pid in meta:
            method = meta[pid].get('method', 'UNKNOWN')
        else:
            method = 'UNKNOWN'
        method_data[method].append(m)

    for method, data in sorted(method_data.items()):
        errs = [d['err'] * 100 for d in data]
        our = [d['our'] for d in data]
        gt = [d['gt'] for d in data]
        r = sp_stats.pearsonr(our, gt)[0] if len(data) > 2 else float('nan')
        print(f"  {method:8s} (n={len(data):3d}): MAE={np.mean(errs):5.1f}%, r={r:.3f}")

    # --- BY DIFFICULTY ---
    print(f"\n--- Error by Paper Difficulty ---")
    diff_data = defaultdict(list)
    for m in matches:
        pid = m['paper']
        if pid in meta:
            diff = meta[pid].get('difficulty', 'UNKNOWN')
        else:
            diff = 'UNKNOWN'
        diff_data[diff].append(m)

    for diff, data in sorted(diff_data.items()):
        errs = [d['err'] * 100 for d in data]
        our = [d['our'] for d in data]
        gt = [d['gt'] for d in data]
        r = sp_stats.pearsonr(our, gt)[0] if len(data) > 2 else float('nan')
        print(f"  {diff:10s} (n={len(data):3d}): MAE={np.mean(errs):5.1f}%, r={r:.3f}")

    # --- BY N_CANDIDATES ---
    print(f"\n--- Error by GT Matching Ambiguity ---")
    for nc in [1, 2, 3]:
        subset = [m for m in matches if m['n_candidates'] == nc]
        if subset:
            sub_err = [m['err'] * 100 for m in subset]
            print(f"  {nc} candidate(s) (n={len(subset)}): MAE={np.mean(sub_err):.1f}%, "
                  f"within 10%: {sum(1 for e in sub_err if e <= 10)/len(subset)*100:.0f}%")
    multi = [m for m in matches if m['n_candidates'] >= 4]
    if multi:
        sub_err = [m['err'] * 100 for m in multi]
        print(f"  4+ candidates (n={len(multi)}): MAE={np.mean(sub_err):.1f}%, "
              f"within 10%: {sum(1 for e in sub_err if e <= 10)/len(multi)*100:.0f}%")

    # --- WORST INDIVIDUAL OBSERVATIONS ---
    print(f"\n--- Top 10 Worst Observations ---")
    worst = sorted(matches, key=lambda m: m['err'], reverse=True)[:10]
    for m in worst:
        print(f"  {m['paper']:25s} {m['element']:4s}: ours={m['our']:+.3f} vs GT={m['gt']:+.3f} "
              f"(err={m['err']*100:.1f}%) [{m['info']}]")

    # --- TIEBREAKER EFFECTIVENESS ---
    print(f"\n--- Tiebreaker Papers ---")
    tb_papers = [pid for pid, m in meta.items() if m.get('tiebreaker_used')]
    for pid in sorted(tb_papers):
        if pid in paper_errors:
            errs = paper_errors[pid]
            print(f"  {pid}: MAE={np.mean(errs):.1f}%, n_matched={len(errs)}")
        else:
            print(f"  {pid}: no GT matches")

    # --- COMBINED EFFECT SIZE ---
    print(f"\n--- Overall Effect Size Reproduction ---")
    gt_mean = np.mean(gt_effects)
    our_mean = np.mean(our_effects)
    print(f"  GT mean effect:   {gt_mean*100:+.2f}%")
    print(f"  Our mean effect:  {our_mean*100:+.2f}%")
    print(f"  Difference:       {(our_mean - gt_mean)*100:+.2f} pp")

    # By element for key elements
    key_els = ['N', 'P', 'K', 'CA', 'MG', 'FE', 'ZN', 'MN', 'CU', 'S', 'NA', 'B']
    print(f"\n  Per-element mean effects (n>=5):")
    for el in key_els:
        data = el_data.get(el, [])
        if len(data) >= 5:
            gt_m = np.mean([d['gt'] for d in data])
            our_m = np.mean([d['our'] for d in data])
            print(f"    {el:4s}: GT={gt_m*100:+6.2f}%, Ours={our_m*100:+6.2f}%, diff={abs(gt_m-our_m)*100:.2f}pp")

    return el_results


def analyze_paper_tiers(matches):
    """Classify papers into accuracy tiers for the paper."""
    print(f"\n{'='*70}")
    print("PAPER TIER CLASSIFICATION")
    print(f"{'='*70}")

    paper_stats = defaultdict(lambda: {'errors': [], 'matches': []})
    for m in matches:
        paper_stats[m['paper']]['errors'].append(m['err'] * 100)
        paper_stats[m['paper']]['matches'].append(m)

    tiers = {'Excellent': [], 'Good': [], 'Fair': [], 'Poor': []}

    for paper, stats in sorted(paper_stats.items()):
        mae = np.mean(stats['errors'])
        n = len(stats['errors'])
        within_10 = sum(1 for e in stats['errors'] if e <= 10) / n * 100

        if mae <= 5:
            tier = 'Excellent'
        elif mae <= 10:
            tier = 'Good'
        elif mae <= 20:
            tier = 'Fair'
        else:
            tier = 'Poor'

        tiers[tier].append((paper, mae, n, within_10))

    for tier_name, papers in tiers.items():
        print(f"\n  {tier_name} (n={len(papers)}):")
        for paper, mae, n, w10 in sorted(papers, key=lambda x: x[1]):
            print(f"    {paper:30s}: MAE={mae:5.1f}%, within 10%={w10:.0f}%, n_matched={n}")

    # Summary
    total = sum(len(p) for p in tiers.values())
    print(f"\n  TIER SUMMARY:")
    for tier_name in ['Excellent', 'Good', 'Fair', 'Poor']:
        n = len(tiers[tier_name])
        print(f"    {tier_name:10s}: {n:2d} papers ({n/total*100:.0f}%)")


def analyze_hui_if_available():
    """Analyze Hui 2023 validation if available."""
    hui_val = HUI_DIR / "validation_matches.csv"
    if not hui_val.exists():
        print(f"\n  Hui 2023 validation not yet available.")
        return

    matches = load_matches(hui_val)
    if not matches:
        print(f"\n  Hui 2023: no validation matches found.")
        return

    print(f"\n{'='*70}")
    print("HUI 2023 ZINC/WHEAT VALIDATION")
    print(f"{'='*70}")

    errors = [m['err'] * 100 for m in matches]
    our = [m['our'] for m in matches]
    gt = [m['gt'] for m in matches]

    print(f"  Matched observations: {len(matches)}")
    if len(matches) > 2:
        print(f"  Pearson r:     {sp_stats.pearsonr(our, gt)[0]:.3f}")
    print(f"  MAE:           {np.mean(errors):.1f}%")
    print(f"  Within 5%:     {sum(1 for e in errors if e <= 5)/len(errors)*100:.0f}%")
    print(f"  Within 10%:    {sum(1 for e in errors if e <= 10)/len(errors)*100:.0f}%")
    print(f"  Within 20%:    {sum(1 for e in errors if e <= 20)/len(errors)*100:.0f}%")

    dir_ok = sum(1 for m in matches if (m['our'] > 0) == (m['gt'] > 0) or abs(m['gt']) < 0.005)
    dir_total = sum(1 for m in matches if abs(m['gt']) >= 0.005)
    if dir_total > 0:
        print(f"  Direction:     {dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)")


def combined_summary(lol_matches, meta):
    """Generate combined summary for paper Table 1."""
    print(f"\n{'='*70}")
    print("TABLE 1: COMBINED VALIDATION RESULTS")
    print(f"{'='*70}")

    # Load Hui if available
    hui_val = HUI_DIR / "validation_matches.csv"
    hui_matches = load_matches(hui_val) if hui_val.exists() else []

    datasets = [("Loladze 2014", lol_matches), ("Hui 2023", hui_matches)]
    if hui_matches:
        datasets.append(("Combined", lol_matches + hui_matches))

    print(f"\n  {'Dataset':<15s} {'n':>5s} {'r':>6s} {'MAE':>6s} {'<5%':>5s} {'<10%':>5s} {'<20%':>5s} {'Dir':>5s}")
    print(f"  {'-'*15} {'-'*5} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")

    for name, matches in datasets:
        if not matches:
            continue
        errors = [m['err'] * 100 for m in matches]
        our = [m['our'] for m in matches]
        gt = [m['gt'] for m in matches]
        r = sp_stats.pearsonr(our, gt)[0] if len(matches) > 2 else float('nan')
        w5 = sum(1 for e in errors if e <= 5) / len(errors) * 100
        w10 = sum(1 for e in errors if e <= 10) / len(errors) * 100
        w20 = sum(1 for e in errors if e <= 20) / len(errors) * 100
        dir_ok = sum(1 for m in matches if (m['our'] > 0) == (m['gt'] > 0) or abs(m['gt']) < 0.005)
        dir_total = sum(1 for m in matches if abs(m['gt']) >= 0.005)
        dir_pct = dir_ok / dir_total * 100 if dir_total > 0 else 0
        print(f"  {name:<15s} {len(matches):5d} {r:6.3f} {np.mean(errors):5.1f}% {w5:4.0f}% {w10:4.0f}% {w20:4.0f}% {dir_pct:4.0f}%")


def consensus_mechanism_analysis(meta):
    """Analyze the consensus mechanism across all papers."""
    print(f"\n{'='*70}")
    print("CONSENSUS MECHANISM ANALYSIS")
    print(f"{'='*70}")

    total_claude = sum(m['claude_obs'] for m in meta.values())
    total_kimi = sum(m['kimi_obs'] for m in meta.values())
    total_gemini = sum(m['gemini_obs'] for m in meta.values())
    total_consensus = sum(m['consensus_obs'] for m in meta.values())

    n_tb = sum(1 for m in meta.values() if m.get('tiebreaker_used'))
    n_hybrid = sum(1 for m in meta.values() if m.get('method') == 'HYBRID')
    n_text = sum(1 for m in meta.values() if m.get('method') == 'TEXT')

    print(f"  Papers: {len(meta)} total ({n_text} TEXT, {n_hybrid} HYBRID)")
    print(f"  Tiebreaker used: {n_tb} papers")
    print(f"\n  Model output volumes:")
    print(f"    Claude: {total_claude} obs")
    print(f"    Kimi:   {total_kimi} obs")
    print(f"    Gemini: {total_gemini} obs (vision+tiebreaker)")
    print(f"    Consensus: {total_consensus} obs")

    if total_claude > 0:
        max_single = max(total_claude, total_kimi)
        print(f"\n  Consensus gain over best single model:")
        print(f"    Best single: {max_single} obs")
        print(f"    Consensus:   {total_consensus} obs ({(total_consensus/max_single-1)*100:+.0f}%)")


def main():
    # Load Loladze data
    lol_csv = LOL_DIR / "validation_matches.csv"
    if not lol_csv.exists():
        print("ERROR: validation_matches.csv not found. Run validate_full_46.py first.")
        return

    matches = load_matches(lol_csv)
    meta = load_paper_metadata(LOL_DIR)

    print(f"Loaded {len(matches)} validation matches from {len(set(m['paper'] for m in matches))} papers")
    print(f"Loaded metadata for {len(meta)} papers")

    # Run all analyses
    analyze_error_patterns(matches, meta)
    analyze_paper_tiers(matches)
    analyze_hui_if_available()
    combined_summary(matches, meta)
    consensus_mechanism_analysis(meta)

    # Also analyze Hui metadata
    hui_meta = load_paper_metadata(HUI_DIR)
    if hui_meta:
        print(f"\n--- Hui 2023 Extraction Summary ---")
        total_obs = sum(m['consensus_obs'] for m in hui_meta.values())
        print(f"  Papers: {len(hui_meta)}")
        print(f"  Total observations: {total_obs}")


if __name__ == "__main__":
    main()
