"""
Programmatic GT Harmonization Classifier for Li 2022

Replaces LLM-based classification of matched observations with purely
algorithmic classification based on metadata, scale ratios, and statistical
patterns. This breaks the "LLM-confirms-LLM" circularity.

Criteria are based on observable data properties, not LLM judgment.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


# ── Clean scale ratio targets (unit conversions) ──────────────────────────
CLEAN_SCALES = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
CLEAN_TOLERANCE = 0.08  # 8% relative tolerance


def is_clean_scale(ratio, tolerance=CLEAN_TOLERANCE):
    """Is this scale ratio close to a clean power-of-10 conversion?"""
    return any(abs(ratio - t) / max(t, 0.001) < tolerance for t in CLEAN_SCALES)


def classify_paper_programmatic(paper_id, obs_list):
    """
    Classify a paper's matches using ONLY algorithmic criteria.

    Returns: (classification, confidence, evidence_dict)

    Classification logic (applied in order):

    1. EXACT_MATCH: If ≥30% of obs have |effect_diff| < 0.01pp
       → Values are identical after scale normalization = verified correct

    2. LOW_ERROR: If paper MAE < 2pp AND direction agreement ≥ 95%
       → Extraction closely matches GT, minor rounding/reading differences

    3. DIRECTION_FAILURE: If direction agreement < 85%
       → Systematic mismatch, likely aggregation-level discordance

    4. SCALE_ANOMALY: If <20% of obs have clean scale ratios AND scale CV > 50%
       → Inconsistent unit matching suggests wrong data matched

    5. MODERATE_ERROR: If MAE 2-5pp AND direction agreement ≥ 90%
       → Partial match, some data alignment issues

    6. HIGH_ERROR: Everything else
       → Significant discrepancies, likely fundamental mismatch
    """
    n = len(obs_list)
    diffs = [abs(float(r['effect_diff_pp'])) for r in obs_list]
    mae = sum(diffs) / n
    median_ae = sorted(diffs)[n // 2]

    # Zero-error fraction
    zero_frac = sum(1 for d in diffs if d < 0.01) / n

    # Direction agreement
    dir_matches = sum(1 for r in obs_list if r['direction_match'] == 'True')
    dir_frac = dir_matches / n

    # Scale ratio analysis
    scales = [float(r['scale_ratio']) for r in obs_list]
    clean_frac = sum(1 for s in scales if is_clean_scale(s)) / n
    scale_mean = statistics.mean(scales) if scales else 1.0
    scale_cv = (statistics.stdev(scales) / scale_mean * 100) if n > 1 and scale_mean > 0 else 0

    # Within-threshold fractions
    within_1pp = sum(1 for d in diffs if d < 1.0) / n
    within_2pp = sum(1 for d in diffs if d < 2.0) / n
    within_5pp = sum(1 for d in diffs if d < 5.0) / n

    evidence = {
        'n_obs': n,
        'mae_pp': round(mae, 2),
        'median_ae_pp': round(median_ae, 2),
        'zero_error_frac': round(zero_frac, 3),
        'direction_agreement': round(dir_frac, 3),
        'clean_scale_frac': round(clean_frac, 3),
        'scale_cv_pct': round(scale_cv, 1),
        'within_1pp': round(within_1pp, 3),
        'within_2pp': round(within_2pp, 3),
        'within_5pp': round(within_5pp, 3),
    }

    # ── Decision tree (purely algorithmic) ──

    # Rule 1: Exact matches prove correct matching
    if zero_frac >= 0.30:
        return 'verified_correct', 'high', evidence

    # Rule 2: Very low error with good direction
    if mae < 2.0 and dir_frac >= 0.95:
        return 'likely_correct', 'medium', evidence

    # Rule 3: Direction failure = aggregation mismatch
    if dir_frac < 0.85:
        return 'aggregation_discordance', 'high', evidence

    # Rule 4: Messy scale ratios = wrong data matched
    if clean_frac < 0.20 and scale_cv > 50:
        # But check if effects are still close despite messy scales
        if mae < 2.0:
            return 'likely_correct', 'low', evidence
        return 'scale_anomaly', 'medium', evidence

    # Rule 5: Moderate error with OK direction
    if mae < 5.0 and dir_frac >= 0.90:
        return 'moderate_discrepancy', 'medium', evidence

    # Rule 6: Everything else
    return 'high_discrepancy', 'low', evidence


def map_to_confidence_tier(prog_class):
    """Map programmatic classification to a simple confidence tier."""
    tier_map = {
        'verified_correct': 'high',
        'likely_correct': 'high',
        'moderate_discrepancy': 'medium',
        'aggregation_discordance': 'low',
        'scale_anomaly': 'low',
        'high_discrepancy': 'low',
    }
    return tier_map.get(prog_class, 'low')


def main():
    base = Path(__file__).parent
    csv_path = base / 'output' / 'li2022_combined' / 'validation_matches_improved.csv'

    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} observations from {csv_path.name}")
    print()

    # ── Group by paper ──
    papers = defaultdict(list)
    for r in rows:
        papers[r['paper_id']].append(r)

    # ── Classify each paper programmatically ──
    results = {}
    for pid, obs in sorted(papers.items()):
        prog_class, confidence, evidence = classify_paper_programmatic(pid, obs)
        llm_class = obs[0]['classification']  # Original LLM classification
        results[pid] = {
            'programmatic': prog_class,
            'llm_original': llm_class,
            'confidence': confidence,
            'tier': map_to_confidence_tier(prog_class),
            'evidence': evidence,
        }

    # ── Print comparison ──
    print(f"{'Paper':<47} {'LLM Class':<20} {'Programmatic':<25} {'Tier':<6} {'MAE':>5} {'Zero%':>5} {'Dir%':>5}")
    print('=' * 130)

    agree = 0
    for pid, res in sorted(results.items()):
        short_pid = pid[:45]
        ev = res['evidence']

        # Check semantic agreement (not exact label match)
        llm_is_good = res['llm_original'] in ('correct',)
        prog_is_good = res['tier'] == 'high'
        match = '✓' if llm_is_good == prog_is_good else '✗'
        if llm_is_good == prog_is_good:
            agree += 1

        print(f"{short_pid:<47} {res['llm_original']:<20} {res['programmatic']:<25} {res['tier']:<6} "
              f"{ev['mae_pp']:>5.2f} {ev['zero_error_frac']*100:>5.0f} {ev['direction_agreement']*100:>5.0f} {match}")

    print()
    print(f"Semantic agreement (good/not-good): {agree}/{len(results)} papers ({agree/len(results)*100:.0f}%)")

    # ── Summary statistics by tier ──
    print()
    print("=" * 80)
    print("STATISTICS BY PROGRAMMATIC CONFIDENCE TIER")
    print("=" * 80)

    for tier in ['high', 'medium', 'low']:
        tier_pids = [pid for pid, res in results.items() if res['tier'] == tier]
        tier_obs = [r for r in rows if r['paper_id'] in tier_pids]
        if not tier_obs:
            continue

        n_obs = len(tier_obs)
        n_papers = len(tier_pids)
        diffs = [abs(float(r['effect_diff_pp'])) for r in tier_obs]
        mae = sum(diffs) / len(diffs)

        gt_effects = [float(r['gt_effect_pct']) for r in tier_obs]
        ext_effects = [float(r['ext_effect_pct']) for r in tier_obs]

        # Correlation
        if len(gt_effects) > 2:
            mean_gt = sum(gt_effects) / len(gt_effects)
            mean_ext = sum(ext_effects) / len(ext_effects)
            cov = sum((g - mean_gt) * (e - mean_ext) for g, e in zip(gt_effects, ext_effects)) / len(gt_effects)
            std_gt = math.sqrt(sum((g - mean_gt)**2 for g in gt_effects) / len(gt_effects))
            std_ext = math.sqrt(sum((e - mean_ext)**2 for e in ext_effects) / len(ext_effects))
            r = cov / (std_gt * std_ext) if std_gt > 0 and std_ext > 0 else 0
        else:
            r = float('nan')

        dir_ok = sum(1 for x in tier_obs if x['direction_match'] == 'True') / n_obs * 100
        zero_frac = sum(1 for d in diffs if d < 0.01) / n_obs * 100

        print(f"\n{tier.upper()} confidence: {n_obs} obs across {n_papers} papers")
        print(f"  r = {r:.3f}")
        print(f"  MAE = {mae:.2f} pp")
        print(f"  Direction agreement = {dir_ok:.1f}%")
        print(f"  Zero-error fraction = {zero_frac:.1f}%")

        # LLM classification breakdown within this tier
        llm_counts = defaultdict(int)
        for pid in tier_pids:
            llm_counts[results[pid]['llm_original']] += 1
        print(f"  LLM classes in this tier: {dict(llm_counts)}")

    # ── Compare with LLM's "correct" subset ──
    print()
    print("=" * 80)
    print("COMPARISON: PROGRAMMATIC HIGH-CONFIDENCE vs LLM 'CORRECT'")
    print("=" * 80)

    llm_correct_obs = [r for r in rows if r['classification'] == 'correct']
    prog_high_pids = [pid for pid, res in results.items() if res['tier'] == 'high']
    prog_high_obs = [r for r in rows if r['paper_id'] in prog_high_pids]

    for label, subset in [("LLM 'correct'", llm_correct_obs), ("Programmatic 'high'", prog_high_obs)]:
        n = len(subset)
        diffs = [abs(float(r['effect_diff_pp'])) for r in subset]
        mae = sum(diffs) / n
        gt = [float(r['gt_effect_pct']) for r in subset]
        ext = [float(r['ext_effect_pct']) for r in subset]
        mean_gt = sum(gt) / n
        mean_ext = sum(ext) / n
        cov = sum((g - mean_gt) * (e - mean_ext) for g, e in zip(gt, ext)) / n
        std_gt = math.sqrt(sum((g - mean_gt)**2 for g in gt) / n)
        std_ext = math.sqrt(sum((e - mean_ext)**2 for e in ext) / n)
        r = cov / (std_gt * std_ext) if std_gt > 0 and std_ext > 0 else 0
        dir_ok = sum(1 for x in subset if x['direction_match'] == 'True') / n * 100

        n_papers = len(set(r['paper_id'] for r in subset))
        print(f"\n{label}: {n} obs, {n_papers} papers")
        print(f"  r = {r:.3f}, MAE = {mae:.2f} pp, Direction = {dir_ok:.1f}%")

    # ── Overlap analysis ──
    llm_correct_papers = set(r['paper_id'] for r in llm_correct_obs)
    prog_high_papers = set(prog_high_pids)

    both = llm_correct_papers & prog_high_papers
    only_llm = llm_correct_papers - prog_high_papers
    only_prog = prog_high_papers - llm_correct_papers

    print(f"\nPaper-level overlap:")
    print(f"  Both agree 'good': {len(both)} papers")
    print(f"  Only LLM says correct: {len(only_llm)} papers")
    if only_llm:
        for p in only_llm:
            ev = results[p]['evidence']
            print(f"    {p[:45]} → prog={results[p]['programmatic']} (MAE={ev['mae_pp']}, zero={ev['zero_error_frac']*100:.0f}%)")
    print(f"  Only programmatic says high: {len(only_prog)} papers")
    if only_prog:
        for p in only_prog:
            ev = results[p]['evidence']
            print(f"    {p[:45]} → llm={results[p]['llm_original']} (MAE={ev['mae_pp']}, zero={ev['zero_error_frac']*100:.0f}%)")

    # ── Save results ──
    out_path = base / 'output' / 'li2022_combined' / 'programmatic_classification.json'
    output = {
        'method': 'Programmatic classification using effect_diff, direction, scale_ratio patterns',
        'criteria': {
            'verified_correct': 'Zero-error fraction >= 30% (exact value matches prove correct alignment)',
            'likely_correct': 'MAE < 2pp AND direction >= 95%',
            'aggregation_discordance': 'Direction agreement < 85%',
            'scale_anomaly': 'Clean scale fraction < 20% AND scale CV > 50% AND MAE >= 2pp',
            'moderate_discrepancy': 'MAE 2-5pp AND direction >= 90%',
            'high_discrepancy': 'All other cases',
        },
        'no_llm_judgment_used': True,
        'papers': {pid: res for pid, res in results.items()},
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")

    # ── Final summary for paper ──
    print()
    print("=" * 80)
    print("BOTTOM LINE FOR THE PAPER")
    print("=" * 80)
    print()
    print("The programmatic classifier uses THREE observable signals:")
    print("  1. Zero-error fraction: exact value matches (after scale normalization)")
    print("  2. Direction agreement: sign concordance between GT and extracted effects")
    print("  3. MAE threshold: magnitude of effect differences")
    print()
    print("These criteria require NO LLM judgment — they are computed directly from")
    print("the matched numerical data. Any researcher can reproduce the classification")
    print("by running this script on the validation CSV.")


if __name__ == '__main__':
    main()
