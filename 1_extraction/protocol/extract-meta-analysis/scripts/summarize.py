#!/usr/bin/env python3
"""
Summarize meta-analysis extraction results.

Computes descriptive statistics for extracted observations without requiring
ground truth. Useful for quality checking before downstream meta-analysis.

Requirements: numpy (pip install numpy)

Usage:
    python summarize.py <extraction_dir> [--output summary.json]
    python summarize.py output/my_extraction/ --output output/summary.json
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not installed. Install with: pip install numpy")
    print("         Some statistics will be approximate.")


def safe_mean(values):
    if HAS_NUMPY:
        return float(np.mean(values))
    return sum(values) / len(values)


def safe_median(values):
    if HAS_NUMPY:
        return float(np.median(values))
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def safe_percentile(values, pct):
    if HAS_NUMPY:
        return float(np.percentile(values, pct))
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])


def load_extractions(directory):
    """Load all extraction JSON files from a directory."""
    directory = Path(directory)
    papers = []

    for f in sorted(directory.glob("*.json")):
        try:
            with open(f, encoding='utf-8') as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"  WARNING: Could not parse {f.name}: {e}")
            continue

        paper_id = data.get('paper_id', f.stem)
        obs_list = data.get('observations', data.get('consensus_observations', []))

        if not obs_list:
            continue

        papers.append({
            'paper_id': paper_id,
            'file': str(f),
            'title': data.get('title', ''),
            'year': data.get('year'),
            'observations': obs_list,
        })

    return papers


def summarize(papers):
    """Compute summary statistics across all papers."""
    total_obs = 0
    effects = []
    outcomes = defaultdict(int)
    variance_types = defaultdict(int)
    has_variance = 0
    has_n = 0
    has_both_means = 0
    has_moderators = 0
    n_values = []
    paper_obs_counts = {}
    units = defaultdict(int)
    data_sources = defaultdict(int)

    # Per-outcome effect sizes
    outcome_effects = defaultdict(list)

    for paper in papers:
        obs_list = paper['observations']
        paper_obs_counts[paper['paper_id']] = len(obs_list)

        for obs in obs_list:
            total_obs += 1

            # Outcome
            outcome = obs.get('outcome', obs.get('element', 'unknown'))
            outcomes[outcome] += 1

            # Effect size
            eff = obs.get('effect_pct')
            if eff is None:
                tm = obs.get('treatment_mean')
                cm = obs.get('control_mean')
                if tm is not None and cm is not None and cm != 0:
                    eff = (tm - cm) / abs(cm) * 100
            if eff is not None:
                effects.append(eff)
                outcome_effects[outcome].append(eff)

            # Means
            if obs.get('treatment_mean') is not None and obs.get('control_mean') is not None:
                has_both_means += 1

            # Variance
            vt = obs.get('variance_type')
            if vt:
                variance_types[vt] += 1
            if obs.get('variance_value') is not None:
                has_variance += 1

            # Sample size
            n = obs.get('n')
            if n is not None:
                has_n += 1
                n_values.append(n)

            # Moderators
            mods = obs.get('moderators', {})
            if mods and any(v is not None for v in mods.values()):
                has_moderators += 1

            # Units
            unit = obs.get('unit')
            if unit:
                units[unit] += 1

            # Data source
            src = obs.get('data_source', '')
            if src:
                data_sources[src] += 1

    # Build report
    report = {
        'summary': {
            'total_papers': len(papers),
            'total_observations': total_obs,
            'observations_with_effect': len(effects),
            'observations_with_both_means': has_both_means,
        },
        'coverage': {
            'variance_value': {
                'count': has_variance,
                'pct': round(has_variance / max(total_obs, 1) * 100, 1),
            },
            'sample_size': {
                'count': has_n,
                'pct': round(has_n / max(total_obs, 1) * 100, 1),
            },
            'moderators': {
                'count': has_moderators,
                'pct': round(has_moderators / max(total_obs, 1) * 100, 1),
            },
            'meta_analysis_ready': {
                'description': 'Observations with both means + variance + n',
                'count': sum(1 for p in papers for o in p['observations']
                             if o.get('treatment_mean') is not None
                             and o.get('control_mean') is not None
                             and o.get('variance_value') is not None
                             and o.get('n') is not None),
                'pct': None,  # filled below
            }
        },
        'outcome_distribution': dict(sorted(outcomes.items(), key=lambda x: -x[1])),
        'variance_types': dict(sorted(variance_types.items(), key=lambda x: -x[1])),
        'units': dict(sorted(units.items(), key=lambda x: -x[1])),
    }

    ready = report['coverage']['meta_analysis_ready']['count']
    report['coverage']['meta_analysis_ready']['pct'] = round(ready / max(total_obs, 1) * 100, 1)

    # Effect size statistics
    if effects:
        report['effect_size'] = {
            'mean': round(safe_mean(effects), 2),
            'median': round(safe_median(effects), 2),
            'sd': round(float(np.std(effects, ddof=1)), 2) if HAS_NUMPY and len(effects) > 1 else None,
            'min': round(min(effects), 2),
            'max': round(max(effects), 2),
            'q25': round(safe_percentile(effects, 25), 2),
            'q75': round(safe_percentile(effects, 75), 2),
            'pct_negative': round(sum(1 for e in effects if e < 0) / len(effects) * 100, 1),
            'pct_positive': round(sum(1 for e in effects if e > 0) / len(effects) * 100, 1),
        }

        # Per-outcome effect sizes (top outcomes)
        report['per_outcome_effects'] = {}
        for outcome in sorted(outcome_effects.keys(), key=lambda k: -len(outcome_effects[k])):
            oe = outcome_effects[outcome]
            if len(oe) >= 2:
                report['per_outcome_effects'][outcome] = {
                    'n': len(oe),
                    'mean': round(safe_mean(oe), 2),
                    'median': round(safe_median(oe), 2),
                    'min': round(min(oe), 2),
                    'max': round(max(oe), 2),
                }

    # Sample size statistics
    if n_values:
        report['sample_size'] = {
            'mean': round(safe_mean(n_values), 1),
            'median': round(safe_median(n_values), 1),
            'min': min(n_values),
            'max': max(n_values),
            'mode': max(set(n_values), key=n_values.count),
        }

    # Per-paper observation counts
    report['per_paper'] = {
        'obs_counts': dict(sorted(paper_obs_counts.items(), key=lambda x: -x[1])),
        'mean_obs_per_paper': round(safe_mean(list(paper_obs_counts.values())), 1),
        'median_obs_per_paper': round(safe_median(list(paper_obs_counts.values())), 1),
        'min_obs': min(paper_obs_counts.values()),
        'max_obs': max(paper_obs_counts.values()),
    }

    # Flags and potential issues
    flags = []

    # Large effects
    large = [e for e in effects if abs(e) > 100]
    if large:
        flags.append(f"{len(large)} observations with |effect| > 100% -- check for unit errors or T/C swaps")

    # Very small effects in unexpected direction
    if report.get('effect_size'):
        neg_pct = report['effect_size']['pct_negative']
        pos_pct = report['effect_size']['pct_positive']
        if neg_pct > 80:
            flags.append(f"Strong negative skew ({neg_pct}% negative effects)")
        elif pos_pct > 80:
            flags.append(f"Strong positive skew ({pos_pct}% positive effects)")

    # Low coverage
    if report['coverage']['variance_value']['pct'] < 30:
        flags.append(f"Very low variance coverage ({report['coverage']['variance_value']['pct']}%)")
    elif report['coverage']['variance_value']['pct'] < 50:
        flags.append(f"Low variance coverage ({report['coverage']['variance_value']['pct']}%)")

    if report['coverage']['sample_size']['pct'] < 50:
        flags.append(f"Low sample size coverage ({report['coverage']['sample_size']['pct']}%)")

    # Papers with very few observations
    papers_with_1 = sum(1 for c in paper_obs_counts.values() if c == 1)
    if papers_with_1 > 0:
        flags.append(f"{papers_with_1} papers with only 1 observation (may need re-extraction)")

    # Papers with zero observations already filtered out, but check for empty files
    report['flags'] = flags

    return report


def print_summary(report):
    """Print formatted summary to stdout."""
    s = report['summary']
    print(f"\n{'='*70}")
    print("META-ANALYSIS EXTRACTION SUMMARY")
    print(f"{'='*70}")

    print(f"\nPapers processed: {s['total_papers']}")
    print(f"Total observations: {s['total_observations']}")
    print(f"With both means: {s['observations_with_both_means']}")
    print(f"With effect size: {s['observations_with_effect']}")

    cov = report['coverage']
    print(f"\nCOVERAGE:")
    print(f"  Variance: {cov['variance_value']['count']} ({cov['variance_value']['pct']}%)")
    print(f"  Sample size: {cov['sample_size']['count']} ({cov['sample_size']['pct']}%)")
    print(f"  Moderators: {cov['moderators']['count']} ({cov['moderators']['pct']}%)")
    mar = cov['meta_analysis_ready']
    print(f"  Meta-analysis ready (means+var+n): {mar['count']} ({mar['pct']}%)")

    if report.get('effect_size'):
        es = report['effect_size']
        print(f"\nEFFECT SIZE (%):")
        print(f"  Mean: {es['mean']}, Median: {es['median']}")
        print(f"  Range: [{es['min']}, {es['max']}]")
        print(f"  IQR: [{es['q25']}, {es['q75']}]")
        print(f"  Negative: {es['pct_negative']}%, Positive: {es['pct_positive']}%")

    print(f"\nOUTCOME DISTRIBUTION:")
    for outcome, count in list(report['outcome_distribution'].items())[:15]:
        print(f"  {outcome}: {count}")
    if len(report['outcome_distribution']) > 15:
        print(f"  ... and {len(report['outcome_distribution']) - 15} more")

    if report.get('variance_types'):
        print(f"\nVARIANCE TYPES:")
        for vt, count in report['variance_types'].items():
            print(f"  {vt}: {count}")

    if report.get('sample_size'):
        ns = report['sample_size']
        print(f"\nSAMPLE SIZES:")
        print(f"  Mean: {ns['mean']}, Median: {ns['median']}, Mode: {ns['mode']}")
        print(f"  Range: [{ns['min']}, {ns['max']}]")

    pp = report['per_paper']
    print(f"\nPER-PAPER:")
    print(f"  Mean obs/paper: {pp['mean_obs_per_paper']}")
    print(f"  Median obs/paper: {pp['median_obs_per_paper']}")
    print(f"  Range: [{pp['min_obs']}, {pp['max_obs']}]")

    if report.get('flags'):
        print(f"\nFLAGS:")
        for flag in report['flags']:
            print(f"  - {flag}")

    print(f"\n{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize meta-analysis extraction results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python summarize.py output/my_extraction/
    python summarize.py output/my_extraction/ --output summary.json
        """
    )
    parser.add_argument('extraction_dir', help='Directory containing extraction JSON files')
    parser.add_argument('--output', '-o', default=None, help='Output path for summary JSON')

    args = parser.parse_args()

    extraction_dir = Path(args.extraction_dir)
    if not extraction_dir.is_dir():
        print(f"ERROR: {extraction_dir} is not a directory")
        sys.exit(1)

    print(f"Loading extractions from: {extraction_dir}")
    papers = load_extractions(extraction_dir)
    print(f"  Loaded {len(papers)} papers")

    if not papers:
        print("ERROR: No extraction files found.")
        sys.exit(1)

    report = summarize(papers)
    print_summary(report)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
