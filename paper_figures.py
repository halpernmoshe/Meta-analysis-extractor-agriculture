"""Generate publication-quality figures for the meta-analysis extraction paper.

Reads validation data from:
  - output/loladze_full_46_v2/validation_matches.csv
  - output/loladze_full_46_v2/validation_report_full.json
  - output/hui2023_v2/validation_hui2023.json
  - output/hui2023_v2/validation_hui2023_matches.csv

Generates figures in output/paper_figures/
"""
import sys, json, csv, math
from pathlib import Path
from collections import defaultdict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy import stats as sp_stats

BASE = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
LOL_DIR = BASE / "output" / "loladze_combined_51"
HUI_DIR = BASE / "output" / "hui2023_full_35"
LI_DIR = BASE / "output" / "li2022_combined"
OUT_DIR = BASE / "output" / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

# Consistent styling
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Element colors for consistent visualization
ELEMENT_COLORS = {
    'N': '#1f77b4', 'P': '#ff7f0e', 'K': '#2ca02c', 'CA': '#d62728',
    'MG': '#9467bd', 'FE': '#8c564b', 'ZN': '#e377c2', 'MN': '#7f7f7f',
    'CU': '#bcbd22', 'S': '#17becf', 'B': '#aec7e8', 'NA': '#ffbb78',
    'AL': '#98df8a', 'CO': '#ff9896', 'NI': '#c5b0d5', 'SI': '#c49c94',
}

# Tier colors
TIER_COLORS = {
    'Excellent': '#2ca02c',
    'Good': '#1f77b4',
    'Fair': '#ff7f0e',
    'Poor': '#d62728',
}


def load_loladze_matches():
    """Load Loladze validation matches CSV."""
    matches = []
    csv_path = LOL_DIR / "validation_matches.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                matches.append({
                    'paper': row['paper'],
                    'ref': row['ref'],
                    'element': row['el'].upper(),
                    'our_effect': float(row['our']),
                    'gt_effect': float(row['gt']),
                    'abs_error': float(row['err']),
                    'info': row.get('info', ''),
                    'n_candidates': int(row.get('n_candidates', 0)),
                })
            except (ValueError, KeyError):
                continue
    return matches


def load_loladze_report():
    """Load Loladze validation report JSON."""
    with open(LOL_DIR / "validation_report_full.json") as f:
        return json.load(f)


def load_hui_matches():
    """Load Hui 2023 validation matches CSV (supports both old and new format)."""
    matches = []
    # Try new format first (expanded 34-paper run)
    csv_path = HUI_DIR / "validation_matches.csv"
    if not csv_path.exists():
        csv_path = HUI_DIR / "validation_hui2023_matches.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Support both old (our_ctrl/our_treat) and new (ext_ctrl/ext_treat) formats
                our_ctrl = float(row.get('ext_ctrl', row.get('our_ctrl', 0)))
                our_treat = float(row.get('ext_treat', row.get('our_treat', 0)))
                gt_ctrl = float(row.get('gt_ctrl', 0))
                gt_treat = float(row.get('gt_treat', 0))
                # Compute lnRR if not present
                our_lnrr = None
                gt_lnrr = None
                if our_ctrl > 0 and our_treat > 0:
                    our_lnrr = math.log(our_treat / our_ctrl)
                if gt_ctrl > 0 and gt_treat > 0:
                    gt_lnrr = math.log(gt_treat / gt_ctrl)
                matches.append({
                    'our_ctrl': our_ctrl,
                    'our_treat': our_treat,
                    'our_lnrr': our_lnrr,
                    'gt_ctrl': gt_ctrl,
                    'gt_treat': gt_treat,
                    'gt_lnrr': gt_lnrr,
                    'gt_pub': row.get('gt_pub', row.get('tissue', '')),
                    'match_qual': float(row.get('match_qual', 0)),
                    'ext_effect': float(row.get('ext_effect', 0)),
                    'gt_effect': float(row.get('gt_effect', 0)),
                })
            except (ValueError, KeyError):
                continue
    return matches


def load_hui_report():
    """Load Hui validation report JSON."""
    rpath = HUI_DIR / "validation_report.json"
    if not rpath.exists():
        rpath = HUI_DIR / "validation_hui2023.json"
    if not rpath.exists():
        return None
    with open(rpath) as f:
        return json.load(f)


def load_li2022_matches():
    """Load Li 2022 validation matches CSV."""
    matches = []
    csv_path = LI_DIR / "validation_matches.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                matches.append({
                    'paper_id': row['paper_id'],
                    'gt_effect': float(row['gt_effect_pct']),
                    'our_effect': float(row['ext_effect_pct']),
                    'direction_match': row['direction_match'] == 'True',
                    'crop': row.get('crop', ''),
                    'category': row.get('category', ''),
                })
            except (ValueError, KeyError):
                continue
    return matches


# ============================================================
# FIGURE 2: Scatter plot - Extracted vs GT effect sizes (Loladze)
# ============================================================
def fig2_scatter_loladze(matches):
    """Scatter plot of extracted vs GT effect sizes, colored by element."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Group by element
    by_element = defaultdict(list)
    for m in matches:
        by_element[m['element']].append(m)

    # Sort elements by count for legend ordering
    sorted_elements = sorted(by_element.keys(), key=lambda e: len(by_element[e]), reverse=True)

    for el in sorted_elements:
        el_matches = by_element[el]
        x = [m['gt_effect'] * 100 for m in el_matches]
        y = [m['our_effect'] * 100 for m in el_matches]
        color = ELEMENT_COLORS.get(el, '#333333')
        ax.scatter(x, y, c=color, alpha=0.6, s=30, label=f"{el} (n={len(el_matches)})",
                   edgecolors='white', linewidth=0.3)

    # Reference line
    lims = [-100, 150]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect agreement')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    # Compute overall stats
    all_our = [m['our_effect'] for m in matches]
    all_gt = [m['gt_effect'] for m in matches]
    r, p = sp_stats.pearsonr(all_gt, all_our)

    ax.set_xlabel('Ground Truth Effect Size (%)')
    ax.set_ylabel('Extracted Effect Size (%)')
    ax.set_title(f'Loladze Dataset: Extracted vs Ground Truth Effect Sizes\n'
                 f'(n={len(matches)}, r={r:.3f}, p<0.001)')
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.8)
    ax.set_xlim(-100, 150)
    ax.set_ylim(-100, 150)

    fig.tight_layout()
    path = OUT_DIR / "fig2_scatter_loladze.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")
    return r


# ============================================================
# FIGURE 3: Per-paper MAE bar chart
# ============================================================
def fig3_paper_mae(report):
    """Bar chart of per-paper MAE, sorted and colored by tier."""
    papers = []
    for p in report['per_paper']:
        mae = p.get('mae')
        if mae is not None and not (isinstance(mae, float) and math.isnan(mae)):
            papers.append({
                'name': p['paper_id'].split('_', 1)[1] if '_' in p['paper_id'] else p['paper_id'],
                'mae': mae,
                'ref': p.get('ref', ''),
                'matched': p.get('matched', 0),
            })

    # Sort by MAE
    papers.sort(key=lambda x: x['mae'])

    fig, ax = plt.subplots(figsize=(12, 7))

    names = [p['name'] for p in papers]
    maes = [p['mae'] for p in papers]

    # Color by tier
    colors = []
    for mae in maes:
        if mae < 5:
            colors.append(TIER_COLORS['Excellent'])
        elif mae < 10:
            colors.append(TIER_COLORS['Good'])
        elif mae < 20:
            colors.append(TIER_COLORS['Fair'])
        else:
            colors.append(TIER_COLORS['Poor'])

    bars = ax.barh(range(len(papers)), maes, color=colors, edgecolor='white', linewidth=0.5)

    # Add tier threshold lines
    ax.axvline(5, color=TIER_COLORS['Excellent'], linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(10, color=TIER_COLORS['Good'], linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(20, color=TIER_COLORS['Fair'], linestyle=':', alpha=0.5, linewidth=1)

    ax.set_yticks(range(len(papers)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Mean Absolute Error (%)')
    ax.set_title('Per-Paper Extraction Accuracy (Loladze Dataset)')
    ax.invert_yaxis()

    # Legend for tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS['Excellent'], label=f'Excellent (<5%, n={sum(1 for m in maes if m < 5)})'),
        Patch(facecolor=TIER_COLORS['Good'], label=f'Good (5-10%, n={sum(1 for m in maes if 5 <= m < 10)})'),
        Patch(facecolor=TIER_COLORS['Fair'], label=f'Fair (10-20%, n={sum(1 for m in maes if 10 <= m < 20)})'),
        Patch(facecolor=TIER_COLORS['Poor'], label=f'Poor (>20%, n={sum(1 for m in maes if m >= 20)})'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "fig3_paper_mae.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# FIGURE 4: Element-level mean effect comparison
# ============================================================
def fig4_element_effects(matches):
    """Element-level comparison of mean extracted vs GT effects."""
    by_element = defaultdict(lambda: {'our': [], 'gt': []})
    for m in matches:
        el = m['element']
        by_element[el]['our'].append(m['our_effect'] * 100)
        by_element[el]['gt'].append(m['gt_effect'] * 100)

    # Filter elements with >= 5 matches
    elements = {el: d for el, d in by_element.items() if len(d['our']) >= 5}
    sorted_els = sorted(elements.keys(), key=lambda e: len(elements[e]['our']), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(sorted_els))
    width = 0.35

    our_means = [np.mean(elements[el]['our']) for el in sorted_els]
    gt_means = [np.mean(elements[el]['gt']) for el in sorted_els]
    our_se = [np.std(elements[el]['our']) / np.sqrt(len(elements[el]['our'])) for el in sorted_els]
    gt_se = [np.std(elements[el]['gt']) / np.sqrt(len(elements[el]['gt'])) for el in sorted_els]

    bars1 = ax.bar(x - width/2, our_means, width, yerr=our_se,
                   label='Extracted', color='#1f77b4', alpha=0.8, capsize=3)
    bars2 = ax.bar(x + width/2, gt_means, width, yerr=gt_se,
                   label='Ground Truth', color='#ff7f0e', alpha=0.8, capsize=3)

    ax.set_xlabel('Element')
    ax.set_ylabel('Mean Effect Size (%)')
    ax.set_title('Element-Level: Extracted vs Ground Truth Mean Effect Sizes')
    ax.set_xticks(x)
    counts = [len(elements[el]['our']) for el in sorted_els]
    ax.set_xticklabels([f"{el}\n(n={c})" for el, c in zip(sorted_els, counts)], fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "fig4_element_effects.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# FIGURE 5: Cross-dataset validation scatter (Hui 2023)
# ============================================================
def fig5_scatter_hui(hui_matches):
    """Scatter plot for Hui 2023 cross-dataset validation."""
    if not hui_matches:
        print("  Skipping fig5: no Hui matches")
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Convert ln_rr to percentage
    our_pct = []
    gt_pct = []
    for m in hui_matches:
        if m['our_lnrr'] is not None and m['gt_lnrr'] is not None:
            our_pct.append((math.exp(m['our_lnrr']) - 1) * 100)
            gt_pct.append((math.exp(m['gt_lnrr']) - 1) * 100)

    ax.scatter(gt_pct, our_pct, c='#e377c2', alpha=0.7, s=40, edgecolors='white', linewidth=0.5,
               label=f'Zn observations (n={len(our_pct)})')

    # Reference line
    all_vals = gt_pct + our_pct
    mn, mx = min(all_vals) - 10, max(all_vals) + 10
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, linewidth=1, label='Perfect agreement')

    r, p = sp_stats.pearsonr(gt_pct, our_pct)

    ax.set_xlabel('Ground Truth Effect Size (%)')
    ax.set_ylabel('Extracted Effect Size (%)')
    ax.set_title(f'Hui 2023 (Zn Biofortification): Cross-Dataset Validation\n'
                 f'(n={len(our_pct)}, r={r:.3f})')
    ax.legend(loc='upper left')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    fig.tight_layout()
    path = OUT_DIR / "fig5_scatter_hui.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# FIGURE: Bland-Altman plot (supplementary)
# ============================================================
def fig_bland_altman(matches):
    """Bland-Altman plot showing agreement between extraction and GT."""
    fig, ax = plt.subplots(figsize=(8, 6))

    means = []
    diffs = []
    for m in matches:
        our = m['our_effect'] * 100
        gt = m['gt_effect'] * 100
        mean_val = (our + gt) / 2
        diff = our - gt
        means.append(mean_val)
        diffs.append(diff)

    means = np.array(means)
    diffs = np.array(diffs)

    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    ax.scatter(means, diffs, alpha=0.4, s=20, c='#1f77b4', edgecolors='white', linewidth=0.3)
    ax.axhline(mean_diff, color='red', linewidth=1, label=f'Mean bias: {mean_diff:.1f}%')
    ax.axhline(mean_diff + 1.96 * std_diff, color='red', linewidth=0.8, linestyle='--',
               label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.1f}%')
    ax.axhline(mean_diff - 1.96 * std_diff, color='red', linewidth=0.8, linestyle='--',
               label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.1f}%')
    ax.axhline(0, color='gray', linewidth=0.5)

    ax.set_xlabel('Mean of Extracted and GT Effect (%)')
    ax.set_ylabel('Difference (Extracted - GT) (%)')
    ax.set_title('Bland-Altman Plot: Extraction Agreement')
    ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "fig_bland_altman.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# FIGURE: Error distribution histogram
# ============================================================
def fig_error_distribution(matches):
    """Histogram of absolute errors."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    errors = [m['abs_error'] * 100 for m in matches]

    # Absolute error histogram
    bins = np.arange(0, max(50, max(errors) + 5), 2)
    ax1.hist(errors, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='white')
    ax1.axvline(5, color=TIER_COLORS['Excellent'], linewidth=1.5, linestyle='--', label='5%')
    ax1.axvline(10, color=TIER_COLORS['Good'], linewidth=1.5, linestyle='--', label='10%')
    ax1.axvline(20, color=TIER_COLORS['Fair'], linewidth=1.5, linestyle='--', label='20%')
    ax1.set_xlabel('Absolute Error (%)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Extraction Errors')
    ax1.legend()

    # Cumulative distribution
    sorted_err = np.sort(errors)
    cumulative = np.arange(1, len(sorted_err) + 1) / len(sorted_err) * 100
    ax2.plot(sorted_err, cumulative, color='#1f77b4', linewidth=2)
    ax2.axvline(5, color=TIER_COLORS['Excellent'], linewidth=1, linestyle='--')
    ax2.axvline(10, color=TIER_COLORS['Good'], linewidth=1, linestyle='--')
    ax2.axvline(20, color=TIER_COLORS['Fair'], linewidth=1, linestyle='--')

    # Annotate key thresholds
    n = len(errors)
    pct5 = sum(1 for e in errors if e <= 5) / n * 100
    pct10 = sum(1 for e in errors if e <= 10) / n * 100
    pct20 = sum(1 for e in errors if e <= 20) / n * 100
    ax2.annotate(f'{pct5:.0f}%', xy=(5, pct5), fontsize=9, color=TIER_COLORS['Excellent'],
                 xytext=(7, pct5 - 5), ha='left')
    ax2.annotate(f'{pct10:.0f}%', xy=(10, pct10), fontsize=9, color=TIER_COLORS['Good'],
                 xytext=(12, pct10 - 5), ha='left')
    ax2.annotate(f'{pct20:.0f}%', xy=(20, pct20), fontsize=9, color=TIER_COLORS['Fair'],
                 xytext=(22, pct20 - 5), ha='left')

    ax2.set_xlabel('Absolute Error (%)')
    ax2.set_ylabel('Cumulative Percentage')
    ax2.set_title('Cumulative Error Distribution')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    path = OUT_DIR / "fig_error_distribution.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# FIGURE: Element-level accuracy heatmap
# ============================================================
def fig_element_heatmap(matches):
    """Element-level accuracy metrics heatmap."""
    by_element = defaultdict(list)
    for m in matches:
        by_element[m['element']].append(m)

    # Filter >= 5 matches
    elements = {el: ms for el, ms in by_element.items() if len(ms) >= 5}
    sorted_els = sorted(elements.keys(), key=lambda e: len(elements[e]), reverse=True)

    # Calculate metrics per element
    data = []
    for el in sorted_els:
        ms = elements[el]
        n = len(ms)
        our = [m['our_effect'] for m in ms]
        gt = [m['gt_effect'] for m in ms]
        errors = [m['abs_error'] * 100 for m in ms]

        r, _ = sp_stats.pearsonr(gt, our) if n > 2 else (0, 1)
        mae = np.mean(errors)
        w5 = sum(1 for e in errors if e <= 5) / n * 100
        w10 = sum(1 for e in errors if e <= 10) / n * 100
        dir_agree = sum(1 for o, g in zip(our, gt) if g != 0 and (o > 0) == (g > 0))
        dir_total = sum(1 for g in gt if g != 0)
        dir_pct = dir_agree / dir_total * 100 if dir_total > 0 else 0

        data.append({
            'element': el, 'n': n, 'r': r, 'mae': mae,
            'w5': w5, 'w10': w10, 'dir': dir_pct,
            'our_mean': np.mean(our) * 100, 'gt_mean': np.mean(gt) * 100,
        })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create table-like figure
    cols = ['Element', 'N', 'r', 'MAE (%)', 'Within 5%', 'Within 10%', 'Direction', 'Our Mean%', 'GT Mean%']
    cell_text = []
    for d in data:
        cell_text.append([
            d['element'], str(d['n']), f"{d['r']:.3f}", f"{d['mae']:.1f}",
            f"{d['w5']:.0f}%", f"{d['w10']:.0f}%", f"{d['dir']:.0f}%",
            f"{d['our_mean']:.1f}", f"{d['gt_mean']:.1f}",
        ])

    ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=cols, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Color code MAE column
    for i, d in enumerate(data):
        mae = d['mae']
        if mae < 5:
            color = '#d4edda'
        elif mae < 10:
            color = '#cce5ff'
        elif mae < 20:
            color = '#fff3cd'
        else:
            color = '#f8d7da'
        table[i + 1, 3].set_facecolor(color)

    ax.set_title('Element-Level Extraction Accuracy (Loladze Dataset)', fontsize=13, pad=20)

    fig.tight_layout()
    path = OUT_DIR / "fig_element_table.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")

    return data


# ============================================================
# FIGURE: Combined scatter (both datasets)
# ============================================================
def fig_combined_scatter(lol_matches, hui_matches, li_matches=None):
    """Combined scatter plot showing all three datasets."""
    n_panels = 3 if li_matches else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 2:
        ax1, ax2 = axes
        ax3 = None
    else:
        ax1, ax2, ax3 = axes

    # Loladze
    our_l = [m['our_effect'] * 100 for m in lol_matches]
    gt_l = [m['gt_effect'] * 100 for m in lol_matches]
    r_l, _ = sp_stats.pearsonr(gt_l, our_l)

    ax1.scatter(gt_l, our_l, alpha=0.4, s=20, c='#1f77b4', edgecolors='white', linewidth=0.3)
    ax1.plot([-100, 150], [-100, 150], 'k--', alpha=0.5, linewidth=1)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Ground Truth Effect Size (%)')
    ax1.set_ylabel('Extracted Effect Size (%)')
    ax1.set_title(f'(A) Loladze 2014: CO$_2$ + Minerals\nn={len(lol_matches)}, r={r_l:.3f}')
    ax1.set_xlim(-100, 150)
    ax1.set_ylim(-100, 150)

    # Hui
    if hui_matches:
        our_h = []
        gt_h = []
        for m in hui_matches:
            if m['our_lnrr'] is not None and m['gt_lnrr'] is not None:
                our_h.append((math.exp(m['our_lnrr']) - 1) * 100)
                gt_h.append((math.exp(m['gt_lnrr']) - 1) * 100)

        if our_h:
            r_h, _ = sp_stats.pearsonr(gt_h, our_h)
            ax2.scatter(gt_h, our_h, alpha=0.6, s=40, c='#e377c2', edgecolors='white', linewidth=0.5)
            ax2.plot([-100, 500], [-100, 500], 'k--', alpha=0.5, linewidth=1)
            ax2.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
            ax2.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
            ax2.set_xlabel('Ground Truth Effect Size (%)')
            ax2.set_ylabel('Extracted Effect Size (%)')
            ax2.set_title(f'(B) Hui 2023: Zn Biofortification\nn={len(our_h)}, r={r_h:.3f}')
    else:
        ax2.text(0.5, 0.5, 'Hui data\nnot yet available', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=14, color='gray')
        ax2.set_title('(B) Hui 2023: Zn Biofortification')

    # Li 2022
    if ax3 is not None and li_matches:
        our_li = [m['our_effect'] for m in li_matches]
        gt_li = [m['gt_effect'] for m in li_matches]
        r_li, _ = sp_stats.pearsonr(gt_li, our_li)

        # Color by biostimulant category
        cat_colors = {
            'SWE': '#2ca02c', 'PHs': '#ff7f0e', 'HFA': '#9467bd',
            'Chi': '#d62728', 'Si': '#17becf', 'Phi': '#bcbd22', 'PE': '#8c564b',
        }
        for m in li_matches:
            c = cat_colors.get(m.get('category', ''), '#333333')
            ax3.scatter(m['gt_effect'], m['our_effect'], alpha=0.5, s=30, c=c,
                       edgecolors='white', linewidth=0.3)

        lims_li = [-50, max(max(our_li), max(gt_li)) * 1.1 + 10]
        ax3.plot(lims_li, lims_li, 'k--', alpha=0.5, linewidth=1)
        ax3.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax3.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
        ax3.set_xlabel('Ground Truth Effect Size (%)')
        ax3.set_ylabel('Extracted Effect Size (%)')
        ax3.set_title(f'(C) Li 2022: Biostimulant + Yield\nn={len(li_matches)}, r={r_li:.3f}')

        # Legend for categories
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=6, label=k)
                   for k, v in cat_colors.items()
                   if any(m.get('category') == k for m in li_matches)]
        if handles:
            ax3.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.8)

    fig.tight_layout()
    path = OUT_DIR / "fig_combined_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# FIGURE: Direction agreement by element
# ============================================================
def fig_direction_by_element(matches):
    """Direction agreement broken down by element."""
    by_element = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'zero': 0})
    for m in matches:
        el = m['element']
        if m['gt_effect'] == 0:
            by_element[el]['zero'] += 1
        elif (m['our_effect'] > 0) == (m['gt_effect'] > 0):
            by_element[el]['correct'] += 1
        else:
            by_element[el]['wrong'] += 1

    # Filter >= 5 matches
    elements = {el: d for el, d in by_element.items()
                if d['correct'] + d['wrong'] >= 5}
    sorted_els = sorted(elements.keys(),
                        key=lambda e: elements[e]['correct'] / max(elements[e]['correct'] + elements[e]['wrong'], 1),
                        reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(sorted_els))

    correct = [elements[el]['correct'] for el in sorted_els]
    wrong = [elements[el]['wrong'] for el in sorted_els]
    totals = [c + w for c, w in zip(correct, wrong)]
    pcts = [c / t * 100 if t > 0 else 0 for c, t in zip(correct, totals)]

    bars = ax.bar(x, pcts, color=['#2ca02c' if p >= 80 else '#ff7f0e' if p >= 70 else '#d62728'
                                    for p in pcts], alpha=0.8, edgecolor='white')

    for i, (p, t) in enumerate(zip(pcts, totals)):
        ax.text(i, p + 1, f'{p:.0f}%\n(n={t})', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_els)
    ax.set_ylabel('Direction Agreement (%)')
    ax.set_title('Direction Agreement by Element')
    ax.set_ylim(0, 110)
    ax.axhline(80, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

    fig.tight_layout()
    path = OUT_DIR / "fig_direction_by_element.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ============================================================
# Summary statistics for paper text
# ============================================================
def print_paper_statistics(lol_matches, lol_report, hui_matches, hui_report):
    """Print comprehensive statistics for the paper."""
    print(f"\n{'='*70}")
    print("PAPER STATISTICS SUMMARY")
    print(f"{'='*70}\n")

    # Loladze
    print("LOLADZE DATASET (CO2 + Mineral Concentrations)")
    print(f"  Papers processed: {lol_report['papers_processed']}")
    print(f"  Papers with GT match: {lol_report['papers_with_gt']}")
    print(f"  Element capture rate: {lol_report['capture_rate']}")
    print(f"  Pearson r: {lol_report['pearson_r']}")
    print(f"  MAE: {lol_report['mae_pct']}%")
    print(f"  Within 5%: {lol_report['within_5pct']}")
    print(f"  Within 10%: {lol_report['within_10pct']}")
    print(f"  Within 20%: {lol_report['within_20pct']}")
    print(f"  Direction: {lol_report['direction_agreement']}")

    # Paper tiers
    maes = [p['mae'] for p in lol_report['per_paper']
            if p.get('mae') is not None and not (isinstance(p['mae'], float) and math.isnan(p['mae']))]
    excellent = sum(1 for m in maes if m < 5)
    good = sum(1 for m in maes if 5 <= m < 10)
    fair = sum(1 for m in maes if 10 <= m < 20)
    poor = sum(1 for m in maes if m >= 20)
    total = len(maes)
    print(f"\n  Paper accuracy tiers:")
    print(f"    Excellent (<5% MAE): {excellent}/{total} ({excellent/total*100:.0f}%)")
    print(f"    Good (5-10%):        {good}/{total} ({good/total*100:.0f}%)")
    print(f"    Fair (10-20%):       {fair}/{total} ({fair/total*100:.0f}%)")
    print(f"    Poor (>20%):         {poor}/{total} ({poor/total*100:.0f}%)")

    # Excluding Baslam
    lol_excl = [m for m in lol_matches if 'Baslam' not in m['paper']]
    if lol_excl:
        our_e = [m['our_effect'] for m in lol_excl]
        gt_e = [m['gt_effect'] for m in lol_excl]
        r_excl, _ = sp_stats.pearsonr(gt_e, our_e)
        mae_excl = np.mean([m['abs_error'] * 100 for m in lol_excl])
        print(f"\n  Excluding Baslam (worst paper):")
        print(f"    n={len(lol_excl)}, r={r_excl:.3f}, MAE={mae_excl:.1f}%")

    # Excluding err>20%
    lol_good = [m for m in lol_matches if m['abs_error'] * 100 <= 20]
    if lol_good:
        our_g = [m['our_effect'] for m in lol_good]
        gt_g = [m['gt_effect'] for m in lol_good]
        r_good, _ = sp_stats.pearsonr(gt_g, our_g)
        print(f"\n  Excluding errors >20%:")
        print(f"    n={len(lol_good)}, r={r_good:.3f}")

    # Overall mean effect
    our_mean = np.mean([m['our_effect'] * 100 for m in lol_matches])
    gt_mean = np.mean([m['gt_effect'] * 100 for m in lol_matches])
    print(f"\n  Overall mean mineral effect:")
    print(f"    Extracted: {our_mean:.1f}%")
    print(f"    Ground truth: {gt_mean:.1f}%")
    print(f"    Difference: {abs(our_mean - gt_mean):.1f} percentage points")

    # Hui
    if hui_report:
        print(f"\nHUI 2023 DATASET (Zn Biofortification)")
        overall = hui_report.get('overall', {})
        print(f"  Papers with GT match: {sum(1 for p in hui_report.get('per_paper', []) if p.get('gt_rows', 0) > 0)}")
        print(f"  Matched obs: {hui_report.get('total_matched', 'N/A')}")
        print(f"  Pearson r: {overall.get('pearson_r', 'N/A')}")
        print(f"  MAE (%): {overall.get('mae_pct', 'N/A')}")
        print(f"  Direction: {overall.get('direction', 'N/A')}")
        print(f"  Within 5% ln_rr: {overall.get('within_5pct_lnrr', 'N/A')}")
        print(f"  Within 10% ln_rr: {overall.get('within_10pct_lnrr', 'N/A')}")
        print(f"  Within 20% ln_rr: {overall.get('within_20pct_lnrr', 'N/A')}")

    print(f"\n{'='*70}")


def main():
    print("Generating paper figures...")
    print(f"Output directory: {OUT_DIR}\n")

    # Load data
    lol_matches = load_loladze_matches()
    lol_report = load_loladze_report()
    hui_matches = load_hui_matches()
    hui_report = load_hui_report()
    li_matches = load_li2022_matches()

    print(f"Loaded {len(lol_matches)} Loladze matches, {len(hui_matches)} Hui matches, {len(li_matches)} Li 2022 matches\n")

    # Generate figures
    print("Generating figures:")
    fig2_scatter_loladze(lol_matches)
    fig3_paper_mae(lol_report)
    fig4_element_effects(lol_matches)
    fig5_scatter_hui(hui_matches)
    fig_bland_altman(lol_matches)
    fig_error_distribution(lol_matches)
    fig_element_heatmap(lol_matches)
    fig_combined_scatter(lol_matches, hui_matches, li_matches)
    fig_direction_by_element(lol_matches)

    # Print statistics
    print_paper_statistics(lol_matches, lol_report, hui_matches, hui_report)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
