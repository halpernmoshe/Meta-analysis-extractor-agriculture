"""Generate publication-quality figures for the revised meta-analysis extraction paper.

New framing: consensus as quality filter, not quantity booster.
Key new figure: Fig 2 showing consensus reliability (MAE by consensus fraction).

Reads validation data from:
  - output/loladze_combined_51/validation_matches.csv
  - output/loladze_combined_51/validation_report_full.json
  - output/hui2023_full_35/validation_matches.csv
  - output/li2022_combined/validation_matches.csv
  - output/paper_supplementary/S3_consensus_stats.csv
  - output/paper_supplementary/S1_per_paper_validation.csv

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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

BASE = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
LOL_DIR = BASE / "output" / "loladze_combined_51"
HUI_DIR = BASE / "output" / "hui2023_full_35"
LI_DIR = BASE / "output" / "li2022_combined"
OUT_DIR = BASE / "output" / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

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

ELEMENT_COLORS = {
    'N': '#1f77b4', 'P': '#ff7f0e', 'K': '#2ca02c', 'CA': '#d62728',
    'MG': '#9467bd', 'FE': '#8c564b', 'ZN': '#e377c2', 'MN': '#7f7f7f',
    'CU': '#bcbd22', 'S': '#17becf', 'B': '#aec7e8', 'NA': '#ffbb78',
    'AL': '#98df8a', 'CO': '#ff9896', 'NI': '#c5b0d5', 'SI': '#c49c94',
}

TIER_COLORS = {
    'Excellent': '#2ca02c', 'Good': '#1f77b4', 'Fair': '#ff7f0e', 'Poor': '#d62728',
}


# ── Data Loading ──────────────────────────────────────────────────────────

def load_loladze_matches():
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
                })
            except (ValueError, KeyError):
                continue
    return matches


def load_loladze_report():
    with open(LOL_DIR / "validation_report_full.json") as f:
        return json.load(f)


def load_consensus_stats():
    """Load S3 consensus stats and compute per-paper consensus fraction."""
    stats = {}
    csv_path = BASE / "output" / "paper_supplementary" / "S3_consensus_stats.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Dataset'] != 'Loladze':
                continue
            paper = row['Paper']
            total = int(row['Consensus Obs'])
            agree = int(row['Both Agree'])
            vision = int(row['Vision'])
            if total > 0:
                stats[paper] = {
                    'total': total,
                    'both_agree': agree,
                    'vision': vision,
                    'consensus_frac': agree / total,
                    'vision_frac': vision / total,
                }
    return stats


def load_per_paper_validation():
    """Load S1 per-paper validation results."""
    papers = {}
    csv_path = BASE / "output" / "paper_supplementary" / "S1_per_paper_validation.csv"
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            paper_id = row['Paper']
            try:
                mae = float(row['MAE (%)'])
            except ValueError:
                mae = None
            papers[paper_id] = {
                'reference': row['Reference'],
                'matched': int(row['Matched']) if row['Matched'] else 0,
                'mae': mae,
                'tier': row['Tier'],
                'direction': row['Direction'],
                'pearson_r': row.get('Pearson r', 'N/A'),
            }
    return papers


def load_hui_matches():
    matches = []
    csv_path = HUI_DIR / "validation_matches.csv"
    if not csv_path.exists():
        csv_path = HUI_DIR / "validation_hui2023_matches.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                our_ctrl = float(row.get('ext_ctrl', row.get('our_ctrl', 0)))
                our_treat = float(row.get('ext_treat', row.get('our_treat', 0)))
                gt_ctrl = float(row.get('gt_ctrl', 0))
                gt_treat = float(row.get('gt_treat', 0))
                our_lnrr = math.log(our_treat / our_ctrl) if our_ctrl > 0 and our_treat > 0 else None
                gt_lnrr = math.log(gt_treat / gt_ctrl) if gt_ctrl > 0 and gt_treat > 0 else None
                matches.append({
                    'our_lnrr': our_lnrr, 'gt_lnrr': gt_lnrr,
                    'ext_effect': float(row.get('ext_effect', 0)),
                    'gt_effect': float(row.get('gt_effect', 0)),
                })
            except (ValueError, KeyError):
                continue
    return matches


def load_li2022_matches():
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
                    'category': row.get('category', ''),
                })
            except (ValueError, KeyError):
                continue
    return matches


# ── FIGURE 1: Pipeline Architecture ──────────────────────────────────────

def fig1_pipeline_architecture():
    """Pipeline architecture with a concrete worked example.

    Left: abstract pipeline flow. Right: worked example showing a real paper
    (Baslam 2012) flowing through each stage with concrete values.
    """
    fig, (ax_pipe, ax_ex) = plt.subplots(1, 2, figsize=(16, 8),
                                          gridspec_kw={'width_ratios': [1, 1.1]})
    # ── Left panel: Pipeline schematic ────────────────────────────────────
    ax = ax_pipe
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Pipeline Architecture', fontsize=13, fontweight='bold', pad=10)

    def _box(ax, x, y, w, h, text, color, fontsize=9, bold=True):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#333',
                              linewidth=1.3, zorder=2, clip_on=False, joinstyle='round')
        ax.add_patch(rect)
        fw = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fw, zorder=3)

    def _arrow(ax, x1, y1, x2, y2, color='#333', lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw))

    # Input
    _box(ax, 0.3, 8.5, 1.8, 0.8, 'PDF\nInput', '#F5F5F5')

    # Stage 1
    _box(ax, 0.3, 6.8, 3.5, 0.9, 'Stage 1: Reconnaissance\n(Claude Sonnet 4)', '#E8F0FE')
    _arrow(ax, 1.2, 8.5, 2.0, 7.7)

    # Routing
    _box(ax, 4.5, 6.8, 3.0, 0.9, 'Routing Decision\nTEXT / HYBRID / VISION', '#FFF3E0')
    _arrow(ax, 3.8, 7.25, 4.5, 7.25)

    # Stage 2
    _box(ax, 0.3, 4.8, 3.5, 0.9, 'Stage 2: Dual Extraction\n(Claude + Kimi)', '#E8F0FE')
    _arrow(ax, 2.0, 6.8, 2.0, 5.7)

    # Stage 3
    _box(ax, 4.5, 4.8, 3.0, 0.9, 'Stage 3: Consensus\n(2-of-3 voting)', '#E8F5E9')
    _arrow(ax, 3.8, 5.25, 4.5, 5.25)

    # Tiebreaker
    _box(ax, 0.3, 3.0, 3.5, 0.9, 'Tiebreaker (if needed)\n(Gemini 3 Flash)', '#FFF3E0')
    _arrow(ax, 2.0, 4.8, 2.0, 3.9)
    _arrow(ax, 3.8, 3.45, 4.5, 4.8)

    # Confidence outputs
    _box(ax, 5.5, 3.0, 3.8, 0.65, 'HIGH: 2+ models agree\nMAE ~ 4%  |  auto-validated', '#C8E6C9', fontsize=8, bold=False)
    _box(ax, 5.5, 2.0, 3.8, 0.65, 'MEDIUM: tiebreaker / single\nflagged for review', '#FFF9C4', fontsize=8, bold=False)
    _box(ax, 5.5, 1.0, 3.8, 0.65, 'LOW: vision-only / OCR\nflagged for review', '#FFCDD2', fontsize=8, bold=False)

    _arrow(ax, 7.5, 4.8, 7.4, 3.65, color='#2ca02c')
    _arrow(ax, 7.5, 4.8, 7.4, 2.65, color='#ff8f00')
    _arrow(ax, 7.5, 4.8, 7.4, 1.65, color='#d32f2f')

    ax.text(7.4, 0.3, 'Confidence-Stratified Output', ha='center', fontsize=10,
            fontweight='bold', style='italic')

    # ── Right panel: Worked example ───────────────────────────────────────
    ax = ax_ex
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Worked Example: Baslam et al. 2012', fontsize=13, fontweight='bold', pad=10)

    # Step annotations
    steps = [
        (0.5, 9.0, 9.0, 0.7, '#E8F0FE',
         'Recon: "Table 2 has mineral concentrations for 2 lettuce cultivars x 2 CO2 levels.\n'
         'Variance: SE in table footnote. Design: 2x2 factorial. Mode: TEXT."'),
        (0.5, 7.6, 9.0, 1.0, '#E8F0FE',
         'Claude extracts 68 obs:  Ca = 8.21 mg/g (control), 7.43 mg/g (elevated)\n'
         'Kimi extracts 68 obs:    Ca = 8.21 mg/g (control), 7.43 mg/g (elevated)'),
        (0.5, 6.2, 9.0, 0.9, '#E8F5E9',
         'Consensus: 68/68 matched (100%)  \u2192  All HIGH confidence\n'
         'Ca effect: -9.5%  |  GT: -9.5%  |  Error: 0.0 pp'),
        (0.5, 4.6, 9.0, 1.1, '#C8E6C9',
         'Result: 18 observations matched to ground truth\n'
         'MAE = 1.0%  |  r = 0.997  |  Direction: 17/17 correct  |  Tier: Excellent'),
    ]

    labels = ['1. Reconnaissance', '2. Dual Extraction', '3. Consensus', '4. Validation']
    y_labels = [9.0, 7.6, 6.2, 4.6]

    for i, (x, y, w, h, color, text) in enumerate(steps):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#555',
                              linewidth=1.0, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + 0.15, y + h - 0.12, labels[i], fontsize=9, fontweight='bold',
                va='top', zorder=3)
        ax.text(x + 0.15, y + h - 0.38, text, fontsize=7.5, va='top', zorder=3,
                family='monospace', linespacing=1.4)

    # Arrows between steps
    for i in range(len(steps) - 1):
        _, y1, _, h1, _, _ = steps[i]
        _, y2, _, h2, _, _ = steps[i + 1]
        _arrow(ax, 5.0, y1, 5.0, y2 + h2, color='#555', lw=1.2)

    # Contrast: a HARD paper example
    ax.text(0.5, 3.6, 'Contrast: Fangmeier et al. 2002 (HARD paper)', fontsize=9,
            fontweight='bold')
    contrast_text = (
        'CO2 x O3 factorial  |  Claude: 0 obs, Kimi: 31 obs, Gemini: 62 obs\n'
        'Consensus: 23% (31/134)  \u2192  mostly vision-dependent  \u2192  MAE = 8.0%\n'
        'Alignment issue: pipeline selected wrong O3 level vs. ground truth'
    )
    rect = plt.Rectangle((0.5, 2.3), 9.0, 1.0, facecolor='#FFECB3', edgecolor='#555',
                          linewidth=1.0, zorder=2, clip_on=False)
    ax.add_patch(rect)
    ax.text(0.65, 3.15, contrast_text, fontsize=7.5, va='top', zorder=3,
            family='monospace', linespacing=1.4)

    # Key takeaway
    ax.text(5.0, 1.5, 'Consensus fraction (100% vs 23%) correctly predicted\n'
            'which paper would be accurate and which would need review.',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E0E0E0', alpha=0.8))

    fig.tight_layout(w_pad=3)
    path = OUT_DIR / "fig1_pipeline_architecture.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── FIGURE 2: Consensus Reliability (KEY NEW FIGURE) ─────────────────────

def fig2_consensus_reliability(lol_matches, consensus_stats, paper_validation):
    """The core figure: consensus fraction predicts extraction quality."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Panel A: Scatter of vision_frac vs MAE per paper ──
    papers_x = []  # consensus fraction
    papers_y = []  # MAE
    papers_n = []  # matched obs
    papers_cat = []  # consensus-dominant or vision-dependent

    for paper_id, pv in paper_validation.items():
        if pv['mae'] is None:
            continue
        cs = consensus_stats.get(paper_id)
        if cs is None or cs['total'] == 0:
            continue
        cf = cs['consensus_frac']
        papers_x.append(cf * 100)
        papers_y.append(pv['mae'])
        papers_n.append(pv['matched'])
        papers_cat.append('consensus' if cf > 0.5 else 'vision')

    papers_x = np.array(papers_x)
    papers_y = np.array(papers_y)
    papers_n = np.array(papers_n)

    # Color by category
    colors = ['#2ca02c' if c == 'consensus' else '#d62728' for c in papers_cat]
    sizes = np.clip(papers_n * 3, 30, 300)

    ax1.scatter(papers_x, papers_y, c=colors, s=sizes, alpha=0.7,
                edgecolors='white', linewidth=0.8, zorder=3)

    # Add trend line
    slope, intercept, r, p, se = sp_stats.linregress(papers_x, papers_y)
    x_line = np.linspace(0, 100, 100)
    ax1.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1,
             label=f'Trend: r={r:.2f}, p={p:.3f}')

    # Threshold line
    ax1.axvline(50, color='gray', linestyle=':', alpha=0.5)
    ax1.text(52, max(papers_y) * 0.9, 'Consensus\nthreshold', fontsize=8, color='gray')

    ax1.set_xlabel('Consensus Fraction (% observations agreed by 2+ models)')
    ax1.set_ylabel('Mean Absolute Error (%)')
    ax1.set_title('(A) Per-Paper: Consensus Fraction vs Accuracy')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c',
               markersize=10, label='Consensus-dominant (>50%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728',
               markersize=10, label='Vision-dependent (<50%)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-1, min(max(papers_y) * 1.1, 65))

    # ── Panel B: Box plot comparing consensus-dominant vs vision-dependent ──
    # Compute per-paper MAE by category
    consensus_maes = [y for y, c in zip(papers_y, papers_cat) if c == 'consensus']
    vision_maes = [y for y, c in zip(papers_y, papers_cat) if c == 'vision']

    # Compute aggregate stats
    n_cons = len(consensus_maes)
    n_vis = len(vision_maes)
    mean_cons = np.mean(consensus_maes) if consensus_maes else 0
    mean_vis = np.mean(vision_maes) if vision_maes else 0

    bp = ax2.boxplot([consensus_maes, vision_maes],
                     tick_labels=[f'Consensus-\ndominant\n(n={n_cons})',
                                  f'Vision-\ndependent\n(n={n_vis})'],
                     patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=2))

    bp['boxes'][0].set_facecolor('#C8E6C9')
    bp['boxes'][1].set_facecolor('#FFCDD2')

    # Add mean markers
    ax2.scatter([1], [mean_cons], marker='D', color='#2ca02c', s=80, zorder=5, label=f'Mean: {mean_cons:.1f}%')
    ax2.scatter([2], [mean_vis], marker='D', color='#d62728', s=80, zorder=5, label=f'Mean: {mean_vis:.1f}%')

    # Add individual points
    ax2.scatter([1] * n_cons, consensus_maes, alpha=0.4, s=20, color='#2ca02c', zorder=4)
    ax2.scatter([2] * n_vis, vision_maes, alpha=0.4, s=20, color='#d62728', zorder=4)

    # Mann-Whitney test
    if consensus_maes and vision_maes:
        stat, p_mw = sp_stats.mannwhitneyu(consensus_maes, vision_maes, alternative='less')
        sig = '***' if p_mw < 0.001 else '**' if p_mw < 0.01 else '*' if p_mw < 0.05 else 'n.s.'
        ax2.text(1.5, max(max(consensus_maes), max(vision_maes)) * 0.95,
                 f'p = {p_mw:.4f} {sig}', ha='center', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Mean Absolute Error (%)')
    ax2.set_title('(B) MAE by Paper Category')
    ax2.legend(loc='upper left', fontsize=9)

    fig.suptitle('Figure 2. Multi-Model Agreement Predicts Extraction Quality', fontsize=14, y=1.01)
    fig.tight_layout()
    path = OUT_DIR / "fig2_consensus_reliability.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")

    # Print stats for paper
    print(f"    Consensus-dominant: n={n_cons}, mean MAE={mean_cons:.1f}%, median={np.median(consensus_maes):.1f}%")
    print(f"    Vision-dependent:   n={n_vis}, mean MAE={mean_vis:.1f}%, median={np.median(vision_maes):.1f}%")


# ── FIGURE 3: Per-paper MAE with consensus indicators ─────────────────────

def fig3_paper_mae_confidence(report, consensus_stats):
    """Bar chart of per-paper MAE with consensus/vision fraction overlay."""
    papers = []
    for p in report['per_paper']:
        mae = p.get('mae')
        if mae is None or (isinstance(mae, float) and math.isnan(mae)):
            continue
        paper_id = p['paper_id']
        name = paper_id.split('_', 1)[1] if '_' in paper_id else paper_id
        cs = consensus_stats.get(paper_id, {})
        cf = cs.get('consensus_frac', 0)
        papers.append({
            'name': name,
            'paper_id': paper_id,
            'mae': mae,
            'matched': p.get('matched', 0),
            'consensus_frac': cf,
        })

    papers.sort(key=lambda x: x['mae'])

    fig, ax = plt.subplots(figsize=(12, 8))
    n = len(papers)
    names = [p['name'] for p in papers]
    maes = [p['mae'] for p in papers]
    cfracs = [p['consensus_frac'] for p in papers]

    # Color by tier
    tier_colors = []
    for mae in maes:
        if mae < 5:
            tier_colors.append(TIER_COLORS['Excellent'])
        elif mae < 10:
            tier_colors.append(TIER_COLORS['Good'])
        elif mae < 20:
            tier_colors.append(TIER_COLORS['Fair'])
        else:
            tier_colors.append(TIER_COLORS['Poor'])

    bars = ax.barh(range(n), maes, color=tier_colors, edgecolor='white', linewidth=0.5, alpha=0.8)

    # Add consensus fraction markers on the right
    ax2 = ax.twiny()
    ax2.scatter(np.array(cfracs) * 100, range(n), marker='|', color='black', s=80, zorder=5,
                label='Consensus %')
    ax2.set_xlabel('Consensus Fraction (%)', fontsize=10)
    ax2.set_xlim(0, 105)

    ax.set_yticks(range(n))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Mean Absolute Error (%)')
    ax.set_title('Per-Paper Accuracy with Consensus Indicators (Loladze Dataset)')
    ax.invert_yaxis()

    # Tier threshold lines
    ax.axvline(5, color=TIER_COLORS['Excellent'], linestyle=':', alpha=0.5)
    ax.axvline(10, color=TIER_COLORS['Good'], linestyle=':', alpha=0.5)
    ax.axvline(20, color=TIER_COLORS['Fair'], linestyle=':', alpha=0.5)

    # Legend
    legend_elements = [
        Patch(facecolor=TIER_COLORS['Excellent'], label=f'Excellent (<5%, n={sum(1 for m in maes if m < 5)})'),
        Patch(facecolor=TIER_COLORS['Good'], label=f'Good (5-10%, n={sum(1 for m in maes if 5 <= m < 10)})'),
        Patch(facecolor=TIER_COLORS['Fair'], label=f'Fair (10-20%, n={sum(1 for m in maes if 10 <= m < 20)})'),
        Patch(facecolor=TIER_COLORS['Poor'], label=f'Poor (>20%, n={sum(1 for m in maes if m >= 20)})'),
        Line2D([0], [0], marker='|', color='black', linestyle='None', markersize=8,
               label='Consensus fraction (top axis)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    fig.tight_layout()
    path = OUT_DIR / "fig3_paper_mae_confidence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── FIGURE: Combined scatter (all 3 datasets) ────────────────────────────

def fig_combined_scatter(lol_matches, hui_matches, li_matches):
    """Combined scatter plot showing all three datasets."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))

    # Loladze
    our_l = [m['our_effect'] * 100 for m in lol_matches]
    gt_l = [m['gt_effect'] * 100 for m in lol_matches]
    r_l, _ = sp_stats.pearsonr(gt_l, our_l)

    ax1.scatter(gt_l, our_l, alpha=0.4, s=20, c='#1f77b4', edgecolors='white', linewidth=0.3)
    ax1.plot([-100, 150], [-100, 150], 'k--', alpha=0.5, linewidth=1)
    ax1.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Ground Truth Effect (%)')
    ax1.set_ylabel('Extracted Effect (%)')
    ax1.set_title(f'(A) Loladze 2014 [Development]\nn={len(lol_matches)}, r={r_l:.3f}')
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
            ax2.set_xlabel('Ground Truth Effect (%)')
            ax2.set_ylabel('Extracted Effect (%)')
            ax2.set_title(f'(B) Hui 2023 [Blind]\nn={len(our_h)}, r={r_h:.3f}')

    # Li 2022
    if li_matches:
        our_li = [m['our_effect'] for m in li_matches]
        gt_li = [m['gt_effect'] for m in li_matches]
        r_li, _ = sp_stats.pearsonr(gt_li, our_li)

        cat_labels = {
            'SWE': 'Seaweed extract', 'PHs': 'Protein hydrolysate',
            'HFA': 'Humic/fulvic acid', 'Chi': 'Chitosan',
            'Si': 'Silicon', 'Phi': 'Phosphite', 'PE': 'Plant extract',
        }
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
        ax3.set_xlabel('Ground Truth Effect (%)')
        ax3.set_ylabel('Extracted Effect (%)')
        ax3.set_title(f'(C) Li 2022 [Cross-domain]\nn={len(li_matches)}, r={r_li:.3f}')

        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v, markersize=6,
                          label=cat_labels.get(k, k))
                   for k, v in cat_colors.items()
                   if any(m.get('category') == k for m in li_matches)]
        if handles:
            ax3.legend(handles=handles, loc='upper left', fontsize=7, framealpha=0.8)

    fig.tight_layout()
    path = OUT_DIR / "fig_combined_scatter.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Supplementary: Scatter by element (Loladze) ──────────────────────────

def fig_scatter_loladze(matches):
    fig, ax = plt.subplots(figsize=(8, 7))
    by_element = defaultdict(list)
    for m in matches:
        by_element[m['element']].append(m)

    sorted_elements = sorted(by_element.keys(), key=lambda e: len(by_element[e]), reverse=True)

    for el in sorted_elements:
        el_matches = by_element[el]
        x = [m['gt_effect'] * 100 for m in el_matches]
        y = [m['our_effect'] * 100 for m in el_matches]
        color = ELEMENT_COLORS.get(el, '#333333')
        ax.scatter(x, y, c=color, alpha=0.6, s=30, label=f"{el} (n={len(el_matches)})",
                   edgecolors='white', linewidth=0.3)

    ax.plot([-100, 150], [-100, 150], 'k--', alpha=0.5, linewidth=1, label='Perfect agreement')
    ax.axhline(0, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='gray', linewidth=0.5, alpha=0.3)

    all_our = [m['our_effect'] for m in matches]
    all_gt = [m['gt_effect'] for m in matches]
    r, _ = sp_stats.pearsonr(all_gt, all_our)

    ax.set_xlabel('Ground Truth Effect Size (%)')
    ax.set_ylabel('Extracted Effect Size (%)')
    ax.set_title(f'Loladze: Extracted vs Ground Truth by Element\n(n={len(matches)}, r={r:.3f})')
    ax.legend(loc='upper left', fontsize=8, ncol=2, framealpha=0.8)
    ax.set_xlim(-100, 150)
    ax.set_ylim(-100, 150)

    fig.tight_layout()
    path = OUT_DIR / "fig2_scatter_loladze.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Supplementary: Element effects ────────────────────────────────────────

def fig_element_effects(matches):
    by_element = defaultdict(lambda: {'our': [], 'gt': []})
    for m in matches:
        el = m['element']
        by_element[el]['our'].append(m['our_effect'] * 100)
        by_element[el]['gt'].append(m['gt_effect'] * 100)

    elements = {el: d for el, d in by_element.items() if len(d['our']) >= 5}
    sorted_els = sorted(elements.keys(), key=lambda e: len(elements[e]['our']), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(sorted_els))
    width = 0.35

    our_means = [np.mean(elements[el]['our']) for el in sorted_els]
    gt_means = [np.mean(elements[el]['gt']) for el in sorted_els]
    our_se = [np.std(elements[el]['our']) / np.sqrt(len(elements[el]['our'])) for el in sorted_els]
    gt_se = [np.std(elements[el]['gt']) / np.sqrt(len(elements[el]['gt'])) for el in sorted_els]

    ax.bar(x - width/2, our_means, width, yerr=our_se, label='Extracted',
           color='#1f77b4', alpha=0.8, capsize=3)
    ax.bar(x + width/2, gt_means, width, yerr=gt_se, label='Ground Truth',
           color='#ff7f0e', alpha=0.8, capsize=3)

    ax.set_xlabel('Element')
    ax.set_ylabel('Mean Effect Size (%)')
    ax.set_title('Element-Level: Extracted vs Ground Truth Mean Effect Sizes')
    counts = [len(elements[el]['our']) for el in sorted_els]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{el}\n(n={c})" for el, c in zip(sorted_els, counts)], fontsize=9)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend()

    fig.tight_layout()
    path = OUT_DIR / "fig4_element_effects.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Supplementary: Bland-Altman (Loladze only) ───────────────────────────

def fig_bland_altman_formal(matches):
    fig, ax = plt.subplots(figsize=(8, 6))

    means, diffs = [], []
    for m in matches:
        our = m['our_effect'] * 100
        gt = m['gt_effect'] * 100
        means.append((our + gt) / 2)
        diffs.append(our - gt)

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
    ax.set_title('Bland-Altman Plot: Extraction Agreement (Loladze)')
    ax.legend(loc='upper left', fontsize=9)

    fig.tight_layout()
    path = OUT_DIR / "fig7_bland_altman_formal.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Supplementary: Error distribution ─────────────────────────────────────

def fig_error_distribution(matches):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    errors = [m['abs_error'] * 100 for m in matches]
    n = len(errors)

    bins = np.arange(0, max(50, max(errors) + 5), 2)
    ax1.hist(errors, bins=bins, color='#1f77b4', alpha=0.7, edgecolor='white')
    ax1.axvline(5, color=TIER_COLORS['Excellent'], linewidth=1.5, linestyle='--', label='5%')
    ax1.axvline(10, color=TIER_COLORS['Good'], linewidth=1.5, linestyle='--', label='10%')
    ax1.axvline(20, color=TIER_COLORS['Fair'], linewidth=1.5, linestyle='--', label='20%')
    ax1.set_xlabel('Absolute Error (%)')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Extraction Errors')
    ax1.legend()

    sorted_err = np.sort(errors)
    cumulative = np.arange(1, n + 1) / n * 100
    ax2.plot(sorted_err, cumulative, color='#1f77b4', linewidth=2)
    ax2.axvline(5, color=TIER_COLORS['Excellent'], linewidth=1, linestyle='--')
    ax2.axvline(10, color=TIER_COLORS['Good'], linewidth=1, linestyle='--')
    ax2.axvline(20, color=TIER_COLORS['Fair'], linewidth=1, linestyle='--')

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


def fig7_consensus_predictors(consensus_stats, paper_validation):
    """Figure 7: What paper attributes predict consensus quality?

    Panel A: Difficulty level vs consensus fraction (bar + strip)
    Panel B: Difficulty level vs MAE (bar + strip)
    """
    import json as _json

    # Load challenge features
    challenge_path = BASE / "output" / "challenge_evaluation_2026-02-03" / "kimi" / "challenges_kimi.json"
    if not challenge_path.exists():
        print("  [SKIP] fig7_consensus_predictors: challenge data not found")
        return

    with open(challenge_path, 'r', encoding='utf-8') as f:
        challenges = _json.load(f)

    challenge_map = {}
    for c in challenges:
        fname = c['filename'].replace('.pdf', '')
        challenge_map[fname] = c

    # Build joined data
    rows = []
    for paper, cs in consensus_stats.items():
        pv = paper_validation.get(paper, {})
        ch = challenge_map.get(paper, {})
        if not ch:
            continue
        mae = pv.get('mae')
        difficulty = ch.get('difficulty', 'UNKNOWN')
        n_challenges = sum(1 for k, v in ch.items()
                          if isinstance(v, bool) and v and k.startswith(('is_', 'has_', 'needs_')))
        rows.append({
            'paper': paper,
            'consensus_frac': cs['consensus_frac'],
            'mae': mae,
            'difficulty': difficulty,
            'n_challenges': n_challenges,
            'has_complex_stats': ch.get('has_complex_stats', False),
            'is_scanned': ch.get('is_scanned', False),
            'has_image_tables': ch.get('has_image_tables', False),
        })

    if not rows:
        print("  [SKIP] fig7_consensus_predictors: no joined data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Difficulty vs Consensus Fraction
    ax = axes[0]
    diff_order = ['MEDIUM', 'HARD']
    diff_colors = {'MEDIUM': '#2ca02c', 'HARD': '#d62728'}
    for i, d in enumerate(diff_order):
        vals = [r['consensus_frac'] for r in rows if r['difficulty'] == d]
        if vals:
            bp = ax.bar(i, np.mean(vals), color=diff_colors[d], alpha=0.7, width=0.6,
                        edgecolor='black', linewidth=0.8)
            # Jitter individual points
            jitter = np.random.uniform(-0.15, 0.15, len(vals))
            ax.scatter([i] * len(vals) + jitter, vals, color='black', alpha=0.4, s=25, zorder=3)
            ax.text(i, np.mean(vals) + 0.03, f'{np.mean(vals):.0%}',
                    ha='center', fontweight='bold', fontsize=11)
            ax.text(i, -0.08, f'n={len(vals)}', ha='center', fontsize=9, color='gray')

    ax.set_xticks(range(len(diff_order)))
    ax.set_xticklabels(diff_order, fontsize=11)
    ax.set_ylabel('Consensus Fraction')
    ax.set_title('A. Difficulty vs. Consensus', fontweight='bold')
    ax.set_ylim(-0.12, 1.15)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend(fontsize=8, loc='upper right')

    # Panel B: Difficulty vs MAE — use box plots to handle outliers properly
    ax = axes[1]
    mae_by_diff = []
    bp_labels = []
    bp_colors_list = []
    for d in diff_order:
        vals = [r['mae'] for r in rows if r['difficulty'] == d and r['mae'] is not None]
        mae_by_diff.append(vals)
        bp_labels.append(d)
        bp_colors_list.append(diff_colors[d])

    bp = ax.boxplot(mae_by_diff, tick_labels=bp_labels, widths=0.5,
                    patch_artist=True, showfliers=True,
                    flierprops=dict(marker='o', markersize=5, alpha=0.5),
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], bp_colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Add jittered points
    for i, (d, vals) in enumerate(zip(diff_order, mae_by_diff)):
        if vals:
            jitter = np.random.uniform(-0.12, 0.12, len(vals))
            ax.scatter([i + 1] * len(vals) + jitter, vals,
                       color='black', alpha=0.35, s=20, zorder=3)
            med = np.median(vals)
            ax.text(i + 1 + 0.32, med, f'med={med:.1f}%',
                    fontsize=9, va='center', color='black', fontweight='bold')

    ax.set_ylabel('MAE (%)')
    ax.set_title('B. Difficulty vs. MAE', fontweight='bold')
    ax.set_ylim(-1, max(v for vals in mae_by_diff for v in vals) * 1.05)

    # Panel C: Number of challenges vs consensus fraction (scatter)
    ax = axes[2]
    nc = [r['n_challenges'] for r in rows]
    cf = [r['consensus_frac'] for r in rows]
    colors = [diff_colors.get(r['difficulty'], 'gray') for r in rows]

    ax.scatter(nc, cf, c=colors, s=50, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=3)

    # Trend line
    z = np.polyfit(nc, cf, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(nc), max(nc), 100)
    ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=1.5)

    r_val, p_val = sp_stats.pearsonr(nc, cf)
    ax.text(0.95, 0.95, f'r = {r_val:.2f}\np = {p_val:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Number of Detected Challenges')
    ax.set_ylabel('Consensus Fraction')
    ax.set_title('C. Challenge Count vs. Consensus', fontweight='bold')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)

    # Legend for difficulty colors
    legend_elements = [Patch(facecolor=diff_colors['MEDIUM'], label='MEDIUM'),
                       Patch(facecolor=diff_colors['HARD'], label='HARD')]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower left')

    fig.tight_layout()
    path = OUT_DIR / "fig7_consensus_predictors.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Generating paper figures (revised framing)...")
    print(f"Output directory: {OUT_DIR}\n")

    # Load data
    lol_matches = load_loladze_matches()
    lol_report = load_loladze_report()
    hui_matches = load_hui_matches()
    li_matches = load_li2022_matches()
    consensus_stats = load_consensus_stats()
    paper_validation = load_per_paper_validation()

    print(f"Loaded: {len(lol_matches)} Loladze matches, {len(hui_matches)} Hui matches, "
          f"{len(li_matches)} Li matches, {len(consensus_stats)} consensus stats\n")

    # Main paper figures
    print("Main figures:")
    fig1_pipeline_architecture()
    fig2_consensus_reliability(lol_matches, consensus_stats, paper_validation)
    fig3_paper_mae_confidence(lol_report, consensus_stats)
    fig_combined_scatter(lol_matches, hui_matches, li_matches)
    fig7_consensus_predictors(consensus_stats, paper_validation)

    # Supplementary figures
    print("\nSupplementary figures:")
    fig_scatter_loladze(lol_matches)
    fig_element_effects(lol_matches)
    fig_bland_altman_formal(lol_matches)
    fig_error_distribution(lol_matches)

    print(f"\nAll figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
