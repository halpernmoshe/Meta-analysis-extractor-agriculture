"""
Generate publication-quality figures for the AGENT-focused paper (v11).

Figures (matching paper body order):
  1. Agent GT Validation Scatter (3-panel)
  2. Per-Paper MAE Summary (horizontal bar chart)
  3. Bland-Altman (3-panel)
  4. Agent-Pipeline Agreement Scatter (3-panel, GT-free)
  5. Error Taxonomy (stacked bar)

Usage:
    ./venv/Scripts/python.exe generate_agent_figures.py
"""

import os
import sys
import json
import math
import re
import warnings
warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor"
os.chdir(BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "output", "paper_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "default"]:
    try:
        plt.style.use(style)
        print(f"Using style: {style}")
        break
    except Exception:
        continue

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Dataset colors
COLOR_LOLADZE = '#2166AC'  # blue
COLOR_HUI     = '#1B7837'  # green
COLOR_LI      = '#E66101'  # orange

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def safe_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return None


def pearson_r(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan
    return np.corrcoef(x, y)[0, 1]


def icc_2_1(x, y):
    """Compute ICC(2,1) for two raters."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    grand = (x + y) / 2
    gm = np.mean(grand)
    ms_between = np.sum((grand - gm) ** 2) * 2 / (n - 1)
    residuals = np.concatenate([x - grand, y - grand])
    ms_within = np.sum(residuals ** 2) / n
    if (ms_between + ms_within) == 0:
        return np.nan
    return (ms_between - ms_within) / (ms_between + ms_within)


# ---------------------------------------------------------------------------
# DATA LOADING: Loladze agent validation (per-obs from validation report)
# ---------------------------------------------------------------------------
def load_loladze_agent_obs():
    """Load Loladze per-obs matched pairs from validation report."""
    path = os.path.join(BASE_DIR, "output", "agent_extraction", "validation_report_agent.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_matches = data.get('all_matches', [])
    gt_vals = []
    agent_vals = []
    papers = []
    for m in all_matches:
        # 'our' and 'gt' are fractional effect sizes (e.g., -0.10 = -10%)
        gt_vals.append(m['gt'] * 100)
        agent_vals.append(m['our'] * 100)
        papers.append(m.get('paper', ''))

    print(f"  Loladze: {len(gt_vals)} matched obs loaded from all_matches")
    return np.array(gt_vals), np.array(agent_vals), papers


# ---------------------------------------------------------------------------
# DATA LOADING: Hui agent validation (per-obs from validation report)
# ---------------------------------------------------------------------------
def load_hui_agent_obs():
    """Load Hui per-obs matched pairs from validation report match_pairs."""
    report_path = os.path.join(BASE_DIR, "output", "hui2023_agent_extraction",
                               "validation_report_agent.json")
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # The Hui validation report HAS match_pairs with per-obs data
    match_pairs = report.get('match_pairs', [])
    if not match_pairs:
        print("  Hui: WARNING - no match_pairs in validation report")
        return np.array([]), np.array([]), []

    gt_vals = []
    agent_vals = []
    papers = []
    for m in match_pairs:
        gt_pct = m.get('gt_pct')
        our_pct = m.get('our_pct')
        if gt_pct is not None and our_pct is not None:
            gt_vals.append(gt_pct)
            agent_vals.append(our_pct)
            papers.append(m.get('paper', ''))

    print(f"  Hui: {len(gt_vals)} matched obs loaded from match_pairs")
    return np.array(gt_vals), np.array(agent_vals), papers


# ---------------------------------------------------------------------------
# DATA LOADING: Li agent validation (HIGH-CONFIDENCE only from harmonized)
# ---------------------------------------------------------------------------
def load_li_agent_obs():
    """Load Li HIGH-CONFIDENCE per-obs matched pairs from harmonized report."""
    harm_path = os.path.join(BASE_DIR, "output", "li2022_agent_extraction",
                             "harmonized_validation_agent.json")
    with open(harm_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Use only the HIGH tier (N=68, r=0.968) as per paper claims
    high_pairs = data.get('match_pairs_by_tier', {}).get('high', [])

    gt_vals = []
    agent_vals = []
    papers = []
    for m in high_pairs:
        gt_eff = m.get('gt_effect')
        ext_eff = m.get('ext_effect')
        if gt_eff is not None and ext_eff is not None:
            gt_vals.append(gt_eff)
            agent_vals.append(ext_eff)
            papers.append(m.get('paper', ''))

    print(f"  Li (high-confidence): {len(gt_vals)} matched obs loaded")
    return np.array(gt_vals), np.array(agent_vals), papers


# ---------------------------------------------------------------------------
# DATA LOADING: Agent-Pipeline Agreement (regenerate per-obs)
# ---------------------------------------------------------------------------
def load_agreement_obs(dataset_key):
    """Regenerate agent-pipeline per-obs matched pairs for a dataset.

    Uses the EXACT same matching logic as agent_pipeline_agreement.py to ensure
    the N values match the stored results (Loladze=1205, Hui=185, Li=499).
    """
    from pathlib import Path

    DATASETS = {
        'loladze': {
            'pipeline_dir': Path(os.path.join(BASE_DIR, "output", "loladze_v3_combined")),
            'agent_dir': Path(os.path.join(BASE_DIR, "output", "agent_extraction")),
            'pipeline_glob': '*_consensus.json',
            'agent_glob': '*_agent*.json',
            'filter_fn': None,
        },
        'hui2023': {
            'pipeline_dir': Path(os.path.join(BASE_DIR, "output", "hui2023_full_35")),
            'agent_dir': Path(os.path.join(BASE_DIR, "output", "hui2023_agent_extraction")),
            'pipeline_glob': '*_consensus.json',
            'agent_glob': '*_agent*.json',
            'filter_fn': lambda obs: 'ZN' in str(obs.get('element', '')).upper() or 'ZINC' in str(obs.get('element', '')).upper(),
        },
        'li2022': {
            'pipeline_dir': Path(os.path.join(BASE_DIR, "output", "li2022_combined")),
            'agent_dir': Path(os.path.join(BASE_DIR, "output", "li2022_agent_extraction")),
            'pipeline_glob': '*_consensus.json',
            'agent_glob': '*_agent*.json',
            'filter_fn': None,
        },
    }

    SCALE_FACTORS = [1, 10, 100, 1000, 0.1, 0.01, 0.001]

    cfg = DATASETS.get(dataset_key)
    if not cfg:
        return np.array([]), np.array([])

    pipe_dir = cfg['pipeline_dir']
    agent_dir = cfg['agent_dir']

    if not pipe_dir.exists() or not agent_dir.exists():
        print(f"  {dataset_key}: directory not found, skipping")
        return np.array([]), np.array([])

    def load_obs(json_path, filter_fn=None):
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
            result.append({'control_mean': ctrl, 'treatment_mean': treat,
                           'effect_pct': (treat - ctrl) / ctrl * 100,
                           'element': obs.get('element', obs.get('outcome', '')),
                           'tissue': obs.get('tissue', '')})
        return result

    def extract_paper_id(filepath, is_agent=False):
        stem = filepath.stem
        if is_agent:
            stem = stem.replace('_agent_v2', '').replace('_agent', '')
        else:
            stem = stem.replace('_consensus', '')
        return stem.lower().strip()

    # Build file maps (same as original script)
    pipe_files = sorted(pipe_dir.glob(cfg['pipeline_glob']))
    agent_files = sorted(agent_dir.glob(cfg['agent_glob']))
    agent_files = [f for f in agent_files if 'validation_report' not in f.name and 'harmonized' not in f.name]

    # Match paper IDs using the EXACT same algorithm as agent_pipeline_agreement.py
    pipe_map = {}
    for f in pipe_files:
        pid = extract_paper_id(f, is_agent=False)
        pipe_map[pid] = f

    agent_map = {}
    for f in agent_files:
        aid = extract_paper_id(f, is_agent=True)
        if '_v2' in f.stem or aid not in agent_map:
            agent_map[aid] = f

    matched_papers = []
    used_agent = set()

    for pid, pfile in pipe_map.items():
        best_match = None
        best_score = 0

        for aid, afile in agent_map.items():
            if aid in used_agent:
                continue

            # Direct match
            if pid == aid:
                best_match = (aid, afile)
                best_score = 1000
                break

            # Numeric prefix match
            p_match = re.match(r'(\d+)_(.+)', pid)
            a_match = re.match(r'(\d+)_(.+)', aid)

            if p_match and a_match:
                p_num, p_rest = p_match.groups()
                a_num, a_rest = a_match.groups()
                if p_num == a_num:
                    score = 100
                    if best_score < score:
                        best_score = score
                        best_match = (aid, afile)
                    continue

            # Author name overlap
            p_parts = set(pid.replace('_', ' ').split())
            a_parts = set(aid.replace('_', ' ').split())
            overlap = len(p_parts & a_parts)
            if overlap >= 2 and overlap > best_score:
                best_score = overlap
                best_match = (aid, afile)

        if best_match:
            used_agent.add(best_match[0])
            matched_papers.append((pid, pfile, best_match[0], best_match[1]))

    # Now match observations within matched papers (tolerance=0.25 as original)
    pipe_effects = []
    agent_effects = []

    filter_fn = cfg.get('filter_fn')
    for pid, pfile, aid, afile in matched_papers:
        p_obs = load_obs(pfile, filter_fn)
        a_obs = load_obs(afile, filter_fn)
        if not p_obs or not a_obs:
            continue

        used_agent_obs = set()
        for p in p_obs:
            best = None
            best_err = float('inf')
            for i, a in enumerate(a_obs):
                if i in used_agent_obs:
                    continue
                for s in SCALE_FACTORS:
                    c_err = abs(p['control_mean'] * s - a['control_mean']) / max(abs(a['control_mean']), 0.001)
                    t_err = abs(p['treatment_mean'] * s - a['treatment_mean']) / max(abs(a['treatment_mean']), 0.001)
                    err = (c_err + t_err) / 2
                    if err < best_err and err < 0.25:
                        best_err = err
                        best = (i, a)
            if best:
                used_agent_obs.add(best[0])
                pipe_effects.append(p['effect_pct'])
                agent_effects.append(best[1]['effect_pct'])

    print(f"  {dataset_key} agreement: {len(pipe_effects)} matched obs")
    return np.array(pipe_effects), np.array(agent_effects)


# ---------------------------------------------------------------------------
# FIGURE 1: Agent GT Validation Scatter (3-panel)
# ---------------------------------------------------------------------------
def figure1_gt_scatter(lol_gt, lol_ag, hui_gt, hui_ag, li_gt, li_ag):
    """Three-panel scatter plot: agent vs GT effect sizes."""
    print("\nGenerating Figure 1: Agent GT Validation Scatter...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    datasets = [
        ('A', 'Loladze 2014 (CO2/minerals)', lol_gt, lol_ag, COLOR_LOLADZE),
        ('B', 'Hui 2023 (Zn/wheat)', hui_gt, hui_ag, COLOR_HUI),
        ('C', 'Li 2022 (biostimulants/yield)', li_gt, li_ag, COLOR_LI),
    ]

    print("\n  --- Figure 1 Panel Stats ---")
    for ax, (label, name, gt, ag, color) in zip(axes, datasets):
        n = len(gt)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'({label}) {name}')
            print(f"  Panel {label}: NO DATA")
            continue

        ax.scatter(gt, ag, alpha=0.35, s=18, color=color, edgecolors='none', rasterized=True)

        # Identity line
        lo = min(gt.min(), ag.min()) * 1.1
        hi = max(gt.max(), ag.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label='Identity')

        # Stats
        r = pearson_r(gt, ag)
        icc = icc_2_1(gt, ag)
        mae = np.mean(np.abs(gt - ag))

        ax.set_title(f'({label}) {name}', fontweight='bold')
        ax.set_xlabel('Ground truth effect (%)')
        ax.set_ylabel('Agent effect (%)')

        # Annotation box
        stats_text = f'N = {n}\nr = {r:.3f}\nICC = {icc:.3f}\nMAE = {mae:.1f}pp'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

        ax.set_aspect('equal', adjustable='datalim')

        print(f"  Panel {label} ({name}): N={n}, r={r:.3f}, ICC={icc:.3f}, MAE={mae:.1f}pp")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_agent_gt_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# FIGURE 2: Per-Paper MAE Summary (was fig4, now fig2 per paper body order)
# ---------------------------------------------------------------------------
def figure2_per_paper_mae():
    """Horizontal bar chart of per-paper MAE across all 3 datasets."""
    print("\nGenerating Figure 2: Per-Paper MAE Summary...")

    papers = []

    # Loladze
    lol_path = os.path.join(BASE_DIR, "output", "agent_extraction", "validation_report_agent.json")
    with open(lol_path, 'r', encoding='utf-8') as f:
        lol_report = json.load(f)
    for p in lol_report.get('per_paper', []):
        if p.get('matched', 0) > 0 and p.get('mae') is not None:
            papers.append({
                'label': p['paper_id'].split('_', 1)[-1] if '_' in p['paper_id'] else p['paper_id'],
                'mae': p['mae'],
                'dataset': 'Loladze',
                'color': COLOR_LOLADZE,
            })

    # Hui
    hui_path = os.path.join(BASE_DIR, "output", "hui2023_agent_extraction", "validation_report_agent.json")
    with open(hui_path, 'r', encoding='utf-8') as f:
        hui_report = json.load(f)
    for p in hui_report.get('paper_results', []):
        if p.get('matched', 0) > 0 and p.get('mae_pct') is not None:
            papers.append({
                'label': p['paper_id'],
                'mae': p['mae_pct'],
                'dataset': 'Hui',
                'color': COLOR_HUI,
            })

    # Li (only high-confidence papers from harmonized report)
    harm_path = os.path.join(BASE_DIR, "output", "li2022_agent_extraction",
                             "harmonized_validation_agent.json")
    with open(harm_path, 'r', encoding='utf-8') as f:
        harm_data = json.load(f)
    high_papers = set()
    for pair in harm_data.get('match_pairs_by_tier', {}).get('high', []):
        high_papers.add(pair.get('paper', ''))

    li_path = os.path.join(BASE_DIR, "output", "li2022_agent_extraction", "validation_report_agent.json")
    with open(li_path, 'r', encoding='utf-8') as f:
        li_report = json.load(f)
    for p in li_report.get('paper_results', []):
        pid = p.get('paper_id', '')
        if p.get('matched', 0) > 0 and p.get('mae_pp') is not None:
            # Only include papers that appear in high-confidence tier
            if pid in high_papers:
                papers.append({
                    'label': pid,
                    'mae': p['mae_pp'],
                    'dataset': 'Li',
                    'color': COLOR_LI,
                })

    if not papers:
        print("  No paper data found for Figure 2!")
        return None

    # Sort by MAE (best to worst)
    papers.sort(key=lambda x: x['mae'])
    total_papers = len(papers)

    # Limit display: show top 20 best + bottom 10 worst with gap indicator
    TOP_N = 20
    BOTTOM_N = 10
    if total_papers > TOP_N + BOTTOM_N + 2:
        top_papers = papers[:TOP_N]
        bottom_papers = papers[-BOTTOM_N:]
        n_hidden = total_papers - TOP_N - BOTTOM_N
        display_papers = top_papers + [{'label': f'... {n_hidden} papers omitted ...',
                                         'mae': 0, 'color': '#FFFFFF', 'dataset': 'gap'}] + bottom_papers
    else:
        display_papers = papers

    # Truncate labels for readability
    for p in display_papers:
        if p['dataset'] != 'gap' and len(p['label']) > 28:
            p['label'] = p['label'][:25] + '...'

    n = len(display_papers)
    fig_height = max(6, n * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n)
    colors = [p['color'] for p in display_papers]
    maes = [p['mae'] for p in display_papers]
    labels = [p['label'] for p in display_papers]

    bars = ax.barh(y_pos, maes, color=colors, alpha=0.8, height=0.7, edgecolor='none')

    # Style the gap row
    for i, p in enumerate(display_papers):
        if p['dataset'] == 'gap':
            bars[i].set_visible(False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    for i, p in enumerate(display_papers):
        if p['dataset'] == 'gap':
            ax.get_yticklabels()[i].set_fontstyle('italic')
            ax.get_yticklabels()[i].set_color('gray')

    ax.set_xlabel('MAE (percentage points)')
    ax.set_title(f'Per-Paper Mean Absolute Error (Agent Extraction, {total_papers} papers total)',
                 fontweight='bold')
    ax.invert_yaxis()

    # Vertical guideline at 5pp
    ax.axvline(5, color='gray', linestyle='--', lw=0.8, alpha=0.5)
    ax.text(5.2, 0.3, '5pp', fontsize=8, color='gray')

    # Legend
    handles = [
        mpatches.Patch(color=COLOR_LOLADZE, label='Loladze 2014'),
        mpatches.Patch(color=COLOR_HUI, label='Hui 2023'),
        mpatches.Patch(color=COLOR_LI, label='Li 2022'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_per_paper_mae.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path} ({total_papers} papers)")
    return path


# ---------------------------------------------------------------------------
# FIGURE 3: Bland-Altman (3-panel) -- was fig2, now fig3 per paper body order
# ---------------------------------------------------------------------------
def figure3_bland_altman(lol_gt, lol_ag, hui_gt, hui_ag, li_gt, li_ag):
    """Three-panel Bland-Altman plots."""
    print("\nGenerating Figure 3: Bland-Altman plots...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    datasets = [
        ('A', 'Loladze 2014', lol_gt, lol_ag, COLOR_LOLADZE),
        ('B', 'Hui 2023', hui_gt, hui_ag, COLOR_HUI),
        ('C', 'Li 2022', li_gt, li_ag, COLOR_LI),
    ]

    print("\n  --- Figure 3 Panel Stats ---")
    for ax, (label, name, gt, ag, color) in zip(axes, datasets):
        n = len(gt)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'({label}) {name}')
            print(f"  Panel {label}: NO DATA")
            continue

        mean_val = (gt + ag) / 2
        diff_val = ag - gt  # Agent minus GT

        mean_diff = np.mean(diff_val)
        sd_diff = np.std(diff_val, ddof=1)
        loa_upper = mean_diff + 1.96 * sd_diff
        loa_lower = mean_diff - 1.96 * sd_diff

        ax.scatter(mean_val, diff_val, alpha=0.3, s=18, color=color, edgecolors='none', rasterized=True)

        # Reference lines
        ax.axhline(mean_diff, color='#333333', linestyle='--', lw=1.2, label=f'Mean diff = {mean_diff:.1f}')
        ax.axhline(loa_upper, color='#999999', linestyle=':', lw=1, label=f'+1.96 SD = {loa_upper:.1f}')
        ax.axhline(loa_lower, color='#999999', linestyle=':', lw=1, label=f'-1.96 SD = {loa_lower:.1f}')
        ax.axhline(0, color='black', lw=0.5, alpha=0.3)

        ax.set_title(f'({label}) {name}', fontweight='bold')
        ax.set_xlabel('Mean of GT and Agent (%)')
        ax.set_ylabel('Agent - GT (%)')

        stats_text = (f'Mean diff = {mean_diff:.2f}pp\n'
                      f'LOA: [{loa_lower:.1f}, {loa_upper:.1f}]\n'
                      f'N = {n}')
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

        print(f"  Panel {label} ({name}): N={n}, mean_diff={mean_diff:.2f}pp, LOA=[{loa_lower:.1f}, {loa_upper:.1f}]")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_bland_altman.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# FIGURE 4: Agent-Pipeline Agreement Scatter (3-panel, GT-free)
# -- was fig3, now fig4 per paper body order
# ---------------------------------------------------------------------------
def figure4_agreement_scatter(lol_pipe, lol_agent, hui_pipe, hui_agent, li_pipe, li_agent):
    """Three-panel scatter: agent vs pipeline (no GT)."""
    print("\nGenerating Figure 4: Agent-Pipeline Agreement Scatter...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    datasets = [
        ('A', 'Loladze 2014', lol_pipe, lol_agent, COLOR_LOLADZE),
        ('B', 'Hui 2023', hui_pipe, hui_agent, COLOR_HUI),
        ('C', 'Li 2022', li_pipe, li_agent, COLOR_LI),
    ]

    print("\n  --- Figure 4 Panel Stats ---")
    for ax, (label, name, pipe, agent, color) in zip(axes, datasets):
        n = len(pipe)
        if n == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'({label}) {name}')
            print(f"  Panel {label}: NO DATA")
            continue

        # Clip extreme outliers for axis limits (use 1st/99th percentile)
        all_vals = np.concatenate([pipe, agent])
        p1, p99 = np.percentile(all_vals, [1, 99])
        margin = (p99 - p1) * 0.1
        clip_lo = p1 - margin
        clip_hi = p99 + margin
        n_outliers = int(np.sum((pipe < clip_lo) | (pipe > clip_hi) |
                                (agent < clip_lo) | (agent > clip_hi)))

        ax.scatter(pipe, agent, alpha=0.3, s=18, color=color, edgecolors='none', rasterized=True)

        # Identity line
        lo = clip_lo
        hi = clip_hi
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6)

        # Set axis limits to clipped range
        ax.set_xlim(clip_lo, clip_hi)
        ax.set_ylim(clip_lo, clip_hi)

        r = pearson_r(pipe, agent)
        mae = np.mean(np.abs(pipe - agent))

        ax.set_title(f'({label}) {name}', fontweight='bold')
        ax.set_xlabel('Pipeline effect (%)')
        ax.set_ylabel('Agent effect (%)')

        stats_text = f'N = {n}\nr = {r:.3f}\nMAE = {mae:.1f}pp'
        if n_outliers > 0:
            stats_text += f'\n({n_outliers} obs beyond axis)'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'))

        ax.set_aspect('equal', adjustable='datalim')

        print(f"  Panel {label} ({name}): N={n}, r={r:.3f}, MAE={mae:.1f}pp")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_gt_free_agreement.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# FIGURE 5: Error Taxonomy
# ---------------------------------------------------------------------------
def figure5_error_taxonomy():
    """Stacked bar showing error source breakdown."""
    print("\nGenerating Figure 5: Error Taxonomy...")

    # Based on 121 diagnosable discrepancies from Loladze dataset:
    # Alignment ambiguity = 113 (93%), True reading errors = 4 (3%), Undiagnosable = 4 (3%)
    # Note: denominator is 121 diagnosable discrepancies, NOT all 655 obs
    categories = ['Alignment\nambiguity', 'True reading\nerror', 'Undiagnosable']
    counts = [113, 4, 4]
    total_discrepancies = sum(counts)
    values = [round(c / total_discrepancies * 100) for c in counts]
    # Fix rounding: 113/121=93.4% -> 93%, 4/121=3.3% -> 3%, 4/121=3.3% -> 3%
    # 93+3+3=99% due to rounding -- this is standard; do NOT adjust to 94%
    # Paper text says 93%, so figure must match
    values = [93, 3, 3]
    colors_tax = ['#4393C3', '#D6604D', '#BABABA']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [1.2, 1]})

    # Panel A: Horizontal stacked bar
    ax = axes[0]
    left = 0
    for cat, val, cnt, col in zip(categories, values, counts, colors_tax):
        ax.barh(['Error sources'], val, left=left, color=col, edgecolor='white', height=0.5,
                label=f'{cat.replace(chr(10), " ")} ({val}%, n={cnt})')
        if val > 8:
            ax.text(left + val / 2, 0, f'{val}%', ha='center', va='center', fontsize=11, fontweight='bold')
        else:
            # Place label above the thin bar segment
            ax.annotate(f'{val}%', xy=(left + val / 2, 0), xytext=(left + val / 2, 0.35),
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
        left += val
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage of error budget')
    ax.set_title('(A) Error source decomposition', fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)

    # Panel B: Explanation text as annotation
    ax2 = axes[1]
    ax2.axis('off')
    explanation = (
        f"Error taxonomy for agent extraction\n"
        f"(Loladze dataset, {total_discrepancies} diagnosable\n"
        f" discrepancies out of 655 obs)\n\n"
        f"Alignment ambiguity ({values[0]}%, n={counts[0]})\n"
        f"  Agent extracts correct values from a\n"
        f"  different subset of the data than GT\n"
        f"  (e.g., different cultivar, year, tissue).\n\n"
        f"True reading error ({values[1]}%, n={counts[1]})\n"
        f"  Agent misreads a numeric value from\n"
        f"  the source table or figure.\n\n"
        f"Undiagnosable ({values[2]}%, n={counts[2]})\n"
        f"  Error source cannot be determined\n"
        f"  without manual PDF inspection."
    )
    ax2.text(0.05, 0.95, explanation, transform=ax2.transAxes,
             fontsize=9.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='gray'))
    ax2.set_title('(B) Category definitions', fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_error_taxonomy.png")
    fig.savefig(path)
    plt.close(fig)

    print(f"  Error taxonomy: {values[0]}% / {values[1]}% / {values[2]}% "
          f"(n={counts[0]}/{counts[1]}/{counts[2]}, total={total_discrepancies})")
    print(f"  Saved: {path}")
    return path


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("AGENT FIGURE GENERATION (v11 Paper) - CORRECTED")
    print("=" * 70)
    print()

    # --- Load GT validation data ---
    print("Loading per-observation data for GT validation...")
    lol_gt, lol_ag, _ = load_loladze_agent_obs()
    hui_gt, hui_ag, _ = load_hui_agent_obs()
    li_gt, li_ag, _ = load_li_agent_obs()

    # --- Load agent-pipeline agreement data ---
    print("\nLoading agent-pipeline agreement data...")
    lol_pipe, lol_ag_agree = load_agreement_obs('loladze')
    hui_pipe, hui_ag_agree = load_agreement_obs('hui2023')
    li_pipe, li_ag_agree = load_agreement_obs('li2022')

    # --- Generate figures ---
    paths = []
    paths.append(figure1_gt_scatter(lol_gt, lol_ag, hui_gt, hui_ag, li_gt, li_ag))
    paths.append(figure2_per_paper_mae())
    paths.append(figure3_bland_altman(lol_gt, lol_ag, hui_gt, hui_ag, li_gt, li_ag))
    paths.append(figure4_agreement_scatter(lol_pipe, lol_ag_agree, hui_pipe, hui_ag_agree, li_pipe, li_ag_agree))
    paths.append(figure5_error_taxonomy())

    # --- Verification Summary ---
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    # Print expected vs actual for paper claims
    print("\n  Figure 1 expected: Loladze N=655 r=0.848 | Hui N=461 r=0.942 | Li N=68 r=0.968")
    print("  Figure 3 expected: Li mean_diff=+0.22 LOA=[-6.5,+7.0] | Hui mean_diff=+0.27")
    print("  Figure 4 expected: Loladze N=1205 | Hui N=185 | Li N=499")
    print("  Figure 5 expected: 93%/3%/3% (121 discrepancies)")

    print("\n" + "=" * 70)
    print("FILE SUMMARY")
    print("=" * 70)
    for p in paths:
        if p and os.path.exists(p):
            size_kb = os.path.getsize(p) / 1024
            print(f"  {os.path.basename(p):45s} {size_kb:7.1f} KB")
        elif p:
            print(f"  {os.path.basename(p):45s} NOT GENERATED")

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
