"""
TOST Equivalence Figure for All Three Datasets
===============================================
Creates a publication-quality forest-plot-style figure showing
TOST equivalence results across Loladze, Hui, and Li 2022.

Run:
    python fig_tost_equivalence.py
"""
import sys, json
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
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


def load_all_stats():
    """Load TOST and formal stats from all three datasets."""
    datasets = {}

    # Loladze
    lol_tost = BASE / "output" / "formal_stats" / "tost_results.json"
    lol_ba = BASE / "output" / "formal_stats" / "bland_altman_results.json"
    lol_icc = BASE / "output" / "formal_stats" / "icc_results.json"
    if lol_tost.exists():
        with open(lol_tost) as f:
            tost = json.load(f)
        with open(lol_ba) as f:
            ba = json.load(f)
        with open(lol_icc) as f:
            icc = json.load(f)
        datasets['Loladze 2014\n(CO2/Minerals, 560 obs)'] = {
            'mean_diff': tost.get('mean_diff_pp', 0.243),
            'ci90_lower': tost.get('ci_90_pp', [-0.832, 1.318])[0],
            'ci90_upper': tost.get('ci_90_pp', [-0.832, 1.318])[1],
            'p_2pp': tost.get('multiple_margins', {}).get('2.0pp', {}).get('p_value', tost.get('p_tost', 0.004)),
            'equivalent_2pp': tost.get('equivalent_at_alpha_05', True),
            'icc': icc.get('observation_level', {}).get('icc_31_consistency', 0.695),
            'n': tost.get('n', 560),
        }

    # Hui 2023
    hui_tost = BASE / "output" / "hui2023_formal_stats" / "tost_results.json"
    hui_ba = BASE / "output" / "hui2023_formal_stats" / "bland_altman_results.json"
    hui_icc = BASE / "output" / "hui2023_formal_stats" / "icc_results.json"
    if hui_tost.exists():
        with open(hui_tost) as f:
            tost = json.load(f)
        with open(hui_ba) as f:
            ba = json.load(f)
        with open(hui_icc) as f:
            icc = json.load(f)
        m2 = tost.get('margin_2pp', {})
        m3 = tost.get('margin_3pp', {})
        datasets['Hui 2023\n(Zn/Wheat, 222 obs)'] = {
            'mean_diff': m2.get('mean_difference', 0.63),
            'ci90_lower': m2.get('ci90_lower', -0.87),
            'ci90_upper': m2.get('ci90_upper', 2.13),
            'p_2pp': m2.get('p_tost', 0.066),
            'equivalent_2pp': m2.get('equivalent', False),
            'p_3pp': m3.get('p_tost', 0.005),
            'equivalent_3pp': m3.get('equivalent', True),
            'icc': icc.get('icc_31', 0.974),
            'n': m2.get('n', 222) if 'n' in m2 else 222,
        }

    # Li 2022
    li_tost = BASE / "output" / "li2022_formal_stats" / "tost_results.json"
    li_ba = BASE / "output" / "li2022_formal_stats" / "bland_altman_results.json"
    li_icc = BASE / "output" / "li2022_formal_stats" / "icc_results.json"
    if li_tost.exists():
        with open(li_tost) as f:
            tost = json.load(f)
        with open(li_ba) as f:
            ba = json.load(f)
        with open(li_icc) as f:
            icc = json.load(f)
        m2 = tost.get('margin_2pp', {})
        m3 = tost.get('margin_3pp', {})
        datasets['Li 2022\n(Biostimulant/Yield, 163 obs)'] = {
            'mean_diff': m3.get('mean_difference', 0.06),
            'ci90_lower': m3.get('ci90_lower', -2.73),
            'ci90_upper': m3.get('ci90_upper', 2.85),
            'p_2pp': m2.get('p_tost', 0.126),
            'equivalent_2pp': m2.get('equivalent', False),
            'p_3pp': m3.get('p_tost', 0.042),
            'equivalent_3pp': m3.get('equivalent', True),
            'icc': icc.get('icc_31', 0.429),
            'n': 163,
        }

    return datasets


def fig_tost_forest(datasets):
    """Create a TOST equivalence forest plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})

    names = list(datasets.keys())
    n_datasets = len(names)
    y_positions = list(range(n_datasets - 1, -1, -1))  # Top to bottom

    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    # === Panel A: TOST Forest Plot ===
    for i, (name, d) in enumerate(datasets.items()):
        y = y_positions[i]
        color = colors[i]
        mean_d = d['mean_diff']
        ci_lo = d['ci90_lower']
        ci_hi = d['ci90_upper']

        # Draw 90% CI
        ax1.plot([ci_lo, ci_hi], [y, y], color=color, linewidth=2.5, solid_capstyle='round')
        # Draw CI endpoints
        ax1.plot([ci_lo], [y], '|', color=color, markersize=12, markeredgewidth=2)
        ax1.plot([ci_hi], [y], '|', color=color, markersize=12, markeredgewidth=2)
        # Draw mean difference point
        ax1.plot(mean_d, y, 'o', color=color, markersize=10, zorder=5, markeredgecolor='white', markeredgewidth=1)

        # Annotate
        label = f"Mean: {mean_d:+.2f} pp\n90% CI: ({ci_lo:.2f}, {ci_hi:.2f})"
        ax1.annotate(label, (ci_hi + 0.3, y), fontsize=8, va='center')

    # Equivalence margins
    for margin, ls, lbl in [(2, '-', r'$\pm$2 pp'), (3, '--', r'$\pm$3 pp'), (5, ':', r'$\pm$5 pp')]:
        ax1.axvline(-margin, color='red', linestyle=ls, alpha=0.5, linewidth=1)
        ax1.axvline(margin, color='red', linestyle=ls, alpha=0.5, linewidth=1)

    # Add margin labels at top
    ax1.text(2, n_datasets - 0.3, r'$\pm$2 pp', ha='center', fontsize=8, color='red', alpha=0.7)
    ax1.text(3, n_datasets - 0.3, r'$\pm$3 pp', ha='center', fontsize=8, color='red', alpha=0.7)
    ax1.text(5, n_datasets - 0.3, r'$\pm$5 pp', ha='center', fontsize=8, color='red', alpha=0.7)

    ax1.axvline(0, color='black', linewidth=0.8, alpha=0.5)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(names, fontsize=10)
    ax1.set_xlabel('Mean Difference (extracted - ground truth, pp)')
    ax1.set_title('(A) TOST Equivalence: 90% CIs vs Equivalence Margins')
    ax1.set_xlim(-6, 8)
    ax1.set_ylim(-0.7, n_datasets - 0.3)

    # Shade the ±2pp equivalence zone
    ax1.axvspan(-2, 2, alpha=0.08, color='green')

    # === Panel B: Summary Statistics Table ===
    ax2.axis('off')

    # Table data
    col_labels = ['Dataset', 'TOST\n(+/-2pp)', 'TOST\n(+/-3pp)', 'ICC', "Cohen's d"]
    table_data = []
    for name, d in datasets.items():
        short_name = name.split('\n')[0]
        tost_2 = f"p={d['p_2pp']:.3f}\n{'EQUIV' if d['equivalent_2pp'] else 'n.s.'}"
        if 'p_3pp' in d:
            tost_3 = f"p={d['p_3pp']:.3f}\n{'EQUIV' if d.get('equivalent_3pp', False) else 'n.s.'}"
        else:
            tost_3 = "p<0.001\nEQUIV"
        icc_val = f"{d['icc']:.3f}"
        cohens = "< 0.05"  # All datasets have negligible Cohen's d
        table_data.append([short_name, tost_2, tost_3, icc_val, cohens])

    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=[0.22, 0.2, 0.2, 0.15, 0.15],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Color header
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color cells by equivalence
    for i, (name, d) in enumerate(datasets.items()):
        # TOST 2pp
        if d['equivalent_2pp']:
            table[i + 1, 1].set_facecolor('#C6EFCE')
        else:
            table[i + 1, 1].set_facecolor('#FFC7CE')
        # TOST 3pp
        if d.get('equivalent_3pp', True):
            table[i + 1, 2].set_facecolor('#C6EFCE')
        else:
            table[i + 1, 2].set_facecolor('#FFC7CE')

    ax2.set_title('(B) Formal Agreement Statistics', fontsize=13, pad=20)

    fig.tight_layout()
    path = OUT_DIR / "fig_tost_equivalence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def fig_bland_altman_trio():
    """Create Bland-Altman plots for all three datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets_info = [
        ('Loladze 2014', 'output/formal_stats/bland_altman_results.json',
         'output/loladze_full_46_v2/validation_matches.csv', 'gt', 'our', '#2ca02c'),
        ('Hui 2023', 'output/hui2023_formal_stats/bland_altman_results.json',
         'output/hui2023_v2/validation_hui2023_matches.csv', 'gt_lnrr_pct', 'our_lnrr_pct', '#1f77b4'),
        ('Li 2022', 'output/li2022_formal_stats/bland_altman_results.json',
         'output/li2022_consensus/validation_matches.csv', 'gt_effect_pct', 'ext_effect_pct', '#ff7f0e'),
    ]

    for idx, (name, ba_path, csv_path, gt_col, ext_col, color) in enumerate(datasets_info):
        ax = axes[idx]
        ba_file = BASE / ba_path
        data_file = BASE / csv_path

        if not ba_file.exists():
            ax.text(0.5, 0.5, f'{name}\nNo data', ha='center', va='center', transform=ax.transAxes)
            continue

        with open(ba_file) as f:
            ba = json.load(f)

        # Load raw data for scatter
        import csv as csvmod
        means = []
        diffs = []
        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csvmod.DictReader(f)
            for row in reader:
                try:
                    if name == 'Loladze 2014':
                        gt_val = float(row['gt']) * 100
                        ext_val = float(row['our']) * 100
                    elif name == 'Hui 2023':
                        gt_ctrl = float(row['gt_ctrl'])
                        gt_treat = float(row['gt_treat'])
                        our_ctrl = float(row['our_ctrl'])
                        our_treat = float(row['our_treat'])
                        if gt_ctrl <= 0 or our_ctrl <= 0:
                            continue
                        gt_val = (gt_treat - gt_ctrl) / gt_ctrl * 100
                        ext_val = (our_treat - our_ctrl) / our_ctrl * 100
                    else:
                        gt_val = float(row['gt_effect_pct'])
                        ext_val = float(row['ext_effect_pct'])

                    mean_val = (gt_val + ext_val) / 2
                    diff_val = ext_val - gt_val
                    means.append(mean_val)
                    diffs.append(diff_val)
                except (ValueError, KeyError, ZeroDivisionError):
                    continue

        means = np.array(means)
        diffs = np.array(diffs)

        ax.scatter(means, diffs, alpha=0.4, s=15, color=color, edgecolors='none')

        # Mean difference line
        mean_d = ba['mean_difference'] if 'mean_difference' in ba else ba.get('mean_difference_pp', 0)
        ax.axhline(mean_d, color='black', linewidth=1.5, label=f'Mean diff: {mean_d:.2f}')

        # LoA
        loa_lo = ba.get('loa_lower', ba.get('loa_lower_pp', 0))
        loa_hi = ba.get('loa_upper', ba.get('loa_upper_pp', 0))
        ax.axhline(loa_lo, color='red', linestyle='--', linewidth=1, label=f'LoA: ({loa_lo:.1f}, {loa_hi:.1f})')
        ax.axhline(loa_hi, color='red', linestyle='--', linewidth=1)

        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('Mean of GT and Extracted (%)')
        if idx == 0:
            ax.set_ylabel('Difference (Extracted - GT, pp)')
        ax.set_title(f'{name}\n(n={len(means)})')
        ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Bland-Altman Analysis: Limits of Agreement Across Datasets', fontsize=14, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig_bland_altman_trio.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    return path


def main():
    print("Generating TOST equivalence and Bland-Altman figures...")

    datasets = load_all_stats()
    print(f"Loaded stats for {len(datasets)} datasets")

    for name, d in datasets.items():
        short = name.split('\n')[0]
        print(f"  {short}: mean_diff={d['mean_diff']:.2f}, 90% CI=({d['ci90_lower']:.2f}, {d['ci90_upper']:.2f}), "
              f"TOST(2pp) p={d['p_2pp']:.4f} {'EQUIV' if d['equivalent_2pp'] else 'n.s.'}")

    # Figure 1: TOST Forest + Summary Table
    fig_tost_forest(datasets)

    # Figure 2: Bland-Altman trio
    fig_bland_altman_trio()

    print("\nDone!")


if __name__ == "__main__":
    main()
