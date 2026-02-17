"""Generate Figure 1: Pipeline Architecture Diagram."""
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\paper_figures")
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Color scheme
C_INPUT = '#E8F4FD'
C_RECON = '#FFF3CD'
C_EXTRACT = '#D4EDDA'
C_CONSENSUS = '#CCE5FF'
C_OUTPUT = '#F8D7DA'
C_TIEBREAK = '#E2D5F1'
C_BORDER = '#333333'
C_ARROW = '#555555'

def box(x, y, w, h, text, color, fontsize=9, bold=False, alpha=1.0):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.2, alpha=alpha)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, wrap=True,
            linespacing=1.3)

def arrow(x1, y1, x2, y2, style='->', color=C_ARROW, lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def label(x, y, text, fontsize=8, color='#666', ha='center'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color, style='italic')

# Title
ax.text(7, 8.7, 'Figure 1: Multi-Model Consensus Pipeline Architecture',
        ha='center', va='center', fontsize=13, fontweight='bold')

# === STAGE 0: INPUT ===
box(0.3, 7.3, 2.2, 0.9, 'PDF Papers\n+ Config JSON', C_INPUT, fontsize=10, bold=True)
label(1.4, 7.05, 'Input', fontsize=8)

# Arrow to recon
arrow(2.5, 7.75, 3.3, 7.75)

# === STAGE 1: RECONNAISSANCE ===
box(3.3, 7.0, 3.0, 1.5, '', C_RECON)
ax.text(4.8, 8.25, 'Stage 1: Reconnaissance', ha='center', fontsize=10, fontweight='bold')
ax.text(4.8, 7.85, 'Claude Sonnet 4', ha='center', fontsize=8, color='#666')
ax.text(4.8, 7.5, 'Variance detection\nTable identification\nChallenge classification',
        ha='center', fontsize=7.5, linespacing=1.4)

# Routing decision diamond
diamond_x, diamond_y = 7.5, 7.75
diamond_size = 0.55
diamond = plt.Polygon([
    [diamond_x, diamond_y + diamond_size],
    [diamond_x + diamond_size, diamond_y],
    [diamond_x, diamond_y - diamond_size],
    [diamond_x - diamond_size, diamond_y],
], facecolor=C_RECON, edgecolor=C_BORDER, linewidth=1.2)
ax.add_patch(diamond)
ax.text(diamond_x, diamond_y, 'Route', ha='center', va='center', fontsize=8, fontweight='bold')

arrow(6.3, 7.75, diamond_x - diamond_size, 7.75)

# Three routing branches
# TEXT branch
arrow(diamond_x + diamond_size, 7.75, 9.5, 7.75, color='#2ca02c')
label(8.8, 8.0, 'TEXT', fontsize=8, color='#2ca02c')

# HYBRID branch
arrow(diamond_x, diamond_y - diamond_size, diamond_x, 6.2, color='#1f77b4')
label(diamond_x + 0.4, 6.7, 'HYBRID', fontsize=8, color='#1f77b4')

# VISION branch
arrow(diamond_x + diamond_size, 7.75, 9.5, 6.6, color='#d62728')
label(9.0, 7.05, 'VISION', fontsize=8, color='#d62728')

# === STAGE 2: DUAL EXTRACTION ===
# Claude box
box(9.5, 7.2, 2.0, 1.1, 'Claude\nSonnet 4\n(Text)', C_EXTRACT, fontsize=9, bold=False)
ax.text(10.5, 8.1, 'Model A', ha='center', fontsize=7, color='#666')

# Kimi box
box(11.8, 7.2, 2.0, 1.1, 'Kimi\nK2.5\n(Text)', C_EXTRACT, fontsize=9, bold=False)
ax.text(12.8, 8.1, 'Model B', ha='center', fontsize=7, color='#666')

# Gemini Vision box (for HYBRID)
box(9.5, 5.8, 2.0, 1.0, 'Gemini 2.5\nFlash\n(Vision)', C_EXTRACT, fontsize=9)
ax.text(10.5, 6.6, 'Vision', ha='center', fontsize=7, color='#666')

# Stage 2 label
ax.text(11.65, 8.55, 'Stage 2: Dual-Model Extraction', ha='center', fontsize=10, fontweight='bold')

# Arrows down from extractors to consensus
arrow(10.5, 7.2, 8.0, 5.0)
arrow(12.8, 7.2, 8.5, 5.0)
arrow(10.5, 5.8, 8.0, 4.6)

# === STAGE 3: CONSENSUS ===
box(5.5, 3.8, 5.5, 1.5, '', C_CONSENSUS)
ax.text(8.25, 5.1, 'Stage 3: Consensus Building', ha='center', fontsize=10, fontweight='bold')
ax.text(8.25, 4.7, 'Element-tissue matching + value tolerance (15%)', ha='center', fontsize=8)
ax.text(8.25, 4.3, 'Both agree: HIGH confidence', ha='center', fontsize=8, color='#2ca02c')
ax.text(8.25, 3.95, 'Disagree: invoke tiebreaker', ha='center', fontsize=8, color='#d62728')

# Tiebreaker path
box(1.0, 3.0, 3.5, 1.5, '', C_TIEBREAK)
ax.text(2.75, 4.25, 'Tiebreaker', ha='center', fontsize=10, fontweight='bold')
ax.text(2.75, 3.85, 'Gemini 2.5 Flash (Text)', ha='center', fontsize=8, color='#666')
ax.text(2.75, 3.45, '2-of-3 voting:\nClaude-Gemini, Kimi-Gemini', ha='center', fontsize=7.5, linespacing=1.3)

# Arrow from consensus to tiebreaker
arrow(5.5, 4.0, 4.5, 3.75, color='#d62728')
label(5.0, 3.55, '<30% match\nor 0 obs', fontsize=7, color='#d62728')

# Arrow back from tiebreaker to consensus
arrow(4.5, 4.25, 5.5, 4.5, color='#2ca02c')
label(5.0, 4.6, 'Recovered obs', fontsize=7, color='#2ca02c')

# === STAGE 4: POST-PROCESSING ===
box(5.5, 1.8, 5.5, 1.2, '', C_OUTPUT)
ax.text(8.25, 2.75, 'Stage 4: Post-Processing', ha='center', fontsize=10, fontweight='bold')
ax.text(8.25, 2.35, 'Dedup | Null filter | T/C swap flag', ha='center', fontsize=8)
ax.text(8.25, 2.0, 'Confidence: HIGH (2+ models) | MEDIUM (single)', ha='center', fontsize=8)

arrow(8.25, 3.8, 8.25, 3.0)

# === OUTPUT ===
box(5.5, 0.3, 5.5, 1.0, 'Consensus Observations\n(JSON + CSV)', C_OUTPUT, fontsize=10, bold=True)
label(8.25, 0.1, 'Output', fontsize=8)

arrow(8.25, 1.8, 8.25, 1.3)

# Legend box
legend_x, legend_y = 0.3, 0.3
box(legend_x, legend_y, 4.5, 2.3, '', '#F5F5F5', alpha=0.9)
ax.text(legend_x + 0.15, legend_y + 2.05, 'Pipeline Statistics (34 papers):', fontsize=8, fontweight='bold')
stats = [
    'Observations: 1,103 consensus',
    'Both agree: 545 (49%)',
    'Vision supplement: 488 (44%)',
    'Tiebreaker resolved: 51 (5%)',
    'High confidence: 95%',
    'Cost: ~$0.43/paper',
    'Time: ~9 min/paper',
]
for i, s in enumerate(stats):
    ax.text(legend_x + 0.25, legend_y + 1.7 - i*0.24, s, fontsize=7.5)

plt.tight_layout()
plt.savefig(OUT / 'fig1_pipeline_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Saved: {OUT / 'fig1_pipeline_architecture.png'}")
plt.close()
