#!/usr/bin/env python3
"""
Generate manuscript Figure S1: per-dataset flow and attrition.

Reviewer 1 (R1.3) requested a transparent per-dataset flow diagram. The counts
here are the solid paper-level and structural-cell numbers reported in
Methods Table 2.6 and the locked scope-matched analysis (EXPECTED_OUTPUT.txt):

  - PDFs processed, mislabelled-PDF exclusions, and the studies that contribute
    at least one common-scope cell to the paired comparison (paper level);
  - the structural-cell partition: common (compared) / reference-only
    (non-coverage) / workflow-only (scope expansion).

Reproducible: no random state, no network. Run: python make_figS1_flow.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "figS1_flow.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

# (display, PDFs processed, mislabelled excluded, contributing studies,
#  common cells, reference-only cells, workflow-only cells)   -- Methods Table 2.6 + EXPECTED_OUTPUT.txt
ROWS = [
    ("Boldorini et al. 2024", 18, 0, 16, 22, 2, 1),
    ("Li X et al. 2024",      28, 1, 14, 17, 65, 57),
    ("Hui et al. 2025",       37, 8, 15, 27, 9, 6),
    ("Li J et al. 2022",      50, 2, 21, 24, 212, 55),
    ("Loladze 2014",          46, 6, 38, 252, 68, 106),
]
labels = [r[0] for r in ROWS]
y = list(range(len(ROWS)))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})

# ---- Left: paper-level flow (processed -> after mislabel exclusion -> contributing studies) ----
C_PROC, C_KEEP, C_STUD = "#cfd8e3", "#9bb2cc", "#3b6ea5"
bar_h = 0.26
for i, (_, proc, mis, stud, *_rest) in enumerate(ROWS):
    kept = proc - mis
    axL.barh(i + bar_h, proc, height=bar_h, color=C_PROC, edgecolor="white", zorder=2)
    axL.barh(i,         kept, height=bar_h, color=C_KEEP, edgecolor="white", zorder=2)
    axL.barh(i - bar_h, stud, height=bar_h, color=C_STUD, edgecolor="white", zorder=2)
    axL.text(proc + 0.6, i + bar_h, f"{proc} processed", va="center", fontsize=7)
    axL.text(kept + 0.6, i, f"{kept} after −{mis} mislabelled", va="center", fontsize=7)
    axL.text(stud + 0.6, i - bar_h, f"{stud} contributing studies", va="center", fontsize=7, color=C_STUD, fontweight="bold")
axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=8)
axL.set_xlabel("Papers", fontsize=8.5)
axL.set_xlim(0, 62)
axL.set_title("A  Paper-level flow", fontsize=9, loc="left", fontweight="bold")
axL.tick_params(labelsize=7.5)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)
# Each bar is labelled inline (processed / after exclusion / contributing studies),
# so no separate legend is needed for panel A.

# ---- Right: structural-cell partition (common / reference-only / workflow-only) ----
C_COM, C_REF, C_WF = "#2c7a4b", "#d98a3c", "#7a76b5"
for i, (_, _p, _m, _s, com, ref, wf) in enumerate(ROWS):
    axR.barh(i, com, color=C_COM, edgecolor="white", zorder=2)
    axR.barh(i, ref, left=com, color=C_REF, edgecolor="white", zorder=2)
    axR.barh(i, wf, left=com + ref, color=C_WF, edgecolor="white", zorder=2)
    total = com + ref + wf
    axR.text(total + 4, i, f"{com} / {ref} / {wf}", va="center", fontsize=7)
axR.set_yticks(y); axR.set_yticklabels([])
axR.set_xlabel("Structural cells", fontsize=8.5)
axR.set_xlim(0, max(sum(r[4:7]) for r in ROWS) * 1.18)
axR.set_title("B  Structural-cell partition", fontsize=9, loc="left", fontweight="bold")
axR.tick_params(labelsize=7.5)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)
axR.legend(handles=[Patch(color=C_COM, label="Common (compared)"),
                    Patch(color=C_REF, label="Reference-only (non-coverage)"),
                    Patch(color=C_WF, label="Workflow-only (scope expansion)")],
           fontsize=6.6, loc="lower right", frameon=False)

fig.suptitle("Figure S1. Per-dataset flow and attrition (paper level and structural-cell partition). "
             "Hui et al. 2025 shown after excluding 8 mislabelled-PDF papers.",
             fontsize=8.5, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT, dpi=200)
print("Wrote:", OUT)
