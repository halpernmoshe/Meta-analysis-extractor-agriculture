#!/usr/bin/env python3
"""
Figure 1 - reading-fidelity scatter. Per dataset, AI vs human cell statistic
(raw treatment mean where both sides store raw means; effect/%-change where the
GT stores an effect), with the identity line, Pearson r and n annotated.

Reuses the bias-free categorical-key pairing of line_by_line_scope_aware.py, so the
plotted points are exactly the matched cells behind EXPECTED_OUTPUT_LINEBYLINE.txt.
Reproducible: no random state. Reads only runs/. Writes figures/fig1_fidelity.png.
"""
import csv, glob, math, os, re
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)

DISPLAY = {"Boldorini": "Boldorini et al. 2024", "Biochar": "Li X et al. 2024",
           "Loladze": "Loladze 2014", "Hui": "Hui et al. 2025", "Li2022": "Li J et al. 2022"}
BASE = {"Boldorini": "boldorini/keys", "Biochar": "biochar_v2/keys", "Loladze": "loladze_v2/keys",
        "Hui": "hui_v4/keys", "Li2022": "li2022_v2/keys"}
EXCLUDE = {
    "Hui": {"zhao_2020", "cakmak_1997", "liu_2014", "dong_2018", "li_2013", "zhang_2012", "khoshgoftarmanesh_2013", "kumar_2018"},
    "Loladze": {"johnson_1997", "ma_2007", "rodenkirchen_2009", "de_2000", "kuehny_1991", "li_2010"},
    "Li2022": {"pramanick_2016", "al-tawaha-et-al-2011"},
    "Biochar": {"jose_2013"},
}
METRIC = {"Boldorini": "raw", "Biochar": "raw", "Hui": "raw", "Loladze": "effect", "Li2022": "effect"}

def npid(s): return re.sub(r'^[\d_]+', '', str(s).strip().lower())
def ff(v):
    try:
        x = float(str(v).strip()); return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None
def low(r, k): return str(r.get(k, "")).strip().lower()
def numtok(r, k):
    x = ff(r.get(k)); return ("%g" % round(x, 4)) if x is not None else low(r, k)
def effect(r):
    if low(r, "unit_canonical") == "ratio": return ff(r.get("treatment_mean"))
    t, c = ff(r.get("treatment_mean")), ff(r.get("control_mean"))
    return (t - c) / c if (t is not None and c is not None and c != 0) else None
def raw_ok(r):
    return low(r, "unit_canonical") not in ("ratio", "unresolved") and ff(r.get("treatment_mean")) is not None
def _bc_base(r): return (npid(low(r, "paper_id")), low(r, "crop"), low(r, "timepoint"))
def biochar_abs_ctrl(rows):
    ac = {}
    for r in rows:
        if ff(r.get("co_amendment_level")) == 0:
            c = ff(r.get("control_mean"))
            if c is not None: ac.setdefault(_bc_base(r), c)
    return ac
def harm_effect(r, ac):
    t = ff(r.get("treatment_mean")); c = ac.get(_bc_base(r))
    return (t - c) / c if (t is not None and c is not None and c != 0) else None
def load(side_dir, excl):
    rows = []
    for f in glob.glob(os.path.join(RUNS, side_dir, "*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            if npid(r.get("paper_id", "")) in excl: continue
            rows.append(r)
    return rows
def pear(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    sx = sum((x-mx)**2 for x in xs); sy = sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(sx*sy) if sx and sy else float("nan")
def keyfn(ds):
    if ds in ("Boldorini", "Biochar"):
        return lambda r: (npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"crop"), low(r,"treatment_level"), low(r,"co_amendment"), numtok(r,"co_amendment_level"), low(r,"timepoint"))
    if ds == "Hui":     return lambda r: (npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"treatment_level"))
    if ds == "Loladze": return lambda r: (npid(low(r,"paper_id")), low(r,"treatment_level"), low(r,"co_amendment"), low(r,"co_amendment_level"))
    return None
def lij_crosswalk(ai):
    LAST = re.compile(r"[a-z][a-z\-]+")
    def fl(n):
        m = LAST.search(n.lower()); return m.group(0) if m else None
    idx = {}
    for r in ai:
        p = npid(low(r, "paper_id")); yr = re.search(r"(19|20)\d{2}", p); la = fl(p)
        if la:
            idx[(la, yr.group(0) if yr else None)] = p; idx.setdefault((la, None), p)
    def remap(r):
        pid = npid(low(r, "paper_id"))
        if not pid.startswith("study") and not pid.startswith("gt_study"): return pid
        m = re.search(r"author='([^']+)'\s*[, ]*((?:19|20)\d{2})", r.get("evidence", ""))
        if not m: return pid
        la, yr = fl(m.group(1)), m.group(2)
        return idx.get((la, yr)) or idx.get((la, None)) or pid
    return remap

def matched_pairs(ds):
    """Return (xs, ys, label) of AI-vs-GT matched cell statistics for the headline metric."""
    excl = {npid(p) for p in EXCLUDE.get(ds, set())}
    ai = load(f"{BASE[ds]}/ai", excl); gt = load(f"{BASE[ds]}/gt", excl)
    if ds == "Li2022":
        remap = lij_crosswalk(ai)
        aiC = defaultdict(list); gtC = defaultdict(list)
        for r in ai:
            e = effect(r)
            if e is not None: aiC[(npid(low(r, "paper_id")),)].append(e)
        for r in gt:
            e = effect(r)
            if e is not None: gtC[(remap(r),)].append(e)
        cells = sorted(set(aiC) & set(gtC), key=str)
        xs = [sum(aiC[c])/len(aiC[c]) for c in cells]; ys = [sum(gtC[c])/len(gtC[c]) for c in cells]
        return [x*100 for x in xs], [y*100 for y in ys], "% change"
    kfn = keyfn(ds)
    if ds == "Biochar":
        ai_ac, gt_ac = biochar_abs_ctrl(ai), biochar_abs_ctrl(gt)
        ef_ai = lambda r: harm_effect(r, ai_ac); ef_gt = lambda r: harm_effect(r, gt_ac)
    else:
        ef_ai = ef_gt = effect
    aiC = defaultdict(lambda: {"eff": [], "raw": []}); gtC = defaultdict(lambda: {"eff": [], "raw": []})
    for r in ai:
        e = ef_ai(r)
        if e is not None: aiC[kfn(r)]["eff"].append(e)
        if raw_ok(r): aiC[kfn(r)]["raw"].append(ff(r.get("treatment_mean")))
    for r in gt:
        e = ef_gt(r)
        if e is not None: gtC[kfn(r)]["eff"].append(e)
        if raw_ok(r): gtC[kfn(r)]["raw"].append(ff(r.get("treatment_mean")))
    cells = sorted(set(aiC) & set(gtC), key=str)
    if METRIC[ds] == "raw":
        rc = [c for c in cells if aiC[c]["raw"] and gtC[c]["raw"]]
        xs = [sum(aiC[c]["raw"])/len(aiC[c]["raw"]) for c in rc]
        ys = [sum(gtC[c]["raw"])/len(gtC[c]["raw"]) for c in rc]
        return xs, ys, "raw treatment mean"
    ec = [c for c in cells if aiC[c]["eff"] and gtC[c]["eff"]]
    xs = [sum(aiC[c]["eff"])/len(aiC[c]["eff"])*100 for c in ec]
    ys = [sum(gtC[c]["eff"])/len(gtC[c]["eff"])*100 for c in ec]
    return xs, ys, "% change"

def main():
    order = ["Boldorini", "Biochar", "Hui", "Loladze", "Li2022"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for ax, ds in zip(axes, order):
        xs, ys, unit = matched_pairs(ds)
        r = pear(xs, ys); n = len(xs)
        ax.scatter(ys, xs, s=18, alpha=0.6, color="#2b6cb0", edgecolors="none")
        lo = min(min(xs), min(ys)); hi = max(max(xs), max(ys))
        pad = 0.05 * (hi - lo if hi > lo else 1)
        lo -= pad; hi += pad
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(DISPLAY[ds], fontsize=10, fontweight="bold")
        ax.set_xlabel(f"Human reference ({unit})", fontsize=8)
        ax.set_ylabel(f"AI workflow ({unit})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.04, 0.93, f"r = {r:.3f}\nn = {n}", transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))
    axes[5].axis("off")
    fig.tight_layout()
    out = os.path.join(FIGS, "fig1_fidelity.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)

if __name__ == "__main__":
    main()
