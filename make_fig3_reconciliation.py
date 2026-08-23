#!/usr/bin/env python3
"""
Figure 3 - reconciliation, two panels.

LEFT: biochar (Li X 2024) pooled CO2-equivalent effect under the AI's matched control
      vs the human's absolute control. Treatment means are byte-identical on both sides;
      only the control differs. Harmonized to the absolute baseline the two estimates
      converge (AI +38.5% vs GT +37.7%).

RIGHT: Loladze per-element AI-minus-human effect difference (percentage points), shown
       for ALL shared cells and for the all_data subset (cells where Loladze applied no
       documented condition selection). The micronutrient gap (~5 pp on all cells)
       collapses to ~1 pp on all_data: the gap is condition selection, not reading error.

Reuses the exact logic of reconciliation_analysis.py, so the plotted numbers match
EXPECTED_OUTPUT_RECONCILIATION.txt. Reproducible: no random state. Reads only runs/.
Writes figures/fig3_reconciliation.png.
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

def npid(s): return re.sub(r'^[\d_]+', '', str(s).strip().lower())
def ff(v):
    try:
        x = float(str(v).strip()); return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None
def low(r, k): return str(r.get(k, "")).strip().lower()
def load(side, excl):
    rows = []
    for f in glob.glob(os.path.join(RUNS, side, "*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            if npid(r.get("paper_id", "")) in excl: continue
            rows.append(r)
    return rows
def lnrr(r):
    if low(r, "unit_canonical") == "ratio":
        x = ff(r.get("treatment_mean")); return math.log(1+x) if (x is not None and 1+x > 0) else None
    t, c = ff(r.get("treatment_mean")), ff(r.get("control_mean"))
    return math.log(t/c) if (t and c and t > 0 and c > 0) else None
def pct(x): return (math.exp(x)-1)*100

LOL_EXCL = {npid(p) for p in {"johnson_1997","ma_2007","rodenkirchen_2009","de_2000","kuehny_1991","li_2010"}}
BIO_EXCL = {npid(p) for p in {"jose_2013"}}

# ============================ (A) BIOCHAR control definition ============================
def _bcbase(r): return (npid(low(r,"paper_id")), low(r,"crop"), low(r,"timepoint"))
def biochar_abs_ctrl(rows):
    ac = {}
    for r in rows:
        if ff(r.get("co_amendment_level")) == 0:
            c = ff(r.get("control_mean"))
            if c is not None: ac.setdefault(_bcbase(r), c)
    return ac
def bc_key(r): return (npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"crop"),
                       low(r,"treatment_level"), low(r,"co_amendment"),
                       ("%g"%round(ff(r.get("co_amendment_level")),4)) if ff(r.get("co_amendment_level")) is not None else low(r,"co_amendment_level"),
                       low(r,"timepoint"))

def biochar_panel():
    ai = load("biochar_v2/keys/ai", BIO_EXCL); gt = load("biochar_v2/keys/gt", BIO_EXCL)
    ac_ai, ac_gt = biochar_abs_ctrl(ai), biochar_abs_ctrl(gt)
    def harm(r, ac):
        t = ff(r.get("treatment_mean")); c = ac.get(_bcbase(r))
        return math.log(t/c) if (t and c and t > 0 and c > 0) else None
    # matched control = native lnRR (AI's own matched control); absolute = harmonized
    aiM = defaultdict(list); aiA = defaultdict(list); gtA = defaultdict(list)
    for r in ai:
        m = lnrr(r); a = harm(r, ac_ai)
        if m is not None: aiM[bc_key(r)].append(m)
        if a is not None: aiA[bc_key(r)].append(a)
    for r in gt:
        a = harm(r, ac_gt)
        if a is not None: gtA[bc_key(r)].append(a)
    shared = set(aiA) & set(gtA)
    # cluster by study
    def pool(C, keys):
        by = defaultdict(list)
        for c in keys:
            if c in C: by[c[0]].append(sum(C[c])/len(C[c]))
        return sum(sum(v)/len(v) for v in by.values())/len(by)
    ai_abs = pool(aiA, shared)
    gt_abs = pool(gtA, shared)
    # AI matched control over the same shared cells where it exists
    ai_matched = pool(aiM, set(aiM) & shared)
    return dict(ai_matched=pct(ai_matched), ai_abs=pct(ai_abs), gt_abs=pct(gt_abs))

# ============================ (B) LOLADZE per-element ============================
def lol_key(r): return (npid(low(r,"paper_id")), low(r,"treatment_level"), low(r,"co_amendment"), low(r,"co_amendment_level"))
def ginfo(r):
    m = re.search(r"AdditionalInfo='([^']*)'", r.get("evidence","")); return (m.group(1).strip() if m else "")

def loladze_panel():
    ai = load("loladze_v2/keys/ai", LOL_EXCL); gt = load("loladze_v2/keys/gt", LOL_EXCL)
    aiC = defaultdict(list); gtC = defaultdict(list); gtinfo = defaultdict(set)
    for r in ai:
        e = lnrr(r)
        if e is not None: aiC[lol_key(r)].append(e)
    for r in gt:
        e = lnrr(r)
        if e is not None: gtC[lol_key(r)].append(e); gtinfo[lol_key(r)].add(ginfo(r))
    shared = set(aiC) & set(gtC)
    def per_element(cells):
        byel = defaultdict(lambda: defaultdict(list))
        for c in cells:
            byel[c[1]][c[0]].append(sum(aiC[c])/len(aiC[c]) - sum(gtC[c])/len(gtC[c]))
            # need GT pooled per study to convert lnRR diff to pp diff
        # diff in pp per element, cluster by study, matching reconciliation_analysis.py
        out = {}
        gtbyel = defaultdict(lambda: defaultdict(list))
        for c in cells:
            gtbyel[c[1]][c[0]].append(sum(gtC[c])/len(gtC[c]))
        for el, bd in byel.items():
            ns = len(bd)
            if ns < 3: continue
            md = sum(sum(v)/len(v) for v in bd.values())/ns
            gp = sum(sum(v)/len(v) for v in gtbyel[el].values())/ns
            out[el] = (pct(gp+md)-pct(gp), ns)
        return out
    allc = per_element(shared)
    clean = per_element([c for c in shared if gtinfo[c] == {""}])
    return allc, clean

def main():
    bio = biochar_panel()
    allc, clean = loladze_panel()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # ---- LEFT panel: biochar control definition ----
    labels = ["AI\nmatched control\n(isolates biochar)", "AI\nabsolute control\n(harmonized)", "Human\nabsolute control"]
    vals = [bio["ai_matched"], bio["ai_abs"], bio["gt_abs"]]
    cols = ["#b04a2f", "#2f6fb0", "#1a7f37"]
    xs = [0, 1, 2]
    bars = axL.bar(xs, vals, color=cols, width=0.6, zorder=3)
    for x, v in zip(xs, vals):
        axL.text(x, v + 0.8, f"{v:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    axL.axhline(0, color="#999999", lw=0.8)
    # bracket showing the harmonized pair agrees
    axL.annotate("", xy=(1, max(bio["ai_abs"], bio["gt_abs"])+5), xytext=(2, max(bio["ai_abs"], bio["gt_abs"])+5),
                 arrowprops=dict(arrowstyle="-", color="#555555", lw=1))
    axL.text(1.5, max(bio["ai_abs"], bio["gt_abs"])+6, "estimands harmonized", ha="center", va="bottom", fontsize=7.5, color="#555555")
    axL.set_xticks(xs); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("Pooled effect on yield (%)", fontsize=9)
    axL.set_ylim(0, max(vals)+12)
    axL.set_title("A  Biochar: control definition (Li X 2024)", fontsize=10, fontweight="bold", loc="left")
    axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)

    # ---- RIGHT panel: Loladze per-element diff, all-cells vs all_data ----
    # order elements by the reconciliation table; show those present in BOTH sets, plus key micros
    order = ["CA", "FE", "MN", "CU", "MG", "ZN", "N", "K", "P"]
    order = [e.lower() for e in order]
    y = list(range(len(order)))[::-1]
    for yi, el in zip(y, order):
        da = allc.get(el, (None, None))[0]
        dc = clean.get(el, (None, None))[0]
        if da is not None:
            axR.plot(da, yi, "o", color="#b04a2f", ms=8, zorder=3)
        if dc is not None:
            axR.plot(dc, yi, "o", color="#1a7f37", ms=8, zorder=3)
        if da is not None and dc is not None:
            axR.annotate("", xy=(dc, yi), xytext=(da, yi),
                         arrowprops=dict(arrowstyle="->", color="#888888", lw=1.1, shrinkA=6, shrinkB=6), zorder=2)
    axR.axvline(0, color="#999999", lw=0.9, ls=":")
    axR.set_yticks(y); axR.set_yticklabels([e.upper() for e in order], fontsize=9)
    axR.set_xlabel("AI - human effect difference (percentage points)", fontsize=9)
    axR.set_ylim(-0.6, len(order)-0.4)
    axR.set_title("B  Loladze 2014: per-element selection", fontsize=10, fontweight="bold", loc="left")
    axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)
    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor="#b04a2f", ms=8, label="all shared cells"),
               Line2D([0],[0], marker="o", color="w", markerfacecolor="#1a7f37", ms=8, label="no-selection subset")]
    axR.legend(handles=handles, fontsize=8, loc="lower right", frameon=False)

    fig.tight_layout()
    out = os.path.join(FIGS, "fig3_reconciliation.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)
    print(f"  LEFT  AI matched={bio['ai_matched']:+.1f}%  AI abs={bio['ai_abs']:+.1f}%  GT abs={bio['gt_abs']:+.1f}%")
    print("  RIGHT all-cells:", {e.upper(): round(allc[e][0],1) for e in order if e in allc})
    print("  RIGHT all_data :", {e.upper(): round(clean[e][0],1) for e in order if e in clean})

if __name__ == "__main__":
    main()
