#!/usr/bin/env python3
"""
Figure 2 - equivalence forest. Per-dataset paired AI-minus-human difference in pooled
effect (percentage points) with its 90% CI, plotted against the +-20%-of-effect
equivalence margin. Datasets whose CI falls entirely inside the margin pass TOST at 20%.

Reuses the paired scope-matched TOST machinery of scope_aware_paired_tost.py, so the
plotted differences and CIs are exactly those in EXPECTED_OUTPUT_PAIRED_TOST.txt.
Reproducible: no random state. Reads only runs/. Writes figures/fig2_equivalence.png.
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
FIGS = os.path.join(REPO, "figures")
os.makedirs(FIGS, exist_ok=True)
DISPLAY = {"Boldorini":"Boldorini et al. 2024","Biochar":"Li X et al. 2024","Loladze":"Loladze 2014","Hui":"Hui et al. 2025","Li2022":"Li J et al. 2022"}
BASE = {"Boldorini":"boldorini/keys","Biochar":"biochar_v2/keys","Loladze":"loladze_v2/keys","Hui":"hui_v4/keys","Li2022":"li2022_v2/keys"}
EXCLUDE = {"Hui":{"zhao_2020","cakmak_1997","liu_2014","dong_2018","li_2013","zhang_2012","khoshgoftarmanesh_2013","kumar_2018"},
           "Loladze":{"johnson_1997","ma_2007","rodenkirchen_2009","de_2000","kuehny_1991","li_2010"},
           "Li2022":{"pramanick_2016","al-tawaha-et-al-2011"},"Biochar":{"jose_2013"}}
def npid(s): return re.sub(r'^[\d_]+','',str(s).strip().lower())
def ff(v):
    try:
        x=float(str(v).strip()); return x if math.isfinite(x) else None
    except (ValueError,TypeError): return None
def low(r,k): return str(r.get(k,"")).strip().lower()
def numtok(r,k):
    x=ff(r.get(k)); return ("%g"%round(x,4)) if x is not None else low(r,k)
def load(side,excl):
    rows=[]
    for f in glob.glob(os.path.join(RUNS,side,"*.csv")):
        for r in csv.DictReader(open(f,encoding="utf-8-sig")):
            if npid(r.get("paper_id",""))in excl: continue
            rows.append(r)
    return rows
def lnrr(r):
    if low(r,"unit_canonical")=="ratio":
        x=ff(r.get("treatment_mean")); return math.log(1+x) if (x is not None and 1+x>0) else None
    t,c=ff(r.get("treatment_mean")),ff(r.get("control_mean"))
    return math.log(t/c) if (t and c and t>0 and c>0) else None
def _bcbase(r): return (npid(low(r,"paper_id")),low(r,"crop"),low(r,"timepoint"))
def biochar_abs_ctrl(rows):
    ac={}
    for r in rows:
        if ff(r.get("co_amendment_level"))==0:
            c=ff(r.get("control_mean"))
            if c is not None: ac.setdefault(_bcbase(r),c)
    return ac
def lij_crosswalk(ai):
    L=re.compile(r"[a-z][a-z\-]+")
    def fl(n):
        m=L.search(n.lower()); return m.group(0) if m else None
    idx={}
    for r in ai:
        p=npid(low(r,"paper_id")); yr=re.search(r"(19|20)\d{2}",p); la=fl(p)
        if la: idx[(la,yr.group(0) if yr else None)]=p; idx.setdefault((la,None),p)
    def remap(r):
        pid=npid(low(r,"paper_id"))
        if not pid.startswith("study") and not pid.startswith("gt_study"): return pid
        m=re.search(r"author='([^']+)'\s*[, ]*((?:19|20)\d{2})",r.get("evidence",""))
        if not m: return pid
        return idx.get((fl(m.group(1)),m.group(2))) or idx.get((fl(m.group(1)),None)) or pid
    return remap
def keyfn(ds):
    if ds=="Boldorini": return lambda r:(npid(low(r,"paper_id")),low(r,"crop"),low(r,"treatment_level"))
    if ds=="Biochar": return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"crop"),low(r,"treatment_level"),low(r,"co_amendment"),numtok(r,"co_amendment_level"),low(r,"timepoint"))
    if ds=="Hui":     return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"treatment_level"))
    if ds=="Loladze": return lambda r:(npid(low(r,"paper_id")),low(r,"treatment_level"),low(r,"co_amendment"),low(r,"co_amendment_level"))
    return None
pct=lambda x:(math.exp(x)-1)*100

def compute(ds):
    excl={npid(p) for p in EXCLUDE.get(ds,set())}
    ai=load(f"{BASE[ds]}/ai",excl); gt=load(f"{BASE[ds]}/gt",excl)
    remap=lij_crosswalk(ai) if ds=="Li2022" else None
    if ds=="Biochar":
        ac_ai,ac_gt=biochar_abs_ctrl(ai),biochar_abs_ctrl(gt)
        ef_ai=lambda r:(math.log(ff(r['treatment_mean'])/ac_ai[_bcbase(r)]) if (ff(r.get('treatment_mean')) and ac_ai.get(_bcbase(r)) and ff(r['treatment_mean'])>0 and ac_ai[_bcbase(r)]>0) else None)
        ef_gt=lambda r:(math.log(ff(r['treatment_mean'])/ac_gt[_bcbase(r)]) if (ff(r.get('treatment_mean')) and ac_gt.get(_bcbase(r)) and ff(r['treatment_mean'])>0 and ac_gt[_bcbase(r)]>0) else None)
    else:
        ef_ai=ef_gt=lnrr
    if ds=="Li2022":
        kf=lambda r:(npid(low(r,"paper_id")),); kg=lambda r:(remap(r),)
    else:
        kf=keyfn(ds); kg=kf
    aiC,gtC=defaultdict(list),defaultdict(list)
    for r in ai:
        e=ef_ai(r)
        if e is not None: aiC[kf(r)].append(e)
    for r in gt:
        e=ef_gt(r)
        if e is not None: gtC[kg(r)].append(e)
    shared=set(aiC)&set(gtC)
    bd=defaultdict(list); bg=defaultdict(list)
    for c in shared:
        st=c[0]; a=sum(aiC[c])/len(aiC[c]); g=sum(gtC[c])/len(gtC[c])
        bd[st].append(a-g); bg[st].append(g)
    sd=[sum(v)/len(v) for v in bd.values()]; ns=len(sd); md=sum(sd)/ns
    pse=(statistics.pstdev(sd)*math.sqrt(ns/(ns-1))/math.sqrt(ns)) if ns>1 else float("nan")
    plo,phi=md-1.645*pse,md+1.645*pse
    gtp=sum(sum(v)/len(v) for v in bg.values())/len(bg)
    margin=0.20*abs(gtp)
    inside = (plo>-margin and phi<margin)
    return dict(diff=pct(md), lo=pct(md-1.645*pse), hi=pct(md+1.645*pse),
                margin=pct(gtp+margin)-pct(gtp), gtp=pct(gtp), ns=ns, inside=inside,
                # symmetric margin in pp for plotting around 0
                marg_pp=abs(gtp)*0.20*100)

def main():
    order=["Boldorini","Biochar","Hui","Loladze","Li2022"]
    res={ds:compute(ds) for ds in order}
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ys=list(range(len(order)))[::-1]
    for y, ds in zip(ys, order):
        R=res[ds]
        m=R["marg_pp"]
        # margin band
        ax.add_patch(plt.Rectangle((-m, y-0.32), 2*m, 0.64, color="#e6eff7", zorder=0))
        ax.plot([-m, -m], [y-0.32, y+0.32], color="#7aa6cf", lw=1, zorder=1)
        ax.plot([ m,  m], [y-0.32, y+0.32], color="#7aa6cf", lw=1, zorder=1)
        col = "#1a7f37" if R["inside"] else "#b04a2f"
        ax.plot([R["lo"], R["hi"]], [y, y], color=col, lw=2, zorder=3)
        ax.plot(R["diff"], y, "o", color=col, ms=7, zorder=4)
        ax.text(0.0, y+0.42, f"{R['diff']:+.2f} pp  [{R['lo']:+.2f}, {R['hi']:+.2f}]   margin ±{m:.1f} pp",
                ha="center", va="bottom", fontsize=7.5, color="#333333")
    ax.axvline(0, color="#999999", lw=0.8, ls=":")
    ax.set_yticks(ys); ax.set_yticklabels([DISPLAY[d] for d in order], fontsize=9)
    ax.set_xlabel("Paired AI - human difference in pooled effect (percentage points)", fontsize=9)
    ax.set_ylim(-0.7, len(order)-0.3)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Aggregate equivalence: paired difference vs ±20%-of-effect margin", fontsize=10, fontweight="bold")
    fig.tight_layout()
    out=os.path.join(FIGS,"fig2_equivalence.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)
    for ds in order:
        R=res[ds]; print(f"  {DISPLAY[ds]:22} diff={R['diff']:+.2f} CI[{R['lo']:+.2f},{R['hi']:+.2f}] margin+-{R['marg_pp']:.2f} inside20%={R['inside']}")

if __name__ == "__main__":
    main()
