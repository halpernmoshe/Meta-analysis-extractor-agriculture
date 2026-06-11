#!/usr/bin/env python3
"""
Observation-level Bland-Altman limits of agreement on the SAME outcome-blind matched cells used
by line_by_line_scope_aware.py (manuscript Supplement S5; Reviewer 1's numeric LoA request).

Per dataset, on the matched cells, the per-cell percentage-change effect difference (AI% - GT%)
is summarised by its mean bias and 95% limits of agreement (mean +/- 1.96 SD), with the median.
Effect is lnRR -> %; biochar uses the harmonized absolute control; Li J is study-level. Pairing is
categorical and never consults outcome values. Deterministic; writes figures/figS5_bland_altman.png
and prints the numeric table.
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(REPO, "runs")
OUT = os.path.join(REPO, "figures", "figS5_bland_altman.png")
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
    for f in sorted(glob.glob(os.path.join(RUNS,side,"*.csv"))):
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
def bc_ac(rows):
    ac={}
    for r in rows:
        if ff(r.get("co_amendment_level"))==0:
            c=ff(r.get("control_mean"))
            if c is not None: ac.setdefault(_bcbase(r),c)
    return ac
def keyfn(ds):
    if ds in("Boldorini","Biochar"): return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"crop"),low(r,"treatment_level"),low(r,"co_amendment"),numtok(r,"co_amendment_level"),low(r,"timepoint"))
    if ds=="Hui": return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"treatment_level"))
    if ds=="Loladze": return lambda r:(npid(low(r,"paper_id")),low(r,"treatment_level"),low(r,"co_amendment"),low(r,"co_amendment_level"))
    return lambda r:(npid(low(r,"paper_id")),)
pct=lambda x:(math.exp(x)-1)*100
order=["Boldorini","Biochar","Loladze","Hui","Li2022"]
fig,axes=plt.subplots(2,3,figsize=(15,9)); axes=axes.ravel()
print(f"{'Dataset':22}{'n':>5}{'bias pp':>9}{'SD':>8}{'LoA_lo':>9}{'LoA_hi':>9}{'median':>9}")
print("-"*71)
for i,ds in enumerate(order):
    excl={npid(p) for p in EXCLUDE.get(ds,set())}
    ai=load(f"{BASE[ds]}/ai",excl); gt=load(f"{BASE[ds]}/gt",excl)
    kf=keyfn(ds)
    if ds=="Biochar":
        ac_a,ac_g=bc_ac(ai),bc_ac(gt)
        ea=lambda r:(math.log(ff(r['treatment_mean'])/ac_a[_bcbase(r)]) if (ff(r.get('treatment_mean')) and ac_a.get(_bcbase(r)) and ff(r['treatment_mean'])>0 and ac_a[_bcbase(r)]>0) else None)
        eg=lambda r:(math.log(ff(r['treatment_mean'])/ac_g[_bcbase(r)]) if (ff(r.get('treatment_mean')) and ac_g.get(_bcbase(r)) and ff(r['treatment_mean'])>0 and ac_g[_bcbase(r)]>0) else None)
    else:
        ea=eg=lnrr
    aiC,gtC=defaultdict(list),defaultdict(list)
    for r in ai:
        e=ea(r)
        if e is not None: aiC[kf(r)].append(e)
    for r in gt:
        e=eg(r)
        if e is not None: gtC[kf(r)].append(e)
    diffs=[]; means=[]
    for c in sorted(set(aiC)&set(gtC)):
        a=pct(sum(aiC[c])/len(aiC[c])); g=pct(sum(gtC[c])/len(gtC[c]))
        diffs.append(a-g); means.append((a+g)/2)
    n=len(diffs)
    ax=axes[i]
    if n>=2:
        bias=statistics.mean(diffs); sd=statistics.stdev(diffs)
        lo,hi=bias-1.96*sd,bias+1.96*sd; med=statistics.median(diffs)
        print(f"{DISPLAY[ds]:22}{n:>5}{bias:>+9.2f}{sd:>8.2f}{lo:>+9.2f}{hi:>+9.2f}{med:>+9.2f}")
        ax.scatter(means,diffs,s=26,alpha=0.55,edgecolor="k",linewidth=0.3,color="#2b6cb0")
        ax.axhline(bias,color="#c53030",lw=1.4); ax.axhline(hi,color="#718096",ls="--",lw=1.1); ax.axhline(lo,color="#718096",ls="--",lw=1.1); ax.axhline(0,color="k",lw=0.6,alpha=0.3)
        ax.set_title(f"{DISPLAY[ds]} (n={n})\nbias {bias:+.1f}, 95% LoA [{lo:+.1f}, {hi:+.1f}] pp",fontsize=9)
    else:
        ax.set_title(f"{DISPLAY[ds]} (n={n}, too few)",fontsize=9)
    ax.set_xlabel("mean of AI & human effect (%)",fontsize=8); ax.set_ylabel("AI - human effect (pp)",fontsize=8); ax.tick_params(labelsize=7)
axes[5].axis("off")
axes[5].text(0.02,0.9,"Figure S5. Bland-Altman agreement of the per-cell\npercentage-change effect (AI - human, pp) on the\noutcome-blind matched cells. Red = mean bias;\ndashed = 95% limits of agreement (mean +/- 1.96 SD).",va="top",ha="left",fontsize=9,family="monospace")
fig.suptitle("Observation-level limits of agreement (matched cells)",fontsize=13,y=0.995)
fig.tight_layout(rect=[0,0,1,0.98]); os.makedirs(os.path.dirname(OUT),exist_ok=True); fig.savefig(OUT,dpi=200); plt.close(fig)
print(f"\nWrote {OUT}")
