#!/usr/bin/env python3
"""
Generate the scope-matched figures from the SAME shipped keys as
scope_matched_equivalence.py. Outputs PNGs to ./figures/ and prints the
computed concordance r per dataset. Reproducible: no random state, no network.

Figures produced (all grounded in the definitive scope-matched method):
  figS4_scatter.png     - per-dataset reference-vs-AI cell-level %-effect scatter with
                          identity line and computed r; the supporting line-by-line
                          concordance discussed in manuscript Section 3.5.1 (manuscript Figure S4)
  figS2_diff_forest.png - per-dataset AI-vs-reference pooled difference (pp) with paired 90% CI
                          and the proportional +/-20% equivalence margin (manuscript Figure S2)
  figS3_margin_grid.png - paired-TOST pass/fail across +/-5/10/15/20% margins (manuscript Figure S3)

The flow figure (manuscript Figure S1) is produced by make_figS1_flow.py, and the
Bland-Altman figure (manuscript Figure S5) by bland_altman_figS5.py. The supporting line-by-line MATCH/coverage outputs
(manuscript Table S6) are deposited under line_by_line_results/ (see its README).

NOT regenerated here (need author input or a different pipeline):
  Figure S1 (flow-diagram schematic) and Figure S5 (variance-coverage); the latter
  needs per-observation dispersion counts.

Requires: Python 3.8+, matplotlib. Run: python make_figures.py
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = HERE + "/runs"
OUT = HERE + "/figures"; os.makedirs(OUT, exist_ok=True)
DS = {"Boldorini":"boldorini/keys", "Biochar":"biochar_v2/keys", "Loladze":"loladze_v2/keys",
      "Hui":"hui_v4/keys", "Li2022":"li2022_v2/keys"}
DISPLAY = {
    "Boldorini": "Boldorini et al. 2024",
    "Biochar": "Li X et al. 2024",
    "Loladze": "Loladze 2014",
    "Hui": "Hui et al. 2025",
    "Li2022": "Li J et al. 2022",
}
EXCLUDE = {
    "Hui": {"zhao_2020","cakmak_1997","liu_2014","dong_2018","li_2013",
            "zhang_2012","khoshgoftarmanesh_2013","kumar_2018"},
    "Loladze": {"johnson_1997","ma_2007","rodenkirchen_2009","de_2000","kuehny_1991","li_2010"},
    "Li2022": {"pramanick_2016","al-tawaha-et-al-2011"},
    "Biochar": {"jose_2013"},
}
def _normpid(s): return re.sub(r'^[\d_]+','',str(s).strip().lower())
def ff(v):
    try: x=float(str(v).strip()); return x if math.isfinite(x) else None
    except (ValueError,TypeError): return None
def lnrr(r):
    t=ff(r.get("treatment_mean")); c=ff(r.get("control_mean"))
    if t is not None and c is not None and t>0 and c>0: return math.log(t/c)
    if str(r.get("unit_canonical","")).strip().lower()=="ratio" and t is not None and (1+t)>0:
        return math.log(1+t)
    return None
def low(r,k): return str(r.get(k,"")).strip().lower()
def numtok(r,k):
    x=ff(r.get(k)); return ("%g"%round(x,4)) if x is not None else low(r,k)
def pid(r): return low(r,"paper_id")
def ctrl(r):
    t=low(r,"control_token")
    if "absolute" in t: return "absolute"
    if "match" in t or "cofactor" in t: return "matched"
    return t or "na"
def cellkey(ds, r):
    if ds=="Loladze":  return (pid(r), low(r,"treatment_level"), low(r,"co_amendment"))
    if ds=="Biochar":  return (pid(r), low(r,"co_amendment"), numtok(r,"co_amendment_level"), ctrl(r))
    if ds=="Li2022":   return (pid(r), low(r,"treatment_level"), low(r,"co_amendment"))
    return (pid(r), low(r,"treatment_level"))
def load(d):
    rows=[]
    for f in glob.glob(f"{RUNS}/{d}/*.csv"):
        for r in csv.DictReader(open(f,encoding="utf-8-sig")): rows.append(r)
    return rows
def cell_effects(ds, rows):
    by=defaultdict(list)
    for r in rows:
        e=lnrr(r)
        if e is not None: by[cellkey(ds,r)].append(e)
    return {k:sum(v)/len(v) for k,v in by.items()}
def pct(x): return (math.exp(x)-1)*100
def pearson(xs, ys):
    n=len(xs)
    if n<3: return float("nan")
    mx=sum(xs)/n; my=sum(ys)/n
    sx=sum((x-mx)**2 for x in xs); sy=sum((y-my)**2 for y in ys)
    if sx<=0 or sy<=0: return float("nan")
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(sx*sy)

# ---- compute per-dataset scope-matched quantities -------------------------
D = {}
for ds,base in DS.items():
    ai=load(f"{base}/ai"); gt=load(f"{base}/gt")
    if not ai or not gt: continue
    excl=EXCLUDE.get(ds,set())
    if excl:
        ai=[r for r in ai if _normpid(pid(r)) not in excl]; gt=[r for r in gt if _normpid(pid(r)) not in excl]
    aiC=cell_effects(ds,ai); gtC=cell_effects(ds,gt)
    common=sorted(set(aiC)&set(gtC))
    gx=[pct(gtC[k]) for k in common]; ax=[pct(aiC[k]) for k in common]
    # paired diff clustered by study
    bystudy=defaultdict(list)
    for k in common: bystudy[k[0]].append(aiC[k]-gtC[k])
    sd=[sum(v)/len(v) for v in bystudy.values()]; ns=len(sd)
    md=sum(sd)/ns; pse=((sum((x-md)**2 for x in sd)/(ns-1))**0.5/math.sqrt(ns)) if ns>1 else float("nan")
    plo,phi=md-1.645*pse, md+1.645*pse
    # GT pooled lnRR for proportional margins
    gstudy=defaultdict(list)
    for k in common: gstudy[k[0]].append(gtC[k])
    gsm=[sum(v)/len(v) for v in gstudy.values()]; gpool=sum(gsm)/len(gsm)
    D[ds]={"gx":gx,"ax":ax,"r":pearson(gx,ax),"n":len(common),
           "md":md,"plo":plo,"phi":phi,"gpool":gpool,"ns":ns}

ORDER=[d for d in ["Boldorini","Biochar","Loladze","Hui","Li2022"] if d in D]

# ---- Figure 1: concordance scatter ---------------------------------------
fig,axes=plt.subplots(1,len(ORDER),figsize=(3.0*len(ORDER),3.2))
if len(ORDER)==1: axes=[axes]
for ax,ds in zip(axes,ORDER):
    d=D[ds]
    ax.scatter(d["gx"],d["ax"],s=16,alpha=0.6,edgecolor="none")
    lo=min(d["gx"]+d["ax"]+[0]); hi=max(d["gx"]+d["ax"]+[0])
    pad=0.05*(hi-lo+1); ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],"k--",lw=0.8)
    ax.set_title(f"{DISPLAY.get(ds, ds)}\nr={d['r']:.3f}, n={d['n']}",fontsize=9)
    ax.set_xlabel("Reference cell effect (%)",fontsize=8)
    if ds==ORDER[0]: ax.set_ylabel("AI cell effect (%)",fontsize=8)
    ax.tick_params(labelsize=7)
fig.suptitle("Figure S4. Supporting line-by-line concordance: AI vs reference cell-level effects (identity line dashed)",fontsize=9)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{OUT}/figS4_scatter.png",dpi=200); plt.close(fig)

# ---- Figure S2: diff forest with paired 90% CI + 20% margin --------------
fig,ax=plt.subplots(figsize=(7,3.0))
ys=list(range(len(ORDER)))[::-1]
for y,ds in zip(ys,ORDER):
    d=D[ds]
    md,plo,phi=pct(d["md"]),pct(d["plo"]),pct(d["phi"])
    ax.errorbar(md,y,xerr=[[md-plo],[phi-md]],fmt="o",color="C0",capsize=3)
    m20=pct(0.20*abs(d["gpool"]))
    ax.plot([-m20,-m20],[y-0.25,y+0.25],color="C3",lw=1)
    ax.plot([ m20, m20],[y-0.25,y+0.25],color="C3",lw=1)
ax.axvline(0,color="k",lw=0.6)
ax.set_yticks(ys); ax.set_yticklabels([DISPLAY.get(d, d) for d in ORDER],fontsize=9)
ax.set_xlabel("AI - reference pooled difference (%), paired 90% CI; red bars = +/-20% margin",fontsize=8)
ax.set_title("Figure S2. Scope-matched aggregate difference with paired-TOST 20% margins",fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/figS2_diff_forest.png",dpi=200); plt.close(fig)

# ---- Figure S3: margin-sensitivity grid ----------------------------------
margins=[0.05,0.10,0.15,0.20]
fig,ax=plt.subplots(figsize=(5,3.0))
for yi,ds in enumerate(ORDER[::-1]):
    d=D[ds]
    for xi,m in enumerate(margins):
        mm=m*abs(d["gpool"])
        ok=(d["plo"]>-mm and d["phi"]<mm)
        ax.add_patch(plt.Rectangle((xi,yi),1,1,facecolor=("#2ca02c" if ok else "#d62728"),
                     edgecolor="white"))
        ax.text(xi+0.5,yi+0.5,"PASS" if ok else "fail",ha="center",va="center",
                color="white",fontsize=8,fontweight="bold")
ax.set_xlim(0,len(margins)); ax.set_ylim(0,len(ORDER))
ax.set_xticks([i+0.5 for i in range(len(margins))]); ax.set_xticklabels([f"{int(m*100)}%" for m in margins])
ax.set_yticks([i+0.5 for i in range(len(ORDER))]); ax.set_yticklabels([DISPLAY.get(d, d) for d in ORDER[::-1]],fontsize=9)
ax.set_xlabel("Proportional equivalence margin"); ax.set_title("Figure S3. Paired-TOST margin sensitivity",fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/figS3_margin_grid.png",dpi=200); plt.close(fig)

print("Computed scope-matched concordance r (cell-level, outcome-blind matching):")
for ds in ORDER:
    print(f"  {DISPLAY.get(ds, ds):24} r={D[ds]['r']:.3f}  (n={D[ds]['n']} common cells, {D[ds]['ns']} studies)")
print(f"\nWrote: {OUT}/figS4_scatter.png, figS2_diff_forest.png, figS3_margin_grid.png")
