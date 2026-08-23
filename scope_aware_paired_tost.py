#!/usr/bin/env python3
"""
PAIRED scope-matched TOST with the scope-aware corrections (lnRR).
Cells matched by categorical key (no values). Paired per-cell lnRR diff, clustered by study.
Corrections: biochar control harmonized to absolute; Loladze MID key (effect from
ratio); Li J study-level + author-year crosswalk.
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict
REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
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
    # Boldorini paired effects use (paper, outcome, crop, treatment level).
    # Strict raw-mean fidelity additionally retains co-amendment, dose, time point and unit;
    # it has 9 cells, distinct from these 16 outcome-blind paired effect cells.
    if ds=="Boldorini": return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"crop"),low(r,"treatment_level"))
    if ds=="Biochar": return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"crop"),low(r,"treatment_level"),low(r,"co_amendment"),numtok(r,"co_amendment_level"),low(r,"timepoint"))
    if ds=="Hui":     return lambda r:(npid(low(r,"paper_id")),low(r,"outcome_canonical"),low(r,"treatment_level"))
    if ds=="Loladze": return lambda r:(npid(low(r,"paper_id")),low(r,"treatment_level"),low(r,"co_amendment"),low(r,"co_amendment_level"))
    return None
pct=lambda x:(math.exp(x)-1)*100
print(f"{'Dataset':22} {'cells':>5} {'studies':>7} {'AI%':>7} {'GT%':>7} {'relative RR diff % (90% CI)':>27}  TOST 5/10/15/20")
print("-"*108)
for ds in ["Boldorini","Biochar","Hui","Loladze","Li2022"]:
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
    if not shared: print(f"{DISPLAY[ds]:22} no cells"); continue
    bd=defaultdict(list); bg=defaultdict(list)
    for c in shared:
        st=c[0]; a=sum(aiC[c])/len(aiC[c]); g=sum(gtC[c])/len(gtC[c])
        bd[st].append(a-g); bg[st].append(g)
    sd=[sum(v)/len(v) for v in bd.values()]; ns=len(sd); md=sum(sd)/ns
    pse=(statistics.pstdev(sd)*math.sqrt(ns/(ns-1))/math.sqrt(ns)) if ns>1 else float("nan")
    plo,phi=md-1.645*pse,md+1.645*pse
    gtp=sum(sum(v)/len(v) for v in bg.values())/len(bg)
    ladder=" ".join("P" if (plo>-m*abs(gtp) and phi<m*abs(gtp)) else "." for m in (0.05,0.10,0.15,0.20))
    print(f"{DISPLAY[ds]:22} {len(shared):>5} {ns:>7} {pct(gtp+md):>6.1f}% {pct(gtp):>6.1f}% {pct(md):>+6.2f} [{pct(plo):>+5.2f},{pct(phi):>+5.2f}]   {ladder}")
print("-"*108)
print("P=pass.")
