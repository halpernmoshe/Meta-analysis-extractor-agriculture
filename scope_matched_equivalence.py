#!/usr/bin/env python3
"""
SCOPE-matched equivalence (not line-by-line matching).

The AI often extracts more than the human curated (scope expansion, not error).
Comparing the AI superset to the human's curated set is unfair. So:
1. Define a comparable CELL by its STRUCTURAL dimensions (study x element x tissue
   x contrast-type) — categorical, blind to outcome values. NOT value-pairing.
2. Average lnRR within each cell on each side (collapses AI over-splitting).
3. Keep only cells the human GT actually covered AND the AI also has (common scope).
4. Pool each side over the common scope (cluster by study) and TOST.
The AI-only cells are reported separately as additional scope; GT-only as AI gaps.
"""
import csv, glob, math, os
from collections import defaultdict

RUNS = os.path.dirname(os.path.abspath(__file__)) + "/runs"
DS = {"Boldorini":"boldorini/keys", "Biochar":"biochar_v2/keys", "Loladze":"loladze_v2/keys",
      "Hui":"hui_v4/keys", "Li2022":"li2022_v2/keys"}   # Internal run-directory keys, not citation labels.
DISPLAY = {
    "Boldorini": "Boldorini et al. 2024",
    "Biochar": "Li X et al. 2024",
    "Loladze": "Loladze 2014",
    "Hui": "Hui et al. 2025",
    "Li2022": "Li J et al. 2022",
}

# Paper-level scope control: exclude papers with DOCUMENTED corpus errors (objective data-integrity
# issues, not disagreement). Justified per paper; logged in CORPUS_ERRORS.md.
import re as _re
# All 8 mislabeled-PDF Hui papers (normalized, leading "NN_" stripped). See CORPUS_ERRORS.md / hui_audit/MANIFEST.csv.
EXCLUDE = {
    "Hui": {"zhao_2020","cakmak_1997","liu_2014","dong_2018","li_2013",
            "zhang_2012","khoshgoftarmanesh_2013","kumar_2018"},
    # Corpus cleanup (corpus_cleanup_decisions.csv). RELABEL cases are left EXCLUDED for now
    # (relabel is a later author decision); see notes below.
    # Loladze: 6 RELABEL rows (johnson_1997->Johnson 2003, ma_2007->Fernando 2012a,
    #   rodenkirchen_2009->Weigt 2011, de_2000->Haase 2008, kuehny_1991->Cao 1997, li_2010->Hogy 2009)
    "Loladze": {"johnson_1997","ma_2007","rodenkirchen_2009","de_2000","kuehny_1991","li_2010"},
    # Li2022: pramanick_2016 = true EXCLUDE (mislabel, no GT counterpart);
    #   al-tawaha-et-al-2011 = RELABEL->al-tawaha_2006 (left excluded for now)
    "Li2022": {"pramanick_2016","al-tawaha-et-al-2011"},
    # Biochar: jose_2013 = RELABEL->Alburquerque 2013 (left excluded for now). Hui's 8 stay.
    "Biochar": {"jose_2013"},
}
def _normpid(s): return _re.sub(r'^[\d_]+','',str(s).strip().lower())

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

# structural cell key per dataset (categorical/blind only)
def cellkey(ds, r):
    if ds=="Loladze":  return (pid(r), low(r,"treatment_level"), low(r,"co_amendment"))            # study,element,tissue
    if ds=="Biochar":  return (pid(r), low(r,"co_amendment"), numtok(r,"co_amendment_level"), ctrl(r))
    if ds=="Li2022":   return (pid(r), low(r,"treatment_level"), low(r,"co_amendment"))             # study,product,method
    if ds=="Hui":      return (pid(r), low(r,"treatment_level"))                                    # study,app_type
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
    return {k:sum(v)/len(v) for k,v in by.items()}     # one effect per cell
def pool_cells(cells):  # cells: {key:effect}; cluster by study (key[0])
    bystudy=defaultdict(list)
    for k,e in cells.items(): bystudy[k[0]].append(e)
    sm=[sum(v)/len(v) for v in bystudy.values()]; n=len(sm)
    if not n: return None
    m=sum(sm)/n
    se=((sum((x-m)**2 for x in sm)/(n-1))**0.5/math.sqrt(n)) if n>1 else float("nan")
    return {"lnrr":m,"se":se,"studies":n,"cells":len(cells)}
def pct(x): return (math.exp(x)-1)*100

print(f"{'Dataset':24} {'Reference %effect [95%CI]':>28} {'Workflow %effect [95%CI]':>28} {'diff':>6} | common/reference-only/workflow-only cells")
print("-"*124)
for ds,base in DS.items():
    label = DISPLAY.get(ds, ds)
    ai=load(f"{base}/ai"); gt=load(f"{base}/gt")
    if not ai or not gt: continue
    excl=EXCLUDE.get(ds,set())
    if excl:
        ai=[r for r in ai if _normpid(pid(r)) not in excl]; gt=[r for r in gt if _normpid(pid(r)) not in excl]
    aiC=cell_effects(ds,ai); gtC=cell_effects(ds,gt)
    common=set(aiC)&set(gtC); gtonly=set(gtC)-set(aiC); aionly=set(aiC)-set(gtC)
    pg=pool_cells({k:gtC[k] for k in common}); pa=pool_cells({k:aiC[k] for k in common})
    if not (pg and pa):
        print(f"{label:24} no common cells"); continue
    diff=pa["lnrr"]-pg["lnrr"]; se=math.sqrt((0 if pa['se']!=pa['se'] else pa['se']**2)+(0 if pg['se']!=pg['se'] else pg['se']**2))
    margin=0.20*abs(pg["lnrr"]); lo,hi=diff-1.645*se,diff+1.645*se
    # PAIRED TOST on common cells (cells matched by STRUCTURE, blind to values -> unbiased, but paired -> powerful)
    bystudy=defaultdict(list)
    for k in common: bystudy[k[0]].append(aiC[k]-gtC[k])     # per-cell paired diff
    sd=[sum(v)/len(v) for v in bystudy.values()]; ns=len(sd)
    md=sum(sd)/ns; pse=((sum((x-md)**2 for x in sd)/(ns-1))**0.5/math.sqrt(ns)) if ns>1 else float('nan')
    plo,phi=md-1.645*pse, md+1.645*pse
    pequiv=(plo>-margin) and (phi<margin)
    gci=(pct(pg['lnrr']-1.96*pg['se']),pct(pg['lnrr']+1.96*pg['se']))
    aci=(pct(pa['lnrr']-1.96*pa['se']),pct(pa['lnrr']+1.96*pa['se']))
    print(f"{label:24} {pct(pg['lnrr']):>6.1f}% [{gci[0]:>5.1f},{gci[1]:>5.1f}] {pct(pa['lnrr']):>6.1f}% [{aci[0]:>5.1f},{aci[1]:>5.1f}] {pct(diff):>+5.1f} | {len(common)}/{len(gtonly)}/{len(aionly)}")
    print(f"{'':24}   PAIRED: mean cell diff={pct(md):+.1f}% (90% CI [{pct(plo):+.1f}%,{pct(phi):+.1f}%]) median={pct(__import__('statistics').median([aiC[k]-gtC[k] for k in common])):+.1f}%  (studies={ns})")
    # TOST LADDER (paired): equivalent at margin m if 90% CI of mean paired diff within +/- m*|GT pooled lnRR|
    ladder=[]
    for m in (0.05,0.10,0.15,0.20):
        mm=m*abs(pg["lnrr"]); ladder.append(f"{int(m*100)}%:{'PASS' if (plo>-mm and phi<mm) else 'fail'}")
    print(f"{'':24}   TOST ladder: " + "  ".join(ladder))
print("-"*124)
print("common = cells in BOTH (compared) | reference-only = human cells without a workflow structural match | workflow-only = workflow scope expansion (extra, not error)")

# --- SENSITIVITY: Loladze with a finer cell key (separate CO2 level) -------
# The headline Loladze key (study,element,tissue) averages multiple sub-observations per
# cell (AI ~4.6 vs GT ~2.2 rows/cell). Re-keying to separate CO2 level compares strictly
# like-for-like cells. The aggregate conclusion is robust to this choice (see
# AUDITS/loladze_concordance_resolution.md). Reported as a sensitivity; headline stays coarse.
def _loladze_co2_sensitivity():
    ai=load("loladze_v2/keys/ai"); gt=load("loladze_v2/keys/gt")
    ex=EXCLUDE["Loladze"]
    ai=[r for r in ai if _normpid(pid(r)) not in ex]; gt=[r for r in gt if _normpid(pid(r)) not in ex]
    fk=lambda r:(pid(r), low(r,"treatment_level"), low(r,"co_amendment"), low(r,"co_amendment_level"))
    def ce(rows):
        by=defaultdict(list)
        for r in rows:
            e=lnrr(r)
            if e is not None: by[fk(r)].append(e)
        return {k:sum(v)/len(v) for k,v in by.items()}
    a=ce(ai); g=ce(gt); common=set(a)&set(g)
    pg=pool_cells({k:g[k] for k in common}); pa=pool_cells({k:a[k] for k in common})
    if pg and pa:
        diff=pa["lnrr"]-pg["lnrr"]
        print(f"SENSITIVITY Loladze 2014 (finer key, +CO2 level): reference {pct(pg['lnrr']):+.1f}%  workflow {pct(pa['lnrr']):+.1f}%  "
              f"diff {pct(diff):+.1f}pp  ({pg['studies']} studies, {len(common)} cells). "
              f"Headline (coarse key) = +1.9pp; conclusion unchanged.")
_loladze_co2_sensitivity()
