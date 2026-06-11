#!/usr/bin/env python3
"""
AGGREGATE equivalence TOST (the 'matching-free' reliability test), scope-aware.

Run the meta-analysis INDEPENDENTLY on each side and compare the pooled effect:
  GT pooled lnRR  = pool ALL GT rows (cluster by study).
  AI pooled lnRR  = pool ALL AI rows that are IN GT'S SCOPE (cluster by study);
                    AI rows for outcomes/topics the GT never measured are REMOVED
                    (true scope expansion), NOT counted against the AI.
No row pairing -> no alignment bias. Effect = lnRR (ratio-encoded GT -> log(1+ratio)).
Scope-aware corrections: biochar control harmonized to absolute; Li J author-year crosswalk.
TOST: 90% CI of the difference of the two independent pooled estimates within
      +-(margin x |GT pooled effect|).  margins 5/10/15/20%.
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
DISPLAY = {"Boldorini": "Boldorini et al. 2024", "Biochar": "Li X et al. 2024", "Loladze": "Loladze 2014",
           "Hui": "Hui et al. 2025", "Li2022": "Li J et al. 2022"}
BASE = {"Boldorini": "boldorini/keys", "Biochar": "biochar_v2/keys", "Loladze": "loladze_v2/keys",
        "Hui": "hui_v4/keys", "Li2022": "li2022_v2/keys"}
EXCLUDE = {"Hui": {"zhao_2020","cakmak_1997","liu_2014","dong_2018","li_2013","zhang_2012","khoshgoftarmanesh_2013","kumar_2018"},
           "Loladze": {"johnson_1997","ma_2007","rodenkirchen_2009","de_2000","kuehny_1991","li_2010"},
           "Li2022": {"pramanick_2016","al-tawaha-et-al-2011"}, "Biochar": {"jose_2013"}}
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
def effect(r):
    if low(r, "unit_canonical") == "ratio":
        x = ff(r.get("treatment_mean")); return math.log(1+x) if (x is not None and 1+x > 0) else None
    t, c = ff(r.get("treatment_mean")), ff(r.get("control_mean"))
    return math.log(t/c) if (t is not None and c is not None and t > 0 and c > 0) else None
def _bcbase(r): return (npid(low(r,"paper_id")), low(r,"crop"), low(r,"timepoint"))
def biochar_abs_ctrl(rows):
    ac = {}
    for r in rows:
        if ff(r.get("co_amendment_level")) == 0:
            c = ff(r.get("control_mean"))
            if c is not None: ac.setdefault(_bcbase(r), c)
    return ac
def lij_crosswalk(ai):
    L = re.compile(r"[a-z][a-z\-]+")
    def fl(n):
        m = L.search(n.lower()); return m.group(0) if m else None
    idx = {}
    for r in ai:
        p = npid(low(r,"paper_id")); yr = re.search(r"(19|20)\d{2}", p); la = fl(p)
        if la: idx[(la, yr.group(0) if yr else None)] = p; idx.setdefault((la, None), p)
    def remap(r):
        pid = npid(low(r,"paper_id"))
        if not pid.startswith("study") and not pid.startswith("gt_study"): return pid
        m = re.search(r"author='([^']+)'\s*[, ]*((?:19|20)\d{2})", r.get("evidence",""))
        if not m: return pid
        return idx.get((fl(m.group(1)), m.group(2))) or idx.get((fl(m.group(1)), None)) or pid
    return remap

# SCOPE unit = the measurement TYPE the GT tracked (coarser than a cell). AI rows whose scope
# is absent from the GT are out-of-scope (removed). study() = clustering unit.
def scope_key(ds):
    if ds == "Loladze": return lambda r:(npid(low(r,"paper_id")), low(r,"treatment_level"), low(r,"co_amendment"))   # paper,element,tissue
    if ds == "Biochar": return lambda r:(npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"crop"))
    if ds == "Hui":     return lambda r:(npid(low(r,"paper_id")), low(r,"outcome_canonical"))
    if ds == "Boldorini":return lambda r:(npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"treatment_level"))
    return None   # Li2022 by remapped study below

print(f"{'Dataset':22} {'AIn/GTn':>9} {'AI%':>7} {'GT%':>7} {'diff (90% CI)':>22}  TOST 5/10/15/20  in/out-scope AI rows")
print("-"*112)
pct = lambda x: (math.exp(x)-1)*100
for ds in ["Boldorini","Biochar","Hui","Loladze","Li2022"]:
    excl = {npid(p) for p in EXCLUDE.get(ds,set())}
    ai = load(f"{BASE[ds]}/ai", excl); gt = load(f"{BASE[ds]}/gt", excl)
    remap = lij_crosswalk(ai) if ds == "Li2022" else None
    if ds == "Biochar":
        ac_ai, ac_gt = biochar_abs_ctrl(ai), biochar_abs_ctrl(gt)
        def eff_ai(r): t=ff(r.get("treatment_mean")); c=ac_ai.get(_bcbase(r)); return math.log(t/c) if (t and c and t>0 and c>0) else None
        def eff_gt(r): t=ff(r.get("treatment_mean")); c=ac_gt.get(_bcbase(r)); return math.log(t/c) if (t and c and t>0 and c>0) else None
    else:
        eff_ai = eff_gt = effect
    if ds == "Li2022":
        sc = lambda r:(remap(r),); sc_ai = lambda r:(npid(low(r,"paper_id")),)
        gt_scope = {sc(r) for r in gt}
        study_ai = lambda r: npid(low(r,"paper_id")); study_gt = lambda r: remap(r)
        in_scope = lambda r: (npid(low(r,"paper_id")),) in {(remap(g),) for g in gt}  # AI paper present in GT (crosswalked)
    else:
        sc = scope_key(ds); gt_scope = {sc(r) for r in gt}
        study_ai = study_gt = lambda r: npid(low(r,"paper_id"))
        in_scope = lambda r: sc(r) in gt_scope

    # shared paper set: only studies BOTH sides extracted (fair like-for-like corpus)
    ai_studies = {study_ai(r) for r in ai if eff_ai(r) is not None}
    gt_studies = {study_gt(r) for r in gt if eff_gt(r) is not None}
    shared_papers = ai_studies & gt_studies

    def pool(rows, eff_fn, study_fn, filt):
        by = defaultdict(list); kept = dropped = 0
        for r in rows:
            e = eff_fn(r)
            if e is None: continue
            if study_fn(r) not in shared_papers: continue
            if filt is not None and not filt(r): dropped += 1; continue
            kept += 1; by[study_fn(r)].append(e)
        sm = [sum(v)/len(v) for v in by.values()]; n = len(sm)
        if n == 0: return None
        m = sum(sm)/n
        se = (statistics.pstdev(sm)*math.sqrt(n/(n-1))/math.sqrt(n)) if n > 1 else float("nan")
        return dict(m=m, se=se, n=n, kept=kept, dropped=dropped)

    G = pool(gt, eff_gt, study_gt, None)
    A = pool(ai, eff_ai, study_ai, in_scope)
    if not (G and A): print(f"{DISPLAY[ds]:22} insufficient"); continue
    diff = A["m"] - G["m"]; se = math.sqrt((A["se"]**2 if A["se"]==A["se"] else 0)+(G["se"]**2 if G["se"]==G["se"] else 0))
    plo, phi = diff-1.645*se, diff+1.645*se
    ladder = " ".join("P" if (plo > -m*abs(G["m"]) and phi < m*abs(G["m"])) else "." for m in (0.05,0.10,0.15,0.20))
    print(f"{DISPLAY[ds]:22} {A['n']:>4}/{G['n']:<4} {pct(A['m']):>6.1f}% {pct(G['m']):>6.1f}% {pct(diff):>+6.2f} [{pct(plo):>+5.2f},{pct(phi):>+5.2f}]   {ladder}   in={A['kept']} out={A['dropped']}")
print("-"*112)
print("Unpaired aggregate of EVERYTHING in scope (AI out-of-scope rows removed). P=90% CI of pooled diff within +-margin*|GT|.")
