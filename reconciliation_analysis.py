#!/usr/bin/env python3
"""
Reconciliation analysis (manuscript Part 2 Table 4 / Figure 3 / the power paragraph).

Deterministic (sorted file reads; sorted element rows with an explicit tiebreaker). Locks the
numbers behind the reconciliation claims, all from the same key tables and the same lnRR effect
used by scope_aware_paired_tost.py:

  (A) BIOCHAR control definition (Li X 2024): AI matched control vs human absolute control.
  (B) LOLADZE per-element CO2 effect (cluster by study) for ALL shared cells vs the all_data
      subset (cells with NO documented condition selection), plus the overall all_data agreement
      (Pearson r and MAE) cited in Results Part 1.
  (C) POWER decomposition (Loladze): observations vs studies; between-study heterogeneity SD vs
      per-study AI-GT paired-difference SD; unpaired vs paired SE.
  (D) LI J units/tokens: study-level effect agreement after author-year crosswalk (Pearson r),
      with the worked grape example (+4.94% on both sides despite raw 4460 vs 1.47).

Read-only; deterministic; prints all four blocks.
"""
import csv, glob, math, os, re, statistics
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
def npid(s): return re.sub(r'^[\d_]+', '', str(s).strip().lower())
def ff(v):
    try:
        x = float(str(v).strip()); return x if math.isfinite(x) else None
    except (ValueError, TypeError): return None
def low(r, k): return str(r.get(k, "")).strip().lower()
def load(side, excl):
    rows = []
    for f in sorted(glob.glob(os.path.join(RUNS, side, "*.csv"))):
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
def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    sx = sum((x-mx)**2 for x in xs); sy = sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(sx*sy) if sx and sy else float("nan")
def mae(xs, ys): return sum(abs(a-b) for a, b in zip(xs, ys))/len(xs) if xs else float("nan")
LOL_EXCL = {npid(p) for p in {"johnson_1997","ma_2007","rodenkirchen_2009","de_2000","kuehny_1991","li_2010"}}
BIO_EXCL = {npid(p) for p in {"jose_2013"}}
LIJ_EXCL = {npid(p) for p in {"pramanick_2016","al-tawaha-et-al-2011"}}

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
def bc_matched(r):  # AI native matched control (its own control_mean)
    t, c = ff(r.get("treatment_mean")), ff(r.get("control_mean"))
    return math.log(t/c) if (t and c and t > 0 and c > 0) else None
ai = load("biochar_v2/keys/ai", BIO_EXCL); gt = load("biochar_v2/keys/gt", BIO_EXCL)
ac_ai, ac_gt = biochar_abs_ctrl(ai), biochar_abs_ctrl(gt)
def harm(r, ac):
    t = ff(r.get("treatment_mean")); c = ac.get(_bcbase(r))
    return math.log(t/c) if (t and c and t > 0 and c > 0) else None
aiC, gtC, aiM = defaultdict(list), defaultdict(list), defaultdict(list)
for r in ai:
    e = harm(r, ac_ai)
    if e is not None: aiC[bc_key(r)].append(e)
    m = bc_matched(r)
    if m is not None: aiM[bc_key(r)].append(m)
for r in gt:
    e = harm(r, ac_gt)
    if e is not None: gtC[bc_key(r)].append(e)
shared = set(aiC) & set(gtC)
ba = defaultdict(list); bg = defaultdict(list); bm = defaultdict(list)
for c in shared:
    ba[c[0]].append(sum(aiC[c])/len(aiC[c])); bg[c[0]].append(sum(gtC[c])/len(gtC[c]))
    if aiM.get(c): bm[c[0]].append(sum(aiM[c])/len(aiM[c]))
ai_abs = sum(sum(v)/len(v) for v in ba.values())/len(ba)
gt_abs = sum(sum(v)/len(v) for v in bg.values())/len(bg)
ai_mat = sum(sum(v)/len(v) for v in bm.values())/len(bm)
print("="*84)
print("(A) BIOCHAR (Li X 2024) control-definition reconciliation")
print("="*84)
print(f"  shared cells={len(shared)}  studies={len(ba)}")
print(f"  AI native MATCHED control (isolates biochar) = {pct(ai_mat):+.1f}%")
print(f"  AI under ABSOLUTE control = {pct(ai_abs):+.1f}%   human ABSOLUTE control = {pct(gt_abs):+.1f}%   diff = {pct(ai_abs)-pct(gt_abs):+.1f} pp")

# ============================ (B) LOLADZE per-element ============================
def lol_key(r): return (npid(low(r,"paper_id")), low(r,"treatment_level"), low(r,"co_amendment"), low(r,"co_amendment_level"))
def ginfo(r):
    m = re.search(r"AdditionalInfo='([^']*)'", r.get("evidence","")); return (m.group(1).strip() if m else "")
ai = load("loladze_v2/keys/ai", LOL_EXCL); gt = load("loladze_v2/keys/gt", LOL_EXCL)
aiC = defaultdict(list); gtC = defaultdict(list); gtinfo = defaultdict(set)
for r in ai:
    e = lnrr(r)
    if e is not None: aiC[lol_key(r)].append(e)
for r in gt:
    e = lnrr(r)
    if e is not None: gtC[lol_key(r)].append(e); gtinfo[lol_key(r)].add(ginfo(r))
shared = set(aiC) & set(gtC)
def per_element(cells, label):
    byel = defaultdict(lambda: (defaultdict(list), defaultdict(list)))
    for c in cells:
        bd, bg = byel[c[1]]
        bd[c[0]].append(sum(aiC[c])/len(aiC[c]) - sum(gtC[c])/len(gtC[c])); bg[c[0]].append(sum(gtC[c])/len(gtC[c]))
    print(f"\n  --- {label} ---")
    print(f"  {'elem':5}{'studies':>8}{'AI%':>8}{'GT%':>8}{'diff pp':>9}")
    for el, (bd, bg) in sorted(byel.items(), key=lambda kv: (-len(kv[1][0]), kv[0])):  # deterministic tiebreaker
        ns = len(bd)
        if ns < 3: continue
        md = sum(sum(v)/len(v) for v in bd.values())/ns
        gp = sum(sum(v)/len(v) for v in bg.values())/ns
        print(f"  {el.upper():5}{ns:>8}{pct(gp+md):>+8.1f}{pct(gp):>+8.1f}{pct(gp+md)-pct(gp):>+9.1f}")
print("\n" + "="*84)
print("(B) LOLADZE per-element CO2 effect (lnRR, cluster by study): ALL cells vs all_data subset")
print("="*84)
allc = sorted(shared)
clean = [c for c in allc if gtinfo[c] == {""}]
per_element(allc, "ALL shared cells")
per_element(clean, "all_data only (Loladze applied NO documented condition selection)")
# overall all_data agreement (cell-level effect %), cited in Results Part 1
ax = [pct(sum(aiC[c])/len(aiC[c])) for c in clean]; gx = [pct(sum(gtC[c])/len(gtC[c])) for c in clean]
print(f"\n  all_data OVERALL cell-level agreement: n={len(clean)}  Pearson r={pearson(gx,ax):.3f}  MAE={mae(gx,ax):.2f} pp")

# ============================ (C) POWER decomposition (Loladze) ============================
n_obs_ai = sum(len(v) for v in aiC.values())
bystudy_gt = defaultdict(list); bystudy_diff = defaultdict(list)
for c in allc:
    bystudy_gt[c[0]].append(sum(gtC[c])/len(gtC[c]))
diff_study = []
for st in sorted(bystudy_gt):
    diffs_pp = [(math.exp(sum(aiC[c])/len(aiC[c]))-1)*100 - (math.exp(sum(gtC[c])/len(gtC[c]))-1)*100
                for c in allc if c[0] == st]
    diff_study.append(sum(diffs_pp)/len(diffs_pp))
gt_study = [pct(sum(v)/len(v)) for v in (bystudy_gt[s] for s in sorted(bystudy_gt))]
ns = len(gt_study)
sd_between = statistics.pstdev(gt_study)*math.sqrt(ns/(ns-1))
sd_diff = statistics.pstdev(diff_study)*math.sqrt(ns/(ns-1))
print("\n" + "="*84)
print("(C) POWER decomposition - Loladze: power runs on STUDIES, not observations")
print("="*84)
print(f"  AI observations extracted: {n_obs_ai}   ->   STUDIES (clustering unit): {ns}")
print(f"  between-study effect SD = {sd_between:.1f} pp   -> pooled SE = {sd_between/math.sqrt(ns):.2f} pp")
print(f"  per-study AI-GT paired-difference SD = {sd_diff:.2f} pp -> paired SE = {sd_diff/math.sqrt(ns):.2f} pp")
print(f"  Unpaired difference-CI ~ 1.645*sqrt(2)*{sd_between/math.sqrt(ns):.2f} = {1.645*math.sqrt(2)*sd_between/math.sqrt(ns):.1f} pp (heterogeneity, doubled)")
print(f"  Paired   difference-CI ~ 1.645*{sd_diff/math.sqrt(ns):.2f} = {1.645*sd_diff/math.sqrt(ns):.1f} pp (heterogeneity cancels)")

# ============================ (D) LI J units/tokens ============================
def fl(n):
    m = re.search(r"[a-z][a-z\-]+", n.lower()); return m.group(0) if m else None
ai = load("li2022_v2/keys/ai", LIJ_EXCL); gt = load("li2022_v2/keys/gt", LIJ_EXCL)
idx = {}
for r in ai:
    p = npid(low(r,"paper_id")); yr = re.search(r"(19|20)\d{2}", p); la = fl(p)
    if la: idx[(la, yr.group(0) if yr else None)] = p; idx.setdefault((la, None), p)
def remap(r):
    pid = npid(low(r,"paper_id"))
    if not pid.startswith("study") and not pid.startswith("gt_study"): return pid
    m = re.search(r"author='([^']+)'\s*[, ]*((?:19|20)\d{2})", r.get("evidence",""))
    if not m: return pid
    # Source-title audit: do not merge distinct same-author/year GT studies.
    if pid in {"gt_study08", "gt_study146"}: return pid
    return idx.get((fl(m.group(1)), m.group(2))) or idx.get((fl(m.group(1)), None)) or pid
def lij_eff(r):
    t, c = ff(r.get("treatment_mean")), ff(r.get("control_mean"))
    return (t-c)/c if (t is not None and c is not None and c != 0) else None
aiP = defaultdict(list); gtP = defaultdict(list)
for r in ai:
    e = lij_eff(r)
    if e is not None: aiP[npid(low(r,"paper_id"))].append(e)
for r in gt:
    e = lij_eff(r)
    if e is not None: gtP[remap(r)].append(e)
shared = sorted(set(aiP) & set(gtP))
ax = [sum(aiP[p])/len(aiP[p])*100 for p in shared]; gx = [sum(gtP[p])/len(gtP[p])*100 for p in shared]
print("\n" + "="*84)
print("(D) LI J 2022 units/tokens: study-level effect agreement (after author-year crosswalk)")
print("="*84)
print(f"  shared papers={len(shared)}  study-level Pearson r={pearson(gx,ax):.3f}")
print(f"  (study-level pools all rows per paper; the units point is clearest at the single-row level below)")
grape = "-s2.0-s0304423819306703-main"
g_ai = [r for r in ai if npid(low(r,"paper_id")) == grape and lij_eff(r) is not None]
g_gt = [r for r in gt if remap(r) == grape and lij_eff(r) is not None]
if g_ai and g_gt:
    a0, g0 = g_ai[0], g_gt[0]
    print(f"  worked grape example (single row, same observation): "
          f"AI T={ff(a0.get('treatment_mean')):g} C={ff(a0.get('control_mean')):g} -> {lij_eff(a0)*100:+.2f}%  |  "
          f"human T={ff(g0.get('treatment_mean')):g} C={ff(g0.get('control_mean')):g} -> {lij_eff(g0)*100:+.2f}%")
    print(f"  (identical {lij_eff(a0)*100:+.2f}% effect despite raw scales {ff(a0.get('treatment_mean')):g} vs {ff(g0.get('treatment_mean')):.2f})")
