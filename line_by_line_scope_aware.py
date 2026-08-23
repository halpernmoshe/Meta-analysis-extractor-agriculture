#!/usr/bin/env python3
"""
Unified BIAS-FREE line-by-line agreement across all five datasets.

Principle: the AI<->GT pairing is decided ONLY by categorical moderator keys (never by
outcome values), so no alignment/detection bias is possible. Outcome values are used only
to COMPARE already-paired cells, never to form the pairing.

Two corrections vs the original report.json line-by-line:
  (1) EFFECT, not raw mean. Some GT datasets store the effect, not raw concentrations:
        Loladze: unit_canonical='ratio', treatment_mean = (E-A)/A, control_mean empty.
        Li J:    unit_canonical='unresolved', yields on a normalised scale.
      Comparing raw means across these scales gives spurious r~=0. We compare the EFFECT
      (% change) which is scale-invariant and is what the meta-analysis uses.
  (2) Granularity / label reconciliation, per dataset, using only categorical info:
        Hui     -> topic cell (study, application-type); GT pools sub-conditions, so the
                   bias-free statistic is the cell mean.
        Loladze -> MID key (study, element, tissue, cultivar); pool over CO2-level/year
                   (the AI pooled there; Loladze's table pools per its Additional-Info rule).
        Li J    -> study level (one effect per paper) + author-year crosswalk, because the
                   AI/GT product & crop TOKENS diverge (other<->seaweed, amino_acid<->
                   protein_hydrolysate) although the effects agree.

Mislabelled-PDF papers are EXCLUDED (same documented set as scope_matched_equivalence.py).
Read-only; prints a single consolidated table.
"""
import csv, glob, math, os, re
from collections import defaultdict

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
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
def npid(s): return re.sub(r'^[\d_]+', '', str(s).strip().lower())
def ff(v):
    try:
        x = float(str(v).strip()); return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None
def low(r, k): return str(r.get(k, "")).strip().lower()
def numtok(r, k):
    x = ff(r.get(k)); return ("%g" % round(x, 4)) if x is not None else low(r, k)

def effect(r):
    if low(r, "unit_canonical") == "ratio": return ff(r.get("treatment_mean"))
    t, c = ff(r.get("treatment_mean")), ff(r.get("control_mean"))
    return (t - c) / c if (t is not None and c is not None and c != 0) else None

# Biochar control harmonization: GT uses an absolute control (vs bare soil), the AI a matched
# control (biochar vs same co-amendment). The AI ALREADY extracted the absolute control (the
# co_amendment_level==0 rows). Re-derive the effect on each side against its OWN absolute
# control so the estimands match. (Scientifically the AI's matched control isolates biochar
# better; we harmonize to the human's definition only to demonstrate agreement.)
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
def raw_ok(r):  # raw treatment mean usable (not an effect-encoded GT)
    return low(r, "unit_canonical") not in ("ratio", "unresolved") and ff(r.get("treatment_mean")) is not None

def load(side_dir, excl):
    rows = []
    for f in glob.glob(os.path.join(RUNS, side_dir, "*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            if npid(r.get("paper_id", "")) in excl: continue
            rows.append(r)
    return rows
def pear(p):
    xs = [a for a, _ in p]; ys = [b for _, b in p]; n = len(xs)
    if n < 2: return float("nan")
    mx, my = sum(xs)/n, sum(ys)/n
    sx = sum((x-mx)**2 for x in xs); sy = sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(sx*sy) if sx and sy else float("nan")
def maepp(p): return sum(abs(a-b) for a, b in p)/len(p)*100 if p else float("nan")

# ---- Li J author-year crosswalk (gt_studyNN -> AI author-year paper_id) ----
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

# per-dataset categorical key (no values); pid_fn lets Li J remap
def keyfn(ds):
    if ds == "Boldorini": return lambda r: (npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"crop"), low(r,"treatment_level"), low(r,"co_amendment"), numtok(r,"co_amendment_level"), low(r,"timepoint"), low(r,"unit_canonical"))
    if ds == "Biochar":   return lambda r: (npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"crop"), low(r,"treatment_level"), low(r,"co_amendment"), numtok(r,"co_amendment_level"), low(r,"timepoint"))
    if ds == "Hui":       return lambda r: (npid(low(r,"paper_id")), low(r,"outcome_canonical"), low(r,"treatment_level"))
    if ds == "Loladze":   return lambda r: (npid(low(r,"paper_id")), low(r,"treatment_level"), low(r,"co_amendment"), low(r,"co_amendment_level"))
    return None  # Li2022 handled specially (study-level + crosswalk)

# Headline metric per dataset: raw treatment-mean r where both sides store raw means;
# effect r where the GT stores the effect (Loladze ratio) or normalised units (Li J).
# Biochar's EFFECT is confounded by a documented control-definition difference (absolute
# vs matched control) -> raw means are the fidelity metric there, effects are not.
METRIC = {"Boldorini": "raw", "Biochar": "raw", "Hui": "raw", "Loladze": "effect", "Li2022": "effect"}

print("="*98)
print("UNIFIED BIAS-FREE LINE-BY-LINE  (categorical-key pairing; values never used to pair)")
print("="*98)
print(f"{'Dataset':22} {'n':>4} {'cov':>5} {'HEADLINE r':>11} {'MAE pp':>7}  metric / notes")
print("-"*98)
results = {}
for ds in ["Boldorini", "Biochar", "Hui", "Loladze", "Li2022"]:
    excl = {npid(p) for p in EXCLUDE.get(ds, set())}
    ai = load(f"{BASE[ds]}/ai", excl); gt = load(f"{BASE[ds]}/gt", excl)
    note = ""
    if ds == "Li2022":
        remap = lij_crosswalk(ai)
        kf = lambda r, _rm=None: (npid(low(r, "paper_id")),)        # AI: study-level
        kg = lambda r: (remap(r),)                                  # GT: remapped study-level
        aiC = defaultdict(lambda: {"eff": [], "raw": []}); gtC = defaultdict(lambda: {"eff": [], "raw": []})
        for r in ai:
            e = effect(r)
            if e is not None: aiC[kf(r)]["eff"].append(e)
        for r in gt:
            e = effect(r)
            if e is not None: gtC[kg(r)]["eff"].append(e)
        note = "study-level (+author-year crosswalk; product/crop tokens diverge)"
        raw_supported = False
    else:
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
        raw_supported = any(gtC[c]["raw"] for c in gtC)
        note = {"Hui": "cell-mean (GT pools sub-conditions)",
                "Loladze": "effect-based (GT stores ratio); MID key pools CO2/year",
                "Biochar": "raw means; dose-level structural key",
                "Boldorini": "raw means"}[ds]

    allcells = set(aiC) & set(gtC)
    eff_cells = [c for c in allcells if aiC[c]["eff"] and gtC[c]["eff"]]
    raw_cells = [c for c in allcells if aiC[c]["raw"] and gtC[c]["raw"]]
    eff_pairs = [(sum(aiC[c]["eff"])/len(aiC[c]["eff"]), sum(gtC[c]["eff"])/len(gtC[c]["eff"])) for c in eff_cells]
    raw_pairs = [(sum(aiC[c]["raw"])/len(aiC[c]["raw"]), sum(gtC[c]["raw"])/len(gtC[c]["raw"])) for c in raw_cells]
    er, em = pear(eff_pairs), maepp(eff_pairs)
    rr, rm = pear(raw_pairs), maepp(raw_pairs)
    head_cells = raw_cells if METRIC[ds] == "raw" else eff_cells
    cov = len(head_cells) / len(gtC) if gtC else float("nan")
    head_r = rr if METRIC[ds] == "raw" else er
    mlabel = "treatment-mean r" if METRIC[ds] == "raw" else "effect r (GT stores effect)"
    aipool = sum(a for a, _ in eff_pairs)/len(eff_pairs)*100 if eff_pairs else float("nan")
    gtpool = sum(b for _, b in eff_pairs)/len(eff_pairs)*100 if eff_pairs else float("nan")
    # secondary: for Biochar show the harmonized-control effect agreement too
    extra = f"  [harmonized-ctrl effect r={er:.3f}, {len(eff_cells)} cells, AI{aipool:+.0f}% vs GT{gtpool:+.0f}%]" if ds == "Biochar" else ""
    results[ds] = dict(n=len(head_cells), cov=cov, eff_r=er, eff_mae=em, raw_r=rr, head_r=head_r, aipool=aipool, gtpool=gtpool)
    print(f"{DISPLAY[ds]:22} {len(head_cells):>4} {cov:>5.0%} {head_r:>11.3f} {em:>7.2f}pp  {mlabel}; {note}{extra}")
print("-"*98)
print("HEADLINE r = bias-free agreement (categorical pairing). effect r/MAE on cell-mean effects; tmean r on raw means.")
print("pooled effect (AI vs GT) over shared cells:")
for ds in ["Boldorini", "Biochar", "Hui", "Loladze", "Li2022"]:
    r = results[ds]
    print(f"  {DISPLAY[ds]:22} AI={r['aipool']:+6.1f}%  GT={r['gtpool']:+6.1f}%  diff={r['aipool']-r['gtpool']:+5.2f}pp")
