#!/usr/bin/env python3
"""
Bland-Altman limits-of-agreement analysis on the CURRENT blind-matched pairs,
for the Environmental Evidence resubmission (Reviewer 1: numeric 95% LoA).

Two complementary metrics are computed per dataset:

PRIMARY  - treatment-mean agreement ("did the AI read the same number?").
           For datasets whose treatment means are native, strictly-positive
           quantities (Boldorini, Li X, Hui, Li J) the Bland-Altman analysis is
           done on log10(treatment_mean); bias and LoA are reported in log10
           units and converted to a multiplicative ratio and an approximate
           +/-% band. Loladze stores its effect canonically as a dimensionless
           ratio (treatment/control - 1, frequently negative), so a log scale is
           undefined; for Loladze the treatment-"mean" BA is reported on the
           linear ratio scale instead, and this is stated explicitly.

SECONDARY - per-cell percentage-change effect difference (pp), the unit-free
           metric Reviewer 1 referenced. Reported as parametric
           mean +/- 1.96*SD AND robust summaries (median, 10% trimmed mean, IQR)
           because Li X is heavy-tailed (near-zero-control cells inflate lnRR).

Pairings are never chosen here: we read the frozen blind pairings + key CSVs and
reproduce the canonical MATCH counts (Boldorini 19, Li X 211, Loladze 346,
Hui 19, Li J 50).

Outputs:
  figures/figS5_bland_altman.png   (primary treatment-mean panels)
  prints both numeric LoA tables to stdout
"""
import csv, glob, json, math, os, statistics
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.abspath(os.path.dirname(__file__))
# Self-contained: the supporting blind-matched pairings and keys are bundled in
# this repository under runs/ (biochar_v3/pairings, loladze_v3/pairings,
# hui_v4/pairings, li2022_v4/pairings + li2022_v4/keys, boldorini join keys).
RUNS = os.path.join(HERE, "runs")
FIG_OUT = os.path.join(HERE, "figures", "figS5_bland_altman.png")


# ----------------------------- helpers --------------------------------------
def ff(v):
    try:
        x = float(str(v).strip())
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def isfig(r):
    return str(r.get("is_figure", "0")).strip().lower() in ("1", "true", "yes")


def lnrr(r):
    """Canonical effect, identical to the manuscript pipeline."""
    t = ff(r.get("treatment_mean")); c = ff(r.get("control_mean"))
    if t is not None and c is not None and t > 0 and c > 0:
        return math.log(t / c)
    if str(r.get("unit_canonical", "")).strip().lower() == "ratio" and t is not None and (1 + t) > 0:
        return math.log(1 + t)
    return None


def eff_pct(r):
    e = lnrr(r)
    return None if e is None else (math.exp(e) - 1) * 100.0


def load_keys(d):
    by = {}
    for f in glob.glob(d + "/*.csv"):
        for _t in range(5):
            try:
                with open(f, encoding="utf-8-sig") as fh:
                    for r in csv.DictReader(fh):
                        if r.get("row_id"):
                            by[r["row_id"]] = r
                break
            except (FileNotFoundError, PermissionError):
                import time; time.sleep(0.4)
    return by


def pairs_from_pairings(pdir, ai, gt):
    pf = {}
    for f in glob.glob(pdir + "/*.jsonl"):
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            if o.get("gt_row_id"):
                pf[o["gt_row_id"]] = o
    out = []
    for gid, g in gt.items():
        if isfig(g):
            continue
        p = pf.get(gid)
        if p and p.get("ai_row_id") and not p.get("ambiguous") and ai.get(p["ai_row_id"]):
            out.append((g, ai[p["ai_row_id"]]))
    return out


KEY = ["paper_id", "outcome_canonical", "crop", "treatment_level", "co_amendment",
       "co_amendment_level", "timepoint", "aggregation_level"]


def norm(v):
    s = str(v).strip().lower()
    try:
        f = float(s)
        if math.isfinite(f):
            return "%g" % round(f, 6)
    except (ValueError, TypeError):
        pass
    return s


def kf(r):
    return tuple(norm(r.get(f, "")) for f in KEY)


def pairs_from_join(ai, gt):
    aiby = defaultdict(list)
    for r in ai.values():
        if not isfig(r):
            aiby[(kf(r), norm(r.get("unit_canonical", "")))].append(r)
    out = []
    for g in gt.values():
        if isfig(g):
            continue
        c = aiby.get((kf(g), norm(g.get("unit_canonical", ""))), [])
        if len(c) == 1:
            out.append((g, c[0]))
    return out


def trimmed_mean(xs, frac=0.10):
    xs = sorted(xs)
    n = len(xs)
    k = int(math.floor(n * frac))
    core = xs[k:n - k] if n - 2 * k >= 1 else xs
    return statistics.mean(core), len(core)


def iqr_bounds(xs):
    a = np.asarray(xs, float)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return float(q1), float(med), float(q3)


# ----------------------------- dataset registry -----------------------------
# display, mode, pairings_dir, ai_dir, gt_dir, expected_match_n, loggable
DS = [
    ("Boldorini et al. 2024", "join", None, "boldorini/keys/ai", "boldorini/keys/gt", 19, True),
    ("Li X et al. 2024",      "pair", "biochar_v3/pairings", "biochar_v2/keys/ai", "biochar_v2/keys/gt", 211, True),
    ("Loladze 2014",          "pair", "loladze_v3/pairings", "loladze_v2/keys/ai", "loladze_v2/keys/gt", 346, False),
    ("Hui et al. 2025",       "pair", "hui_v4/pairings", "hui_v4/keys/ai", "hui_v4/keys/gt", 19, True),
    ("Li J et al. 2022",      "pair", "li2022_v4/pairings", "li2022_v4/keys/ai", "li2022_v4/keys/gt", 50, True),
]


def get_pairs(disp, mode, pdir, aid, gtd):
    ai = load_keys(f"{RUNS}/{aid}")
    gt = load_keys(f"{RUNS}/{gtd}")
    return pairs_from_join(ai, gt) if mode == "join" else pairs_from_pairings(f"{RUNS}/{pdir}", ai, gt)


def main():
    results = {}
    for disp, mode, pdir, aid, gtd, expn, loggable in DS:
        prs = get_pairs(disp, mode, pdir, aid, gtd)
        # ---- PRIMARY: treatment-mean ----
        tm_ai, tm_gt = [], []
        for g, a in prs:
            tg = ff(g.get("treatment_mean")); ta = ff(a.get("treatment_mean"))
            if tg is not None and ta is not None:
                tm_gt.append(tg); tm_ai.append(ta)
        # ---- SECONDARY: pp-effect diff ----
        pp = []
        for g, a in prs:
            pg = eff_pct(g); pa = eff_pct(a)
            if pg is not None and pa is not None:
                pp.append(pa - pg)
        results[disp] = dict(prs=prs, n_match=len(prs), expn=expn, loggable=loggable,
                             tm_ai=tm_ai, tm_gt=tm_gt, pp=pp)

    # ================= PRIMARY table (treatment-mean) =================
    print("\n" + "=" * 96)
    print("PRIMARY metric: Bland-Altman on EXTRACTED TREATMENT MEANS  (\"did the AI read the same number?\")")
    print("Log datasets: bias/LoA in log10 units -> multiplicative ratio (10^x) -> approx +/-% band.")
    print("Loladze: ratio scale (dimensionless effect; log undefined) -> bias/LoA in ratio units.")
    print("=" * 96)
    hdr = f"{'Dataset':22}{'n':>5}{'scale':>8}{'bias':>10}{'LoA_lo':>10}{'LoA_hi':>10}{'interpretation':>30}"
    print(hdr); print("-" * len(hdr))
    primary_rows = {}
    for disp, _m, _p, _a, _g, _e, loggable in DS:
        R = results[disp]
        ai = np.asarray(R["tm_ai"], float); gt = np.asarray(R["tm_gt"], float)
        n = len(ai)
        if loggable:
            la = np.log10(ai); lg = np.log10(gt)
            d = la - lg                       # log10 ratio AI/GT
            mean = float(np.mean(d)); sd = float(np.std(d, ddof=1))
            lo, hi = mean - 1.96 * sd, mean + 1.96 * sd
            ratio = 10 ** mean
            pct_lo = (10 ** lo - 1) * 100; pct_hi = (10 ** hi - 1) * 100
            interp = f"x{ratio:.3f}; ~[{pct_lo:+.1f},{pct_hi:+.1f}]%"
            print(f"{disp:22}{n:>5}{'log10':>8}{mean:>+10.4f}{lo:>+10.4f}{hi:>+10.4f}{interp:>30}")
            primary_rows[disp] = dict(n=n, scale="log10", bias=mean, lo=lo, hi=hi,
                                      ratio=ratio, pct_lo=pct_lo, pct_hi=pct_hi)
        else:
            d = ai - gt
            mean = float(np.mean(d)); sd = float(np.std(d, ddof=1))
            lo, hi = mean - 1.96 * sd, mean + 1.96 * sd
            interp = "ratio units (effect)"
            print(f"{disp:22}{n:>5}{'ratio':>8}{mean:>+10.4f}{lo:>+10.4f}{hi:>+10.4f}{interp:>30}")
            primary_rows[disp] = dict(n=n, scale="ratio", bias=mean, lo=lo, hi=hi)

    # ================= SECONDARY table (pp-effect) =================
    print("\n" + "=" * 96)
    print("SECONDARY metric: 95% LoA on per-cell PERCENTAGE-CHANGE EFFECT difference (pp = AI%% - GT%%)")
    print("Parametric mean+/-1.96SD AND robust (median, 10%% trimmed mean, IQR). Li X is heavy-tailed.")
    print("=" * 96)
    hdr2 = (f"{'Dataset':22}{'n':>5}{'bias':>8}{'SD':>8}{'LoA_lo':>9}{'LoA_hi':>9}"
            f"{'median':>9}{'trim10':>9}{'IQR_lo':>9}{'IQR_hi':>9}")
    print(hdr2); print("-" * len(hdr2))
    secondary_rows = {}
    for disp, _m, _p, _a, _g, _e, _l in DS:
        pp = results[disp]["pp"]
        n = len(pp)
        if n < 2:
            print(f"{disp:22}{n:>5}  too few"); continue
        bias = statistics.mean(pp); sd = statistics.stdev(pp)
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        q1, med, q3 = iqr_bounds(pp)
        tm, _ = trimmed_mean(pp, 0.10)
        print(f"{disp:22}{n:>5}{bias:>+8.2f}{sd:>8.2f}{lo:>+9.2f}{hi:>+9.2f}"
              f"{med:>+9.2f}{tm:>+9.2f}{q1:>+9.2f}{q3:>+9.2f}")
        secondary_rows[disp] = dict(n=n, bias=bias, sd=sd, lo=lo, hi=hi,
                                    median=med, trim10=tm, iqr_lo=q1, iqr_hi=q3)

    # ================= FIGURE (primary treatment-mean panels) =================
    order = ["Boldorini et al. 2024", "Li X et al. 2024", "Loladze 2014",
             "Hui et al. 2025", "Li J et al. 2022"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for i, disp in enumerate(order):
        ax = axes[i]
        R = results[disp]
        loggable = dict((d[0], d[6]) for d in DS)[disp]
        ai = np.asarray(R["tm_ai"], float); gt = np.asarray(R["tm_gt"], float)
        n = len(ai)
        if loggable:
            la = np.log10(ai); lg = np.log10(gt)
            mean_axis = (la + lg) / 2.0
            d = la - lg
            xlabel = "Mean of AI & reference treatment mean  (log10 units)"
            ylabel = "log10(AI / reference)"
            unit = "log10"
        else:
            mean_axis = (ai + gt) / 2.0
            d = ai - gt
            xlabel = "Mean of AI & reference effect  (ratio units)"
            ylabel = "AI - reference  (ratio units)"
            unit = "ratio"
        bias = float(np.mean(d)); sd = float(np.std(d, ddof=1))
        lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
        ax.scatter(mean_axis, d, s=26, alpha=0.55, edgecolor="k", linewidth=0.3, color="#2b6cb0")
        ax.axhline(bias, color="#c53030", lw=1.4, label=f"bias {bias:+.3f}")
        ax.axhline(hi, color="#718096", ls="--", lw=1.1)
        ax.axhline(lo, color="#718096", ls="--", lw=1.1, label=f"95% LoA [{lo:+.3f}, {hi:+.3f}]")
        ax.axhline(0, color="#000", lw=0.6, alpha=0.3)
        ax.set_title(f"{disp}  (n={n}, {unit})", fontsize=11)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
        ax.tick_params(labelsize=8)
    axes[5].axis("off")
    axes[5].text(0.02, 0.95,
                 "Figure S5. Bland-Altman agreement of AI-extracted vs reference\n"
                 "TREATMENT MEANS, per dataset (primary metric).\n\n"
                 "Boldorini, Li X, Hui, Li J: log10 scale (native positive units);\n"
                 "y = log10(AI/reference), so 0 = exact agreement.\n\n"
                 "Loladze: linear ratio scale (canonical effect is a dimensionless\n"
                 "ratio, frequently negative, so log is undefined).\n\n"
                 "Red = mean bias; dashed = 95% limits of agreement\n"
                 "(mean +/- 1.96 SD).",
                 va="top", ha="left", fontsize=9, family="monospace")
    fig.suptitle("Bland-Altman: AI vs reference treatment-mean agreement (blind-matched pairs)",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(FIG_OUT), exist_ok=True)
    fig.savefig(FIG_OUT, dpi=200)
    print(f"\nFigure written: {FIG_OUT}")
    return primary_rows, secondary_rows


if __name__ == "__main__":
    main()
