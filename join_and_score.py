#!/usr/bin/env python3
"""
Deterministic join + scoring for the unbiased matching protocol.

Bias-critical step (protocol P1/P8): pairing is a pure deterministic equality
join on the canonical key. No outcome value is consulted to CHOOSE a pairing;
treatment/control means are read only to SCORE already-paired rows.

Inputs : two frozen decoded-key CSVs (ai side, gt side) following
         CANONICAL_SCHEMA.md.
Outputs: <out>/classification.csv  (every GT row -> MATCH/AMBIGUOUS/NO_MATCH)
         <out>/report.json         (counts, match rate, decomposed agreement)
         prints a one-line summary.

Usage:
  python join_and_score.py --ai ai_keys.csv --gt gt_keys.csv --out resultsdir [--dataset name]
"""
import argparse, csv, json, math, os, sys
from collections import defaultdict

KEY_FIELDS = ["paper_id", "outcome_canonical", "crop", "treatment_level",
              "co_amendment", "co_amendment_level", "timepoint", "aggregation_level"]

def norm(v):
    if v is None:
        return ""
    s = str(v).strip().lower()
    # numeric canonicalization: 25.0 -> 25, 2.50 -> 2.5
    try:
        f = float(s)
        if math.isfinite(f):
            return ("%g" % round(f, 6))
    except (ValueError, TypeError):
        pass
    return s

def key_of(row):
    return tuple(norm(row.get(f, "")) for f in KEY_FIELDS)

def to_float(v):
    try:
        f = float(str(v).strip())
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def is_fig(row):
    return norm(row.get("is_figure", "0")) in ("1", "true", "yes")

def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs)
    syy = sum((y-my)**2 for y in ys)
    sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx*syy)

def _longpath(p):
    r"""Windows MAX_PATH (260) workaround: prefix absolute paths with \\?\ so
    files with long names still open. No-op on non-Windows. Bias-neutral I/O fix."""
    if os.name == "nt":
        ap = os.path.abspath(p)
        if not ap.startswith("\\\\?\\"):
            return "\\\\?\\" + ap
        return ap
    return p

def load(path):
    """Load one CSV, or concat every *.csv in a directory (per-paper decode files).
    Recurses only the top level (subdirs like ai_backup_* are ignored)."""
    files = []
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path)
                       if f.lower().endswith(".csv") and os.path.isfile(os.path.join(path, f)))
    else:
        files = [path]
    rows = []
    for fp in files:
        with open(_longpath(fp), newline="", encoding="utf-8-sig") as fh:
            rows.extend(csv.DictReader(fh))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ai_rows = load(a.ai)
    gt_rows = load(a.gt)

    # index AI by key (table rows only; figure rows scored separately)
    ai_by_key = defaultdict(list)
    for r in ai_rows:
        if not is_fig(r):
            ai_by_key[(key_of(r), norm(r.get("unit_canonical", "")))].append(r)

    classified = []
    matched_pairs = []          # (gt, ai) for MATCH
    counts = defaultdict(int)
    fig_gt = 0

    for g in gt_rows:
        if is_fig(g):
            fig_gt += 1
            classified.append({**{f: g.get(f, "") for f in KEY_FIELDS},
                               "row_id": g.get("row_id", ""), "verdict": "FIGURE_TIER",
                               "n_ai_candidates": ""})
            continue
        cand = ai_by_key.get((key_of(g), norm(g.get("unit_canonical", ""))), [])
        if len(cand) == 1:
            verdict = "MATCH"; matched_pairs.append((g, cand[0]))
        elif len(cand) == 0:
            verdict = "NO_MATCH"
        else:
            verdict = "AMBIGUOUS"
        counts[verdict] += 1
        classified.append({**{f: g.get(f, "") for f in KEY_FIELDS},
                           "row_id": g.get("row_id", ""), "verdict": verdict,
                           "n_ai_candidates": len(cand)})

    total_gt_table = sum(counts.values())
    match_rate = counts["MATCH"] / total_gt_table if total_gt_table else 0.0

    # decomposed agreement on MATCH pairs
    t_ai = [to_float(a_["treatment_mean"]) for g, a_ in matched_pairs]
    t_gt = [to_float(g["treatment_mean"]) for g, a_ in matched_pairs]
    pairs_t = [(x, y) for x, y in zip(t_ai, t_gt) if x is not None and y is not None]
    c_ai = [to_float(a_["control_mean"]) for g, a_ in matched_pairs]
    c_gt = [to_float(g["control_mean"]) for g, a_ in matched_pairs]
    pairs_c = [(x, y) for x, y in zip(c_ai, c_gt) if x is not None and y is not None]

    def agree(pairs):
        if not pairs:
            return {"n": 0, "r": None, "mae": None}
        xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
        mae = sum(abs(x-y) for x, y in pairs)/len(pairs)
        return {"n": len(pairs), "r": pearson(xs, ys), "mae": mae}

    ctrl_conc = None
    if matched_pairs:
        same = sum(1 for g, a_ in matched_pairs
                   if norm(g.get("control_token", "")) == norm(a_.get("control_token", "")))
        ctrl_conc = same/len(matched_pairs)

    report = {
        "dataset": a.dataset,
        "gt_rows_total": len(gt_rows),
        "gt_rows_figure_tier": fig_gt,
        "gt_rows_table": total_gt_table,
        "ai_rows_total": len(ai_rows),
        "counts": dict(counts),
        "match_rate_table": round(match_rate, 4),
        "treatment_mean_agreement": agree(pairs_t),
        "control_mean_agreement": agree(pairs_c),
        "control_token_concordance": ctrl_conc,
        "passes_75pct": match_rate >= 0.75,
    }
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(os.path.join(a.out, "classification.csv"), "w", newline="", encoding="utf-8") as fh:
        if classified:
            w = csv.DictWriter(fh, fieldnames=list(classified[0].keys()))
            w.writeheader(); w.writerows(classified)

    tm = report["treatment_mean_agreement"]
    print(f"[{a.dataset}] match_rate={match_rate:.1%} "
          f"({counts['MATCH']}/{total_gt_table}) | "
          f"NO_MATCH={counts['NO_MATCH']} AMBIG={counts['AMBIGUOUS']} fig={fig_gt} | "
          f"treat-mean r={tm['r']} MAE={tm['mae']} n={tm['n']} | "
          f"ctrl_concordance={ctrl_conc} | PASS75={report['passes_75pct']}")

if __name__ == "__main__":
    main()
