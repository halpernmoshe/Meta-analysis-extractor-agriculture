#!/usr/bin/env python3
"""
qc_hard_filters.py — Stage 5: Deterministic QC hard filters for Pipeline V2.

Universal, config-driven hard filters applied AFTER extraction and BEFORE
LLM semantic adjudication. These are purely programmatic checks — no LLM needed.

Checks:
  1. Structural completeness (both means present, numeric, positive for lnRR)
  2. Variance integrity (SE/SD/LSD/CV/CI detection and conversion)
  3. Duplicate detection (same paper + same means = likely duplicate)
  4. Effect size computation (lnRR + percentage change)
  5. Outlier flagging (extreme lnRR values)
  6. Provenance tracking (source_type, confidence preserved)

Usage:
    python qc_hard_filters.py <topic_dir>
    python qc_hard_filters.py --all
    python qc_hard_filters.py legume_rotation --dry-run
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent

# ── Constants ────────────────────────────────────────────────────────────────

EXTREME_LNRR_THRESHOLD = 2.0  # |lnRR| > 2 means >600% increase or <-86% decrease; kept as secondary check

# Plausibility thresholds based on V1 lesson (notill_tillage, 2026-03-26):
# AbdulsattarAlrijabo 2014 (Iraq drought) contributed rows with +194% to +609% effects.
# lnRR for +609% = ln(7.09) = 1.96, which slips UNDER the lnRR > 2.0 threshold.
# A percent-change filter catches these directly without relying on log-scale arithmetic.
# +200% chosen as a hard agronomic ceiling: no intervention doubles yield in a single
# season under field conditions. -80% chosen as floor: near-total crop failure is possible
# in extreme drought but effects below -80% are almost always T/C swap artifacts.
EFFECT_PCT_LOWER = -80   # Flag percent_change < -80% (V1: Alrijabo outlier lesson)
EFFECT_PCT_UPPER = 200   # Flag percent_change > 200% (V1: Alrijabo +609% slipped lnRR filter)
CV_LOWER = 0.5   # Flag CV < 0.5%
CV_UPPER = 150    # Flag CV > 150%
DUPLICATE_MEAN_TOLERANCE = 0.001  # Relative tolerance for duplicate detection


# ── Variance conversion ─────────────────────────────────────────────────────

def convert_variance_to_sd(row):
    """Convert any variance type to SD for both treatment and control.

    Returns dict with sd_treatment, sd_control, n_treatment, n_control,
    variance_status, and conversion_notes.
    """
    result = {
        "sd_treatment": None,
        "sd_control": None,
        "n_treatment": None,
        "n_control": None,
        "variance_status": "missing",
        "conversion_notes": "",
    }

    # Get sample sizes
    n_t = _safe_float(row.get("treatment_n"))
    n_c = _safe_float(row.get("control_n"))
    # Fallback: if one is missing, use the other
    if n_t is None and n_c is not None:
        n_t = n_c
    if n_c is None and n_t is not None:
        n_c = n_t
    result["n_treatment"] = n_t
    result["n_control"] = n_c

    # Try direct SD columns first
    sd_t = _safe_float(row.get("sd_treatment"))
    sd_c = _safe_float(row.get("sd_control"))

    # Try SE → SD conversion
    if sd_t is None:
        se_t = _safe_float(row.get("se_treatment"))
        if se_t is not None and n_t is not None and n_t > 0:
            sd_t = se_t * math.sqrt(n_t)
            result["conversion_notes"] += "sd_t from SE*sqrt(n); "

    if sd_c is None:
        se_c = _safe_float(row.get("se_control"))
        if se_c is not None and n_c is not None and n_c > 0:
            sd_c = se_c * math.sqrt(n_c)
            result["conversion_notes"] += "sd_c from SE*sqrt(n); "

    # Try generic variance_value + variance_type
    if sd_t is None or sd_c is None:
        vtype = str(row.get("variance_type", "")).upper().strip()
        vval = _safe_float(row.get("variance_value"))

        if vval is not None and vtype:
            n = n_t if n_t else 3.0  # Default assumption

            if vtype in ("SD", "STD"):
                if sd_t is None:
                    sd_t = vval
                if sd_c is None:
                    sd_c = vval
                result["conversion_notes"] += f"SD from variance_value; "

            elif vtype in ("SE", "SEM"):
                if n > 0:
                    sd_conv = vval * math.sqrt(n)
                    if sd_t is None:
                        sd_t = sd_conv
                    if sd_c is None:
                        sd_c = sd_conv
                    result["conversion_notes"] += f"SD from SE*sqrt({n}); "

            elif vtype == "LSD":
                if n > 0:
                    df = 2 * (n - 1)
                    if df > 0:
                        t_crit = stats.t.ppf(0.975, df)
                        se_diff = vval / (t_crit * math.sqrt(2))
                        sd_conv = se_diff * math.sqrt(n)
                        if sd_t is None:
                            sd_t = sd_conv
                        if sd_c is None:
                            sd_c = sd_conv
                        result["conversion_notes"] += f"SD from LSD/(t*sqrt2)*sqrt(n); "

            elif vtype in ("CV", "CV%"):
                mean_t = _safe_float(row.get("treatment_mean"))
                mean_c = _safe_float(row.get("control_mean"))
                if mean_t and mean_t > 0:
                    sd_t = vval * mean_t / 100.0
                if mean_c and mean_c > 0:
                    sd_c = vval * mean_c / 100.0
                result["conversion_notes"] += "SD from CV*mean/100; "

            elif vtype in ("CI_95", "CI", "95CI"):
                # Assume vval is half-width of 95% CI
                if n and n > 0:
                    se = vval / 1.96
                    sd_conv = se * math.sqrt(n)
                    if sd_t is None:
                        sd_t = sd_conv
                    if sd_c is None:
                        sd_c = sd_conv
                    result["conversion_notes"] += "SD from CI_95 half-width; "

    result["sd_treatment"] = sd_t
    result["sd_control"] = sd_c

    if sd_t is not None and sd_c is not None:
        result["variance_status"] = "present"
    elif sd_t is not None or sd_c is not None:
        result["variance_status"] = "partial"
    else:
        result["variance_status"] = "missing"

    return result


# ── Effect size computation ──────────────────────────────────────────────────

def compute_lnRR(t_mean, c_mean):
    """Compute log response ratio."""
    try:
        t, c = float(t_mean), float(c_mean)
        if t > 0 and c > 0:
            return math.log(t / c)
    except (TypeError, ValueError):
        pass
    return None


def compute_var_lnRR(sd_t, sd_c, n_t, n_c, mean_t, mean_c):
    """Compute variance of lnRR from SD and n."""
    try:
        vals = [float(x) for x in (sd_t, sd_c, n_t, n_c, mean_t, mean_c)]
        if any(v <= 0 for v in vals):
            return None
        sd_t, sd_c, n_t, n_c, mean_t, mean_c = vals
        return (sd_t**2 / (n_t * mean_t**2)) + (sd_c**2 / (n_c * mean_c**2))
    except (TypeError, ValueError):
        return None


def compute_effect_pct(t_mean, c_mean):
    """Compute percentage change: (T-C)/|C| * 100."""
    try:
        t, c = float(t_mean), float(c_mean)
        if c != 0:
            return (t - c) / abs(c) * 100
    except (TypeError, ValueError):
        pass
    return None


# ── Duplicate detection ──────────────────────────────────────────────────────

def detect_duplicates(df):
    """Detect likely duplicate rows within the same paper.

    Duplicates: same paper + same outcome + same treatment_mean + same control_mean.
    """
    flags = pd.Series(False, index=df.index)
    dup_groups = []

    for paper_id, group in df.groupby("paper_id"):
        if len(group) < 2:
            continue

        seen = {}
        for idx, row in group.iterrows():
            key = (
                str(row.get("outcome", "")).lower().strip(),
                _round_sig(row.get("treatment_mean"), 4),
                _round_sig(row.get("control_mean"), 4),
            )
            if key in seen:
                flags[idx] = True
                if not flags[seen[key]]:
                    # Keep the first occurrence, flag subsequent
                    pass
                dup_groups.append({
                    "paper_id": paper_id,
                    "original_idx": seen[key],
                    "duplicate_idx": idx,
                    "outcome": key[0],
                    "t_mean": key[1],
                    "c_mean": key[2],
                })
            else:
                seen[key] = idx

    return flags, dup_groups


# ── Main QC function ─────────────────────────────────────────────────────────

def run_qc(topic_dir: Path, config: dict, dry_run: bool = False):
    """Run all hard QC filters on a topic's extraction output.

    Reads summary.csv (or merges individual JSONs), applies filters,
    writes summary_qc.csv and qc_audit.json.
    """
    extract_dir = topic_dir / "4_extract"
    if not extract_dir.exists():
        print(f"  No 4_extract/ directory for {topic_dir.name}")
        return None

    # Load data: try summary.csv first, then merge JSONs
    csv_path = extract_dir / "summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = _merge_extraction_jsons(extract_dir)
        if df is None or len(df) == 0:
            print(f"  No extraction data found for {topic_dir.name}")
            return None

    n_input = len(df)
    n_papers_input = df["paper_id"].nunique()
    audit = {
        "topic": topic_dir.name,
        "input_rows": n_input,
        "input_papers": n_papers_input,
        "checks": [],
    }

    # ── Check 1: Structural completeness ─────────────────────────────────
    # Both means must be present and numeric
    has_t = pd.to_numeric(df["treatment_mean"], errors="coerce").notna()
    has_c = pd.to_numeric(df["control_mean"], errors="coerce").notna()
    struct_ok = has_t & has_c

    n_struct_fail = int((~struct_ok).sum())
    audit["checks"].append({
        "check": "structural_completeness",
        "description": "Both treatment_mean and control_mean must be present and numeric",
        "rows_flagged": n_struct_fail,
        "action": "exclude",
    })
    df = df[struct_ok].copy()

    # Ensure numeric types
    df["treatment_mean"] = pd.to_numeric(df["treatment_mean"], errors="coerce")
    df["control_mean"] = pd.to_numeric(df["control_mean"], errors="coerce")

    # ── Check 2: Positive means (for lnRR) ───────────────────────────────
    positive_mask = (df["treatment_mean"] > 0) & (df["control_mean"] > 0)
    n_neg = int((~positive_mask).sum())
    audit["checks"].append({
        "check": "positive_means",
        "description": "Both means must be positive for log response ratio",
        "rows_flagged": n_neg,
        "action": "exclude",
    })
    df = df[positive_mask].copy()

    # ── Check 3: Effect size computation ─────────────────────────────────
    df["effect_pct"] = df.apply(
        lambda r: compute_effect_pct(r["treatment_mean"], r["control_mean"]),
        axis=1,
    )
    df["lnRR"] = df.apply(
        lambda r: compute_lnRR(r["treatment_mean"], r["control_mean"]),
        axis=1,
    )

    # ── Check 4: Variance conversion ─────────────────────────────────────
    var_results = df.apply(convert_variance_to_sd, axis=1, result_type="expand")
    df["sd_treatment_qc"] = var_results["sd_treatment"]
    df["sd_control_qc"] = var_results["sd_control"]
    df["n_treatment_qc"] = var_results["n_treatment"]
    df["n_control_qc"] = var_results["n_control"]
    df["variance_status"] = var_results["variance_status"]
    df["variance_conversion_notes"] = var_results["conversion_notes"]

    # Compute variance of lnRR
    df["var_lnRR"] = df.apply(
        lambda r: compute_var_lnRR(
            r["sd_treatment_qc"], r["sd_control_qc"],
            r["n_treatment_qc"], r["n_control_qc"],
            r["treatment_mean"], r["control_mean"],
        ),
        axis=1,
    )

    n_var_present = int((df["variance_status"] == "present").sum())
    n_var_missing = int((df["variance_status"] == "missing").sum())
    audit["checks"].append({
        "check": "variance_integrity",
        "description": "Convert SE/SD/LSD/CV/CI to common SD",
        "variance_present": n_var_present,
        "variance_partial": int((df["variance_status"] == "partial").sum()),
        "variance_missing": n_var_missing,
        "action": "flag_only",
    })

    # ── Check 5: Duplicate detection ─────────────────────────────────────
    dup_flags, dup_groups = detect_duplicates(df)
    n_dups = int(dup_flags.sum())
    audit["checks"].append({
        "check": "duplicate_detection",
        "description": "Same paper + outcome + treatment_mean + control_mean",
        "rows_flagged": n_dups,
        "duplicate_groups": dup_groups[:20],  # Cap at 20 examples
        "action": "exclude",
    })
    df = df[~dup_flags].copy()

    # ── Check 6: Outlier flagging ────────────────────────────────────────
    # Primary filter: percent_change thresholds (catches V1 Alrijabo-style outliers
    # that slip under the lnRR > 2.0 ceiling, e.g. +609% has lnRR=1.96 < 2.0).
    pct_extreme = (
        df["effect_pct"].notna() &
        ((df["effect_pct"] < EFFECT_PCT_LOWER) | (df["effect_pct"] > EFFECT_PCT_UPPER))
    )
    # Secondary filter: |lnRR| > 2.0 catches cases where means are positive but
    # effect_pct cannot be computed (e.g. asymmetric ratio issues).
    extreme_mask = (
        df["lnRR"].notna() &
        (df["lnRR"].abs() > EXTREME_LNRR_THRESHOLD)
    )
    outlier_mask = pct_extreme | extreme_mask
    n_outliers = int(outlier_mask.sum())

    df["_qc_outlier"] = outlier_mask
    audit["checks"].append({
        "check": "outlier_flagging",
        "description": (
            f"percent_change outside [{EFFECT_PCT_LOWER}%, {EFFECT_PCT_UPPER}%] "
            f"(primary) OR |lnRR| > {EXTREME_LNRR_THRESHOLD} (secondary). "
            f"V1 lesson: +609% Alrijabo rows had lnRR=1.96, slipping the lnRR-only filter."
        ),
        "rows_flagged": n_outliers,
        "action": "flag_only",
    })

    # ── Check 7: CV bounds check ─────────────────────────────────────────
    cv_flags = pd.Series(False, index=df.index)
    for col, mean_col in [("sd_treatment_qc", "treatment_mean"),
                           ("sd_control_qc", "control_mean")]:
        sd = pd.to_numeric(df[col], errors="coerce")
        mn = pd.to_numeric(df[mean_col], errors="coerce")
        cv = (sd / mn * 100).where(mn > 0)
        cv_flags |= cv.notna() & ((cv < CV_LOWER) | (cv > CV_UPPER))

    df["_qc_cv_suspicious"] = cv_flags
    audit["checks"].append({
        "check": "cv_bounds",
        "description": f"CV outside [{CV_LOWER}%, {CV_UPPER}%] is suspicious",
        "rows_flagged": int(cv_flags.sum()),
        "action": "flag_only",
    })

    # ── Summary ──────────────────────────────────────────────────────────
    n_output = len(df)
    n_papers_output = df["paper_id"].nunique()
    n_weighted = int(df["var_lnRR"].notna().sum())

    audit["output_rows"] = n_output
    audit["output_papers"] = n_papers_output
    audit["rows_excluded"] = n_input - n_output
    audit["rows_with_variance"] = n_weighted
    audit["variance_coverage_pct"] = round(n_weighted / n_output * 100, 1) if n_output > 0 else 0

    # ── Write outputs ────────────────────────────────────────────────────
    if not dry_run:
        out_csv = extract_dir / "summary_qc.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")

        audit_path = extract_dir / "qc_audit.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False, default=str)

        print(f"  QC complete: {n_input} -> {n_output} rows ({n_input - n_output} removed)")
        print(f"  Variance coverage: {audit['variance_coverage_pct']}% ({n_weighted}/{n_output})")
        print(f"  Written: {out_csv.name}, {audit_path.name}")
    else:
        print(f"  [DRY RUN] Would produce: {n_input} → {n_output} rows")
        print(f"  Variance coverage: {audit['variance_coverage_pct']}%")

    return audit


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    try:
        v = float(val)
        return v if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _round_sig(val, digits=4):
    try:
        v = float(val)
        if v == 0:
            return 0.0
        return round(v, digits - int(math.floor(math.log10(abs(v)))) - 1)
    except (TypeError, ValueError):
        return None


def _merge_extraction_jsons(extract_dir: Path):
    """Merge individual *_agent.json files into a single DataFrame."""
    rows = []
    for jf in sorted(extract_dir.glob("*_agent.json")):
        if jf.name.startswith("_") or jf.name == "extraction_config.json":
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        obs_list = data.get("observations", [])
        for obs in obs_list:
            if isinstance(obs, dict):
                # Flatten moderators
                mods = obs.pop("moderators", {})
                if isinstance(mods, dict):
                    for k, v in mods.items():
                        obs[f"mod_{k}"] = v
                rows.append(obs)

    if not rows:
        return None
    return pd.DataFrame(rows)


def load_config(topic_dir: Path):
    """Load topic config.json."""
    cfg_path = topic_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V2 Stage 5: Deterministic QC hard filters")
    parser.add_argument("topic", nargs="?", help="Topic directory name (or --all)")
    parser.add_argument("--all", action="store_true", help="Run on all topics")
    parser.add_argument("--dry-run", action="store_true", help="Print results without writing")
    args = parser.parse_args()

    if args.all:
        topics = [d for d in ROOT.iterdir()
                  if d.is_dir() and (d / "4_extract").exists()]
    elif args.topic:
        topics = [ROOT / args.topic]
    else:
        parser.print_help()
        return

    print("=" * 60)
    print("PIPELINE V2 — STAGE 5: DETERMINISTIC QC")
    print("=" * 60)

    all_audits = []
    for topic_dir in sorted(topics):
        if not topic_dir.exists():
            print(f"\n--- {topic_dir.name} --- SKIP (not found)")
            continue
        print(f"\n--- {topic_dir.name} ---")
        config = load_config(topic_dir)
        audit = run_qc(topic_dir, config, dry_run=args.dry_run)
        if audit:
            all_audits.append(audit)

    # Summary table
    if all_audits:
        print(f"\n{'='*60}")
        print(f"{'Topic':25s} {'In':>6s} {'Out':>6s} {'Cut':>5s} {'Var%':>5s}")
        print("-" * 60)
        for a in all_audits:
            print(f"{a['topic']:25s} {a['input_rows']:6d} {a['output_rows']:6d} "
                  f"{a['rows_excluded']:5d} {a['variance_coverage_pct']:5.1f}")


if __name__ == "__main__":
    main()
