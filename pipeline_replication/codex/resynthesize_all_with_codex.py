#!/usr/bin/env python3
"""
Apply codex adjudication decisions to all 6 topics and produce a comprehensive
before/after comparison table.

Reads:
- Original validated CSVs from {topic}/4_extract/summary_validated.csv
- Codex decisions from codex/outputs/codex_decisions/{topic}/decisions.jsonl
- Effector labels from codex/outputs/effector_labels/{topic}/labels.jsonl

Produces:
- codex/outputs/codex_filtered_results/{topic}_kept.csv
- codex/outputs/codex_filtered_results/{topic}_flagged.csv
- codex/outputs/codex_filtered_results/all_topics_comparison.json
- codex/outputs/codex_filtered_results/all_topics_comparison.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
DECISIONS_ROOT = CODEX_ROOT / "outputs" / "codex_decisions"
EFFECTOR_ROOT = CODEX_ROOT / "outputs" / "effector_labels"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "codex_filtered_results"

ALL_TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]

BENCHMARKS = {
    "organic_yield_gap": -19.2,
    "notill_tillage": -5.7,
    "mycorrhiza_yield": 23.0,
    "legume_rotation": 20.0,
    "biochar_crop_yield": 16.0,
    "intercropping_yield": 22.0,
}

BENCHMARK_SOURCES = {
    "organic_yield_gap": "Ponisio et al. 2015",
    "notill_tillage": "Pittelkow et al. 2015",
    "mycorrhiza_yield": "Hoeksema et al. 2010",
    "legume_rotation": "Zhao et al. 2022",
    "biochar_crop_yield": "Ye et al. 2020",
    "intercropping_yield": "Yu et al. 2015",
}


# ── Meta-analysis helpers ──────────────────────────────────────────────────

def compute_lnrr(t_mean, c_mean) -> float | None:
    try:
        t, c = float(t_mean), float(c_mean)
        if t > 0 and c > 0:
            return math.log(t / c)
    except (TypeError, ValueError):
        pass
    return None


def get_sd(row) -> tuple:
    sd_t = row.get("sd_treatment")
    sd_c = row.get("sd_control")
    n_t = row.get("treatment_n") or row.get("control_n")
    n_c = row.get("control_n") or row.get("treatment_n")

    if _missing(sd_t) and not _missing(row.get("se_treatment")) and _pos(n_t):
        sd_t = float(row["se_treatment"]) * math.sqrt(float(n_t))
    if _missing(sd_c) and not _missing(row.get("se_control")) and _pos(n_c):
        sd_c = float(row["se_control"]) * math.sqrt(float(n_c))

    if _missing(sd_t) and not _missing(row.get("variance_value")):
        vtype = str(row.get("variance_type", "")).upper()
        if vtype == "LSD" and _pos(n_t):
            lsd = float(row["variance_value"])
            n_val = float(n_t)
            df_val = 2 * (n_val - 1)
            if df_val > 0:
                t_crit = stats.t.ppf(0.975, df_val)
                se_diff = lsd / (t_crit * math.sqrt(2))
                sd_est = se_diff * math.sqrt(n_val)
                sd_t = sd_est
                sd_c = sd_est
    return sd_t, sd_c, n_t, n_c


def dl_meta(rows) -> dict | None:
    """DerSimonian-Laird random effects on lnRR."""
    yi, vi = [], []
    for row in rows:
        lnrr = compute_lnrr(row.get("treatment_mean"), row.get("control_mean"))
        if lnrr is None:
            continue
        sd_t, sd_c, n_t, n_c = get_sd(row)
        try:
            vals = [float(x) for x in (sd_t, sd_c, n_t, n_c)]
            t_m, c_m = float(row["treatment_mean"]), float(row["control_mean"])
            if any(v <= 0 for v in vals) or t_m <= 0 or c_m <= 0:
                continue
            sd_t, sd_c, n_t, n_c = vals
            vr = (sd_t**2 / (n_t * t_m**2)) + (sd_c**2 / (n_c * c_m**2))
            if vr > 0:
                yi.append(lnrr)
                vi.append(vr)
        except (TypeError, ValueError):
            continue

    if len(yi) < 3:
        return None

    yi_a = np.array(yi, dtype=float)
    vi_a = np.array(vi, dtype=float)
    wi = 1.0 / vi_a
    sum_w = wi.sum()
    mu_fe = (wi * yi_a).sum() / sum_w
    q = (wi * (yi_a - mu_fe)**2).sum()
    k = len(yi_a)
    c_val = sum_w - (wi**2).sum() / sum_w
    tau2 = max(0.0, (q - (k - 1)) / c_val) if c_val > 0 else 0.0
    wi_re = 1.0 / (vi_a + tau2)
    sum_w_re = wi_re.sum()
    mu_re = (wi_re * yi_a).sum() / sum_w_re
    se_re = 1.0 / math.sqrt(sum_w_re)
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    i2 = max(0.0, (q - (k - 1)) / q * 100) if q > 0 else 0.0

    return {
        "k": int(k),
        "pooled_pct": round((math.exp(mu_re) - 1) * 100, 2),
        "ci_lo_pct": round((math.exp(ci_lo) - 1) * 100, 2),
        "ci_hi_pct": round((math.exp(ci_hi) - 1) * 100, 2),
        "I2": round(i2, 1),
    }


def simple_mean_pct(rows) -> float | None:
    effects = []
    for row in rows:
        try:
            t = float(row["treatment_mean"])
            c = float(row["control_mean"])
            if c > 0:
                effects.append((t / c - 1) * 100)
        except (TypeError, ValueError):
            continue
    return round(np.mean(effects), 2) if effects else None


def _missing(val) -> bool:
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _pos(val) -> bool:
    try:
        return float(val) > 0
    except (TypeError, ValueError):
        return False


# ── Topic processing ───────────────────────────────────────────────────────

def process_topic(topic: str) -> dict:
    """Process a single topic: load validated CSV, apply decisions, compute meta."""

    # Load validated CSV
    validated_csv = ROOT / topic / "4_extract" / "summary_validated.csv"
    if not validated_csv.exists():
        validated_csv = ROOT / topic / "4_extract" / "summary.csv"
    if not validated_csv.exists():
        return {"topic": topic, "error": "no_validated_csv"}

    df = pd.read_csv(validated_csv)
    before_rows = df.to_dict("records")
    before_meta = dl_meta(before_rows)
    before_simple = simple_mean_pct(before_rows)

    # Load decisions
    decisions_path = DECISIONS_ROOT / topic / "decisions.jsonl"
    if not decisions_path.exists():
        return {
            "topic": topic,
            "error": "no_decisions",
            "before_n": len(df),
            "before_meta": before_meta,
        }

    decisions = {}
    with decisions_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            decisions[obj["row_id"]] = obj

    # Load effector labels if available
    effector_labels = {}
    labels_path = EFFECTOR_ROOT / topic / "labels.jsonl"
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                effector_labels[obj["row_id"]] = obj

    # Separate kept/flagged/excluded
    kept_rows = []
    flagged_rows = []
    for row_id, dec in decisions.items():
        if dec["decision"] in ("keep", "swap_treatment_control"):
            kept_rows.append(row_id)
        elif dec["decision"] == "flag":
            flagged_rows.append(row_id)

    # Load the kept rows CSV directly (already built by adjudication)
    kept_csv = DECISIONS_ROOT / topic / "strict_kept_rows.csv"
    if not kept_csv.exists():
        kept_csv = DECISIONS_ROOT / topic / "kept_rows.csv"
    if kept_csv.exists():
        kept_df = pd.read_csv(kept_csv)
        kept_data = kept_df.to_dict("records")
    else:
        kept_data = []

    after_meta = dl_meta(kept_data)
    after_simple = simple_mean_pct(kept_data)

    # Benchmark-aligned subset (from effector labels)
    aligned_rows = []
    for row in kept_data:
        rid = row.get("row_id", "")
        label = effector_labels.get(rid, {})
        if label.get("normalized_estimand_context") == "benchmark_aligned":
            aligned_rows.append(row)

    aligned_meta = dl_meta(aligned_rows) if aligned_rows else None
    aligned_simple = simple_mean_pct(aligned_rows) if aligned_rows else None

    # Save filtered CSVs
    if kept_data:
        pd.DataFrame(kept_data).to_csv(
            OUTPUT_ROOT / f"{topic}_kept.csv", index=False
        )

    benchmark = BENCHMARKS.get(topic)
    result = {
        "topic": topic,
        "benchmark_pct": benchmark,
        "benchmark_source": BENCHMARK_SOURCES.get(topic),
        "before": {
            "n_rows": len(before_rows),
            "n_papers": len(set(r.get("paper_id", "") for r in before_rows)),
            "dl_re": before_meta,
            "simple_mean_pct": before_simple,
        },
        "after_codex": {
            "n_rows": len(kept_data),
            "n_papers": len(set(r.get("paper_id", "") for r in kept_data)),
            "n_flagged": len(flagged_rows),
            "n_excluded": len(decisions) - len(kept_rows) - len(flagged_rows),
            "retention_pct": round(len(kept_data) / max(len(decisions), 1) * 100, 1),
            "dl_re": after_meta,
            "simple_mean_pct": after_simple,
        },
        "benchmark_aligned_subset": {
            "n_rows": len(aligned_rows),
            "n_papers": len(set(r.get("paper_id", "") for r in aligned_rows)),
            "dl_re": aligned_meta,
            "simple_mean_pct": aligned_simple,
        },
    }

    # Compute diffs
    if after_meta and benchmark is not None:
        result["after_codex"]["diff_vs_benchmark"] = round(
            after_meta["pooled_pct"] - benchmark, 2
        )
        result["after_codex"]["direction_match"] = (
            (after_meta["pooled_pct"] > 0) == (benchmark > 0)
        )
        result["after_codex"]["ci_overlap"] = (
            after_meta["ci_lo_pct"] <= benchmark <= after_meta["ci_hi_pct"]
        )

    if aligned_meta and benchmark is not None:
        result["benchmark_aligned_subset"]["diff_vs_benchmark"] = round(
            aligned_meta["pooled_pct"] - benchmark, 2
        )

    return result


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    results = []
    for topic in ALL_TOPICS:
        print(f"\nProcessing {topic}...")
        result = process_topic(topic)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        before = result["before"]
        after = result["after_codex"]
        aligned = result["benchmark_aligned_subset"]
        bench = result["benchmark_pct"]

        print(f"  Before: {before['n_rows']} rows, "
              f"DL RE = {before['dl_re']['pooled_pct']:+.2f}%" if before['dl_re'] else "  Before: insufficient data")
        print(f"  After codex: {after['n_rows']} rows ({after['retention_pct']}% retained), "
              f"DL RE = {after['dl_re']['pooled_pct']:+.2f}%" if after['dl_re'] else f"  After codex: {after['n_rows']} rows, insufficient data")
        if aligned["dl_re"]:
            print(f"  Benchmark-aligned: {aligned['n_rows']} rows, "
                  f"DL RE = {aligned['dl_re']['pooled_pct']:+.2f}%")
        print(f"  Benchmark: {bench}% ({result.get('benchmark_source', '')})")

    # Write JSON
    (OUTPUT_ROOT / "all_topics_comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write markdown comparison table
    md_lines = [
        "# All Topics: Before vs After Codex Adjudication",
        "",
        "| Topic | Benchmark | Before (DL RE) | After Codex (DL RE) | Aligned Subset | Direction | CI Overlap |",
        "|-------|-----------|----------------|---------------------|----------------|-----------|------------|",
    ]
    for r in results:
        if "error" in r:
            md_lines.append(f"| {r['topic']} | -- | ERROR: {r.get('error','')} | -- | -- | -- | -- |")
            continue
        bench = r["benchmark_pct"]
        before = r["before"]["dl_re"]
        after = r["after_codex"]["dl_re"]
        aligned = r["benchmark_aligned_subset"]["dl_re"]

        before_str = f"{before['pooled_pct']:+.1f}% (k={before['k']})" if before else "N/A"
        after_str = f"{after['pooled_pct']:+.1f}% (k={after['k']})" if after else "N/A"
        aligned_str = f"{aligned['pooled_pct']:+.1f}% (n={r['benchmark_aligned_subset']['n_rows']})" if aligned else "N/A"
        direction = "YES" if r["after_codex"].get("direction_match") else "NO"
        ci_overlap = "YES" if r["after_codex"].get("ci_overlap") else "NO"

        md_lines.append(
            f"| {r['topic']} | {bench:+.1f}% | {before_str} | {after_str} | {aligned_str} | {direction} | {ci_overlap} |"
        )

    md_lines.extend([
        "",
        "## Summary",
        "",
    ])

    n_direction = sum(1 for r in results if r.get("after_codex", {}).get("direction_match"))
    n_ci = sum(1 for r in results if r.get("after_codex", {}).get("ci_overlap"))
    n_total = sum(1 for r in results if "error" not in r)

    md_lines.append(f"- Direction agreement: {n_direction}/{n_total}")
    md_lines.append(f"- CI overlap with benchmark: {n_ci}/{n_total}")

    # Did codex improve things?
    md_lines.extend(["", "## Did Codex Adjudication Help?", ""])
    for r in results:
        if "error" in r:
            continue
        topic = r["topic"]
        bench = r["benchmark_pct"]
        before = r["before"]["dl_re"]
        after = r["after_codex"]["dl_re"]
        if before and after and bench is not None:
            before_diff = abs(before["pooled_pct"] - bench)
            after_diff = abs(after["pooled_pct"] - bench)
            improved = after_diff < before_diff
            md_lines.append(
                f"- **{topic}**: {'IMPROVED' if improved else 'WORSENED'} "
                f"(before: {before_diff:.1f}pp gap, after: {after_diff:.1f}pp gap)"
            )

    (OUTPUT_ROOT / "all_topics_comparison.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print("  COMPARISON TABLE WRITTEN")
    print("=" * 60)
    for line in md_lines[3:3+len(results)]:
        print(line)


if __name__ == "__main__":
    main()
