#!/usr/bin/env python3
"""
Build a first consilience-profile layer from merged claim features.

The aim is to replace a single catch-all risk score with a structured profile:
- numeric_grounding
- cross_model_concordance
- within_paper_support
- construct_drift
- benchmark_comparability
- structural_risk
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
OUT = ROOT / "pipeline_replication" / "codex" / "outputs" / "combined_analysis"


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def to_int(value: str | None) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def band(score: int) -> str:
    if score >= 3:
        return "high"
    if score == 2:
        return "medium"
    return "low"


def profile_row(row: dict) -> dict:
    strict_support = to_int(row.get("cross_model_support_n"))
    relaxed_support = to_int(row.get("pairwise_relaxed_support_n"))
    disagreements = to_int(row.get("claim_disagreement_count"))
    drift_count = to_int(row.get("construct_drift_count"))
    channel_count = to_int(row.get("report_channel_count"))
    weighted_support = to_int(row.get("report_channel_weighted_support"))

    numeric_grounding = 0
    if row.get("consensus_data_source") not in ("", "unknown", None):
        numeric_grounding += 1
    if truthy(row.get("has_n")):
        numeric_grounding += 1
    if truthy(row.get("has_variance")):
        numeric_grounding += 1

    cross_model_concordance = 0
    if strict_support >= 2:
        cross_model_concordance += 1
    if strict_support >= 3:
        cross_model_concordance += 1
    if disagreements == 0:
        cross_model_concordance += 1
    elif disagreements <= 1:
        cross_model_concordance += 0
    else:
        cross_model_concordance = max(cross_model_concordance - 1, 0)

    within_paper_support = 0
    if weighted_support >= 1 or channel_count >= 1:
        within_paper_support += 1
    if weighted_support >= 3 or channel_count >= 3:
        within_paper_support += 1
    if weighted_support >= 5 or truthy(row.get("report_mentions_results_text")) or truthy(row.get("report_mentions_table_evidence")):
        within_paper_support += 1

    benchmark_comparability = 3
    if drift_count >= 1:
        benchmark_comparability -= 1
    if truthy(row.get("drift_concentration_vs_content")) or truthy(row.get("drift_tissue_mismatch")):
        benchmark_comparability -= 1
    if truthy(row.get("report_status_skip")) or truthy(row.get("report_no_concentration_data")):
        benchmark_comparability -= 1
    benchmark_comparability = max(0, benchmark_comparability)

    structural_risk = 0
    if truthy(row.get("warning_factorial_risk")) or truthy(row.get("report_factorial_conflict")):
        structural_risk += 1
    if truthy(row.get("warning_multi_condition_risk")) or truthy(row.get("report_treatment_arm_confusion")):
        structural_risk += 1
    if truthy(row.get("report_overall_partial")) or truthy(row.get("warning_sparse_recon_risk")):
        structural_risk += 1

    overall = (
        numeric_grounding
        + cross_model_concordance
        + within_paper_support
        + benchmark_comparability
        - drift_count
        - structural_risk
    )

    out = dict(row)
    out.update(
        {
            "profile_numeric_grounding": numeric_grounding,
            "profile_numeric_grounding_band": band(numeric_grounding),
            "profile_cross_model_concordance": cross_model_concordance,
            "profile_cross_model_concordance_band": band(cross_model_concordance),
            "profile_within_paper_support": within_paper_support,
            "profile_within_paper_support_band": band(within_paper_support),
            "profile_construct_drift": drift_count,
            "profile_construct_drift_band": "high" if drift_count >= 2 else ("medium" if drift_count == 1 else "low"),
            "profile_benchmark_comparability": benchmark_comparability,
            "profile_benchmark_comparability_band": band(benchmark_comparability),
            "profile_structural_risk": structural_risk,
            "profile_structural_risk_band": "high" if structural_risk >= 2 else ("medium" if structural_risk == 1 else "low"),
            "profile_overall_score": overall,
        }
    )
    return out


def summarize(rows: list[dict]) -> dict:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["initial_label"]].append(row)

    summary = {"total_claims": len(rows), "by_label": {}}
    for label, group in sorted(by_label.items()):
        summary["by_label"][label] = {
            "n_claims": len(group),
            "mean_numeric_grounding": round(sum(to_int(r["profile_numeric_grounding"]) for r in group) / len(group), 3),
            "mean_cross_model_concordance": round(sum(to_int(r["profile_cross_model_concordance"]) for r in group) / len(group), 3),
            "mean_within_paper_support": round(sum(to_int(r["profile_within_paper_support"]) for r in group) / len(group), 3),
            "mean_construct_drift": round(sum(to_int(r["profile_construct_drift"]) for r in group) / len(group), 3),
            "mean_benchmark_comparability": round(sum(to_int(r["profile_benchmark_comparability"]) for r in group) / len(group), 3),
            "mean_structural_risk": round(sum(to_int(r["profile_structural_risk"]) for r in group) / len(group), 3),
            "mean_overall_score": round(sum(to_int(r["profile_overall_score"]) for r in group) / len(group), 3),
            "papers": sorted({r["paper_id"] for r in group}),
        }
    return summary


def build_markdown(summary: dict) -> str:
    lines = ["# Consilience Profiles", "", f"Total claims: {summary['total_claims']}", ""]
    for label, entry in summary["by_label"].items():
        lines.extend(
            [
                f"## {label}",
                f"- Claims: {entry['n_claims']}",
                f"- Mean numeric grounding: {entry['mean_numeric_grounding']}",
                f"- Mean cross-model concordance: {entry['mean_cross_model_concordance']}",
                f"- Mean within-paper support: {entry['mean_within_paper_support']}",
                f"- Mean construct drift: {entry['mean_construct_drift']}",
                f"- Mean benchmark comparability: {entry['mean_benchmark_comparability']}",
                f"- Mean structural risk: {entry['mean_structural_risk']}",
                f"- Mean overall score: {entry['mean_overall_score']}",
                f"- Papers: {', '.join(entry['papers'])}",
                "",
            ]
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="audit_subset")
    parser.add_argument("--date-stamp", type=str, default="2026-03-27")
    args = parser.parse_args()

    rows = load_csv(OUT / f"{args.prefix}_claim_features_merged_{args.date_stamp}.csv")
    profiled = [profile_row(row) for row in rows]
    summary = summarize(profiled)
    md = build_markdown(summary)

    out_csv = OUT / f"{args.prefix}_consilience_profiles_{args.date_stamp}.csv"
    out_json = OUT / f"{args.prefix}_consilience_profiles_{args.date_stamp}.json"
    out_md = OUT / f"{args.prefix}_consilience_profiles_{args.date_stamp}.md"

    write_csv(out_csv, profiled)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(md)

    print(out_csv)
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
