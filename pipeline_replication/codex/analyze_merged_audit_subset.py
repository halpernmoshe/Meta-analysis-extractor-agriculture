#!/usr/bin/env python3
"""
Merge the audit subset layers and produce a small summary of which features
distinguish clean, alignment/structure, and coverage cases.
"""

from __future__ import annotations

import csv
import json
import argparse
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


def to_float(value: str | None) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def merge_rows(prefix: str, date_stamp: str) -> list[dict]:
    claims = load_csv(OUT / f"{prefix}_claim_features_{date_stamp}.csv")
    labels = {r["claim_key"]: r for r in load_csv(OUT / f"{prefix}_claim_labels_{date_stamp}.csv")}
    within = {r["paper_id"]: r for r in load_csv(OUT / f"{prefix}_within_paper_features_{date_stamp}.csv")}

    merged = []
    for row in claims:
        m = dict(row)
        m.update(labels.get(row["claim_key"], {}))
        wp = within.get(row["paper_id"], {})
        for k, v in wp.items():
            if k != "paper_id":
                m[k] = v
        merged.append(m)
    return merged


def summarize_by_label(rows: list[dict]) -> dict:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["initial_label"]].append(row)

    important_flags = [
        "drift_concentration_vs_content",
        "drift_tissue_mismatch",
        "drift_arm_mismatch",
        "drift_timepoint_mismatch",
        "drift_pooled_vs_subgroup_mismatch",
        "drift_figure_only_target",
        "warning_timepoint_risk",
        "warning_figure_only_risk",
        "warning_averaging_risk",
        "warning_factorial_risk",
        "warning_multi_condition_risk",
        "report_timepoint_conflict",
        "report_averaging_conflict",
        "report_factorial_conflict",
        "report_figure_digitization_limit",
        "report_mentions_results_text",
        "report_mentions_abstract",
    ]

    summary = {"total_claims": len(rows), "by_label": {}}
    for label, group in sorted(by_label.items()):
        entry = {
            "n_claims": len(group),
            "papers": sorted({r["paper_id"] for r in group}),
            "mean_claim_disagreement_count": round(
                sum(to_int(r.get("claim_disagreement_count")) for r in group) / len(group), 3
            ),
            "mean_cross_model_support_n": round(
                sum(to_int(r.get("cross_model_support_n")) for r in group) / len(group), 3
            ),
            "mean_relaxed_support_n": round(
                sum(to_int(r.get("pairwise_relaxed_support_n")) for r in group) / len(group), 3
            ),
            "mean_risk_flag_count": round(sum(to_int(r.get("risk_flag_count")) for r in group) / len(group), 3),
            "mean_construct_drift_count": round(
                sum(to_int(r.get("construct_drift_count")) for r in group) / len(group), 3
            ),
            "mean_report_channel_count": round(
                sum(to_int(r.get("report_channel_count")) for r in group) / len(group), 3
            ),
            "paper_root_causes": dict(Counter((r.get("paper_root_cause") or "none") for r in group)),
            "feature_true_rates": {},
        }
        for flag in important_flags:
            entry["feature_true_rates"][flag] = round(
                sum(1 for r in group if truthy(r.get(flag))) / len(group), 3
            )
        summary["by_label"][label] = entry
    return summary


def build_markdown(summary: dict) -> str:
    lines = [
        "# Merged Audit Subset Analysis",
        "",
        f"Total claims: {summary['total_claims']}",
        "",
    ]
    for label, entry in summary["by_label"].items():
        lines.extend(
            [
                f"## {label}",
                f"- Claims: {entry['n_claims']}",
                f"- Mean disagreement count: {entry['mean_claim_disagreement_count']}",
                f"- Mean strict support: {entry['mean_cross_model_support_n']}",
                f"- Mean relaxed support: {entry['mean_relaxed_support_n']}",
                f"- Mean risk-flag count: {entry['mean_risk_flag_count']}",
                f"- Mean construct-drift count: {entry['mean_construct_drift_count']}",
                f"- Mean report-channel count: {entry['mean_report_channel_count']}",
                f"- Paper root causes: {json.dumps(entry['paper_root_causes'], sort_keys=True)}",
                f"- Papers: {', '.join(entry['papers'])}",
                "",
                "Feature true-rates:",
            ]
        )
        for flag, rate in entry["feature_true_rates"].items():
            lines.append(f"- {flag}: {rate}")
        lines.append("")
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

    rows = merge_rows(args.prefix, args.date_stamp)
    summary = summarize_by_label(rows)
    md = build_markdown(summary)

    merged_csv = OUT / f"{args.prefix}_claim_features_merged_{args.date_stamp}.csv"
    summary_json = OUT / f"{args.prefix}_merged_analysis_{args.date_stamp}.json"
    summary_md = OUT / f"{args.prefix}_merged_analysis_{args.date_stamp}.md"

    write_csv(merged_csv, rows)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(md)

    print(merged_csv)
    print(summary_json)
    print(summary_md)


if __name__ == "__main__":
    main()
