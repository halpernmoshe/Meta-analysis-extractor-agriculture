#!/usr/bin/env python3
"""
Apply a simple profile-based policy to consilience profiles.
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


def score_row(row: dict) -> str:
    numeric_grounding = to_int(row.get("profile_numeric_grounding"))
    concordance = to_int(row.get("profile_cross_model_concordance"))
    within_paper = to_int(row.get("profile_within_paper_support"))
    drift = to_int(row.get("profile_construct_drift"))
    comparability = to_int(row.get("profile_benchmark_comparability"))
    structural_risk = to_int(row.get("profile_structural_risk"))

    figure_only = truthy(row.get("drift_figure_only_target"))
    concentration_vs_content = truthy(row.get("drift_concentration_vs_content"))
    tissue_mismatch = truthy(row.get("drift_tissue_mismatch"))
    arm_mismatch = truthy(row.get("drift_arm_mismatch"))
    timepoint_mismatch = truthy(row.get("drift_timepoint_mismatch"))
    pooled_mismatch = truthy(row.get("drift_pooled_vs_subgroup_mismatch"))
    disagreements = to_int(row.get("claim_disagreement_count"))

    if figure_only and (concentration_vs_content or timepoint_mismatch) and comparability <= 2:
        return "extraction_coverage_problem"

    if tissue_mismatch or arm_mismatch or pooled_mismatch:
        return "alignment_or_structure_problem"

    if disagreements > 0 and (concordance <= 1 or structural_risk >= 1):
        return "alignment_or_structure_problem"

    if concordance >= 2 and drift == 0 and comparability >= 3 and structural_risk <= 1:
        return "clean_support"

    if concordance <= 1 and numeric_grounding <= 1 and drift == 0:
        return "low_support_uncertain"

    if concordance >= 2 and figure_only and comparability >= 2 and not concentration_vs_content and not timepoint_mismatch and not tissue_mismatch and not arm_mismatch:
        return "clean_support"

    return "alignment_or_structure_problem"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="audit_subset")
    parser.add_argument("--date-stamp", type=str, default="2026-03-27")
    args = parser.parse_args()

    rows = load_csv(OUT / f"{args.prefix}_consilience_profiles_{args.date_stamp}.csv")

    scored = []
    confusion = Counter()
    by_true: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pred = score_row(row)
        scored_row = dict(row)
        scored_row["profile_predicted_label"] = pred
        scored.append(scored_row)
        confusion[(row["initial_label"], pred)] += 1
        by_true[row["initial_label"]].append(scored_row)

    summary = {
        "n_rows": len(scored),
        "confusion": {f"{k[0]} -> {k[1]}": v for k, v in sorted(confusion.items())},
        "by_true_label": {},
    }
    for label, group in sorted(by_true.items()):
        summary["by_true_label"][label] = {
            "n": len(group),
            "predicted_counts": dict(Counter(r["profile_predicted_label"] for r in group)),
            "papers": sorted({r["paper_id"] for r in group}),
        }

    md_lines = ["# Consilience Profile Score", "", f"Rows scored: {len(scored)}", "", "## Confusion"]
    for key, value in summary["confusion"].items():
        md_lines.append(f"- {key}: {value}")
    md = "\n".join(md_lines)

    out_csv = OUT / f"{args.prefix}_consilience_scored_{args.date_stamp}.csv"
    out_json = OUT / f"{args.prefix}_consilience_score_summary_{args.date_stamp}.json"
    out_md = OUT / f"{args.prefix}_consilience_score_summary_{args.date_stamp}.md"

    fieldnames = sorted({k for row in scored for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write(md)

    print(out_csv)
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
