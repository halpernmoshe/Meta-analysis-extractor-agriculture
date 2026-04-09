#!/usr/bin/env python3
"""
Apply a first simple rule-based three-way classifier to the merged audit subset.

Goal:
- clean_support
- alignment_or_structure_problem
- extraction_coverage_problem
- low_support_uncertain

This is intentionally simple and readable. It is meant to test whether the
current feature stack is already sufficient to separate the major regimes.
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


def score_row(row: dict) -> tuple[str, dict]:
    strict_support = to_int(row.get("cross_model_support_n"))
    relaxed_support = to_int(row.get("pairwise_relaxed_support_n"))
    disagreements = to_int(row.get("claim_disagreement_count"))
    risk_flags = to_int(row.get("risk_flag_count"))

    timepoint = truthy(row.get("warning_timepoint_risk")) or truthy(row.get("report_timepoint_conflict"))
    figure_limit = truthy(row.get("warning_figure_only_risk")) or truthy(row.get("report_figure_digitization_limit"))
    factorial = truthy(row.get("warning_factorial_risk")) or truthy(row.get("report_factorial_conflict"))
    averaging = truthy(row.get("warning_averaging_risk")) or truthy(row.get("report_averaging_conflict"))
    multi_condition = truthy(row.get("warning_multi_condition_risk"))
    skip_status = truthy(row.get("report_status_skip")) or truthy(row.get("report_zero_match")) or truthy(row.get("report_no_concentration_data"))
    wrong_tissue = truthy(row.get("report_wrong_tissue"))
    arm_confusion = truthy(row.get("report_treatment_arm_confusion")) or truthy(row.get("report_factorial_arm_selection_issue"))
    partial_rating = truthy(row.get("report_overall_partial"))

    alignment_gap = max(relaxed_support - strict_support, 0)

    component = {
        "strict_support": strict_support,
        "relaxed_support": relaxed_support,
        "disagreements": disagreements,
        "risk_flags": risk_flags,
        "alignment_gap": alignment_gap,
        "timepoint_or_figure": int(timepoint or figure_limit),
        "factorial_or_averaging": int(factorial or averaging or multi_condition),
        "report_skip_or_noncomparable": int(skip_status),
        "report_wrong_tissue_or_arm": int(wrong_tissue or arm_confusion),
        "report_partial": int(partial_rating),
    }

    # Coverage pattern: highly convergent, zero disagreement, but timepoint/figure limitation.
    if (strict_support >= 3 and disagreements == 0 and timepoint and figure_limit) or (skip_status and figure_limit):
        return "extraction_coverage_problem", component

    # Weak-support pattern: low support, no disagreement, and no explicit coverage/alignment trigger.
    if strict_support <= 1 and relaxed_support <= 1 and disagreements == 0 and risk_flags <= 3 and not timepoint and not averaging and not multi_condition:
        return "low_support_uncertain", component
    if strict_support == 0 and relaxed_support >= 2 and disagreements == 0 and risk_flags <= 3 and not timepoint and not averaging and not multi_condition:
        return "low_support_uncertain", component

    # Clean pattern: strong support, no disagreement, and no explicit timepoint/averaging warning.
    clean_support_like = (
        (strict_support >= 3)
        or (strict_support >= 2 and relaxed_support >= 3 and risk_flags <= 2)
        or (strict_support >= 2 and relaxed_support >= 2 and risk_flags <= 3)
    )
    if clean_support_like and disagreements == 0 and not timepoint and not averaging and not multi_condition and not skip_status and not wrong_tissue and not arm_confusion and not partial_rating:
        return "clean_support", component

    # Alignment/structure pattern: disagreement OR a gap between relaxed and strict support
    # OR explicit averaging/multi-condition/factorial structure without the coverage pattern.
    if disagreements > 0 or alignment_gap >= 1 or averaging or multi_condition or wrong_tissue or arm_confusion or partial_rating or (factorial and strict_support < 3):
        return "alignment_or_structure_problem", component

    return "alignment_or_structure_problem", component


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="audit_subset")
    parser.add_argument("--date-stamp", type=str, default="2026-03-27")
    args = parser.parse_args()

    path = OUT / f"{args.prefix}_claim_features_merged_{args.date_stamp}.csv"
    rows = load_csv(path)

    scored = []
    confusion = Counter()
    by_pred: dict[str, list[dict]] = defaultdict(list)
    by_true: dict[str, list[dict]] = defaultdict(list)

    for row in rows:
        pred, component = score_row(row)
        scored_row = dict(row)
        scored_row["predicted_label"] = pred
        for k, v in component.items():
            scored_row[f"score_{k}"] = v
        scored.append(scored_row)
        true_label = row["initial_label"]
        confusion[(true_label, pred)] += 1
        by_pred[pred].append(scored_row)
        by_true[true_label].append(scored_row)

    summary = {
        "n_rows": len(scored),
        "confusion": {f"{k[0]} -> {k[1]}": v for k, v in sorted(confusion.items())},
        "by_predicted_label": {},
        "by_true_label": {},
    }

    for label, group in sorted(by_pred.items()):
        summary["by_predicted_label"][label] = {
            "n": len(group),
            "papers": sorted({r["paper_id"] for r in group}),
        }

    for label, group in sorted(by_true.items()):
        correct = sum(1 for r in group if r["predicted_label"] == label.replace("likely_", "").replace("_problem", "_problem"))
        # keep the raw count breakdown instead of pretending to compute formal accuracy
        pred_counts = Counter(r["predicted_label"] for r in group)
        summary["by_true_label"][label] = {
            "n": len(group),
            "predicted_counts": dict(pred_counts),
            "papers": sorted({r["paper_id"] for r in group}),
        }

    md_lines = [
        "# Audit Subset Rule Score",
        "",
        f"Rows scored: {len(scored)}",
        "",
        "## Confusion",
    ]
    for key, value in summary["confusion"].items():
        md_lines.append(f"- {key}: {value}")
    md_lines.extend(["", "## By True Label"])
    for label, info in summary["by_true_label"].items():
        md_lines.append(f"- {label}: {json.dumps(info['predicted_counts'], sort_keys=True)}")
    md = "\n".join(md_lines)

    scored_csv = OUT / f"{args.prefix}_rule_scored_{args.date_stamp}.csv"
    summary_json = OUT / f"{args.prefix}_rule_score_summary_{args.date_stamp}.json"
    summary_md = OUT / f"{args.prefix}_rule_score_summary_{args.date_stamp}.md"

    fieldnames = sorted({k for row in scored for k in row.keys()})
    with scored_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(md)

    print(scored_csv)
    print(summary_json)
    print(summary_md)


if __name__ == "__main__":
    main()
