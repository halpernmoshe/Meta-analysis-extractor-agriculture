#!/usr/bin/env python3
"""
Build a small corpus-level extraction-bias dashboard across batches.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
OUT = ROOT / "pipeline_replication" / "codex" / "outputs" / "combined_analysis"
DATE_STAMP = "2026-03-27"
PREFIXES = ["audit_subset", "heldout_subset", "heldout_subset2"]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def rate(rows: list[dict], pred) -> float:
    return round(sum(1 for r in rows if pred(r)) / len(rows), 3) if rows else 0.0


def summary_for_prefix(prefix: str) -> dict:
    rows = load_csv(OUT / f"{prefix}_claim_features_merged_{DATE_STAMP}.csv")
    return {
        "n_claims": len(rows),
        "figure_only_target_rate": rate(rows, lambda r: truthy(r.get("drift_figure_only_target"))),
        "factorial_paper_rate": rate(rows, lambda r: truthy(r.get("warning_factorial_risk")) or truthy(r.get("report_factorial_conflict"))),
        "wrong_tissue_rate": rate(rows, lambda r: truthy(r.get("drift_tissue_mismatch"))),
        "arm_mismatch_rate": rate(rows, lambda r: truthy(r.get("drift_arm_mismatch"))),
        "timepoint_mismatch_rate": rate(rows, lambda r: truthy(r.get("drift_timepoint_mismatch"))),
        "pooled_subgroup_mismatch_rate": rate(rows, lambda r: truthy(r.get("drift_pooled_vs_subgroup_mismatch"))),
        "concentration_content_mismatch_rate": rate(rows, lambda r: truthy(r.get("drift_concentration_vs_content"))),
        "narrative_only_support_rate": rate(
            rows,
            lambda r: not truthy(r.get("report_mentions_table_evidence"))
            and not truthy(r.get("report_mentions_results_text"))
            and (
                truthy(r.get("report_mentions_abstract"))
                or truthy(r.get("report_mentions_conclusion"))
                or truthy(r.get("report_mentions_figure_evidence"))
            ),
        ),
        "clean_label_rate": rate(rows, lambda r: r.get("initial_label") == "clean_support"),
        "alignment_label_rate": rate(rows, lambda r: r.get("initial_label") == "likely_alignment_or_structure_problem"),
        "coverage_label_rate": rate(rows, lambda r: r.get("initial_label") == "likely_extraction_coverage_problem"),
    }


def main() -> None:
    summary = {prefix: summary_for_prefix(prefix) for prefix in PREFIXES}
    lines = ["# Extraction Bias Dashboard", ""]
    for prefix, info in summary.items():
        lines.append(f"## {prefix}")
        for key, value in info.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    out_json = OUT / f"extraction_bias_dashboard_{DATE_STAMP}.json"
    out_md = OUT / f"extraction_bias_dashboard_{DATE_STAMP}.md"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
