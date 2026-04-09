#!/usr/bin/env python3
"""
Analyze counterfactual rescue opportunities for construct-drift claims.
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


def rescue_modes(row: dict) -> list[str]:
    modes = []
    if truthy(row.get("drift_tissue_mismatch")):
        modes.append("restrict_to_correct_tissue")
    if truthy(row.get("drift_arm_mismatch")):
        modes.append("restrict_to_correct_arm")
    if truthy(row.get("drift_pooled_vs_subgroup_mismatch")):
        modes.append("restrict_to_correct_subgroup")
    if truthy(row.get("drift_timepoint_mismatch")):
        modes.append("restrict_to_correct_timepoint")
    if truthy(row.get("drift_concentration_vs_content")):
        modes.append("restrict_to_correct_scale")
    if truthy(row.get("drift_figure_only_target")):
        modes.append("add_figure_digitization")
    return modes


def is_rescuable(row: dict) -> bool:
    support = to_int(row.get("cross_model_support_n"))
    relaxed = to_int(row.get("pairwise_relaxed_support_n"))
    disagreements = to_int(row.get("claim_disagreement_count"))
    return bool(rescue_modes(row)) and (support >= 2 or relaxed >= 2) and disagreements <= 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="audit_subset")
    parser.add_argument("--date-stamp", type=str, default="2026-03-27")
    args = parser.parse_args()

    rows = load_csv(OUT / f"{args.prefix}_claim_features_merged_{args.date_stamp}.csv")
    enriched = []
    by_label: dict[str, list[dict]] = defaultdict(list)
    rescue_counter = Counter()

    for row in rows:
        modes = rescue_modes(row)
        resc = is_rescuable(row)
        enriched_row = dict(row)
        enriched_row["rescue_modes"] = json.dumps(modes)
        enriched_row["rescuable_by_restriction"] = resc
        enriched.append(enriched_row)
        by_label[row["initial_label"]].append(enriched_row)
        if resc:
            for mode in modes:
                rescue_counter[mode] += 1

    summary = {
        "total_claims": len(rows),
        "rescue_mode_counts": dict(rescue_counter),
        "by_label": {},
    }

    for label, group in sorted(by_label.items()):
        rescuable = [r for r in group if truthy(r.get("rescuable_by_restriction"))]
        summary["by_label"][label] = {
            "n_claims": len(group),
            "n_rescuable": len(rescuable),
            "rescuable_fraction": round(len(rescuable) / len(group), 3) if group else 0.0,
            "papers": sorted({r["paper_id"] for r in rescuable}),
        }

    md_lines = [
        "# Counterfactual Rescue Analysis",
        "",
        f"Total claims: {len(rows)}",
        "",
        "## Rescue modes",
    ]
    for mode, count in summary["rescue_mode_counts"].items():
        md_lines.append(f"- {mode}: {count}")
    md_lines.extend(["", "## By label"])
    for label, info in summary["by_label"].items():
        md_lines.append(
            f"- {label}: rescuable {info['n_rescuable']}/{info['n_claims']} ({info['rescuable_fraction']})"
        )

    out_csv = OUT / f"{args.prefix}_counterfactual_rescue_{args.date_stamp}.csv"
    out_json = OUT / f"{args.prefix}_counterfactual_rescue_{args.date_stamp}.json"
    out_md = OUT / f"{args.prefix}_counterfactual_rescue_{args.date_stamp}.md"

    fieldnames = sorted({k for row in enriched for k in row.keys()})
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(out_csv)
    print(out_json)
    print(out_md)


if __name__ == "__main__":
    main()
