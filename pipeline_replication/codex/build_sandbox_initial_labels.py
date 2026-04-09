#!/usr/bin/env python3
"""
Build initial weak labels for sandbox claim features.

These are intentionally conservative. The goal is to create a pilot label set
for analysis, not a final truth table.

Labels:
- clean_support
- likely_alignment_or_structure_problem
- unclear
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
OUTPUT_DIR = ROOT / "pipeline_replication" / "codex" / "outputs" / "combined_analysis"


def truthy(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def assign_label(row: dict) -> tuple[str, str, str]:
    paper_id = row["paper_id"]
    disagreements = int(row.get("claim_disagreement_count") or 0)
    support_n = int(row.get("cross_model_support_n") or 0)
    direction_agree = row.get("cross_model_direction_agreement")
    source = row.get("claim_source") or ""
    flags = set(json.loads(row.get("risk_flags") or "[]"))
    notes = (row.get("consensus_notes") or "").lower()

    if disagreements == 0 and source == "consensus_observation":
        return (
            "clean_support",
            "high",
            "Consensus claim with no recorded Claude/Kimi disagreement.",
        )

    if paper_id in {"020_Overdieck_1993", "031_Pal_2003"} and disagreements == 0:
        return (
            "clean_support",
            "high",
            "Control paper with complete model agreement in sandbox audit.",
        )

    if "proxy" in notes or "averaged across" in notes:
        return (
            "likely_alignment_or_structure_problem",
            "medium",
            "Claim depends on proxy use or averaging across non-target design dimensions.",
        )

    if "paper_swap_risk" in flags or "paper_high_disagreement" in flags:
        return (
            "likely_alignment_or_structure_problem",
            "medium",
            "Paper-level disagreement or swap risk suggests structure/matching problem rather than clean support.",
        )

    if "factorial_structure" in flags and "tc_confusion" in flags and disagreements > 0:
        return (
            "likely_alignment_or_structure_problem",
            "medium",
            "Factorial structure and treatment/control ambiguity make claim alignment risky.",
        )

    if source == "disagreement_only":
        return (
            "unclear",
            "low",
            "Claim appears only in disagreement structure without stable consensus support.",
        )

    if support_n >= 2 and str(direction_agree).strip().lower() == "true" and disagreements <= 1:
        return (
            "clean_support",
            "medium",
            "Cross-model direction support is present despite limited disagreement.",
        )

    return (
        "unclear",
        "low",
        "Current evidence is mixed or weak; retain as unresolved in pilot set.",
    )


def main() -> None:
    in_csv = OUTPUT_DIR / "sandbox_claim_features_2026-03-26.csv"
    out_csv = OUTPUT_DIR / "sandbox_claim_labels_2026-03-26.csv"
    out_json = OUTPUT_DIR / "sandbox_claim_labels_2026-03-26.json"

    rows = []
    with in_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            label, confidence, rationale = assign_label(row)
            rows.append(
                {
                    "claim_key": row["claim_key"],
                    "paper_id": row["paper_id"],
                    "element": row["element"],
                    "tissue": row["tissue"],
                    "initial_label": label,
                    "label_confidence": confidence,
                    "label_rationale": rationale,
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} labels")
    print(out_csv)
    print(out_json)


if __name__ == "__main__":
    main()
