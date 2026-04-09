#!/usr/bin/env python3
"""
Create a first multi-role, full-context pilot input package for selected papers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
CODEX = ROOT / "pipeline_replication" / "codex"
OUT = CODEX / "outputs" / "multi_role_pilot"
REPORT_DIR = ROOT / "output" / "loladze_combined_51" / "per_paper"

DEFAULT_PAPERS = [
    "019_Baxter_1994",
    "026_Seneweera_1997",
    "035_Oksanen_2005",
    "015_Pleijel_2009",
]

ROLE_PROMPTS = {
    "design_agent": "Read the full paper and extract only design constraints: intervention/comparator structure, valid treatment arms, tissues, timepoints, units, factorial structure, and what numeric comparisons are scientifically valid.",
    "narrative_agent": "Read the full paper and extract only narrative claims: abstract/results/conclusion direction, qualitative strength, significance language, and the paper's own stated interpretation.",
    "table_agent": "Read the full paper and extract only table-grounded numeric claims: means, variance, n, source table/row, and direct treatment-control contrasts from tables.",
    "figure_agent": "Read the full paper and extract only figure-grounded claims: figure-only targets, caption evidence, approximate direction, and whether relevant outcomes exist only graphically.",
    "benchmark_agent": "Read the full paper and identify what claims are benchmark-comparable: likely benchmark construct, matching rows, nearby but non-equivalent rows, and likely construct drift.",
    "consistency_agent": "Given role outputs for the same paper, identify contradictions, support relations, construct drift, and produce a final consilience verdict.",
}

OUTPUT_SCHEMA_TEMPLATE = {
    "paper_id": "",
    "role": "",
    "claims": [
        {
            "claim_id": "",
            "claim_key": "",
            "claim_text": "",
            "element": "",
            "outcome": "",
            "tissue": "",
            "arm": "",
            "timepoint": "",
            "direction": "",
            "unit": "",
            "source_channel": "",
            "source_locator": "",
            "benchmark_comparable": None,
            "confidence": "",
            "evidence_quote": "",
            "notes": "",
        }
    ],
    "constraints": [
        {
            "constraint_type": "",
            "constraint_text": "",
            "applies_to_claim_keys": [],
            "confidence": "",
            "source_locator": "",
        }
    ],
    "contradictions": [
        {
            "contradiction_type": "",
            "claim_keys": [],
            "description": "",
            "severity": "",
        }
    ],
    "notes": "",
}


def paper_context(paper_id: str) -> dict:
    report_path = REPORT_DIR / f"{paper_id}_report.md"
    input_candidates = [
        ROOT / "input" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "validated" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "validation_input" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "mineral_validation_input" / f"{paper_id}.pdf",
    ]
    pdf_paths = [str(p) for p in input_candidates if p.exists()]
    report_text = report_path.read_text(encoding="utf-8", errors="ignore") if report_path.exists() else ""
    return {
        "paper_id": paper_id,
        "pdf_paths": pdf_paths,
        "report_path": str(report_path) if report_path.exists() else "",
        "report_excerpt": report_text[:8000],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", nargs="*", default=DEFAULT_PAPERS)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    existing_index_path = OUT / "index.json"
    existing = {"papers": [], "roles": list(ROLE_PROMPTS)}
    if existing_index_path.exists():
        existing = json.loads(existing_index_path.read_text(encoding="utf-8"))
        existing.setdefault("papers", [])
        existing["roles"] = list(ROLE_PROMPTS)
    papers_by_id = {row.get("paper_id"): row for row in existing["papers"] if row.get("paper_id")}

    for paper_id in args.papers:
        context = paper_context(paper_id)
        paper_dir = OUT / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        role_records = []
        for role, prompt in ROLE_PROMPTS.items():
            record = {
                "paper_id": paper_id,
                "role": role,
                "prompt": prompt,
                "context": context,
                "output_schema": {
                    **OUTPUT_SCHEMA_TEMPLATE,
                    "paper_id": paper_id,
                    "role": role,
                    "claims": [],
                    "constraints": [],
                    "contradictions": [],
                },
            }
            role_records.append(record)
            out_path = paper_dir / f"{role}.json"
            if not out_path.exists():
                out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        papers_by_id[paper_id] = {
            "paper_id": paper_id,
            "dir": str(paper_dir),
            "pdf_paths": context["pdf_paths"],
            "role_files": [str(paper_dir / f"{role}.json") for role in ROLE_PROMPTS],
        }

    index_path = OUT / "index.json"
    updated_index = {
        "papers": [papers_by_id[k] for k in sorted(papers_by_id)],
        "roles": list(ROLE_PROMPTS),
    }
    index_path.write_text(json.dumps(updated_index, indent=2), encoding="utf-8")
    print(index_path)


if __name__ == "__main__":
    main()
