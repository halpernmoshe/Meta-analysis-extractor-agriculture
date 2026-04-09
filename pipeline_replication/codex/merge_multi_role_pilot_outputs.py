#!/usr/bin/env python3
"""
Merge multi-role full-context outputs into a contradiction / consilience table.

This works even if role files are only partially populated. It is meant to make
the prototype executable end-to-end before the role readers themselves exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
PILOT_DIR = ROOT / "pipeline_replication" / "codex" / "outputs" / "multi_role_pilot"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def normalize_text(value: object) -> str:
    return str(value or "").strip()


MEASUREMENT_UNIT_HINTS = [
    "mg/kg",
    "g/kg",
    "mg/g",
    "ug/g",
    "µg/g",
    "ppm",
    "%",
    "ppb",
    "mg/plant",
    "g/plant",
]

EFFECT_METRIC_HINTS = [
    "lnrr",
    "effect_pct",
    "effect%",
    "percent_change",
    "rr",
    "hedges_g",
    "smd",
]


def classify_unit_like(value: str) -> tuple[str, str]:
    text = normalize_text(value).lower()
    if not text:
        return "", ""
    for hint in EFFECT_METRIC_HINTS:
        if hint in text:
            return "effect_metric", normalize_text(value)
    for hint in MEASUREMENT_UNIT_HINTS:
        if hint in text:
            return "measurement_unit", normalize_text(value)
    return "other_unit_like", normalize_text(value)


def role_payload(data: dict) -> dict:
    # Support both the scaffold format and a future "filled" direct format.
    if "output_schema" in data and isinstance(data["output_schema"], dict):
        payload = dict(data["output_schema"])
        payload.setdefault("paper_id", data.get("paper_id"))
        payload.setdefault("role", data.get("role"))
        return payload
    return data


def claim_identity(role: str, claim: dict, ordinal: int) -> str:
    if normalize_text(claim.get("claim_key")):
        return normalize_text(claim["claim_key"])
    if normalize_text(claim.get("claim_id")):
        return normalize_text(claim["claim_id"])
    parts = [
        normalize_text(claim.get("element")),
        normalize_text(claim.get("outcome")),
        normalize_text(claim.get("tissue")),
        normalize_text(claim.get("arm")),
        normalize_text(claim.get("timepoint")),
        normalize_text(claim.get("direction")),
    ]
    parts = [p for p in parts if p]
    if parts:
        return "::".join(parts)
    return f"{role}::claim_{ordinal:03d}"


def merge_paper(paper_dir: Path) -> tuple[list[dict], dict]:
    role_files = [
        p
        for p in sorted(paper_dir.glob("*.json"))
        if p.stem in {"design_agent", "narrative_agent", "table_agent", "figure_agent", "benchmark_agent", "consistency_agent"}
    ]
    roles_loaded = []
    claims_by_key: dict[str, dict] = {}
    paper_contradictions = []
    paper_constraints = []
    explicit_claim_flags: dict[str, set[str]] = defaultdict(set)
    explicit_claim_links: dict[str, list[dict]] = defaultdict(list)

    for role_file in role_files:
        data = load_json(role_file)
        payload = role_payload(data)
        role = payload.get("role") or role_file.stem
        roles_loaded.append(role)

        for i, claim in enumerate(payload.get("claims", []), start=1):
            key = claim_identity(role, claim, i)
            base = claims_by_key.setdefault(
                key,
                {
                    "paper_id": payload.get("paper_id") or paper_dir.name,
                    "claim_key": key,
                    "element": normalize_text(claim.get("element")),
                    "outcome": normalize_text(claim.get("outcome")),
                    "tissue": normalize_text(claim.get("tissue")),
                    "arm": normalize_text(claim.get("arm")),
                    "timepoint": normalize_text(claim.get("timepoint")),
                    "role_support_n": 0,
                    "roles_present": [],
                    "directions": [],
                    "measurement_units": [],
                    "effect_metrics": [],
                    "other_unit_like": [],
                    "source_channels": [],
                    "benchmark_comparable_values": [],
                    "role_confidences": [],
                    "claim_texts": [],
                    "evidence_quotes": [],
                    "local_contradiction_flags": [],
                },
            )
            base["role_support_n"] += 1
            base["roles_present"].append(role)
            if normalize_text(claim.get("direction")):
                base["directions"].append(normalize_text(claim.get("direction")))
            if normalize_text(claim.get("unit")):
                unit_type, unit_value = classify_unit_like(claim.get("unit"))
                if unit_type == "measurement_unit":
                    base["measurement_units"].append(unit_value)
                elif unit_type == "effect_metric":
                    base["effect_metrics"].append(unit_value)
                else:
                    base["other_unit_like"].append(unit_value)
            if normalize_text(claim.get("source_channel")):
                base["source_channels"].append(normalize_text(claim.get("source_channel")))
            if claim.get("benchmark_comparable") is not None:
                base["benchmark_comparable_values"].append(str(claim.get("benchmark_comparable")))
            if normalize_text(claim.get("confidence")):
                base["role_confidences"].append(normalize_text(claim.get("confidence")))
            if normalize_text(claim.get("claim_text")):
                base["claim_texts"].append(normalize_text(claim.get("claim_text")))
            if normalize_text(claim.get("evidence_quote")):
                base["evidence_quotes"].append(normalize_text(claim.get("evidence_quote")))

        for c in payload.get("constraints", []):
            paper_constraints.append(
                {
                    "paper_id": payload.get("paper_id") or paper_dir.name,
                    "role": role,
                    "constraint_type": normalize_text(c.get("constraint_type")),
                    "constraint_text": normalize_text(c.get("constraint_text")),
                    "applies_to_claim_keys": json.dumps(c.get("applies_to_claim_keys", [])),
                    "confidence": normalize_text(c.get("confidence")),
                    "source_locator": normalize_text(c.get("source_locator")),
                }
            )

        for c in payload.get("contradictions", []):
            ctype = normalize_text(c.get("contradiction_type"))
            claim_keys = list(c.get("claim_keys", []))
            if normalize_text(c.get("claim_key")):
                claim_keys.append(normalize_text(c.get("claim_key")))
            if normalize_text(c.get("against_claim_key")):
                claim_keys.append(normalize_text(c.get("against_claim_key")))
            claim_keys = [k for k in claim_keys if k]

            rec = {
                "paper_id": payload.get("paper_id") or paper_dir.name,
                "role": role,
                "contradiction_type": ctype,
                "claim_keys": json.dumps(claim_keys),
                "description": normalize_text(c.get("description")),
                "severity": normalize_text(c.get("severity")),
            }
            paper_contradictions.append(rec)
            for k in claim_keys:
                explicit_claim_flags[k].add(ctype)
                explicit_claim_links[k].append(rec)

    merged_rows = []
    for key, row in sorted(claims_by_key.items()):
        direction_set = sorted(set(row["directions"]))
        measurement_unit_set = sorted(set(row["measurement_units"]))
        effect_metric_set = sorted(set(row["effect_metrics"]))
        other_unit_like_set = sorted(set(row["other_unit_like"]))
        benchmark_vals = sorted(set(row["benchmark_comparable_values"]))
        contradiction_flags = []
        if len(direction_set) > 1:
            contradiction_flags.append("direction_conflict")
        if len(measurement_unit_set) > 1:
            contradiction_flags.append("unit_conflict")
        if len(effect_metric_set) > 1:
            contradiction_flags.append("effect_metric_conflict")
        if len(benchmark_vals) > 1:
            contradiction_flags.append("benchmark_comparability_conflict")
        if row["role_support_n"] <= 1:
            contradiction_flags.append("single_role_only")
        if not row["source_channels"]:
            contradiction_flags.append("missing_source_channel")
        contradiction_flags.extend(sorted(explicit_claim_flags.get(key, set())))

        merged_rows.append(
            {
                "paper_id": row["paper_id"],
                "claim_key": key,
                "element": row["element"],
                "outcome": row["outcome"],
                "tissue": row["tissue"],
                "arm": row["arm"],
                "timepoint": row["timepoint"],
                "role_support_n": row["role_support_n"],
                "roles_present": json.dumps(sorted(set(row["roles_present"]))),
                "direction_set": json.dumps(direction_set),
                "measurement_unit_set": json.dumps(measurement_unit_set),
                "effect_metric_set": json.dumps(effect_metric_set),
                "other_unit_like_set": json.dumps(other_unit_like_set),
                "source_channel_set": json.dumps(sorted(set(row["source_channels"]))),
                "benchmark_comparable_set": json.dumps(benchmark_vals),
                "role_confidence_set": json.dumps(sorted(set(row["role_confidences"]))),
                "claim_text_examples": json.dumps(row["claim_texts"][:3]),
                "evidence_quote_examples": json.dumps(row["evidence_quotes"][:2]),
                "explicit_contradiction_details": json.dumps(explicit_claim_links.get(key, [])[:3]),
                "contradiction_flags": json.dumps(sorted(set(contradiction_flags))),
                "contradiction_count": len(set(contradiction_flags)),
            }
        )

    summary = {
        "paper_id": paper_dir.name,
        "roles_loaded": roles_loaded,
        "n_claim_rows": len(merged_rows),
        "n_constraints": len(paper_constraints),
        "n_role_contradictions": len(paper_contradictions),
        "claim_contradiction_counts": dict(
            Counter(flag for row in merged_rows for flag in json.loads(row["contradiction_flags"]))
        ),
    }
    return merged_rows, {"summary": summary, "constraints": paper_constraints, "contradictions": paper_contradictions}


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", type=str, default=None, help="Optional single paper_id under outputs/multi_role_pilot")
    args = parser.parse_args()

    paper_dirs = [PILOT_DIR / args.paper] if args.paper else [p for p in PILOT_DIR.iterdir() if p.is_dir()]
    overall_path = PILOT_DIR / "merged_index.json"
    overall = {}
    if overall_path.exists():
        overall = load_json(overall_path)
    for paper_dir in paper_dirs:
        merged_rows, meta = merge_paper(paper_dir)
        out_csv = paper_dir / "merged_claims.csv"
        out_json = paper_dir / "merged_summary.json"
        out_md = paper_dir / "merged_summary.md"

        write_csv(out_csv, merged_rows)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        lines = [
            f"# Multi-Role Merge: {paper_dir.name}",
            "",
            f"- Roles loaded: {', '.join(meta['summary']['roles_loaded'])}",
            f"- Claim rows: {meta['summary']['n_claim_rows']}",
            f"- Constraints: {meta['summary']['n_constraints']}",
            f"- Role contradictions: {meta['summary']['n_role_contradictions']}",
            f"- Claim contradiction counts: {json.dumps(meta['summary']['claim_contradiction_counts'], sort_keys=True)}",
        ]
        with out_md.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        overall[paper_dir.name] = meta["summary"]
        print(out_csv)
        print(out_json)
        print(out_md)

    with overall_path.open("w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2)
    print(overall_path)


if __name__ == "__main__":
    main()
