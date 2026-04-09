#!/usr/bin/env python3
"""
Build a starter paper-level convergence table for the sandbox papers.

This script integrates:
- PDF paths
- per-paper Claude/Kimi consensus JSONs
- inter-model agreement rows
- model comparison disagreement flags

Output:
- codex/outputs/combined_analysis/sandbox_paper_features_2026-03-26.csv
- codex/outputs/combined_analysis/sandbox_paper_features_2026-03-26.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
CODEX_DIR = ROOT / "pipeline_replication" / "codex"
OUTPUT_DIR = CODEX_DIR / "outputs" / "combined_analysis"

SANDBOX_PAPERS = [
    "020_Overdieck_1993",
    "031_Pal_2003",
    "002_Ziska_1997",
    "003_Baslam_2012",
    "004_Finzi_2001",
    "007_Woodin_1992",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_pdf_paths(paper_id: str) -> list[str]:
    candidates = [
        ROOT / "input" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "validated" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "validation_input" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "mineral_validation_input" / f"{paper_id}.pdf",
    ]
    return [str(p) for p in candidates if p.exists()]


def load_pairwise_counts() -> dict[str, dict]:
    path = ROOT / "output" / "inter_model_agreement" / "pairwise_comparison.csv"
    rows_by_paper: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_paper[row["paper_id"]].append(row)

    out: dict[str, dict] = {}
    for paper_id, rows in rows_by_paper.items():
        total = len(rows)
        all_agree = sum(str(r.get("all_agree", "")).lower() == "true" for r in rows)
        direction_complete = 0
        direction_all_same = 0
        for r in rows:
            dirs = [r.get("claude_dir"), r.get("kimi_dir"), r.get("gemini_dir")]
            dirs = [d for d in dirs if d]
            if len(dirs) >= 2:
                direction_complete += 1
                if len(set(dirs)) == 1:
                    direction_all_same += 1
        out[paper_id] = {
            "pairwise_rows": total,
            "pairwise_all_agree_rows": all_agree,
            "pairwise_all_agree_rate": round(all_agree / total, 4) if total else None,
            "direction_comparable_rows": direction_complete,
            "direction_agree_rows": direction_all_same,
            "direction_agreement_rate": round(direction_all_same / direction_complete, 4)
            if direction_complete
            else None,
        }
    return out


def load_disagreement_flags() -> dict[str, dict]:
    path = ROOT / "output" / "model_comparison" / "disagreement_analysis.json"
    data = load_json(path)

    element_mismatch = {}
    for item in data.get("papers_by_issue", {}).get("element_mismatch", []):
        element_mismatch[item["paper"]] = {
            "gemini_only_count": len(item.get("gemini_only", [])),
            "kimi_only_count": len(item.get("kimi_only", [])),
            "element_mismatch_flag": True,
        }

    high_disagreement = {
        item["paper"]: {
            "high_disagreement_count": item.get("disagree", 0),
            "high_disagreement_exact_count": item.get("exact", 0),
            "high_disagreement_flag": True,
        }
        for item in data.get("papers_by_issue", {}).get("high_disagreement", [])
    }

    tc_swaps = {
        item["paper"]: {
            "swap_count": item.get("swap_count", 0),
            "swap_risk_flag": True,
        }
        for item in data.get("papers_by_issue", {}).get("treatment_control_swaps", [])
    }

    merged: dict[str, dict] = defaultdict(dict)
    for source in (element_mismatch, high_disagreement, tc_swaps):
        for paper_id, vals in source.items():
            merged[paper_id].update(vals)
    return merged


def summarize_disagreements(disagreements: list[dict]) -> dict:
    type_counts = Counter()
    missing_control = 0
    missing_treatment = 0
    null_effect = 0

    for d in disagreements:
        dtype = d.get("type", "unknown")
        type_counts[dtype] += 1
        for side in ("claude", "kimi"):
            obs = d.get(side)
            if not obs:
                continue
            if obs.get("control_mean") in (None, ""):
                missing_control += 1
            if obs.get("treatment_mean") in (None, ""):
                missing_treatment += 1
            if obs.get("effect_pct") in (None, ""):
                null_effect += 1

    out = {
        "disagreement_count": len(disagreements),
        "disagreement_types": json.dumps(dict(type_counts), sort_keys=True),
        "missing_control_mentions": missing_control,
        "missing_treatment_mentions": missing_treatment,
        "null_effect_mentions": null_effect,
    }
    for key, value in type_counts.items():
        out[f"disagree_type_{key}"] = value
    return out


def summarize_consensus_observations(obs: list[dict]) -> dict:
    confidence_counts = Counter()
    data_sources = Counter()
    has_variance = 0
    has_notes = 0
    effect_values = []

    for row in obs:
        conf = row.get("confidence") or "unknown"
        confidence_counts[conf] += 1
        src = row.get("data_source") or "unknown"
        data_sources[src] += 1
        if row.get("treatment_variance") not in (None, "") or row.get("control_variance") not in (None, ""):
            has_variance += 1
        if row.get("notes"):
            has_notes += 1
        effect = row.get("effect_pct")
        if isinstance(effect, (int, float)):
            effect_values.append(effect)

    out = {
        "consensus_observation_count": len(obs),
        "consensus_confidence_counts": json.dumps(dict(confidence_counts), sort_keys=True),
        "consensus_data_sources": json.dumps(dict(data_sources), sort_keys=True),
        "consensus_rows_with_variance": has_variance,
        "consensus_rows_with_notes": has_notes,
    }
    if effect_values:
        out["consensus_mean_effect_pct"] = round(sum(effect_values) / len(effect_values), 4)
        out["consensus_min_effect_pct"] = round(min(effect_values), 4)
        out["consensus_max_effect_pct"] = round(max(effect_values), 4)
    else:
        out["consensus_mean_effect_pct"] = None
        out["consensus_min_effect_pct"] = None
        out["consensus_max_effect_pct"] = None
    return out


def build_rows() -> list[dict]:
    pairwise = load_pairwise_counts()
    disagreement_flags = load_disagreement_flags()

    rows = []
    for paper_id in SANDBOX_PAPERS:
        consensus_path = ROOT / "output" / "claude_kimi_full_comparison" / f"{paper_id}_consensus.json"
        data = load_json(consensus_path)
        recon = data.get("recon", {})
        warnings = recon.get("warnings", []) or []
        disagreements = data.get("disagreements", []) or []
        consensus_obs = data.get("consensus_observations", []) or []

        row = {
            "paper_id": paper_id,
            "pdf_paths": json.dumps(find_pdf_paths(paper_id)),
            "consensus_json_path": str(consensus_path),
            "claude_obs": data.get("claude_obs"),
            "kimi_obs": data.get("kimi_obs"),
            "matched_obs": data.get("matched_obs"),
            "agreement_fraction_vs_max_obs": round(
                (data.get("matched_obs") or 0) / max(data.get("claude_obs") or 0, data.get("kimi_obs") or 0, 1),
                4,
            ),
            "recon_warning_count": len(warnings),
            "recon_warnings": json.dumps(warnings, ensure_ascii=True),
            "has_tc_confusion": bool(recon.get("potential_tc_confusion")),
            "potential_tc_confusion": recon.get("potential_tc_confusion"),
            "variance_type": recon.get("variance_type"),
            "variance_confidence": recon.get("variance_confidence"),
            "sample_size_found": recon.get("sample_size_found"),
            "experimental_design": recon.get("experimental_design"),
            "factorial_structure_raw": recon.get("factorial_structure"),
            "has_factorial_structure": bool(recon.get("factorial_structure")),
            "tables_with_mineral_data_n": len(recon.get("tables_with_mineral_data", []) or []),
            "tables_with_mineral_data": json.dumps(recon.get("tables_with_mineral_data", [])),
        }
        row.update(summarize_disagreements(disagreements))
        row.update(summarize_consensus_observations(consensus_obs))
        row.update(pairwise.get(paper_id, {}))
        row.update(disagreement_flags.get(paper_id, {}))

        row.setdefault("element_mismatch_flag", False)
        row.setdefault("high_disagreement_flag", False)
        row.setdefault("swap_risk_flag", False)
        row.setdefault("gemini_only_count", 0)
        row.setdefault("kimi_only_count", 0)
        row.setdefault("high_disagreement_count", 0)
        row.setdefault("high_disagreement_exact_count", 0)
        row.setdefault("swap_count", 0)
        rows.append(row)

    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()

    csv_path = OUTPUT_DIR / "sandbox_paper_features_2026-03-26.csv"
    json_path = OUTPUT_DIR / "sandbox_paper_features_2026-03-26.json"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} sandbox paper rows")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
