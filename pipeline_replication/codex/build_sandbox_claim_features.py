#!/usr/bin/env python3
"""
Build a starter claim-level convergence table for the sandbox papers.

This version intentionally stays simple:
- one row per paper x normalized claim key
- claim key is paper_id + element + tissue
- integrates Claude/Kimi consensus/disagreement structure
- integrates Gemini/Claude/Kimi direction/effect agreement where available
- carries forward paper-level risk features

Outputs:
- codex/outputs/combined_analysis/sandbox_claim_features_2026-03-26.csv
- codex/outputs/combined_analysis/sandbox_claim_features_2026-03-26.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
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


def key_for_claim(paper_id: str, element: str | None, tissue: str | None) -> str:
    return f"{paper_id}::{(element or 'UNKNOWN').strip()}::{(tissue or 'UNKNOWN').strip()}"


def load_paper_features() -> dict[str, dict]:
    path = OUTPUT_DIR / "sandbox_paper_features_2026-03-26.csv"
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["paper_id"]] = row
    return out


def load_pairwise_rows() -> dict[str, dict]:
    path = ROOT / "output" / "inter_model_agreement" / "pairwise_comparison.csv"
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            paper_id = row["paper_id"]
            if paper_id not in SANDBOX_PAPERS:
                continue
            claim_key = key_for_claim(paper_id, row.get("element"), row.get("tissue"))
            out[claim_key] = {
                "pairwise_claim_available": True,
                "pairwise_element": row.get("element"),
                "pairwise_tissue": row.get("tissue"),
                "claude_eff": row.get("claude_eff"),
                "kimi_eff": row.get("kimi_eff"),
                "gemini_eff": row.get("gemini_eff"),
                "claude_dir": row.get("claude_dir"),
                "kimi_dir": row.get("kimi_dir"),
                "gemini_dir": row.get("gemini_dir"),
                "all_agree": row.get("all_agree"),
            }
    return out


def load_consensus_claims() -> dict[str, dict]:
    claims: dict[str, dict] = {}
    for paper_id in SANDBOX_PAPERS:
        path = ROOT / "output" / "claude_kimi_full_comparison" / f"{paper_id}_consensus.json"
        data = load_json(path)

        # Consensus observations become high-support claims.
        for obs in data.get("consensus_observations", []):
            claim_key = key_for_claim(paper_id, obs.get("element"), obs.get("tissue"))
            claims[claim_key] = {
                "paper_id": paper_id,
                "claim_key": claim_key,
                "element": obs.get("element"),
                "tissue": obs.get("tissue"),
                "claim_source": "consensus_observation",
                "consensus_support": 2,
                "consensus_effect_pct": obs.get("effect_pct"),
                "consensus_confidence": obs.get("confidence"),
                "consensus_data_source": obs.get("data_source"),
                "consensus_notes": obs.get("notes"),
                "has_variance": bool(
                    obs.get("treatment_variance") not in (None, "")
                    or obs.get("control_variance") not in (None, "")
                ),
                "has_n": obs.get("n") not in (None, ""),
                "claim_disagreement_type": "",
                "claim_disagreement_count": 0,
            }

        # Disagreements become claims too, often lower support / riskier.
        for d in data.get("disagreements", []):
            element = d.get("element")
            tissue = d.get("tissue")
            claim_key = key_for_claim(paper_id, element, tissue)
            base = claims.setdefault(
                claim_key,
                {
                    "paper_id": paper_id,
                    "claim_key": claim_key,
                    "element": element,
                    "tissue": tissue,
                    "claim_source": "disagreement_only",
                    "consensus_support": 0,
                    "consensus_effect_pct": None,
                    "consensus_confidence": None,
                    "consensus_data_source": None,
                    "consensus_notes": None,
                    "has_variance": False,
                    "has_n": False,
                    "claim_disagreement_type": "",
                    "claim_disagreement_count": 0,
                },
            )
            base["claim_disagreement_count"] += 1
            dtype = d.get("type", "unknown")
            if base["claim_disagreement_type"]:
                base["claim_disagreement_type"] += f";{dtype}"
            else:
                base["claim_disagreement_type"] = dtype

            for side in ("claude", "kimi"):
                obs = d.get(side)
                if obs:
                    base["has_variance"] = base["has_variance"] or bool(
                        obs.get("treatment_variance") not in (None, "")
                        or obs.get("control_variance") not in (None, "")
                    )
                    base["has_n"] = base["has_n"] or (obs.get("n") not in (None, ""))
                    if base["consensus_effect_pct"] is None and obs.get("effect_pct") not in (None, ""):
                        base["consensus_effect_pct"] = obs.get("effect_pct")
                    if base["consensus_confidence"] is None and obs.get("confidence"):
                        base["consensus_confidence"] = obs.get("confidence")
                    if base["consensus_data_source"] is None and obs.get("data_source"):
                        base["consensus_data_source"] = obs.get("data_source")
                    if base["consensus_notes"] is None and obs.get("notes"):
                        base["consensus_notes"] = obs.get("notes")
    return claims


def derive_direction_support(row: dict) -> dict:
    dirs = [row.get("claude_dir"), row.get("kimi_dir"), row.get("gemini_dir")]
    dirs = [d for d in dirs if d]
    unique_dirs = sorted(set(dirs))
    return {
        "cross_model_support_n": len(dirs),
        "cross_model_unique_directions_n": len(unique_dirs),
        "cross_model_direction_set": json.dumps(unique_dirs),
        "cross_model_direction_agreement": len(unique_dirs) == 1 if dirs else None,
    }


def derive_risk_flags(row: dict) -> dict:
    risk_flags = []

    if str(row.get("claim_disagreement_count", 0)) not in ("0", "", "None"):
        risk_flags.append("disagreement_present")
    if row.get("claim_disagreement_type"):
        if "claude_only" in row["claim_disagreement_type"] or "kimi_only" in row["claim_disagreement_type"]:
            risk_flags.append("partial_model_support")
    if str(row.get("swap_risk_flag", "")).lower() == "true":
        risk_flags.append("paper_swap_risk")
    if str(row.get("high_disagreement_flag", "")).lower() == "true":
        risk_flags.append("paper_high_disagreement")
    if str(row.get("element_mismatch_flag", "")).lower() == "true":
        risk_flags.append("paper_element_mismatch")
    if str(row.get("has_tc_confusion", "")).lower() == "true":
        risk_flags.append("tc_confusion")
    if str(row.get("has_factorial_structure", "")).lower() == "true":
        risk_flags.append("factorial_structure")
    if row.get("consensus_data_source") in (None, "", "unknown"):
        risk_flags.append("weak_source_specificity")
    if str(row.get("has_variance", "")).lower() != "true":
        risk_flags.append("missing_variance")

    row["risk_flags"] = json.dumps(sorted(set(risk_flags)))
    row["risk_flag_count"] = len(set(risk_flags))
    return row


def build_rows() -> list[dict]:
    paper_features = load_paper_features()
    pairwise_rows = load_pairwise_rows()
    claim_rows = load_consensus_claims()

    merged: list[dict] = []
    for claim_key, claim in sorted(claim_rows.items()):
        paper_id = claim["paper_id"]
        row = dict(claim)
        row.update(pairwise_rows.get(claim_key, {}))

        paper = paper_features[paper_id]
        carry = {
            "paper_agreement_fraction": paper.get("agreement_fraction_vs_max_obs"),
            "paper_recon_warning_count": paper.get("recon_warning_count"),
            "paper_recon_warnings": paper.get("recon_warnings"),
            "has_tc_confusion": paper.get("has_tc_confusion"),
            "has_factorial_structure": paper.get("has_factorial_structure"),
            "paper_pairwise_all_agree_rate": paper.get("pairwise_all_agree_rate"),
            "paper_direction_agreement_rate": paper.get("direction_agreement_rate"),
            "element_mismatch_flag": paper.get("element_mismatch_flag"),
            "high_disagreement_flag": paper.get("high_disagreement_flag"),
            "swap_risk_flag": paper.get("swap_risk_flag"),
        }
        row.update(carry)
        row.update(derive_direction_support(row))
        row = derive_risk_flags(row)
        merged.append(row)

    return merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()

    csv_path = OUTPUT_DIR / "sandbox_claim_features_2026-03-26.csv"
    json_path = OUTPUT_DIR / "sandbox_claim_features_2026-03-26.json"

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Wrote {len(rows)} sandbox claim rows")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
