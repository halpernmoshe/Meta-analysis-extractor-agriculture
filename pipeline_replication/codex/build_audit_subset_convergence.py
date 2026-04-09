#!/usr/bin/env python3
"""
Build a larger convergence/audit subset from existing paper-level audit files.

This extends the earlier 6-paper sandbox into a broader pilot set that includes:
- clean controls
- high-disagreement structural cases
- papers from concordant_error_audit_v2 with known root-cause diagnoses

Outputs:
- codex/outputs/combined_analysis/audit_subset_paper_features_2026-03-26.csv
- codex/outputs/combined_analysis/audit_subset_claim_features_2026-03-26.csv
- codex/outputs/combined_analysis/audit_subset_claim_labels_2026-03-26.csv
- codex/outputs/combined_analysis/audit_subset_label_analysis_2026-03-26.json
- codex/outputs/combined_analysis/audit_subset_label_analysis_2026-03-26.md
"""

from __future__ import annotations

import csv
import json
import argparse
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
CODEX_DIR = ROOT / "pipeline_replication" / "codex"
OUTPUT_DIR = CODEX_DIR / "outputs" / "combined_analysis"
REPORT_DIR = ROOT / "output" / "loladze_combined_51" / "per_paper"

DEFAULT_PAPERS = [
    "020_Overdieck_1993",
    "031_Pal_2003",
    "002_Ziska_1997",
    "003_Baslam_2012",
    "004_Finzi_2001",
    "005_Niinemets_1999",
    "007_Woodin_1992",
    "011_Huluka_1994",
    "016_Fernando_2012a",
    "017_Fangmeier_2002",
    "021_Wilsey_1994",
    "040_Pfirrmann_1996",
    "044_Housman_2012",
    "051_Niu_2013",
]
ACTIVE_PAPERS = list(DEFAULT_PAPERS)

CONTROL_PAPERS = {"020_Overdieck_1993", "031_Pal_2003"}

PAPER_ROOT_CAUSE = {
    "011_Huluka_1994": {
        "paper_root_cause": "extraction_coverage_limitation",
        "paper_root_cause_confidence": "high",
        "paper_root_cause_note": "Wrong temporal point in extracted data; GT uses DOY 247 values not accessible to all models.",
    },
    "017_Fangmeier_2002": {
        "paper_root_cause": "matching_alignment_artifact",
        "paper_root_cause_confidence": "high",
        "paper_root_cause_note": "Correct condition-specific values existed in extractions but matcher averaged across tissue/harvest/condition.",
    },
    "021_Wilsey_1994": {
        "paper_root_cause": "matching_alignment_artifact",
        "paper_root_cause_confidence": "high",
        "paper_root_cause_note": "Clipped and unclipped conditions were averaged together; GT corresponds to unclipped only.",
    },
    "044_Housman_2012": {
        "paper_root_cause": "matching_alignment_artifact",
        "paper_root_cause_confidence": "high",
        "paper_root_cause_note": "Models extracted per-year values while GT used rainfall-averaged value.",
    },
}

ELEMENT_ALIASES = {
    "CA": "Ca",
    "FE": "Fe",
    "MG": "Mg",
    "MN": "Mn",
    "ZN": "Zn",
    "CU": "Cu",
    "MO": "Mo",
    "CO": "Co",
    "NA": "Na",
    "S": "S",
    "N": "N",
    "P": "P",
    "K": "K",
    "B": "B",
    "C": "C",
    "TNC": "TNC",
    "LIGNIN": "LIGNIN",
    "AMYLOSE": "AMYLOSE",
}

TISSUE_ALIASES = {
    "whole plant": "whole",
    "whole": "whole",
    "shoot": "whole",
    "aboveground": "aboveground",
    "above ground": "aboveground",
    "leaf": "leaf",
    "leaves": "leaf",
    "needle": "needle",
    "needles": "needle",
    "grain": "grain",
    "seed": "grain",
    "tuber": "tuber",
    "foliar": "leaf",
}

TISSUE_BASE_GROUPS = {
    "whole": "whole",
    "leaf": "leaf",
    "aboveground": "aboveground",
    "needle": "needle",
    "grain": "grain",
    "tuber": "tuber",
    "unknown": "unknown",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_paper_list(path: Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PAPERS)
    papers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            papers.append(line)
    return papers


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def is_zero_match_summary(text: str) -> bool:
    return ("zero-match" in text) or bool(__import__("re").search(r"\*\*match summary:\*\*\s*0/\d+", text))


def infer_report_root_cause(paper_id: str) -> dict:
    path = REPORT_DIR / f"{paper_id}_report.md"
    if not path.exists():
        return {}

    lower = read_text(path).lower()

    if (
        "**status:** skip" in lower
        or "figure-digitization failure" in lower
        or "requires figure digitization" in lower
        or "no concentration data" in lower
        or "only in figure" in lower
        or "figure-only data" in lower
    ):
        return {
            "paper_root_cause": "extraction_coverage_limitation",
            "paper_root_cause_confidence": "high",
            "paper_root_cause_note": "Inferred from per-paper report: skip / figure-only / non-tabular target data.",
        }

    if (
        "wrong tissue" in lower
        or "tissue mismatch" in lower
        or "tissue-mismatched" in lower
        or "treatment-arm confusion" in lower
        or "treatment arm ambiguity" in lower
        or "clean-air (20 nmol o3) condition" in lower
        or "factorial aggregation problem" in lower
        or "ec-only" in lower
        or "ec+eo" in lower
        or "averaged across both o3 levels" in lower
        or "wrong biological arm" in lower
    ):
        return {
            "paper_root_cause": "matching_alignment_artifact",
            "paper_root_cause_confidence": "high",
            "paper_root_cause_note": "Inferred from per-paper report: tissue / arm / factorial alignment issue.",
        }

    return {}


def infer_report_signals(paper_id: str) -> dict:
    path = REPORT_DIR / f"{paper_id}_report.md"
    if not path.exists():
        return {}

    lower = read_text(path).lower()
    return {
        "report_status_skip": "**status:** skip" in lower,
        "report_zero_match": is_zero_match_summary(lower),
        "report_wrong_tissue": ("wrong tissue" in lower) or ("tissue mismatch" in lower) or ("tissue-mismatched" in lower),
        "report_treatment_arm_confusion": ("treatment-arm confusion" in lower)
        or ("treatment arm ambiguity" in lower)
        or ("ec-only" in lower)
        or ("ec+eo" in lower)
        or ("clean-air (20 nmol o3) condition" in lower)
        or ("factorial aggregation problem" in lower)
        or ("averaged across both o3 levels" in lower),
        "report_no_concentration_data": ("no concentration data" in lower) or ("non-concentration units" in lower),
        "report_figure_digitization_limit": ("figure-digitization failure" in lower)
        or ("requires figure digitization" in lower)
        or ("only in figure" in lower)
        or ("figure-only data" in lower),
        "report_overall_partial": "**overall rating:** partial" in lower,
    }


def find_pdf_paths(paper_id: str) -> list[str]:
    candidates = [
        ROOT / "input" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "validated" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "validation_input" / f"{paper_id}.pdf",
        ROOT.parent / "Loladze" / "mineral_validation_input" / f"{paper_id}.pdf",
    ]
    return [str(p) for p in candidates if p.exists()]


def normalize_element(element: str | None) -> str:
    raw = (element or "UNKNOWN").strip()
    if not raw:
        return "UNKNOWN"
    upper = raw.upper()
    if upper in ELEMENT_ALIASES:
        return ELEMENT_ALIASES[upper]
    return raw


def normalize_tissue(tissue: str | None) -> str:
    raw = (tissue or "UNKNOWN").strip().lower()
    if not raw:
        return "unknown"
    return TISSUE_ALIASES.get(raw, raw)


def key_for_claim(paper_id: str, element: str | None, tissue: str | None) -> str:
    return f"{paper_id}::{normalize_element(element)}::{normalize_tissue(tissue)}"


def tissue_base_group(tissue: str | None) -> str:
    return TISSUE_BASE_GROUPS.get(normalize_tissue(tissue), normalize_tissue(tissue))


def truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def derive_warning_flags(warnings: list[str], recon: dict) -> dict:
    joined = " || ".join(warnings).lower()
    return {
        "warning_timepoint_risk": ("multiple sampling dates" in joined) or ("timepoint" in joined),
        "warning_figure_only_risk": ("figure" in joined) or ("figures" in joined),
        "warning_averaging_risk": ("average" in joined)
        or ("averaged" in joined)
        or ("multiple years" in joined)
        or ("rainfall" in joined)
        or ("clipped vs unclipped" in joined)
        or ("clipped" in joined and "unclipped" in joined)
        or ("harvest dates combined" in joined)
        or ("combined" in joined and "sites" in joined),
        "warning_factorial_risk": ("factorial" in joined) or bool(recon.get("factorial_structure")),
        "warning_tc_confusion_risk": ("confusion" in joined) or bool(recon.get("potential_tc_confusion")),
        "warning_multi_condition_risk": ("multiple years" in joined)
        or ("multiple treatment" in joined)
        or ("multiple co2 levels" in joined)
        or ("multiple sites" in joined)
        or ("site" in joined and "varying" in joined),
        "warning_sparse_recon_risk": any("recon error" in w.lower() for w in warnings),
    }


def derive_construct_drift_flags(row: dict) -> dict:
    flags: list[str] = []

    if truthy(row.get("report_no_concentration_data")):
        flags.append("concentration_vs_content")
    if truthy(row.get("report_wrong_tissue")):
        flags.append("tissue_mismatch")
    if truthy(row.get("report_treatment_arm_confusion")):
        flags.append("arm_mismatch")
    if truthy(row.get("warning_timepoint_risk")) or truthy(row.get("report_timepoint_conflict")):
        flags.append("timepoint_mismatch")
    if (
        truthy(row.get("warning_averaging_risk"))
        or truthy(row.get("report_averaging_conflict"))
        or truthy(row.get("warning_multi_condition_risk"))
        or truthy(row.get("report_factorial_arm_selection_issue"))
    ):
        flags.append("pooled_vs_subgroup_mismatch")
    if truthy(row.get("warning_figure_only_risk")) or truthy(row.get("report_figure_digitization_limit")):
        flags.append("figure_only_target")

    unique_flags = sorted(set(flags))
    return {
        "construct_drift_flags": json.dumps(unique_flags),
        "construct_drift_count": len(unique_flags),
        "drift_concentration_vs_content": "concentration_vs_content" in unique_flags,
        "drift_tissue_mismatch": "tissue_mismatch" in unique_flags,
        "drift_arm_mismatch": "arm_mismatch" in unique_flags,
        "drift_timepoint_mismatch": "timepoint_mismatch" in unique_flags,
        "drift_pooled_vs_subgroup_mismatch": "pooled_vs_subgroup_mismatch" in unique_flags,
        "drift_figure_only_target": "figure_only_target" in unique_flags,
    }


def summarize_disagreements(disagreements: list[dict]) -> dict:
    type_counts = Counter()
    for item in disagreements:
        type_counts[item.get("type", "unknown")] += 1
    out = {
        "disagreement_count": len(disagreements),
        "disagreement_types": json.dumps(dict(type_counts), sort_keys=True),
    }
    for key, value in type_counts.items():
        out[f"disagree_type_{key}"] = value
    return out


def summarize_consensus_observations(obs: list[dict]) -> dict:
    confidence_counts = Counter()
    data_sources = Counter()
    effect_values = []
    for row in obs:
        confidence_counts[row.get("confidence") or "unknown"] += 1
        data_sources[row.get("data_source") or "unknown"] += 1
        effect = row.get("effect_pct")
        if isinstance(effect, (int, float)):
            effect_values.append(effect)
    return {
        "consensus_observation_count": len(obs),
        "consensus_confidence_counts": json.dumps(dict(confidence_counts), sort_keys=True),
        "consensus_data_sources": json.dumps(dict(data_sources), sort_keys=True),
        "consensus_mean_effect_pct": round(sum(effect_values) / len(effect_values), 4) if effect_values else None,
    }


def load_pairwise_rows() -> dict[str, list[dict]]:
    path = ROOT / "output" / "inter_model_agreement" / "pairwise_comparison.csv"
    rows_by_paper: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            paper_id = row["paper_id"]
            if paper_id in ACTIVE_PAPERS:
                rows_by_paper[paper_id].append(row)
    return rows_by_paper


def load_pairwise_claim_rows() -> dict[str, dict]:
    out: dict[str, dict] = {}
    element_level: dict[str, dict] = {}
    for paper_id, rows in load_pairwise_rows().items():
        for row in rows:
            norm_element = normalize_element(row.get("element"))
            norm_tissue = normalize_tissue(row.get("tissue"))
            claim_key = f"{paper_id}::{norm_element}::{norm_tissue}"
            base = out.setdefault(
                claim_key,
                {
                    "pairwise_claim_available": True,
                    "pairwise_element": norm_element,
                    "pairwise_tissue": norm_tissue,
                    "pairwise_row_count": 0,
                    "claude_dirs": set(),
                    "kimi_dirs": set(),
                    "gemini_dirs": set(),
                    "all_agree_true_count": 0,
                },
            )
            base["pairwise_row_count"] += 1
            if str(row.get("all_agree", "")).lower() == "true":
                base["all_agree_true_count"] += 1
            for src, field in (("claude_dir", "claude_dirs"), ("kimi_dir", "kimi_dirs"), ("gemini_dir", "gemini_dirs")):
                val = row.get(src)
                if val:
                    base[field].add(val)

            relaxed_key = f"{paper_id}::{norm_element}"
            relaxed = element_level.setdefault(
                relaxed_key,
                {
                    "pairwise_relaxed_row_count": 0,
                    "relaxed_tissues": set(),
                    "claude_dirs": set(),
                    "kimi_dirs": set(),
                    "gemini_dirs": set(),
                },
            )
            relaxed["pairwise_relaxed_row_count"] += 1
            relaxed["relaxed_tissues"].add(norm_tissue)
            for src, field in (("claude_dir", "claude_dirs"), ("kimi_dir", "kimi_dirs"), ("gemini_dir", "gemini_dirs")):
                val = row.get(src)
                if val:
                    relaxed[field].add(val)

    final: dict[str, dict] = {}
    for claim_key, row in out.items():
        claude_dirs = sorted(row.pop("claude_dirs"))
        kimi_dirs = sorted(row.pop("kimi_dirs"))
        gemini_dirs = sorted(row.pop("gemini_dirs"))
        final[claim_key] = {
            **row,
            "claude_dir": claude_dirs[0] if len(claude_dirs) == 1 else None,
            "kimi_dir": kimi_dirs[0] if len(kimi_dirs) == 1 else None,
            "gemini_dir": gemini_dirs[0] if len(gemini_dirs) == 1 else None,
            "claude_dir_set": json.dumps(claude_dirs),
            "kimi_dir_set": json.dumps(kimi_dirs),
            "gemini_dir_set": json.dumps(gemini_dirs),
            "pairwise_all_agree_rate_claim": round(row["all_agree_true_count"] / row["pairwise_row_count"], 4)
            if row["pairwise_row_count"]
            else None,
        }
    for relaxed_key, row in element_level.items():
        element_level[relaxed_key] = {
            "pairwise_relaxed_row_count": row["pairwise_relaxed_row_count"],
            "pairwise_relaxed_tissues": json.dumps(sorted(row["relaxed_tissues"])),
            "pairwise_relaxed_claude_dir_set": json.dumps(sorted(row["claude_dirs"])),
            "pairwise_relaxed_kimi_dir_set": json.dumps(sorted(row["kimi_dirs"])),
            "pairwise_relaxed_gemini_dir_set": json.dumps(sorted(row["gemini_dirs"])),
            "pairwise_relaxed_support_n": sum(1 for dirs in (row["claude_dirs"], row["kimi_dirs"], row["gemini_dirs"]) if dirs),
        }
    return {"exact": final, "relaxed": element_level}


def load_pairwise_paper_counts() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for paper_id, rows in load_pairwise_rows().items():
        total = len(rows)
        all_agree = sum(str(r.get("all_agree", "")).lower() == "true" for r in rows)
        direction_comparable = 0
        direction_agree = 0
        for row in rows:
            dirs = [row.get("claude_dir"), row.get("kimi_dir"), row.get("gemini_dir")]
            dirs = [d for d in dirs if d]
            if len(dirs) >= 2:
                direction_comparable += 1
                if len(set(dirs)) == 1:
                    direction_agree += 1
        out[paper_id] = {
            "pairwise_rows": total,
            "pairwise_all_agree_rows": all_agree,
            "pairwise_all_agree_rate": round(all_agree / total, 4) if total else None,
            "direction_comparable_rows": direction_comparable,
            "direction_agree_rows": direction_agree,
            "direction_agreement_rate": round(direction_agree / direction_comparable, 4) if direction_comparable else None,
        }
    return out


def load_disagreement_flags() -> dict[str, dict]:
    path = ROOT / "output" / "model_comparison" / "disagreement_analysis.json"
    data = load_json(path)
    merged: dict[str, dict] = defaultdict(dict)

    for item in data.get("papers_by_issue", {}).get("element_mismatch", []):
        merged[item["paper"]].update(
            {
                "element_mismatch_flag": True,
                "gemini_only_count": len(item.get("gemini_only", [])),
                "kimi_only_count": len(item.get("kimi_only", [])),
            }
        )
    for item in data.get("papers_by_issue", {}).get("high_disagreement", []):
        merged[item["paper"]].update(
            {
                "high_disagreement_flag": True,
                "high_disagreement_count": item.get("disagree", 0),
                "high_disagreement_exact_count": item.get("exact", 0),
            }
        )
    for item in data.get("papers_by_issue", {}).get("treatment_control_swaps", []):
        merged[item["paper"]].update(
            {
                "swap_risk_flag": True,
                "swap_count": item.get("swap_count", 0),
            }
        )
    return merged


def build_paper_rows() -> list[dict]:
    pairwise_counts = load_pairwise_paper_counts()
    disagreement_flags = load_disagreement_flags()
    rows = []

    for paper_id in ACTIVE_PAPERS:
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
            "variance_type": recon.get("variance_type"),
            "variance_confidence": recon.get("variance_confidence"),
            "sample_size_found": recon.get("sample_size_found"),
            "experimental_design": recon.get("experimental_design"),
            "has_factorial_structure": bool(recon.get("factorial_structure")),
            "tables_with_mineral_data_n": len(recon.get("tables_with_mineral_data", []) or []),
            "is_control_paper": paper_id in CONTROL_PAPERS,
        }
        row.update(derive_warning_flags(warnings, recon))
        row.update(summarize_disagreements(disagreements))
        row.update(summarize_consensus_observations(consensus_obs))
        row.update(pairwise_counts.get(paper_id, {}))
        row.update(disagreement_flags.get(paper_id, {}))
        row.update(infer_report_signals(paper_id))
        row.update(PAPER_ROOT_CAUSE.get(paper_id, {}))
        if not row.get("paper_root_cause"):
            row.update(infer_report_root_cause(paper_id))

        row.setdefault("element_mismatch_flag", False)
        row.setdefault("high_disagreement_flag", False)
        row.setdefault("swap_risk_flag", False)
        row.setdefault("gemini_only_count", 0)
        row.setdefault("kimi_only_count", 0)
        row.setdefault("high_disagreement_count", 0)
        row.setdefault("high_disagreement_exact_count", 0)
        row.setdefault("swap_count", 0)
        row.setdefault("paper_root_cause", "")
        row.setdefault("paper_root_cause_confidence", "")
        row.setdefault("paper_root_cause_note", "")
        row.setdefault("report_status_skip", False)
        row.setdefault("report_zero_match", False)
        row.setdefault("report_wrong_tissue", False)
        row.setdefault("report_treatment_arm_confusion", False)
        row.setdefault("report_no_concentration_data", False)
        row.setdefault("report_figure_digitization_limit", False)
        row.setdefault("report_overall_partial", False)
        rows.append(row)
    return rows


def build_claim_rows(paper_rows: list[dict]) -> list[dict]:
    paper_index = {row["paper_id"]: row for row in paper_rows}
    pairwise_struct = load_pairwise_claim_rows()
    pairwise_claim_rows = pairwise_struct["exact"]
    pairwise_relaxed_rows = pairwise_struct["relaxed"]
    claims: dict[str, dict] = {}

    for paper_id in ACTIVE_PAPERS:
        path = ROOT / "output" / "claude_kimi_full_comparison" / f"{paper_id}_consensus.json"
        data = load_json(path)

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

        for item in data.get("disagreements", []):
            claim_key = key_for_claim(paper_id, item.get("element"), item.get("tissue"))
            base = claims.setdefault(
                claim_key,
                {
                    "paper_id": paper_id,
                    "claim_key": claim_key,
                    "element": item.get("element"),
                    "tissue": item.get("tissue"),
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
            dtype = item.get("type", "unknown")
            base["claim_disagreement_type"] = f"{base['claim_disagreement_type']};{dtype}".strip(";")
            for side in ("claude", "kimi"):
                obs = item.get(side)
                if not obs:
                    continue
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

    merged = []
    for claim_key, claim in sorted(claims.items()):
        row = dict(claim)
        row.update(pairwise_claim_rows.get(claim_key, {}))
        relaxed_key = f"{row['paper_id']}::{normalize_element(row.get('element'))}"
        row.update(pairwise_relaxed_rows.get(relaxed_key, {}))
        paper = paper_index[row["paper_id"]]
        row.update(
            {
                "paper_agreement_fraction": paper.get("agreement_fraction_vs_max_obs"),
                "paper_recon_warning_count": paper.get("recon_warning_count"),
                "paper_recon_warnings": paper.get("recon_warnings"),
                "paper_pairwise_all_agree_rate": paper.get("pairwise_all_agree_rate"),
                "paper_direction_agreement_rate": paper.get("direction_agreement_rate"),
                "has_tc_confusion": paper.get("has_tc_confusion"),
                "has_factorial_structure": paper.get("has_factorial_structure"),
                "element_mismatch_flag": paper.get("element_mismatch_flag"),
                "high_disagreement_flag": paper.get("high_disagreement_flag"),
                "swap_risk_flag": paper.get("swap_risk_flag"),
                "paper_root_cause": paper.get("paper_root_cause"),
                "paper_root_cause_confidence": paper.get("paper_root_cause_confidence"),
                "paper_root_cause_note": paper.get("paper_root_cause_note"),
                "is_control_paper": paper.get("is_control_paper"),
                "warning_timepoint_risk": paper.get("warning_timepoint_risk"),
                "warning_figure_only_risk": paper.get("warning_figure_only_risk"),
                "warning_averaging_risk": paper.get("warning_averaging_risk"),
                "warning_factorial_risk": paper.get("warning_factorial_risk"),
                "warning_tc_confusion_risk": paper.get("warning_tc_confusion_risk"),
                "warning_multi_condition_risk": paper.get("warning_multi_condition_risk"),
                "warning_sparse_recon_risk": paper.get("warning_sparse_recon_risk"),
                "report_status_skip": paper.get("report_status_skip"),
                "report_zero_match": paper.get("report_zero_match"),
                "report_wrong_tissue": paper.get("report_wrong_tissue"),
                "report_treatment_arm_confusion": paper.get("report_treatment_arm_confusion"),
                "report_no_concentration_data": paper.get("report_no_concentration_data"),
                "report_figure_digitization_limit": paper.get("report_figure_digitization_limit"),
                "report_overall_partial": paper.get("report_overall_partial"),
            }
        )

        dirs = [row.get("claude_dir"), row.get("kimi_dir"), row.get("gemini_dir")]
        dirs = [d for d in dirs if d]
        direction_set = sorted(set(dirs))
        row["cross_model_support_n"] = len(dirs)
        row["cross_model_direction_set"] = json.dumps(direction_set)
        row["cross_model_direction_agreement"] = len(direction_set) == 1 if dirs else None

        risk_flags = []
        if int(row.get("claim_disagreement_count") or 0) > 0:
            risk_flags.append("disagreement_present")
        if "claude_only" in (row.get("claim_disagreement_type") or "") or "kimi_only" in (
            row.get("claim_disagreement_type") or ""
        ):
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
        if str(row.get("warning_timepoint_risk", "")).lower() == "true":
            risk_flags.append("warning_timepoint_risk")
        if str(row.get("warning_figure_only_risk", "")).lower() == "true":
            risk_flags.append("warning_figure_only_risk")
        if str(row.get("warning_averaging_risk", "")).lower() == "true":
            risk_flags.append("warning_averaging_risk")
        if str(row.get("warning_multi_condition_risk", "")).lower() == "true":
            risk_flags.append("warning_multi_condition_risk")
        if str(row.get("warning_sparse_recon_risk", "")).lower() == "true":
            risk_flags.append("warning_sparse_recon_risk")
        if row.get("consensus_data_source") in (None, "", "unknown"):
            risk_flags.append("weak_source_specificity")
        if str(row.get("has_variance", "")).lower() != "true":
            risk_flags.append("missing_variance")
        if row.get("paper_root_cause") == "matching_alignment_artifact":
            risk_flags.append("paper_alignment_known")
        if row.get("paper_root_cause") == "extraction_coverage_limitation":
            risk_flags.append("paper_extraction_coverage_known")

        row["risk_flags"] = json.dumps(sorted(set(risk_flags)))
        row["risk_flag_count"] = len(set(risk_flags))
        row.update(derive_construct_drift_flags(row))
        merged.append(row)
    return merged


def assign_label(row: dict) -> tuple[str, str, str]:
    disagreements = int(row.get("claim_disagreement_count") or 0)
    support_n = int(row.get("cross_model_support_n") or 0)
    direction_agree = str(row.get("cross_model_direction_agreement")).strip().lower() == "true"
    source = row.get("claim_source") or ""
    flags = set(json.loads(row.get("risk_flags") or "[]"))
    notes = (row.get("consensus_notes") or "").lower()
    paper_root_cause = row.get("paper_root_cause") or ""
    report_status_skip = str(row.get("report_status_skip", "")).lower() == "true"
    report_zero_match = str(row.get("report_zero_match", "")).lower() == "true"
    report_wrong_tissue = str(row.get("report_wrong_tissue", "")).lower() == "true"
    report_treatment_arm_confusion = str(row.get("report_treatment_arm_confusion", "")).lower() == "true"
    report_no_concentration_data = str(row.get("report_no_concentration_data", "")).lower() == "true"
    report_figure_digitization_limit = str(row.get("report_figure_digitization_limit", "")).lower() == "true"

    if str(row.get("is_control_paper", "")).lower() == "true" and disagreements == 0:
        return ("clean_support", "high", "Control paper with consensus support and no recorded disagreement.")

    if report_status_skip or report_zero_match or report_no_concentration_data:
        return (
            "likely_extraction_coverage_problem",
            "high",
            "Report explicitly marks the paper as skipped / zero-match / non-comparable concentration data.",
        )

    if paper_root_cause == "extraction_coverage_limitation":
        if source == "disagreement_only" or disagreements > 0 or report_wrong_tissue or report_treatment_arm_confusion:
            return (
                "likely_alignment_or_structure_problem",
                "medium",
                "Coverage-limited paper, but this specific claim shows disagreement or wrong-tissue/arm structure.",
            )
        if source == "consensus_observation" and support_n >= 2 and direction_agree and disagreements == 0 and not report_figure_digitization_limit:
            return (
                "clean_support",
                "medium",
                "Coverage-limited paper, but this specific claim is stable and consensus-backed rather than figure-missing.",
            )
        return (
            "likely_extraction_coverage_problem",
            "high",
            "Paper is audited as a coverage limitation and this claim is not strongly rescued by stable consensus support.",
        )

    if paper_root_cause == "matching_alignment_artifact" or report_wrong_tissue or report_treatment_arm_confusion:
        return (
            "likely_alignment_or_structure_problem",
            "high",
            "Paper/report indicates tissue, arm, or matching/alignment artifacts.",
        )

    if "proxy" in notes or "averaged across" in notes or "avg of" in notes:
        return (
            "likely_alignment_or_structure_problem",
            "medium",
            "Claim notes indicate averaging or proxy use across non-target design dimensions.",
        )

    if "paper_swap_risk" in flags or "paper_high_disagreement" in flags or "paper_element_mismatch" in flags:
        return (
            "likely_alignment_or_structure_problem",
            "medium",
            "Paper-level disagreement, swap risk, or element mismatch suggests structure/matching trouble.",
        )

    if "factorial_structure" in flags and "tc_confusion" in flags and disagreements > 0:
        return (
            "likely_alignment_or_structure_problem",
            "medium",
            "Factorial structure and treatment/control ambiguity raise alignment risk.",
        )

    if disagreements == 0 and source == "consensus_observation":
        return (
            "clean_support",
            "medium",
            "Consensus-backed claim with no recorded Claude/Kimi disagreement.",
        )

    if source == "disagreement_only":
        return (
            "unclear",
            "low",
            "Claim appears only in disagreement structure without stable consensus support.",
        )

    if support_n >= 2 and direction_agree and disagreements <= 1:
        return (
            "clean_support",
            "medium",
            "Cross-model direction support is present with limited disagreement.",
        )

    return ("unclear", "low", "Current evidence is mixed; keep unresolved in the pilot subset.")


def build_labels(claim_rows: list[dict]) -> list[dict]:
    rows = []
    for row in claim_rows:
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
    return rows


def build_analysis(claim_rows: list[dict], label_rows: list[dict]) -> tuple[dict, str]:
    labels_by_claim = {row["claim_key"]: row for row in label_rows}
    merged = []
    for row in claim_rows:
        merged_row = dict(row)
        merged_row.update(labels_by_claim[row["claim_key"]])
        merged.append(merged_row)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in merged:
        by_label[row["initial_label"]].append(row)

    summary = {
        "total_claims": len(merged),
        "label_counts": {label: len(rows) for label, rows in sorted(by_label.items())},
        "by_label": {},
    }

    lines = ["# Audit Subset Label Analysis", "", f"Total claims: {len(merged)}", ""]
    for label, rows in sorted(by_label.items()):
        mean_disagreement = sum(int(r.get("claim_disagreement_count") or 0) for r in rows) / len(rows)
        mean_support = sum(int(r.get("cross_model_support_n") or 0) for r in rows) / len(rows)
        mean_risk_flags = sum(int(r.get("risk_flag_count") or 0) for r in rows) / len(rows)
        root_causes = Counter(r.get("paper_root_cause") or "none" for r in rows)
        papers = sorted({r["paper_id"] for r in rows})
        summary["by_label"][label] = {
            "n_claims": len(rows),
            "mean_claim_disagreement_count": round(mean_disagreement, 3),
            "mean_cross_model_support_n": round(mean_support, 3),
            "mean_risk_flag_count": round(mean_risk_flags, 3),
            "paper_root_causes": dict(root_causes),
            "papers": papers,
        }
        lines.extend(
            [
                f"## {label}",
                f"- Claims: {len(rows)}",
                f"- Mean disagreement count: {mean_disagreement:.3f}",
                f"- Mean cross-model support: {mean_support:.3f}",
                f"- Mean risk-flag count: {mean_risk_flags:.3f}",
                f"- Paper root causes: {json.dumps(dict(root_causes), sort_keys=True)}",
                f"- Papers: {', '.join(papers)}",
                "",
            ]
        )
    return summary, "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-list", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="audit_subset")
    args = parser.parse_args()

    global ACTIVE_PAPERS
    ACTIVE_PAPERS = load_paper_list(args.paper_list)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paper_rows = build_paper_rows()
    claim_rows = build_claim_rows(paper_rows)
    label_rows = build_labels(claim_rows)
    analysis_json, analysis_md = build_analysis(claim_rows, label_rows)

    paper_csv = OUTPUT_DIR / f"{args.prefix}_paper_features_2026-03-27.csv"
    claim_csv = OUTPUT_DIR / f"{args.prefix}_claim_features_2026-03-27.csv"
    label_csv = OUTPUT_DIR / f"{args.prefix}_claim_labels_2026-03-27.csv"
    analysis_json_path = OUTPUT_DIR / f"{args.prefix}_label_analysis_2026-03-27.json"
    analysis_md_path = OUTPUT_DIR / f"{args.prefix}_label_analysis_2026-03-27.md"

    write_csv(paper_csv, paper_rows)
    write_csv(claim_csv, claim_rows)
    write_csv(label_csv, label_rows)

    with analysis_json_path.open("w", encoding="utf-8") as f:
        json.dump(analysis_json, f, indent=2)
    with analysis_md_path.open("w", encoding="utf-8") as f:
        f.write(analysis_md)

    print(f"Wrote {len(paper_rows)} paper rows -> {paper_csv}")
    print(f"Wrote {len(claim_rows)} claim rows -> {claim_csv}")
    print(f"Wrote {len(label_rows)} label rows -> {label_csv}")
    print(analysis_json_path)
    print(analysis_md_path)


if __name__ == "__main__":
    main()
