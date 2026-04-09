#!/usr/bin/env python3
"""
Universal post-processing diagnostics on current validated extraction outputs.

This script does not modify main pipeline files. It annotates rows with
universal canonical classes and produces per-topic diagnostics under codex.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "universal_postprocess_diagnostics"

TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]


def topic_config(topic: str) -> dict:
    return json.loads((ROOT / topic / "config.json").read_text(encoding="utf-8"))


def row_text(row: pd.Series) -> str:
    cols = [
        "outcome",
        "outcome_unit",
        "treatment_description",
        "control_description",
        "title",
        "notes",
    ]
    return " | ".join(str(row.get(c, "")) for c in cols).lower()


def canonical_outcome_class(text: str) -> str:
    if "land equivalent ratio" in text or "ler" in text:
        return "system_productivity"
    if any(k in text for k in ["protein", "quality", "concentration", "heme", "hme", "hectolitre", "ratio", "npq", "quantum"]):
        return "quality_trait"
    if any(k in text for k in ["root biomass", "root dry weight", "belowground"]):
        return "belowground_biomass"
    if any(k in text for k in ["shoot dry weight", "shoot biomass", "total biomass", "dry weight", "fresh weight", "biological yield"]):
        return "biomass"
    if any(k in text for k in ["grain yield", "fruit yield", "seed yield", "tuber yield", "yield per plant", "crop yield", "final grain yield"]):
        return "harvest_yield"
    if "yield" in text:
        return "generic_yield"
    return "other_unknown"


def canonical_study_setting(text: str, mods: dict[str, str]) -> str:
    merged = text + " | " + " | ".join(f"{k}={v}" for k, v in mods.items())
    merged = merged.lower()
    if any(k in merged for k in ["greenhouse", "growth chamber"]):
        return "greenhouse"
    if any(k in merged for k in ["pot", "g/pot", "mg/pot", "w/w"]):
        return "pot"
    if any(k in merged for k in ["field", "on-farm", "long-term trial", "field trial"]):
        return "field"
    return "unknown"


def canonical_estimand_class(text: str) -> str:
    lower = text.lower()
    if "land equivalent ratio" in lower or "ler" in lower:
        return "system_productivity"
    if "intercropped maize" in lower or "intercropped soybean" in lower or "grain yield of" in lower:
        return "component_yield"
    if any(k in lower for k in ["grain yield", "fruit yield", "seed yield", "tuber yield", "crop yield"]):
        return "direct_yield"
    if any(k in lower for k in ["dry weight", "biomass"]):
        return "biomass_proxy"
    if any(k in lower for k in ["quality", "concentration", "protein", "ratio"]):
        return "quality_proxy"
    return "unknown"


def canonical_intervention_class(text: str) -> str:
    lower = text.lower()
    if "organic" in lower:
        return "organic_vs_conventional"
    if any(k in lower for k in ["no-till", "no till", "zero till", "zero-till", "direct sow", "direct seed", "direct drill"]):
        return "strict_notill"
    if any(k in lower for k in ["reduced tillage", "minimum tillage", "strip till", "strip-till", "conservation agriculture", "conservation tillage"]):
        return "broad_conservation_tillage"
    if "biochar" in lower:
        return "biochar_vs_no_biochar"
    if any(k in lower for k in ["amf", "mycorrh", "glomus", "rhizophagus"]):
        return "amf_vs_control"
    if any(k in lower for k in ["rotation", "following", "preceding crop", "continuous", "monoculture"]):
        return "rotation_vs_continuous"
    if any(k in lower for k in ["intercrop", "relay strip", "sole crop", "monoculture maize", "monoculture soybean"]):
        return "intercrop_vs_sole"
    return "other_unknown"


def benchmark_alignment(topic: str, outcome_class: str, setting: str, estimand: str, intervention: str) -> str:
    if topic == "organic_yield_gap":
        if intervention != "organic_vs_conventional":
            return "misaligned"
        if outcome_class in {"harvest_yield", "generic_yield"}:
            return "benchmark_aligned"
        if outcome_class in {"biomass"}:
            return "partially_aligned"
        return "misaligned"
    if topic == "notill_tillage":
        if intervention not in {"strict_notill", "broad_conservation_tillage"}:
            return "misaligned"
        if intervention == "broad_conservation_tillage":
            return "partially_aligned"
        if outcome_class in {"harvest_yield", "generic_yield"}:
            return "benchmark_aligned"
        return "misaligned"
    if topic == "mycorrhiza_yield":
        if intervention != "amf_vs_control":
            return "misaligned"
        if outcome_class in {"harvest_yield", "generic_yield", "biomass"}:
            return "partially_aligned" if setting != "field" else "benchmark_aligned"
        return "misaligned"
    if topic == "legume_rotation":
        if intervention != "rotation_vs_continuous":
            return "misaligned"
        if outcome_class in {"harvest_yield", "generic_yield"}:
            return "benchmark_aligned"
        if outcome_class == "biomass":
            return "partially_aligned"
        return "misaligned"
    if topic == "biochar_crop_yield":
        if intervention != "biochar_vs_no_biochar":
            return "misaligned"
        if setting == "field" and outcome_class in {"harvest_yield", "generic_yield"}:
            return "benchmark_aligned"
        if outcome_class in {"harvest_yield", "generic_yield", "biomass"}:
            return "partially_aligned"
        return "misaligned"
    if topic == "intercropping_yield":
        if intervention != "intercrop_vs_sole":
            return "misaligned"
        if estimand == "system_productivity":
            return "benchmark_aligned"
        if estimand == "component_yield":
            return "misaligned"
        return "partially_aligned"
    return "unknown"


def variance_status(row: pd.Series) -> str:
    if pd.notna(row.get("sd_treatment")) or pd.notna(row.get("se_treatment")) or pd.notna(row.get("variance_value")):
        return "present"
    return "missing"


def likely_duplicate_key(row: pd.Series) -> str:
    parts = [
        str(row.get("paper_id", "")).strip().lower(),
        str(row.get("outcome", "")).strip().lower(),
        str(row.get("treatment_description", "")).strip().lower(),
        str(row.get("control_description", "")).strip().lower(),
        str(row.get("treatment_mean", "")).strip(),
        str(row.get("control_mean", "")).strip(),
    ]
    return "||".join(parts)


def annotate_topic(topic: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / topic / "4_extract" / "summary_validated.csv")
    annotations = []
    for idx, row in df.iterrows():
        mods = {
            c: row[c]
            for c in df.columns
            if c.startswith("mod_") and pd.notna(row[c]) and str(row[c]).strip()
        }
        text = row_text(row)
        outcome_class = canonical_outcome_class(text)
        setting = canonical_study_setting(text, mods)
        estimand = canonical_estimand_class(text)
        intervention = canonical_intervention_class(text)
        alignment = benchmark_alignment(topic, outcome_class, setting, estimand, intervention)
        annotations.append(
            {
                "row_id": f"{topic}::{row.get('paper_id', 'unknown')}::{idx}",
                "canonical_outcome_class": outcome_class,
                "canonical_study_setting": setting,
                "canonical_estimand_class": estimand,
                "canonical_intervention_class": intervention,
                "benchmark_alignment": alignment,
                "variance_status": variance_status(row),
                "duplicate_key": likely_duplicate_key(row),
            }
        )
    ann = pd.DataFrame(annotations)
    return pd.concat([df.reset_index(drop=True), ann], axis=1)


def diagnostics_for_topic(topic: str, df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated(subset=["duplicate_key"]).sum())
    return {
        "topic": topic,
        "n_obs": int(len(df)),
        "n_papers": int(df["paper_id"].nunique()),
        "outcome_classes": df["canonical_outcome_class"].value_counts(dropna=False).to_dict(),
        "study_settings": df["canonical_study_setting"].value_counts(dropna=False).to_dict(),
        "estimand_classes": df["canonical_estimand_class"].value_counts(dropna=False).to_dict(),
        "intervention_classes": df["canonical_intervention_class"].value_counts(dropna=False).to_dict(),
        "benchmark_alignment": df["benchmark_alignment"].value_counts(dropna=False).to_dict(),
        "variance_status": df["variance_status"].value_counts(dropna=False).to_dict(),
        "duplicate_rows": dup_count,
        "low_confidence_rows": int((df["confidence"].astype(str).str.lower() == "low").sum()) if "confidence" in df.columns else 0,
        "figure_rows": int((df["source_type"].astype(str).str.lower() == "figure").sum()) if "source_type" in df.columns else 0,
    }


def write_markdown_report(summaries: list[dict]) -> None:
    lines = ["# Universal Post-Processing Diagnostics", ""]
    lines.append("These diagnostics annotate validated rows with universal canonical classes and show which gaps can be closed on current local data.")
    lines.append("")
    for item in summaries:
        lines.append(f"## {item['topic']}")
        lines.append(f"- Rows: {item['n_obs']} across {item['n_papers']} papers")
        lines.append(f"- Benchmark-aligned rows: {item['benchmark_alignment'].get('benchmark_aligned', 0)}")
        lines.append(f"- Partially aligned rows: {item['benchmark_alignment'].get('partially_aligned', 0)}")
        lines.append(f"- Misaligned rows: {item['benchmark_alignment'].get('misaligned', 0)}")
        lines.append(f"- Missing variance rows: {item['variance_status'].get('missing', 0)}")
        lines.append(f"- Figure rows: {item['figure_rows']}")
        lines.append(f"- Low-confidence rows: {item['low_confidence_rows']}")
        lines.append(f"- Duplicate rows by strict key: {item['duplicate_rows']}")
        lines.append(f"- Outcome classes: {json.dumps(item['outcome_classes'], ensure_ascii=False)}")
        lines.append(f"- Study settings: {json.dumps(item['study_settings'], ensure_ascii=False)}")
        lines.append("")
    (OUTPUT_ROOT / "UNIVERSAL_POSTPROCESS_DIAGNOSTICS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for topic in TOPICS:
        annotated = annotate_topic(topic)
        annotated.to_csv(OUTPUT_ROOT / f"{topic}_annotated.csv", index=False)
        summary = diagnostics_for_topic(topic, annotated)
        summaries.append(summary)
        (OUTPUT_ROOT / f"{topic}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    (OUTPUT_ROOT / "all_topics_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_markdown_report(summaries)


if __name__ == "__main__":
    main()
