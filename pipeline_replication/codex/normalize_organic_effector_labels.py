#!/usr/bin/env python3
"""
Normalize effector classes for organic_yield_gap kept rows.

This is a universal-style, config-driven post-processing step for the organic
topic only. It writes outputs under codex/outputs/effector_labels/organic_yield_gap.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "effector_review_batches" / "organic_yield_gap"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "effector_labels" / "organic_yield_gap"


def load_topic_brief() -> dict:
    config = json.loads((ROOT / "organic_yield_gap" / "config.json").read_text(encoding="utf-8"))
    pico = config.get("pico", {})
    benchmark = config.get("benchmark", {})
    return {
        "review_id": config.get("review_id"),
        "title": config.get("title"),
        "research_question": config.get("research_question"),
        "population_description": pico.get("population", {}).get("description"),
        "intervention_description": pico.get("intervention", {}).get("description"),
        "comparator_description": pico.get("comparator", {}).get("description"),
        "outcome_description": pico.get("outcome", {}).get("primary", {}).get("description"),
        "important_moderators": config.get("important_moderators", []),
        "benchmark_source": benchmark.get("source"),
        "benchmark_notes": benchmark.get("published_pooled_effect", {}).get("notes"),
    }


def iter_rows() -> list[dict]:
    rows = []
    for batch_path in sorted(INPUT_ROOT.glob("batch_*.json")):
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


def first_nonempty(*values):
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return None


def normalize_crop_class(row: dict) -> str | None:
    moderators = row.get("moderators", {})
    raw = first_nonempty(
        moderators.get("mod_crop_type"),
        moderators.get("mod_crop_species"),
        row.get("outcome"),
        row.get("title"),
    )
    if raw is None:
        return None
    text = raw.lower()

    patterns = [
        ("grain_cereal", ["cereal", "grain", "wheat", "maize", "corn", "rice", "barley", "oat", "sorghum", "millet"]),
        ("legume", ["legume", "bean", "pea", "chickpea", "lentil", "soybean", "cowpea"]),
        ("vegetable", ["vegetable", "okra", "lettuce", "cabbage", "tomato", "pepper", "eggplant", "onion", "bean?"]),
        ("root_tuber", ["root/tuber", "root tuber", "potato", "cassava", "tuber", "yam", "sweet potato"]),
        ("oilseed", ["oilseed", "sunflower", "rapeseed", "canola", "mustard"]),
        ("fiber", ["fiber", "fibre", "cotton", "hemp", "flax"]),
        ("tree_crop", ["tree crop", "cacao", "coffee", "banana", "citrus", "apple", "grape"]),
        ("mixed_grain_dairy", ["mixed grain + dairy", "grain + dairy", "milk", "dairy"]),
        ("mixed_grain", ["mixed grain crops", "cereals, oilseeds, grain legumes"]),
        ("mixed_rotation", ["rotation (", "mixed rotation"]),
    ]
    for label, pats in patterns:
        if any(p in text for p in pats):
            return label
    if text in {"mixed", "other", "unknown"}:
        return text
    return re.sub(r"\s+", "_", text.strip())


def normalize_study_setting(row: dict) -> str:
    moderators = row.get("moderators", {})
    parts = [
        row.get("title"),
        row.get("outcome"),
        row.get("treatment_description"),
        row.get("control_description"),
        moderators.get("mod_experiment_type"),
        moderators.get("mod_cropping_system"),
    ]
    text = " ".join([str(x) for x in parts if x is not None]).lower()

    has_pot = any(k in text for k in ["pot", "greenhouse", "growth chamber", "screenhouse"])
    has_field = any(k in text for k in ["field trial", "field experiment", "field", "on-farm", "farm", "large plot", "long-term trial", "comparison"])

    if has_pot and has_field:
        return "mixed"
    if has_pot:
        return "pot" if "greenhouse" not in text and "screenhouse" not in text else "greenhouse"
    if has_field:
        return "field"
    return "unknown"


def normalize_climate_class(row: dict) -> str:
    moderators = row.get("moderators", {})
    raw = first_nonempty(moderators.get("mod_climate_zone"), moderators.get("mod_region"))
    if raw is None:
        return "unknown"
    text = raw.lower()

    if any(k in text for k in ["subarctic", "boreal"]):
        return "boreal"
    if "mediterranean" in text:
        return "mediterranean"
    if "semi-arid" in text or "semi arid" in text:
        return "semi_arid"
    if "arid" in text:
        return "arid"
    if "temperate" in text:
        return "temperate"
    if "subtropical" in text:
        return "subtropical"
    if "tropical" in text:
        return "tropical"
    return "unknown"


def normalize_soil_class(row: dict) -> str | None:
    moderators = row.get("moderators", {})
    raw = first_nonempty(moderators.get("mod_soil_type"))
    if raw is None:
        return None
    text = re.sub(r"\(.*?\)", "", raw).strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(",; ")
    return text or None


def normalize_management_class(row: dict) -> str:
    text = " ".join(
        [
            str(row.get("treatment_description", "")),
            str(row.get("control_description", "")),
            str(row.get("notes", "")),
            str(row.get("moderators", {}).get("mod_cropping_system", "")),
            str(row.get("moderators", {}).get("mod_rotation", "")),
            str(row.get("moderators", {}).get("mod_organic_inputs", "")),
            str(row.get("moderators", {}).get("mod_organic_amendment", "")),
            str(row.get("moderators", {}).get("mod_organic_preceding_crop", "")),
            str(row.get("moderators", {}).get("mod_conventional_preceding_crop", "")),
        ]
    ).lower()

    residue_terms = ["residue", "straw", "mulch", "green manure", "cover crop", "compost", "manure", "vermicompost"]
    rotation_terms = ["rotation", "preceding crop", "preceeding crop", "preceded", "cycle", "sequence", "intercropping"]

    has_residue = any(term in text for term in residue_terms)
    has_rotation = any(term in text for term in rotation_terms)

    if has_residue and has_rotation:
        return "residue_rotation"
    if has_residue:
        return "residue_only"
    if has_rotation:
        return "rotation_only"
    return "standard"


def normalize_estimand_context(row: dict, crop_class: str | None, study_setting: str, climate_class: str) -> str:
    outcome = str(row.get("outcome", "")).lower()
    text = " ".join(
        [
            outcome,
            str(row.get("outcome_unit", "")),
            str(row.get("treatment_description", "")),
            str(row.get("control_description", "")),
            str(row.get("notes", "")),
        ]
    ).lower()

    misaligned_terms = [
        "human metabolizable energy",
        "hme",
        "ratio",
        "quality",
        "protein",
        "concentration",
        "energy",
        "profitability",
        "sustainability",
        "equivalent grain yield",
        "total equivalent",
        "yield equivalent",
    ]
    if any(term in text for term in misaligned_terms):
        return "misaligned"
    if study_setting in {"pot", "greenhouse"}:
        return "misaligned"
    if crop_class == "grain_cereal" and study_setting == "field":
        return "benchmark_aligned"
    if study_setting == "field":
        return "partially_aligned"
    if climate_class == "boreal":
        return "partially_aligned"
    return "unknown"


def normalize_row(row: dict) -> dict:
    crop_class = normalize_crop_class(row)
    study_setting = normalize_study_setting(row)
    climate_class = normalize_climate_class(row)
    soil_class = normalize_soil_class(row)
    management_class = normalize_management_class(row)
    estimand_context = normalize_estimand_context(row, crop_class, study_setting, climate_class)

    note_bits = []
    if crop_class:
        note_bits.append(f"crop={crop_class}")
    if study_setting != "unknown":
        note_bits.append(f"setting={study_setting}")
    if climate_class != "unknown":
        note_bits.append(f"climate={climate_class}")
    if management_class != "unknown":
        note_bits.append(f"management={management_class}")
    if estimand_context:
        note_bits.append(f"context={estimand_context}")

    return {
        "row_id": row["row_id"],
        "normalized_crop_class": crop_class,
        "normalized_study_setting": study_setting,
        "normalized_climate_class": climate_class,
        "normalized_soil_class": soil_class,
        "normalized_management_class": management_class,
        "normalized_estimand_context": estimand_context,
        "normalization_notes": "; ".join(note_bits[:3]) if note_bits else "insufficient metadata",
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = iter_rows()
    labels = [normalize_row(row) for row in rows]

    labels_path = OUTPUT_ROOT / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as handle:
        for label in labels:
            handle.write(json.dumps(label, ensure_ascii=False) + "\n")

    summary = {
        "total_rows": len(labels),
        "crop_class_counts": dict(Counter(x["normalized_crop_class"] or "null" for x in labels)),
        "study_setting_counts": dict(Counter(x["normalized_study_setting"] for x in labels)),
        "climate_class_counts": dict(Counter(x["normalized_climate_class"] for x in labels)),
        "management_class_counts": dict(Counter(x["normalized_management_class"] for x in labels)),
        "estimand_context_counts": dict(Counter(x["normalized_estimand_context"] for x in labels)),
        "null_soil_class_count": sum(1 for x in labels if x["normalized_soil_class"] is None),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
