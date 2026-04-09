#!/usr/bin/env python3
"""
Universal effector normalization for all pipeline replication topics.

Reads kept rows from codex/outputs/codex_decisions/{topic}/strict_kept_rows.csv
and writes normalized effector labels to codex/outputs/effector_labels/{topic}/.

Universal normalization of:
- crop class
- study setting (field/greenhouse/pot)
- climate class
- soil class
- management class
- estimand context (benchmark-aligned/partially aligned/misaligned)
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


CODEX_ROOT = Path(__file__).resolve().parent
ROOT = CODEX_ROOT.parent
DECISIONS_ROOT = CODEX_ROOT / "outputs" / "codex_decisions"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "effector_labels"

TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]


# ── Crop class normalization ───────────────────────────────────────────────

CROP_PATTERNS = [
    ("grain_cereal", [
        "wheat", "maize", "corn", "rice", "barley", "oat", "sorghum",
        "millet", "rye", "triticale",
    ]),
    ("legume", [
        "legume", "bean", "pea", "chickpea", "lentil", "soybean", "cowpea",
        "pigeon pea", "mung bean", "groundnut", "faba bean", "lupin",
        "vetch", "clover", "alfalfa", "medic",
    ]),
    ("vegetable", [
        "vegetable", "okra", "lettuce", "cabbage", "tomato", "pepper",
        "eggplant", "onion", "garlic", "carrot", "spinach", "broccoli",
        "cucumber", "squash", "zucchini", "pumpkin",
    ]),
    ("root_tuber", [
        "potato", "cassava", "tuber", "yam", "sweet potato", "beet",
    ]),
    ("oilseed", [
        "oilseed", "sunflower", "rapeseed", "canola", "mustard", "sesame",
        "safflower",
    ]),
    ("fiber", [
        "fiber", "fibre", "cotton", "hemp", "flax", "jute",
    ]),
    ("tree_crop", [
        "cacao", "coffee", "banana", "citrus", "apple", "grape", "olive",
        "mango", "oil palm", "coconut",
    ]),
    ("grass_forage", [
        "grass", "forage", "pasture", "hay", "ryegrass", "fescue",
        "bermudagrass", "clover",
    ]),
]


def normalize_crop_class(row: dict) -> str | None:
    """Classify crop from outcome, title, treatment description, moderators."""
    parts = [
        str(row.get("outcome", "")),
        str(row.get("title", "")),
        str(row.get("treatment_description", "")),
        str(row.get("notes", "")),
    ]
    # Check moderators if present
    mods = row.get("moderators", {})
    if isinstance(mods, dict):
        for key in ["mod_crop_type", "mod_crop_species", "mod_crop"]:
            if key in mods and mods[key]:
                parts.append(str(mods[key]))

    text = " ".join(parts).lower()

    for label, keywords in CROP_PATTERNS:
        if any(kw in text for kw in keywords):
            return label
    return None


# ── Study setting normalization ────────────────────────────────────────────

def normalize_study_setting(row: dict) -> str:
    """Classify as field/greenhouse/pot/mixed/unknown."""
    parts = [
        str(row.get("title", "")),
        str(row.get("outcome", "")),
        str(row.get("treatment_description", "")),
        str(row.get("control_description", "")),
        str(row.get("notes", "")),
    ]
    mods = row.get("moderators", {})
    if isinstance(mods, dict):
        for key in ["mod_experiment_type", "mod_study_type", "mod_cropping_system"]:
            if key in mods and mods[key]:
                parts.append(str(mods[key]))

    text = " ".join(parts).lower()

    pot_terms = ["pot", "greenhouse", "growth chamber", "screenhouse",
                 "controlled environment", "phytotron", "glasshouse"]
    field_terms = ["field", "farm", "station", "on-farm", "large plot",
                   "long-term trial", "irrigated plot", "rainfed"]

    has_pot = any(t in text for t in pot_terms)
    has_field = any(t in text for t in field_terms)

    if has_pot and has_field:
        return "mixed"
    if has_pot:
        if "greenhouse" in text or "glasshouse" in text or "screenhouse" in text:
            return "greenhouse"
        return "pot"
    if has_field:
        return "field"
    return "unknown"


# ── Climate class normalization ────────────────────────────────────────────

def normalize_climate_class(row: dict) -> str:
    """Classify climate zone from moderators and text."""
    mods = row.get("moderators", {})
    parts = []
    if isinstance(mods, dict):
        for key in ["mod_climate_zone", "mod_region", "mod_country"]:
            if key in mods and mods[key]:
                parts.append(str(mods[key]))
    parts.append(str(row.get("title", "")))
    parts.append(str(row.get("notes", "")))

    text = " ".join(parts).lower()

    if "boreal" in text or "subarctic" in text:
        return "boreal"
    if "mediterranean" in text:
        return "mediterranean"
    if "semi-arid" in text or "semi arid" in text or "semiarid" in text:
        return "semi_arid"
    if "arid" in text and "semi" not in text:
        return "arid"
    if "temperate" in text:
        return "temperate"
    if "subtropical" in text or "sub-tropical" in text:
        return "subtropical"
    if "tropical" in text:
        return "tropical"
    return "unknown"


# ── Soil class normalization ───────────────────────────────────────────────

def normalize_soil_class(row: dict) -> str | None:
    """Extract soil type from moderators."""
    mods = row.get("moderators", {})
    if not isinstance(mods, dict):
        return None
    raw = mods.get("mod_soil_type") or mods.get("mod_soil")
    if not raw or str(raw).strip().lower() in ("nan", "none", ""):
        return None
    text = re.sub(r"\(.*?\)", "", str(raw)).strip()
    text = re.sub(r"\s+", " ", text).rstrip(",; ")
    return text or None


# ── Management class normalization ─────────────────────────────────────────

def normalize_management_class(row: dict) -> str:
    """Classify management context."""
    parts = [
        str(row.get("treatment_description", "")),
        str(row.get("control_description", "")),
        str(row.get("notes", "")),
    ]
    mods = row.get("moderators", {})
    if isinstance(mods, dict):
        for key in ["mod_cropping_system", "mod_rotation", "mod_organic_inputs",
                     "mod_organic_amendment", "mod_n_rate", "mod_fertilizer"]:
            if key in mods and mods[key]:
                parts.append(str(mods[key]))

    text = " ".join(parts).lower()

    residue_terms = ["residue", "straw", "mulch", "green manure", "cover crop",
                     "compost", "manure", "vermicompost", "organic amendment"]
    rotation_terms = ["rotation", "preceding crop", "sequence", "cycle",
                      "intercropping", "relay"]

    has_residue = any(t in text for t in residue_terms)
    has_rotation = any(t in text for t in rotation_terms)

    if has_residue and has_rotation:
        return "residue_rotation"
    if has_residue:
        return "residue_only"
    if has_rotation:
        return "rotation_only"
    return "standard"


# ── Estimand context normalization ─────────────────────────────────────────

TOPIC_ESTIMAND_RULES = {
    "organic_yield_gap": {
        "benchmark_aligned_conditions": lambda row, crop, setting, climate: (
            setting == "field" and crop in ("grain_cereal", "oilseed", "legume", "vegetable", "root_tuber")
        ),
        "misaligned_terms": [
            "human metabolizable energy", "hme", "profitability",
            "sustainability", "equivalent grain yield",
        ],
    },
    "notill_tillage": {
        "benchmark_aligned_conditions": lambda row, crop, setting, climate: (
            setting == "field" and crop in ("grain_cereal", "oilseed", "legume")
        ),
        "misaligned_terms": [
            "straw yield", "biological yield", "stover",
        ],
    },
    "mycorrhiza_yield": {
        "benchmark_aligned_conditions": lambda row, crop, setting, climate: (
            setting == "field" or setting == "greenhouse"
        ),
        "misaligned_terms": [
            "colonization", "infection", "spore", "root length",
            "photosynthesis", "chlorophyll", "stomatal",
        ],
    },
    "legume_rotation": {
        "benchmark_aligned_conditions": lambda row, crop, setting, climate: (
            setting == "field" and crop in ("grain_cereal", "oilseed")
        ),
        "misaligned_terms": [
            "nodule", "nodulation", "rhizob", "root biomass",
            "nitrogen fixation rate",
        ],
    },
    "biochar_crop_yield": {
        "benchmark_aligned_conditions": lambda row, crop, setting, climate: (
            setting == "field"
        ),
        "misaligned_terms": [
            "root biomass", "root dry weight", "root length",
            "soil organic carbon", "soil respiration",
        ],
    },
    "intercropping_yield": {
        "benchmark_aligned_conditions": lambda row, crop, setting, climate: (
            "ler" in str(row.get("outcome", "")).lower() or
            "land equivalent ratio" in str(row.get("outcome", "")).lower()
        ),
        "misaligned_terms": [],
    },
}


def normalize_estimand_context(topic: str, row: dict, crop_class: str | None,
                                study_setting: str, climate_class: str) -> str:
    """Classify whether row is benchmark-aligned for the given topic."""
    rules = TOPIC_ESTIMAND_RULES.get(topic, {})
    text = " ".join([
        str(row.get("outcome", "")),
        str(row.get("outcome_unit", "")),
        str(row.get("treatment_description", "")),
        str(row.get("notes", "")),
    ]).lower()

    # Check misalignment first
    misaligned_terms = rules.get("misaligned_terms", [])
    if any(t in text for t in misaligned_terms):
        return "misaligned"

    # Check benchmark alignment
    cond_fn = rules.get("benchmark_aligned_conditions")
    if cond_fn and cond_fn(row, crop_class, study_setting, climate_class):
        return "benchmark_aligned"

    return "partially_aligned"


# ── Main ───────────────────────────────────────────────────────────────────

def normalize_row(topic: str, row: dict) -> dict:
    crop_class = normalize_crop_class(row)
    study_setting = normalize_study_setting(row)
    climate_class = normalize_climate_class(row)
    soil_class = normalize_soil_class(row)
    management_class = normalize_management_class(row)
    estimand_context = normalize_estimand_context(
        topic, row, crop_class, study_setting, climate_class
    )

    notes = []
    if crop_class:
        notes.append(f"crop={crop_class}")
    if study_setting != "unknown":
        notes.append(f"setting={study_setting}")
    if climate_class != "unknown":
        notes.append(f"climate={climate_class}")

    return {
        "row_id": row.get("row_id", ""),
        "normalized_crop_class": crop_class,
        "normalized_study_setting": study_setting,
        "normalized_climate_class": climate_class,
        "normalized_soil_class": soil_class,
        "normalized_management_class": management_class,
        "normalized_estimand_context": estimand_context,
        "normalization_notes": "; ".join(notes[:3]) if notes else "insufficient metadata",
    }


def process_topic(topic: str) -> dict:
    """Process a single topic's kept rows."""
    input_csv = DECISIONS_ROOT / topic / "strict_kept_rows.csv"
    out_dir = OUTPUT_ROOT / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        print(f"  [SKIP] No kept rows CSV for {topic}")
        return {"topic": topic, "error": "no_kept_rows_csv"}

    df = pd.read_csv(input_csv)
    rows = df.to_dict("records")

    labels = [normalize_row(topic, row) for row in rows]

    # Write JSONL
    labels_path = out_dir / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + "\n")

    # Summary stats
    summary = {
        "topic": topic,
        "total_rows": len(labels),
        "crop_class_counts": dict(Counter(
            x["normalized_crop_class"] or "null" for x in labels
        )),
        "study_setting_counts": dict(Counter(
            x["normalized_study_setting"] for x in labels
        )),
        "climate_class_counts": dict(Counter(
            x["normalized_climate_class"] for x in labels
        )),
        "management_class_counts": dict(Counter(
            x["normalized_management_class"] for x in labels
        )),
        "estimand_context_counts": dict(Counter(
            x["normalized_estimand_context"] for x in labels
        )),
        "null_soil_class_count": sum(
            1 for x in labels if x["normalized_soil_class"] is None
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    report = [
        f"# {topic} -- Effector Normalization",
        "",
        f"Total rows: {len(labels)}",
        "",
        "## Crop Class Distribution",
    ]
    for cls, cnt in sorted(summary["crop_class_counts"].items(), key=lambda x: -x[1]):
        report.append(f"- {cls}: {cnt}")
    report.extend(["", "## Study Setting Distribution"])
    for cls, cnt in sorted(summary["study_setting_counts"].items(), key=lambda x: -x[1]):
        report.append(f"- {cls}: {cnt}")
    report.extend(["", "## Climate Class Distribution"])
    for cls, cnt in sorted(summary["climate_class_counts"].items(), key=lambda x: -x[1]):
        report.append(f"- {cls}: {cnt}")
    report.extend(["", "## Estimand Context Distribution"])
    for cls, cnt in sorted(summary["estimand_context_counts"].items(), key=lambda x: -x[1]):
        report.append(f"- {cls}: {cnt}")

    (out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")

    return summary


def main():
    import sys
    topics = sys.argv[1:] if len(sys.argv) > 1 else TOPICS

    all_summaries = {}
    for topic in topics:
        print(f"\n{'='*50}")
        print(f"  Normalizing effectors: {topic}")
        print(f"{'='*50}")
        summary = process_topic(topic)
        all_summaries[topic] = summary
        if "error" not in summary:
            print(f"  -> {summary['total_rows']} rows labeled")
            print(f"  -> Crop classes: {summary['crop_class_counts']}")
            print(f"  -> Settings: {summary['study_setting_counts']}")
            print(f"  -> Estimand context: {summary['estimand_context_counts']}")

    # Combined summary
    combined_path = OUTPUT_ROOT / "universal_effector_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*50}")
    print("  ALL TOPICS COMPLETE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
