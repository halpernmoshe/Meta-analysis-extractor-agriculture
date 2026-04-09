#!/usr/bin/env python3
"""
Normalize notill_tillage effector classes for already-kept rows.

Inputs:
- codex/outputs/effector_review_batches/notill_tillage/*.json

Outputs:
- codex/outputs/effector_labels/notill_tillage/labels.jsonl
- codex/outputs/effector_labels/notill_tillage/summary.json
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "effector_review_batches" / "notill_tillage"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "effector_labels" / "notill_tillage"


def normalize_crop(text: str) -> str | None:
    text = text.lower()
    hits = []
    patterns = [
        ("wheat", ["wheat", "triticum"]),
        ("maize", ["maize", "corn", "zea mays"]),
        ("rice", ["rice", "oryza"]),
        ("soybean", ["soybean", "soya", "glycine max"]),
        ("cotton", ["cotton", "gossypium"]),
        ("barley", ["barley", "hordeum"]),
        ("sorghum", ["sorghum"]),
        ("millet", ["millet", "pennisetum", "proso millet"]),
        ("legume", ["chickpea", "pea", "bean", "pigeonpea", "cowpea", "mung bean", "lentil", "vigna"]),
        ("oilseed_rape", ["oilseed rape", "canola", "rapeseed"]),
    ]
    for label, pats in patterns:
        if any(p in text for p in pats):
            hits.append(label)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "mixed"
    return "other"


def normalize_setting(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["greenhouse", "growth chamber"]):
        if any(k in t for k in ["field", "on-farm"]):
            return "mixed"
        return "greenhouse"
    if any(k in t for k in ["pot", "lysimeter"]):
        if "field" in t:
            return "mixed"
        return "pot"
    if "field" in t or "on-farm" in t:
        return "field"
    return "field"


def normalize_climate(text: str) -> str:
    t = text.lower()
    if "boreal" in t:
        return "boreal"
    if "semi-arid" in t or "semiarid" in t:
        return "semi_arid"
    if re.search(r"\barid\b", t):
        return "arid"
    if "mediterranean" in t:
        return "mediterranean"
    if "tropical" in t or "tropical savanna" in t or "tropical highland" in t:
        return "tropical"
    if "subtropical" in t or "sub-tropical" in t or "humid sub-tropical" in t or "subtropical humid" in t:
        return "subtropical"
    if "temperate" in t or "pannonian" in t or "sub-temperate" in t or "subtemperate" in t:
        return "temperate"
    return "unknown"


def normalize_soil(text: str) -> str | None:
    t = text.lower().strip()
    if not t:
        return None
    order_patterns = [
        "vertisol",
        "lixisol",
        "alfisol",
        "oxisol",
        "nitosol",
        "ferralsol",
        "inceptisol",
        "fluvaquent",
        "chernozem",
        "haploxeralf",
        "luvisol",
        "mollisol",
        "ultisol",
        "aridisol",
        "entisol",
        "spodosol",
        "histosol",
        "andisol",
        "oxisol",
        "ferralsol",
    ]
    for pat in order_patterns:
        if pat in t:
            return pat
    texture_patterns = [
        "clay loam",
        "sandy clay loam",
        "silty clay loam",
        "sandy loam",
        "silty loam",
        "silt loam",
        "sandy clay",
        "silty clay",
        "clay",
        "loam",
        "sand",
        "silt",
    ]
    for pat in texture_patterns:
        if pat in t:
            return pat
    if "alluvial" in t:
        return "alluvial"
    return None


def normalize_management(text: str) -> str:
    t = text.lower()
    has_residue = any(k in t for k in ["residue", "straw", "mulch", "trash", "cover crop", "retained"])
    has_rotation = any(k in t for k in ["rotation", "rotational", "cropping system", "preceding crop", "crop rotation"])
    if has_residue and has_rotation:
        return "residue_rotation"
    if has_residue:
        return "residue_only"
    if has_rotation:
        return "rotation_only"
    return "standard"


def normalize_estimand_context(climate: str, crop: str, management: str, setting: str, row_text: str) -> str:
    t = row_text.lower()
    if setting in {"pot", "greenhouse", "mixed"}:
        return "misaligned"
    if any(k in t for k in ["straw yield", "biological yield", "forage", "quality trait"]):
        return "misaligned"
    core_crop = crop in {"wheat", "maize", "rice", "soybean", "cotton", "barley", "sorghum", "oilseed_rape", "millet", "legume"}
    if climate == "temperate" and core_crop and management in {"standard", "residue_rotation"}:
        return "benchmark_aligned"
    if core_crop and climate in {"subtropical", "semi_arid", "arid", "tropical", "mediterranean", "boreal"}:
        return "partially_aligned"
    if core_crop:
        return "partially_aligned"
    return "unknown"


def collect_rows() -> list[dict]:
    rows = []
    for p in sorted(INPUT_ROOT.glob("batch_*.json")):
        obj = json.loads(p.read_text(encoding="utf-8"))
        rows.extend(obj["rows"])
    return rows


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    label_path = OUTPUT_ROOT / "labels.jsonl"
    summary_path = OUTPUT_ROOT / "summary.json"

    counters = {
        "crop": Counter(),
        "setting": Counter(),
        "climate": Counter(),
        "soil": Counter(),
        "management": Counter(),
        "estimand_context": Counter(),
    }

    with label_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            rid = row["row_id"]
            text_blob = " ".join(
                [
                    str(row.get("paper_id", "")),
                    str(row.get("title", "")),
                    str(row.get("outcome", "")),
                    str(row.get("outcome_unit", "")),
                    str(row.get("treatment_description", "")),
                    str(row.get("control_description", "")),
                    str(row.get("notes", "")),
                    " ".join(f"{k}:{v}" for k, v in row.get("moderators", {}).items()),
                ]
            )

            crop = normalize_crop(
                " ".join(
                    [
                        str(row.get("outcome", "")),
                        str(row.get("treatment_description", "")),
                        str(row.get("control_description", "")),
                        str(row.get("moderators", {}).get("mod_crop_species", "")),
                    ]
                )
            )
            setting = normalize_setting(text_blob)
            climate_raw = " ".join(
                [
                    str(row.get("moderators", {}).get("mod_climate", "")),
                    str(row.get("moderators", {}).get("mod_koppen_class", "")),
                    str(row.get("moderators", {}).get("mod_climate_character", "")),
                    str(row.get("moderators", {}).get("mod_region", "")),
                    str(row.get("moderators", {}).get("mod_country", "")),
                ]
            )
            climate = normalize_climate(climate_raw)
            soil_raw = " ".join(
                [
                    str(row.get("moderators", {}).get("mod_soil_type", "")),
                    str(row.get("moderators", {}).get("mod_soil_texture_class", "")),
                    str(row.get("moderators", {}).get("mod_soil_class", "")),
                    str(row.get("moderators", {}).get("mod_soil_texture", "")),
                    str(row.get("moderators", {}).get("mod_soil_taxonomy", "")),
                ]
            )
            soil = normalize_soil(soil_raw)
            management = normalize_management(
                " ".join(
                    [
                        str(row.get("treatment_description", "")),
                        str(row.get("control_description", "")),
                        str(row.get("moderators", {}).get("mod_residue_management", "")),
                        str(row.get("moderators", {}).get("mod_residue_management_treatment", "")),
                        str(row.get("moderators", {}).get("mod_crop_rotation", "")),
                        str(row.get("moderators", {}).get("mod_rotation", "")),
                    ]
                )
            )
            estimand_context = normalize_estimand_context(climate, crop or "unknown", management, setting, text_blob)
            notes = {
                "benchmark_aligned": "field grain-yield no-till comparison in benchmark-relevant climate context.",
                "partially_aligned": "field no-till comparison with valid crop-yield outcome but non-temperate or mixed effector context.",
                "misaligned": "effector context is outside the benchmark-style comparison.",
                "unknown": "insufficient effector detail to classify confidently.",
            }[estimand_context]

            record = {
                "row_id": rid,
                "normalized_crop_class": crop,
                "normalized_study_setting": setting,
                "normalized_climate_class": climate,
                "normalized_soil_class": soil,
                "normalized_management_class": management,
                "normalized_estimand_context": estimand_context,
                "normalization_notes": notes,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            counters["crop"][crop or "null"] += 1
            counters["setting"][setting] += 1
            counters["climate"][climate] += 1
            counters["soil"][soil or "null"] += 1
            counters["management"][management] += 1
            counters["estimand_context"][estimand_context] += 1

    summary = {
        "rows_labeled": len(rows),
        "counts": {key: dict(val.most_common()) for key, val in counters.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
