#!/usr/bin/env python3
"""
Build smaller validated-row review batches for Codex adjudication.

This packages `summary_validated.csv` rows for selected topics into JSON files
under `codex/outputs/validated_review_batches/<topic>/batch_XXX.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "validated_review_batches"


def topic_brief(config: dict) -> dict:
    pico = config.get("pico", {})
    benchmark = config.get("benchmark", {})
    return {
        "review_id": config.get("review_id"),
        "title": config.get("title"),
        "research_question": config.get("research_question"),
        "population_description": pico.get("population", {}).get("description"),
        "intervention_description": pico.get("intervention", {}).get("description"),
        "intervention_terms": pico.get("intervention", {}).get("search_terms", []),
        "comparator_description": pico.get("comparator", {}).get("description"),
        "comparator_terms": pico.get("comparator", {}).get("search_terms", []),
        "outcome_description": pico.get("outcome", {}).get("primary", {}).get("description"),
        "outcome_terms": pico.get("outcome", {}).get("primary", {}).get("search_terms", []),
        "expected_direction": config.get("expected_direction"),
        "tc_confusion_warnings": config.get("tc_confusion_warnings", []),
        "benchmark_source": benchmark.get("source"),
        "benchmark_notes": benchmark.get("published_pooled_effect", {}).get("notes"),
    }


def row_to_payload(topic: str, idx: int, row: pd.Series) -> dict:
    keep_cols = [
        "paper_id",
        "title",
        "year",
        "outcome",
        "outcome_unit",
        "treatment_mean",
        "control_mean",
        "effect_pct",
        "treatment_n",
        "control_n",
        "variance_type",
        "variance_value",
        "sd_treatment",
        "sd_control",
        "se_treatment",
        "se_control",
        "treatment_description",
        "control_description",
        "confidence",
        "source_type",
        "notes",
    ]
    payload = {"row_id": f"{topic}::{row.get('paper_id','unknown')}::{idx}"}
    for col in keep_cols:
        if col in row.index:
            val = row[col]
            payload[col] = None if pd.isna(val) else val
    payload["moderators"] = {
        col: row[col]
        for col in row.index
        if col.startswith("mod_") and pd.notna(row[col]) and str(row[col]).strip()
    }
    return payload


def build_batches(topic: str, batch_size: int = 80) -> None:
    topic_dir = OUTPUT_ROOT / topic
    topic_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((ROOT / topic / "config.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / topic / "4_extract" / "summary_validated.csv")

    brief = topic_brief(config)
    payloads = [row_to_payload(topic, idx, row) for idx, row in df.iterrows()]

    batches = [payloads[i : i + batch_size] for i in range(0, len(payloads), batch_size)]
    index = []
    for i, batch in enumerate(batches, start=1):
        batch_obj = {
            "topic": topic,
            "topic_brief": brief,
            "rows": batch,
        }
        batch_path = topic_dir / f"batch_{i:03d}.json"
        batch_path.write_text(json.dumps(batch_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({"batch": i, "rows": len(batch), "path": str(batch_path)})

    (topic_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    for topic in ["organic_yield_gap", "notill_tillage"]:
        build_batches(topic)


if __name__ == "__main__":
    main()
