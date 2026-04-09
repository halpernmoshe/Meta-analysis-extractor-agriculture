#!/usr/bin/env python3
"""
Package Codex-kept rows for LLM-based effector normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "codex_filtered_results"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "effector_review_batches"

TOPICS = ["organic_yield_gap", "notill_tillage"]


def topic_brief(config: dict) -> dict:
    pico = config.get("pico", {})
    benchmark = config.get("benchmark", {})
    return {
        "review_id": config.get("review_id"),
        "title": config.get("title"),
        "research_question": config.get("research_question"),
        "intervention_description": pico.get("intervention", {}).get("description"),
        "comparator_description": pico.get("comparator", {}).get("description"),
        "outcome_description": pico.get("outcome", {}).get("primary", {}).get("description"),
        "important_moderators": config.get("important_moderators", []),
        "benchmark_source": benchmark.get("source"),
        "benchmark_notes": benchmark.get("published_pooled_effect", {}).get("notes"),
    }


def row_payload(topic: str, idx: int, row: pd.Series) -> dict:
    payload = {
        "row_id": f"{topic}::{row.get('paper_id', 'unknown')}::{idx}",
        "paper_id": row.get("paper_id"),
        "title": row.get("title"),
        "outcome": row.get("outcome"),
        "outcome_unit": row.get("outcome_unit"),
        "treatment_description": row.get("treatment_description"),
        "control_description": row.get("control_description"),
        "notes": row.get("notes"),
    }
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
    kept_df = pd.read_csv(INPUT_ROOT / f"{topic}_kept.csv")

    brief = topic_brief(config)
    payloads = [row_payload(topic, idx, row) for idx, row in kept_df.iterrows()]
    batches = [payloads[i : i + batch_size] for i in range(0, len(payloads), batch_size)]

    index = []
    for i, batch in enumerate(batches, start=1):
        batch_obj = {"topic": topic, "topic_brief": brief, "rows": batch}
        path = topic_dir / f"batch_{i:03d}.json"
        path.write_text(json.dumps(batch_obj, indent=2, ensure_ascii=False), encoding="utf-8")
        index.append({"batch": i, "rows": len(batch), "path": str(path)})

    (topic_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    for topic in TOPICS:
        build_batches(topic)


if __name__ == "__main__":
    main()
