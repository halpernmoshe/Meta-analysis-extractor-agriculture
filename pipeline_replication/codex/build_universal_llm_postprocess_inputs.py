#!/usr/bin/env python3
"""
Build universal LLM post-processing inputs from topic configs and extracted rows.

This script does not run an LLM. It packages rows for a future Claude Opus 4.6
 adjudication pass using only:

- topic config
- extracted row fields

Outputs are written under `codex/outputs/universal_llm_inputs`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "universal_llm_inputs"

TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]


def load_config(topic: str) -> dict:
    return json.loads((ROOT / topic / "config.json").read_text(encoding="utf-8"))


def load_rows(topic: str) -> pd.DataFrame:
    path = ROOT / topic / "4_extract" / "summary.csv"
    return pd.read_csv(path)


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
        "extraction_priorities": config.get("extraction_priorities", []),
        "benchmark_source": benchmark.get("source"),
        "benchmark_notes": benchmark.get("published_pooled_effect", {}).get("notes"),
    }


def row_payload(topic: str, row_idx: int, row: pd.Series) -> dict:
    keep_cols = [
        "paper_id",
        "title",
        "authors",
        "year",
        "journal",
        "doi",
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
        "table_or_figure",
        "notes",
    ]
    payload = {
        "row_id": f"{topic}::{row.get('paper_id', 'unknown')}::{row_idx}",
        "topic": topic,
    }
    for col in keep_cols:
        if col in row.index:
            value = row[col]
            payload[col] = None if pd.isna(value) else value

    mod_cols = [c for c in row.index if c.startswith("mod_")]
    mods = {}
    for col in mod_cols:
        value = row[col]
        if pd.notna(value) and str(value).strip():
            mods[col] = value
    payload["moderators"] = mods
    return payload


def heuristic_flags(config: dict, row: pd.Series) -> dict:
    pico = config.get("pico", {})
    intervention_terms = [str(x).lower() for x in pico.get("intervention", {}).get("search_terms", [])]
    comparator_terms = [str(x).lower() for x in pico.get("comparator", {}).get("search_terms", [])]
    outcome_terms = [str(x).lower() for x in pico.get("outcome", {}).get("primary", {}).get("search_terms", [])]

    treatment_text = str(row.get("treatment_description", "")).lower()
    control_text = str(row.get("control_description", "")).lower()
    outcome_text = (str(row.get("outcome", "")) + " " + str(row.get("outcome_unit", ""))).lower()

    return {
        "missing_means": pd.isna(row.get("treatment_mean")) or pd.isna(row.get("control_mean")),
        "nonpositive_means": (
            pd.notna(row.get("treatment_mean"))
            and pd.notna(row.get("control_mean"))
            and (float(row.get("treatment_mean")) <= 0 or float(row.get("control_mean")) <= 0)
        ),
        "intervention_term_hit": any(term in treatment_text for term in intervention_terms if term),
        "comparator_term_hit": any(term in control_text for term in comparator_terms if term),
        "outcome_term_hit": any(term in outcome_text for term in outcome_terms if term),
        "low_confidence": str(row.get("confidence", "")).lower() == "low",
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for topic in TOPICS:
        topic_dir = OUTPUT_ROOT / topic
        topic_dir.mkdir(exist_ok=True)

        config = load_config(topic)
        df = load_rows(topic)
        brief = topic_brief(config)

        prompt_items = []
        for idx, row in df.iterrows():
            item = {
                "topic_brief": brief,
                "row": row_payload(topic, idx, row),
                "heuristic_flags": heuristic_flags(config, row),
            }
            prompt_items.append(item)

        jsonl_path = topic_dir / "llm_review_inputs.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in prompt_items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        (topic_dir / "topic_brief.json").write_text(
            json.dumps(brief, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        index_rows.append(
            {
                "topic": topic,
                "rows_packaged": len(prompt_items),
                "output_jsonl": str(jsonl_path),
            }
        )

    (OUTPUT_ROOT / "index.json").write_text(
        json.dumps(index_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
