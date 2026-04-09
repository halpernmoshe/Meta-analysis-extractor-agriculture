#!/usr/bin/env python3
"""
Small helper to rank topic candidates from topic_scoring_template.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "topic_scoring_template.csv"
OUT_PATH = ROOT / "topic_scoring_ranked.csv"

SCORE_COLS = [
    "estimand_clarity_1_5",
    "intervention_clarity_1_5",
    "comparator_clarity_1_5",
    "study_setting_consistency_1_5",
    "oa_feasibility_1_5",
    "benchmark_spec_richness_1_5",
    "moderator_extractability_1_5",
    "low_estimand_trap_risk_1_5",
]


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    for col in SCORE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["total_score"] = df[SCORE_COLS].sum(axis=1)
    df = df.sort_values(["total_score", "oa_feasibility_1_5", "estimand_clarity_1_5"], ascending=False)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote ranked candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()
