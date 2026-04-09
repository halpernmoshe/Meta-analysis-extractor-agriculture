#!/usr/bin/env python3
"""
Apply Codex row decisions to validated CSVs and compare pooled effects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
DECISIONS_ROOT = CODEX_ROOT / "outputs" / "codex_decisions"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "codex_filtered_results"

TOPICS = ["organic_yield_gap", "notill_tillage"]
BENCHMARKS = {"organic_yield_gap": -19.2, "notill_tillage": -5.7}


def compute_lnrr(t_mean: float, c_mean: float) -> float | None:
    try:
        t_val = float(t_mean)
        c_val = float(c_mean)
        if t_val > 0 and c_val > 0:
            return math.log(t_val / c_val)
    except (TypeError, ValueError):
        return None
    return None


def lnrr_to_pct(lnrr: float) -> float:
    return (math.exp(lnrr) - 1.0) * 100.0


def get_sd(row: pd.Series) -> tuple[float | None, float | None, float | None, float | None]:
    sd_t = row.get("sd_treatment")
    sd_c = row.get("sd_control")
    n_t = row.get("treatment_n") or row.get("control_n")
    n_c = row.get("control_n") or row.get("treatment_n")

    if pd.isna(sd_t) and not pd.isna(row.get("se_treatment")) and pd.notna(n_t) and float(n_t) > 0:
        sd_t = float(row["se_treatment"]) * math.sqrt(float(n_t))
    if pd.isna(sd_c) and not pd.isna(row.get("se_control")) and pd.notna(n_c) and float(n_c) > 0:
        sd_c = float(row["se_control"]) * math.sqrt(float(n_c))

    if pd.isna(sd_t) and not pd.isna(row.get("variance_value")):
        if str(row.get("variance_type", "")).upper() == "LSD":
            lsd = float(row["variance_value"])
            n_val = float(n_t) if pd.notna(n_t) else 3.0
            df_val = 2 * (n_val - 1)
            if df_val > 0:
                t_crit = stats.t.ppf(0.975, df_val)
                se_diff = lsd / (t_crit * math.sqrt(2))
                sd_est = se_diff * math.sqrt(n_val)
                sd_t = sd_est
                sd_c = sd_est

    return sd_t, sd_c, n_t, n_c


def variance_lnrr(
    sd_t: float | None,
    sd_c: float | None,
    n_t: float | None,
    n_c: float | None,
    mean_t: float,
    mean_c: float,
) -> float | None:
    try:
        vals = [float(x) for x in (sd_t, sd_c, n_t, n_c, mean_t, mean_c)]
        if any(v <= 0 for v in vals):
            return None
        sd_t, sd_c, n_t, n_c, mean_t, mean_c = vals
        return (sd_t**2 / (n_t * mean_t**2)) + (sd_c**2 / (n_c * mean_c**2))
    except (TypeError, ValueError):
        return None


def dl_meta(df: pd.DataFrame) -> dict | None:
    yi = []
    vi = []
    for _, row in df.iterrows():
        lnrr = compute_lnrr(row["treatment_mean"], row["control_mean"])
        if lnrr is None:
            continue
        sd_t, sd_c, n_t, n_c = get_sd(row)
        var = variance_lnrr(sd_t, sd_c, n_t, n_c, row["treatment_mean"], row["control_mean"])
        if var is not None and var > 0:
            yi.append(lnrr)
            vi.append(var)

    if len(yi) < 3:
        return None

    yi_arr = np.array(yi, dtype=float)
    vi_arr = np.array(vi, dtype=float)
    wi = 1.0 / vi_arr
    sum_w = wi.sum()
    mu_fe = (wi * yi_arr).sum() / sum_w
    q_stat = (wi * (yi_arr - mu_fe) ** 2).sum()
    k = len(yi_arr)
    df_q = k - 1
    c_val = sum_w - (wi**2).sum() / sum_w
    tau2 = max(0.0, (q_stat - df_q) / c_val) if c_val > 0 else 0.0
    wi_re = 1.0 / (vi_arr + tau2)
    sum_w_re = wi_re.sum()
    mu_re = (wi_re * yi_arr).sum() / sum_w_re
    se_re = 1.0 / math.sqrt(sum_w_re)
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    return {
        "k": int(k),
        "pooled_pct": float(lnrr_to_pct(mu_re)),
        "ci_lo_pct": float(lnrr_to_pct(ci_lo)),
        "ci_hi_pct": float(lnrr_to_pct(ci_hi)),
    }


def load_decisions(topic: str) -> dict[str, dict]:
    path = DECISIONS_ROOT / topic / "decisions.jsonl"
    decisions = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            decisions[obj["row_id"]] = obj
    return decisions


def row_id(topic: str, idx: int, row: pd.Series) -> str:
    return f"{topic}::{row.get('paper_id', 'unknown')}::{idx}"


def apply_swaps(df: pd.DataFrame, topic: str, decisions: dict[str, dict]) -> pd.DataFrame:
    df = df.copy()
    for idx, row in df.iterrows():
        decision = decisions.get(row_id(topic, idx, row))
        if not decision:
            continue
        if decision.get("decision") == "swap_treatment_control":
            df.loc[idx, ["treatment_mean", "control_mean"]] = df.loc[idx, ["control_mean", "treatment_mean"]].values
            for a, b in [
                ("treatment_n", "control_n"),
                ("sd_treatment", "sd_control"),
                ("se_treatment", "se_control"),
                ("treatment_description", "control_description"),
            ]:
                if a in df.columns and b in df.columns:
                    df.loc[idx, [a, b]] = df.loc[idx, [b, a]].values
    return df


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for topic in TOPICS:
        df = pd.read_csv(ROOT / topic / "4_extract" / "summary_validated.csv")
        before_meta = dl_meta(df[(df["treatment_mean"] > 0) & (df["control_mean"] > 0)].copy())

        decisions = load_decisions(topic)
        df = apply_swaps(df, topic, decisions)

        keep_ids = {
            rid
            for rid, obj in decisions.items()
            if obj.get("decision") in {"keep", "swap_treatment_control"}
        }
        flag_ids = {
            rid
            for rid, obj in decisions.items()
            if obj.get("decision") == "flag"
        }

        kept_rows = []
        flagged_rows = []
        for idx, row in df.iterrows():
            rid = row_id(topic, idx, row)
            if rid in keep_ids:
                kept_rows.append(row)
            elif rid in flag_ids:
                flagged_rows.append(row)

        kept_df = pd.DataFrame(kept_rows) if kept_rows else pd.DataFrame(columns=df.columns)
        flagged_df = pd.DataFrame(flagged_rows) if flagged_rows else pd.DataFrame(columns=df.columns)

        if not kept_df.empty:
            kept_df.to_csv(OUTPUT_ROOT / f"{topic}_kept.csv", index=False)
        if not flagged_df.empty:
            flagged_df.to_csv(OUTPUT_ROOT / f"{topic}_flagged.csv", index=False)

        after_meta = dl_meta(kept_df[(kept_df["treatment_mean"] > 0) & (kept_df["control_mean"] > 0)].copy()) if not kept_df.empty else None

        summary = {
            "topic": topic,
            "benchmark_pct": BENCHMARKS[topic],
            "before_n": int(len(df)),
            "after_keep_n": int(len(kept_df)),
            "after_flag_n": int(len(flagged_df)),
            "before_meta": before_meta,
            "after_meta": after_meta,
        }
        summaries.append(summary)

    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
