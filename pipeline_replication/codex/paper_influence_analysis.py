#!/usr/bin/env python3
"""
Assess whether pooled estimates are driven by a few influential papers.
Writes outputs under codex/outputs/paper_influence.
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
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "paper_influence"

TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]


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
    return {
        "k": int(k),
        "pooled_pct": float(lnrr_to_pct(mu_re)),
        "ci_lo_pct": float(lnrr_to_pct(mu_re - 1.96 * se_re)),
        "ci_hi_pct": float(lnrr_to_pct(mu_re + 1.96 * se_re)),
    }


def analyze_topic(topic: str) -> dict:
    df = pd.read_csv(ROOT / topic / "4_extract" / "summary_validated.csv")
    df = df[(df["treatment_mean"] > 0) & (df["control_mean"] > 0)].copy()
    overall = dl_meta(df)
    if overall is None:
        return {"topic": topic, "error": "insufficient rows"}

    paper_counts = df.groupby("paper_id").size().sort_values(ascending=False)
    top_papers = []
    for paper_id, n_obs in paper_counts.head(10).items():
        sub = df[df["paper_id"] == paper_id].copy()
        meta = dl_meta(sub)
        top_papers.append(
            {
                "paper_id": paper_id,
                "n_obs": int(n_obs),
                "paper_only_meta": meta,
            }
        )

    loo = []
    for paper_id in df["paper_id"].unique():
        sub = df[df["paper_id"] != paper_id].copy()
        meta = dl_meta(sub)
        if meta:
            loo.append({"paper_id": paper_id, "pooled_pct": meta["pooled_pct"]})

    loo_df = pd.DataFrame(loo)
    range_pct = None
    most_influential = None
    if not loo_df.empty:
        loo_df["abs_shift"] = (loo_df["pooled_pct"] - overall["pooled_pct"]).abs()
        loo_df = loo_df.sort_values("abs_shift", ascending=False)
        range_pct = float(loo_df["pooled_pct"].max() - loo_df["pooled_pct"].min())
        most_influential = loo_df.head(10).to_dict(orient="records")

    summary = {
        "topic": topic,
        "overall_meta": overall,
        "n_obs": int(len(df)),
        "n_papers": int(df["paper_id"].nunique()),
        "top_papers_by_rows": top_papers,
        "leave_one_out_range_pct": range_pct,
        "most_influential_papers": most_influential,
    }
    return summary


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_topic(topic) for topic in TOPICS]
    (OUTPUT_ROOT / "paper_influence_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
