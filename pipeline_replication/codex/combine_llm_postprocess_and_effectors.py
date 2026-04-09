#!/usr/bin/env python3
"""
Combine Codex keep/exclude decisions with LLM-normalized effector labels and
test whether aligned subsets improve replication.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CODEX_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "codex_filtered_results"
LABEL_ROOT = CODEX_ROOT / "outputs" / "effector_labels"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "llm_combined_results"

BENCHMARKS = {
    "organic_yield_gap": -19.2,
    "notill_tillage": -5.7,
}


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


def load_labels(topic: str) -> pd.DataFrame:
    path = LABEL_ROOT / topic / "labels.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def annotate_rows(topic: str) -> pd.DataFrame:
    kept = pd.read_csv(INPUT_ROOT / f"{topic}_kept.csv")
    kept["row_id"] = [f"{topic}::{row.paper_id}::{idx}" for idx, row in kept.reset_index(drop=True).iterrows()]
    labels = load_labels(topic)
    merged = kept.merge(labels, on="row_id", how="left")
    return merged


def subset_summary(topic: str, label: str, df: pd.DataFrame) -> dict:
    meta = dl_meta(df)
    result = {
        "label": label,
        "n_obs": int(len(df)),
        "n_papers": int(df["paper_id"].nunique()) if len(df) else 0,
        "meta": meta,
    }
    if meta:
        result["abs_diff_vs_benchmark"] = round(abs(meta["pooled_pct"] - BENCHMARKS[topic]), 2)
    return result


def analyze_topic(topic: str) -> list[dict]:
    df = annotate_rows(topic)
    results = [subset_summary(topic, "codex_kept_all", df)]

    aligned = df[df["normalized_estimand_context"] == "benchmark_aligned"].copy()
    if len(aligned) >= 8:
        results.append(subset_summary(topic, "llm_benchmark_aligned_only", aligned))

    if topic == "organic_yield_gap":
        cereal = df[df["normalized_crop_class"] == "grain_cereal"].copy()
        if len(cereal) >= 8:
            results.append(subset_summary(topic, "llm_crop=grain_cereal", cereal))
        cereal_aligned = df[
            (df["normalized_crop_class"] == "grain_cereal")
            & (df["normalized_estimand_context"].isin(["benchmark_aligned", "partially_aligned"]))
        ].copy()
        if len(cereal_aligned) >= 8:
            results.append(subset_summary(topic, "llm_crop=grain_cereal_plus_aligned", cereal_aligned))

    if topic == "notill_tillage":
        temp = df[df["normalized_climate_class"] == "temperate"].copy()
        if len(temp) >= 8:
            results.append(subset_summary(topic, "llm_climate=temperate", temp))
        resrot = df[df["normalized_management_class"] == "residue_rotation"].copy()
        if len(resrot) >= 8:
            results.append(subset_summary(topic, "llm_management=residue_rotation", resrot))
        both = df[
            (df["normalized_climate_class"] == "temperate")
            & (df["normalized_management_class"] == "residue_rotation")
        ].copy()
        if len(both) >= 8:
            results.append(subset_summary(topic, "llm_temperate_and_residue_rotation", both))

    return sorted(results, key=lambda x: (x.get("abs_diff_vs_benchmark", 999), -x["n_obs"]))


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "organic_yield_gap": analyze_topic("organic_yield_gap"),
        "notill_tillage": analyze_topic("notill_tillage"),
    }
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
