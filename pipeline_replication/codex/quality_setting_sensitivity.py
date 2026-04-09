#!/usr/bin/env python3
"""
Run simple quality/setting sensitivity analyses on validated topic outputs.
Writes outputs under codex/outputs/quality_setting_sensitivity.
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
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "quality_setting_sensitivity"

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


def setting_mask(df: pd.DataFrame) -> dict[str, pd.Series]:
    mods = pd.Series("", index=df.index)
    for col in ["mod_experiment_type", "mod_study_type"]:
        if col in df.columns:
            mods = mods + " " + df[col].astype(str).str.lower()
    text = mods + " " + df.get("outcome_unit", pd.Series("", index=df.index)).astype(str).str.lower()
    return {
        "field_only": text.str.contains("field|on-farm|trial", na=False),
        "pot_only": text.str.contains("pot|g/pot|mg/pot|w/w", na=False),
        "greenhouse_only": text.str.contains("greenhouse|growth chamber", na=False),
    }


def variance_present_mask(df: pd.DataFrame) -> pd.Series:
    return (
        pd.notna(df.get("sd_treatment"))
        | pd.notna(df.get("se_treatment"))
        | pd.notna(df.get("variance_value"))
    )


def analyze_topic(topic: str) -> dict:
    df = pd.read_csv(ROOT / topic / "4_extract" / "summary_validated.csv")
    df = df[(df["treatment_mean"] > 0) & (df["control_mean"] > 0)].copy()
    masks = {
        "all_rows": pd.Series(True, index=df.index),
        "table_only": df["source_type"].astype(str).str.lower() == "table" if "source_type" in df.columns else pd.Series(False, index=df.index),
        "figure_only": df["source_type"].astype(str).str.lower() == "figure" if "source_type" in df.columns else pd.Series(False, index=df.index),
        "high_conf_only": df["confidence"].astype(str).str.lower() == "high" if "confidence" in df.columns else pd.Series(False, index=df.index),
        "medium_or_high": df["confidence"].astype(str).str.lower().isin(["high", "medium"]) if "confidence" in df.columns else pd.Series(False, index=df.index),
        "variance_present": variance_present_mask(df),
    }
    masks.update(setting_mask(df))

    results = {"topic": topic, "subsets": []}
    for label, mask in masks.items():
        sub = df[mask].copy()
        if len(sub) < 8:
            continue
        meta = dl_meta(sub)
        results["subsets"].append(
            {
                "label": label,
                "n_obs": int(len(sub)),
                "n_papers": int(sub["paper_id"].nunique()),
                "meta": meta,
            }
        )
    return results


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = [analyze_topic(topic) for topic in TOPICS]
    (OUTPUT_ROOT / "quality_setting_sensitivity_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
