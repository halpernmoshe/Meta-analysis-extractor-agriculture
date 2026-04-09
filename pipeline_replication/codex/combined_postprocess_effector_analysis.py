#!/usr/bin/env python3
"""
Test whether Codex post-processing plus benchmark-relevant effectors
improves replication for the hardest topics.

All outputs are written under codex/outputs/combined_analysis.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "codex_filtered_results"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "combined_analysis"

BENCHMARKS = {
    "organic_yield_gap": -19.2,
    "notill_tillage": -5.7,
}


def ensure_output_dir() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


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


def classify_organic_crop(row: pd.Series) -> str:
    for col in ["mod_crop_type", "mod_crop_species", "outcome", "title"]:
        txt = str(row.get(col, "")).lower()
        if any(k in txt for k in ["wheat", "maize", "corn", "rice", "barley", "sorghum", "oat", "millet", "cereal", "grain"]):
            return "grain_cereal"
        if any(k in txt for k in ["tomato", "okra", "lettuce", "vegetable", "cabbage", "pepper", "eggplant", "onion"]):
            return "vegetable"
        if any(k in txt for k in ["fruit", "apple", "grape", "banana", "strawberry", "citrus", "raspberry"]):
            return "fruit"
        if any(k in txt for k in ["potato", "tuber", "root"]):
            return "root_tuber"
        if any(k in txt for k in ["soybean", "legume", "bean", "pea", "chickpea", "lentil"]):
            return "legume"
    return "other_unknown"


def classify_notill_crop(row: pd.Series) -> str:
    txt = " ".join(str(row.get(c, "")) for c in ["mod_crop_species", "outcome", "title"]).lower()
    if "wheat" in txt:
        return "wheat"
    if "maize" in txt or "corn" in txt:
        return "maize"
    if "soybean" in txt:
        return "soybean"
    if "rice" in txt:
        return "rice"
    return "other"


def residue_rotation_mask(df: pd.DataFrame) -> pd.Series:
    residue_text = (
        df.get("mod_residue_management", pd.Series("", index=df.index)).astype(str).str.lower()
        + " "
        + df.get("mod_residue_management_treatment", pd.Series("", index=df.index)).astype(str).str.lower()
    )
    rotation_text = (
        df.get("mod_crop_rotation", pd.Series("", index=df.index)).astype(str).str.lower()
        + " "
        + df.get("mod_rotation", pd.Series("", index=df.index)).astype(str).str.lower()
    )
    residue_mask = residue_text.str.contains("retain|retained|straw return|mulch|residue retained", na=False)
    rotation_mask = rotation_text.str.contains("rotation|legume|divers", na=False)
    return residue_mask & rotation_mask


def climate_temperate_mask(df: pd.DataFrame) -> pd.Series:
    climate_text = (
        df.get("mod_climate", pd.Series("", index=df.index)).astype(str).str.lower()
        + " "
        + df.get("mod_climate_character", pd.Series("", index=df.index)).astype(str).str.lower()
    )
    return climate_text.str.contains("temperate", na=False)


def summarize_subset(topic: str, label: str, df: pd.DataFrame) -> dict:
    meta = dl_meta(df)
    result = {
        "label": label,
        "n_obs": int(len(df)),
        "n_papers": int(df["paper_id"].nunique()) if len(df) else 0,
        "meta": meta,
    }
    if meta is not None:
        result["abs_diff_vs_benchmark"] = round(abs(meta["pooled_pct"] - BENCHMARKS[topic]), 2)
    return result


def analyze_organic() -> list[dict]:
    df = pd.read_csv(INPUT_ROOT / "organic_yield_gap_kept.csv")
    df = df[(df["treatment_mean"] > 0) & (df["control_mean"] > 0)].copy()
    df["crop_class"] = df.apply(classify_organic_crop, axis=1)

    results = [summarize_subset("organic_yield_gap", "codex_kept_all", df)]
    for group_name, group_df in df.groupby("crop_class"):
        if len(group_df) >= 8:
            results.append(summarize_subset("organic_yield_gap", f"codex_kept_crop={group_name}", group_df.copy()))
    return sorted(results, key=lambda x: (x.get("abs_diff_vs_benchmark", 999), -x["n_obs"]))


def analyze_notill() -> list[dict]:
    df = pd.read_csv(INPUT_ROOT / "notill_tillage_kept.csv")
    df = df[(df["treatment_mean"] > 0) & (df["control_mean"] > 0)].copy()
    df["crop_class"] = df.apply(classify_notill_crop, axis=1)

    results = [summarize_subset("notill_tillage", "codex_kept_all", df)]

    mask_res_rot = residue_rotation_mask(df)
    if mask_res_rot.sum() >= 8:
        results.append(summarize_subset("notill_tillage", "codex_kept_residue_plus_rotation", df[mask_res_rot].copy()))

    mask_temp = climate_temperate_mask(df)
    if mask_temp.sum() >= 8:
        results.append(summarize_subset("notill_tillage", "codex_kept_temperate", df[mask_temp].copy()))

    mask_both = mask_res_rot & mask_temp
    if mask_both.sum() >= 8:
        results.append(summarize_subset("notill_tillage", "codex_kept_temperate_residue_rotation", df[mask_both].copy()))

    for group_name, group_df in df.groupby("crop_class"):
        if len(group_df) >= 8:
            results.append(summarize_subset("notill_tillage", f"codex_kept_crop={group_name}", group_df.copy()))

    return sorted(results, key=lambda x: (x.get("abs_diff_vs_benchmark", 999), -x["n_obs"]))


def write_markdown(summary: dict) -> None:
    lines = ["# Combined Post-Processing + Effector Analysis", ""]
    lines.append("This analysis tests whether Codex-cleaned rows plus benchmark-relevant effectors jointly improve replication.")
    lines.append("")
    for topic, items in summary.items():
        lines.append(f"## {topic}")
        lines.append(f"- Benchmark: {BENCHMARKS[topic]}%")
        for item in items:
            if item["meta"] is None:
                lines.append(f"- {item['label']}: {item['n_obs']} obs / {item['n_papers']} papers; insufficient rows for DL synthesis")
            else:
                lines.append(
                    f"- {item['label']}: {item['n_obs']} obs / {item['n_papers']} papers; "
                    f"{item['meta']['pooled_pct']:.2f}% "
                    f"[{item['meta']['ci_lo_pct']:.2f}, {item['meta']['ci_hi_pct']:.2f}] "
                    f"(abs diff {item['abs_diff_vs_benchmark']} pp)"
                )
        lines.append("")
    (OUTPUT_ROOT / "COMBINED_ANALYSIS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dir()
    summary = {
        "organic_yield_gap": analyze_organic(),
        "notill_tillage": analyze_notill(),
    }
    (OUTPUT_ROOT / "combined_analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary)


if __name__ == "__main__":
    main()
