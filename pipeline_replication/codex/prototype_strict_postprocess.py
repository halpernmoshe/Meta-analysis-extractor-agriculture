#!/usr/bin/env python3
"""
Prototype stricter post-extraction filter layer.

Reads existing `summary_validated.csv` files from topic folders, applies
stricter topic-specific rules, and writes outputs under `codex/outputs`.

This script does not modify the main pipeline artifacts.
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
OUTPUT_ROOT = CODEX_ROOT / "outputs"

TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]

BENCHMARKS = {
    "organic_yield_gap": -19.2,
    "notill_tillage": -5.7,
    "mycorrhiza_yield": 23.0,
    "legume_rotation": 20.0,
    "biochar_crop_yield": 16.0,
    "intercropping_yield": 22.0,
}


def ensure_output_dirs() -> None:
    OUTPUT_ROOT.mkdir(exist_ok=True)
    for topic in TOPICS:
        (OUTPUT_ROOT / topic).mkdir(exist_ok=True)


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


def dersimonian_laird(df: pd.DataFrame) -> dict | None:
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


def load_validated(topic: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / topic / "4_extract" / "summary_validated.csv")


def outcome_text(df: pd.DataFrame) -> pd.Series:
    return (
        df.get("outcome", "").astype(str).str.lower()
        + " | "
        + df.get("outcome_unit", "").astype(str).str.lower()
        + " | "
        + df.get("treatment_description", "").astype(str).str.lower()
        + " | "
        + df.get("control_description", "").astype(str).str.lower()
    )


def keep_positive_means(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["treatment_mean"] > 0) & (df["control_mean"] > 0)].copy()


def filter_organic(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    text = outcome_text(df)
    exclude_terms = [
        "concentration",
        "protein",
        "energy",
        "hectolitre",
        "hme",
        "cu concentration",
        "fe concentration",
        "zn concentration",
        "mn concentration",
        "p/l ratio",
        "quality",
        "fatty acid",
        "nitrogen content",
    ]
    keep_terms = [
        "yield",
        "grain yield",
        "fruit yield",
        "tuber yield",
        "equivalent grain yield",
        "total yield",
    ]
    keep_mask = pd.Series(False, index=df.index)
    for term in keep_terms:
        keep_mask |= text.str.contains(term, regex=False, na=False)
    exclude_mask = pd.Series(False, index=df.index)
    for term in exclude_terms:
        exclude_mask |= text.str.contains(term, regex=False, na=False)
    return keep_positive_means(df[keep_mask & ~exclude_mask].copy()), exclude_terms


def filter_notill(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    text = outcome_text(df)
    keep_mask = text.str.contains("grain yield", regex=False, na=False) | (
        text.str.contains("yield", regex=False, na=False)
        & ~text.str.contains("straw yield", regex=False, na=False)
        & ~text.str.contains("biological yield", regex=False, na=False)
    )
    exclude_terms = ["straw yield", "biological yield", "forage", "biomass"]
    exclude_mask = pd.Series(False, index=df.index)
    for term in exclude_terms:
        exclude_mask |= text.str.contains(term, regex=False, na=False)
    return keep_positive_means(df[keep_mask & ~exclude_mask].copy()), exclude_terms


def filter_mycorrhiza(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    text = outcome_text(df)
    keep_terms = [
        "grain yield",
        "yield",
        "fruit yield",
        "pod yield",
        "shoot dry weight",
        "shoot biomass",
        "total biomass",
        "plant dry weight",
    ]
    exclude_terms = [
        "root dry weight",
        "root biomass",
        "colonization",
        "quantum",
        "uptake",
        "npq",
        "root length",
    ]
    keep_mask = pd.Series(False, index=df.index)
    for term in keep_terms:
        keep_mask |= text.str.contains(term, regex=False, na=False)
    exclude_mask = pd.Series(False, index=df.index)
    for term in exclude_terms:
        exclude_mask |= text.str.contains(term, regex=False, na=False)
    return keep_positive_means(df[keep_mask & ~exclude_mask].copy()), exclude_terms


def filter_legume(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    text = outcome_text(df)
    keep_mask = (
        text.str.contains("yield", regex=False, na=False)
        | text.str.contains("grain", regex=False, na=False)
        | text.str.contains("dry matter yield", regex=False, na=False)
    )
    exclude_terms = [
        "pgpr",
        "rhizobia",
        "mycorrhiza",
        "amf",
        "inoculation",
        "pod weight",
    ]
    exclude_mask = pd.Series(False, index=df.index)
    for term in exclude_terms:
        exclude_mask |= text.str.contains(term, regex=False, na=False)
    return keep_positive_means(df[keep_mask & ~exclude_mask].copy()), exclude_terms


def filter_biochar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    text = outcome_text(df)
    study_type = df.get("mod_experiment_type", pd.Series("", index=df.index)).astype(str).str.lower()
    is_field = study_type.str.contains("field", regex=False, na=False)
    keep_mask = text.str.contains("yield", regex=False, na=False) | text.str.contains("grain", regex=False, na=False)
    exclude_terms = [
        "pot",
        "growth chamber",
        "root biomass",
        "shoot dry weight",
        "w/w",
    ]
    exclude_mask = pd.Series(False, index=df.index)
    for term in exclude_terms:
        exclude_mask |= text.str.contains(term, regex=False, na=False)
    return keep_positive_means(df[is_field & keep_mask & ~exclude_mask].copy()), exclude_terms


def filter_intercropping(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    obs_type = df.get("_obs_type", pd.Series("", index=df.index)).astype(str)
    filtered = df[obs_type == "LER"].copy()
    return keep_positive_means(filtered), ["individual_crop_yield"]


FILTERS = {
    "organic_yield_gap": filter_organic,
    "notill_tillage": filter_notill,
    "mycorrhiza_yield": filter_mycorrhiza,
    "legume_rotation": filter_legume,
    "biochar_crop_yield": filter_biochar,
    "intercropping_yield": filter_intercropping,
}


def summarize_topic(topic: str, before_df: pd.DataFrame, after_df: pd.DataFrame, excluded_terms: list[str]) -> dict:
    before_meta = dersimonian_laird(before_df)
    after_meta = dersimonian_laird(after_df)

    summary = {
        "topic": topic,
        "benchmark_pct": BENCHMARKS.get(topic),
        "before": {
            "n_obs": int(len(before_df)),
            "n_papers": int(before_df["paper_id"].nunique()),
            "meta": before_meta,
        },
        "after": {
            "n_obs": int(len(after_df)),
            "n_papers": int(after_df["paper_id"].nunique()),
            "meta": after_meta,
        },
        "excluded_term_rules": excluded_terms,
    }
    if before_meta and BENCHMARKS.get(topic) is not None:
        summary["before"]["abs_diff_vs_benchmark"] = round(
            abs(before_meta["pooled_pct"] - BENCHMARKS[topic]), 2
        )
    if after_meta and BENCHMARKS.get(topic) is not None:
        summary["after"]["abs_diff_vs_benchmark"] = round(
            abs(after_meta["pooled_pct"] - BENCHMARKS[topic]), 2
        )
    return summary


def write_markdown_report(summaries: list[dict]) -> None:
    lines = ["# Strict Post-Processing Prototype Results", ""]
    lines.append("These outputs were generated from the current `summary_validated.csv` files.")
    lines.append("They are diagnostic strict-pass analyses only.")
    lines.append("")
    for item in summaries:
        lines.append(f"## {item['topic']}")
        lines.append(f"- Benchmark: {item['benchmark_pct']}%")
        before = item["before"]
        after = item["after"]
        lines.append(
            f"- Before strict pass: {before['n_obs']} obs / {before['n_papers']} papers"
        )
        if before["meta"]:
            lines.append(
                f"- Before pooled: {before['meta']['pooled_pct']:.2f}% "
                f"[{before['meta']['ci_lo_pct']:.2f}, {before['meta']['ci_hi_pct']:.2f}] "
                f"(abs diff {before.get('abs_diff_vs_benchmark', 'n/a')} pp)"
            )
        else:
            lines.append("- Before pooled: insufficient variance-bearing rows for DL synthesis")
        lines.append(
            f"- After strict pass: {after['n_obs']} obs / {after['n_papers']} papers"
        )
        if after["meta"]:
            lines.append(
                f"- After pooled: {after['meta']['pooled_pct']:.2f}% "
                f"[{after['meta']['ci_lo_pct']:.2f}, {after['meta']['ci_hi_pct']:.2f}] "
                f"(abs diff {after.get('abs_diff_vs_benchmark', 'n/a')} pp)"
            )
        else:
            lines.append("- After pooled: insufficient variance-bearing rows for DL synthesis")
        lines.append(
            "- Strict rules used: " + ", ".join(item["excluded_term_rules"])
            if item["excluded_term_rules"]
            else "- Strict rules used: topic-specific primary target only"
        )
        lines.append("")

    (OUTPUT_ROOT / "STRICT_RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_output_dirs()
    summaries = []
    for topic in TOPICS:
        validated_df = load_validated(topic)
        strict_df, excluded_terms = FILTERS[topic](validated_df)
        strict_path = OUTPUT_ROOT / topic / "summary_strict.csv"
        strict_df.to_csv(strict_path, index=False)

        summary = summarize_topic(topic, validated_df, strict_df, excluded_terms)
        summaries.append(summary)
        (OUTPUT_ROOT / topic / "strict_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    (OUTPUT_ROOT / "strict_results.json").write_text(
        json.dumps(summaries, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(summaries)


if __name__ == "__main__":
    main()
