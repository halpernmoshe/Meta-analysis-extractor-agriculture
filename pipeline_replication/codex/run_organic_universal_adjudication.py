#!/usr/bin/env python3
"""
Universal semantic adjudication prototype for organic_yield_gap.

This uses only:
- the organic config
- the extracted validated row fields

It writes row-level decisions and a kept-row CSV under codex/outputs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
TOPIC = "organic_yield_gap"
CONFIG_PATH = ROOT / TOPIC / "config.json"
BATCH_DIR = CODEX_ROOT / "outputs" / "validated_review_batches" / TOPIC
OUT_DIR = CODEX_ROOT / "outputs" / "codex_decisions" / TOPIC


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_batches() -> list[dict]:
    batch_paths = sorted(BATCH_DIR.glob("batch_*.json"))
    batches = []
    for path in batch_paths:
        batches.append(json.loads(path.read_text(encoding="utf-8")))
    return batches


def normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def gather_text(row: dict) -> str:
    parts = [
        row.get("outcome", ""),
        row.get("outcome_unit", ""),
        row.get("treatment_description", ""),
        row.get("control_description", ""),
        row.get("title", ""),
        row.get("notes", ""),
    ]
    return " | ".join(normalize_text(p) for p in parts if p is not None)


def keyword_union(*items) -> list[str]:
    out = []
    for item in items:
        if not item:
            continue
        if isinstance(item, str):
            out.append(normalize_text(item))
        else:
            for sub in item:
                if sub is not None and str(sub).strip():
                    out.append(normalize_text(sub))
    return [x for x in out if x]


def build_terms(config: dict) -> dict:
    pico = config["pico"]
    return {
        "intervention": keyword_union(
            pico["intervention"]["description"],
            pico["intervention"]["search_terms"],
            config.get("intervention"),
        ),
        "comparator": keyword_union(
            pico["comparator"]["description"],
            pico["comparator"]["search_terms"],
            config.get("control"),
        ),
        "outcome": keyword_union(
            pico["outcome"]["primary"]["description"],
            pico["outcome"]["primary"]["search_terms"],
            config.get("primary_outcomes", []),
        ),
        "population_exclude": keyword_union(
            pico["population"].get("exclude_terms", [])
        ),
    }


def has_any(text: str, terms: list[str]) -> bool:
    return any(term in text for term in terms if term)


def has_all(text: str, terms: list[str]) -> bool:
    return all(term in text for term in terms if term)


def classify_outcome(text: str) -> tuple[str, str, str]:
    """
    Returns (outcome_match, estimand_match, normalized_outcome_class).
    """
    yield_terms = [
        "yield",
        "productivity",
        "harvest",
        "grain",
        "fruit",
        "seed",
        "tuber",
        "total biomass",
        "shoot biomass",
        "shoot dry weight",
        "plant dry weight",
        "dry matter yield",
        "equivalent yield",
    ]
    hard_off_target = [
        "concentration",
        "content",
        "protein",
        "energy",
        "ratio",
        "quality",
        "hectolitre",
        "number of",
        "count",
        "leaf",
        "height",
        "chlorophyll",
        "uptake",
        "colonization",
        "infection",
        "spore",
        "photosynthesis",
        "npq",
        "p/l ratio",
        "brightness",
        "color",
        "cu ",
        "fe ",
        "zn ",
        "mn ",
    ]
    biomass_terms = ["biomass", "dry matter", "dry weight"]
    root_terms = ["root biomass", "root dry weight", "root length"]
    system_terms = ["equivalent yield", "system productivity", "ler", "land equivalent ratio"]

    outcome_class = "other"
    if "grain" in text:
        outcome_class = "grain_yield"
    elif "fruit" in text or "pod" in text:
        outcome_class = "harvest_yield"
    elif "tuber" in text:
        outcome_class = "harvest_yield"
    elif any(term in text for term in biomass_terms):
        outcome_class = "biomass"
    elif any(term in text for term in ["concentration", "content", "protein", "quality", "ratio"]):
        outcome_class = "quality_trait"

    if any(term in text for term in hard_off_target) and not any(term in text for term in yield_terms):
        return "no", "no", outcome_class

    if any(term in text for term in system_terms):
        return "partial", "no", "system_productivity"

    if any(term in text for term in root_terms):
        return "no", "no", "other"

    if any(term in text for term in yield_terms):
        if "yield" in text or "harvest" in text or "grain" in text or "fruit" in text or "seed" in text or "tuber" in text:
            return "yes", "yes", outcome_class
        return "partial", "partial", outcome_class

    return "no", "no", outcome_class


def classify_intervention(row_text: str) -> tuple[str, bool]:
    organic_terms = [
        "organic farming",
        "organic agriculture",
        "organic production",
        "organic management",
        "organic cultivation",
        "organic system",
        "organic crop",
        "organically grown",
        "organically managed",
        "organic-principles",
        "organic principles",
        "fym",
        "vermicompost",
        "green manure",
        "biodynamic",
        "natural farming",
    ]
    conventional_terms = [
        "conventional farming",
        "conventional agriculture",
        "conventional system",
        "conventional management",
        "high-input",
        "synthetic fertilizer",
        "synthetic fertilizers",
        "synthetic pesticide",
        "synthetic pesticides",
        "npk",
        "chemical fertilizer",
        "chemical fertilizers",
    ]
    organic = has_any(row_text, organic_terms)
    conventional = has_any(row_text, conventional_terms)
    if organic and not conventional:
        return "yes", False
    if organic and conventional:
        return "partial", False
    if conventional and not organic:
        return "no", False
    if has_any(row_text, ["low-input", "no synthetic", "without synthetic", "without chemical"]):
        return "partial", False
    return "no", False


def classify_comparator(row_text: str) -> str:
    conventional_terms = [
        "conventional farming",
        "conventional agriculture",
        "conventional system",
        "conventional management",
        "high-input",
        "synthetic fertilizer",
        "synthetic fertilizers",
        "synthetic pesticide",
        "synthetic pesticides",
        "npk",
        "chemical fertilizer",
        "chemical fertilizers",
    ]
    if has_any(row_text, conventional_terms):
        return "yes"
    if has_any(row_text, ["unamended", "no fertilizer", "control", "check", "untreated"]):
        return "no"
    return "partial"


def detect_swap(row_text: str) -> bool:
    organic_terms = [
        "organic farming",
        "organic agriculture",
        "organic production",
        "organic management",
        "organic cultivation",
        "organic system",
        "organically grown",
        "organically managed",
        "organic-principles",
        "organic principles",
        "fym",
        "vermicompost",
        "green manure",
        "biodynamic",
        "natural farming",
    ]
    conventional_terms = [
        "conventional farming",
        "conventional agriculture",
        "conventional system",
        "conventional management",
        "high-input",
        "synthetic fertilizer",
        "synthetic fertilizers",
        "synthetic pesticide",
        "synthetic pesticides",
        "npk",
        "chemical fertilizer",
        "chemical fertilizers",
    ]
    treatment = normalize_text(row_text.split(" | ")[2] if " | " in row_text else row_text)
    control = normalize_text(row_text.split(" | ")[3] if row_text.count(" | ") >= 3 else "")
    return has_any(treatment, conventional_terms) and has_any(control, organic_terms)


def confidence_flag(confidence: str) -> bool:
    return normalize_text(confidence) == "low"


def hard_unusable(row: dict) -> tuple[bool, str | None]:
    t_mean = row.get("treatment_mean")
    c_mean = row.get("control_mean")
    if pd.isna(t_mean) or pd.isna(c_mean):
        return True, "missing_mean"
    try:
        if float(t_mean) <= 0 or float(c_mean) <= 0:
            return True, "nonpositive_mean"
    except Exception:
        return True, "non_numeric_mean"
    return False, None


def adjudicate_row(config: dict, terms: dict, row: dict) -> dict:
    row_text = gather_text(row)
    unusable, reason = hard_unusable(row)
    outcome_match, estimand_match, outcome_class = classify_outcome(row_text)
    intervention_match, _ = classify_intervention(row_text)
    comparator_match = classify_comparator(row_text)
    swap = detect_swap(row_text)

    if unusable:
        return {
            "row_id": row["row_id"],
            "decision": "exclude",
            "intervention_match": "no",
            "comparator_match": "no",
            "outcome_match": "no",
            "estimand_match": "no",
            "needs_tc_swap": False,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": "other",
            "exclusion_reason": reason,
            "rationale_short": f"Row is structurally unusable: {reason}.",
        }

    if swap:
        return {
            "row_id": row["row_id"],
            "decision": "swap_treatment_control",
            "intervention_match": "no",
            "comparator_match": "no",
            "outcome_match": outcome_match,
            "estimand_match": estimand_match,
            "needs_tc_swap": True,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": "other",
            "exclusion_reason": None,
            "rationale_short": "Treatment and control appear reversed relative to the organic versus conventional config.",
        }

    if intervention_match == "no":
        return {
            "row_id": row["row_id"],
            "decision": "exclude",
            "intervention_match": intervention_match,
            "comparator_match": comparator_match,
            "outcome_match": outcome_match,
            "estimand_match": estimand_match,
            "needs_tc_swap": False,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": "other",
            "exclusion_reason": "intervention_mismatch",
            "rationale_short": "Treatment does not clearly match the configured organic intervention.",
        }

    if comparator_match == "no":
        return {
            "row_id": row["row_id"],
            "decision": "exclude",
            "intervention_match": intervention_match,
            "comparator_match": comparator_match,
            "outcome_match": outcome_match,
            "estimand_match": estimand_match,
            "needs_tc_swap": False,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": "other",
            "exclusion_reason": "comparator_mismatch",
            "rationale_short": "Control does not clearly match the configured conventional comparator.",
        }

    if outcome_match == "no":
        return {
            "row_id": row["row_id"],
            "decision": "exclude",
            "intervention_match": intervention_match,
            "comparator_match": comparator_match,
            "outcome_match": outcome_match,
            "estimand_match": estimand_match,
            "needs_tc_swap": False,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": outcome_class if outcome_class != "other" else "other",
            "exclusion_reason": "outcome_mismatch",
            "rationale_short": "Outcome does not match the configured crop-yield target.",
        }

    if estimand_match == "no":
        if outcome_class == "system_productivity":
            return {
                "row_id": row["row_id"],
                "decision": "flag",
                "intervention_match": intervention_match,
                "comparator_match": comparator_match,
                "outcome_match": outcome_match,
                "estimand_match": estimand_match,
                "needs_tc_swap": False,
                "normalized_outcome_class": outcome_class,
                "normalized_study_setting": "unknown",
                "normalized_estimand_class": outcome_class,
                "exclusion_reason": "estimand_mismatch_system_productivity",
                "rationale_short": "Outcome is yield-like but appears to measure system productivity rather than the crop-yield gap target.",
            }
        return {
            "row_id": row["row_id"],
            "decision": "exclude",
            "intervention_match": intervention_match,
            "comparator_match": comparator_match,
            "outcome_match": outcome_match,
            "estimand_match": estimand_match,
            "needs_tc_swap": False,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": outcome_class if outcome_class != "other" else "other",
            "exclusion_reason": "estimand_mismatch",
            "rationale_short": "Row measures a different target than the configured harvested-yield estimand.",
        }

    if confidence_flag(row.get("confidence", "")):
        return {
            "row_id": row["row_id"],
            "decision": "flag",
            "intervention_match": intervention_match,
            "comparator_match": comparator_match,
            "outcome_match": outcome_match,
            "estimand_match": estimand_match,
            "needs_tc_swap": False,
            "normalized_outcome_class": outcome_class,
            "normalized_study_setting": "unknown",
            "normalized_estimand_class": outcome_class if outcome_class != "other" else "other",
            "exclusion_reason": "low_confidence",
            "rationale_short": "Row matches the topic semantically but was labeled low confidence in extraction.",
        }

    return {
        "row_id": row["row_id"],
        "decision": "keep",
        "intervention_match": intervention_match,
        "comparator_match": comparator_match,
        "outcome_match": outcome_match,
        "estimand_match": estimand_match,
        "needs_tc_swap": False,
        "normalized_outcome_class": outcome_class,
        "normalized_study_setting": "unknown",
        "normalized_estimand_class": outcome_class if outcome_class != "other" else "other",
        "exclusion_reason": None,
        "rationale_short": "Row matches the configured organic versus conventional yield comparison closely enough for synthesis.",
    }


def compute_meta(df: pd.DataFrame) -> dict | None:
    effect_sizes = []
    variances = []
    for _, row in df.iterrows():
        try:
            t = float(row["treatment_mean"])
            c = float(row["control_mean"])
            if t <= 0 or c <= 0:
                continue
            lnrr = math.log(t / c)
        except Exception:
            continue
        sd_t = row.get("sd_treatment")
        sd_c = row.get("sd_control")
        n_t = row.get("treatment_n") or row.get("control_n")
        n_c = row.get("control_n") or row.get("treatment_n")
        if pd.isna(sd_t) and not pd.isna(row.get("se_treatment")) and pd.notna(n_t) and float(n_t) > 0:
            sd_t = float(row["se_treatment"]) * math.sqrt(float(n_t))
        if pd.isna(sd_c) and not pd.isna(row.get("se_control")) and pd.notna(n_c) and float(n_c) > 0:
            sd_c = float(row["se_control"]) * math.sqrt(float(n_c))
        if pd.isna(sd_t) and not pd.isna(row.get("variance_value")) and str(row.get("variance_type", "")).upper() == "LSD":
            lsd = float(row["variance_value"])
            n = float(n_t) if pd.notna(n_t) else 3.0
            df_val = 2 * (n - 1)
            if df_val > 0:
                t_crit = stats.t.ppf(0.975, df_val)
                se_diff = lsd / (t_crit * math.sqrt(2))
                sd_est = se_diff * math.sqrt(n)
                sd_t = sd_est
                sd_c = sd_est
        try:
            vals = [float(x) for x in (sd_t, sd_c, n_t, n_c, t, c)]
            if any(v <= 0 for v in vals):
                continue
            sd_t, sd_c, n_t, n_c, t, c = vals
            var = (sd_t**2 / (n_t * t**2)) + (sd_c**2 / (n_c * c**2))
            if var > 0:
                effect_sizes.append(lnrr)
                variances.append(var)
        except Exception:
            continue

    if len(effect_sizes) < 3:
        return None
    yi = np.array(effect_sizes, dtype=float)
    vi = np.array(variances, dtype=float)
    wi = 1.0 / vi
    sum_w = wi.sum()
    mu_fe = (wi * yi).sum() / sum_w
    q_stat = (wi * (yi - mu_fe) ** 2).sum()
    k = len(yi)
    df_q = k - 1
    c_val = sum_w - (wi**2).sum() / sum_w
    tau2 = max(0.0, (q_stat - df_q) / c_val) if c_val > 0 else 0.0
    wi_re = 1.0 / (vi + tau2)
    sum_w_re = wi_re.sum()
    mu_re = (wi_re * yi).sum() / sum_w_re
    se_re = 1.0 / math.sqrt(sum_w_re)
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    return {
        "k": int(k),
        "pooled_pct": (math.exp(mu_re) - 1.0) * 100.0,
        "ci_lo_pct": (math.exp(ci_lo) - 1.0) * 100.0,
        "ci_hi_pct": (math.exp(ci_hi) - 1.0) * 100.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config()
    terms = build_terms(config)
    batches = load_batches()

    decisions = []
    counts = Counter()
    reasons = Counter()
    kept_rows = []
    raw_validated_rows = []

    for batch in batches:
        for row in batch["rows"]:
            raw_validated_rows.append(row)
            decision = adjudicate_row(config, terms, row)
            decisions.append(decision)
            counts[decision["decision"]] += 1
            if decision["exclusion_reason"]:
                reasons[decision["exclusion_reason"]] += 1
            if decision["decision"] == "keep":
                kept_rows.append(row)

    decisions_path = OUT_DIR / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    kept_df = pd.DataFrame(kept_rows)
    kept_csv = OUT_DIR / "kept_rows.csv"
    kept_df.to_csv(kept_csv, index=False)

    validated_df = pd.DataFrame(raw_validated_rows)
    summary = {
        "topic": TOPIC,
        "rows_reviewed": len(decisions),
        "decision_counts": dict(counts),
        "major_exclusion_reasons": dict(reasons.most_common(12)),
        "kept_rows": int(len(kept_df)),
        "kept_papers": int(kept_df["paper_id"].nunique()) if len(kept_df) else 0,
        "validated_rows": int(len(validated_df)),
        "validated_papers": int(validated_df["paper_id"].nunique()) if len(validated_df) else 0,
        "validated_meta": compute_meta(validated_df),
        "kept_meta": compute_meta(kept_df),
        "benchmark_pct": config.get("benchmark", {}).get("published_pooled_effect", {}).get("estimate"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Organic Universal Adjudication",
        "",
        f"- Reviewed rows: {summary['rows_reviewed']}",
        f"- Decision counts: {summary['decision_counts']}",
        f"- Major exclusion reasons: {summary['major_exclusion_reasons']}",
        f"- Validated pooled: {summary['validated_meta']}",
        f"- Kept pooled: {summary['kept_meta']}",
        "",
        f"- Decisions: {decisions_path}",
        f"- Kept CSV: {kept_csv}",
        f"- Summary: {OUT_DIR / 'summary.json'}",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
