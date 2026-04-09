#!/usr/bin/env python3
"""
Universal adjudication pass for notill_tillage.

This consumes the prebuilt review batches in codex/outputs/validated_review_batches/
and writes row-level decisions under codex/outputs/codex_decisions/notill_tillage/.

The logic is deliberately universal and config-driven:
- keep rows that match the no-till vs conventional tillage contrast and report crop yield
- flag composite/system productivity rows
- exclude non-yield or off-target outcomes
- swap only if treatment/control are clearly reversed
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
CODEX_ROOT = Path(__file__).resolve().parent
BATCH_DIR = CODEX_ROOT / "outputs" / "validated_review_batches" / "notill_tillage"
OUT_DIR = CODEX_ROOT / "outputs" / "codex_decisions" / "notill_tillage"


NOTILL_PATTERNS = [
    r"\bno[- ]?till\b",
    r"\bno[- ]?tillage\b",
    r"\bzero[- ]?till\b",
    r"\bzero[- ]?tillage\b",
    r"\bdirect seeding\b",
    r"\bdirect sow(?:ing)?\b",
    r"\bdirect drilling\b",
    r"\bdirect drill\b",
    r"\bnt\b",
    r"\bzt\b",
]

CONV_PATTERNS = [
    r"\bconventional till(?:age)?\b",
    r"\bmoldboard\b",
    r"\bmouldboard\b",
    r"\bplow(?:ed|ing)?\b",
    r"\bplough(?:ed|ing)?\b",
    r"\bdisk\b",
    r"\btilled\b",
    r"\btraditional tillage\b",
    r"\binversion tillage\b",
]

EXCLUDE_OUTCOME_PATTERNS = [
    r"straw yield",
    r"biological yield",
    r"hectol(?:i|e)ter weight",
    r"1000 seed weight",
    r"thousand seed weight",
    r"mass of 100 seeds",
    r"mass of 100 seed",
    r"number of produced seeds",
    r"boll weight",
    r"bolls per plant",
    r"ginning out turn",
    r"soil organic carbon",
    r"nitrogen uptake",
    r"phosphorus uptake",
    r"protein",
    r"quality",
    r"plant height",
    r"leaf area",
    r"root length",
    r"stover yield",
]

FLAG_OUTCOME_PATTERNS = [
    r"system productivity",
    r"system yield",
    r"equivalent rice yield",
    r"maize equivalent yield",
    r"combined grain yield",
    r"\bery\b",
]

KEEP_OUTCOME_PATTERNS = [
    r"grain yield",
    r"crop grain yield",
    r"crop yield",
    r"seed yield",
]

AMBIGUOUS_TILLAGE_PATTERNS = [
    r"reduced tillage",
    r"minimum tillage",
    r"strip[- ]till",
    r"conservation agriculture",
]


def load_config() -> dict:
    return json.loads((ROOT / "notill_tillage" / "config.json").read_text(encoding="utf-8"))


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(BATCH_DIR.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    return rows


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
        "tc_confusion_warnings": config.get("tc_confusion_warnings", []),
        "benchmark_source": benchmark.get("source"),
        "benchmark_notes": benchmark.get("published_pooled_effect", {}).get("notes"),
    }


def text_blob(row: dict) -> str:
    return " | ".join(
        str(row.get(key, "")).lower()
        for key in [
            "outcome",
            "outcome_unit",
            "treatment_description",
            "control_description",
            "title",
            "notes",
        ]
    )


def has_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify_outcome(text: str) -> tuple[str, str, str]:
    if has_any(EXCLUDE_OUTCOME_PATTERNS, text):
        return "exclude", "no", "off-target outcome"
    if has_any(FLAG_OUTCOME_PATTERNS, text):
        return "flag", "partial", "composite productivity metric"
    if has_any(KEEP_OUTCOME_PATTERNS, text) or "yield" in text or "productivity" in text:
        return "keep", "yes", ""
    return "exclude", "no", "outcome does not match crop yield target"


def classify_tillage(row: dict) -> tuple[str, str, str, bool]:
    treatment = str(row.get("treatment_description", "")).lower()
    control = str(row.get("control_description", "")).lower()

    treatment_notill = has_any(NOTILL_PATTERNS, treatment)
    control_notill = has_any(NOTILL_PATTERNS, control)
    treatment_conv = has_any(CONV_PATTERNS, treatment)
    control_conv = has_any(CONV_PATTERNS, control)
    treatment_ambiguous = has_any(AMBIGUOUS_TILLAGE_PATTERNS, treatment)

    if treatment_conv and control_notill and not treatment_notill:
        return "swap_treatment_control", "yes", "treatment/control reversed relative to config", True

    if not treatment_notill:
        if treatment_ambiguous:
            return "exclude", "no", "treatment is reduced/minimum/strip/conservation tillage, not strict no-till", False
        return "exclude", "no", "treatment does not clearly match no-till", False

    if not control_conv:
        if control_notill:
            return "exclude", "no", "control is also no-till, not conventional tillage", False
        return "flag", "partial", "control is not clearly conventional tillage", False

    if treatment_ambiguous and not treatment_notill:
        return "exclude", "no", "ambiguous tillage definition", False

    return "keep", "yes", "", False


def adjudicate_row(row: dict) -> dict:
    blob = text_blob(row)
    decision, outcome_match, outcome_reason = classify_outcome(blob)
    tillage_decision, intervention_match, tillage_reason, needs_swap = classify_tillage(row)

    # If treatment/control are reversed, that takes precedence.
    if tillage_decision == "swap_treatment_control":
        return {
            "row_id": row["row_id"],
            "decision": "swap_treatment_control",
            "intervention_match": "yes",
            "comparator_match": "yes",
            "outcome_match": outcome_match,
            "estimand_match": "partial" if decision != "exclude" else "no",
            "needs_tc_swap": True,
            "normalized_outcome_class": "grain_yield" if "grain yield" in blob or "crop yield" in blob else "yield",
            "normalized_study_setting": "field" if "greenhouse" not in blob and "pot" not in blob else "mixed",
            "normalized_estimand_class": "crop_yield",
            "exclusion_reason": None,
            "rationale_short": "Treatment and control are reversed relative to the no-till vs conventional tillage config.",
        }

    if tillage_decision == "exclude" or decision == "exclude":
        reason = tillage_reason if tillage_decision == "exclude" else outcome_reason
        return {
            "row_id": row["row_id"],
            "decision": "exclude",
            "intervention_match": intervention_match if tillage_decision != "exclude" else "no",
            "comparator_match": "yes" if "conventional" in str(row.get("control_description", "")).lower() else "partial",
            "outcome_match": outcome_match,
            "estimand_match": "no",
            "needs_tc_swap": False,
            "normalized_outcome_class": "other",
            "normalized_study_setting": "field" if "greenhouse" not in blob and "pot" not in blob else "mixed",
            "normalized_estimand_class": "other",
            "exclusion_reason": reason,
            "rationale_short": reason,
        }

    if tillage_decision == "flag" or decision == "flag":
        reason = tillage_reason or outcome_reason
        return {
            "row_id": row["row_id"],
            "decision": "flag",
            "intervention_match": intervention_match,
            "comparator_match": "yes" if "conventional" in str(row.get("control_description", "")).lower() else "partial",
            "outcome_match": outcome_match,
            "estimand_match": "partial",
            "needs_tc_swap": False,
            "normalized_outcome_class": "composite_yield" if "system" in blob or "equivalent" in blob else "yield",
            "normalized_study_setting": "field" if "greenhouse" not in blob and "pot" not in blob else "mixed",
            "normalized_estimand_class": "system_productivity" if "system" in blob or "equivalent" in blob else "crop_yield",
            "exclusion_reason": reason,
            "rationale_short": reason or "ambiguous but potentially relevant row",
        }

    # Keep only strict crop-yield rows with clear no-till and conventional comparator.
    return {
        "row_id": row["row_id"],
        "decision": "keep",
        "intervention_match": "yes",
        "comparator_match": "yes",
        "outcome_match": outcome_match,
        "estimand_match": "yes" if outcome_match == "yes" else "partial",
        "needs_tc_swap": False,
        "normalized_outcome_class": "grain_yield" if "grain yield" in blob else "yield",
        "normalized_study_setting": "field" if "greenhouse" not in blob and "pot" not in blob else "mixed",
        "normalized_estimand_class": "crop_yield",
        "exclusion_reason": None,
        "rationale_short": "Row matches strict no-till vs conventional tillage and reports a crop-yield endpoint.",
    }


def compute_meta_from_rows(rows: list[dict]) -> dict | None:
    import math
    import numpy as np
    from scipy import stats

    yi = []
    vi = []
    for row in rows:
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
            n_val = float(n_t) if pd.notna(n_t) else 3.0
            df_val = 2 * (n_val - 1)
            if df_val > 0:
                t_crit = stats.t.ppf(0.975, df_val)
                se_diff = lsd / (t_crit * math.sqrt(2))
                sd_est = se_diff * math.sqrt(n_val)
                sd_t = sd_est
                sd_c = sd_est

        try:
            vals = [float(x) for x in (sd_t, sd_c, n_t, n_c, t, c)]
            if any(v <= 0 for v in vals):
                continue
            sd_t, sd_c, n_t, n_c, t, c = vals
            vr = (sd_t**2 / (n_t * t**2)) + (sd_c**2 / (n_c * c**2))
        except Exception:
            continue

        if vr > 0:
            yi.append(lnrr)
            vi.append(vr)

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
        "pooled_pct": float((math.exp(mu_re) - 1.0) * 100.0),
        "ci_lo_pct": float((math.exp(ci_lo) - 1.0) * 100.0),
        "ci_hi_pct": float((math.exp(ci_hi) - 1.0) * 100.0),
        "I2": float(max(0.0, (q_stat - df_q) / q_stat * 100.0) if q_stat > 0 else 0.0),
    }


def main() -> None:
    config = load_config()
    brief = topic_brief(config)
    rows = load_rows()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    decisions = []
    counts = Counter()
    reasons = Counter()
    kept_rows = []

    for row in rows:
        decision = adjudicate_row(row)
        decisions.append(decision)
        counts[decision["decision"]] += 1
        if decision.get("exclusion_reason"):
            reasons[decision["exclusion_reason"]] += 1
        if decision["decision"] in {"keep", "swap_treatment_control"}:
            kept_rows.append(row)

    decisions_path = OUT_DIR / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for item in decisions:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "topic": "notill_tillage",
        "topic_brief": brief,
        "total_rows": len(rows),
        "decision_counts": dict(counts),
        "major_exclusion_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reasons.most_common(10)
        ],
        "kept_rows_for_synthesis": len(kept_rows),
        "swapped_rows": counts.get("swap_treatment_control", 0),
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Optional helper for follow-up synthesis.
    pd_rows = []
    for row in kept_rows:
        # Preserve the original row payload for downstream synthesis work.
        pd_rows.append(row)
    if pd_rows:
        try:
            pd.DataFrame(pd_rows).to_csv(OUT_DIR / "strict_kept_rows.csv", index=False)
        except Exception:
            pass

    strict_meta = compute_meta_from_rows(kept_rows)
    if strict_meta is not None:
        strict_meta.update(
            {
                "benchmark_pct": -5.7,
                "benchmark_source": "Pittelkow et al. 2015",
                "abs_diff_vs_benchmark": abs(strict_meta["pooled_pct"] - (-5.7)),
            }
        )
        (OUT_DIR / "strict_synthesis.json").write_text(
            json.dumps(strict_meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
