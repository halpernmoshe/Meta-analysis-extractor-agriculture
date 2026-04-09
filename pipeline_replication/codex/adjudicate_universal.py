#!/usr/bin/env python3
"""
Universal config-driven semantic adjudication for all pipeline replication topics.

Reads pre-built JSONL inputs from codex/outputs/universal_llm_inputs/{topic}/
and writes decisions to codex/outputs/codex_decisions/{topic}/.

The logic is deliberately universal and config-driven:
- uses topic_brief from the JSONL to determine intervention/comparator/outcome matching
- keyword-based matching with topic-specific term lists from config
- deterministic hard checks (missing means, non-positive means)
- writes keep/exclude/flag/swap decisions

This is the deterministic baseline. The intended production version will use
Claude Opus 4.6 as the semantic adjudicator, but this provides a strong
keyword-based baseline for all 6 topics.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CODEX_ROOT = Path(__file__).resolve().parent
ROOT = CODEX_ROOT.parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "universal_llm_inputs"
OUT_ROOT = CODEX_ROOT / "outputs" / "codex_decisions"


# ── Topic-specific term banks ──────────────────────────────────────────────

# Hard-exclude outcome patterns (always off-target regardless of topic)
UNIVERSAL_HARD_EXCLUDE_OUTCOMES = [
    r"colonization",
    r"infection rate",
    r"spore density",
    r"spore count",
    r"photosynthesis rate",
    r"chlorophyll content",
    r"leaf area index",
    r"plant height$",
    r"root length$",
    r"stomatal conductance",
    r"transpiration rate",
    r"water use efficiency",
    r"\bwue\b",
    r"soil organic carbon",
    r"soil respiration",
    r"microbial biomass",
    r"enzyme activity",
    r"nitrogen uptake",
    r"phosphorus uptake",
    r"potassium uptake",
]

# Yield-related keywords (universally positive for crop-yield topics)
YIELD_TERMS = [
    r"\byield\b",
    r"\bgrain yield\b",
    r"\bcrop yield\b",
    r"\bseed yield\b",
    r"\bfruit yield\b",
    r"\btuber yield\b",
    r"\bpod yield\b",
    r"\bpod weight\b",
    r"\bharvest\b",
    r"\bproductivity\b",
    r"\bkg[/ ]ha\b",
    r"\bMg[/ ]ha\b",
    r"\bt[/ ]ha\b",
    r"\bg[/ ]plant\b",
    r"\bg[/ ]pot\b",
]

# Biomass terms (partial match for yield topics)
BIOMASS_TERMS = [
    r"\bbiomass\b",
    r"\bdry weight\b",
    r"\bdry matter\b",
    r"\bfresh weight\b",
    r"\bshoot biomass\b",
    r"\bplant biomass\b",
    r"\btotal biomass\b",
    r"\bshoot dry weight\b",
    r"\baboveground biomass\b",
]

# Quality/concentration traits (usually off-target for yield topics)
QUALITY_EXCLUDE = [
    r"\bprotein\b",
    r"\bconcentration\b",
    r"\bcontent\b",
    r"\bquality\b",
    r"\bhectolitre weight\b",
    r"\btest weight\b",
    r"\bflour\b",
    r"\benergy\b",
    r"\bcolor\b",
    r"\bbrightness\b",
    r"\bcu\b",
    r"\bfe\b",
    r"\bzn\b",
    r"\bmn\b",
]

# Root/belowground (off-target for above-ground yield topics)
ROOT_EXCLUDE = [
    r"\broot biomass\b",
    r"\broot dry weight\b",
    r"\broot length\b",
    r"\broot volume\b",
    r"\bbelowground biomass\b",
]

# System-level metrics (flag, not exclude — may be benchmark-aligned for some topics)
SYSTEM_TERMS = [
    r"\bsystem productivity\b",
    r"\bequivalent yield\b",
    r"\bland equivalent ratio\b",
    r"\bler\b",
    r"\bcombined grain yield\b",
]


# ── Per-topic configuration ────────────────────────────────────────────────

TOPIC_RULES = {
    "organic_yield_gap": {
        "intervention_patterns": [
            r"\borganic\b", r"\borganically\b", r"\bbiodynamic\b",
            r"\bnatural farming\b", r"\borganic farming\b",
            r"\borganic agriculture\b", r"\borganic system\b",
            r"\borganic management\b", r"\borganic production\b",
            r"\bfym\b", r"\bvermicompost\b", r"\bgreen manure\b",
        ],
        "comparator_patterns": [
            r"\bconventional\b", r"\bhigh[- ]input\b",
            r"\bsynthetic fertilizer", r"\bchemical fertilizer",
            r"\bnpk\b", r"\bsynthetic pesticide",
        ],
        "exclude_outcomes": ROOT_EXCLUDE + QUALITY_EXCLUDE + [
            r"\bstraw yield\b",
        ],
        "flag_outcomes": SYSTEM_TERMS,
        "keep_outcomes": YIELD_TERMS + BIOMASS_TERMS,
        "allow_biomass": True,
        "benchmark_pct": -19.2,
        "benchmark_source": "Ponisio et al. 2015",
        "swap_detection": "intervention_in_control",
    },
    "notill_tillage": {
        "intervention_patterns": [
            r"\bno[- ]?till\b", r"\bno[- ]?tillage\b",
            r"\bzero[- ]?till\b", r"\bzero[- ]?tillage\b",
            r"\bdirect seeding\b", r"\bdirect sow", r"\bdirect drill",
            r"\bnt\b", r"\bzt\b",
        ],
        "comparator_patterns": [
            r"\bconventional till", r"\bmoldboard\b", r"\bmouldboard\b",
            r"\bplow", r"\bplough", r"\bdisk\b", r"\btilled\b",
            r"\btraditional tillage\b", r"\binversion tillage\b",
        ],
        "exclude_outcomes": ROOT_EXCLUDE + QUALITY_EXCLUDE + [
            r"\bstraw yield\b", r"\bbiological yield\b",
            r"\bstover yield\b", r"\bboll weight\b", r"\bbolls per plant\b",
            r"\bginning out turn\b",
        ],
        "flag_outcomes": [
            r"\bsystem productivity\b", r"\bequivalent.*yield\b",
        ],
        "keep_outcomes": YIELD_TERMS,
        "allow_biomass": False,
        "benchmark_pct": -5.7,
        "benchmark_source": "Pittelkow et al. 2015",
        "swap_detection": "intervention_in_control",
        # Exclude ambiguous tillage (reduced/minimum/strip/conservation)
        "extra_exclude_check": "notill_ambiguous_check",
    },
    "mycorrhiza_yield": {
        "intervention_patterns": [
            r"\bamf\b", r"\bmycorrhiz", r"\bglomus\b", r"\brhizophagus\b",
            r"\bfunneliformis\b", r"\bclaroideoglomus\b", r"\binocul",
            r"\bvam\b", r"\bam fungi\b",
        ],
        "comparator_patterns": [
            r"\bnon[- ]?mycorrhiz", r"\buninoculated\b", r"\bwithout amf\b",
            r"\b-amf\b", r"\bnm\b", r"\bno amf\b", r"\bnon[- ]?inoculated\b",
            r"\bcontrol\b", r"\bsterilized\b",
        ],
        "exclude_outcomes": ROOT_EXCLUDE + [
            r"\bcolonization\b", r"\binfection\b", r"\bspore\b",
            r"\bphotosynthesis\b", r"\bstomatal\b", r"\btranspiration\b",
            r"\bleaf area\b", r"\bplant height\b",
        ],
        "flag_outcomes": [],
        "keep_outcomes": YIELD_TERMS + BIOMASS_TERMS,
        "allow_biomass": True,  # mycorrhiza benchmark uses biomass
        "benchmark_pct": 23.0,
        "benchmark_source": "Hoeksema et al. 2010",
        "swap_detection": "intervention_in_control",
    },
    "legume_rotation": {
        "intervention_patterns": [
            r"\blegume\b", r"\bsoybean\b", r"\bpea\b", r"\blentil\b",
            r"\bchickpea\b", r"\bfaba bean\b", r"\bclover\b", r"\blupin\b",
            r"\bbean\b", r"\bgroundnut\b", r"\bcowpea\b", r"\bpigeon ?pea\b",
            r"\bmedic\b", r"\bvetch\b", r"\bgreen manure\b",
            r"\bn[2] ?fix", r"\bpre[- ]?crop\b", r"\bpreceding crop\b",
            r"\brotation\b", r"\bafter\b",
        ],
        "comparator_patterns": [
            r"\bcontinuous\b", r"\bmonoculture\b", r"\bmonocrop\b",
            r"\bnon[- ]?legume\b", r"\bcereal[- ]cereal\b", r"\bfallow\b",
            r"\bcontinuous wheat\b", r"\bcontinuous maize\b",
            r"\bwheat after wheat\b", r"\bmaize after maize\b",
            r"\bcontrol\b",
        ],
        "exclude_outcomes": ROOT_EXCLUDE + [
            r"\bstraw yield\b", r"\bbiological yield\b",
            r"\bnodule\b", r"\bnodulation\b",
        ],
        "flag_outcomes": SYSTEM_TERMS,
        "keep_outcomes": YIELD_TERMS + BIOMASS_TERMS,
        "allow_biomass": True,  # Some legume rotation papers report biomass
        "benchmark_pct": 20.0,
        "benchmark_source": "Zhao et al. 2022",
        "swap_detection": "intervention_in_control",
        # Removed legume_yield_check - too aggressive, PICO validation already handles this
    },
    "biochar_crop_yield": {
        "intervention_patterns": [
            r"\bbiochar\b", r"\bbio[- ]?char\b", r"\bcharcoal\b",
            r"\bpyrolysis char\b", r"\bbiomass char\b",
        ],
        "comparator_patterns": [
            r"\bno biochar\b", r"\bunamended\b", r"\bwithout biochar\b",
            r"\bno char\b", r"\bcontrol\b", r"\b0 ?t[/ ]ha\b",
        ],
        "exclude_outcomes": ROOT_EXCLUDE + QUALITY_EXCLUDE + [
            r"\bstraw yield\b", r"\bbiological yield\b",
        ],
        "flag_outcomes": [],
        "keep_outcomes": YIELD_TERMS + BIOMASS_TERMS,
        "allow_biomass": True,
        "benchmark_pct": 16.0,
        "benchmark_source": "Ye et al. 2020",
        "swap_detection": "intervention_in_control",
    },
    "intercropping_yield": {
        "intervention_patterns": [
            r"\bintercrop", r"\binter[- ]?crop", r"\bmixed cropping\b",
            r"\bstrip intercrop", r"\brelay intercrop", r"\bcompanion crop",
            r"\bpolyculture\b", r"\bcrop mixture\b",
            r"\bcereal[- ]legume\b", r"\bmaize[- ]bean\b",
            r"\badditive intercrop", r"\breplacement intercrop",
        ],
        "comparator_patterns": [
            r"\bsole crop", r"\bmonoculture\b", r"\bmonocrop\b",
            r"\bpure stand\b", r"\bsingle crop\b",
        ],
        "exclude_outcomes": ROOT_EXCLUDE + QUALITY_EXCLUDE,
        "flag_outcomes": [],
        "keep_outcomes": YIELD_TERMS + BIOMASS_TERMS + SYSTEM_TERMS,
        "allow_biomass": True,
        "benchmark_pct": 22.0,
        "benchmark_source": "Yu et al. 2015",
        "swap_detection": "intervention_in_control",
        # Special: LER is the primary benchmark estimand
        "primary_estimand": "LER",
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def normalize_text(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower()) if value is not None else ""


def gather_text(row: dict) -> str:
    parts = [
        row.get("outcome_variable", ""),
        row.get("outcome_unit", ""),
        row.get("treatment_description", ""),
        row.get("control_description", ""),
        row.get("title", ""),
        row.get("notes", ""),
    ]
    return " | ".join(normalize_text(p) for p in parts)


def has_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify_study_setting(text: str) -> str:
    if re.search(r"\bgreenhouse\b|\bglass\s*house\b|\bgrowth chamber\b", text, re.IGNORECASE):
        return "greenhouse"
    if re.search(r"\bpot\b|\bpotted\b|\bcontainer\b", text, re.IGNORECASE):
        return "pot"
    if re.search(r"\bfield\b|\bfarm\b|\bstation\b|\bsite\b|\blocati", text, re.IGNORECASE):
        return "field"
    return "unknown"


def classify_outcome_class(text: str) -> str:
    if re.search(r"\bgrain yield\b", text, re.IGNORECASE):
        return "grain_yield"
    if re.search(r"\bfruit yield\b|\bpod yield\b|\btuber yield\b", text, re.IGNORECASE):
        return "harvest_yield"
    if re.search(r"\bland equivalent ratio\b|\bler\b", text, re.IGNORECASE):
        return "system_productivity"
    if re.search(r"\bequivalent yield\b|\bsystem productivity\b", text, re.IGNORECASE):
        return "system_productivity"
    if has_any_pattern(text, BIOMASS_TERMS):
        return "biomass"
    if has_any_pattern(text, QUALITY_EXCLUDE):
        return "quality_trait"
    if re.search(r"\byield\b", text, re.IGNORECASE):
        return "harvest_yield"
    return "other"


# ── Legume-specific check ──────────────────────────────────────────────────

def is_notill_ambiguous(row: dict) -> bool:
    """Check if treatment is reduced/minimum/strip/conservation tillage, not strict no-till."""
    treatment = normalize_text(row.get("treatment_description", ""))
    notill_patterns = [
        r"\bno[- ]?till", r"\bzero[- ]?till", r"\bdirect seeding\b",
        r"\bdirect sow", r"\bdirect drill", r"\bnt\b", r"\bzt\b",
    ]
    ambiguous_patterns = [
        r"\breduced tillage\b", r"\bminimum tillage\b",
        r"\bstrip[- ]?till\b", r"\bconservation agriculture\b",
    ]
    # If treatment has a clear no-till term, it's fine
    if has_any_pattern(treatment, notill_patterns):
        return False
    # If treatment only has ambiguous terms, exclude
    if has_any_pattern(treatment, ambiguous_patterns):
        return True
    return False


def is_legume_yield_row(row: dict) -> bool:
    """Check if this row reports the LEGUME crop yield rather than subsequent crop yield."""
    outcome = normalize_text(row.get("outcome_variable", ""))
    treatment_desc = normalize_text(row.get("treatment_description", ""))

    legume_names = [
        "soybean", "pea", "lentil", "chickpea", "faba bean", "clover",
        "lupin", "bean", "groundnut", "cowpea", "pigeon pea", "mung bean",
        "medic", "vetch",
    ]

    # If the outcome explicitly mentions a legume crop name
    for name in legume_names:
        if name in outcome:
            # Check if it's "yield of soybean" vs "yield after soybean"
            if "after" not in outcome and "following" not in outcome:
                return True

    return False


# ── Core adjudication ──────────────────────────────────────────────────────

def adjudicate_row(topic: str, rules: dict, row: dict) -> dict:
    """Adjudicate a single extracted row against topic rules."""
    text = gather_text(row)
    row_id = row.get("row_id", "unknown")

    # Hard structural checks
    t_mean = row.get("treatment_mean")
    c_mean = row.get("control_mean")
    try:
        t_val = float(t_mean)
        c_val = float(c_mean)
    except (TypeError, ValueError):
        return make_decision(row_id, "exclude", text, reason="missing_or_non_numeric_mean",
                           rationale="Row has missing or non-numeric treatment/control means.")

    if t_val <= 0 or c_val <= 0:
        return make_decision(row_id, "exclude", text, reason="nonpositive_mean",
                           rationale="Treatment or control mean is non-positive.")

    # Outcome classification
    if has_any_pattern(text, UNIVERSAL_HARD_EXCLUDE_OUTCOMES):
        return make_decision(row_id, "exclude", text,
                           reason="universal_hard_exclude_outcome",
                           rationale="Outcome matches universally off-target pattern.")

    if has_any_pattern(text, rules.get("exclude_outcomes", [])):
        return make_decision(row_id, "exclude", text,
                           reason="topic_exclude_outcome",
                           rationale="Outcome does not match configured primary outcome target.")

    # Intervention match
    treatment_text = normalize_text(row.get("treatment_description", ""))
    control_text = normalize_text(row.get("control_description", ""))

    intervention_in_treatment = has_any_pattern(treatment_text, rules["intervention_patterns"])
    intervention_in_control = has_any_pattern(control_text, rules["intervention_patterns"])
    comparator_in_control = has_any_pattern(control_text, rules["comparator_patterns"])
    comparator_in_treatment = has_any_pattern(treatment_text, rules["comparator_patterns"])

    # Also check title/outcome for context
    full_text_intervention = has_any_pattern(text, rules["intervention_patterns"])

    # Swap detection
    needs_swap = False
    if not intervention_in_treatment and intervention_in_control:
        if comparator_in_treatment:
            needs_swap = True

    if needs_swap:
        return make_decision(row_id, "swap_treatment_control", text,
                           intervention_match="yes", comparator_match="yes",
                           rationale="Treatment and control appear reversed relative to config.")

    # Intervention match scoring
    if intervention_in_treatment:
        intervention_match = "yes"
    elif full_text_intervention:
        intervention_match = "partial"
    else:
        return make_decision(row_id, "exclude", text,
                           reason="intervention_mismatch",
                           rationale="Treatment does not match configured intervention.")

    # Comparator match scoring
    if comparator_in_control:
        comparator_match = "yes"
    elif has_any_pattern(control_text, [r"\bcontrol\b", r"\bck\b", r"\bcheck\b"]):
        comparator_match = "partial"
    else:
        comparator_match = "partial"  # Be lenient on comparator

    # Outcome match
    if has_any_pattern(text, rules.get("flag_outcomes", [])):
        return make_decision(row_id, "flag", text,
                           intervention_match=intervention_match,
                           comparator_match=comparator_match,
                           outcome_match="partial",
                           estimand_match="partial",
                           reason="flagged_outcome",
                           rationale="Outcome matches a flagged pattern (composite/system metric).")

    if has_any_pattern(text, rules.get("keep_outcomes", [])):
        outcome_match = "yes"
    elif rules.get("allow_biomass") and has_any_pattern(text, BIOMASS_TERMS):
        outcome_match = "partial"
    else:
        return make_decision(row_id, "exclude", text,
                           intervention_match=intervention_match,
                           comparator_match=comparator_match,
                           reason="outcome_mismatch",
                           rationale="Outcome does not match configured yield/biomass target.")

    # Topic-specific extra checks
    extra_check = rules.get("extra_exclude_check")
    if extra_check == "legume_yield_check" and is_legume_yield_row(row):
        return make_decision(row_id, "exclude", text,
                           intervention_match=intervention_match,
                           comparator_match=comparator_match,
                           outcome_match=outcome_match,
                           reason="legume_yield_not_subsequent",
                           rationale="Row reports legume crop yield, not subsequent crop yield after legume rotation.")
    if extra_check == "notill_ambiguous_check" and is_notill_ambiguous(row):
        return make_decision(row_id, "exclude", text,
                           intervention_match="no",
                           comparator_match=comparator_match,
                           outcome_match=outcome_match,
                           reason="ambiguous_tillage",
                           rationale="Treatment is reduced/minimum/strip/conservation tillage, not strict no-till.")

    # Confidence check
    confidence = normalize_text(row.get("confidence", ""))
    if confidence == "low":
        return make_decision(row_id, "flag", text,
                           intervention_match=intervention_match,
                           comparator_match=comparator_match,
                           outcome_match=outcome_match,
                           estimand_match="partial",
                           reason="low_confidence",
                           rationale="Row matches topic but was labeled low confidence.")

    # All checks passed → keep
    estimand_match = "yes"
    if topic == "intercropping_yield":
        # For intercropping, LER is the primary estimand
        if has_any_pattern(text, [r"\bler\b", r"\bland equivalent ratio\b"]):
            estimand_match = "yes"
        else:
            estimand_match = "partial"  # Component yield, not LER

    return make_decision(row_id, "keep", text,
                        intervention_match=intervention_match,
                        comparator_match=comparator_match,
                        outcome_match=outcome_match,
                        estimand_match=estimand_match,
                        rationale=f"Row matches configured {topic} comparison for synthesis.")


def make_decision(row_id: str, decision: str, text: str,
                  intervention_match: str = "no",
                  comparator_match: str = "no",
                  outcome_match: str = "no",
                  estimand_match: str = "no",
                  reason: str | None = None,
                  rationale: str = "") -> dict:
    return {
        "row_id": row_id,
        "decision": decision,
        "intervention_match": intervention_match,
        "comparator_match": comparator_match,
        "outcome_match": outcome_match,
        "estimand_match": estimand_match,
        "needs_tc_swap": decision == "swap_treatment_control",
        "normalized_outcome_class": classify_outcome_class(text),
        "normalized_study_setting": classify_study_setting(text),
        "normalized_estimand_class": classify_outcome_class(text),
        "exclusion_reason": reason,
        "rationale_short": rationale,
    }


# ── Meta-analysis computation ──────────────────────────────────────────────

def compute_meta(rows: list[dict]) -> dict | None:
    """DerSimonian-Laird random-effects meta-analysis on lnRR."""
    yi_list = []
    vi_list = []

    for row in rows:
        try:
            t = float(row["treatment_mean"])
            c = float(row["control_mean"])
            if t <= 0 or c <= 0:
                continue
            lnrr = math.log(t / c)
        except (TypeError, ValueError, KeyError):
            continue

        # Try to get variance
        sd_t = row.get("sd_treatment")
        sd_c = row.get("sd_control")
        n_t = row.get("treatment_n") or row.get("control_n")
        n_c = row.get("control_n") or row.get("treatment_n")

        # SE → SD conversion
        if _is_missing(sd_t) and not _is_missing(row.get("se_treatment")) and _pos(n_t):
            sd_t = float(row["se_treatment"]) * math.sqrt(float(n_t))
        if _is_missing(sd_c) and not _is_missing(row.get("se_control")) and _pos(n_c):
            sd_c = float(row["se_control"]) * math.sqrt(float(n_c))

        # LSD → SD conversion
        if _is_missing(sd_t) and not _is_missing(row.get("variance_value")):
            vtype = str(row.get("variance_type", "")).upper()
            if vtype == "LSD" and _pos(n_t):
                lsd = float(row["variance_value"])
                n_val = float(n_t)
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
            vr = (sd_t ** 2 / (n_t * t ** 2)) + (sd_c ** 2 / (n_c * c ** 2))
            if vr > 0:
                yi_list.append(lnrr)
                vi_list.append(vr)
        except (TypeError, ValueError):
            continue

    if len(yi_list) < 3:
        return None

    yi = np.array(yi_list, dtype=float)
    vi = np.array(vi_list, dtype=float)
    wi = 1.0 / vi
    sum_w = wi.sum()
    mu_fe = (wi * yi).sum() / sum_w
    q_stat = (wi * (yi - mu_fe) ** 2).sum()
    k = len(yi)
    df_q = k - 1
    c_val = sum_w - (wi ** 2).sum() / sum_w
    tau2 = max(0.0, (q_stat - df_q) / c_val) if c_val > 0 else 0.0
    wi_re = 1.0 / (vi + tau2)
    sum_w_re = wi_re.sum()
    mu_re = (wi_re * yi).sum() / sum_w_re
    se_re = 1.0 / math.sqrt(sum_w_re)
    ci_lo = mu_re - 1.96 * se_re
    ci_hi = mu_re + 1.96 * se_re
    i2 = max(0.0, (q_stat - df_q) / q_stat * 100.0) if q_stat > 0 else 0.0

    return {
        "k": int(k),
        "pooled_pct": round((math.exp(mu_re) - 1.0) * 100.0, 2),
        "ci_lo_pct": round((math.exp(ci_lo) - 1.0) * 100.0, 2),
        "ci_hi_pct": round((math.exp(ci_hi) - 1.0) * 100.0, 2),
        "I2": round(i2, 1),
        "tau2": round(tau2, 6),
    }


def _is_missing(val) -> bool:
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (TypeError, ValueError):
        return False


def _pos(val) -> bool:
    try:
        return float(val) > 0
    except (TypeError, ValueError):
        return False


# ── Main runner ────────────────────────────────────────────────────────────

def process_topic(topic: str) -> dict:
    """Process a single topic and write results."""
    rules = TOPIC_RULES[topic]
    input_dir = INPUT_ROOT / topic
    out_dir = OUT_ROOT / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load JSONL rows
    jsonl_path = input_dir / "llm_review_inputs.jsonl"
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                rows.append(entry["row"])

    # Adjudicate each row
    decisions = []
    counts = Counter()
    reasons = Counter()
    kept_rows = []
    all_rows = []

    for row in rows:
        all_rows.append(row)
        decision = adjudicate_row(topic, rules, row)
        decisions.append(decision)
        counts[decision["decision"]] += 1
        if decision["exclusion_reason"]:
            reasons[decision["exclusion_reason"]] += 1
        if decision["decision"] in ("keep", "swap_treatment_control"):
            kept_rows.append(row)

    # Write decisions JSONL
    decisions_path = out_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Write kept rows CSV
    if kept_rows:
        pd.DataFrame(kept_rows).to_csv(out_dir / "strict_kept_rows.csv", index=False)

    # Compute meta-analysis
    all_meta = compute_meta(all_rows)
    kept_meta = compute_meta(kept_rows)

    # Synthesis comparison
    benchmark_pct = rules.get("benchmark_pct")
    synthesis = None
    if kept_meta:
        synthesis = dict(kept_meta)
        synthesis["benchmark_pct"] = benchmark_pct
        synthesis["benchmark_source"] = rules.get("benchmark_source")
        synthesis["abs_diff_vs_benchmark"] = round(
            abs(kept_meta["pooled_pct"] - benchmark_pct), 2
        ) if benchmark_pct is not None else None
        (out_dir / "strict_synthesis.json").write_text(
            json.dumps(synthesis, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # Summary
    summary = {
        "topic": topic,
        "total_rows": len(rows),
        "decision_counts": dict(counts),
        "major_exclusion_reasons": [
            {"reason": r, "count": c} for r, c in reasons.most_common(10)
        ],
        "kept_rows": len(kept_rows),
        "kept_papers": len(set(r.get("paper_id", "") for r in kept_rows)),
        "all_meta": all_meta,
        "kept_meta": kept_meta,
        "benchmark_pct": benchmark_pct,
        "benchmark_source": rules.get("benchmark_source"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown report
    report = [
        f"# {topic} — Universal Adjudication Results",
        "",
        f"- Total rows reviewed: {len(rows)}",
        f"- Decision counts: {dict(counts)}",
        f"- Kept for synthesis: {len(kept_rows)} ({len(kept_rows)/max(len(rows),1)*100:.1f}%)",
        f"- Kept papers: {summary['kept_papers']}",
        "",
        f"## Before Adjudication (all validated rows)",
        f"- {all_meta}" if all_meta else "- Not enough data for DL RE",
        "",
        f"## After Adjudication (kept rows only)",
        f"- {kept_meta}" if kept_meta else "- Not enough data for DL RE",
        "",
        f"## Benchmark Comparison",
        f"- Benchmark: {benchmark_pct}% ({rules.get('benchmark_source')})",
    ]
    if kept_meta and benchmark_pct is not None:
        diff = kept_meta["pooled_pct"] - benchmark_pct
        report.append(f"- Pipeline: {kept_meta['pooled_pct']:.2f}%")
        report.append(f"- Difference: {diff:+.2f} pp")
        report.append(f"- Direction match: {'YES' if (kept_meta['pooled_pct'] > 0) == (benchmark_pct > 0) else 'NO'}")

    report.extend([
        "",
        "## Top Exclusion Reasons",
    ])
    for item in summary["major_exclusion_reasons"][:5]:
        report.append(f"- {item['reason']}: {item['count']}")

    (out_dir / "summary.md").write_text("\n".join(report), encoding="utf-8")

    return summary


def main():
    import sys

    topics = sys.argv[1:] if len(sys.argv) > 1 else list(TOPIC_RULES.keys())

    results = {}
    for topic in topics:
        if topic not in TOPIC_RULES:
            print(f"[SKIP] No rules for {topic}")
            continue
        print(f"\n{'='*60}")
        print(f"  Processing: {topic}")
        print(f"{'='*60}")
        summary = process_topic(topic)
        results[topic] = summary
        print(f"  -> {summary['kept_rows']}/{summary['total_rows']} kept "
              f"({summary['kept_rows']/max(summary['total_rows'],1)*100:.1f}%)")
        if summary.get("kept_meta"):
            m = summary["kept_meta"]
            print(f"  -> Pooled: {m['pooled_pct']:+.2f}% [{m['ci_lo_pct']:.2f}, {m['ci_hi_pct']:.2f}]")
            if summary.get("benchmark_pct") is not None:
                diff = m["pooled_pct"] - summary["benchmark_pct"]
                print(f"  -> Benchmark: {summary['benchmark_pct']}% | Diff: {diff:+.2f} pp")

    # Write combined summary
    combined_path = OUT_ROOT / "universal_adjudication_summary.json"
    combined_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*60}")
    print("  ALL TOPICS COMPLETE")
    print(f"{'='*60}")
    for topic, s in results.items():
        kept = s.get("kept_meta")
        bench = s.get("benchmark_pct")
        if kept and bench is not None:
            diff = kept["pooled_pct"] - bench
            print(f"  {topic:30s} -> {kept['pooled_pct']:+7.2f}% vs {bench:+7.2f}% (diff {diff:+.2f} pp)")
        else:
            print(f"  {topic:30s} -> insufficient data for DL RE")


if __name__ == "__main__":
    main()
