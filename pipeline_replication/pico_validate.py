#!/usr/bin/env python3
"""
pico_validate.py — Post-extraction PICO validation for pipeline replications.

Reads summary.csv from extraction, applies strict PICO filtering rules based on
the config, and outputs a cleaned CSV + audit log. This is the missing validation
layer between extraction and synthesis.

Usage:
    python pico_validate.py organic_yield_gap
    python pico_validate.py notill_tillage
    python pico_validate.py mycorrhiza_yield
    python pico_validate.py legume_rotation
    python pico_validate.py biochar_crop_yield
    python pico_validate.py --all
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent  # pipeline_replication/

# ── Topic-specific PICO validators ───────────────────────────────────────────

def validate_organic(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate organic vs conventional yield gap observations."""
    audit = []

    # --- Rule 1: Exclude review articles ---
    def is_review(row):
        notes = str(row.get("notes", "")).lower()
        title = str(row.get("title", "")).lower()
        return ("review" in notes and ("secondary data" in notes or "compiling" in notes)) \
            or "review article" in notes \
            or "meta-analysis" in title \
            or "systematic review" in title

    review_mask = df.apply(is_review, axis=1)
    if review_mask.any():
        papers = df[review_mask]["paper_id"].unique().tolist()
        audit.append({"rule": "exclude_reviews", "removed_papers": papers,
                       "n_removed": int(review_mask.sum())})
    df = df[~review_mask].copy()

    # --- Rule 2: Treatment must be organic-related ---
    organic_keywords = [
        "organic", "compost", "manure", "vermicompost", "fym",
        "farmyard manure", "green manure", "biodynamic",
        "no synthetic", "without synthetic", "without chemical",
        "organically", "organic farming", "organic system",
    ]
    # Control must be conventional-related
    conventional_keywords = [
        "conventional", "npk", "chemical fertiliz", "synthetic fertiliz",
        "inorganic fertiliz", "mineral fertiliz", "urea",
        "conventional system", "conventional farming",
        "chemical", "synthetic", "high-input",
    ]

    def is_valid_organic_comparison(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()

        has_organic_treatment = any(k in td for k in organic_keywords)
        has_conv_control = any(k in cd for k in conventional_keywords)

        # Also accept if control desc mentions conventional in any form
        # But reject if control is just "unamended soil" (that's not conventional farming)
        if cd.strip() in ("unamended soil", "unamended", "control", "no amendment",
                          "unfertilized", "no fertilizer", "check"):
            has_conv_control = False

        # Reject if treatment is just an amendment vs nothing (not organic farming system)
        amendment_only = ["mulch", "biochar", "lime", "gypsum", "nanobubble",
                          "foliar spray", "humic acid", "seaweed", "biostimulant"]
        is_amendment = any(k in td for k in amendment_only) and not any(k in td for k in ["organic", "compost", "manure", "fym"])

        return has_organic_treatment and has_conv_control and not is_amendment

    pico_mask = df.apply(is_valid_organic_comparison, axis=1)
    excluded = df[~pico_mask]
    if len(excluded) > 0:
        ex_papers = excluded.groupby("paper_id").size().to_dict()
        audit.append({
            "rule": "pico_organic_vs_conventional",
            "description": "Treatment must be organic system, control must be conventional with synthetic inputs",
            "removed_papers": list(ex_papers.keys()),
            "n_removed": int((~pico_mask).sum()),
            "examples": [
                {"paper": row["paper_id"],
                 "treatment": str(row.get("treatment_description", "")),
                 "control": str(row.get("control_description", ""))}
                for _, row in excluded.head(10).iterrows()
            ]
        })
    df = df[pico_mask].copy()

    # --- Rule 3: Detect T/C swaps ---
    # If treatment_description contains conventional keywords and control has organic keywords, swap
    def needs_swap(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()
        t_is_conv = any(k in td for k in conventional_keywords)
        c_is_organic = any(k in cd for k in organic_keywords)
        return t_is_conv and c_is_organic

    swap_mask = df.apply(needs_swap, axis=1)
    if swap_mask.any():
        n_swapped = int(swap_mask.sum())
        # Perform the swap
        df.loc[swap_mask, ["treatment_mean", "control_mean"]] = \
            df.loc[swap_mask, ["control_mean", "treatment_mean"]].values
        df.loc[swap_mask, ["treatment_n", "control_n"]] = \
            df.loc[swap_mask, ["control_n", "treatment_n"]].values
        df.loc[swap_mask, ["sd_treatment", "sd_control"]] = \
            df.loc[swap_mask, ["sd_control", "sd_treatment"]].values
        df.loc[swap_mask, ["se_treatment", "se_control"]] = \
            df.loc[swap_mask, ["se_control", "se_treatment"]].values
        df.loc[swap_mask, ["treatment_description", "control_description"]] = \
            df.loc[swap_mask, ["control_description", "treatment_description"]].values
        # Recalculate effect
        df.loc[swap_mask, "effect_pct"] = (
            (df.loc[swap_mask, "treatment_mean"] - df.loc[swap_mask, "control_mean"])
            / df.loc[swap_mask, "control_mean"].abs() * 100
        )
        audit.append({"rule": "tc_swap_correction", "n_swapped": n_swapped})

    # --- Rule 4: Confidence filter ---
    low_conf_mask = df["confidence"].astype(str).str.lower() == "low"
    if low_conf_mask.any():
        audit.append({"rule": "exclude_low_confidence", "n_removed": int(low_conf_mask.sum())})
    df = df[~low_conf_mask].copy()

    return df, audit


def validate_notill(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate no-till vs conventional tillage observations."""
    import re
    audit = []

    # --- Rule 1: Treatment must be no-till related ---
    # Use word-boundary matching for short abbreviations to avoid false positives
    # e.g. "nt" matching inside "conventional", "ct" matching inside "direct"
    notill_phrases = [
        "no-till", "no till", "zero-till", "zero till", "direct seed",
        "direct drill", "no tillage", "zero tillage",
        "conservation agriculture", "conservation tillage",
        "minimum tillage", "reduced tillage", "strip-till",
    ]
    # Short abbreviations need word boundary matching
    notill_abbrevs_re = re.compile(r'\b(NT|ZT)\b', re.IGNORECASE)

    conv_phrases = [
        "conventional till", "conventional tillage", "conventional sowing",
        "moldboard", "mouldboard", "plough", "plow", "disk harrow",
        "inversion tillage", "traditional tillage", "full tillage",
        "puddled", "ploughing", "disc plough", "chisel plough",
    ]
    conv_abbrevs_re = re.compile(r'\bCT\b')  # case-sensitive: CT only

    def has_notill(text):
        return any(k in text for k in notill_phrases) or bool(notill_abbrevs_re.search(text))

    def has_conv_till(text):
        return any(k in text for k in conv_phrases) or bool(conv_abbrevs_re.search(text))

    def classify_tc(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()
        # For abbreviation regex, use original case
        td_orig = str(row.get("treatment_description", ""))
        cd_orig = str(row.get("control_description", ""))

        t_is_notill = any(k in td for k in notill_phrases) or bool(notill_abbrevs_re.search(td_orig))
        c_is_conv = any(k in cd for k in conv_phrases) or bool(conv_abbrevs_re.search(cd_orig))
        c_is_notill = any(k in cd for k in notill_phrases) or bool(notill_abbrevs_re.search(cd_orig))
        t_is_conv = (any(k in td for k in conv_phrases) or bool(conv_abbrevs_re.search(td_orig))) and not t_is_notill

        if t_is_notill and c_is_conv:
            return "correct"
        elif t_is_conv and c_is_notill:
            return "swapped"
        elif t_is_notill and not c_is_conv:
            return "unclear_control"
        elif c_is_notill and not t_is_conv:
            return "unclear_treatment"
        else:
            return "no_match"

    df["_tc_status"] = df.apply(classify_tc, axis=1)
    status_counts = df["_tc_status"].value_counts().to_dict()
    audit.append({"rule": "tc_classification", "counts": status_counts})

    # --- Rule 2: Swap T/C where needed ---
    swap_mask = df["_tc_status"] == "swapped"
    if swap_mask.any():
        n_swapped = int(swap_mask.sum())
        df.loc[swap_mask, ["treatment_mean", "control_mean"]] = \
            df.loc[swap_mask, ["control_mean", "treatment_mean"]].values
        df.loc[swap_mask, ["treatment_n", "control_n"]] = \
            df.loc[swap_mask, ["control_n", "treatment_n"]].values
        df.loc[swap_mask, ["sd_treatment", "sd_control"]] = \
            df.loc[swap_mask, ["sd_control", "sd_treatment"]].values
        df.loc[swap_mask, ["se_treatment", "se_control"]] = \
            df.loc[swap_mask, ["se_control", "se_treatment"]].values
        df.loc[swap_mask, ["treatment_description", "control_description"]] = \
            df.loc[swap_mask, ["control_description", "treatment_description"]].values
        df.loc[swap_mask, "effect_pct"] = (
            (df.loc[swap_mask, "treatment_mean"] - df.loc[swap_mask, "control_mean"])
            / df.loc[swap_mask, "control_mean"].abs() * 100
        )
        audit.append({"rule": "tc_swap_correction", "n_swapped": n_swapped})

    # --- Rule 3: Exclude observations with no PICO match ---
    no_match = df["_tc_status"] == "no_match"
    if no_match.any():
        papers = df[no_match]["paper_id"].unique().tolist()
        audit.append({"rule": "exclude_no_pico_match",
                       "removed_papers": papers,
                       "n_removed": int(no_match.sum())})
    df = df[~no_match].copy()

    # --- Rule 4: Keep unclear_control/unclear_treatment but flag ---
    unclear = df["_tc_status"].isin(["unclear_control", "unclear_treatment"])
    if unclear.any():
        audit.append({"rule": "flagged_unclear_tc",
                       "n_flagged": int(unclear.sum()),
                       "papers": df[unclear]["paper_id"].unique().tolist()})

    # --- Rule 5: Direction sanity check ---
    # Expected direction is negative (no-till typically hurts yield)
    # but paper says "either" — so just flag papers where ALL observations
    # are strongly positive (>30%) as potential T/C confusion
    suspicious = []
    for pid, grp in df.groupby("paper_id"):
        effs = grp["effect_pct"].dropna()
        if len(effs) >= 2 and effs.mean() > 30:
            suspicious.append(pid)
    if suspicious:
        audit.append({"rule": "suspicious_all_positive",
                       "papers": suspicious,
                       "note": "All effects >30% positive — possible T/C swap or wrong comparison"})

    df = df.drop(columns=["_tc_status"], errors="ignore")
    return df, audit


def validate_mycorrhiza(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate mycorrhiza yield observations."""
    audit = []

    # --- Rule 1: Treatment must be AMF inoculation ---
    amf_keywords = [
        "mycorrhiz", "amf", "am fungi", "glomus", "rhizophagus",
        "funneliformis", "claroideoglomus", "inocul", "vam",
        "vesicular-arbuscular",
    ]
    control_keywords = [
        "non-inocul", "uninocul", "control", "without amf", "-amf",
        "non-mycorrhiz", "sterilized inocul", "no amf", "nm",
    ]

    def is_valid_amf_comparison(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()
        has_amf = any(k in td for k in amf_keywords)
        has_control = any(k in cd for k in control_keywords) or "control" in cd
        return has_amf and has_control

    pico_mask = df.apply(is_valid_amf_comparison, axis=1)
    excluded = df[~pico_mask]
    if len(excluded) > 0:
        audit.append({
            "rule": "pico_amf_vs_control",
            "n_removed": int((~pico_mask).sum()),
            "removed_papers": excluded["paper_id"].unique().tolist(),
        })
    df = df[pico_mask].copy()

    # --- Rule 2: Outcome must be yield or biomass ---
    yield_keywords = [
        "yield", "biomass", "dry weight", "fresh weight", "dry matter",
        "shoot weight", "grain", "fruit", "tuber", "seed weight",
        "plant weight", "total weight", "pod", "ear weight",
        "kg/ha", "g/plant", "mg/ha", "t/ha",
    ]
    exclude_outcome = [
        "colonization", "infection", "spore", "root length",
        "p uptake", "phosphorus", "nitrogen", "nutrient",
        "chlorophyll", "photosynthesis", "stomatal",
        "antioxidant", "enzyme", "hormone", "proline",
    ]

    def is_yield_or_biomass(row):
        text = "|".join(str(row.get(c, "")).lower()
                        for c in ["outcome", "outcome_unit"])
        has_yield = any(k in text for k in yield_keywords)
        is_excluded = any(k in text for k in exclude_outcome)
        return has_yield and not is_excluded

    outcome_mask = df.apply(is_yield_or_biomass, axis=1)
    if (~outcome_mask).any():
        audit.append({
            "rule": "filter_to_yield_biomass",
            "n_removed": int((~outcome_mask).sum()),
            "removed_outcomes": df[~outcome_mask]["outcome"].value_counts().head(10).to_dict(),
        })
    df = df[outcome_mask].copy()

    # --- Rule 3: Detect T/C swaps ---
    def needs_swap(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()
        t_is_control = any(k in td for k in control_keywords) and not any(k in td for k in amf_keywords)
        c_is_amf = any(k in cd for k in amf_keywords)
        return t_is_control and c_is_amf

    swap_mask = df.apply(needs_swap, axis=1)
    if swap_mask.any():
        n_swapped = int(swap_mask.sum())
        df.loc[swap_mask, ["treatment_mean", "control_mean"]] = \
            df.loc[swap_mask, ["control_mean", "treatment_mean"]].values
        df.loc[swap_mask, ["treatment_n", "control_n"]] = \
            df.loc[swap_mask, ["control_n", "treatment_n"]].values
        df.loc[swap_mask, ["sd_treatment", "sd_control"]] = \
            df.loc[swap_mask, ["sd_control", "sd_treatment"]].values
        df.loc[swap_mask, ["se_treatment", "se_control"]] = \
            df.loc[swap_mask, ["se_control", "se_treatment"]].values
        df.loc[swap_mask, ["treatment_description", "control_description"]] = \
            df.loc[swap_mask, ["control_description", "treatment_description"]].values
        df.loc[swap_mask, "effect_pct"] = (
            (df.loc[swap_mask, "treatment_mean"] - df.loc[swap_mask, "control_mean"])
            / df.loc[swap_mask, "control_mean"].abs() * 100
        )
        audit.append({"rule": "tc_swap_correction", "n_swapped": n_swapped})

    return df, audit


def validate_legume_rotation(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate legume rotation yield observations."""
    import re
    audit = []

    # --- Rule 1: Exclude review articles ---
    def is_review(row):
        notes = str(row.get("notes", "")).lower()
        title = str(row.get("title", "")).lower()
        return "review" in title or "meta-analysis" in title or "systematic review" in title
    review_mask = df.apply(is_review, axis=1)
    if review_mask.any():
        audit.append({"rule": "exclude_reviews", "n_removed": int(review_mask.sum())})
    df = df[~review_mask].copy()

    # --- Rule 2: Treatment must involve legume rotation ---
    legume_keywords = [
        "legume", "soybean", "soy", "pea", "lentil", "chickpea",
        "faba bean", "fababean", "clover", "lupin", "alfalfa", "lucerne",
        "vetch", "cowpea", "pigeon pea", "mung bean", "groundnut",
        "bean rotation", "legume rotation", "legume pre-crop",
        "preceding legume", "after legume", "legume-cereal",
        "biological nitrogen", "n fixation",
    ]
    control_keywords = [
        "continuous", "monoculture", "fallow", "cereal-cereal",
        "non-legume", "without legume", "wheat-wheat", "maize-maize",
        "continuous cropping", "continuous wheat", "continuous maize",
    ]

    def check_legume_pico(row):
        t_desc = str(row.get("treatment_description", "")).lower()
        c_desc = str(row.get("control_description", "")).lower()
        outcome = str(row.get("outcome", "")).lower()
        notes = str(row.get("notes", "")).lower()
        all_text = f"{t_desc}|{c_desc}|{outcome}|{notes}"

        has_legume = any(k in all_text for k in legume_keywords)
        # Exclude if this is measuring the legume yield itself (not subsequent crop)
        is_legume_yield = any(
            k in outcome for k in ["legume yield", "soybean yield", "pea yield",
                                     "lentil yield", "chickpea yield"]
        )
        return has_legume and not is_legume_yield

    pico_mask = df.apply(check_legume_pico, axis=1)
    n_removed = int((~pico_mask).sum())
    if n_removed:
        audit.append({"rule": "require_legume_rotation_pico", "n_removed": n_removed})
    df = df[pico_mask].copy()

    # --- Rule 3: Yield outcomes only ---
    yield_keywords = [
        "yield", "grain", "biomass", "dry weight", "fresh weight",
        "kg/ha", "t/ha", "mg/ha", "g/plant", "g/m2", "productivity",
        "fruit", "tuber", "seed", "harvest",
    ]
    exclude_keywords = [
        "soil", "nitrogen content", "protein", "quality", "weed",
        "disease", "pest", "root length", "colonization", "n uptake",
        "carbon", "ph", "nutrient concentration",
    ]

    def is_yield(row):
        text = "|".join(str(row.get(c, "")).lower() for c in ["outcome", "outcome_unit"])
        has = any(k in text for k in yield_keywords)
        excl = any(k in text for k in exclude_keywords)
        return has and not excl

    yield_mask = df.apply(is_yield, axis=1)
    n_removed = int((~yield_mask).sum())
    if n_removed:
        audit.append({"rule": "yield_outcomes_only", "n_removed": n_removed})
    df = df[yield_mask].copy()

    return df, audit


def validate_biochar_yield(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate biochar crop yield observations."""
    audit = []

    # --- Rule 1: Exclude review articles ---
    def is_review(row):
        notes = str(row.get("notes", "")).lower()
        title = str(row.get("title", "")).lower()
        return "review" in title or "meta-analysis" in title or "systematic review" in title
    review_mask = df.apply(is_review, axis=1)
    if review_mask.any():
        audit.append({"rule": "exclude_reviews", "n_removed": int(review_mask.sum())})
    df = df[~review_mask].copy()

    # --- Rule 2: Treatment must involve biochar ---
    biochar_keywords = [
        "biochar", "bio-char", "charcoal", "pyrolysis char",
        "biomass char", "biochar amend", "char addition",
        "biochar application", "biochar treatment",
    ]

    def check_biochar_pico(row):
        t_desc = str(row.get("treatment_description", "")).lower()
        c_desc = str(row.get("control_description", "")).lower()
        notes = str(row.get("notes", "")).lower()
        all_text = f"{t_desc}|{c_desc}|{notes}"
        return any(k in all_text for k in biochar_keywords)

    pico_mask = df.apply(check_biochar_pico, axis=1)
    n_removed = int((~pico_mask).sum())
    if n_removed:
        audit.append({"rule": "require_biochar_pico", "n_removed": n_removed})
    df = df[pico_mask].copy()

    # --- Rule 3: Yield/biomass outcomes only ---
    yield_keywords = [
        "yield", "grain", "biomass", "dry weight", "fresh weight",
        "kg/ha", "t/ha", "mg/ha", "g/plant", "g/pot", "g/m2",
        "productivity", "fruit", "tuber", "seed", "harvest",
        "dry matter", "above-ground", "aboveground",
    ]
    exclude_keywords = [
        "soil carbon", "soil organic", "soil ph", "heavy metal",
        "adsorption", "removal efficiency", "nutrient concentration",
        "microbial", "enzyme", "respiration", "leaching",
        "greenhouse gas", "emission", "root colonization",
    ]

    def is_yield(row):
        text = "|".join(str(row.get(c, "")).lower() for c in ["outcome", "outcome_unit"])
        has = any(k in text for k in yield_keywords)
        excl = any(k in text for k in exclude_keywords)
        return has and not excl

    yield_mask = df.apply(is_yield, axis=1)
    n_removed = int((~yield_mask).sum())
    if n_removed:
        audit.append({"rule": "yield_outcomes_only", "n_removed": n_removed})
    df = df[yield_mask].copy()

    # --- Rule 4: T/C swap check ---
    # If expected positive but most effects are strongly negative, check for swap
    if len(df) > 5:
        df["_effect"] = (df["treatment_mean"] - df["control_mean"]) / df["control_mean"].abs() * 100
        negative_frac = (df["_effect"] < -20).mean()
        if negative_frac > 0.6:
            audit.append({"rule": "tc_swap_warning",
                         "message": f"{negative_frac:.0%} of effects are < -20%, possible T/C swap"})
        df = df.drop(columns=["_effect"], errors="ignore")

    return df, audit


# ── Universal filters ────────────────────────────────────────────────────────

def apply_universal_filters(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Filters that apply to all topics."""
    audit = []
    n_start = len(df)

    # 1. Require both means
    mask = df["treatment_mean"].notna() & df["control_mean"].notna()
    n_dropped = int((~mask).sum())
    if n_dropped:
        audit.append({"rule": "require_both_means", "n_removed": n_dropped})
    df = df[mask].copy()

    # 2. Require positive means (for lnRR)
    mask = (df["treatment_mean"] > 0) & (df["control_mean"] > 0)
    n_dropped = int((~mask).sum())
    if n_dropped:
        audit.append({"rule": "require_positive_means", "n_removed": n_dropped})
    df = df[mask].copy()

    # 3. Outlier removal: effect between -90% and +500%
    df["_effect"] = (df["treatment_mean"] - df["control_mean"]) / df["control_mean"].abs() * 100
    mask = (df["_effect"] > -90) & (df["_effect"] < 500)
    n_dropped = int((~mask).sum())
    if n_dropped:
        audit.append({"rule": "outlier_removal_-90_to_500", "n_removed": n_dropped})
    df = df[mask].copy()
    df = df.drop(columns=["_effect"], errors="ignore")

    audit.append({"rule": "total_after_universal", "n_remaining": len(df),
                   "n_removed_total": n_start - len(df)})
    return df, audit


# ── Main ─────────────────────────────────────────────────────────────────────

TOPIC_VALIDATORS = {
    "organic_yield_gap": validate_organic,
    "notill_tillage": validate_notill,
    "mycorrhiza_yield": validate_mycorrhiza,
    "legume_rotation": validate_legume_rotation,
    "biochar_crop_yield": validate_biochar_yield,
}


def process_topic(topic: str):
    topic_dir = ROOT / topic
    config_path = topic_dir / "config.json"
    summary_csv = topic_dir / "4_extract" / "summary.csv"

    if not summary_csv.exists():
        print(f"ERROR: {summary_csv} not found")
        return

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    df = pd.read_csv(summary_csv)
    print(f"\n{'='*60}")
    print(f"PICO VALIDATION: {topic}")
    print(f"{'='*60}")
    print(f"Input: {len(df)} observations across {df['paper_id'].nunique()} papers")

    all_audit = []

    # Topic-specific validation
    validator = TOPIC_VALIDATORS.get(topic)
    if validator:
        df, topic_audit = validator(df, config)
        all_audit.extend(topic_audit)
        print(f"After topic-specific filters: {len(df)} obs, {df['paper_id'].nunique()} papers")

    # Universal filters
    df, univ_audit = apply_universal_filters(df, config)
    all_audit.extend(univ_audit)
    print(f"After universal filters: {len(df)} obs, {df['paper_id'].nunique()} papers")

    # Recalculate effect_pct
    df["effect_pct"] = (
        (df["treatment_mean"] - df["control_mean"])
        / df["control_mean"].abs() * 100
    )

    # Summary stats
    effs = df["effect_pct"].dropna()
    if len(effs) > 0:
        print(f"\nEffect distribution after PICO validation:")
        print(f"  Mean:   {effs.mean():+.2f}%")
        print(f"  Median: {effs.median():+.2f}%")
        print(f"  Negative: {(effs < 0).sum()}/{len(effs)} ({(effs < 0).mean()*100:.0f}%)")
        print(f"  Positive: {(effs > 0).sum()}/{len(effs)} ({(effs > 0).mean()*100:.0f}%)")

    # Show benchmark comparison
    bench = config.get("benchmark", {}).get("published_pooled_effect", {})
    if bench:
        bench_est = bench.get("estimate")
        if bench_est is not None:
            print(f"\n  Benchmark: {bench_est:+.1f}%")
            print(f"  Our mean:  {effs.mean():+.2f}%")
            print(f"  Our median: {effs.median():+.2f}%")
            direction_match = (effs.mean() < 0) == (bench_est < 0)
            print(f"  Direction match: {direction_match}")

    # Write outputs
    out_csv = topic_dir / "4_extract" / "summary_validated.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nValidated CSV: {out_csv}")

    audit_path = topic_dir / "4_extract" / "pico_validation_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(all_audit, f, indent=2, ensure_ascii=False, default=str)
    print(f"Audit log:     {audit_path}")

    # Print audit summary
    print("\nAudit summary:")
    for entry in all_audit:
        rule = entry.get("rule", "unknown")
        if "n_removed" in entry:
            print(f"  {rule}: removed {entry['n_removed']} obs")
        elif "n_swapped" in entry:
            print(f"  {rule}: swapped {entry['n_swapped']} obs")
        elif "n_flagged" in entry:
            print(f"  {rule}: flagged {entry['n_flagged']} obs")
        elif "counts" in entry:
            print(f"  {rule}: {entry['counts']}")
        elif "n_remaining" in entry:
            print(f"  {rule}: {entry['n_remaining']} obs remaining")

    return df, all_audit


def main():
    parser = argparse.ArgumentParser(description="Post-extraction PICO validation")
    parser.add_argument("topic", nargs="?", help="Topic directory name")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    args = parser.parse_args()

    if args.all:
        for topic in TOPIC_VALIDATORS:
            if (ROOT / topic / "4_extract" / "summary.csv").exists():
                process_topic(topic)
    elif args.topic:
        process_topic(args.topic)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
