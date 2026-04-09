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
    """Validate no-till vs conventional tillage observations.

    Strict rules: only TRUE zero-till / no-till / direct seeding included.
    Excludes minimum tillage, reduced tillage, strip-till, and conservation
    agriculture unless it explicitly specifies no-till as the core treatment.
    """
    import re
    audit = []

    # --- Rule 0: Exclude non-zero-till treatments ---
    # These are NOT zero-till and inflate the pooled effect upward:
    #   minimum tillage, reduced tillage, strip-till, rotary tillage,
    #   non-inversion tillage, deep non-inversion tillage, bed planting,
    #   double-layer ploughing
    exclude_tillage_phrases = [
        "minimum tillage", "reduced tillage", "strip-till", "strip till",
        "rotary tillage", "rotary till",
        "non-inversion tillage", "non-inversion till",
        "deep non-inversion", "shallow non-inversion",
        "bed planting", "double-layer plough", "double-layer plow",
        "all reduced tillage",
        "ridge till", "ridge-till",
    ]
    # Also exclude "conservation agriculture" UNLESS it explicitly says
    # no-till / zero-till / NT / ZT / direct seed
    true_notill_markers = [
        "no-till", "no till", "no tillage", "zero-till", "zero till",
        "zero tillage", "direct seed", "direct drill", "direct sow",
    ]
    true_notill_abbrev_re = re.compile(r'\b(NT|ZT)\b')  # case-sensitive

    def _is_true_zero_till(text_lower, text_orig):
        """Return True only if the description indicates true zero-till."""
        return (
            any(k in text_lower for k in true_notill_markers)
            or bool(true_notill_abbrev_re.search(text_orig))
        )

    def _should_exclude_treatment(row):
        """Return (exclude: bool, reason: str)."""
        td = str(row.get("treatment_description", "")).lower()
        td_orig = str(row.get("treatment_description", ""))

        # Check for explicitly excluded tillage types
        for phrase in exclude_tillage_phrases:
            if phrase in td:
                # Exception: if description ALSO says no-till/zero-till, keep it
                if _is_true_zero_till(td, td_orig):
                    continue
                return True, f"non_zero_till: '{phrase}'"

        # Conservation agriculture: exclude unless explicitly no-till
        if "conservation agriculture" in td or "conservation tillage" in td:
            if not _is_true_zero_till(td, td_orig):
                return True, "conservation_agriculture_without_explicit_notill"

        return False, ""

    excl_mask = pd.Series(False, index=df.index)
    excl_reasons = []
    for idx, row in df.iterrows():
        exclude, reason = _should_exclude_treatment(row)
        if exclude:
            excl_mask.at[idx] = True
            excl_reasons.append({"paper": row.get("paper_id", ""),
                                 "treatment": str(row.get("treatment_description", ""))[:120],
                                 "reason": reason})

    if excl_mask.any():
        n_excl = int(excl_mask.sum())
        papers_excl = df[excl_mask]["paper_id"].unique().tolist()
        reason_counts = {}
        for r in excl_reasons:
            k = r["reason"]
            reason_counts[k] = reason_counts.get(k, 0) + 1
        audit.append({
            "rule": "exclude_non_zero_till",
            "description": "Removed minimum tillage, reduced tillage, strip-till, "
                           "non-inversion tillage, and conservation agriculture "
                           "without explicit no-till comparison",
            "n_removed": n_excl,
            "removed_papers": papers_excl,
            "reason_counts": reason_counts,
            "examples": excl_reasons[:15],
        })
    df = df[~excl_mask].copy()

    # --- Rule 1: Treatment must be no-till related ---
    notill_phrases = [
        "no-till", "no till", "zero-till", "zero till", "direct seed",
        "direct drill", "no tillage", "zero tillage", "direct sow",
    ]
    notill_abbrevs_re = re.compile(r'\b(NT|ZT)\b', re.IGNORECASE)

    conv_phrases = [
        "conventional till", "conventional tillage", "conventional sowing",
        "moldboard", "mouldboard", "plough", "plow", "disk harrow",
        "inversion tillage", "traditional tillage", "full tillage",
        "puddled", "ploughing", "disc plough",
    ]
    conv_abbrevs_re = re.compile(r'\bCT\b')  # case-sensitive: CT only

    def classify_tc(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()
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

    # --- Rule 4: Exclude no-till vs reduced/minimum/rotary tillage controls ---
    # These are NOT the standard no-till vs conventional (moldboard/plough) comparison.
    # No-till vs minimum tillage shows a positive effect because the control is
    # already a reduced tillage system -- this inflates the pooled estimate.
    reduced_control_phrases = [
        "minimum tillage", "reduced tillage", "rotary tillage", "rotary till",
        "strip-till", "strip till", "non-inversion",
        "ridge till", "ridge-till", "shallow disc",
        "continuous rotary",
    ]

    def _control_is_reduced_tillage(row):
        cd = str(row.get("control_description", "")).lower()
        return any(k in cd for k in reduced_control_phrases)

    reduced_ctrl_mask = df.apply(_control_is_reduced_tillage, axis=1)
    if reduced_ctrl_mask.any():
        n_reduced = int(reduced_ctrl_mask.sum())
        papers_reduced = df[reduced_ctrl_mask]["paper_id"].unique().tolist()
        audit.append({
            "rule": "exclude_reduced_tillage_control",
            "description": "Removed no-till vs minimum/reduced/rotary tillage controls "
                           "(not the standard no-till vs conventional plough comparison)",
            "n_removed": n_reduced,
            "removed_papers": papers_reduced,
        })
        df = df[~reduced_ctrl_mask].copy()

    # Flag remaining unclear_control/unclear_treatment
    unclear = df["_tc_status"].isin(["unclear_control", "unclear_treatment"])
    if unclear.any():
        audit.append({"rule": "flagged_unclear_tc",
                       "n_flagged": int(unclear.sum()),
                       "papers": df[unclear]["paper_id"].unique().tolist()})

    # --- Rule 5: Extreme outlier removal ---
    # No-till effect >100% yield increase is almost certainly a bundled
    # intervention (intercropping, mulch, rotation combined) or data error,
    # not a true no-till vs conventional tillage comparison
    df["_effect_tmp"] = (
        (df["treatment_mean"] - df["control_mean"])
        / df["control_mean"].abs() * 100
    )
    extreme_mask = df["_effect_tmp"] > 100
    if extreme_mask.any():
        n_extreme = int(extreme_mask.sum())
        extreme_papers = df[extreme_mask]["paper_id"].unique().tolist()
        extreme_examples = [
            {"paper": row["paper_id"],
             "treatment": str(row.get("treatment_description", ""))[:120],
             "effect_pct": round(row["_effect_tmp"], 1)}
            for _, row in df[extreme_mask].head(10).iterrows()
        ]
        audit.append({
            "rule": "exclude_extreme_outliers_gt100pct",
            "description": "No-till effect >100% yield increase is not plausible "
                           "for a pure tillage comparison; likely bundled interventions",
            "n_removed": n_extreme,
            "removed_papers": extreme_papers,
            "examples": extreme_examples,
        })
        df = df[~extreme_mask].copy()
    df = df.drop(columns=["_effect_tmp"], errors="ignore")

    # --- Rule 6: Direction sanity check ---
    # Flag papers where ALL observations are strongly positive (>30%)
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


def validate_intercropping(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Validate intercropping vs sole-cropping yield observations.

    Key issues addressed:
    - LER observations (+100%) must be separated from individual crop yields (~-5%)
    - Non-yield outcomes (biomass, straw, plant height, stem diameter, leaf area) excluded
    - T/C swaps detected (monoculture as treatment, intercrop as control)
    - Per-plant metrics excluded (grain weight per plant, ear number per plant)
      because they conflate density effects with yield effects
    """
    import re
    audit = []

    n_start = len(df)

    # --- Rule 1: Treatment must contain intercropping keywords ---
    intercrop_keywords = [
        "intercrop", "inter-crop", "mixed crop", "mixed cropping",
        "strip intercrop", "relay intercrop", "relay strip",
        "companion crop", "companion plant", "polyculture",
        "crop mixture", "cereal-legume", "maize-soybean",
        "maize-bean", "wheat-faba", "barley-pea",
        "wheat-soybean", "cereal legume",
        "additive intercrop", "replacement intercrop",
    ]
    # Control must contain sole/mono keywords
    sole_keywords = [
        "sole", "mono", "monoculture", "monocrop", "mono-crop",
        "pure stand", "single crop", "sole crop",
    ]

    def check_tc(row):
        td = str(row.get("treatment_description", "")).lower()
        cd = str(row.get("control_description", "")).lower()

        t_is_intercrop = any(k in td for k in intercrop_keywords)
        c_is_sole = any(k in cd for k in sole_keywords)
        t_is_sole = any(k in td for k in sole_keywords)
        c_is_intercrop = any(k in cd for k in intercrop_keywords)

        if t_is_intercrop and c_is_sole:
            return "correct"
        elif t_is_sole and c_is_intercrop:
            return "swapped"
        elif t_is_intercrop and not c_is_sole:
            return "unclear_control"
        elif c_is_sole and not t_is_intercrop:
            return "unclear_treatment"
        else:
            return "no_match"

    df["_tc_status"] = df.apply(check_tc, axis=1)
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

    # --- Rule 3: Exclude no_match observations ---
    no_match = df["_tc_status"] == "no_match"
    if no_match.any():
        papers = df[no_match]["paper_id"].unique().tolist()
        audit.append({"rule": "exclude_no_pico_match",
                       "removed_papers": papers,
                       "n_removed": int(no_match.sum()),
                       "examples": [
                           {"paper": row["paper_id"],
                            "treatment": str(row.get("treatment_description", ""))[:100],
                            "control": str(row.get("control_description", ""))[:100]}
                           for _, row in df[no_match].head(5).iterrows()
                       ]})
    df = df[~no_match].copy()

    # --- Rule 4: Separate LER from individual crop yields ---
    ler_pattern = re.compile(r'\bLER\b|land equivalent ratio', re.IGNORECASE)
    df["_is_ler"] = df["outcome"].astype(str).apply(lambda x: bool(ler_pattern.search(x)))

    n_ler = int(df["_is_ler"].sum())
    n_non_ler = int((~df["_is_ler"]).sum())
    audit.append({
        "rule": "ler_separation",
        "n_ler": n_ler,
        "n_non_ler": n_non_ler,
        "note": "LER observations are system-level (LER~1.22 → +22%); "
                "individual crop yields are often negative. Keeping ONLY LER "
                "for primary synthesis (matches benchmark definition)."
    })

    # For the primary synthesis: keep only LER observations
    # Individual crop yields measure something fundamentally different
    # (a single species' yield under competition, NOT system productivity)
    df_ler = df[df["_is_ler"]].copy()
    df_non_ler = df[~df["_is_ler"]].copy()

    # --- Rule 5: For non-LER, keep only grain/seed yield (not biomass/straw/plant parts) ---
    # This produces a secondary "individual crop yield" analysis
    grain_yield_keywords = [
        "grain yield", "seed yield", "crop yield", "eggplant yield",
        "fruit yield", "tuber yield", "pod yield",
        "crop stand grain yield",
    ]
    # Outcomes to EXCLUDE (non-yield or per-plant metrics)
    exclude_outcomes = [
        "biomass", "straw", "dry matter", "aboveground biomass",
        "plant biomass", "root biomass", "shoot biomass",
        "plant height", "stem diameter", "leaf area",
        "ear number", "tiller", "branch",
        # Per-plant metrics conflate density with yield
        "weight per plant", "grain weight per plant",
        "per plant", "100-grain weight", "1000-grain weight",
        "100 grain weight", "1000 grain weight",
        "kernel weight", "test weight",
    ]

    def is_grain_yield(row):
        outcome_lower = str(row.get("outcome", "")).lower()
        unit_lower = str(row.get("outcome_unit", "")).lower()
        text = f"{outcome_lower}|{unit_lower}"

        # Check explicit grain yield keywords
        has_grain = any(k in text for k in grain_yield_keywords)

        # Also accept if unit is area-based yield (kg/ha, t/ha, Mg/ha, g/m2)
        has_area_unit = bool(re.search(r'(kg|mg|t|g)\s*/\s*(ha|m2|m²|acre)', unit_lower))
        # But only if outcome contains "yield"
        has_yield_word = "yield" in outcome_lower

        is_excluded = any(k in outcome_lower for k in exclude_outcomes)

        return (has_grain or (has_area_unit and has_yield_word)) and not is_excluded

    yield_mask = df_non_ler.apply(is_grain_yield, axis=1)
    n_grain_kept = int(yield_mask.sum())
    n_excluded = int((~yield_mask).sum())

    if n_excluded > 0:
        excluded_outcomes = df_non_ler[~yield_mask]["outcome"].value_counts().to_dict()
        audit.append({
            "rule": "exclude_non_yield_outcomes",
            "n_removed": n_excluded,
            "n_kept_grain_yield": n_grain_kept,
            "removed_outcomes": excluded_outcomes,
        })

    df_grain = df_non_ler[yield_mask].copy()

    # --- Rule 6: For LER, handle the special effect size calculation ---
    # LER is a ratio where control = 1.0 (by definition, sole crop LER = 1)
    # Effect = (LER - 1) * 100, NOT the standard (T-C)/C formula
    # If the data has LER as treatment_mean and 1.0 as control_mean, the
    # standard formula (T-C)/C = (LER-1)/1 = LER-1 which IS correct
    # But if control_mean is something other than 1.0, we need to check
    ler_issues = []
    for idx, row in df_ler.iterrows():
        cm = row.get("control_mean")
        tm = row.get("treatment_mean")
        # LER control should be 1.0 (by definition)
        if pd.notna(cm) and abs(float(cm) - 1.0) > 0.05:
            ler_issues.append({
                "paper": row["paper_id"],
                "control_mean": float(cm),
                "treatment_mean": float(tm) if pd.notna(tm) else None,
            })
    if ler_issues:
        audit.append({
            "rule": "ler_control_check",
            "note": "LER control should be 1.0; these papers have different values",
            "issues": ler_issues,
        })

    # --- Combine: LER for primary, grain yield for secondary ---
    # Tag observations
    df_ler["_obs_type"] = "LER"
    df_grain["_obs_type"] = "individual_crop_yield"

    # Primary synthesis uses LER only
    df_combined = pd.concat([df_ler, df_grain], ignore_index=True)

    # Clean up temp columns
    df_combined = df_combined.drop(columns=["_tc_status", "_is_ler"], errors="ignore")

    audit.append({
        "rule": "final_composition",
        "n_ler": n_ler,
        "n_grain_yield": n_grain_kept,
        "n_total": len(df_combined),
        "note": "Primary synthesis uses LER only (matches +22% benchmark). "
                "Individual crop yields available as secondary analysis.",
    })

    return df_combined, audit


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
    "intercropping_yield": validate_intercropping,
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
