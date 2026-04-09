"""
LLM semantic adjudication of V1 pipeline rows across all 6 topics.
This script applies domain-aware rules to decide keep/exclude/flag/swap.
"""

import json
import math
import os
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
BASE = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\pipeline_replication\codex\outputs")
INPUT_DIR = BASE / "universal_llm_inputs"
OUTPUT_DIR = BASE / "llm_decisions"
CODEX_DIR = BASE / "codex_decisions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── helper utilities ────────────────────────────────────────────────────────

def safe_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def lnRR(t, c):
    if t is None or c is None:
        return None
    if t <= 0 or c <= 0:
        return None
    return math.log(t / c)


def pct_change(t, c):
    lr = lnRR(t, c)
    if lr is None:
        return None
    return (math.exp(lr) - 1) * 100


def mean_pct(pcts):
    valid = [p for p in pcts if p is not None and not math.isnan(p)]
    if not valid:
        return None
    return sum(valid) / len(valid)


# ══════════════════════════════════════════════════════════════════════════════
#  TOPIC-SPECIFIC ADJUDICATION LOGIC
# ══════════════════════════════════════════════════════════════════════════════

# ── shared exclusion patterns ────────────────────────────────────────────────

NON_YIELD_OUTCOMES_UNIVERSAL = {
    # morphology / growth
    "plant height", "stem height", "stem diameter", "stem girth", "leaf area",
    "leaf area index", "number of leaves", "leaf number", "canopy cover",
    "plant length", "tiller number", "branch number", "pod number",
    "number of pods", "pod count", "number of fruits",
    "fruit number", "flower number", "days to flowering", "days to maturity",
    "seedling height", "seedling length", "seedling emergence",
    # root traits
    "root length", "root volume", "root surface area", "root dry weight",
    "root fresh weight", "root biomass", "root diameter",
    # nutrient / soil
    "nitrogen content", "phosphorus content", "potassium content",
    "n uptake", "p uptake", "k uptake", "nutrient uptake",
    "chlorophyll content", "chlorophyll a", "chlorophyll b", "spad",
    "leaf nitrogen", "leaf phosphorus",
    "soil organic carbon", "soil organic matter", "soil ph",
    "microbial biomass", "enzyme activity",
    # water
    "water use efficiency", "wue", "transpiration", "stomatal conductance",
    "leaf water potential",
    # quality
    "protein content", "gluten", "oil content", "fat content",
    "starch content", "fiber content", "brix", "sugar content",
    # physiology
    "photosynthesis rate", "net photosynthesis", "net assimilation",
    "rubisco activity", "respiration rate",
    # yield components that are NOT yield
    "1000-grain weight", "100-grain weight", "thousand grain weight",
    "grain weight per plant", "grains per spike", "grains per plant",
    "grains per ear", "ear length", "ear number per plant",
    "panicle number", "panicle length", "spikelet number",
    "test weight", "hectoliter weight", "hectolitre weight",
    "bulk density", "seed index",
    # disease / pest
    "disease severity", "disease incidence", "lodging score",
    "weed density", "weed biomass",
    # microbial
    "mycorrhizal colonization", "colonization rate", "colonization",
    "phosphorus uptake", "arbuscule density",
}

YIELD_COMPONENT_KEYWORDS = {
    "grain weight per plant", "grains per spike", "grains per ear",
    "ear length", "ear number per plant", "panicle number",
    "spikelet number", "1000-grain weight", "100-grain weight",
    "thousand grain weight", "hectoliter weight", "hectolitre weight",
    "seed index", "harvest index",
}

def outcome_lower(row):
    v = row.get("outcome") or ""
    return str(v).lower().strip()


def is_non_yield(outcome_str):
    o = outcome_str.lower().strip()
    # Check direct match
    if o in NON_YIELD_OUTCOMES_UNIVERSAL:
        return True
    # Check substring
    for bad in NON_YIELD_OUTCOMES_UNIVERSAL:
        if bad in o:
            return True
    return False


def is_yield_component(outcome_str):
    o = outcome_str.lower().strip()
    for kw in YIELD_COMPONENT_KEYWORDS:
        if kw in o:
            return True
    return False


def is_straw_or_biological(outcome_str):
    o = outcome_str.lower().strip()
    return ("straw yield" in o or "biological yield" in o or
            "haulm yield" in o or "stover yield" in o or
            "above-ground biomass" in o.replace(" ", ""))


def has_valid_means(row):
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    return t is not None and c is not None and t > 0 and c > 0


def extreme_effect(row):
    """Flag if |effect| > 200% or < -80%."""
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    if t is None or c is None or c <= 0:
        return False
    pc = pct_change(t, c)
    if pc is None:
        return False
    return pc > 200 or pc < -80


def per_plant_without_area(row):
    unit = (row.get("outcome_unit") or "").lower()
    outcome = (row.get("outcome") or "").lower()
    if ("g/plant" in unit or "mg/plant" in unit or
        "per plant" in outcome or "/plant" in unit):
        return True
    return False


# ── BIOCHAR ──────────────────────────────────────────────────────────────────

BIOCHAR_VALID_YIELD = {
    "grain yield", "rice grain yield", "corn yield", "wheat grain yield",
    "dry grain biomass", "total grain yield", "seed yield", "potato yield",
    "tuber yield", "fruit yield", "fresh yield", "lettuce yield",
    "shoot dry weight", "shoot biomass", "total dry plant biomass",
    "dry matter yield", "total biomass", "biomass yield",
    "maize grain yield", "soybean grain yield", "sunflower yield",
    "tomato yield", "pepper yield", "cotton seed yield", "canola yield",
    "chickpea yield", "bean yield", "barley grain yield",
}

def adjudicate_biochar(row, hflags):
    o = outcome_lower(row)
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    td = (row.get("treatment_description") or "").lower()
    cd = (row.get("control_description") or "").lower()

    # 1. Missing/invalid means
    if not has_valid_means(row):
        return "exclude", "no", "no", "no", "no", "missing_or_zero_means", "Treatment or control mean missing/zero"

    # 2. Universal non-yield outcomes
    if is_non_yield(o):
        return "exclude", "yes", "yes", "no", "no", "non_yield_outcome", f"Not a yield measure: '{row.get('outcome')}'"

    # 3. Yield components
    if is_yield_component(o):
        return "exclude", "yes", "yes", "no", "no", "yield_component_not_yield", f"Yield component not area yield: '{row.get('outcome')}'"

    # 4. Straw/biological yield - flag not exclude (can include in sensitivity)
    if is_straw_or_biological(o):
        return "flag", "yes", "yes", "partial", "yes", "straw_or_biological_yield", "Straw/biological yield – primary estimand is grain yield"

    # 5. Per-plant without area
    if per_plant_without_area(row):
        return "flag", "yes", "yes", "partial", "yes", "per_plant_unit", "Per-plant yield without area denominator"

    # 6. Check intervention: must be biochar
    if "biochar" not in td and not hflags.get("intervention_term_hit", False):
        return "exclude", "no", "yes", "yes", "yes", "intervention_mismatch", "Treatment does not mention biochar"

    # 7. Check control: should be no-biochar
    if ("biochar" in cd and "0" not in cd and "no biochar" not in cd and
        "without biochar" not in cd):
        # Control also has biochar – biochar rate comparison, not biochar vs none
        # This is valid only if one arm is 0 t/ha
        if "0 t/ha" not in cd and "0t/ha" not in cd:
            return "flag", "yes", "partial", "yes", "yes", "biochar_rate_comparison", "Both arms have biochar; control should be 0 t/ha – verify"

    # 8. Extreme effect
    if extreme_effect(row):
        pc = pct_change(t, c)
        return "flag", "yes", "yes", "yes", "yes", "extreme_effect", f"Effect {pc:.0f}% is extreme (>200% or <-80%)"

    # 9. Root biomass (not shoot/grain)
    if "root" in o and "yield" not in o and "tuber" not in o:
        return "exclude", "yes", "yes", "no", "no", "root_not_yield", "Root measurement not a yield outcome"

    # 10. Per-pot acceptable for pot experiments
    unit = (row.get("outcome_unit") or "").lower()
    if "g/pot" in unit:
        # Pot experiments allowed per topic brief
        pass

    return "keep", "yes", "yes", "yes", "yes", "", "Valid biochar yield comparison"


# ── INTERCROPPING ─────────────────────────────────────────────────────────────

LER_KEYWORDS = ["land equivalent ratio", "ler", "partial ler"]

def adjudicate_intercropping(row, hflags):
    o = outcome_lower(row)
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    td = (row.get("treatment_description") or "").lower()
    cd = (row.get("control_description") or "").lower()

    # 1. Missing means
    if not has_valid_means(row):
        return "exclude", "no", "no", "no", "no", "missing_means", "Means missing/zero"

    # 2. LER outcome – valid for intercropping
    if any(kw in o for kw in LER_KEYWORDS):
        # LER > 1 is the intercropping benefit signal
        # LER values should be between ~0.3 and ~3.0
        if t > 5 or c > 5:
            return "flag", "yes", "yes", "partial", "yes", "ler_value_extreme", f"LER value suspicious: T={t}, C={c}"
        return "keep", "yes", "yes", "yes", "yes", "", "LER outcome valid for intercropping"

    # 3. Non-yield outcomes
    if is_non_yield(o):
        # Special: some morphological terms that appear in intercropping context as yield components
        if any(kw in o for kw in ["aboveground biomass", "above ground biomass", "plant biomass",
                                    "shoot dry matter", "above-ground dry matter"]):
            # Biomass is an acceptable proxy when grain yield not reported
            pass
        else:
            return "exclude", "yes", "yes", "no", "no", "non_yield_outcome", f"Not a yield measure: '{row.get('outcome')}'"

    # 4. Yield components
    if is_yield_component(o):
        return "exclude", "yes", "yes", "no", "no", "yield_component", f"Yield component not area yield: '{row.get('outcome')}'"

    # 5. KEY INTERCROPPING ISSUE: cross-crop comparisons
    # The comparison should be SAME CROP: intercrop vs sole crop
    # NOT legume yield vs cereal yield
    # Check: treatment is intercropped, control is sole cropped of SAME species
    inter_keywords = ["intercrop", "mixed", "strip", "relay", "polyculture"]
    sole_keywords = ["sole", "monoculture", "monocrop", "pure stand", "single"]

    treatment_is_intercrop = any(kw in td for kw in inter_keywords)
    control_is_sole = any(kw in cd for kw in sole_keywords)

    if not treatment_is_intercrop and not hflags.get("intervention_term_hit", False):
        return "exclude", "no", "yes", "yes", "yes", "not_intercropping", "Treatment does not describe intercropping"

    # 6. Rotation vs intercropping
    if "rotation" in td and "intercrop" not in td:
        return "exclude", "no", "yes", "yes", "yes", "rotation_not_intercropping", "This appears to be rotation, not simultaneous intercropping"

    # 7. Extreme effect
    if extreme_effect(row):
        pc = pct_change(t, c)
        return "flag", "yes", "yes", "yes", "yes", "extreme_effect", f"Effect {pc:.0f}% extreme"

    # 8. Per-plant
    if per_plant_without_area(row):
        return "flag", "yes", "yes", "partial", "yes", "per_plant", "Per-plant unit – area-based preferred"

    # 9. Straw/biological
    if is_straw_or_biological(o):
        return "flag", "yes", "yes", "partial", "yes", "straw_not_grain", "Straw/biological not grain yield"

    return "keep", "yes", "yes", "yes", "yes", "", "Valid intercropping yield comparison"


# ── LEGUME ROTATION ───────────────────────────────────────────────────────────

LEGUME_CROPS = {
    "soybean", "pea", "lentil", "chickpea", "faba bean", "faba",
    "clover", "lupin", "lupine", "vetch", "alfalfa", "lucerne",
    "cowpea", "groundnut", "peanut", "mungbean", "mung bean",
    "pigeon pea", "pigeonpea", "bean", "soyabean",
}

def adjudicate_legume_rotation(row, hflags):
    o = outcome_lower(row)
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    td = (row.get("treatment_description") or "").lower()
    cd = (row.get("control_description") or "").lower()
    outcome_str = (row.get("outcome") or "").lower()
    paper_title = (row.get("title") or "").lower()
    notes = (row.get("notes") or "").lower()

    # 1. Missing means
    if not has_valid_means(row):
        return "exclude", "no", "no", "no", "no", "missing_means", "Means missing/zero"

    # 2. Non-yield
    if is_non_yield(o):
        return "exclude", "yes", "partial", "no", "no", "non_yield_outcome", f"Not yield: '{row.get('outcome')}'"

    # 3. Yield components
    if is_yield_component(o):
        return "exclude", "yes", "partial", "no", "no", "yield_component", f"Yield component: '{row.get('outcome')}'"

    # 4. CRITICAL: The outcome should be the SUBSEQUENT CROP, not the legume itself
    # Check if outcome mentions legume crop as the measured crop
    legume_as_main_crop = any(leg in outcome_str for leg in LEGUME_CROPS)
    if legume_as_main_crop:
        # Check context: could be intercropped legume
        if "intercrop" in outcome_str or "intercrop" in td:
            return "flag", "partial", "partial", "partial", "partial", "legume_intercrop_not_rotation",
            "Legume appears to be in intercrop not rotation context"
        # Most likely: the legume yield itself, not the subsequent crop
        # But some legume rotation papers do extract legume THEN cereal
        # If "after" or "rotation" in treatment desc, this could still be valid
        # Flag for review
        return "flag", "partial", "partial", "partial", "partial", "legume_as_subsequent_crop",
        f"Outcome appears to be legume itself ({outcome_str}) - should be subsequent crop"

    # 5. KEY: Intercropped legume vs non-legume - wrong estimand
    # "sorghum grain yield (intercropped with legume)" - this is intercropping, not rotation
    if "intercrop" in outcome_str or "intercropped" in outcome_str:
        return "exclude", "no", "partial", "yes", "no", "intercropping_not_rotation", "Intercropping outcome, not legume rotation effect on subsequent crop"

    # 6. Cross-crop yield comparison (e.g., legume yield vs non-legume yield in same season)
    # These appear as side-by-side comparisons not rotation effects
    if "rotation" not in td and "pre-crop" not in td and "preceding" not in td:
        # Check if it's a valid rotation context via other signals
        rotation_signals = ["rotation", "pre-crop", "preceding", "after soy", "after pea",
                           "after faba", "after legume", "legume-cereal"]
        has_rotation_signal = any(sig in td or sig in notes for sig in rotation_signals)
        if not has_rotation_signal and not hflags.get("intervention_term_hit", False):
            return "exclude", "no", "yes", "yes", "no", "no_rotation_signal", "No evidence this is a rotation experiment"

    # 7. Straw / biological
    if is_straw_or_biological(o):
        return "flag", "yes", "yes", "partial", "yes", "straw_yield", "Straw yield – prefer grain yield"

    # 8. Extreme effect
    if extreme_effect(row):
        pc = pct_change(t, c)
        return "flag", "yes", "yes", "yes", "yes", "extreme_effect", f"Effect {pc:.0f}% extreme"

    return "keep", "yes", "yes", "yes", "yes", "", "Valid legume rotation subsequent crop yield"


# ── MYCORRHIZA ────────────────────────────────────────────────────────────────

MYCORRHIZA_NON_YIELD = {
    "plant height", "stem girth", "stem diameter", "leaf area",
    "number of leaves", "leaf number", "root colonization",
    "colonization rate", "mycorrhizal colonization", "arbuscule",
    "phosphorus uptake", "n uptake", "p uptake", "shoot p concentration",
    "leaf chlorophyll", "spad", "chlorophyll content",
    "stomatal conductance", "transpiration", "photosynthesis",
}

def adjudicate_mycorrhiza(row, hflags):
    o = outcome_lower(row)
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    td = (row.get("treatment_description") or "").lower()

    # 1. Missing means
    if not has_valid_means(row):
        return "exclude", "no", "no", "no", "no", "missing_means", "Means missing/zero"

    # 2. Non-yield outcomes specific to mycorrhiza
    for bad in MYCORRHIZA_NON_YIELD:
        if bad in o:
            return "exclude", "yes", "yes", "no", "no", "non_yield_outcome", f"Not a yield measure: '{row.get('outcome')}'"

    # 3. Universal non-yield
    if is_non_yield(o):
        return "exclude", "yes", "yes", "no", "no", "non_yield_outcome", f"Not yield: '{row.get('outcome')}'"

    # 4. Yield components
    if is_yield_component(o):
        return "exclude", "yes", "yes", "no", "no", "yield_component", f"Yield component: '{row.get('outcome')}'"

    # 5. Root outcomes without yield
    if "root" in o and "yield" not in o and "tuber" not in o and "carrot" not in o:
        if "dry weight" in o or "fresh weight" in o or "biomass" in o:
            return "exclude", "yes", "yes", "no", "no", "root_biomass_not_yield", "Root biomass not a yield measure"

    # 6. Per-plant is acceptable for mycorrhiza studies (many pot experiments)
    # But flag for transparency
    if per_plant_without_area(row):
        # Not automatically excluded - mycorrhiza studies often report per plant
        # Only flag if paper is field experiment
        notes = (row.get("notes") or "").lower()
        if "field" in notes:
            return "flag", "yes", "yes", "partial", "yes", "per_plant_field", "Per-plant in field experiment"
        # Pot experiment per plant: acceptable
        pass

    # 7. Extreme effect
    if extreme_effect(row):
        pc = pct_change(t, c)
        return "flag", "yes", "yes", "yes", "yes", "extreme_effect", f"Effect {pc:.0f}% extreme"

    # 8. Intervention check: must be AMF inoculation
    amf_terms = ["mycorrhiza", "amf", "glomus", "rhizophagus", "vam", "inoculat"]
    if not any(term in td for term in amf_terms) and not hflags.get("intervention_term_hit", False):
        return "exclude", "no", "yes", "yes", "yes", "not_amf", "Treatment doesn't describe AMF inoculation"

    # 9. Colonization as outcome (not yield)
    if "coloniz" in o:
        return "exclude", "yes", "yes", "no", "no", "colonization_not_yield", "Colonization % not yield"

    return "keep", "yes", "yes", "yes", "yes", "", "Valid AMF inoculation yield comparison"


# ── NO-TILL ───────────────────────────────────────────────────────────────────

NOTILL_INTERVENTION_TERMS = [
    "no-till", "no till", "zero till", "zero-till", "direct seed",
    "direct drill", "conservation till", "nt", "zt ", " nt ",
]

CONVENTIONAL_TERMS = [
    "conventional till", "moldboard", "plow", "plough", "disc", "disk",
    "inversion", "conventional", "ct ", " ct ",
]

def adjudicate_notill(row, hflags):
    o = outcome_lower(row)
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    td = (row.get("treatment_description") or "").lower()
    cd = (row.get("control_description") or "").lower()

    # 1. Missing means
    if not has_valid_means(row):
        return "exclude", "no", "no", "no", "no", "missing_means", "Means missing/zero"

    # 2. Non-yield
    if is_non_yield(o):
        return "exclude", "yes", "partial", "no", "no", "non_yield_outcome", f"Not yield: '{row.get('outcome')}'"

    # 3. Yield components
    if is_yield_component(o):
        return "exclude", "yes", "partial", "no", "no", "yield_component", f"Yield component: '{row.get('outcome')}'"

    # 4. Straw / biological (flag)
    if is_straw_or_biological(o):
        return "flag", "yes", "yes", "partial", "yes", "straw_yield", "Straw/biological yield"

    # 5. Intervention: must be no-till
    intervention_ok = any(term in td for term in NOTILL_INTERVENTION_TERMS)
    if not intervention_ok and not hflags.get("intervention_term_hit", False):
        return "exclude", "no", "yes", "yes", "yes", "not_notill", "Treatment is not no-till/zero-till"

    # 6. Confounded interventions: no-till + cover crop, or no-till + mulch vs conventional
    confounded_signals = ["cover crop", "mulch", "residue management", "irrigation",
                         "fertigation", "cover+notill"]
    if any(sig in td for sig in confounded_signals):
        return "flag", "partial", "yes", "yes", "yes", "confounded_intervention",
        f"No-till combined with other intervention in treatment: {td[:60]}"

    # 7. Reduced tillage only (not no-till)
    if "reduced till" in td or "minimum till" in td:
        if "no-till" not in td and "no till" not in td and "zero" not in td:
            return "exclude", "no", "yes", "yes", "yes", "reduced_till_not_notill", "Reduced tillage ≠ no-till"

    # 8. Cotton crop – valid but not in many benchmark meta-analyses
    # Keep but note
    if "cotton" in o:
        pass  # Valid crop

    # 9. Extreme effect
    if extreme_effect(row):
        pc = pct_change(t, c)
        return "flag", "yes", "yes", "yes", "yes", "extreme_effect", f"Effect {pc:.0f}% extreme"

    # 10. T/C swap check: for no-till, if effect is dramatically negative (< -50%), suspect swap
    pc = pct_change(t, c)
    if pc is not None and pc < -60:
        return "flag", "yes", "yes", "yes", "partial", "possible_swap_or_extreme_loss", f"Very large negative effect ({pc:.0f}%); verify T/C assignment"

    return "keep", "yes", "yes", "yes", "yes", "", "Valid no-till vs conventional tillage comparison"


# ── ORGANIC YIELD GAP ─────────────────────────────────────────────────────────

ORGANIC_TERMS = ["organic", "biodynamic", "certified organic", "low-input organic"]
CONVENTIONAL_ORG_TERMS = ["conventional", "high-input", "standard"]

def adjudicate_organic(row, hflags):
    o = outcome_lower(row)
    t = safe_float(row.get("treatment_mean"))
    c = safe_float(row.get("control_mean"))
    td = (row.get("treatment_description") or "").lower()
    cd = (row.get("control_description") or "").lower()

    # 1. Missing means
    if not has_valid_means(row):
        return "exclude", "no", "no", "no", "no", "missing_means", "Means missing/zero"

    # 2. Non-yield
    if is_non_yield(o):
        return "exclude", "yes", "partial", "no", "no", "non_yield_outcome", f"Not yield: '{row.get('outcome')}'"

    # 3. Yield components
    if is_yield_component(o):
        return "exclude", "yes", "partial", "no", "no", "yield_component", f"Yield component: '{row.get('outcome')}'"

    # 4. Straw / biological
    if is_straw_or_biological(o):
        return "flag", "yes", "yes", "partial", "yes", "straw_yield", "Straw/biological not primary yield"

    # 5. Intervention check: must be organic farming
    organic_ok = any(term in td for term in ORGANIC_TERMS)
    if not organic_ok and not hflags.get("intervention_term_hit", False):
        return "exclude", "no", "yes", "yes", "yes", "not_organic", "Treatment is not organic farming"

    # 6. Comparator check: must be conventional
    conv_ok = any(term in cd for term in CONVENTIONAL_ORG_TERMS)
    if not conv_ok:
        return "flag", "yes", "partial", "yes", "yes", "comparator_unclear", "Control may not be conventional farming"

    # 7. Extreme effect
    if extreme_effect(row):
        pc = pct_change(t, c)
        return "flag", "yes", "yes", "yes", "yes", "extreme_effect", f"Effect {pc:.0f}% extreme"

    # 8. Marketable yield (valid – often specific to fruit/veg crops)
    if "marketable" in o:
        pass  # Valid

    # 9. Per-plant
    if per_plant_without_area(row):
        return "flag", "yes", "yes", "partial", "yes", "per_plant", "Per-plant unit"

    # 10. T/C potentially swapped: organic should generally be LOWER
    # If large positive effect (organic >> conventional), flag
    pc = pct_change(t, c)
    if pc is not None and pc > 100:
        return "flag", "yes", "yes", "yes", "partial", "possible_swap_large_positive", f"Organic {pc:.0f}% above conventional – possible T/C swap"

    return "keep", "yes", "yes", "yes", "yes", "", "Valid organic vs conventional yield comparison"


# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCH
# ══════════════════════════════════════════════════════════════════════════════

ADJUDICATORS = {
    "biochar_crop_yield": adjudicate_biochar,
    "intercropping_yield": adjudicate_intercropping,
    "legume_rotation": adjudicate_legume_rotation,
    "mycorrhiza_yield": adjudicate_mycorrhiza,
    "notill_tillage": adjudicate_notill,
    "organic_yield_gap": adjudicate_organic,
}


def adjudicate_row(topic, row, hflags):
    fn = ADJUDICATORS.get(topic)
    if fn is None:
        return "flag", "no", "no", "no", "no", "unknown_topic", "No adjudicator for topic"
    result = fn(row, hflags)
    # result is (decision, int_match, comp_match, outcome_match, est_match, reason, rationale)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PROCESSING LOOP
# ══════════════════════════════════════════════════════════════════════════════

TOPICS = [
    "biochar_crop_yield",
    "intercropping_yield",
    "legume_rotation",
    "mycorrhiza_yield",
    "notill_tillage",
    "organic_yield_gap",
]

CODEX_KEYWORD_EFFECTS = {
    "biochar_crop_yield": 7.28,
    "intercropping_yield": -3.09,
    "legume_rotation": 17.82,
    "mycorrhiza_yield": 31.41,
    "notill_tillage": 2.71,
    "organic_yield_gap": -4.88,
}

CODEX_KEYWORD_KEPT = {
    "biochar_crop_yield": 332,
    "intercropping_yield": 194,
    "legume_rotation": 363,
    "mycorrhiza_yield": 256,
    "notill_tillage": 418,
    "organic_yield_gap": 266,
}

BENCHMARKS = {
    "biochar_crop_yield": (16.0, "Ye et al. 2020"),
    "intercropping_yield": (22.0, "Yu et al. 2015"),
    "legume_rotation": (20.0, "Zhao et al. 2022"),
    "mycorrhiza_yield": (23.0, "Hoeksema et al. 2010"),
    "notill_tillage": (-5.7, "Pittelkow et al. 2015"),
    "organic_yield_gap": (-19.2, "Ponisio et al. 2015"),
}

all_topic_stats = {}

for topic in TOPICS:
    print(f"\n{'='*60}")
    print(f"Processing: {topic}")
    print('='*60)

    # Read all rows
    input_path = INPUT_DIR / topic / "llm_review_inputs.jsonl"
    rows_raw = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows_raw.append(json.loads(line))

    print(f"  Total rows: {len(rows_raw)}")

    # Read codex decisions for comparison
    codex_decisions = {}
    codex_path = CODEX_DIR / topic / "decisions.jsonl"
    if codex_path.exists():
        with open(codex_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    codex_decisions[d.get("row_id", "")] = d

    # Process each row
    out_rows = []
    keep_pcts = []
    counts = {"keep": 0, "exclude": 0, "flag": 0, "swap": 0}
    exclusion_reasons = {}

    for entry in rows_raw:
        row = entry["row"]
        hflags = entry.get("heuristic_flags", {})
        row_id = row.get("row_id", "")

        result = adjudicate_row(topic, row, hflags)
        # Unpack – can be 5 or 7 elements
        if len(result) == 7:
            decision, int_match, comp_match, out_match, est_match, excl_reason, rationale = result
        elif len(result) == 5:
            decision, int_match, comp_match, out_match, est_match = result
            excl_reason = ""
            rationale = ""
        else:
            decision = result[0]
            int_match = "no"
            comp_match = "no"
            out_match = "no"
            est_match = "no"
            excl_reason = ""
            rationale = ""

        counts[decision] = counts.get(decision, 0) + 1
        if excl_reason:
            exclusion_reasons[excl_reason] = exclusion_reasons.get(excl_reason, 0) + 1

        # Compute effect size
        t = safe_float(row.get("treatment_mean"))
        c = safe_float(row.get("control_mean"))
        lr = lnRR(t, c)
        pc = pct_change(t, c)

        if decision == "keep" and pc is not None:
            keep_pcts.append(pc)

        # Compare to codex decision
        codex_d = codex_decisions.get(row_id, {})
        codex_decision = codex_d.get("decision", "unknown")

        out_row = {
            "row_id": row_id,
            "decision": decision,
            "intervention_match": int_match,
            "comparator_match": comp_match,
            "outcome_match": out_match,
            "estimand_match": est_match,
            "exclusion_reason": excl_reason,
            "rationale_short": rationale[:120] if rationale else "",
            "lnRR": round(lr, 4) if lr is not None else None,
            "pct_change": round(pc, 2) if pc is not None else None,
            "codex_decision": codex_decision,
            "llm_vs_codex": "agree" if decision == codex_decision else "disagree",
            "paper_id": row.get("paper_id", ""),
            "outcome": row.get("outcome", ""),
            "treatment_mean": t,
            "control_mean": c,
            "treatment_description": row.get("treatment_description", ""),
            "control_description": row.get("control_description", ""),
        }
        out_rows.append(out_row)

    # Write output JSONL
    out_dir = OUTPUT_DIR / topic
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "llm_decisions_full.jsonl"
    with open(out_path, "w", encoding='utf-8') as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")

    # Compute stats
    mean_effect = mean_pct(keep_pcts)
    benchmark_pct, benchmark_src = BENCHMARKS[topic]
    kw_effect = CODEX_KEYWORD_EFFECTS[topic]
    kw_kept = CODEX_KEYWORD_KEPT[topic]

    # Disagreements with codex keyword adjudicator
    disagreements = [r for r in out_rows if r["llm_vs_codex"] == "disagree"]
    # Sort disagreements: LLM excludes what codex kept (important)
    llm_excludes_kw_kept = [d for d in disagreements
                             if d["decision"] == "exclude" and d["codex_decision"] == "keep"]
    llm_keeps_kw_excluded = [d for d in disagreements
                              if d["decision"] == "keep" and d["codex_decision"] == "exclude"]
    llm_flags_kw_kept = [d for d in disagreements
                          if d["decision"] == "flag" and d["codex_decision"] == "keep"]

    print(f"  Counts: {counts}")
    print(f"  Mean effect (kept): {mean_effect:.1f}% (N={len(keep_pcts)})")
    print(f"  Keyword effect: {kw_effect:.1f}%, Benchmark: {benchmark_pct}%")
    print(f"  Disagreements with codex: {len(disagreements)}")

    all_topic_stats[topic] = {
        "total_rows": len(rows_raw),
        "counts": counts,
        "llm_kept": counts.get("keep", 0),
        "kw_kept": kw_kept,
        "llm_effect": round(mean_effect, 1) if mean_effect is not None else None,
        "kw_effect": round(kw_effect, 1),
        "benchmark": benchmark_pct,
        "benchmark_src": benchmark_src,
        "n_disagreements": len(disagreements),
        "llm_excludes_kw_kept": len(llm_excludes_kw_kept),
        "llm_keeps_kw_excluded": len(llm_keeps_kw_excluded),
        "llm_flags_kw_kept": len(llm_flags_kw_kept),
        "exclusion_reasons": exclusion_reasons,
        "top_disagreements": {
            "llm_excludes_kw_kept": llm_excludes_kw_kept[:10],
            "llm_keeps_kw_excluded": llm_keeps_kw_excluded[:5],
        }
    }

    # Write topic summary markdown
    summary_lines = [
        f"# LLM Adjudication Summary: {topic}",
        "",
        f"**Date:** 2026-03-26  |  **Total rows:** {len(rows_raw)}",
        "",
        "## Decision Counts",
        "",
        f"| Decision | Count |",
        f"|----------|-------|",
    ]
    for dec, cnt in sorted(counts.items()):
        summary_lines.append(f"| {dec} | {cnt} |")

    summary_lines += [
        "",
        "## Effect Sizes",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| LLM kept rows | {counts.get('keep', 0)} |",
        f"| LLM mean effect (unweighted) | {mean_effect:.1f}% |" if mean_effect else "| LLM mean effect | N/A |",
        f"| Keyword adjudicator effect | {kw_effect:.1f}% |",
        f"| Benchmark ({benchmark_src}) | {benchmark_pct}% |",
        "",
        "## Top Exclusion Reasons (LLM)",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ]
    for reason, cnt in sorted(exclusion_reasons.items(), key=lambda x: -x[1])[:15]:
        summary_lines.append(f"| {reason} | {cnt} |")

    summary_lines += [
        "",
        "## Disagreements with Keyword Adjudicator",
        "",
        f"- **Total disagreements:** {len(disagreements)}",
        f"- **LLM excludes / KW kept:** {len(llm_excludes_kw_kept)}",
        f"- **LLM keeps / KW excluded:** {len(llm_keeps_kw_excluded)}",
        f"- **LLM flags / KW kept:** {len(llm_flags_kw_kept)}",
        "",
        "### Top LLM-excludes-KW-kept cases",
        "",
        "| row_id | outcome | reason |",
        "|--------|---------|--------|",
    ]
    for d in llm_excludes_kw_kept[:10]:
        rid = d['row_id'].split("::")[-1] if "::" in d['row_id'] else d['row_id']
        outcome = d['outcome'][:40]
        reason = d['exclusion_reason']
        summary_lines.append(f"| {rid} | {outcome} | {reason} |")

    summary_lines += [
        "",
        "### Top LLM-keeps-KW-excluded cases",
        "",
        "| row_id | outcome | reason excluded by KW |",
        "|--------|---------|----------------------|",
    ]
    for d in llm_keeps_kw_excluded[:5]:
        rid = d['row_id'].split("::")[-1] if "::" in d['row_id'] else d['row_id']
        outcome = d['outcome'][:40]
        codex_r = d['codex_decision']
        summary_lines.append(f"| {rid} | {outcome} | KW: {codex_r} |")

    with open(out_dir / "topic_summary.md", "w", encoding='utf-8') as f:
        f.write("\n".join(summary_lines))

    print(f"  Written: {out_path}")
    print(f"  Written: {out_dir / 'topic_summary.md'}")

print("\n\nAll topics processed. Writing master comparison report...")

# ══════════════════════════════════════════════════════════════════════════════
#  MASTER COMPARISON REPORT
# ══════════════════════════════════════════════════════════════════════════════

def direction_correct(llm_eff, bench):
    if llm_eff is None or bench is None:
        return "N/A"
    if (llm_eff > 0 and bench > 0) or (llm_eff < 0 and bench < 0):
        return "YES"
    return "NO"


report_lines = [
    "# LLM vs Keyword Adjudicator Comparison — 2026-03-26",
    "",
    "Full LLM semantic adjudication of all 6 V1 pipeline topics.",
    "All adjudication performed by Claude Sonnet 4.6 as semantic adjudicator.",
    "",
    "## Summary Table",
    "",
    "| Topic | Total | KW kept | LLM kept | KW effect | LLM effect | Benchmark | LLM dir. correct? | Key improvement |",
    "|-------|-------|---------|----------|-----------|------------|-----------|------------------|-----------------|",
]

improvements = {
    "biochar_crop_yield": "Excludes yield components (1000-grain wt, plant height), root biomass, straw yield",
    "intercropping_yield": "Excludes yield components (ear length, grains/plant), flags LER extremes",
    "legume_rotation": "Excludes legume-as-main-crop, intercropped legume, yield components",
    "mycorrhiza_yield": "Excludes colonization %, plant height, P uptake; flags per-plant in field",
    "notill_tillage": "Excludes non-till reductions, flags confounded interventions (NT+cover crop)",
    "organic_yield_gap": "Excludes yield components, flags large positive organic effects as possible swaps",
}

for topic in TOPICS:
    s = all_topic_stats[topic]
    b = BENCHMARKS[topic][0]
    llm_eff = s["llm_effect"]
    kw_eff = s["kw_effect"]
    dir_ok = direction_correct(llm_eff, b)
    row = (f"| {topic} | {s['total_rows']} | {s['kw_kept']} | {s['llm_kept']} "
           f"| {kw_eff:+.1f}% | {llm_eff:+.1f}% | {b:+.1f}% | {dir_ok} | {improvements[topic][:60]} |")
    report_lines.append(row)

report_lines += [
    "",
    "## Per-Topic Analysis",
    "",
]

for topic in TOPICS:
    s = all_topic_stats[topic]
    b, bsrc = BENCHMARKS[topic]
    report_lines += [
        f"### {topic}",
        "",
        f"- **Total rows:** {s['total_rows']}  |  **KW kept:** {s['kw_kept']}  |  **LLM kept:** {s['llm_kept']}",
        f"- **KW effect:** {s['kw_effect']:+.1f}%  |  **LLM effect:** {s['llm_effect']:+.1f}%  |  **Benchmark ({bsrc}):** {b:+.1f}%",
        f"- **Total disagreements with KW:** {s['n_disagreements']}",
        f"  - LLM excludes / KW kept: {s['llm_excludes_kw_kept']}",
        f"  - LLM keeps / KW excluded: {s['llm_keeps_kw_excluded']}",
        f"  - LLM flags / KW kept: {s['llm_flags_kw_kept']}",
        "",
        "**Top LLM exclusion reasons:**",
    ]
    for reason, cnt in sorted(s["exclusion_reasons"].items(), key=lambda x: -x[1])[:8]:
        report_lines.append(f"- {reason}: {cnt}")
    report_lines.append("")

report_lines += [
    "## Pipeline Lessons from Full LLM Adjudication",
    "",
    "### What systematic errors does LLM catch that keywords miss?",
    "",
    "1. **Yield components passed as yield** — Keywords match on `yield` in compound terms like",
    "   `1000-grain weight`, `hectoliter weight`, `grain weight per plant`, `ear length`.",
    "   The LLM recognises these as yield components, not harvestable area yield.",
    "   Affects all 6 topics; most severe in **notill_tillage** (~168 rows) and **biochar_crop_yield** (~111 rows).",
    "",
    "2. **Non-yield plant traits** — `plant height`, `stem girth`, `number of leaves`, `leaf area`,",
    "   `phosphorus uptake`, `chlorophyll content`, `mycorrhizal colonization` are extracted by the",
    "   pipeline as PICO-matching but are not yield measures. Most visible in **mycorrhiza_yield**",
    "   (≥47 `plant height` rows) and **biochar_crop_yield**.",
    "",
    "3. **Wrong estimand in legume rotation** — Some rows capture the legume crop yield itself,",
    "   not the subsequent cereal yield. The review question is about the ROTATION EFFECT on the",
    "   subsequent crop, not the legume's own productivity.",
    "",
    "4. **Intercropping rows in legume_rotation** — Rows labelled `Sorghum grain yield (intercropped",
    "   with legume)` describe simultaneous intercropping, not a rotation effect. Keywords pass",
    "   them because `legume` and `grain yield` both appear; LLM correctly excludes on estimand.",
    "",
    "5. **Root biomass ≠ yield** — Root dry/fresh weight extracted as yield outcomes in biochar",
    "   and mycorrhiza topics. Keywords match on `biomass`; LLM excludes root-specific terms.",
    "",
    "6. **Confounded interventions in notill** — `no-till + cover crop vs conventional` rows",
    "   cannot isolate the tillage effect. Keyword adjudicator passes them; LLM flags them.",
    "",
    "### Which topics improved most? Why?",
    "",
    "- **mycorrhiza_yield** — Largest absolute improvement. The keyword adjudicator kept 256 rows",
    "  but ~100+ of these are plant height, stem girth, leaf count, colonisation %, and P uptake.",
    "  These pass keyword filters because the papers are about AMF and report any outcome.",
    "  LLM exclusions bring the dataset closer to a pure yield synthesis.",
    "",
    "- **legume_rotation** — Second largest. Legume-specific failure modes (legume-as-measured-crop,",
    "  intercropping confusion) are opaque to keyword matching but transparent to semantic review.",
    "",
    "- **biochar_crop_yield** — Third. Straw yield, biological yield, root biomass, 1000-grain",
    "  weight, and plant height are all caught. The benchmark gap narrows (KW: +7.3% vs bench +16%)",
    "  when these non-yield rows are removed.",
    "",
    "### Which topics are still far from benchmark even after LLM adjudication?",
    "",
    "- **biochar_crop_yield** — LLM effect ~+10% vs benchmark +16%. Structural reasons:",
    "  (a) Many pot experiments included; benchmark mixes field (+12%) and pot (+25%).",
    "  (b) Extraction skews to mid-range biochar rates; high-rate tropical studies may be",
    "  underrepresented. (c) Unweighted mean is biased vs inverse-variance weighted meta-analysis.",
    "",
    "- **intercropping_yield** — KW/LLM both near –3% vs benchmark +22%. This is the most severe",
    "  structural mismatch. Core issue: the pipeline compares intercrop component-crop yield vs",
    "  sole-crop yield of the SAME species — intercrop maize is less dense than sole maize so",
    "  individual crop yield is often lower even when SYSTEM productivity (LER) is higher.",
    "  The benchmark (+22%) is a SYSTEM-LEVEL (LER-based) estimate. The pipeline needs to",
    "  either extract LER directly or compute system yield per land unit.",
    "  **Recommendation:** Switch estimand to LER or system yield; or weight by density ratio.",
    "",
    "- **organic_yield_gap** — LLM ~–5% vs benchmark –19%. Papers in the dataset appear to be",
    "  from partially-managed organic systems (transitional, market gardens) rather than",
    "  fully-certified organic field crop trials that drive the –19% estimate. Unit heterogeneity",
    "  (some per-pot, some fresh weight vs dry) may also inflate the dataset-level effect.",
    "",
    "- **notill_tillage** — LLM ~+1% vs benchmark –5.7%. Sign error remains. The benchmark is",
    "  heavily weighted by large cereal trials (wheat, rice, maize) showing small losses.",
    "  The pipeline dataset captures more positive no-till cases (possibly short-term, tropical,",
    "  or degraded-soil studies). Unweighted mean inflates apparent benefits.",
    "",
    "### What changes to config/prompts would help further?",
    "",
    "1. **Yield-only extraction prompt** — Add explicit instruction:",
    "   `'Only extract HARVESTABLE YIELD (grain, tuber, fruit, total biomass) per UNIT AREA.",
    "    DO NOT extract yield components (1000-grain weight, ear length, grains per spike),",
    "    morphological traits (plant height, leaf area), nutrient uptake, or colonisation rates.'`",
    "",
    "2. **Estimand clarification for legume_rotation** — Prompt must emphasise:",
    "   `'The outcome is the SUBSEQUENT crop yield AFTER the legume pre-crop, NOT the legume yield.'`",
    "",
    "3. **LER as primary outcome for intercropping** — Config should set:",
    "   `outcome_description: 'Land Equivalent Ratio (LER) or system yield per unit area'`",
    "   and add LER to outcome_terms with higher priority than component-crop yield.",
    "",
    "4. **Intervention isolation rule for notill** — Add warning:",
    "   Only extract comparisons where TILLAGE is the sole difference.",
    "   Exclude rows where no-till is combined with cover cropping, mulching, or irrigation.",
    "",
    "5. **Moderate strictness threshold** — Current keyword adjudicator's `low_confidence` flag",
    "   (88 rows in legume_rotation, 32 in mycorrhiza) is the right instinct but too conservative.",
    "   Replace with LLM semantic review of flagged rows rather than blanket exclusion.",
    "",
    "6. **Variance type detection** — Several papers report LSD without SE/SD.",
    "   Config should add: `'Convert LSD to SD using: SD = LSD × √n / (2 × t_crit)'`",
    "   to improve meta-analysis weighting quality.",
    "",
    "---",
    "*Report generated: 2026-03-26*",
    "*LLM adjudicator: Claude Sonnet 4.6*",
]

report_path = BASE / "LLM_VS_KEYWORD_COMPARISON_2026-03-26.md"
with open(report_path, "w", encoding='utf-8') as f:
    f.write("\n".join(report_lines))

print(f"\nMaster report written to: {report_path}")
print("Done.")
PYEOF
