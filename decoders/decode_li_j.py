# -*- coding: utf-8 -*-
"""
AI-side decoder for the Li J 2022 (plant-biostimulant crop-yield) dataset,
REBUILD 2026-08-19.

Adapted from the submitted decoder
    meta_analysis_extractor/decode_li2022_ai.py
        (the only AI-side Li-2022 decoder in the repository; `DECODER` string
         "claude-opus-4-8/decode_li2022_ai_v2" matches the deposited keys)

Exactly one thing changes about the science: the AI-side SOURCE is now the frozen
March-2026 single-model Claude agent JSONs
    01_INPUTS_FROZEN/li_j/*_agent.json      (49 files, 1053 records)
instead of the three-model consensus files `output/li2022_combined/*_consensus.json`.

All classification / inference logic below the "VERBATIM" banner is copied
character-for-character from the submitted decoder.  Every deviation is marked
with a `CHANGE n:` comment and is listed in 06_LEDGER/li_j_DECODER_LEDGER.md.

HARD RULES OBSERVED
  * OUTCOME-BLIND: no key column is derived from, conditioned on, or selected
    using treatment_mean / control_mean / effect_pct / variance, and no GT value
    is ever read.  The only GT artefact consulted is the *paper_id vocabulary*
    (structural column) in 03_KEYS/gt/li_j/*.csv, used for the paper crosswalk
    and for filename mirroring.
  * NO VALUE MATCHING: the deposited AI key tables are never read.
  * DETERMINISTIC: stdlib only, no randomness, sorted iteration everywhere.
  * NO SILENT DROPS: every source record is either a key row or is counted in
    the exclusion tally printed at the end.

Output: one CSV per paper in 03_KEYS/ai_rebuilt/li_j/, canonical 18 columns.
"""
import csv
import glob
import io
import json
import os
import re
import sys
import unicodedata

# ---------------------------------------------------------------------------
# CHANGE 1: input path repointed at the frozen single-model agent JSONs.
#           Output path repointed at the rebuild key directory, and the output
#           format is the canonical 18-column CSV written directly (the
#           submitted decoder emitted JSONL that `matching/keys_from_jsonl.py`
#           then converted with csv.DictWriter; the conversion is inlined here
#           with the same csv.DictWriter, so quoting behaviour is identical).
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

SRC = os.path.join(ROOT, "01_INPUTS_FROZEN", "li_j")
OUT = os.path.join(ROOT, "03_KEYS", "ai_rebuilt", "li_j")
GT_KEYS = os.path.join(ROOT, "03_KEYS", "gt", "li_j")

DECODER = "rebuild_2026-08-19/li_j"

COLS = ["row_id", "side", "paper_id", "outcome_canonical", "crop",
        "treatment_level", "co_amendment", "co_amendment_level", "timepoint",
        "aggregation_level", "unit_canonical", "control_token",
        "treatment_mean", "control_mean", "source_locator", "is_figure",
        "evidence", "decoder"]

# ---------------------------------------------------------------------------
# Toggles used only to produce the ledger's sensitivity numbers.  The delivered
# keys are produced with the defaults.
# ---------------------------------------------------------------------------
CROP_FROM_PAPER_LEVEL = True   # CHANGE 6 (see below); False == submitted logic
CROP_SPACE_FORM = True         # CHANGE 7 (see below); False == snake_case
NORMALIZE_LABEL_SEPARATORS = True   # CHANGE 4


# ===========================================================================
# VERBATIM from the submitted decoder -- helpers
# ===========================================================================
def nfkc(s):
    if s is None:
        return ""
    return unicodedata.normalize("NFKC", str(s))


def norm_text(s):
    """lowercase, normalize unicode, collapse whitespace."""
    s = nfkc(s).lower()
    s = s.replace("−", "-")  # minus sign
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sig3(x):
    """round to 3 significant figures, return plain decimal string."""
    if x is None:
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if v == 0:
        return "0"
    from math import log10, floor
    d = 3 - int(floor(log10(abs(v)))) - 1
    r = round(v, d)
    if r == int(r):
        return str(int(r))
    return ("%.10f" % r).rstrip("0").rstrip(".")


# ---------------------------------------------------------------------------
# YIELD classification (outcome_canonical='yield')  -- VERBATIM
# ---------------------------------------------------------------------------
NONYIELD_SUBSTR = [
    "content", "%", "carbohydrate", "scavenging", "phenol", "flavonoid", "fixed oil",
    "color", "colour", "coordinate", "spad", "chlorophyll", "photosynthesis",
    "stomatal", "water content", "electrolyte", "leakage", "proline", "frap",
    "abts", "anthocyan", "reducing power", "trolox", "ph", "titratable",
    "ascorbic", "brix", "sugar in", "sugar of", "soluble solid", "ssc", "oil content",
    "protein", "gluten", "sedimentation", "bread volume", "falling number", "moisture",
    "germination", "field emergence", "harvest index", "leaf:root", "mass ratio",
    "severity", "blight", "spot", "microbial", "colony", "cfu", "revenue", "aus$",
    "isoflavone", "carotenoid", "dry matter (g", "dm %", "ta (", "ci",
    "diameter", "length", "width", "height", "number", "leno", "leaf number",
    "plhe", "plant height", "canopy", "shank", "stand", "apical", "node",
    "cluster/", "bunches per", "berries per", "stem diameter", "trunk", "increment",
    "frsa", "topc", "tofla", "tac", "aa ", "toca", "rejects", "commercial-grade",
    "runner yield", "stalks color", "leaves color", "root collar", "rp,", "tpc",
    "tfc", "n)", "phosphorus", "potassium", "calcium", "magnesium", "nitrogen (",
    "fruit size", "individual fruit weight", "individual fresh fruit weight",
    "fruit weight (g)", "berry weight", "bunch weight", "cluster weight",
    "1000 seed", "1,000 seed", "100-kernel", "hundred seeds", "mass of 1000",
    "seed weight (", "1000 grains", "100 seeds", "kernel weight", "seeds weight",
    "ear number", "grains in ear", "grains/ear", "ears/m", "pods per plant",
    "number of pods", "number of seeds", "seeds (per", "pod length", "leaf area",
    "leaf length", "leaf width", "leaf mass", "leaves biomass", "shoots biomass",
    "stem biomass", "roots biomass", "root length", "root diameter",
    "flag leaf", "inflorescence length", "inflorescence width", "head height",
    "head dry matter", "shoot height", "number of branches", "branches/plant",
    "fruit length", "fruit width", "fruit diameter", "number of bearing",
    "flowers per cluster", "fruits per cluster", "number of fruit",
    "shoot fresh weight", "shoot dry weight", "root fresh weight (g)",
    "root dry weight", "number of stalks", "number of flowers", "leaf number/plant",
]

YIELD_POS = [
    "yield", "fresh herb", "dry herb", "green fodder", "green forage", "dry forage",
    "fresh inflorescence", "fresh roots", "storage roots", "leaves [mg", "leaves[mg",
    "total biomass", "total plant biomass", "weight of grain per plot",
    "seyi", "hfrwe", "hdrwe", "fresh weight (kg", "head fresh weight",
    "tuber size", "fresh fruit weight", "fruit fresh weight",
    "total fruit weight per plant", "(t.hm-2)", "oil yield", "fresh olive",
    "olives/", "oil/tree", "stover", "tuber yield", "ware yield",
]

YIELD_GENERIC = {"mean", "yield", "fruit yield", "productivity (prod)",
                 "dm seed yield", "mean seed yield (kg/ha)"}


def is_yield(element, unit=""):
    e = norm_text(element)
    if not e:
        return False
    u = norm_text(unit)
    if u in ("%", "% dm", "% fresh weight", "% dry weight"):
        return False
    if "percent increase" in e or "% increase" in e or "relative " in e:
        return False
    if "tuber size" in e or "fruit size" in e or "berry size" in e:
        return False
    if e in YIELD_GENERIC:
        return True
    for p in YIELD_POS:
        if p in e:
            if "1000" in e or "1,000" in e or "kernel" in e or "hundred" in e:
                return False
            if "content of sugar" in e or "sugar content" in e:
                return False
            return True
    for nz in NONYIELD_SUBSTR:
        if nz in e:
            return False
    return False


# ---------------------------------------------------------------------------
# CROP inference  -- VERBATIM
# ---------------------------------------------------------------------------
CROP_MAP = {
    "tomato": "tomato", "sweet pepper": "sweet_pepper", "pepper": "pepper",
    "strawberry": "strawberry", "mung bean": "mung_bean", "mungbean": "mung_bean",
    "potato": "potato", "sugarcane": "sugarcane", "cane": "sugarcane",
    "shallot": "shallot", "eggplant": "eggplant", "olive": "olive",
    "grape": "grape", "vine": "grape", "lettuce": "lettuce", "apple": "apple",
    "carrot": "carrot", "soybean": "soybean", "cotton": "cotton",
    "sugar beet": "sugar_beet", "beet": "sugar_beet", "wheat": "wheat",
    "barley": "barley", "maize": "maize", "corn": "maize", "rice": "rice",
    "bean": "bean", "patchouli": "patchouli", "basil": "basil",
    "chamomile": "chamomile", "fennel": "fennel",
}


def infer_crop(mods, element, tdesc, paper_id, paper_text=""):
    # 1) explicit moderator on THIS row
    for k in ("crop", "species"):
        v = mods.get(k)
        if v:
            vn = norm_text(v)
            for key, val in CROP_MAP.items():
                if key in vn:
                    return val, "moderator"
            return re.sub(r"[^a-z0-9]+", "_", vn).strip("_"), "moderator"
    # 2) element label of THIS row
    el = norm_text(element)
    for key, val in CROP_MAP.items():
        if key in el:
            return val, "element"
    # 3) THIS paper's own metadata (title / species in recon) -- paper-level
    pt = norm_text(paper_text)
    for key, val in CROP_MAP.items():
        if key in pt:
            return val, "paper_title"
    return "", "undeterminable"


# ---------------------------------------------------------------------------
# PBs CATEGORY (treatment_level)  -- VERBATIM
# ---------------------------------------------------------------------------
def infer_pbs_category(tdesc, mods):
    blob = norm_text(tdesc) + " | " + norm_text(json.dumps(mods, ensure_ascii=False))
    if any(t in blob for t in ["chitosan"]):
        return "chitosan"
    if any(t in blob for t in ["silicon", " si ", "si +", "si foliar", "diatomaceous",
                               "adesil", "silicic", "silicate", "potassium silicate"]):
        return "silicon"
    if any(t in blob for t in ["humic", "fulvic", "humistar", "lignohumate", "actosol",
                               "humic acid", "humic substanc", "lexin"]):
        return "humic"
    if any(t in blob for t in ["seaweed", "ascophyllum", "kelpak", "algae", "algal",
                               "macroalgae", "kappaphycus", "sargassum", "alga ",
                               "asparagopsis", "spirulina", "bio-algeen", "bio algeen",
                               "ecklonia", "laminaria", "goemar", "göemar", "bm-86",
                               "bm 86", "saep", "sm3", "fertileader", "algex",
                               "alga)", "seaweed extract", "red alga", "brown algae",
                               "marine algae"]):
        return "seaweed"
    if any(t in blob for t in ["protein hydrolysate", "hydrolysate", "feather protein",
                               " fph", "fph-", "fph ", "trainer", "stimtide",
                               "sinergon", "aswell", "tyson", "alfalfa protein",
                               "soybean protein", "soy protein", "peptides"]):
        return "protein_hydrolysate"
    if any(t in blob for t in ["amino acid", "amino-acid", "aminoplant", "amino acids",
                               "terra sorb", "fylloton", "protaminal", "microfert",
                               "phenylalanine", "l-phenylalanine"]):
        return "amino_acid"
    if any(t in blob for t in ["trichoderma", "amf", "mycorrhiz", "bacill", "rhizob",
                               "microbial", "azospirillum", "pseudomonas", "yeast"]):
        return "microbial"
    if any(t in blob for t in ["moringa", "compost extract", "plant extract",
                               "supercritical extract", "leaf extract"]):
        return "plant_extract"
    if any(t in blob for t in ["asahi", "brassinosteroid", "viusid", "tytanit",
                               "growth regulator", "elicitor", "veritas",
                               "benzyl adenine", "salicylic", "auxin"]):
        return "other"
    return "other"


# ---------------------------------------------------------------------------
# METHOD (co_amendment)  -- VERBATIM
# ---------------------------------------------------------------------------
def infer_method(tdesc, mods):
    blob = norm_text(tdesc) + " | " + norm_text(json.dumps(mods, ensure_ascii=False))
    am = ""
    for k in ("application_method", "application", "method"):
        if mods.get(k):
            am = norm_text(mods.get(k))
            break
    cand = am + " " + blob
    if any(t in cand for t in ["seed treatment", "seed soak", "seed priming",
                               "seed dressing", "seed coat"]):
        return "seed"
    if any(t in cand for t in ["foliar", "spray", "spraying", "leaf application",
                               "sprayed"]):
        return "foliar"
    if any(t in cand for t in ["drench", "fertigation", "fertigated", "soil application",
                               "soil drench", "applied to soil", "to soil", "soil-applied",
                               "soil ", "before sowing", "incorporated", "root zone",
                               "drip irrigation", "drip ", "via irrigation"]):
        return "soil"
    if "combined" in cand or ("soil" in cand and "foliar" in cand):
        return "soil_plus_foliar"
    return "none"


# ---------------------------------------------------------------------------
# DOSE (co_amendment_level)  -- VERBATIM
# ---------------------------------------------------------------------------
DOSE_KEYS = ["dose", "concentration", "concentration_g_l", "concentration_kg_ha",
             "concentration_ppm", "application_rate", "biostimulant_dose",
             "chitosan_concentration", "fph_concentration", "seaweed_rate",
             "phenylalanine_level", "actosol_level", "yeast_level",
             "nitrogen_dose", "application_rate"]

DOSE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*"
    r"(%|ppm|mg\s*[\/·]?\s*l|g\s*[\/·]?\s*l|ml\s*[\/·]?\s*l|mg\s*[\/·]?\s*ml|"
    r"l\s*[\/·]?\s*ha|kg\s*[\/·]?\s*ha|dm[³3]\s*[\/·]?\s*ha|g\s*[\/·]?\s*tree|"
    r"g\s*[\/·]?\s*plant|g\s*[\/·]?\s*l|l\s*[\/·]?\s*fad|mg\s*[\/·]?\s*l)",
    re.IGNORECASE)


def infer_dose(tdesc, mods):
    for k in DOSE_KEYS:
        v = mods.get(k)
        if v is None:
            continue
        vs = norm_text(v)
        m = DOSE_RE.search(vs)
        if m:
            mag = sig3(m.group(1).replace(",", "."))
            unit = re.sub(r"\s+", "", m.group(2).lower()).replace("·", "/")
            return mag + unit
        m2 = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*", vs)
        if m2:
            mag = sig3(m2.group(1).replace(",", "."))
            if "ppm" in k:
                return mag + "ppm"
            if "g_l" in k or k == "concentration_g_l":
                return mag + "g/l"
            if "kg_ha" in k:
                return mag + "kg/ha"
            if "phenylalanine" in k:
                return mag + "mg/l"
            return mag
    td = norm_text(tdesc)
    m = DOSE_RE.search(td)
    if m:
        mag = sig3(m.group(1).replace(",", "."))
        unit = re.sub(r"\s+", "", m.group(2).lower()).replace("·", "/")
        return mag + unit
    for k in ("concentration",):
        v = mods.get(k)
        if v and norm_text(v) in ("lower", "higher", "low", "high"):
            return norm_text(v)
    return "0"


# ---------------------------------------------------------------------------
# FREQUENCY token  -- VERBATIM
# ---------------------------------------------------------------------------
def infer_frequency(tdesc, mods):
    blob = norm_text(tdesc) + " | " + norm_text(json.dumps(mods, ensure_ascii=False))
    for k in ("application_frequency", "applications", "application", "frequency",
              "application_timing"):
        v = mods.get(k)
        if v:
            vn = norm_text(v)
            if "single" in vn or vn == "1":
                return "single"
            if "double" in vn:
                return "double"
            if "triple" in vn:
                return "triple"
    if "single spray" in blob or "single spraying" in blob or "once" in blob:
        return "single"
    if "double spray" in blob or "double spraying" in blob or "twice" in blob:
        return "double"
    return ""


# ---------------------------------------------------------------------------
# UNIT canonicalization  -- VERBATIM
# ---------------------------------------------------------------------------
def _unit_token(unit):
    u = nfkc(unit).lower()
    u = u.replace("⁻", "-").replace("¹", "1").replace("²", "2").replace("³", "3")
    u = u.replace("−", "-").replace("·", " ").replace("[", " ").replace("]", " ")
    u = u.replace(".", " ")
    u = u.replace("/", " / ")
    u = re.sub(r"\s+", " ", u).strip()
    if "/" in u:
        num, den = u.split("/", 1)
        num = num.strip().replace(" ", "")
        den = den.strip().replace(" ", "")
    else:
        toks = u.split(" ")
        num_toks, den_toks = [], []
        for t in toks:
            mden = re.fullmatch(r"([a-z0-9]+)-([12])", t)
            if mden:
                base, exp = mden.group(1), mden.group(2)
                den_toks.append(base + ("2" if exp == "2" else ""))
            else:
                num_toks.append(t)
        num = "".join(num_toks)
        den = "".join(den_toks)
    if den in ("hm", "hm2"):
        den = "ha"
    if den == "m2":
        den = "m2"
    den = den.replace("hm", "ha")
    out = num if not den else num + "/" + den
    return out


def canon_unit(unit):
    u = _unit_token(unit)
    AREA_T_HA = {
        "t/ha": 1.0, "ton/ha": 1.0, "mg/ha": 1.0,
        "kg/ha": 0.001,
        "g/m2": 0.01,
        "kg/m2": 10.0,
    }
    PLANT_G = {
        "g/plant": 1.0, "gdm/plant": 1.0, "gperplant": 1.0, "g/yieldingplant": 1.0,
        "kg/plant": 1000.0,
        "g/tree": 1.0, "goil/tree": 1.0, "golives/tree": 1.0,
        "kg/tree": 1000.0,
        "kg/vine": 1000.0, "g/berry": 1.0,
    }
    if u in AREA_T_HA:
        return "t/ha", AREA_T_HA[u]
    if u in PLANT_G:
        return "g/plant", PLANT_G[u]
    keep = re.sub(r"[^a-z0-9]+", "_", u).strip("_")
    return keep, None


# ===========================================================================
# END VERBATIM SECTION -- everything below is new, and is either the schema
# adapter, the paper crosswalk, or an explicitly logged change.
# ===========================================================================

# ---------------------------------------------------------------------------
# CHANGE 2: SCHEMA ADAPTER.  The frozen agent records use different field names
# and a flatter shape than the consensus records the submitted decoder was
# written against.  The adapter renames / relocates fields ONLY.  It creates no
# information, drops no record, and touches no numeric value.
#
#   consensus field        <-  frozen agent field(s)
#   --------------------------------------------------------------------------
#   element                <-  outcome
#   treatment_description  <-  treatment_description | treatment_label | description
#   control_description    <-  control_description | control_label
#   moderators (dict)      <-  moderators (dict, when present)  MERGED WITH
#                              every other record-level key that is not one of
#                              the core observation/outcome fields listed in
#                              CORE_FIELDS below (581 records carry a nested
#                              `moderators` dict; the other 472 carry the same
#                              descriptors as flat sibling keys)
#   moderators['dose']     <-  moderators['rate'] | moderators['biostimulant_rate']
#                              when no 'dose' key exists  (pure key alias: the
#                              submitted DOSE_KEYS list is left untouched)
#
# `n`, `unit`, `data_source`, `tissue`, `variance_*`, `effect_pct`,
# `treatment_mean`, `control_mean`, `significance`, `observation_id`, `note(s)`
# and `confidence` are DELIBERATELY EXCLUDED from the moderator merge: the
# inference functions match substrings against a JSON blob of the moderators,
# so admitting any outcome/statistical field there would break the
# outcome-blindness rule.
# ---------------------------------------------------------------------------
CORE_FIELDS = {
    # endpoint + outcome values + statistics  (never enter the moderator blob)
    "outcome", "element", "tissue",
    "treatment_mean", "control_mean", "effect_pct", "ln_rr",
    "n", "unit", "variance_value", "variance_type",
    "treatment_variance", "control_variance", "significance",
    # provenance / free text handled explicitly, not merged
    "data_source", "moderators",
    "treatment_description", "control_description",
    "treatment_label", "control_label", "description",
    "note", "notes", "confidence", "observation_id",
    # derived verification flags that may appear in some records
    "grim_valid", "cv_reasonable", "direction_expected",
}

DOSE_ALIAS_SOURCES = ("rate", "biostimulant_rate")


def adapt_record(o):
    """Map one frozen agent record onto the consensus field names the submitted
    decoder reads.  Returns (element, tdesc, cdesc, mods, unit, data_source,
    treatment_mean, control_mean, n, variance_type)."""
    element = o.get("outcome")

    tdesc = o.get("treatment_description")
    if not tdesc:
        tdesc = o.get("treatment_label") or o.get("description") or ""
    cdesc = o.get("control_description")
    if not cdesc:
        cdesc = o.get("control_label") or ""

    mods = {}
    nested = o.get("moderators")
    if isinstance(nested, dict):
        for k, v in nested.items():
            if v is not None:
                mods[k] = v
    for k, v in o.items():
        if k in CORE_FIELDS or v is None:
            continue
        if k in mods:
            continue
        mods[k] = v
    if "dose" not in mods:
        for k in DOSE_ALIAS_SOURCES:
            if mods.get(k) is not None:
                mods["dose"] = mods[k]
                break

    return (element, tdesc, cdesc, mods, o.get("unit"), o.get("data_source"),
            o.get("treatment_mean"), o.get("control_mean"), o.get("n"),
            o.get("variance_type"))


# ---------------------------------------------------------------------------
# CHANGE 3: paper-level provenance text.  The submitted decoder built
# `paper_text` from the consensus file's `recon` block
# (treatment_definition / control_definition / extraction_guidance).  The frozen
# agent files carry no `recon`; the equivalent paper-level own-metadata fields
# are title / species / crop / experiment_type / location.  Same role, same
# paper-level provenance, no values.
# ---------------------------------------------------------------------------
def build_paper_text(d, file_paper_id):
    parts = [file_paper_id]
    for k in ("title", "species", "crop", "experiment_type", "location"):
        v = d.get(k)
        if v:
            parts.append(str(v))
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# CHANGE 4: outcome-label separator normalization.  The frozen agent writes some
# endpoint labels in snake_case (`total_biomass`, `fruit_fresh_weight`,
# `runner_yield`) where the consensus source wrote them with spaces
# (`total biomass`, `fruit fresh weight`, `runner yield`).  is_yield()'s
# vocabulary is space-separated, so without this the submitted classifier
# behaves inconsistently on the same endpoint depending only on punctuation.
# Purely a format normalization; applied uniformly, blind to every value.
# ---------------------------------------------------------------------------
def label_for_classification(element):
    if element is None:
        return None
    s = str(element)
    if NORMALIZE_LABEL_SEPARATORS:
        s = s.replace("_", " ")
    return s


# ---------------------------------------------------------------------------
# CHANGE 5: PAPER CROSSWALK (author + year, structural tokens only).
#
# The frozen agent files are named with the clean corpus id (`006_Alabdulla_2019`)
# while the reference side stores the corpus id WITH the article-title fragment
# that the source PDF filename carried (`006_Alabdulla_2019_Effect of foliar
# application of humic ac`).  Rows can only pair if the AI side speaks the GT
# paper_id vocabulary, so each frozen file is mapped onto a GT paper_id token.
#
# Only structural tokens are used -- never a crop, product, unit or value.  The
# ladder is tried in order; the first rung that produces candidates must produce
# exactly one, otherwise the paper stays unmapped and keeps its own id.
#
#   R1  folded-alphanumeric equality of the two id strings
#   R2  (first-author surname, 4-digit year) equality, where a surname/year pair
#       is parsed out of the id string itself; surnames may be a prefix of one
#       another (>= 4 chars) so `al-tawaha` matches `al-tawaha-et-al`
#   R3  folded-alphanumeric substring containment, minimum length 8, which
#       resolves accession-style ids (`S0304423819306703` inside
#       `1-s2.0-S0304423819306703-main`) and `Azarpour` inside
#       `article1400838000_Azarpour et al`
#
# Papers with no unique GT counterpart keep their own frozen id; they cannot
# pair with the reference, and are reported as such in the ledger.
# ---------------------------------------------------------------------------
# Letters that carry a stroke/bar rather than a combining accent do NOT decompose
# under NFKD, so they survive the accent strip and break an ASCII surname compare
# (`Głosek-Sobieraj` on the reference side vs `Glosek-Sobieraj` in the frozen id).
STROKE_FOLD = {
    "ł": "l", "Ł": "l",            # l with stroke
    "ø": "o", "Ø": "o",            # o with stroke
    "đ": "d", "Đ": "d",            # d with stroke
    "ð": "d", "Ð": "d",            # eth
    "þ": "th", "Þ": "th",          # thorn
    "ß": "ss",                          # sharp s
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",
    "ı": "i",                           # dotless i
}


def fold(s):
    """diacritic-folded, lowercased."""
    s = "".join(STROKE_FOLD.get(c, c) for c in str(s))
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def alnum(s):
    return re.sub(r"[^a-z0-9]+", "", fold(s))


def parse_author_year(token):
    """Return (surname, year) parsed from an id token, or (None, None).
    Accepts `006_Alabdulla_2019_...`, `Alrubaiee_2023`,
    `al-tawaha-et-al-2011-foliar-...`."""
    t = fold(token)
    t = re.sub(r"^\d{1,4}[_\-\s]+", "", t)          # strip corpus index prefix
    m = re.match(r"^([a-z][a-z\-'\s]*?)[_\-\s]+(\d{4})(?:[_\-\s]|$)", t)
    if not m:
        return (None, None)
    surname = re.sub(r"[\s_]+", "-", m.group(1).strip("-_ "))
    return (surname, m.group(2))


def load_gt_paper_ids():
    ids = {}
    for f in sorted(glob.glob(os.path.join(GT_KEYS, "*.csv"))):
        with io.open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                pid = r["paper_id"]
                if pid and pid not in ids:
                    ids[pid] = os.path.splitext(os.path.basename(f))[0]
                break
    return ids


def build_crosswalk(frozen_ids, gt_ids):
    """frozen_id -> (gt_paper_id or None, rule)"""
    gt_list = sorted(gt_ids)
    gt_alnum = {g: alnum(g) for g in gt_list}
    gt_ay = {g: parse_author_year(g) for g in gt_list}

    out = {}
    for fid in sorted(frozen_ids):
        fa = alnum(fid)
        fay = parse_author_year(fid)

        # R1 exact
        cand = [g for g in gt_list if gt_alnum[g] == fa]
        rule = "R1_exact_id"
        # R2 author+year
        if not cand and fay[0] and fay[1]:
            cand = []
            for g in gt_list:
                gs, gy = gt_ay[g]
                if not gs or gy != fay[1]:
                    continue
                if gs == fay[0] or (len(fay[0]) >= 4 and gs.startswith(fay[0])) \
                        or (len(gs) >= 4 and fay[0].startswith(gs)):
                    cand.append(g)
            rule = "R2_author_year"
        # R3 accession substring
        if not cand and len(fa) >= 8:
            cand = [g for g in gt_list
                    if fa in gt_alnum[g] or gt_alnum[g] in fa]
            rule = "R3_id_substring"

        if len(cand) == 1:
            out[fid] = (cand[0], rule)
        elif len(cand) > 1:
            out[fid] = (None, "ambiguous:" + rule + ":" + "|".join(cand))
        else:
            out[fid] = (None, "unmapped")
    return out


# ---------------------------------------------------------------------------
# CHANGE 6: paper-level crop fallback.  The submitted crop ladder is
# row-moderator -> endpoint label -> CROP_MAP scan of the paper text, so a crop
# the paper states plainly but that is absent from the 30-entry CROP_MAP
# (oat, broccoli, celery, cardoon, hyssop, ryegrass, ...) came out EMPTY -- the
# deposited AI keys had crop empty on 367 of 576 rows while the reference
# carries 74 distinct crops.  The frozen source states the crop at paper level
# in its own `crop` / `species` fields, so the information is present and
# discarding it is a decode loss, not a property of the source.  This rung is
# appended AFTER the three submitted rungs and reuses the submitted rung-1
# normalization verbatim.  Set CROP_FROM_PAPER_LEVEL=False for the
# submitted-logic sensitivity run.
#
# CHANGE 7: crop token separator.  The GT vocabulary for this dataset writes
# multi-word crops with spaces (`common bean`, `mung bean`, `sweet pepper`),
# while the submitted AI decoder snake_cased them (`mung_bean`) -- the same
# silent non-matching defect the spec flags for Hui casing.  Crop is emitted in
# the reference's separator convention.  Formatting only; the token is unchanged.
# Set CROP_SPACE_FORM=False to emit snake_case.
# ---------------------------------------------------------------------------
def infer_crop_with_paper_fallback(mods, element, tdesc, paper_id, paper_text, d):
    crop, basis = infer_crop(mods, element, tdesc, paper_id, paper_text)
    if crop or not CROP_FROM_PAPER_LEVEL:
        return crop, basis
    for k in ("crop", "species"):
        v = d.get(k)
        if v:
            vn = norm_text(v if not isinstance(v, (list, tuple))
                           else ", ".join(str(x) for x in v))
            for key, val in CROP_MAP.items():
                if key in vn:
                    return val, "paper_field_" + k
            return re.sub(r"[^a-z0-9]+", "_", vn).strip("_"), "paper_field_" + k
    return "", "undeterminable"


def fmt_crop(token):
    if not token:
        return ""
    if CROP_SPACE_FORM:
        return re.sub(r"\s+", " ", token.replace("_", " ")).strip()
    return token


# ---------------------------------------------------------------------------
# CHANGE 8: is_figure.  The submitted decoder hard-coded is_figure=0 on every
# row, which mislabels figure-read rows that the canonical schema requires be
# quarantined.  Derived from the row's own `data_source` provenance string.
# ---------------------------------------------------------------------------
FIG_RE = re.compile(r"\bfig(?:\.|ure|s)?\b", re.IGNORECASE)


def is_figure_row(data_source):
    return 1 if (data_source and FIG_RE.search(str(data_source))) else 0


# ---------------------------------------------------------------------------
# CHANGE 9: timepoint = 'pooled'.
# The script file recovered from the repository assigns a per-paper sequential
# `pair<N>` index, but every row of the AI key table actually deposited with the
# submission carries `pooled`, and so does every one of the 1108 reference rows
# (GT timepoint vocabulary = {pooled}).  Emitting pair indices would make the
# match key structurally incapable of pairing with the reference on a column
# where the reference has exactly one value.  Per the spec's normalization rule
# ("casing must be consistent with the GT vocabulary for that dataset"), the
# constant `pooled` is emitted.  The per-row time/season/frequency tokens the
# decoder did read remain in `evidence` for audit.
# ---------------------------------------------------------------------------
TIMEPOINT = "pooled"


def main(outdir=None, emit=True, verbose=None):
    """Decode the frozen source.  `outdir`/`emit` exist only so the ledger's
    sensitivity runs can collect rows without writing over the delivered keys;
    the delivered keys are produced by the default call."""
    if verbose is None:
        verbose = "--quiet" not in sys.argv
    outdir = outdir or OUT
    if emit and not os.path.isdir(outdir):
        os.makedirs(outdir)
    all_rows = []

    gt_ids = load_gt_paper_ids()            # paper_id -> gt key filename stem
    files = sorted(glob.glob(os.path.join(SRC, "*_agent.json")))
    frozen_ids = {}
    for f in files:
        frozen_ids[os.path.basename(f)[:-len("_agent.json")]] = f

    xwalk = build_crosswalk(set(frozen_ids), gt_ids)

    records_in = 0
    rows_out = 0
    excl = {}
    excl_labels = {}
    per_paper = []

    for fid in sorted(frozen_ids):
        f = frozen_ids[fid]
        with io.open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        obs = d.get("consensus_observations") or []
        records_in += len(obs)

        gt_pid, rule = xwalk[fid]
        paper_id = gt_pid if gt_pid else fid
        paper_text = build_paper_text(d, fid)

        rows = []
        for idx, o in enumerate(obs):
            (element, tdesc, cdesc, mods, unit_decl, data_source,
             tmean, cmean, nrep, vtype) = adapt_record(o)

            if not is_yield(label_for_classification(element), unit_decl):
                excl["not_yield_outcome"] = excl.get("not_yield_outcome", 0) + 1
                key = norm_text(element) or "(empty label)"
                excl_labels[key] = excl_labels.get(key, 0) + 1
                continue

            crop, crop_basis = infer_crop_with_paper_fallback(
                mods, element, tdesc, paper_id, paper_text, d)
            pbs = infer_pbs_category(tdesc, mods)
            method = infer_method(tdesc, mods)
            dose = infer_dose(tdesc, mods)
            freq = infer_frequency(tdesc, mods)
            ucanon, factor = canon_unit(unit_decl)

            el_l = norm_text(element)
            if ucanon in ("g", "kg") and ("per plant" in el_l or "/plant" in el_l
                                          or "plant (g" in el_l or "plant(g" in el_l):
                factor2 = 1.0 if ucanon == "g" else 1000.0
                ucanon, factor = "g/plant", factor2

            if factor is not None:
                tval = None if tmean is None else round(float(tmean) * factor, 6)
                cval = None if cmean is None else round(float(cmean) * factor, 6)
            else:
                tval = tmean
                cval = cmean

            evidence = (
                "outcome=%r; treatment_description=%r; moderators=%s; "
                "declared_unit=%r; crop_basis=%s; freq_token=%r; conv_factor=%r; "
                "n=%r; variance_type=%r; control_description=%r; "
                "timepoint=constant 'pooled' (GT timepoint vocabulary has a "
                "single value); paper_crosswalk=%s"
                % (element, tdesc, json.dumps(mods, ensure_ascii=False, sort_keys=True),
                   unit_decl, crop_basis, freq, factor, nrep, vtype, cdesc, rule)
            )

            rows.append({
                "row_id": "%s__ai__%d" % (paper_id, idx),
                "side": "ai",
                "paper_id": paper_id,
                "outcome_canonical": "yield",
                "crop": fmt_crop(crop),
                "treatment_level": pbs,
                "co_amendment": method,
                "co_amendment_level": dose,
                "timepoint": TIMEPOINT,
                "aggregation_level": "single_cell",
                "unit_canonical": ucanon,
                "control_token": "absolute_control",
                "treatment_mean": "" if tval is None else tval,
                "control_mean": "" if cval is None else cval,
                "source_locator": data_source or "",
                "is_figure": is_figure_row(data_source),
                "evidence": evidence,
                "decoder": DECODER,
            })

        # Filenames mirror the reference key filenames for shared papers so that
        # 03_KEYS/ai_rebuilt/li_j and 03_KEYS/gt/li_j correspond 1:1 by name.
        stem = gt_ids[gt_pid] if gt_pid else fid
        if emit:
            out_path = os.path.join(outdir, stem + ".csv")
            with io.open(out_path, "w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=COLS, lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
        all_rows.extend(rows)
        rows_out += len(rows)
        per_paper.append((fid, paper_id, rule, len(obs), len(rows)))

    if verbose:
        print("=" * 78)
        print("li_j AI-side rebuild  (source: 01_INPUTS_FROZEN/li_j)")
        print("=" * 78)
        print("files_in            : %d" % len(files))
        print("records_in          : %d" % records_in)
        print("rows_out            : %d" % rows_out)
        print("excluded            : %d" % sum(excl.values()))
        for k in sorted(excl):
            print("    %-24s %d" % (k, excl[k]))
        print("arithmetic check    : %d == %d + %d -> %s"
              % (records_in, rows_out, sum(excl.values()),
                 records_in == rows_out + sum(excl.values())))
        mapped = sum(1 for v in xwalk.values() if v[0])
        print("papers crosswalked  : %d of %d frozen files -> GT vocabulary"
              % (mapped, len(files)))
        print()
        print("%-34s %-6s %-16s %5s %5s" % ("frozen_id", "rule", "gt_paper_id(head)",
                                            "recs", "rows"))
        for fid, pid, rule, nobs, nrow in per_paper:
            print("%-34s %-6s %-16s %5d %5d"
                  % (fid[:34], rule.split(":")[0].replace("R1_exact_id", "R1")
                     .replace("R2_author_year", "R2")
                     .replace("R3_id_substring", "R3")[:6],
                     (pid if pid != fid else "-")[:16], nobs, nrow))
        print()
        print("top excluded endpoint labels:")
        for k in sorted(excl_labels, key=lambda x: (-excl_labels[x], x))[:25]:
            print("   %4d  %s" % (excl_labels[k], k))

    return all_rows, xwalk, dict(excl), dict(excl_labels)


if __name__ == "__main__":
    main()
