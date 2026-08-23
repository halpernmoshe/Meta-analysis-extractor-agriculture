# -*- coding: utf-8 -*-
"""
AI-side decoder for dataset 'hui' (Hui et al. 2025, wheat grain-Zn biofortification).
INDEPENDENT REBUILD, 2026-08-19.

Adapted from the submitted two-stage chain:
  stage 1  SUBMISSION_Environmental_Evidence/resubmission/matching/runs/hui/gen_ai_keys.py
           (+ the documented iter-0 key canonicalisations in runs/hui/diagnosis_iter0.md
            and runs/hui_v2/reemit_ai_aggregation.py)
  stage 2  SUBMISSION_Environmental_Evidence/resubmission/matching/runs/hui_v4/build_hui_v4.py
           (canonical publication-level paper_id + corpus cleaning)

Exactly one variable changes vs the submission: the AI-side SOURCE is now the frozen
March-2026 single-model Claude agent JSONs (01_INPUTS_FROZEN/hui/*_agent.json) instead
of the multi-model consensus folder output/hui2023_full_35/*_consensus.json.

TWO KEY SETS ARE EMITTED, differing ONLY in how `treatment_level` is decoded, so that the
effect of the source change and the effect of the parser change can be attributed separately:

  03_KEYS/ai_rebuilt_strict/hui/   variant "strict" -- treatment_level from the
        treatment_description ONLY, by a literal port of gen_ai_keys.py::app_type
        (no method-field precedence, no keyword-union additions). Like-for-like baseline.
  03_KEYS/ai_rebuilt/hui/          variant "method_field_first" -- treatment_level from the
        record's own explicit Zn-application-method field when present, descriptor keyword
        decode (union of the four submitted sibling decoders) as fallback.

Both variants apply the mandated casing normalisation to the GT vocabulary, and both are
identical in every other column.

OUTCOME-BLIND: every key column is derived from structural/provenance metadata only
(element, tissue/plant_part, unit, treatment_description, control_description,
moderators, data_source, top-level species/paper_id). `treatment_mean` / `control_mean`
are copied through verbatim -- the single exception being the deterministic mg/100g -> mg/kg
unit conversion, which is decided from the unit STRING alone and never from a mean. No mean
is ever read to decide a key, to drop a row, or to choose between candidate rows. No GT file
is opened here.

DETERMINISTIC: stdlib only, sorted file iteration, Decimal arithmetic for the unit
conversion, no randomness, no network, no LLM.

Usage:  python decode_hui.py            (paths are resolved relative to this file)
"""
import csv
import glob
import json
import os
import re
import sys
import unicodedata
from decimal import Decimal

# --------------------------------------------------------------------------------------
# paths / variants
# --------------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))          # claude_rebuild/
IN_DIR = os.path.join(ROOT, "01_INPUTS_FROZEN", "hui")
DECODER_TAG = "rebuild_2026-08-19/hui"

# variant key -> output directory (relative to ROOT)
VARIANTS = [
    ("strict", os.path.join("03_KEYS", "ai_rebuilt_strict", "hui")),
    ("method_field_first", os.path.join("03_KEYS", "ai_rebuilt", "hui")),
]

HEADER = ["row_id", "side", "paper_id", "outcome_canonical", "crop", "treatment_level",
          "co_amendment", "co_amendment_level", "timepoint", "aggregation_level",
          "unit_canonical", "control_token", "treatment_mean", "control_mean",
          "source_locator", "is_figure", "evidence", "decoder"]

# --------------------------------------------------------------------------------------
# corpus cleaning (stage 2). Reasons transcribed from
# 04_ANALYSIS/_AS_SUBMITTED/corpus_mislabels_D2.csv (dataset = "Hui et al. 2025").
# build_hui_v4.py excluded only zhao_2020 at key-build time; the remaining seven were
# excluded downstream by every submitted analysis script (line_by_line_scope_aware.py,
# scope_aware_{paired,aggregate}_tost.py, make_fig1_fidelity.py, make_fig2_equivalence.py,
# make_bland_altman.py all carry the identical 8-token EXCLUDE set). The manuscript
# describes hui_v4 as "clean corpus after excluding 8 mislabelled PDFs", so all eight are
# applied here, at the key-build stage, and logged.
# --------------------------------------------------------------------------------------
EXCLUDE_MISLABELLED = {
    "zhao_2020": 'PDF is Mirbolook et al., Commun. Soil Sci. Plant Anal. (Zn-Gly vs ZnSO4); header "A. MIRBOLOOK ET AL."',
    "cakmak_1997": 'PDF is Plant Soil (2016) 401:331-346, Portuguese INIAV wheat lines, Elvas 2010-2013 (= Gomez-Coronado et al. 2016)',
    "liu_2014": "PDF is Uddin, Kaczmarczyk & Vincze (barley hordein transcripts & Zn, Aarhus)",
    "li_2013": 'PDF is Impa, Morete et al., J. Exp. Bot. 2013, "Zn uptake, translocation and grain Zn loading in rice"',
    "dong_2018": "filename-vs-content mismatch (B2 source-verification); actual identity not separately documented",
    "zhang_2012": "filename-vs-content mismatch (B2 source-verification); actual identity not separately documented",
    "khoshgoftarmanesh_2013": "filename-vs-content mismatch (B2 source-verification); actual identity not separately documented",
    "kumar_2018": "filename-vs-content mismatch (B2 source-verification); actual identity not separately documented",
}

# Paper-level scope decision preserved verbatim from the submitted sibling decoder
# runs/hui/_gen_ai_keys.py: the focal treatment axis is nitrogen rate, not a Zn
# application, so no app_type coordinate exists. The frozen source file states the same
# in its own `note` field ("All plots received basal 30 kg/ha ZnSO4.7H2O. The treatment
# factor is N rate, not Zn rate."). Row is still emitted, with a blank treatment_level.
# (This paper is also on the mislabelled-PDF exclusion list, so it never reaches analysis.)
# Applied identically in BOTH variants.
N_AXIS_ONLY_PAPERS = {"zhang_2012"}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def norm(s):
    """NFKC-normalise, collapse whitespace, unify minus/dot lookalikes. (union of the
    submitted siblings' norm(): gen_ai_keys.py whitespace collapse + decode_ai_batch.py
    NFKC/unicode-minus handling)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("−", "-").replace("·", ".")
    return re.sub(r"\s+", " ", s).strip()


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def strip_num_prefix(stem):
    """'11_Zhao_2020' -> 'Zhao_2020'; '14_37_Liu_2019' -> 'Liu_2019'.
    (verbatim from build_hui_v4.py)"""
    return re.sub(r"^[0-9]+(_[0-9]+)*_", "", stem)


def canon_paper_id(label):
    """Canonical publication token: <first-author-surname>_<year>, alpha-only surname,
    deaccented, lowercased. Same construction rule build_hui_v4.py::derive_paper_id used
    to build the GT tokens, so both sides land on the same vocabulary
    (e.g. 'Pahlavan-Rad' -> 'pahlavanrad_2009', 'Gomez-Coronado' -> 'gomezcoronado_2016').
    STRUCTURAL ONLY: author+year label tokens; no outcome value is consulted."""
    base = strip_num_prefix(deaccent(norm(label)))
    m = re.search(r"((?:19|20)\d{2})", base)
    year = m.group(1) if m else "NA"
    surname = base[:m.start()] if m else base
    surname = re.sub(r"[^A-Za-z]", "", surname)
    return "%s_%s" % (surname.lower(), year)


def get(o, *names):
    """First non-empty value among the record's own fields or its moderators.
    SCHEMA ADAPTER: the frozen corpus carries two record layouts -- 26 files nest
    moderators under `moderators`, 5 files inline the same descriptors at record level
    (`plant_part`, `zn_method`, `zn_rate`, `cultivar`, `year`, ...)."""
    mods = o.get("moderators")
    if not isinstance(mods, dict):
        mods = {}
    for n in names:
        v = o.get(n)
        if v not in (None, ""):
            return v
        v = mods.get(n)
        if v not in (None, ""):
            return v
    return None


def moderator_blob(o):
    mods = o.get("moderators")
    if isinstance(mods, dict):
        return dict(mods)
    # flat layout: everything that is not a canonical observation field is a moderator
    core = {"element", "tissue", "plant_part", "treatment_mean", "control_mean",
            "effect_pct", "n", "unit", "variance_value", "variance_type",
            "control_variance_value", "control_variance_type", "n_control", "n_treatment",
            "data_source", "treatment_description", "control_description",
            "obs_id", "observation_id", "moderators"}
    return {k: v for k, v in o.items() if k not in core}


def first_number(s):
    m = re.search(r"(\d+(?:\.\d+)?)", norm(s))
    return m.group(1) if m else ""


# --------------------------------------------------------------------------------------
# scope filter: grain Zn CONCENTRATION only, plus canonical-unit resolution
# --------------------------------------------------------------------------------------
# Unit canonicalisation (protocol section 5). The canonical unit for this dataset is mg/kg.
#   mg/kg, mg/kg DW, mg kg-1, ppm, ug/g, ug g-1  -> factor 1 (numerically identical)
#   mg/100g                                      -> factor 10  (pure decimal conversion)
# The factor is chosen from the UNIT STRING ALONE. No mean is inspected, so this cannot
# violate the outcome-blind rule; a magnitude-based "does this look like mg/kg?" check is
# deliberately NOT performed, because unit_canonical is a match-key field and spec rule 1
# forbids conditioning a key field on any mean.
UNIT_TARGET = "mg/kg"


def unit_factor(un):
    """Return (Decimal factor, None) if the unit maps onto mg/kg, else (None, reason)."""
    u = norm(un).lower().replace("⁻", "-").replace("¹", "1")
    if re.search(r"/\s*ha\b", u) or re.search(r"\bha\b", u):
        return None, "area_basis_unit_is_content_not_concentration:%s" % norm(un)
    if re.search(r"mg\s*[ /]?\s*kg", u) or "ppm" in u or "ug/g" in u or "µg/g" in u:
        return Decimal(1), None
    if re.match(r"^mg\s*[ /]?\s*100\s*g", u):
        return Decimal(10), None
    return None, "unit_not_canonical_mass_per_mass:%s" % norm(un)


def scope_reject_reason(o):
    """Return (factor, None) if the record is in scope, else (None, reason).
    Logic is gen_ai_keys.py::is_grain_zn_conc with the unit test widened to the equivalent
    regex used by the sibling decoder decode_ai_batch.py, plus the mg/100g conversion."""
    el = norm(get(o, "element")).lower().replace("zinc", "zn")
    tis = norm(get(o, "tissue", "plant_part")).lower()

    if "zn" not in el:
        return None, "non_Zn_element"
    for bad in ["content", "uptake", "yield", "harvest", "phytate", "phytic",
                "efficiency", "accumul", "bioavail", "protein", "biomass", "remobil"]:
        if bad in el:
            return None, "non_concentration_Zn_metric"
    if re.search(r"\bratio\b", el) or re.search(r"\bhi\b", el) or "znhi" in el:
        return None, "non_concentration_Zn_metric"

    if tis != "grain":
        return None, "non_grain_tissue:%s" % (tis if tis else "<missing>")

    return unit_factor(get(o, "unit"))


# --------------------------------------------------------------------------------------
# treatment_level = Zn application-method axis (the GT `app_type` axis)
# --------------------------------------------------------------------------------------
# canonical GT vocabulary for this dataset: Soil | Foliar | Soil+Foliar
# (03_KEYS/gt/hui/*.csv carry only these three). Seed / Seed+Foliar are AI-side surplus
# tokens that the submitted chain also emitted; they are kept in GT casing so that the
# only reason they do not pair is that GT has no such cell.
GT_LEVEL_VOCAB = ("Soil", "Foliar", "Soil+Foliar")

METHOD_MAP = {
    "soil": "Soil",
    "foliar": "Foliar",
    "soil+foliar": "Soil+Foliar",
    "foliar+soil": "Soil+Foliar",
    "soil + foliar": "Soil+Foliar",
    "foliar + soil": "Soil+Foliar",
    "seed": "Seed",
    "seed priming": "Seed",
    "seed coating": "Seed",
    "seed treatment": "Seed",
    "seed biofortification": "Seed",
    "seed+foliar": "Seed+Foliar",
    "seed + foliar": "Seed+Foliar",
    "foliar+seed": "Seed+Foliar",
    "foliar + seed": "Seed+Foliar",
    # Zn delivered to the root zone by solution culture / fertigation is decoded as the
    # soil/root Zn-supply axis -- rule preserved from the submitted sibling _decode_ai.py
    # ("Zn-sufficient solution culture = soil/root Zn supply").
    "nutrient solution": "Soil",
    "fertigation": "Soil",
    # the Zn method is foliar; the pesticide is a tank-mix co-applicant, not a Zn method
    "foliar+pesticide": "Foliar",
    "foliar + pesticide": "Foliar",
    "none": "",
}


def app_type_strict(tdesc):
    """VARIANT "strict": literal port of the submitted runs/hui/gen_ai_keys.py::app_type.
    treatment_description ONLY. Keyword sets, precedence and the kg/ha fallback are exactly
    as that script implements them; the only alteration is that the returned token is passed
    through the casing canonicaliser (a no-op here, since gen_ai_keys.py already emitted GT
    casing -- the lowercase defect came from its sibling decode_ai_batch.py)."""
    t = norm(tdesc).lower()
    has_foliar = "foliar" in t
    has_soil = ("soil" in t) or ("broadcast" in t)
    has_seed = ("seed priming" in t) or ("seed-priming" in t) or ("priming" in t)
    explicit_combined = (("soil" in t and "foliar" in t) or "soil+foliar" in t
                         or "soil + foliar" in t)
    if explicit_combined:
        return "Soil+Foliar"
    if has_seed and has_foliar:
        return "Seed+Foliar"
    if has_foliar:
        return "Foliar"
    if has_seed:
        return "Seed"
    if has_soil:
        return "Soil"
    # A Zn fertilizer rate with no application-method word: in this corpus a plain ZnSO4
    # soil/basal rate (no 'foliar'/'seed' word) is Soil application.
    if re.search(r"\d", t) and ("zn" in t or "znso4" in t):
        return "Soil"
    return ""


def app_type_union_descriptor(tdesc):
    """Descriptor fallback for the "method_field_first" variant: union of the four submitted
    sibling decoders' rules (gen_ai_keys.py + _gen_ai_keys.py + decode_ai_batch.py +
    _decode_ai.py), casing normalised to the GT vocabulary."""
    t = norm(tdesc).lower()
    # negated mentions must not create a method token, e.g.
    # "High-Zn biofortified seeds (no soil Zn)" is seed-applied, not soil-applied.
    t_pos = re.sub(r"\b(no|nil|without)\s+(additional\s+)?(soil|foliar|seed)\b", " ", t)

    has_foliar = ("foliar" in t_pos) or ("spray" in t_pos) or ("leaf" in t_pos)
    has_soil = ("soil" in t_pos) or ("broadcast" in t_pos)
    has_seed = ("seed" in t_pos) or ("coat" in t_pos) or ("prim" in t_pos)

    if has_soil and has_foliar:
        return "Soil+Foliar"
    if has_seed and has_foliar:
        return "Seed+Foliar"
    if has_foliar:
        return "Foliar"
    # soil outranks seed: an explicit soil application is the focal method, whereas a
    # seed-Zn level alongside it is a co-factor ("Soil Zn at 23 kg/ha, seed Zn 355 ng/seed").
    if has_soil:
        return "Soil"
    if has_seed:
        return "Seed"
    # dose-form fallbacks, from decode_ai_batch.py / gen_ai_keys.py: a % w/v solution is a
    # spray; a plain kg/ha Zn-fertiliser rate with no method word is a basal soil rate.
    if "%" in t_pos and not re.search(r"\bha\b", t_pos):
        return "Foliar"
    if re.search(r"\d", t_pos) and ("zn" in t_pos or "znso4" in t_pos):
        return "Soil"
    return ""


def canon_level(tok):
    """Casing guard: the emitted token must sit in the GT vocabulary's casing.
    Returns (token, changed_flag)."""
    t = (tok or "").strip()
    mapped = METHOD_MAP.get(t.lower(), t)
    return mapped, (mapped != tok)


def treatment_level(o, variant):
    """Returns (token, provenance, case_changed)."""
    if variant == "strict":
        tok = app_type_strict(get(o, "treatment_description"))
        tok, changed = canon_level(tok)
        return tok, "strict:treatment_description(gen_ai_keys.py port)", changed
    # variant "method_field_first"
    raw = get(o, "application_method", "zn_method", "zn_application_method")
    if raw is not None:
        key = norm(raw).lower()
        if key in METHOD_MAP:
            tok, changed = canon_level(METHOD_MAP[key])
            return tok, "method_field=%s" % norm(raw), changed
        tok = app_type_union_descriptor(norm(raw))
        if tok:
            tok, changed = canon_level(tok)
            return tok, "method_field(parsed)=%s" % norm(raw), changed
    tok = app_type_union_descriptor(get(o, "treatment_description"))
    tok, changed = canon_level(tok)
    return tok, "union:treatment_description", changed


# --------------------------------------------------------------------------------------
# co-amendment axis
# --------------------------------------------------------------------------------------
def co_amendment(o):
    """(name, level). Rules and their field names come from the submitted siblings:
    lime  -> gen_ai_keys.py  (mod lime_rate / lime_treatment)
    N     -> decode_ai_batch.py (mod nitrogen_rate / nitrogen_level; frozen corpus spells
             the same axis nitrogen_kg_ha / n_rate_kg_ha / n_rate)
    sucrose -> _decode_ai.py (Dong 2018 tank-mix, 3.0 % w/v)"""
    lr = get(o, "lime_rate", "lime_treatment")
    if lr is not None:
        v = norm(lr)
        if "no lime" in v.lower():
            return "lime", "0"
        return "lime", first_number(v) or ""
    nr = get(o, "nitrogen_rate", "nitrogen_level", "nitrogen_kg_ha",
             "n_rate_kg_ha", "soil_N_rate", "N_rate", "n_rate")
    if nr is not None:
        return "nitrogen", first_number(nr) or "0"
    if "sucrose" in norm(get(o, "treatment_description")).lower():
        return "sucrose", "3.0"
    return "none", "0"


# --------------------------------------------------------------------------------------
# control_definition_token (protocol section 4 closed vocabulary; NOT a key field)
# --------------------------------------------------------------------------------------
NO_ZN_PATTERNS = [
    r"\bno\s+zn\b", r"\bno\s+zinc\b", r"\bnon\s+zn\b", r"\bnil\s+zn\b", r"\bnil\b",
    r"\bno\s+soil\s+zn\b", r"\bno\s+foliar\s+zn\b", r"\bno\s+application\b",
    r"\bnot\s+applied\b", r"\buntreated\b", r"\bcontrol\b", r"\bctrl\b",
    r"\b0\s*kg\b", r"\b0\s*mg\b", r"\bzn\s*0\b", r"\b0\s*zn\b", r"\bzn0\b",
    r"-\s*zn\b", r"\bdeficient\b", r"\blow\s+zn\b", r"\bno\s+micronutrient",
    r"\bdistilled\s+water\b", r"\bdeionized\b", r"\bwater\s+control\b",
    r"\bfoliar\s+dw\b", r"\bno\s+additional\s+zn\b",
]


def control_token(cdesc):
    c = norm(cdesc).lower()
    if c == "":
        return "other"
    for p in NO_ZN_PATTERNS:
        if re.search(p, c):
            return "absolute_control"
    # a comparator that itself carries a Zn application (e.g. ZnSO4 vs a Zn chelate):
    # focal treatment carries a co-factor the control does not match.
    # (rule preserved from the submitted sibling _gen_ai_keys.py)
    if "zn" in c:
        return "co_factor_present_unmatched"
    return "other"


# --------------------------------------------------------------------------------------
# crop
# --------------------------------------------------------------------------------------
def crop_of(species, o):
    blob = (norm(species) + " " + norm(get(o, "species", "species_note", "crop", "wheat_type"))).lower()
    if "oryza" in blob or "rice" in blob:
        return "rice"
    if "hordeum" in blob or "barley" in blob:
        return "barley"
    if "zea" in blob or "maize" in blob or "corn" in blob:
        return "maize"
    if "triticum" in blob or "wheat" in blob:
        return "wheat"
    return "wheat"


# --------------------------------------------------------------------------------------
# timepoint (decoded, then canonicalised to blank -- see ledger / diagnosis_iter0.md)
# --------------------------------------------------------------------------------------
def decoded_timepoint(o):
    y = norm(get(o, "year", "growing_season", "years"))
    if not y:
        return ""
    if any(w in y.lower() for w in ("average", "avg", "mean", "pooled")):
        return "pooled"
    yrs = sorted({int(x) for x in re.findall(r"(?:19|20)\d{2}", y)})
    if not yrs:
        return re.sub(r"[^0-9a-z]", "", y.lower())
    if len(yrs) == 2 and yrs[1] - yrs[0] == 1:
        return "y%d" % yrs[0]        # 2013-2014 = one growing season
    if len(yrs) > 1 and yrs[-1] - yrs[0] > 1:
        return "pooled"
    return "y%d" % yrs[0]


def decoded_aggregation(o):
    blob = json.dumps(moderator_blob(o), ensure_ascii=False, sort_keys=True).lower()
    blob += " " + norm(get(o, "treatment_description")).lower()
    if "average" in blob or "averaged" in blob or "mean of" in blob or \
       "mean across" in blob or "pooled" in blob or "all cultivars mean" in blob:
        return "pooled"
    return "single_cell"


def dec_str(d):
    """Deterministic plain-decimal rendering of a Decimal."""
    d = d.normalize()
    if d == d.to_integral_value():
        d = d.quantize(Decimal(1))
    return format(d, "f")


def cell(v, factor):
    """Pass a mean through. factor == 1 -> verbatim string. factor != 1 -> exact decimal
    conversion into the canonical unit (protocol section 5)."""
    if v is None:
        return ""
    if factor == 1:
        return str(v)
    try:
        return dec_str(Decimal(str(v)) * factor)
    except Exception:
        return str(v)


# --------------------------------------------------------------------------------------
# decode one variant
# --------------------------------------------------------------------------------------
def decode(variant, out_dir):
    files = sorted(glob.glob(os.path.join(IN_DIR, "*_agent.json")))
    if not files:
        sys.exit("no frozen input files found under %s" % IN_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for stale in sorted(glob.glob(os.path.join(out_dir, "*.csv"))):
        os.remove(stale)

    stat = {
        "files_in": len(files), "records_in": 0, "rows_out": 0,
        "excl": {}, "per_paper": [], "excluded_papers": [], "empty_files": [],
        "tl_blank": [], "case_normalised": 0, "unit_converted": 0,
        "unit_convertible_in_excluded_papers": 0,
        "levels": {},          # row_id -> treatment_level  (for the variant diff)
        "paper_of": {},        # row_id -> paper_id
    }

    def bump(reason, k=1):
        stat["excl"][reason] = stat["excl"].get(reason, 0) + k

    for fp in files:
        fn = os.path.basename(fp)
        with open(fp, encoding="utf-8") as fh:
            doc = json.load(fh)
        obs = doc.get("consensus_observations") or []
        stat["records_in"] += len(obs)

        label = doc.get("paper_id") or re.sub(r"_agent$", "", fn[:-5])
        token = canon_paper_id(label)

        if token in EXCLUDE_MISLABELLED:
            bump("paper_excluded_mislabelled_pdf[%s]" % token, len(obs))
            stat["excluded_papers"].append((token, fn, len(obs), EXCLUDE_MISLABELLED[token]))
            # diagnostic only: how many records in excluded papers would have needed a
            # unit conversion had the paper survived
            for o in obs:
                f, _r = unit_factor(get(o, "unit"))
                if f is not None and f != 1:
                    stat["unit_convertible_in_excluded_papers"] += 1
            continue

        if not obs:
            stat["empty_files"].append((token, fn, norm(doc.get("note") or doc.get("notes"))[:160]))

        species = doc.get("species")
        rows = []
        for i, o in enumerate(obs):
            factor, reason = scope_reject_reason(o)
            if reason is not None:
                bump(reason)
                continue
            if factor != 1:
                stat["unit_converted"] += 1

            tdesc = norm(get(o, "treatment_description"))
            cdesc = norm(get(o, "control_description"))
            tl, tl_prov, case_changed = treatment_level(o, variant)
            if case_changed:
                stat["case_normalised"] += 1
            note = ""
            if token in N_AXIS_ONLY_PAPERS:
                tl = ""
                note = ("focal treatment axis is N rate with basal Zn on both arms; "
                        "no Zn application-method coordinate (scope note preserved from "
                        "submitted _gen_ai_keys.py)")
            if tl == "":
                stat["tl_blank"].append((token, i, tdesc[:70]))

            co_name, co_level = co_amendment(o)
            src = norm(get(o, "data_source"))
            is_fig = 1 if re.search(r"fig", src, re.I) else 0
            dtp = decoded_timepoint(o)
            dagg = decoded_aggregation(o)
            raw_unit = norm(get(o, "unit"))

            ev = " || ".join([
                "T:%s" % tdesc,
                "C:%s" % cdesc,
                "el:%s" % norm(get(o, "element")),
                "tissue:%s" % norm(get(o, "tissue", "plant_part")),
                "unit:%s" % raw_unit,
                "n:%s" % norm(get(o, "n")),
                "obs:%s" % norm(get(o, "obs_id", "observation_id")),
                "variant:%s" % variant,
                "app_type_src:%s" % tl_prov,
                "unit_conversion:%s->%s x%s" % (raw_unit or "<none>", UNIT_TARGET, dec_str(factor)),
                "mods:%s" % json.dumps(moderator_blob(o), ensure_ascii=False, sort_keys=True),
                "decoded_timepoint(blanked):%s" % dtp,
                "decoded_aggregation(canonicalised):%s" % dagg,
            ] + (["NOTE:%s" % note] if note else []))

            row_id = "%s__ai__%d" % (token, i)
            stat["levels"][row_id] = tl
            stat["paper_of"][row_id] = token
            rows.append([
                row_id,
                "ai",
                token,
                "grain_zn",
                crop_of(species, o),
                tl,
                co_name,
                co_level,
                "",              # timepoint: undeterminable on the GT side -> blank both sides
                "single_cell",   # aggregation_level: GT default, canonicalised on the AI side
                UNIT_TARGET,
                control_token(cdesc),
                cell(o.get("treatment_mean"), factor),
                cell(o.get("control_mean"), factor),
                src,
                is_fig,
                ev,
                DECODER_TAG,
            ])

        with open(os.path.join(out_dir, token + ".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(HEADER)
            w.writerows(rows)
        stat["rows_out"] += len(rows)
        stat["per_paper"].append((token, fn, len(obs), len(rows)))

    return stat


def report(variant, out_dir, stat):
    excluded_total = sum(stat["excl"].values())
    print("=" * 78)
    print("decode_hui.py  |  variant=%s  ->  %s" % (variant, out_dir))
    print("=" * 78)
    print("files in                : %d" % stat["files_in"])
    print("records in              : %d" % stat["records_in"])
    print("rows out                : %d" % stat["rows_out"])
    print("excluded                : %d" % excluded_total)
    print("arithmetic check        : %d = %d + %d  -> %s"
          % (stat["records_in"], stat["rows_out"], excluded_total,
             "OK" if stat["records_in"] == stat["rows_out"] + excluded_total else "MISMATCH"))
    print("key CSVs written        : %d" % len(stat["per_paper"]))
    print("treatment_level case-normalised rows        : %d" % stat["case_normalised"])
    print("rows unit-converted to mg/kg                : %d" % stat["unit_converted"])
    print("convertible records inside excluded papers  : %d" % stat["unit_convertible_in_excluded_papers"])
    print()
    print("-- exclusions by reason --")
    for r in sorted(stat["excl"]):
        print("   %5d  %s" % (stat["excl"][r], r))
    print()
    print("-- rows out per paper --")
    for t, fn, n, k in sorted(stat["per_paper"]):
        print("   %-26s %-40s recs=%3d rows=%3d" % (t, fn, n, k))
    print()
    print("-- rows with blank treatment_level (emitted, cannot pair) : %d --" % len(stat["tl_blank"]))
    for t, i, td in stat["tl_blank"]:
        print("   %-26s idx=%-4d %s" % (t, i, td))
    print()


def main():
    stats = {}
    for variant, rel in VARIANTS:
        out_dir = os.path.join(ROOT, rel)
        stats[variant] = decode(variant, out_dir)
        report(variant, rel, stats[variant])

    a, b = stats["strict"], stats["method_field_first"]
    print("=" * 78)
    print("VARIANT DIFF  strict (descriptor-only)  vs  method_field_first")
    print("=" * 78)
    assert set(a["levels"]) == set(b["levels"]), "row sets must be identical across variants"
    diff = sorted(k for k in a["levels"] if a["levels"][k] != b["levels"][k])
    print("rows in each set        : %d" % len(a["levels"]))
    print("rows differing in treatment_level : %d (%.1f%%)"
          % (len(diff), 100.0 * len(diff) / max(1, len(a["levels"]))))
    by_paper = {}
    by_transition = {}
    for k in diff:
        p = a["paper_of"][k]
        by_paper[p] = by_paper.get(p, 0) + 1
        tr = "%s -> %s" % (a["levels"][k] or "<blank>", b["levels"][k] or "<blank>")
        by_transition[tr] = by_transition.get(tr, 0) + 1
    print()
    print("-- by paper --")
    for p in sorted(by_paper, key=lambda x: (-by_paper[x], x)):
        print("   %-26s %3d" % (p, by_paper[p]))
    print()
    print("-- by transition (strict -> method_field_first) --")
    for t in sorted(by_transition, key=lambda x: (-by_transition[x], x)):
        print("   %-34s %3d" % (t, by_transition[t]))
    print()
    print("-- differing rows --")
    for k in diff:
        print("   %-34s %-14s -> %-14s" % (k, a["levels"][k] or "<blank>", b["levels"][k] or "<blank>"))


if __name__ == "__main__":
    main()
