# -*- coding: utf-8 -*-
"""AI-side decoder for Loladze -- INDEPENDENT REBUILD (19 Aug 2026).

Adapted from the SUBMITTED decoder:
  matching/runs/loladze_v2/keys/ai/_decode_ai_loladze_v3.py

Exactly one substantive variable changes vs the submission: the AI-side SOURCE.
The submitted decoder read the multi-model consensus folder
`output/loladze_v3_combined/*_consensus.json`; this rebuild reads the FROZEN
March-2026 single-model Claude agent JSONs
`source_records/loladze/*_agent.json` (46 files, 1646 records).

All element / tissue / species / CO2 / suffix / pooling / effect logic is the
submitted v3 logic, byte-for-byte, except for the changes enumerated in
`06_LEDGER/loladze_DECODER_LEDGER.md` (each marked `# [REBUILD-CHANGE n]`
below). Changes are limited to: (a) input path + filename glob, (b) schema
adapters where the frozen files hold the same semantic field at a different
location or under a different name, (c) one genuine closed-list parse defect,
(d) emitting canonical-schema CSV directly instead of JSONL + a separate
`keys_from_jsonl.py` pass.

BIAS: single-sided, OUTCOME-BLIND. Every key coordinate is derived from this
row's own structural descriptors / moderators / provenance. No GT file is read,
no deposited AI key table is read, and `treatment_mean` / `control_mean` /
`effect_pct` are never used to choose a key, drop a row, or pick between
candidate rows. (Note: `022_Blank_2011_agent.json` carries a top-level
`ground_truth_comparison` block containing Loladze GT effect sizes. This decoder
never reads that key. It is listed here so the omission is auditable.)

DETERMINISTIC: stdlib only, no randomness, no network, no LLM. Re-running on the
same input produces byte-identical CSVs.
"""
import csv
import glob
import json
import os
import re

# ---------------------------------------------------------------------------
# [REBUILD-CHANGE 1] input path + glob repointed at the frozen single-model run.
# submitted: SRC = ".../output/loladze_v3_combined" ; glob "*_consensus.json"
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
SRC = os.environ.get("LOLADZE_SOURCE_DIR", os.path.join(_ROOT, "source_records", "loladze"))
SRC_GLOB = "*_agent.json"

# ---------------------------------------------------------------------------
# [REBUILD-CHANGE 4] emit canonical-schema CSV directly (18 cols, csv.DictWriter,
# same column order / quoting behaviour as the submitted `keys_from_jsonl.py`).
# submitted: wrote .jsonl into runs/loladze_v2/jsonl/ai, converted in a 2nd pass.
# ---------------------------------------------------------------------------
GENERATED_ROOT = os.environ.get("DECODER_OUTPUT_ROOT", os.path.join(_ROOT, "generated_keys"))
OUTDIR = os.path.join(GENERATED_ROOT, "loladze")

COLS = ["row_id", "side", "paper_id", "outcome_canonical", "crop",
        "treatment_level", "co_amendment", "co_amendment_level", "timepoint",
        "aggregation_level", "unit_canonical", "control_token",
        "treatment_mean", "control_mean", "source_locator", "is_figure",
        "evidence", "decoder"]

OUTCOME_CANONICAL = "mineral_concentration"
CROP_CONST = "na"
DECODER = "rebuild_2026-08-19/loladze"   # [REBUILD-CHANGE 4b] spec-mandated tag

# ---------------------------------------------------------------------------
# element vocabulary (submitted v3, plus [REBUILD-CHANGE 3])
# [REBUILD-CHANGE 3] "se"/"selenium" ADDED. The submitted closed list omitted
# selenium, so source rows whose element field literally reads "Se" were parsed
# to an EMPTY treatment_level and could never pair. Selenium is a mineral
# nutrient and occurs in the GT structural vocabulary. This is a closed-list
# parse defect, fixed by correctly parsing the source string -- no value was
# consulted. Affects 3 rows (014_Lieffering_2004 x2, 010_Li_2010 x1).
# Deliberately NOT added: "h" (hydrogen, 4 rows: 006_Azam, 048_Khan) -- not a
# mineral nutrient and absent from the GT vocabulary.
# ---------------------------------------------------------------------------
ELEMENT_SYMBOLS = {"ca", "k", "fe", "zn", "n", "p", "mg", "mn", "cu", "c", "b",
                   "s", "si", "mo", "al", "ba", "na", "cd", "cr", "ni", "pb",
                   "co", "sr", "v",
                   "se"}  # [REBUILD-CHANGE 3]
ELEMENT_WORD = {
    "calcium": "ca", "potassium": "k", "iron": "fe", "zinc": "zn",
    "nitrogen": "n", "phosphorus": "p",
    "magnesium": "mg", "manganese": "mn", "copper": "cu", "carbon": "c",
    "boron": "b", "sulfur": "s",
    "sulphur": "s", "silicon": "si", "molybdenum": "mo", "aluminum": "al",
    "aluminium": "al",
    "barium": "ba", "sodium": "na", "cadmium": "cd", "chromium": "cr",
    "nickel": "ni", "lead": "pb",
    "cobalt": "co", "strontium": "sr",
    "selenium": "se",  # [REBUILD-CHANGE 3]
}
NON_MINERAL_HINTS = (
    "ratio", "protein", "amylose", "fiber", "fibre", "adf", "ndf", "ash",
    "amax", "jmax", "vcmax",
    "rd", "pmax", "pn", "photosynth", "stomatal", "conductance", "fluoresc",
    "chlorophyll", "spad",
    "sla", "lamina", "rubisco", "lignin", "cellulose", "mannan", "glucan",
    "starch", "sugar",
    "carbohydrate", "tannin", "phenolic", "extractive", "uronic", "anthocyan",
    "yield", "biomass",
    "dry mass", "dry weight basis", "bread", "dough", "flour yield", "lysine",
    "vitamin", "tnc",
    "soluble protein", "resorption", "retranslocation", "reductase",
    "nos activity", "nr activity",
    "phosphatase", "fluorescence", "toughness", "thickness", "no fluorescence",
    "bor1",
    "fraction of leaf n", "uptake rate", "absorbed per weight",
    "mixing requirement", "breakdown",
    "peak height", "peak width", "estimated bread", "optimum mixing",
    "root:shoot", "specific uptake",
    "soluble carbohydrates", "ci)", "internal co2", "net photosynthesis",
)


def element_symbol(elem_raw):
    e = (elem_raw or "").strip(); el = e.lower()
    if re.search(r"\bratio\b", el) or re.search(r"\b[a-z]{1,2}\s*[:/]\s*[a-z]{1,2}\b", el):
        if re.search(r"\bratio\b", el) or re.search(r"\b(?:c|n|p|k|ca|mg|mn|zn|fe|cu|s|b)\s*[:/]\s*(?:n|p|c)\b", el):
            return "", "ratio variable (no single element symbol): %r" % e
    m = re.match(r"^\s*([A-Za-z]{1,2})\b", e); cand = (m.group(1).lower() if m else "")
    if cand in ELEMENT_SYMBOLS and (m and m.group(1)[0].isupper()):
        return cand, ""
    for word, sym in ELEMENT_WORD.items():
        if re.search(r"\b" + word + r"\b", el): return sym, ""
    for tok in re.findall(r"[A-Za-z]{1,3}", e):
        if tok.lower() in ELEMENT_SYMBOLS and tok[0].isupper() and tok == tok.upper():
            return tok.lower(), ""
    for tok in re.findall(r"\b([A-Z][a-z]?)\b", e):
        if tok.lower() in ELEMENT_SYMBOLS: return tok.lower(), ""
    for h in NON_MINERAL_HINTS:
        if h in el: return "", "non-mineral variable (no element symbol): %r" % e
    return "", "element not in closed mineral list (undeterminable symbol): %r" % e


# ---- tissue -> co_amendment (ALIGNED to GT decode_gt.TISSUE_MAP) -------------
# GT routing: shoots/shoot/stems/tillers/stover -> above_ground ;
#             leaves/blades/needles/seedlings/frond/leaf/foliage/sheath -> foliar ;
#             seed/grain/flour -> grain ; tuber/fruit/(root-vegetable edible) -> edible ;
#             whole-plant/total-vegetation/aboveground-biomass -> above_ground.
def tissue_token(tissue_raw, element_raw=""):
    t = (tissue_raw or "").strip().lower()
    if not t:
        return "", "tissue undeterminable (blank)"
    if ("above-ground" in t or "aboveground" in t or "above ground" in t
            or "whole plant" in t or "total vegetation" in t
            or "total above" in t or "weighted mean" in t):
        return "above_ground", ""
    base = re.split(r"\s*[-(]\s*", t)[0].strip()
    # grain
    if base in {"grain", "seed", "flour"}: return "grain", ""
    # GT maps shoots/stems/tillers/stover to ABOVE_GROUND (not foliar)
    if any(w in base for w in ("shoot", "stem", "tiller", "stover", "straw", "culm", "stalk")):
        return "above_ground", "shoot/stem-class -> above_ground (GT convention)"
    # foliar (leaves/needles/blades/foliage/sheath/frond/seedling/flower)
    if base in {"leaf", "leaves", "foliage", "needles", "needle", "blade", "blades", "sheath", "frond", "seedling", "seedlings"} \
       or any(w in base for w in ("leaf", "leav", "foliage", "needle", "blade", "sheath", "frond")):
        return "foliar", ""
    if "flower" in base:
        return "foliar", "flower -> foliar (above-ground vegetative)"
    # edible storage organs: tuber, fruit, AND root-vegetables (carrot/radish/beet root = edible)
    if base in {"tuber", "fruit", "bulb", "pod"}:
        return "edible", ""
    if "root" in base or "hypocotyl" in base:
        # GT routes carrot/radish/beet 'root' (the edible organ) to 'edible'. The GT
        # Loladze dataset contains no inedible-root mineral rows, so map root->edible.
        return "edible", "storage-root tissue -> edible (GT convention)"
    if "litter" in base:
        return "litter", "tissue outside closed list (litter) -> documented attrition"
    return base.replace(" ", "_"), "tissue outside closed list: %r" % tissue_raw


# ---- species/cultivar base token --------------------------------------------
def slug(s):
    s = (s or "").strip().lower(); s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_"); return s


SPECIES_NORM = {
    "red bud": "cercis_canadensis", "redbud": "cercis_canadensis", "dogwood": "cornus_florida",
    "loblolly pine": "pinus_taeda", "sweet gum": "liquidambar_styraciflua",
    "sweetgum": "liquidambar_styraciflua", "red maple": "acer_rubrum",
}


def species_base(mods):
    cv = (mods.get("cultivar") or "").strip(); sp = (mods.get("species") or "").strip()
    clone = (mods.get("clone") or "").strip(); eco = (mods.get("ecotype") or "").strip()
    pooled = False; chosen = ""; src = ""
    if cv: chosen, src = cv, "cultivar"
    elif clone: chosen, src = clone, "clone"
    elif eco: chosen, src = eco, "ecotype"
    elif sp: chosen, src = sp, "species"
    if not chosen: return "", False, "no species/cultivar -> blank"
    low = chosen.lower()
    if any(p in low for p in ("pool", "both species", "combined", "mean of", " and ")): pooled = True
    norm = None
    for k, v in SPECIES_NORM.items():
        if k in low: norm = v; break
    return (norm if norm else slug(chosen)), pooled, "%s=%r%s" % (src, chosen, " [pooled]" if pooled else "")


# ---- GT-ALIGNED factor-suffix recovery from AI moderators -------------------
# Mirrors decode_gt.parse_additional_info routing, but reads structured AI mods.
def level_suffixes(mods):
    """Return (list_of_level_suffix_tokens, notes). Each token mirrors a GT
    co_amendment_level suffix vocabulary item. Only deterministic mappings; never
    guessed. Pure facility/site-design noise is dropped (as GT drops it)."""
    suf = []; notes = []

    def g(*keys):
        for k in keys:
            if k in mods and mods[k] not in (None, ""): return str(mods[k]).strip()
        return ""
    # Nitrogen level: GT tokens n50/n100, highn/lown
    nl = g("nitrogen_level", "N_treatment", "nitrogen_treatment", "N_amendment", "nitrogen_form")
    if nl:
        m = re.search(r"n?\s*(\d+)", nl.lower())
        low = nl.lower()
        if re.match(r"n?\s*\d+$", low) or re.fullmatch(r"n\d+", low.replace(" ", "")):
            if m: suf.append("n" + m.group(1)); notes.append("N=%r->n%s" % (nl, m.group(1)))
        elif low.startswith("high"): suf.append("highn"); notes.append("N=%r->highn" % nl)
        elif low.startswith("low"): suf.append("lown"); notes.append("N=%r->lown" % nl)
        elif m: suf.append("n" + m.group(1)); notes.append("N=%r->n%s" % (nl, m.group(1)))
    # Phosphorus level mg/kg soil: GT tokens p0/p30/p60/p120/p240/p480
    pl = g("P_level_mg_kg_soil", "phosphorus_level", "P_treatment")
    if pl:
        m = re.search(r"(\d+)", pl)
        if m: suf.append("p" + m.group(1)); notes.append("P=%r->p%s" % (pl, m.group(1)))
    # leaf/needle age: GT '1yr_old', 'old_leaves', 'inner_nm', 'needles'
    la = g("needle_age", "leaf_age", "leaf_position", "needle_cohort", "cohort", "age", "maturity_stage")
    if la:
        l = la.lower()
        if "1-year" in l or "1 year" in l or "1yr" in l or "1-yr" in l or l.startswith("1 "):
            suf.append("1yr_old"); notes.append("age=%r->1yr_old" % la)
        elif "0-year" in l or "current" in l or l.startswith("0"):
            pass  # current-year is the GT default (no suffix on the base needle rows)
        elif "old" in l:
            suf.append("old_leaves"); notes.append("age=%r->old_leaves" % la)
        elif "inner" in l:
            suf.append("inner_nm"); notes.append("age=%r->inner_nm" % la)
    # 'inner NM' leaf descriptor sometimes in tissue_detail/tissue_fraction/component
    td = g("tissue_detail", "tissue_fraction", "tissue_type", "component", "tissue_specific", "leaf_position", "organ")
    if td and ("inner" in td.lower()) and "inner_nm" not in suf:
        suf.append("inner_nm"); notes.append("tissue_detail=%r->inner_nm" % td)
    # site as distinguishing experimental site: GT 'duke','ornl' (Natali FACE sites)
    site = g("site")
    if site:
        sl = site.lower()
        if "duke" in sl: suf.append("duke"); notes.append("site=%r->duke" % site)
        elif "ornl" in sl: suf.append("ornl"); notes.append("site=%r->ornl" % site)
        # other sites (SERC/OTC/institute names) are design context, dropped like GT
    # soil type as distinguishing factor: GT 'basalt','rhyolite' (Kanowski parent material)
    soil = g("soil_type", "soil")
    if soil:
        slx = soil.lower()
        if "basalt" in slx: suf.append("basalt"); notes.append("soil=%r->basalt" % soil)
        elif "rhyolite" in slx: suf.append("rhyolite"); notes.append("soil=%r->rhyolite" % soil)
    # +/-K treatment: GT 'kplus','kminus'
    kt = g("K_treatment", "boron_treatment")
    if kt:
        kl = kt.lower()
        if re.search(r"\+\s*k\b", kl) or kl in ("+k", "plus k", "with k", "high k"): suf.append("kplus"); notes.append("K=%r->kplus" % kt)
        elif re.search(r"-\s*k\b", kl) or kl in ("-k", "minus k", "without k", "no k", "low k"): suf.append("kminus"); notes.append("K=%r->kminus" % kt)
    return suf, notes


def time_suffixes(mods):
    """Return (list_of_time_suffix_tokens, notes). GT routes year/season/harvest/DOY
    into timepoint suffixes."""
    suf = []; notes = []

    def g(*keys):
        for k in keys:
            if k in mods and mods[k] not in (None, ""): return str(mods[k]).strip()
        return ""
    yr = g("year", "cohort_year", "rainfall_year", "sampling_date", "harvest_dates", "time_point", "season")
    if yr:
        yl = yr.lower()
        # bare 4-digit year -> yYYYY
        m = re.fullmatch(r"(19|20)\d{2}", yr.strip())
        if m:
            suf.append("y" + yr.strip()); notes.append("year=%r->y%s" % (yr, yr.strip()))
        else:
            # ranges '2004-2006' -> y2004_2006 ; '2009-2010' -> y2009_2010 (GT y<a>_<b>)
            m2 = re.fullmatch(r"((?:19|20)\d{2})\s*[-/]\s*(\d{2,4})", yr.strip())
            if m2:
                suf.append("y" + m2.group(1) + "_" + m2.group(2)); notes.append("year=%r->y%s_%s" % (yr, m2.group(1), m2.group(2)))
            elif yl in ("spring", "summer", "winter", "autumn", "fall"):
                suf.append(yl); notes.append("season=%r" % yr)
    harv = g("harvest", "harvest_time", "harvest_day")
    if harv:
        m = re.search(r"(\d{4})", harv)
        if m: suf.append("h" + m.group(1)); notes.append("harvest=%r->h%s" % (harv, m.group(1)))
    doy = g("sampling_details", "sampling_date")
    if doy:
        m = re.search(r"doy\s*(\d+)", doy.lower())
        if m: suf.append("doy" + m.group(1)); notes.append("DOY=%r" % doy)
    return suf, notes


# ---- CO2 contrast (unchanged from submitted v3) ------------------------------
PPM_RE = re.compile(r"(\d{2,4})(?:\s*[-–]\s*(\d{2,4}))?\s*(?:ppm|ppmv|µmol|μmol|µl|μl|µL|umol|ul)", re.IGNORECASE)


def _ppm_from_text(txt):
    if not txt: return None
    m = PPM_RE.search(txt)
    if m:
        a = int(m.group(1))
        if m.group(2): return int(round((a + int(m.group(2))) / 2.0))
        return a
    m2 = re.search(r"(\d{2,4})", txt)
    if m2 and ("co2" in txt.lower() or "ppm" in txt.lower()): return int(m2.group(1))
    return None


def _only_int(s):
    m = re.search(r"(\d{2,4})", s or ""); return int(m.group(1)) if m else None


def co2_contrast(o, recon):
    td = o.get("treatment_description") or ""; cd = o.get("control_description") or ""
    mods = o.get("moderators") or {}
    e = _ppm_from_text(td); a = _ppm_from_text(cd)
    if e is None:
        for k in ("CO2_elevated_ppm", "CO2_elevated", "CO2_level_ppm", "co2_level", "CO2_level", "co2_treatment_level", "CO2_level_ppm"):
            if mods.get(k):
                v = _ppm_from_text(str(mods[k])) or _only_int(str(mods[k]))
                if v: e = v; break
    if a is None:
        for k in ("CO2_control",):
            if mods.get(k):
                a = _ppm_from_text(str(mods[k])) or _only_int(str(mods[k]))
                if a: break
    if e is None or a is None:
        td_def = recon.get("treatment_definition") or ""; cd_def = recon.get("control_definition") or ""
        if e is None: e = _ppm_from_text(td_def)
        if a is None: a = _ppm_from_text(cd_def)
    if e is None or a is None:
        raw = recon.get("raw_response") or ""
        mm = re.search(r'"co2_levels"\s*:\s*\{[^}]*"control"\s*:\s*"?(\d{2,4})"?[^}]*"elevated"\s*:\s*"?(\d{2,4})"?', raw)
        if mm:
            if a is None: a = int(mm.group(1))
            if e is None: e = int(mm.group(2))
    if e is None or a is None:
        return "co2_unresolved", "CO2 ppm undeterminable (td=%r c=%r)" % (td[:40], cd[:40])
    return "eco2_%d_amb_%d" % (e, a), "CO2 e=%d a=%d" % (e, a)


POOL_MOD_KEYS = ("sites_pooled",)


def is_pooled(o, sp_pooled):
    mods = o.get("moderators") or {}; pooled = sp_pooled; notes = []
    for k, v in mods.items():
        vs = str(v).lower()
        if ("pool" in vs or "main effect" in vs or "averaged across" in vs or "avg of" in vs
                or "mean of" in vs or "(pooled)" in vs):
            pooled = True; notes.append("%s=%r" % (k, str(v)[:40]))
    return pooled, ("; ".join(notes) if notes else "")


def ratio_effect(o):
    ep = o.get("effect_pct")
    if ep is not None:
        try: return round(float(ep) / 100.0, 6), ""
        except (TypeError, ValueError): pass
    tm, cm = o.get("treatment_mean"), o.get("control_mean")
    try:
        if tm is not None and cm not in (None, 0):
            return round((float(tm) - float(cm)) / float(cm), 6), "ratio from own means"
    except (TypeError, ValueError, ZeroDivisionError): pass
    return "", "effect undeterminable"


def fmt_num(x):
    if x == "" or x is None: return ""
    if isinstance(x, float) and x == int(x): return str(int(x))
    return ("%.6f" % x).rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


# ---------------------------------------------------------------------------
# [REBUILD-CHANGE 2a] SCHEMA ADAPTER: synthesise the `recon` block.
# The submitted decoder's paper-level CO2 fallback read
#   recon["treatment_definition"], recon["control_definition"],
#   recon["raw_response"] (regex'd for a "co2_levels" object), recon["is_fig_only"].
# The frozen single-model agent JSONs carry NO `recon` block; the same
# paper-level information lives in the top-level keys `co2_elevated`,
# `co2_ambient`, `co2_levels`. This adapter re-points the identical fallback at
# them without adding any new inference. `is_fig_only` has no counterpart in the
# frozen schema and is therefore absent (documented in the ledger); row-level
# figure detection via `data_source` is unchanged.
# ---------------------------------------------------------------------------
def build_recon(d):
    recon = {}
    ele = d.get("co2_elevated")
    amb = d.get("co2_ambient")
    recon["treatment_definition"] = "" if ele in (None, "") else str(ele)
    recon["control_definition"] = "" if amb in (None, "") else str(amb)
    # `co2_levels` -> the shape the submitted raw_response regex expects. Only the
    # unambiguous ambient/control + elevated key pair is forwarded; multi-level
    # dicts (elevated_1/elevated_2/..) are NOT collapsed, because choosing among
    # them would be a guess.
    lv = d.get("co2_levels")
    raw = ""
    if isinstance(lv, dict):
        ctrl = lv.get("ambient", lv.get("control"))
        elev = lv.get("elevated")
        if ctrl not in (None, "") and elev not in (None, ""):
            raw = json.dumps({"co2_levels": {"control": str(ctrl), "elevated": str(elev)}})
    recon["raw_response"] = raw
    recon["is_fig_only"] = False
    return recon


# ---------------------------------------------------------------------------
# [REBUILD-CHANGE 2b] SCHEMA ADAPTER: moderator view.
# `species_base()` and `level_suffixes()` read `species` and `site` out of the
# row's `moderators` dict. In the frozen schema 043_Natali_2009 (70 rows) puts
# those two fields on the OBSERVATION instead of inside `moderators`. This
# adapter surfaces exactly those two named keys, with `moderators` winning any
# collision (034_Johnson_1997 has both). No other observation-level key is
# promoted, so pooling detection and every other suffix rule see the same
# key-set the submitted decoder saw.
# ---------------------------------------------------------------------------
PROMOTE_OBS_KEYS = ("species", "site")


def moderator_view(o):
    mods = dict(o.get("moderators") or {})
    promoted = []
    for k in PROMOTE_OBS_KEYS:
        if k in o and o[k] not in (None, "") and k not in mods:
            mods[k] = o[k]; promoted.append(k)
    return mods, promoted


# ================================================================ main
def main():
    os.makedirs(OUTDIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, SRC_GLOB)))
    summary = []
    excl = {}   # reason -> count (rows emitted but structurally unpairable)
    for path in files:
        d = json.load(open(path, encoding="utf-8"))
        pid = d["paper_id"]
        recon = build_recon(d)                      # [REBUILD-CHANGE 2a]
        obs = d.get("consensus_observations") or []
        rows = []
        for i, o in enumerate(obs):
            mods, promoted = moderator_view(o)      # [REBUILD-CHANGE 2b]
            o_v = dict(o); o_v["moderators"] = mods
            sym, sym_note = element_symbol(o.get("element"))
            tis, tis_note = tissue_token(o.get("tissue"), o.get("element") or "")
            spc, sp_pooled, sp_note = species_base(mods)
            tp_base, tp_note = co2_contrast(o_v, recon)
            pooled, agg_note = is_pooled(o_v, sp_pooled)
            lvl_suf, lvl_notes = level_suffixes(mods)
            tp_suf, tp_notes = time_suffixes(mods)
            # assemble co_amendment_level = species/cultivar base + factor suffixes (GT order)
            cal = spc
            if lvl_suf:
                extra = "__".join(lvl_suf)
                cal = (cal + "__" + extra) if cal else extra
            # assemble timepoint
            if pooled:
                timepoint = "pooled"; agg = "pooled"
            else:
                timepoint = tp_base
                if tp_suf and tp_base != "co2_unresolved":
                    timepoint = tp_base + "".join("__" + s for s in tp_suf)
                elif tp_suf:  # unresolved base but time known
                    timepoint = "co2_unresolved" + "".join("__" + s for s in tp_suf)
                agg = "single_cell"
            rr, rr_note = ratio_effect(o)
            src = o.get("data_source") or ""
            is_fig = 1 if ("fig" in src.lower() or recon.get("is_fig_only")) else 0
            notes = [n for n in ([sym_note, tis_note, sp_note, tp_note, agg_note, rr_note]
                                 + lvl_notes + tp_notes) if n]
            if promoted:
                notes.append("obs-level %s promoted into moderator view" % ",".join(promoted))
            evidence = ("element=%r->%s | tissue=%r->%s | %s | CO2->%s | lvl_suffix=%s tp_suffix=%s | "
                        "T_desc=%r C_desc=%r | unit=%r | src=%r"
                        % (o.get("element"), sym or "(blank)", o.get("tissue"), tis or "(blank)",
                           sp_note, timepoint, lvl_suf, tp_suf,
                           (o.get("treatment_description") or "")[:50],
                           (o.get("control_description") or "")[:50], o.get("unit"), src))
            if notes: evidence += " || NOTE: " + " ; ".join(notes)
            # ---- exclusion / unpairability bookkeeping (rows are still emitted;
            # nothing is silently dropped -- see NO SILENT DROPS in the spec)
            if not sym:
                excl["unpairable: element unparseable (%s)"
                     % ("non-mineral variable" if "non-mineral" in sym_note else "not in closed mineral list")] = \
                    excl.get("unpairable: element unparseable (%s)"
                             % ("non-mineral variable" if "non-mineral" in sym_note else "not in closed mineral list"), 0) + 1
            if tis not in ("foliar", "grain", "above_ground", "edible"):
                excl["unpairable: tissue outside GT closed list (%s)" % tis] = \
                    excl.get("unpairable: tissue outside GT closed list (%s)" % tis, 0) + 1
            if timepoint.startswith("co2_unresolved"):
                excl["unpairable: CO2 ppm pair undeterminable"] = excl.get("unpairable: CO2 ppm pair undeterminable", 0) + 1
            if not cal:
                excl["diagnostic: co_amendment_level blank (no species/cultivar/clone/ecotype)"] = \
                    excl.get("diagnostic: co_amendment_level blank (no species/cultivar/clone/ecotype)", 0) + 1
            rows.append({
                "row_id": "%s__ai__%d" % (pid, i), "side": "ai", "paper_id": pid,
                "outcome_canonical": OUTCOME_CANONICAL, "crop": CROP_CONST,
                "treatment_level": sym, "co_amendment": tis, "co_amendment_level": cal,
                "timepoint": timepoint, "aggregation_level": agg, "unit_canonical": "ratio",
                "control_token": "ambient_co2", "treatment_mean": fmt_num(rr), "control_mean": "",
                "source_locator": src, "is_figure": is_fig, "evidence": evidence,
                "decoder": DECODER,
            })
        out = os.path.join(OUTDIR, pid + ".csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader(); w.writerows(rows)
        summary.append((pid, len(obs), len(rows), sum(1 for r in rows if r["treatment_level"])))

    recs = sum(n for _, n, _, _ in summary)
    tot = sum(n for _, _, n, _ in summary)
    totm = sum(m for _, _, _, m in summary)
    print("SRC=%s" % SRC)
    print("FILES=%d RECORDS_IN=%d ROWS_OUT=%d ROWS_WITH_ELEMENT=%d" % (len(summary), recs, tot, totm))
    print("ARITHMETIC: records_in(%d) == rows_out(%d) + hard_drops(0)" % (recs, tot))
    print("--- unpairability / diagnostic tally (rows are emitted, not dropped) ---")
    for k in sorted(excl):
        print("  %5d  %s" % (excl[k], k))
    print("OUT=%s" % OUTDIR)


if __name__ == "__main__":
    main()
