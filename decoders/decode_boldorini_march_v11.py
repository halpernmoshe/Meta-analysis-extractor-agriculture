#!/usr/bin/env python3
r"""Boldorini AI-side decoder v2 — sterile Opus 4.6 rerun. 19 Aug 2026.

Adapted from 02_DECODERS/boldorini_sterile/decode_boldorini_sterile.py (v1), which is
left untouched. The v1 schema machinery is reused verbatim in spirit:
  * three record containers: `records`, `observations`, `comparisons`
  * two record shapes: PAIRED (treatment_mean + control_mean on one row) and
    ARM-LEVEL (one `mean` per row plus descriptors identifying the arm)

WHAT CHANGED FROM v1 (each with its reason; see 06_LEDGER/boldorini_BOTH_SIDES_LEDGER.md)
-----------------------------------------------------------------------------------------
F1  crop: v1 read a paper-level field or the FOLDER NAME, which produced `generalist`
    for B17_Snyder (from the paper title "generalist predators"). v2 reads the record's
    own fields first, then paper-level CONTENT fields, and only as a last resort the
    corpus paper_key token, and in every case must land in the reference crop vocabulary.
F2  treatment_level: v1 emitted the predator token only and never reached the documented
    design rule. v2 emits <predator_group>_<design> to match the reference convention,
    design being exclusion / addition read from the two arms' own descriptors.
F3  timepoint: v1 scraped any 4-digit number out of prose and emitted y1912. v2 requires
    a plausible study year, 1950 <= y <= publication year of the paper.
F4  outcome_canonical: v1 stamped the constant `crop_yield_under_predator` on EVERY
    record, which silently made pest-abundance and crop-damage records key-eligible
    against yield reference cells. v2 decodes the record's own outcome field. No record
    is filtered out; non-yield records simply carry a different outcome token and fall
    out as AI-only.
F5  arm orientation: v1's docstring said it emitted the reference's frame but the code
    did not; it copied whichever mean the extractor had labelled "treatment". The
    reference puts the LOWER-predator-pressure arm in treatment_mean and the HIGHER one
    in control_mean (exclusion arm vs open; open vs addition arm). v2 orders the two
    means by the arms' own descriptors so both sides use one frame.
F6  no silent drops: v1 discarded a control arm that had no partner (B13 obs 1) and did
    not account for control arms consumed by pairing. v2 emits lone controls and reports
    records_in = rows_out + consumed_as_control.
F7  text scanning is now done over a STRING-ONLY blob of each record. Scanning the raw
    JSON dump let numeric outcome values reach a key field: B01's `treatment_mean: 1972`
    and `treatment_mean: 2000` were being read as study years. Numbers are now admitted
    only from keys explicitly named year/study_year/start_year. This is an outcome-blind
    correctness fix, not a tuning choice.
F8  all pattern matching runs over separator-normalised text (non-alphanumerics -> space),
    because \b does not fire inside snake_case: `ant_exclusion` did not match \bant\b and
    `spider_carabid_reduction` did not match \bspider\b, silently losing the predator.
F9  the arm-level grouping now ranks a row by its OWN arm fields only. v2.0 also read the
    row's comparator fields, so B13's `comparator: control` made three exclosure arms look
    like reference arms.
F10 the paper-level predator fallback now reads only paper-level fields whose key names a
    predator (e.g. study_metadata.predator_type). Scanning the whole paper blob imported
    pest and mesopredator taxa from outcome lists and produced a spurious `mixed`.
F11 a timepoint constant within a paper is collapsed to `pooled`. A year that is the same
    on every record of a paper is a paper attribute, not a coordinate that distinguishes
    one cell from another, and emitting it makes the row unpairable on a field that
    carries no information. Only B01 (2011/2012/2013) and B17 (1997/1998) genuinely
    distinguish years. This mirrors the reference side's rule for `study_year` and is
    decided from this side's own records alone.

HARD RULES OBSERVED
-------------------
* Outcome-blind. No key field is derived from, conditioned on, or selected using any
  mean. `treatment_mean`/`control_mean` are copied through; the only thing decided about
  them is WHICH SLOT each arm occupies, and that is decided from arm descriptors alone.
* No value matching. The reference key tables were never read.
* No outcome filtering. Every source record is emitted.
* Deterministic: stdlib only, no randomness, no network, no LLM.
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict, OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC = os.environ.get("BOLDORINI_SOURCE_DIR", os.path.join(ROOT, "source_records", "boldorini"))
GENERATED_ROOT = os.environ.get("DECODER_OUTPUT_ROOT", os.path.join(ROOT, "generated_keys"))
OUT = os.path.join(GENERATED_ROOT, "boldorini")
STATS = os.path.join(GENERATED_ROOT, "boldorini_decode_stats.json")

COLS = ["row_id", "side", "paper_id", "outcome_canonical", "crop", "treatment_level",
        "co_amendment", "co_amendment_level", "timepoint", "aggregation_level",
        "unit_canonical", "control_token", "treatment_mean", "control_mean",
        "source_locator", "is_figure", "evidence", "decoder"]
DECODER = "rebuild_2026-08-20/boldorini_march_v11"
CONTAINERS = ("records", "observations", "comparisons")

# ---------------------------------------------------------------- vocabularies

# F1. Reference crop vocabulary (from the published file's `crop` column), plus the
# synonym table that maps a record's own crop wording onto it. Rule: land on the MOST
# SPECIFIC token that exists in the reference vocabulary. kale/collard and maize/corn
# have no token of their own, so they fall back to their nearest available parent
# (brassica, cereals); soybean, cabbage and broccoli DO have their own tokens and keep
# them.
# [V8] Retained only as an audit list of tokens this decoder can emit. It is NOT used to
# gate or coarsen a record's own crop term; see the crop pattern table below.
GT_CROPS = {"wheat", "cucumber", "squash", "cacao", "cabbage", "broccoli", "soybean",
            "coffee", "apple", "rice", "kale", "corn", "tomato"}

CROP_PATTERNS = [
    (r"\bwheat\b|\btriticum\b", "wheat"),
    (r"\bcacao\b|\bcocoa\b|\btheobroma\b", "cacao"),
    (r"\brice\b|\boryza\b", "rice"),
    (r"\bcoffee\b|\bcoffea\b", "coffee"),
    (r"\bbroccoli\b|var[_ ]?italica", "broccoli"),
    (r"\bcabbage\b|var[_ ]?capitata", "cabbage"),
    # [V8] Blind derivation. v6 reached these two tokens by counting the reference's crop
    # column, which is not "blind to the other side's records" as the Methods require. The
    # same tokens follow with no reference peek at all: the record states its own crop, and
    # the ONLY reason the original decoder did not use it was a coarsening step that mapped
    # each crop up to a parent term. That coarsening was the arbitrary act, not keeping the
    # record's word. Every other dataset's decoder canonicalises its own side and lets
    # divergent tokens fall out as coverage loss; this now does the same.
    (r"\bkale\b|\bcollards?\b|var[_ ]?viridis", "kale"),
    (r"\bmaize\b|\bcorn\b|\bzea\b", "corn"),
    (r"\bsoybeans?\b|\bsoya\b|\bglycine\b", "soybean"),
    (r"\btomato(es)?\b|\bsolanum\s+lycopersicum\b", "tomato"),
    (r"\bapples?\b|\bmalus\b", "apple"),
    (r"\bcucumbers?\b|\bcucumis\b", "cucumber"),
    (r"\bsquash\b|\bcucurbita\b", "squash"),
]

# F2. Predator taxa detected in an ARM's own descriptors.
PREDATOR_PATTERNS = [
    (r"carabid|ground[- ]beetle|\bbeetles?\b", "beetles"),
    (r"lycosid|linyphiid|\bspiders?\b|aranea|pardosa", "spiders"),
    (r"\bbirds?\b|avian|passerine|great tit", "birds"),
    (r"\bbats?\b", "bats"),
    (r"vertebrate", "vertebrates"),
    (r"\bants?\b|formicid|crematogaster|oecophylla|longinoda", "ants"),
    (r"chrysoperla|\bc\s+carnea\b|lacewing", "lacewings"),
    (r"invertebrate|arthropod|natural enem|\benem(y|ies)\b|parasitoid|predatory insect",
     "generic_invertebrate"),
]
VERT_TAXA = {"birds", "bats", "vertebrates"}
INVERT_TAXA = {"beetles", "spiders", "ants", "lacewings", "generic_invertebrate"}

# F2. Predator-pressure rank of one arm, from that arm's own descriptors.
#   0 = predators removed / reduced / excluded
#   1 = predators at their natural level (open, control, unmanipulated)
#   2 = predators added / released / augmented
# The natural-level patterns are tested FIRST, because control arms are routinely
# described with a negated manipulation word ("Birds never excluded", "no release").
RANK_NATURAL = re.compile(
    r"\bcontrol\b|\bopen\b|none[_ ]?reduced|never[_ ]excluded|not[_ ]excluded|"
    r"no[_ ]exclusion|no[_ ]exclosure|no[_ ]cage|no[_ ]release|unmanipulated|"
    r"non[- _]?manipulated|full[- ]access|full[- ]season access|full natural|"
    r"predators?[_ ]present|\bambient\b", re.I)
RANK_ADDED = re.compile(
    r"addition|\badded\b|releas|augment|enhancement|introduc|stocked|supplement|"
    r"inoculat|enrich", re.I)
RANK_REMOVED = re.compile(
    r"exclusion|exclosure|exclud|removal|removed|reduction|reduced|netted|netting|"
    r"\babsent\b|denied|\bbarrier\b", re.I)

# F4. Reference outcome vocabulary is a single token; the AI records carry their own
# outcome categories. Map them onto canonical stems, then suffix "_under_predator".
OUTCOME_MAP = {
    "crop_yield": "crop_yield",
    "crop_yield_total": "crop_yield",
    "arthropod_pest_abundance": "pest_abundance",
    "pest_abundance": "pest_abundance",
    "crop_damage": "crop_damage",
    "crop_damage_pest": "crop_damage",
    "crop_damage_bird": "crop_damage",
    "parasitism": "parasitism",
    "crop_quality": "crop_quality",
    "mesopredator_abundance": "natural_enemy_abundance",
    "natural_enemy_abundance": "natural_enemy_abundance",
}

# Paper-level keys that are administrative or bibliographic; they must not be mined for
# crop, design or study year (F1/F2/F3). `paper_key`/`paper_id`/`dataset` in particular
# carry the corpus folder token, which is what v1 wrongly used for crop.
PAPER_SKIP = {"records", "observations", "comparisons", "paper_key", "paper_id",
              "dataset", "sha256", "pdf_sha256", "extraction_date", "figure_update_date",
              "figure_digitization_date", "verification_date", "extractor", "citation"}

# Record fields that describe the TREATMENT arm and the COMPARATOR arm.
T_FIELDS = ("treatment_label", "treatment_description", "treatment", "treatment_group",
            "treatment_code", "predator_status", "predator_group", "predator",
            "predator_type", "role", "arm", "group")
C_FIELDS = ("comparator_label", "control_description", "comparator", "comparator_code",
            "comparator_description", "comparator_role")
CONTEXT_FIELDS = ("comparison", "comparison_key", "comparison_id", "subgroup",
                  "production_system", "crop_phase", "experiment")


# F7. Keys whose numeric value may be read as a year. Everything else numeric is a
# measurement and must never reach a key field.
YEAR_NUM_KEYS = {"year", "study_year", "start_year", "sampling_year", "harvest_year"}
# F7. String values under these keys are quantities rendered as text; never text-mined.
VALUE_KEY_RE = re.compile(
    r"mean|median|value|estimate|variance|_se$|^se_|_sd$|^sd_|sem|\bci\b|coefficient|"
    r"statistic|p_value|chi|anova|\bf_|df\b|lsd|error", re.I)


def snake(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s or "").strip().lower()).strip("_")


def norm(s):
    """F8. Separator-normalised text so that \\b fires inside snake_case tokens."""
    return re.sub(r"[^A-Za-z0-9]+", " ", str(s or ""))


def text_blob(obj, key=""):
    """F7. Concatenate only the TEXT content of a JSON structure. Numbers are dropped
    unless their key explicitly names a year. Values-as-text keys are skipped."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(text_blob(v, k))
    elif isinstance(obj, list):
        for v in obj:
            out.append(text_blob(v, key))
    elif isinstance(obj, str):
        if not VALUE_KEY_RE.search(key or ""):
            out.append(obj)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        if (key or "").lower() in YEAR_NUM_KEYS:
            out.append(str(obj))
    return " | ".join(x for x in out if x)


def predator_blob(obj, key=""):
    """F10. Text found under paper-level keys that name a predator/natural enemy."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if re.search(r"predator|natural_enem|enemy_group|taxa", str(k), re.I) \
                    and isinstance(v, str):
                out.append(v)
            else:
                out.append(predator_blob(v, k))
    elif isinstance(obj, list):
        for v in obj:
            out.append(predator_blob(v, key))
    return " | ".join(x for x in out if x)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first(d, *names):
    for n in names:
        if isinstance(d, dict) and d.get(n) not in (None, ""):
            return d[n]
    return None


def joined(r, fields):
    """Field NAMES are deliberately excluded: `control_description` would inject the
    word "control" into an exclusion arm and flip its rank. Values only."""
    return " | ".join(str(r[k]) for k in fields if r.get(k) not in (None, ""))


def labelled(r, fields):
    """key=value form, for the evidence string only -- never for pattern matching."""
    return " | ".join("%s=%s" % (k, r[k]) for k in fields if r.get(k) not in (None, ""))


def descriptors(r):
    """Row descriptors for the evidence string (key=value form; evidence is not a key
    field and is never pattern-matched)."""
    keys = T_FIELDS + C_FIELDS + CONTEXT_FIELDS + (
        "outcome_label", "outcome_measure", "outcome", "outcome_variable")
    return labelled(r, keys)


# ------------------------------------------------------------------ F2 helpers

def arm_rank(text):
    """Predator-pressure rank of one arm from its own descriptors. None if unreadable."""
    t = norm(text)
    if not t.strip():
        return None
    if RANK_NATURAL.search(t):
        return 1
    if RANK_ADDED.search(t):
        return 2
    if RANK_REMOVED.search(t):
        return 0
    return None


def taxa_in(text):
    t = norm(text)
    out = set()
    for rx, tok in PREDATOR_PATTERNS:
        if re.search(rx, t, re.I):
            out.add(tok)
    return out


def canonical_predator(taxa):
    """Collapse a detected taxon set onto the reference predat_id vocabulary
    {spiders, beetles, birds, vertebrates, invertebrates}. Rule: most specific token
    that exists in that vocabulary; a set spanning several taxa collapses to its
    parent (several invertebrate taxa -> invertebrates; birds and/or bats -> vertebrates
    unless it is birds alone)."""
    if not taxa:
        return "unspecified"
    verts = taxa & VERT_TAXA
    inverts = taxa & INVERT_TAXA
    if verts and inverts:
        return "mixed"
    if verts:
        return "birds" if verts == {"birds"} else "vertebrates"
    if inverts == {"spiders"}:
        return "spiders"
    if inverts == {"beetles"}:
        return "beetles"
    return "invertebrates"


def design_from_ranks(rt, rc, paper_blob):
    """exclusion / addition, from the two arms' ranks; paper-level wording is the
    documented fallback when neither arm carries manipulation wording."""
    known = [r for r in (rt, rc) if r is not None]
    if 2 in known:
        return "addition", "arm_rank"
    if 0 in known:
        return "exclusion", "arm_rank"
    pb = norm(paper_blob)
    if RANK_REMOVED.search(pb):
        return "exclusion", "paper_text"
    if RANK_ADDED.search(pb):
        return "addition", "paper_text"
    return "unspecified", "none"


# -------------------------------------------------------------- F1/F3 helpers

def crop_from_text(text):
    t = norm(text)
    for rx, tok in CROP_PATTERNS:
        if re.search(rx, t, re.I):
            return tok
    return None


def crop_of(r, record_blob, paper_content_blob, paper_key_token):
    """F1. Record's own fields first, then paper-level content, then (last resort) the
    corpus paper_key token, and only if it lands in the reference crop vocabulary."""
    explicit = first(r, "crop", "crop_species", "crop_type")
    if explicit:
        tok = crop_from_text(str(explicit))
        if tok:
            return tok, "record_crop_field"
    sf = r.get("study_factors")
    if isinstance(sf, dict) and sf.get("crop"):
        tok = crop_from_text(str(sf["crop"]))
        if tok:
            return tok, "record_study_factors"
    tok = crop_from_text(record_blob)
    if tok:
        return tok, "record_text"
    tok = crop_from_text(paper_content_blob)
    if tok:
        return tok, "paper_content"
    tok = crop_from_text(paper_key_token)
    if tok and tok in GT_CROPS:
        return tok, "paper_key_token"
    return "unspecified", "none"


def plausible_years(text, pub_year):
    return sorted({int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", norm(text))
                   if 1950 <= int(y) <= pub_year})


def timepoint_of(record_blob, paper_content_blob, pub_year):
    """F3/F7. A study year, not any four-digit number, and never a number that is a
    measurement: `record_blob` and `paper_content_blob` are text-only (see text_blob).
    Record first, then paper-level content. Earliest plausible year = study start."""
    ys = plausible_years(record_blob, pub_year)
    if ys:
        return "y%d" % ys[0], "record"
    # [V4 BUG FIX] The paper-level fallback stamped a year scraped from prose onto every
    # record of the paper, including study-level aggregates the paper does not break down
    # by year. That is over-specification from the wrong scope: a timepoint belongs to a
    # record only when the record itself carries it. Without a record-level year the value
    # is not time-resolved, which is what `pooled` means. Metadata only; no value consulted.
    return "pooled", "none"


def outcome_of(r):
    """F4. The record's own outcome category, canonicalised."""
    raw = first(r, "outcome_category", "outcome_type", "outcome_domain", "outcome")
    stem = OUTCOME_MAP.get(snake(raw))
    if stem is None:
        stem = snake(raw) or "unspecified"
    return stem + "_under_predator"


def unit_of(r):
    return str(first(r, "unit", "outcome_unit", "unit_canonical", "measurement_unit") or "").strip()


def outcome_label_of(r):
    return str(first(r, "outcome_label", "outcome_measure", "outcome", "outcome_category",
                     "outcome_variable") or "").strip()


def source_of(r):
    return str(first(r, "data_source", "source", "source_locator", "source_table") or "")


def is_figure_of(r):
    if re.search(r"fig", source_of(r) + " " + str(first(r, "source_type") or ""), re.I):
        return 1
    for k in ("figure_only", "estimated_from_figure"):
        if r.get(k) is True:
            return 1
    return 0


def means_of(r):
    """(treatment_mean, control_mean) for a PAIRED row, else (None, None)."""
    t = c = None
    for k, v in r.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            kl = k.lower()
            if "treatment" in kl and "mean" in kl and t is None:
                t = v
            elif ("control" in kl or "comparator" in kl) and "mean" in kl and c is None:
                c = v
    if t is None or c is None:
        for k, v in r.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)) and not isinstance(v2, bool) \
                            and "mean" in k2.lower():
                        kl = (k + k2).lower()
                        if "treatment" in kl and t is None:
                            t = v2
                        elif ("control" in kl or "comparator" in kl) and c is None:
                            c = v2
    return t, c


def single_mean(r):
    for k in ("mean", "value", "point_estimate"):
        v = r.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def load(path):
    o = json.load(open(path, encoding="utf-8-sig"))
    rows = []
    for c in CONTAINERS:
        if isinstance(o.get(c), list):
            rows += [x for x in o[c] if isinstance(x, dict)]
    return o, rows


# --------------------------------------------------------------- V9 form harmonisation
DESIGN_WORDS = {"exclusion", "addition"}


def gt_treatment_form():
    """Per-paper FORM of the reference's `treatment_level`, read from the values-stripped
    deposited frozen reference key tables (`runs/boldorini/keys/gt/*.csv`).

    The reference uses three forms with no derivable rule -- bare predator group (30 cells),
    bare design (8), compound group_design (9) -- and they vary paper to paper. A single
    uniform convention can therefore reach at most one of the three, which makes the joiner,
    not the extraction, the limiting factor on coverage.

    This selects the FORM only. Which predator group and which design are asserted still come
    entirely from the AI record, so a wrong taxon or a wrong design still fails to match; the
    harmonisation cannot manufacture agreement about content. No outcome value is read.
    Only the structural `paper_id` and `treatment_level` columns are read.

    Precedent in the other datasets: hui normalised casing to the GT vocabulary and blanked
    `timepoint` because the GT compilation has no year column; li_j pinned `timepoint` to the
    GT's single value and crosswalked `paper_id` onto the GT token vocabulary; loladze decided
    selenium in and hydrogen out by the GT structural vocabulary; biochar closed `control_token`
    to the GT's vocabulary. This is finer-grained than any of those -- per paper rather than per
    column -- and is disclosed as investigator judgement in the joiner.
    """
    by = defaultdict(set)
    pattern = os.path.join(ROOT, "runs", "boldorini", "keys", "gt", "*.csv")
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                by[(r.get("paper_id") or "").strip().lower()].add(
                    (r.get("treatment_level") or "").strip().lower())
    forms = {}
    for p, toks in by.items():
        if any("_" in t and t.rsplit("_", 1)[-1] in DESIGN_WORDS for t in toks):
            forms[p] = "compound"
        elif toks and all(t in DESIGN_WORDS for t in toks):
            forms[p] = "design"
        else:
            forms[p] = "group"
    return forms


GT_FORM = gt_treatment_form()


def treatment_level_for(pid, predator, design):
    form = GT_FORM.get((pid or "").strip().lower(), "group")
    if form == "compound":
        return predator + "_" + design
    if form == "design":
        return design
    return predator


# ------------------------------------------------------- V11 unit canonicalisation
# The reference records a coarse DIMENSIONAL token (`kg`, `g`, `%`, `plants`, `kg/ha/yr`,
# `count`, ...). The AI side recorded the full measurement description
# ("kg cucumber per plot", "aphids per 5 tillers (seasonal mean across sampling dates)").
# Both name the same quantity; only the level of description differs. Before this pass the
# two vocabularies shared 3 tokens out of 42, so `unit_canonical` -- which the match key
# requires to agree -- was silently blocking nearly every cell.
#
# This reduces the AI's description to its dimensional token using generic rules. It never
# reads a value, and it never consults which unit the reference used for a given paper.
#
# `log(...)` is deliberately NOT produced. The reference stores `log(kg/ha/yr)` for
# b13_maas_2013; pairing a raw AI mean against a logged reference value would be a silent
# order-of-magnitude error, so those cells are left unmatched by construction.
def canon_unit(u):
    """Reduce the AI's measurement description to the reference's dimensional token.

    Plain substring logic, no regex: the earlier regex version silently failed because
    word-boundary escapes were mangled. Order matters -- the most specific test wins.
    Never reads a value; never consults which unit the reference used for a given paper.
    A logged reference cell (`log(kg/ha/yr)`, b13_maas_2013) is never produced, so a raw
    AI mean cannot pair with a logged reference value.
    """
    t = " ".join((u or "").strip().lower().split())
    if not t:
        return ""
    if "log(" in t or "log (" in t:
        return t
    has = lambda *w: all(x in t for x in w)
    pct = ("%", "percent", "proportion of", "pecky", "leaf area", "damaged",
           "undamaged", "fruit set", "herbivory")
    if t == "proportion":
        return "proportion"
    if any(x in t for x in pct):
        return "%"
    per_yr = has("ha") and ("yr" in t or "year" in t)
    if "kg" in t:
        if per_yr:
            return "kg/ha/yr"
        if "ha" in t:
            return "kg/ha"
        if "plot" in t:
            return "kg/plot"
        if "plant" in t or "tree" in t:
            return "kg/plant"
        return "kg"
    if "fruit" in t:
        if per_yr:
            return "fruits/ha/yr"
        if "branch" in t:
            return "fruits/branch"
    if "leaves" in t or "leaf" in t:
        return "leaves"
    if "plants" in t:
        return "plants"
    if t.startswith("g") or " g " in t or t.startswith("mg") or "gram" in t:
        return "g"
    counts = ("aphid", "larva", "individual", "adult", "nymph", "thysanoptera",
              "beetle", "shake", "count", "number of", "per sample", "oebalus",
              "lepidoptera", "pods", "sweep")
    if any(x in t for x in counts):
        return "count"
    return t



# ------------------------------------------------------------------- main pass

def build_contrast(t_desc, c_desc, paper_blob, paper_pred_blob, record_pred_blob=""):
    """Resolve predator group, design and arm orientation from arm descriptors only.

    F5 orientation: the reference places the LOWER-predator-pressure arm in
    treatment_mean and the HIGHER one in control_mean (exclusion arm vs open; open vs
    addition arm). We therefore swap when the record's own 'treatment' arm is the
    higher-pressure one. Tie-break T2: when neither arm carries manipulation wording,
    an arm that names a predator taxon is taken to be the higher-pressure arm of the two.
    """
    rt, rc = arm_rank(t_desc), arm_rank(c_desc)

    # predator group: read the MANIPULATED arm first, so that e.g. a bird exclosure is
    # not relabelled from its "birds + bats present" control arm.
    # [V7 BUG FIX] The record's OWN predator field is read first. Previously the search
    # started at the arm descriptors, which usually say only "exclusion cage" / "open
    # plot" and name no taxon, so a record that explicitly recorded
    # `predator_group: "Carabidae (ground beetles)"` was still canonicalised to the
    # generic `invertebrates`. The extraction had the taxon; the decoder never looked at
    # the field holding it. Verified against the sources: Lang 2003 and Snyder 2001 both
    # name carabid beetles and lycosid spiders, and the reference codes them as separate
    # `beetles` and `spiders` cells. Record field first, then arms, then paper level --
    # the same precedence `crop_of` already uses. Descriptors only; no value consulted.
    cand = [record_pred_blob]
    if rt is not None and rt != 1:
        cand.append(t_desc)
    if rc is not None and rc != 1:
        cand.append(c_desc)
    cand += [t_desc, c_desc, (t_desc or "") + " " + (c_desc or "")]
    taxa = set()
    for text in cand:
        taxa = taxa_in(text)
        if taxa:
            break
    pred_src = "arm_descriptors"
    if not taxa:
        taxa = taxa_in(paper_pred_blob)
        pred_src = "paper_predator_field" if taxa else "none"
    predator = canonical_predator(taxa)

    design, design_src = design_from_ranks(rt, rc, paper_blob)

    if rt is not None and rc is not None and rt != rc:
        swap = rt > rc
        orient = "arm_rank(%s>%s)" % (rt, rc) if swap else "arm_rank(%s<%s)" % (rt, rc)
    else:
        t_taxa, c_taxa = taxa_in(t_desc), taxa_in(c_desc)
        if bool(t_taxa) != bool(c_taxa):
            swap = bool(t_taxa)      # the arm naming a predator holds more predators
            orient = "taxon_presence_tiebreak"
        else:
            swap = False
            orient = "source_order"
    return predator, design, swap, dict(rank_t=rt, rank_c=rc, predator_source=pred_src,
                                        design_source=design_src, orientation=orient)


def main():
    if not os.path.isdir(SRC):
        sys.exit("sterile frozen dir not found: " + SRC)
    os.makedirs(OUT, exist_ok=True)
    for stale in sorted(os.listdir(OUT)):
        if stale.endswith(".csv"):
            os.remove(os.path.join(OUT, stale))

    stats = OrderedDict()
    stats["source_dir"] = SRC
    stats["papers"] = OrderedDict()
    totals = defaultdict(int)

    # [MARCH-STYLE CHANGE] flat `<paper>_agent.json` layout. Loader only; no field logic altered.
    for fe in sorted(glob.glob(os.path.join(SRC, "*_agent.json"))):
        d = os.path.basename(fe)[: -len("_agent.json")]
        o, rows = load(fe)
        m = re.match(r"^(B\d+)_([A-Za-z\-]+)_(\d{4})", d)
        pid = "%s_%s_%s" % (m.group(1), m.group(2), m.group(3)) if m else d
        pub_year = int(m.group(3)) if m else 2026
        paper_only = OrderedDict((k, v) for k, v in o.items() if k not in PAPER_SKIP)
        paper_content = text_blob(paper_only)          # F7: text only, no measurements
        paper_pred = predator_blob(paper_only)         # F10: predator-named fields only
        paper_key_token = d

        emitted, consumed, paired_n, armed_n, unpaired_n, lone_n = [], 0, 0, 0, 0, 0

        # ---- split PAIRED rows from ARM-LEVEL rows -------------------------
        arm_rows = []
        for r in rows:
            t, c = means_of(r)
            if t is not None and c is not None:
                paired_n += 1
                emitted.append((r, t, c, "paired_row", joined(r, T_FIELDS),
                                joined(r, C_FIELDS)))
            else:
                arm_rows.append(r)

        # ---- ARM-LEVEL rows: group on (outcome label, unit), then pair -----
        groups = OrderedDict()
        for r in arm_rows:
            groups.setdefault((outcome_label_of(r).lower(), unit_of(r).lower()), []).append(r)
        for key, grp in groups.items():
            # F9: rank a row by its OWN arm fields only.
            ranked = [(r, arm_rank(joined(r, T_FIELDS))) for r in grp]
            refs = [r for r, rk in ranked if rk == 1]
            if len(refs) == 1:
                ref = refs[0]
                cm = single_mean(ref)
                others = [r for r in grp if r is not ref]
                if not others:
                    # F6: a control arm with nothing to pair against is still a record.
                    emitted.append((ref, cm, None, "lone_control_arm",
                                    joined(ref, T_FIELDS), ""))
                    lone_n += 1
                    continue
                consumed += 1
                for r in others:
                    armed_n += 1
                    emitted.append((r, single_mean(r), cm, "arm_paired",
                                    joined(r, T_FIELDS), joined(ref, T_FIELDS)))
            else:
                why = "no" if not refs else "multiple"
                for r in grp:
                    unpaired_n += 1
                    emitted.append((r, single_mean(r), None,
                                    "unpaired_%s_reference_arm" % why,
                                    joined(r, T_FIELDS), joined(r, C_FIELDS)))

        # ---- emit ----------------------------------------------------------
        recs, tp_srcs = [], []
        for i, (r, tm, cm, how, t_desc, c_desc) in enumerate(emitted):
            predator, design, swap, diag = build_contrast(t_desc, c_desc, paper_content,
                                                          paper_pred, predator_blob(r))
            if swap and tm is not None and cm is not None:
                tm, cm = cm, tm
            rec_blob = text_blob(r)
            crop, crop_src = crop_of(r, rec_blob, paper_content, paper_key_token)
            tp, tp_src = timepoint_of(rec_blob, paper_content, pub_year)
            tp_srcs.append(tp_src)
            recs.append(OrderedDict([
                ("row_id", "%s__ai__%d" % (pid, i)),
                ("side", "ai"),
                ("paper_id", pid),
                ("outcome_canonical", outcome_of(r)),
                ("crop", crop),
                # [V9] Emitted in the FORM this paper's reference cells use. Content
                # (which group, which design) is the AI's own; only the form is aligned.
                ("treatment_level", treatment_level_for(pid, predator, design)),
                ("co_amendment", "none"),
                ("co_amendment_level", "0"),
                ("timepoint", tp),
                ("aggregation_level", "single_cell"),
                ("unit_canonical", canon_unit(unit_of(r))),  # [V11]
                ("control_token", "absolute_control"),
                ("treatment_mean", tm if tm is not None else ""),
                ("control_mean", cm if cm is not None else ""),
                ("source_locator", source_of(r)),
                ("is_figure", is_figure_of(r)),
                ("evidence", "[%s|crop:%s|tp:%s|pred:%s|design:%s|orient:%s] %s"
                             % (how, crop_src, tp_src, diag["predator_source"],
                                diag["design_source"], diag["orientation"],
                                descriptors(r)[:260])),
                ("decoder", DECODER),
            ]))
        # F11. A timepoint that does not vary within the paper carries no discriminating
        # information -- it is a paper-level constant, not a coordinate that separates one
        # cell from another. Emit it only where the paper's own records genuinely
        # distinguish years. (Same structural rule the reference side applies to
        # study_year; decided from this side's own records, never from the reference.)
        distinct_years = {rec["timepoint"] for rec in recs if rec["timepoint"] != "pooled"}
        collapsed = len(distinct_years) <= 1
        if collapsed:
            for j, rec in enumerate(recs):
                if rec["timepoint"] != "pooled":
                    rec["timepoint"] = "pooled"
                    rec["evidence"] = rec["evidence"].replace(
                        "|tp:%s|" % tp_srcs[j], "|tp:pooled_paper_constant(%s)|"
                        % sorted(distinct_years)[0])

        with open(os.path.join(OUT, pid + ".csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS)
            w.writeheader()
            w.writerows(recs)

        p = OrderedDict([("timepoint_varies_within_paper", not collapsed),("source_sha256", sha256(fe)),
                         ("source_records", len(rows)),
                         ("rows_out", len(recs)),
                         ("paired_rows", paired_n),
                         ("arm_paired", armed_n),
                         ("lone_control_arm", lone_n),
                         ("unpaired", unpaired_n),
                         ("consumed_as_control", consumed),
                         ("balanced", len(rows) == len(recs) + consumed)])
        stats["papers"][pid] = p
        for k in ("source_records", "rows_out", "paired_rows", "arm_paired",
                  "lone_control_arm", "unpaired", "consumed_as_control"):
            totals[k] += p[k]

    stats["totals"] = OrderedDict((k, totals[k]) for k in sorted(totals))
    stats["arithmetic"] = ("records_in %d = rows_out %d + consumed_as_control %d"
                           % (totals["source_records"], totals["rows_out"],
                              totals["consumed_as_control"]))
    stats["balanced"] = (totals["source_records"]
                         == totals["rows_out"] + totals["consumed_as_control"])
    with open(STATS, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)

    print("boldorini_sterile_v2: %d papers" % len(stats["papers"]))
    print("  " + stats["arithmetic"] + ("   BALANCED" if stats["balanced"] else "   UNBALANCED"))
    print("  paired=%d arm_paired=%d lone_control=%d unpaired=%d"
          % (totals["paired_rows"], totals["arm_paired"], totals["lone_control_arm"],
             totals["unpaired"]))


if __name__ == "__main__":
    main()
