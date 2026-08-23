# -*- coding: utf-8 -*-
"""AI-side decoder for the biochar dataset -- INDEPENDENT REBUILD (2026-08-19).

Outcome: crop_yield.  Single-sided, outcome-blind.  Every key coordinate is
derived from the source record's OWN structural fields; treatment_mean /
control_mean are copied through (with the submitted decoders' declared-unit
conversions only) and are never used to choose a key, drop a row, or pick
between candidate rows.

SOURCE (the only thing that changes vs the submission):
    01_INPUTS_FROZEN/biochar/*.json   -- frozen March 2026 single-model Claude
                                         agent JSONs (mtime 2026-03-18), copied
                                         from output/biochar_extraction.
    The May 2026 PDF re-extractions of 016_Li_B_2016 / 063_Asai_2009 /
    145_Omara_2020 are NOT used.  Those three papers are decoded from their
    March records instead.

ADAPTED FROM (see 06_LEDGER/biochar_DECODER_LEDGER.md for the full change log):
  A) runs/biochar/decode_ai_batch.py   ->  STAGE A   (8 papers)
  B) runs/biochar/keys/ai/_decode_ai.py ->  STAGE B  (16 papers)
  C) (no submitted script existed)     ->  STAGE C   (4 papers: 229/231/234/242)
  D) runs/biochar/patch_iter0.py       ->  STAGE D   (AI-side patches only)
  E) new: declared-unit STRING normalisation (notation synonyms only)
  F) new: control_token closed-vocabulary mapping

Pure stdlib, no randomness, no LLM calls.  Deterministic: same input gives
byte-identical output.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "01_INPUTS_FROZEN", "biochar")
OUT = os.path.join(ROOT, "03_KEYS", "ai_rebuilt", "biochar")
LOGDIR = os.path.join(ROOT, "06_LEDGER")

DECODER = "rebuild_2026-08-19/biochar"

HEADER = ["row_id", "side", "paper_id", "outcome_canonical", "crop",
          "treatment_level", "co_amendment", "co_amendment_level", "timepoint",
          "aggregation_level", "unit_canonical", "control_token",
          "treatment_mean", "control_mean", "source_locator", "is_figure",
          "evidence", "decoder"]


def is_paper_extraction(path: str) -> bool:
    """STRUCTURAL filter (no filename hardcoding).

    The frozen input dir also holds pipeline artefacts that are not paper
    extractions.  A file qualifies only if it is a JSON object with a truthy
    top-level `paper_id` AND a top-level `observations` list.  Only the two
    structural keys are inspected; nothing else in a rejected file is read.
    """
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (ValueError, OSError):
        return False
    return (isinstance(d, dict) and bool(d.get("paper_id"))
            and isinstance(d.get("observations"), list))

CLOSED_CONTROL_VOCAB = {"absolute_control", "cofactor_matched_control"}

# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------
rows_by_paper: dict[str, list[dict]] = {}
records_in: dict[str, int] = {}
exclusions: list[tuple[str, str, str]] = []      # (paper_id, ref, reason)
unit_log: list[tuple[str, str, str, str]] = []   # (paper, row_id, before, after)
ctok_log: list[tuple[str, str, str, str, str]] = []  # (paper,row_id,before,after,rule)
notes: list[str] = []


def load(pid: str) -> dict:
    with open(os.path.join(SRC, pid + ".json"), encoding="utf-8") as f:
        return json.load(f)


def add(paper: str, **kw) -> dict:
    r = {k: "" for k in HEADER}
    r.update(kw)
    r["side"] = "ai"
    r["paper_id"] = paper
    r["outcome_canonical"] = "crop_yield"
    r["decoder"] = DECODER
    rows_by_paper.setdefault(paper, []).append(r)
    return r


# --------------------------------------------------------------------------
# shared numeric helpers (verbatim from the submitted decoders)
# --------------------------------------------------------------------------
def sig3(x):
    """3 significant figures, plain decimal string.  From _decode_ai.py."""
    if x is None:
        return ""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f == 0:
        return "0"
    d = math.floor(math.log10(abs(f)))
    ndig = 2 - d
    r = round(f, ndig)
    if r == int(r):
        return str(int(r))
    return ("%.10f" % r).rstrip("0").rstrip(".")


def round3(x):
    """3 significant figures via %g.  From decode_ai_batch.py."""
    if x is None:
        return ""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return ""
    if f == 0:
        return "0"
    d = round(f, -int(math.floor(math.log10(abs(f)))) + 2)
    return "%g" % d


def to_kgha(val, unit):
    """Declared-dimension yield conversion to kg/ha.  From _decode_ai.py.
    Blind: depends only on the unit STRING, never on the value or on GT."""
    if val is None:
        return None
    u = (unit or "").lower()
    if "kg/ha" in u or "kg/hm" in u:
        return float(val)
    if "t/ha" in u or "mg/ha" in u or "mg ha" in u or "t ha" in u:
        return float(val) * 1000.0
    return float(val)


def fmt_mean(val, unit):
    v = to_kgha(val, unit)
    if v is None:
        return ""
    if v == int(v):
        return str(int(v))
    return ("%.6f" % v).rstrip("0").rstrip(".")


def fmt_raw(x):
    """Pass a mean through unconverted.  From decode_ai_batch.py fmt_mean()."""
    if x is None:
        return ""
    return "%g" % float(x)


def clean_crop(c):
    """From decode_ai_batch.py."""
    if not c:
        return ""
    c = c.lower().strip()
    c = re.sub(r"\(.*?\)", "", c).strip()
    c = c.replace("upland rice", "rice").replace("spring barley", "barley")
    c = re.sub(r"^mixed vegetables.*", "vegetable", c)
    c = c.replace("vegetables", "vegetable")
    return c.strip()


# ==========================================================================
# STAGE A -- adapted from runs/biochar/decode_ai_batch.py
#            papers 001, 007, 016, 021, 041, 063, 077, 078
# ==========================================================================
STAGE_A_PAPERS = [
    "001_Adekiya_2019", "007_Gathorne-Hardy_2009", "016_Li_B_2016",
    "021_Nobile_2022", "041_Guerena_2013", "063_Asai_2009",
    "077_Zhang_J_2019", "078_Wang_2012",
]


def stage_a():
    for pid in STAGE_A_PAPERS:
        data = load(pid)
        obs_list = data.get("observations", [])
        records_in[pid] = len(obs_list)
        for i, ob in enumerate(obs_list):
            mods = ob.get("moderators", {}) or {}
            tdesc = (ob.get("treatment_description") or ob.get("treatment_label")
                     or ob.get("treatment") or "")
            cdesc = (ob.get("control_description") or ob.get("control_label")
                     or ob.get("control") or "")
            crop = clean_crop(mods.get("crop") or ob.get("crop"))
            unit = ob.get("unit") or ob.get("yield_unit") or ""
            src = ob.get("data_source") or ""
            is_fig = 1 if (ob.get("source_type") == "figure") else 0
            tmean = ob.get("treatment_mean")
            cmean = ob.get("control_mean")

            rate = mods.get("biochar_rate_tha")
            if rate is None:
                rate = ob.get("biochar_rate_tha")

            co_amend = "none"
            co_level = "0"
            timepoint = "pooled"
            agg = "single_cell"
            note = None

            if pid == "001_Adekiya_2019":
                m = re.search(r"PM\s*([\d.]+)\s*t/ha", tdesc)
                if "no PM" in tdesc:
                    co_amend, co_level = "poultry_manure", "0"
                elif m:
                    co_amend, co_level = "poultry_manure", round3(m.group(1))
                ym = re.search(r"(20\d\d)", tdesc)
                if ym:
                    timepoint = "y" + ym.group(1)
                tl = round3(rate)

            elif pid == "007_Gathorne-Hardy_2009":
                nm = re.search(r"(\d+)\s*N\s*kg/ha", tdesc)
                co_amend = "nitrogen"
                co_level = round3(nm.group(1)) if nm else "0"
                tl = round3(rate)

            elif pid == "016_Li_B_2016":
                co_amend = "nitrogen"
                if "no N" in tdesc or "N0" in tdesc:
                    co_level = "0"
                elif "conv. N" in tdesc and "4/3" not in tdesc:
                    co_level = "conventional"
                elif "4/3" in tdesc:
                    co_level = "1.33conventional"
                tl = round3(rate)

            elif pid == "021_Nobile_2022":
                co_amend = "compost"
                co_level = "8"
                ym = re.search(r"(20\d\d)", src) or re.search(r"(20\d\d)", tdesc)
                if ym:
                    timepoint = "y" + ym.group(1)
                tl = round3(rate)

            elif pid == "041_Guerena_2013":
                co_amend = "nitrogen"
                co_level = "90pct"
                ym = re.search(r"(20\d\d)", tdesc)
                if ym:
                    timepoint = "y" + ym.group(1)
                tl = round3(rate)

            elif pid == "063_Asai_2009":
                nm = re.search(r"N\s*(\d+)\s*kg/ha", tdesc)
                co_amend = "nitrogen"
                if "no N" in tdesc:
                    co_level = "0"
                elif nm:
                    co_level = round3(nm.group(1))
                sm = re.search(r",\s*([A-Z]{2}\d?|SO|SN|LS)\b", tdesc)
                if sm:
                    timepoint = sm.group(1)
                tl = round3(rate)

            elif pid == "077_Zhang_J_2019":
                co_amend, co_level = "none", "0"
                ym = re.search(r"(20\d\d)", tdesc)
                if ym:
                    timepoint = "y" + ym.group(1)
                tl = ""  # biochar dose is % of soil mass, not t/ha
                note = ("%s obs %d: biochar dose given as %% of soil mass "
                        "(C-5/10/15/20), not t/ha; treatment_level left blank"
                        % (pid, i))

            elif pid == "078_Wang_2012":
                fert = ob.get("fertilizer") or ""
                co_amend = "nitrogen"
                nm = re.search(r"(\d+)\s*kg\s*N/ha", fert)
                if "Nil" in fert:
                    co_level = "0"
                elif nm:
                    co_level = round3(nm.group(1))
                tl = round3(rate)

            else:
                tl = round3(rate)

            add(pid,
                row_id="%s__ai__%d" % (pid, i),
                crop=crop,
                treatment_level=tl,
                co_amendment=co_amend,
                co_amendment_level=co_level,
                timepoint=timepoint,
                aggregation_level=agg,
                unit_canonical=unit,
                control_token=cdesc.strip(),      # free text; closed in STAGE F
                treatment_mean=fmt_raw(tmean),
                control_mean=fmt_raw(cmean),
                source_locator=src,
                is_figure=str(is_fig),
                evidence=("T:%s || C:%s" % (tdesc, cdesc)).strip())
            if note:
                notes.append(note)


# ==========================================================================
# STAGE B -- adapted from runs/biochar/keys/ai/_decode_ai.py
#            papers 081,082,101,116,126,130,133,145,153,166,184,193,207,219,223,227
# ==========================================================================
def stage_b():
    # ---------------- 153_Wei_2022 ----------------
    d = load("153_Wei_2022")
    records_in["153_Wei_2022"] = len(d["observations"])
    for o in d["observations"]:
        npct = o["N_rate_pct"]
        ctl = ("cofactor_matched_control" if npct == 100
               else "co_factor_present_unmatched")
        add("153_Wei_2022",
            row_id="153_Wei_2022__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower().replace(" ", "_"),
            treatment_level=sig3(o["biochar_rate_t_ha"]),
            co_amendment="nitrogen",
            co_amendment_level=sig3(npct),
            timepoint="y%d" % o["year"],
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token=ctl,
            treatment_mean=fmt_mean(o["treatment_mean"], o["yield_unit"]),
            control_mean=fmt_mean(o["control_mean"], o["yield_unit"]),
            source_locator="Table 2",
            is_figure="0",
            evidence="treatment_label=%s; control_label=%s; %d"
                     % (o["treatment_label"], o["control_label"], o["year"]))

    # ---------------- 166_Haefele_2011 ----------------
    d = load("166_Haefele_2011")
    records_in["166_Haefele_2011"] = len(d["observations"])

    def hae_time(season):
        s = season.strip()
        if s.lower() == "mean":
            return "pooled"
        parts = s.split()
        yr = parts[0]
        ss = parts[1].lower() if len(parts) > 1 else ""
        return "y%s_%s" % (yr, ss)

    for o in d["observations"]:
        fert = "+F" in o["treatment_label"]
        is_mean = o["season"].strip().lower() == "mean"
        add("166_Haefele_2011",
            row_id="166_Haefele_2011__ai__%d" % o["obs_id"],
            crop="rice",
            treatment_level="41.3",
            co_amendment=("fertilizer" if fert else "none"),
            co_amendment_level=("present" if fert else "0"),
            timepoint=hae_time(o["season"]),
            aggregation_level=("pooled" if is_mean else "single_cell"),
            unit_canonical="kg/ha",
            control_token=("cofactor_matched_control" if fert
                           else "absolute_control"),
            treatment_mean=fmt_mean(o["treatment_mean"], o["yield_unit"]),
            control_mean=fmt_mean(o["control_mean"], o["yield_unit"]),
            source_locator="Table 6",
            is_figure="0",
            evidence="site=%s; season=%s; %s vs %s"
                     % (o["site"], o["season"], o["treatment_label"],
                        o["control_label"]))

    # ---------------- 184_Yeboah_2018 ----------------
    d = load("184_Yeboah_2018")
    records_in["184_Yeboah_2018"] = len(d["observations"])
    for o in d["observations"]:
        add("184_Yeboah_2018",
            row_id="184_Yeboah_2018__ai__%d" % o["obs_id"],
            crop="spring_wheat",
            treatment_level=sig3(o["biochar_rate_t_ha"]),
            co_amendment="nitrogen",
            co_amendment_level=sig3(o["N_rate_kg_ha"]),
            timepoint="y%d" % o["year"],
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token="cofactor_matched_control",
            treatment_mean=fmt_mean(o["treatment_mean"], o["yield_unit"]),
            control_mean=fmt_mean(o["control_mean"], o["yield_unit"]),
            source_locator="Table 5",
            is_figure="0",
            evidence="%s vs %s; %d"
                     % (o["treatment_label"], o["control_label"], o["year"]))

    # ---------------- 193_Islami_2011 ----------------
    d = load("193_Islami_2011")
    records_in["193_Islami_2011"] = len(d["observations"])

    def isl_time(y):
        y = str(y)
        return "y" + (y.split("-")[0] if "-" in y else y)

    for o in d["observations"]:
        isfig = o.get("source_type") == "figure"
        bt = o.get("biochar_type", "")
        add("193_Islami_2011",
            row_id="193_Islami_2011__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o["biochar_rate_Mg_ha"]),
            co_amendment="none",
            co_amendment_level="0",
            timepoint=isl_time(o["year"]),
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token="absolute_control",
            treatment_mean=fmt_mean(o["treatment_mean"], o["yield_unit"]),
            control_mean=fmt_mean(o["control_mean"], o["yield_unit"]),
            source_locator=(o.get("data_source")
                            or ("Fig 1" if isfig else "Table 2")),
            is_figure=("1" if isfig else "0"),
            evidence="biochar_type=%s; system=%s; %s vs %s; year=%s"
                     % (bt, o.get("system", ""), o["treatment_label"],
                        o["control_label"], o["year"]))

    # ---------------- 207_Liu_2019 ----------------
    d = load("207_Liu_2019")
    records_in["207_Liu_2019"] = len(d["observations"])
    for o in d["observations"]:
        add("207_Liu_2019",
            row_id="207_Liu_2019__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o["biochar_rate_t_ha"]),
            co_amendment="nitrogen",
            co_amendment_level=sig3(o["N_rate_kg_ha"]),
            timepoint="y%d" % o["year"],
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token="cofactor_matched_control",
            treatment_mean=fmt_mean(o["treatment_mean"], o["yield_unit"]),
            control_mean=fmt_mean(o["control_mean"], o["yield_unit"]),
            source_locator=(o.get("data_source") or "Fig 1"),
            is_figure="1",
            evidence="%s vs %s; %d"
                     % (o["treatment_label"], o["control_label"], o["year"]))

    # ---------------- 219_Xie_2021 ----------------
    d = load("219_Xie_2021")
    records_in["219_Xie_2021"] = len(d["observations"])
    n = 0
    for o in d["observations"]:
        n += 1
        season = o["season"]
        if "mean" in season.lower():
            tp, agg = "pooled", "pooled"
        else:
            tp, agg = "season7", "single_cell"
        add("219_Xie_2021",
            row_id="219_Xie_2021__ai__%d" % n,
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o["biochar_rate_t_ha"]),
            co_amendment="none",
            co_amendment_level="0",
            timepoint=tp,
            aggregation_level=agg,
            unit_canonical="kg/ha",
            control_token="absolute_control",
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("yield_unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("yield_unit")),
            source_locator=(o.get("data_source") or "Fig 3"),
            is_figure="1",
            evidence="%s vs %s; season=%s; effect_pct=%s"
                     % (o["treatment"], o["control"], season,
                        o.get("effect_pct")))

    # ---------------- 223_Dong_2019 ----------------
    d = load("223_Dong_2019")
    records_in["223_Dong_2019"] = len(d["observations"])
    n = 0
    for o in d["observations"]:
        n += 1
        add("223_Dong_2019",
            row_id="223_Dong_2019__ai__%d" % n,
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o["biochar_rate_t_ha"]),
            co_amendment="nitrogen",
            co_amendment_level="250",
            timepoint="y" + o["season"].split("-")[0],
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token="cofactor_matched_control",
            treatment_mean=fmt_mean(o.get("treatment_mean"), "t/ha"),
            control_mean=fmt_mean(o.get("control_mean"), "t/ha"),
            source_locator="Table 3",
            is_figure="0",
            evidence="biochar_type=%s; %s vs %s; season=%s"
                     % (o.get("biochar_type", ""), o["treatment"],
                        o["control"], o["season"]))

    # ---------------- 227_Niu_2017 ----------------
    d = load("227_Niu_2017")
    records_in["227_Niu_2017"] = len(d["observations"])
    n = 0
    for o in d["observations"]:
        n += 1
        fert = (o.get("fertilizer") or "none").lower()
        has_fert = fert not in ("none", "", "0")
        st = o.get("source_type")
        isfig = st in ("figure",)
        add("227_Niu_2017",
            row_id="227_Niu_2017__ai__%d" % n,
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o["biochar_rate_t_ha"]),
            co_amendment=("nitrogen" if has_fert else "none"),
            co_amendment_level=("200" if has_fert else "0"),
            timepoint="y" + str(o["season"]).split("-")[0],
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token=("cofactor_matched_control" if has_fert
                           else "absolute_control"),
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("yield_unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("yield_unit")),
            source_locator=(o.get("data_source") or "Fig 2"),
            is_figure=("1" if isfig else "0"),
            evidence="%s vs %s; season=%s; src=%s"
                     % (o["treatment"], o["control"], o["season"], st))

    # ---------------- 081_Deenik_2010 ----------------
    d = load("081_Deenik_2010")
    records_in["081_Deenik_2010"] = len(d["observations"])
    for o in d["observations"]:
        fl = (o.get("fertilizer") or "").lower()
        if fl == "none":
            coa, coal, ctl = "none", "0", "absolute_control"
        elif "lime" in fl or "npk" in fl:
            coa, coal, ctl = "lime_npk_fertilizer", "present", "cofactor_matched_control"
        elif "n fertilizer" in fl or fl.startswith("+n"):
            coa, coal, ctl = "nitrogen_fertilizer", "present", "cofactor_matched_control"
        else:
            coa, coal, ctl = "none", "0", "absolute_control"
        add("081_Deenik_2010",
            row_id="081_Deenik_2010__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o.get("biochar_rate_pct")),
            co_amendment=coa,
            co_amendment_level=coal,
            timepoint="pooled",
            aggregation_level="single_cell",
            unit_canonical="g/pot",
            control_token=ctl,
            treatment_mean=fmt_mean(o.get("treatment_mean"), "g/pot"),
            control_mean=fmt_mean(o.get("control_mean"), "g/pot"),
            source_locator=(o.get("data_source") or ""),
            is_figure="1",
            evidence="%s vs %s; biochar_pct=%s; fert=%s; feedstock=%s; exp=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_pct"), o.get("fertilizer"),
                        o["moderators"].get("feedstock"), o.get("experiment")))

    # ---------------- 082_Jose_2013 ----------------
    d = load("082_Jose_2013")
    records_in["082_Jose_2013"] = len(d["observations"])
    for o in d["observations"]:
        fert = o.get("fertilizer") or ""
        if fert.startswith("F0"):
            coa, coal, ctl = "none", "0", "absolute_control"
        elif fert.startswith("F40"):
            coa, coal, ctl = "mineral_fertilizer", "40", "cofactor_matched_control"
        elif fert.startswith("F100"):
            coa, coal, ctl = "mineral_fertilizer", "100", "cofactor_matched_control"
        else:
            coa, coal, ctl = "none", "0", "absolute_control"
        add("082_Jose_2013",
            row_id="082_Jose_2013__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o.get("biochar_rate_pct")),
            co_amendment=coa,
            co_amendment_level=coal,
            timepoint="pooled",
            aggregation_level="single_cell",
            unit_canonical="g/pot",
            control_token=ctl,
            treatment_mean=fmt_mean(o.get("treatment_mean"), "g/pot"),
            control_mean=fmt_mean(o.get("control_mean"), "g/pot"),
            source_locator=(o.get("data_source") or "Fig 2"),
            is_figure="1",
            evidence="%s vs %s; biochar_type=%s; biochar_pct=%s; fert=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_type"), o.get("biochar_rate_pct"), fert))

    # ---------------- 101_Liang_Feng_2014 ----------------
    d = load("101_Liang_Feng_2014")
    records_in["101_Liang_Feng_2014"] = len(d["observations"])
    for o in d["observations"]:
        add("101_Liang_Feng_2014",
            row_id="101_Liang_Feng_2014__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower().replace(" + ", "+").replace(" ", "_"),
            treatment_level=sig3(o.get("biochar_rate_tha")),
            co_amendment="npk_fertilizer",
            co_amendment_level="112.5",
            timepoint="pooled",
            aggregation_level="pooled",
            unit_canonical="kg/ha",
            control_token="cofactor_matched_control",
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("unit")),
            source_locator=(o.get("data_source") or "Table 3"),
            is_figure="0",
            evidence="%s vs %s; biochar_tha=%s; season=%s; fert=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_tha"), o.get("season"),
                        o.get("fertilizer")))

    # ---------------- 116_Farrell_2014 ----------------
    d = load("116_Farrell_2014")
    records_in["116_Farrell_2014"] = len(d["observations"])
    for o in d["observations"]:
        fert = o.get("fertilizer") or ""
        if fert.startswith("0"):
            coa, coal, ctl = "none", "0", "absolute_control"
        elif "35 kg DAP" in fert:
            coa, coal, ctl = "phosphorus_dap", "8.8", "cofactor_matched_control"
        elif "70 kg DAP" in fert:
            coa, coal, ctl = "phosphorus_dap", "17.6", "cofactor_matched_control"
        else:
            coa, coal, ctl = "none", "0", "absolute_control"
        yr = str(o.get("season") or "")
        tp = "y" + yr if yr.isdigit() else "pooled"
        add("116_Farrell_2014",
            row_id="116_Farrell_2014__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o.get("biochar_rate_tha")),
            co_amendment=coa,
            co_amendment_level=coal,
            timepoint=tp,
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token=ctl,
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("unit")),
            source_locator=(o.get("data_source") or "Fig 2"),
            is_figure="1",
            evidence="%s vs %s; biochar_tha=%s; application=%s; P=%s; season=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_tha"), o.get("biochar_application"),
                        fert, yr))

    # ---------------- 126_Arif_2017 ----------------
    d = load("126_Arif_2017")
    records_in["126_Arif_2017"] = len(d["observations"])

    def _lvl(s):
        return "100" if "100" in s else ("75" if "75" in s
                                         else ("50" if "50" in s else ""))

    for o in d["observations"]:
        ps = o["moderators"].get("P_source") or o.get("fertilizer") or ""
        psl = ps.lower()
        if psl.startswith("control"):
            coa, coal = "none", "0"
        elif "fym" in psl:
            coa, coal = "farmyard_manure", _lvl(ps)
        elif "pm" in psl:
            coa, coal = "poultry_manure", _lvl(ps)
        elif "cf" in psl or "dap" in psl:
            coa, coal = "chemical_fertilizer_dap", "100"
        else:
            coa, coal = "none", "0"
        add("126_Arif_2017",
            row_id="126_Arif_2017__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o.get("biochar_rate_tha")),
            co_amendment=coa,
            co_amendment_level=coal,
            timepoint="pooled",
            aggregation_level="pooled",
            unit_canonical="kg/ha",
            control_token="cofactor_matched_control",
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("unit")),
            source_locator=(o.get("data_source") or "Fig 2"),
            is_figure="1",
            evidence="%s vs %s; biochar_tha=%s; P_source=%s; n=%s(2yr)"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_tha"), ps, o.get("n")))

    # ---------------- 130_Azeem_2019 ----------------
    d = load("130_Azeem_2019")
    records_in["130_Azeem_2019"] = len(d["observations"])
    for o in d["observations"]:
        fert = (o.get("fertilizer") or "").strip()
        if fert.lower() == "none":
            coa, coal, ctl = "none", "0", "absolute_control"
        elif fert.upper() == "NPK":
            coa, coal, ctl = "npk_fertilizer", "present", "cofactor_matched_control"
        else:
            coa, coal, ctl = "none", "0", "absolute_control"
        season = str(o.get("season") or "")
        tp = "y" + season[:4] if season[:4].isdigit() else "pooled"
        add("130_Azeem_2019",
            row_id="130_Azeem_2019__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower().replace(" ", "_"),
            treatment_level=sig3(o.get("biochar_rate_tha")),
            co_amendment=coa,
            co_amendment_level=coal,
            timepoint=tp,
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token=ctl,
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("unit")),
            source_locator=(o.get("data_source") or "Table 5"),
            is_figure="0",
            evidence="%s vs %s; biochar_tha=%s; fert=%s; season=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_tha"), fert, season))

    # ---------------- 133_Pandit_2018 ----------------
    d = load("133_Pandit_2018")
    records_in["133_Pandit_2018"] = len(d["observations"])
    for o in d["observations"]:
        yr = str(o.get("year") or "")
        m = re.search(r"(20\d\d)", yr)
        tp = "y" + m.group(1) if m else "pooled"
        isfig = "1" if o.get("source_type") == "figure" else "0"
        add("133_Pandit_2018",
            row_id="133_Pandit_2018__ai__%d" % o["obs_id"],
            crop=o["crop"].strip().lower(),
            treatment_level=sig3(o.get("biochar_rate_t_ha")),
            co_amendment="none",
            co_amendment_level="0",
            timepoint=tp,
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token="absolute_control",
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("yield_unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("yield_unit")),
            source_locator=(o.get("data_source") or ""),
            is_figure=isfig,
            evidence="%s vs %s; biochar_tha=%s; year=%s; src=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_t_ha"), yr, o.get("source_type")))

    # ---------------- 145_Omara_2020 ----------------
    d = load("145_Omara_2020")
    records_in["145_Omara_2020"] = len(d["observations"])
    for o in d["observations"]:
        yr = str(o.get("year") or "")
        tp = "y" + yr if yr.isdigit() else "pooled"
        add("145_Omara_2020",
            row_id="145_Omara_2020__ai__%d" % o["obs_id"],
            crop=(o.get("crop") or "maize").strip().lower(),
            treatment_level=sig3(o.get("biochar_rate_Mg_ha")),
            co_amendment="nitrogen",
            co_amendment_level=sig3(o.get("N_rate_kg_ha")),
            timepoint=tp,
            aggregation_level="single_cell",
            unit_canonical="kg/ha",
            control_token="cofactor_matched_control",
            treatment_mean=fmt_mean(o.get("treatment_mean"), o.get("yield_unit")),
            control_mean=fmt_mean(o.get("control_mean"), o.get("yield_unit")),
            source_locator=("site %s" % o.get("site")),
            is_figure="0",
            evidence="%s vs %s; biochar_Mgha=%s; N_kgha=%s; site=%s; year=%s; src=%s"
                     % (o.get("treatment_label"), o.get("control_label"),
                        o.get("biochar_rate_Mg_ha"), o.get("N_rate_kg_ha"),
                        o.get("site"), yr, o.get("source_type")))


# ==========================================================================
# STAGE C -- NEW.  papers 229, 231, 234, 242.
# No submitted decoder script exists for these four (their deposited AI key
# CSVs were hand-authored).  Decoded here from the March records' own explicit
# treatment / crop / season / amendment / unit / source fields, following the
# same conventions as STAGE A/B.
# ==========================================================================
def legacy_time(value) -> str:
    text = str(value or "")
    return "season" + text.replace("-", "_") if "-" in text else "y" + text


def legacy_isfig(o) -> str:
    hit = ("fig" in str(o.get("data_source") or "").lower()
           or str(o.get("source_type") or "").lower() == "figure")
    return "1" if hit else "0"


def stage_c():
    # ---------------- 229_Shi_2022 ----------------
    d = load("229_Shi_2022")
    records_in["229_Shi_2022"] = len(d["observations"])
    for i, o in enumerate(d["observations"]):
        annual = str(o.get("crop") or "").lower() == "annual total"
        add("229_Shi_2022",
            row_id="229_Shi_2022__ai__%d" % i,
            crop=re.sub(r"\s+", "_", str(o.get("crop") or "").lower()),
            treatment_level=sig3(o.get("biochar_rate_t_ha")),
            co_amendment="none",
            co_amendment_level="0",
            timepoint=("pooled" if annual else legacy_time(o.get("season"))),
            aggregation_level=("pooled" if annual else "single_cell"),
            unit_canonical=(o.get("yield_unit") or ""),
            control_token="absolute_control",   # control label is "CK"
            treatment_mean=fmt_raw(o.get("treatment_mean")),
            control_mean=fmt_raw(o.get("control_mean")),
            source_locator=(o.get("data_source") or ""),
            is_figure=legacy_isfig(o),
            evidence="%s vs %s; crop=%s; season=%s; rate=%s"
                     % (o.get("treatment"), o.get("control"), o.get("crop"),
                        o.get("season"), o.get("biochar_rate_t_ha")))

    # ---------------- 231_Zhang_2021 ----------------
    d = load("231_Zhang_2021")
    records_in["231_Zhang_2021"] = len(d["observations"])
    for i, o in enumerate(d["observations"]):
        fert = str(o.get("fertilizer") or "").lower()
        if "chemical" in fert:
            coa, coal = "chemical_fertilizer", "240_kgN_ha"
        else:
            coa, coal = "organic_fertilizer", "20pct_N_replaced"
        ctl_label = str(o.get("control") or "")
        # control_token from the row's OWN control label: CK = unamended.
        ctok = ("absolute_control" if ctl_label.upper().startswith("CK")
                else "cofactor_matched_control")
        add("231_Zhang_2021",
            row_id="231_Zhang_2021__ai__%d" % i,
            crop=re.sub(r"\s+", "_", str(o.get("crop") or "").lower()),
            treatment_level=sig3(o.get("biochar_rate_t_ha")),
            co_amendment=coa,
            co_amendment_level=coal,
            timepoint=legacy_time(o.get("season")),
            aggregation_level="single_cell",
            unit_canonical=(o.get("yield_unit") or ""),
            control_token=ctok,
            treatment_mean=fmt_raw(o.get("treatment_mean")),
            control_mean=fmt_raw(o.get("control_mean")),
            source_locator=(o.get("data_source") or ""),
            is_figure=legacy_isfig(o),
            evidence="%s vs %s; fert=%s; season=%s; rate=%s"
                     % (o.get("treatment"), o.get("control"),
                        o.get("fertilizer"), o.get("season"),
                        o.get("biochar_rate_t_ha")))

    # ---------------- 234_Malik_2018 ----------------
    # SCHEMA ADAPTER: this paper's records carry NO 'treatment_mean' field.
    # Each record supplies two outcomes explicitly:
    #   treatment_mean_biomass / control_mean_biomass  (straw biomass)
    #   treatment_mean_grain   / control_mean_grain    (grain yield)
    # -> one key row per outcome (14 records -> 28 rows).
    d = load("234_Malik_2018")
    records_in["234_Malik_2018"] = len(d["observations"])
    for i, o in enumerate(d["observations"]):
        feed = str(o.get("biochar_feedstock") or "").lower()
        raw_rate = o.get("biochar_rate_pct")
        rate = "" if raw_rate is None else sig3(raw_rate)
        tl = "0" if rate == "0" else "%s_%spct" % (feed, rate)
        lime = o.get("lime_rate_pct")
        common = dict(
            treatment_level=tl,
            co_amendment="quicklime",
            co_amendment_level=("0" if lime is None else sig3(lime)),
            timepoint="pooled",
            aggregation_level="single_cell",
            unit_canonical=(o.get("yield_unit") or ""),
            control_token="absolute_control",   # control label is "CK"
            source_locator=(o.get("data_source") or ""),
            is_figure=legacy_isfig(o),
        )
        ev = ("%s vs %s; feedstock=%s; biochar_pct=%s; lime_pct=%s"
              % (o.get("treatment"), o.get("control"), feed, raw_rate, lime))
        add("234_Malik_2018",
            row_id="234_Malik_2018__ai__%d_biomass" % i,
            crop="wheat_biomass",
            treatment_mean=fmt_raw(o.get("treatment_mean_biomass")),
            control_mean=fmt_raw(o.get("control_mean_biomass")),
            evidence=ev + "; outcome=biomass", **common)
        add("234_Malik_2018",
            row_id="234_Malik_2018__ai__%d_grain" % i,
            crop="wheat",
            treatment_mean=fmt_raw(o.get("treatment_mean_grain")),
            control_mean=fmt_raw(o.get("control_mean_grain")),
            evidence=ev + "; outcome=grain", **common)

    # ---------------- 242_Liu_2014 ----------------
    d = load("242_Liu_2014")
    records_in["242_Liu_2014"] = len(d["observations"])
    for i, o in enumerate(d["observations"]):
        add("242_Liu_2014",
            row_id="242_Liu_2014__ai__%d" % i,
            crop=re.sub(r"\s+", "_", str(o.get("crop") or "").lower()),
            treatment_level=sig3(o.get("biochar_rate_t_ha")),
            co_amendment="none",
            co_amendment_level="0",
            timepoint=legacy_time(o.get("season")),
            aggregation_level="single_cell",
            unit_canonical=(o.get("yield_unit") or ""),
            control_token="absolute_control",   # control label is "CK (0 t/ha)"
            treatment_mean=fmt_raw(o.get("treatment_mean")),
            control_mean=fmt_raw(o.get("control_mean")),
            source_locator=(o.get("data_source") or ""),
            is_figure=legacy_isfig(o),
            evidence="%s vs %s; crop=%s; season=%s; rate=%s"
                     % (o.get("treatment"), o.get("control"), o.get("crop"),
                        o.get("season"), o.get("biochar_rate_t_ha")))


# ==========================================================================
# STAGE D -- AI-side canonicalisations carried over from
#            runs/biochar/patch_iter0.py.  GT-side patches are NOT reproduced
#            (03_KEYS/gt is frozen as-submitted and already carries them).
#            Every edit is justified from the row's OWN fields/evidence.
# ==========================================================================
def num(s):
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def fmt(v):
    if v is None:
        return ""
    if v == int(v):
        return str(int(v))
    return ("%.6f" % v).rstrip("0").rstrip(".")


def patch(pid, fn):
    for r in rows_by_paper.get(pid, []):
        fn(r)


def stage_d():
    # ---- 153_Wei_2022 ----
    def fix_153(r):
        if r["crop"] == "winter_rapeseed":
            r["crop"] = "rapeseed"
        if r["co_amendment"] == "nitrogen":
            r["co_amendment"] = "nitrogen_fertilizer"
    patch("153_Wei_2022", fix_153)

    # ---- 041_Guerena_2013 ----
    def fix_041(r):
        if r["co_amendment"] == "nitrogen":
            r["co_amendment"] = "nitrogen_fertilizer"
    patch("041_Guerena_2013", fix_041)

    # ---- 130_Azeem_2019 ----
    def fix_130(r):
        if r["co_amendment"] == "npk_fertilizer":
            r["co_amendment"] = "npk"
        if r["co_amendment_level"] == "present":
            r["co_amendment_level"] = "23"
    patch("130_Azeem_2019", fix_130)

    # ---- 223_Dong_2019: aged+fresh biochar TOTAL rate, read off own labels ----
    def fix_223(r):
        ev = r.get("evidence", "")
        if "N1Ba2f1" in ev or "fresh biochar 10" in ev:
            r["treatment_level"] = "30"
        elif "N1Ba4f2" in ev or "fresh biochar 20" in ev:
            r["treatment_level"] = "60"
    patch("223_Dong_2019", fix_223)

    # ---- 184_Yeboah_2018 ----
    def fix_184(r):
        if r["crop"] == "spring_wheat":
            r["crop"] = "wheat"
    patch("184_Yeboah_2018", fix_184)

    # ---- 145_Omara_2020: recover site from own source_locator into timepoint --
    def fix_145(r):
        m = re.search(r"site\s+(\w+)", r.get("source_locator") or "", re.I)
        if m:
            site = m.group(1).strip().lower()
            if not r["timepoint"].endswith("_" + site):
                r["timepoint"] = r["timepoint"] + "_" + site
    patch("145_Omara_2020", fix_145)

    # ---- 193_Islami_2011: recover biochar feedstock from own evidence ----
    def fix_193(r):
        ev = (r.get("evidence") or "").lower()
        if "biochar_type=fym" in ev or "fym biochar" in ev:
            r["co_amendment"] = "fym_biochar"
        elif "cassava stem biochar" in ev or "biochar_type=cassava" in ev:
            r["co_amendment"] = "cassava_stem_biochar"
    patch("193_Islami_2011", fix_193)

    # ---- 166_Haefele_2011: recover site from own evidence into timepoint ----
    def fix_166(r):
        ev = r.get("evidence") or ""
        m = re.search(r"site=(\w+)", ev)
        if r["co_amendment"] == "fertilizer":
            r["co_amendment"] = "npk"
            r["co_amendment_level"] = "medium"
        if m and r["timepoint"] != "pooled":
            site = m.group(1).strip().lower()
            mm = re.search(r"y?(\d{4})_?(ws|ds)", r["timepoint"], re.I)
            if mm:
                r["timepoint"] = "%s_%s%s" % (site, mm.group(1),
                                              mm.group(2).lower())
    patch("166_Haefele_2011", fix_166)

    # ---- 234_Malik_2018 ----
    # patch_iter0's fix_234 rebuilt treatment_level as '<feedstock>_<pct>pct'
    # and set co_amendment 'quicklime'.  STAGE C already emits both, so the
    # patch is a no-op here; retained as an explicit assertion.
    for r in rows_by_paper.get("234_Malik_2018", []):
        assert r["co_amendment"] == "quicklime"

    # ---- 101_Liang_Feng_2014: 4-season TOTAL -> documented pooled MEAN ----
    def fix_101(r):
        if r["co_amendment"] == "npk_fertilizer":
            r["co_amendment"] = "none"
            r["co_amendment_level"] = "0"
        if r["aggregation_level"] == "pooled":
            r["aggregation_level"] = "documented_pooled"
        for col in ("treatment_mean", "control_mean"):
            v = num(r.get(col))
            if v is not None:
                r[col] = fmt(v / 4.0)
    patch("101_Liang_Feng_2014", fix_101)

    # ---- 063_Asai_2009: recover cultivar from own evidence into timepoint ----
    def fix_063(r):
        ev = (r.get("evidence") or "").lower()
        if r["co_amendment"] == "nitrogen" and r["co_amendment_level"] in ("0", "0.0"):
            r["co_amendment"] = "none"
        if r["timepoint"].upper() == "HK1":
            if "apo cultivar" in ev:
                r["timepoint"] = "hk1_apo"
            elif "vieng cultivar" in ev:
                r["timepoint"] = "hk1_vieng"
    patch("063_Asai_2009", fix_063)


# ==========================================================================
# STAGE E -- NEW.  Declared-unit STRING normalisation.
# Notation-only: strips parenthetical qualifiers and maps dimensionally
# IDENTICAL notations onto one spelling.  No value is rescaled here; no
# factor other than 1 is involved.  Purely a string operation on the unit
# the source record itself declared, so it is outcome-blind.
# ==========================================================================
UNIT_SYNONYMS = {
    "mg/ha": "t/ha",        # 1 Mg == 1 tonne  (exact identity)
    "g pot-1": "g/pot",
    "g pot^-1": "g/pot",
    "g/pot": "g/pot",
    "g/250 cm2": "g/250cm2",
}


def stage_e():
    for pid in sorted(rows_by_paper):
        for r in rows_by_paper[pid]:
            before = r["unit_canonical"]
            u = re.sub(r"\(.*?\)", "", before).strip()
            u = re.sub(r"\s+", " ", u).strip()
            u = UNIT_SYNONYMS.get(u.lower(), u)
            if u != before:
                r["unit_canonical"] = u
                unit_log.append((pid, r["row_id"], before, u))


# ==========================================================================
# STAGE F -- NEW.  control_token -> closed vocabulary.
#
# GT uses only {absolute_control, cofactor_matched_control}.  The submitted AI
# decoders wrote (a) raw control descriptions as free text for the STAGE A
# papers and (b) a third token 'co_factor_present_unmatched'.  control_token is
# deliberately NOT part of the match key, so this changes no pairing; it makes
# the recorded control-definition concordance measurable.
#
# Rule (deterministic, from each row's OWN control description + its own
# decoded co_amendment_level):
#   F1  control records NO co-amendment / fertiliser background (zero/absent)
#         -> absolute_control
#   F2  control records the co-amendment at the SAME level this row carries
#         -> cofactor_matched_control
#   F3  control records a co-amendment background at a DIFFERENT level
#         -> absolute_control   (no unmatched-cofactor token exists;
#                                logged as 'demoted_unmatched_cofactor')
# ==========================================================================
ZERO_LEVELS = {"0", "0.0", "", "none"}


def _ctrl_desc(row):
    """The control description this row itself recorded."""
    ev = row.get("evidence") or ""
    m = re.search(r"\|\|\s*C:\s*(.*)$", ev)
    if m:
        return m.group(1).strip()
    return row.get("control_token") or ""


def stage_f():
    for pid in sorted(rows_by_paper):
        for r in rows_by_paper[pid]:
            before = r["control_token"]
            if before in CLOSED_CONTROL_VOCAB:
                continue

            if before == "co_factor_present_unmatched":
                r["control_token"] = "absolute_control"
                ctok_log.append((pid, r["row_id"], before, "absolute_control",
                                 "F3 demoted_unmatched_cofactor"))
                continue

            cdesc = _ctrl_desc(r)
            cl = cdesc.lower()
            lvl = (r["co_amendment_level"] or "").strip()

            if r["co_amendment"] == "none" or lvl.lower() in ZERO_LEVELS:
                after, rule = "absolute_control", "F1 no_cofactor_in_control"
            else:
                # does the control description carry this row's co-amendment
                # at this row's level?
                matched = False
                if re.search(r"\bno\s+pm\b", cl) or "nil n" in cl:
                    matched = False
                elif pid == "001_Adekiya_2019":
                    matched = bool(re.search(r"pm\s*[\d.]+\s*t/ha", cl))
                elif pid == "007_Gathorne-Hardy_2009":
                    m = re.search(r"(\d+)\s*n\s*kg/ha", cl)
                    matched = bool(m) and m.group(1) == lvl
                elif pid == "016_Li_B_2016":
                    matched = ("conv. n" in cl) or ("4/3" in cl)
                elif pid == "021_Nobile_2022":
                    # control = mineral fertiliser, i.e. compost@8 is ABSENT
                    matched = False
                elif pid == "041_Guerena_2013":
                    matched = "90% n" in cl
                elif pid == "063_Asai_2009":
                    m = re.search(r"n\s*(\d+)\s*kg/ha", cl)
                    matched = bool(m) and m.group(1) == lvl
                elif pid == "078_Wang_2012":
                    matched = bool(re.search(r"\bb0\s+n\b", cl))
                if matched:
                    after, rule = ("cofactor_matched_control",
                                   "F2 cofactor_matched_in_control")
                else:
                    after, rule = ("absolute_control",
                                   "F3 demoted_unmatched_cofactor")
            r["control_token"] = after
            ctok_log.append((pid, r["row_id"], before, after, rule))


# ==========================================================================
# write + audit
# ==========================================================================
def main():
    all_json = sorted(f for f in os.listdir(SRC) if f.endswith(".json"))
    present = []
    for f in all_json:
        if is_paper_extraction(os.path.join(SRC, f)):
            present.append(f[:-5])
        else:
            exclusions.append((f, "whole file",
                               "non-extraction pipeline artefact, not a paper "
                               "(no top-level paper_id + observations list); "
                               "contents not read"))
    print("frozen input dir: %d .json files -> %d paper extractions, "
          "%d artefacts excluded"
          % (len(all_json), len(present), len(exclusions)))

    stage_a()
    stage_b()
    stage_c()
    stage_d()
    stage_e()
    stage_f()

    os.makedirs(OUT, exist_ok=True)
    total = 0
    for pid in sorted(rows_by_paper):
        path = os.path.join(OUT, pid + ".csv")
        with open(path, "w", newline="\n", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\n")
            w.writeheader()
            for r in rows_by_paper[pid]:
                w.writerow(r)
        total += len(rows_by_paper[pid])

    decoded = set(rows_by_paper)
    missing = [p for p in present if p not in decoded]
    for p in missing:
        exclusions.append((p, "whole paper", "no decode branch"))

    # ---- audit log (deterministic) ----
    lines = []
    lines.append("# biochar AI-side rebuild -- decoder audit log")
    lines.append("")
    lines.append("source: 01_INPUTS_FROZEN/biochar (frozen March 2026 JSONs)")
    lines.append("decoder tag: %s" % DECODER)
    lines.append("")
    lines.append("input filter (structural, no filename hardcoding): a file is a "
                 "paper extraction iff it is a JSON object with a truthy "
                 "top-level `paper_id` AND a top-level `observations` list.")
    lines.append("")
    lines.append("- .json files in frozen dir: %d" % len(all_json))
    lines.append("- paper extractions accepted: %d" % len(present))
    lines.append("- non-extraction artefacts excluded: %d" % len(exclusions))
    lines.append("- source observations across accepted papers: %d"
                 % sum(records_in.values()))
    lines.append("")
    lines.append("## record arithmetic")
    lines.append("")
    lines.append("| paper_id | march_records_in | key_rows_out | delta | note |")
    lines.append("|---|---|---|---|---|")
    for pid in sorted(rows_by_paper):
        ri = records_in.get(pid, 0)
        ro = len(rows_by_paper[pid])
        note = ""
        if ro != ri:
            note = "1 record -> 2 rows (biomass + grain outcomes)" if pid == "234_Malik_2018" else "see ledger"
        lines.append("| %s | %d | %d | %+d | %s |" % (pid, ri, ro, ro - ri, note))
    lines.append("| **TOTAL** | **%d** | **%d** | **%+d** | |"
                 % (sum(records_in.values()), total,
                    total - sum(records_in.values())))
    lines.append("")
    lines.append("papers decoded: %d ; paper JSONs present: %d ; "
                 "records dropped: 0" % (len(decoded), len(present)))
    lines.append("")
    lines.append("## exclusions")
    lines.append("")
    lines.append("| item | ref | reason |")
    lines.append("|---|---|---|")
    for a, b, c in exclusions:
        lines.append("| %s | %s | %s |" % (a, b, c))
    lines.append("")
    lines.append("## STAGE E unit-string normalisations (%d rows)" % len(unit_log))
    lines.append("")
    lines.append("| paper_id | rows | before | after |")
    lines.append("|---|---|---|---|")
    agg = {}
    for pid, rid, b, a in unit_log:
        agg[(pid, b, a)] = agg.get((pid, b, a), 0) + 1
    for (pid, b, a), n in sorted(agg.items()):
        lines.append("| %s | %d | `%s` | `%s` |" % (pid, n, b, a))
    lines.append("")
    lines.append("## STAGE F control_token closures (%d rows)" % len(ctok_log))
    lines.append("")
    lines.append("| paper_id | rows | before (free text / non-closed) | after | rule |")
    lines.append("|---|---|---|---|---|")
    agg2 = {}
    for pid, rid, b, a, rule in ctok_log:
        agg2[(pid, b, a, rule)] = agg2.get((pid, b, a, rule), 0) + 1
    for (pid, b, a, rule), n in sorted(agg2.items()):
        lines.append("| %s | %d | `%s` | %s | %s |" % (pid, n, b, a, rule))
    lines.append("")
    if notes:
        lines.append("## decoder notes")
        lines.append("")
        for n in sorted(set(notes)):
            lines.append("- %s" % n)
        lines.append("")

    os.makedirs(LOGDIR, exist_ok=True)
    with open(os.path.join(LOGDIR, "biochar_DECODER_AUDIT.md"),
              "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    for pid in sorted(rows_by_paper):
        print("%-28s %3d records -> %3d rows"
              % (pid, records_in.get(pid, 0), len(rows_by_paper[pid])))
    print("TOTAL rows: %d (from %d March records, %d papers)"
          % (total, sum(records_in.values()), len(decoded)))
    if missing:
        print("UNDECODED PAPERS:", missing)


if __name__ == "__main__":
    main()
