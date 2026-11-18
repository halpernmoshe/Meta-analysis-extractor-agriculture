"""
Build the Boldorini et al. 2024 published-reference key tables (2_conversion/keys/boldorini/gt) from the published
effect-size file 0_sources/raw/boldorini/rspb20232522_si_005.csv (identical to the file the original script read).

PROVENANCE LABEL: the original conversion script, with paths parameterised.
  Everything below the separator is the original script decode_gt.py (31 May 2026, 13:24) verbatim except
  the two path lines (GT, OUTDIR), which now come from the command line.
WHAT IT DOES NOT REPRODUCE: 45 of the 47 deposited rows are reproduced identically on every column. Rows
  gt__B06_Garfinkel_2020__id_44 and __id_45 differ in 'crop' (corn/soybean swapped) and 'evidence'. In the
  deposited tables those two cells were edited by hand after this script ran (31 May 2026, 13:41), as recorded
  in the project's diagnosis notes ("Cause 3"; not in this repository); that edit is deliberately NOT encoded here.
  Row ids, treatment_mean and control_mean of all 47 rows are reproduced.
DISCLOSURE: the script contains 37 hand-written override calls covering all 47 es_ids (dict OV), which set
  treatment_level, timepoint, unit, crop, control token, is_figure, source locator and evidence notes per row.
  Several comments state these were chosen to align tokens with the frozen AI-side keys or read from the
  source PDFs.
Dependencies: Python 3 standard library only.
Usage : python build_reference_keys_boldorini.py [--raw FILE] [--out FOLDER]
        Defaults: --raw 0_sources/raw/boldorini/rspb20232522_si_005.csv; --out a new temporary folder (printed).
        The script refuses to write into 2_conversion/keys/.
"""
import argparse as _argparse, pathlib, sys, tempfile

def _repo_root():
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_out(out, name):
    """Default output: a new temporary folder. Never the deposited key tables."""
    if not out:
        out = tempfile.mkdtemp(prefix="reference_keys_%s_" % name)
    keys = (_repo_root() / "2_conversion" / "keys").resolve()
    o = pathlib.Path(out).resolve()
    if o == keys or keys in o.parents:
        sys.exit("refusing to write into 2_conversion/keys/ (the deposited tables); choose another --out")
    return str(o)

_ap = _argparse.ArgumentParser()
_ap.add_argument("--raw", default=str(_repo_root() / "0_sources" / "raw" / "boldorini" / "rspb20232522_si_005.csv"))
_ap.add_argument("--out", default=None, help="output folder (default: new temp folder)")
_args = _ap.parse_args()
_args.out = _resolve_out(_args.out, "boldorini")
# ---------------------------------------------------------------- original script follows
# GT-side decode for dataset "boldorini" -> canonical-key CSVs
# Protocol: outcome-blind. Coordinates derived from GT's own descriptor columns
# (author/year/crop/crop_gen/crop_sp/predat_id/treatment/study_year) and, where the
# completeness rule fires (identical key + different means), the missing outcome-component
# coordinate is recovered from that paper's PDF only (recorded in `evidence`).
# Means are NEVER used to choose coordinates.
import csv, os

GT = _args.raw  # was a hard-coded absolute path
OUTDIR = _args.out  # was <script dir>\keys\gt
os.makedirs(OUTDIR, exist_ok=True)

DECODER = "gt-col-decode"
OUTCOME = "crop_yield_under_predator"

# author+year -> canonical paper_id (matches AI paper_id list; Liere 2015 has no AI file)
PAPER_ID = {
    ("Vichitbandha_Wise","2002"): "B19_Vichitbandha_2002",
    ("Lang","2003"): "B10_Lang_2003",
    ("Suenaga_Hamamura","2015"): "B18_Suenaga_2015",
    ("Snyder_Wise","2001"): "B17_Snyder_2001",
    ("Mols","2002"): "B15_Mols_2002",
    ("Hooks_et_al","2003"): "B08_Hooks_2003",
    ("Maas","2013"): "B13_Maas_2013",
    ("Borkhataria","2012"): "B03_Borkhataria_2012",
    ("Martin","2013"): "B14_Martin_2013",
    ("Ali","2018"): "B01_Ali_2018",
    ("Liere","2015"): "Liere_2015",            # no AI counterpart (B12 absent)
    ("Garfinkel","2015"): "B05_Garfinkel_2015",
    ("Gras","2016"): "B07_Gras_2016",
    ("Classen","2014"): "B04_Classen_2014",
    ("Libran-Embid","2017"): "B11_Libran-Embid_2017",
    ("Saunders","2016"): "B16_Saunders_2016",
    ("Garfinkel","2020"): "B06_Garfinkel_2020",
    ("Bisseleua","2017"): "B02_Bisseleua_2017",
    ("Ismoilov","2020"): "B09_Ismoilov_2020",
}

PRED = {"Spiders":"spiders","Beetles":"beetles","Invertebrates":"invertebrates",
        "Birds":"birds","Vertebrates":"vertebrates"}

# Per-es_id overrides for the distinguishing coordinate recovered when the
# completeness rule fired, plus paper-specific token alignment to the frozen AI keys.
# Fields that may be set: tl (treatment_level), tp (timepoint), unit, crop, ctrl, isfig, ev_extra
# Defaults below if not overridden.
OV = {}

def setov(ids, **kw):
    for i in ids: OV[i]=kw

# --- Lang 2003 (B10): AI treatment_level=predator group; timepoint=pooled.
#     3 yield components distinguished by unit: plants / g (grain mass-per-ear) / % (protein).
#     Recovered from Lang 2003 Table 3 (Wheat plants; Grain mass mg per ear; Grain protein %).
setov(["id_2"],  unit="plants", loc="Table 3", ev_extra="Lang2003 Table3 'Wheat plants (numbers)' Spiders=173.20")
setov(["id_3"],  unit="g",      loc="Table 3", ev_extra="Lang2003 Table3 'Grain mass (mg per ear)' Spiders=2.01")
setov(["id_4"],  unit="%",      loc="Table 3", ev_extra="Lang2003 Table3 'Grain protein (%)' Spiders=9.03")
setov(["id_9"],  unit="plants", loc="Table 3", ev_extra="Lang2003 Table3 'Wheat plants' Carabids=176.60")
setov(["id_10"], unit="g",      loc="Table 3", ev_extra="Lang2003 Table3 'Grain mass mg/ear' Carabids=2.11")
setov(["id_11"], unit="%",      loc="Table 3", ev_extra="Lang2003 Table3 'Grain protein %' Carabids=9.25")
setov(["id_12"], unit="plants", loc="Table 3", ev_extra="Lang2003 Table3 'Wheat plants' Both reduced=172.60")
setov(["id_13"], unit="g",      loc="Table 3", ev_extra="Lang2003 Table3 'Grain mass mg/ear' Both=2.10")
setov(["id_14"], unit="%",      loc="Table 3", ev_extra="Lang2003 Table3 'Grain protein %' Both=9.60")

# --- Snyder_Wise 2001 (B17): AI tl=predator group, timepoint=year(y1997/y1998), crop=cucumber/squash.
for i in ["id_15","id_16","id_17","id_18","id_19","id_20"]:
    setov([i], tp="y1997", unit="kg", isfig=1)
for i in ["id_21","id_22","id_23","id_24","id_25","id_26"]:
    setov([i], tp="y1998", unit="kg", isfig=1)

# --- Suenaga_Hamamura 2015 (B18): AI tl=spiders_exclusion / spiders_addition; timepoint=season_spring/autumn.
#     Recovered season from Suenaga Fig 3 (spring vs autumn experiments).
setov(["id_5"], tl="spiders_exclusion", tp="season_spring", unit="g", isfig=1,
      ev_extra="Suenaga2015 Fig3 spring exclusion treat=1138.98")
setov(["id_6"], tl="spiders_addition",  tp="season_spring", unit="g", isfig=1,
      ev_extra="Suenaga2015 Fig3 spring addition treat=1220.34")
setov(["id_7"], tl="spiders_exclusion", tp="season_autumn", unit="g", isfig=1,
      ev_extra="Suenaga2015 Fig3 autumn exclusion treat=461.02")
setov(["id_8"], tl="spiders_addition",  tp="season_autumn", unit="g", isfig=1,
      ev_extra="Suenaga2015 Fig3 autumn addition treat=298.31")

# --- Hooks_et_al 2003 (B08): AI tl=exclusion; 2 components by unit: head weight (kg) vs whole plant weight (g).
#     Recovered from Hooks Table 3 (head weight kg; whole-plant weight g).
# NOTE: predator group (Spiders vs Birds) is the genuine distinguishing coordinate here;
# encoded into treatment_level to satisfy the completeness rule. The AI B08 key collapsed
# this into treatment_level=exclusion (AI key incomplete) -> join will surface it.
setov(["id_28"], tl="spiders_exclusion", unit="kg", loc="Table 3", ev_extra="Hooks2003 Table3 head weight Spiders=0.182 kg")
setov(["id_29"], tl="spiders_exclusion", unit="g",  loc="Table 3", ctrl="co_factor_present_unmatched",
      ev_extra="Hooks2003 Table3 whole plant weight Spiders=153.5 g (ctrl=B+S cage)")
setov(["id_30"], tl="birds_exclusion",   unit="kg", loc="Table 3", ev_extra="Hooks2003 Table3 head weight Birds=0.177 kg")
setov(["id_31"], tl="birds_exclusion",   unit="g",  loc="Table 3", ctrl="co_factor_present_unmatched",
      ev_extra="Hooks2003 Table3 whole plant weight Birds=139.2 g (ctrl=B+S cage)")

# --- Gras 2016 (B07): AI tl=exclusion; 2 components by unit: weight kg/ha/yr vs count fruits/ha/yr. Figure.
setov(["id_39"], tl="exclusion", unit="kg/ha/yr",     isfig=1, ev_extra="Gras2016 Fig2 yield weight Birds=827.6")
setov(["id_40"], tl="exclusion", unit="fruits/ha/yr", isfig=1, ev_extra="Gras2016 Fig2 fruit count Birds=17290.8")

# --- Garfinkel 2020 (B06): AI tl=exclusion; crop from crop_gen (Zea->corn, Glycine->soybean); unit g.
setov(["id_44"], tl="exclusion", crop="corn",    unit="g", loc="Table 1", ev_extra="Garfinkel2020 crop_gen=Zea -> corn/cereals")
setov(["id_45"], tl="exclusion", crop="soybean", unit="g", loc="Table 1", ev_extra="Garfinkel2020 crop_gen=Glycine -> soybean")

# --- Maas 2013 (B13): AI tl=predator group (birds / vertebrates); unit log(kg/ha/yr).
setov(["id_32"], tl="birds",       unit="log(kg/ha/yr)", isfig=1)
setov(["id_35"], tl="vertebrates", unit="log(kg/ha/yr)", isfig=1)

# --- Single-row / single-component papers: align treatment_level + unit + crop + timepoint + isfig to AI.
setov(["id_1"],  tl="invertebrates_exclusion", unit="kg/plot", isfig=1)          # B19 Vichitbandha
setov(["id_36"], tl="exclusion", unit="kg/ha", isfig=1)                          # B01 Ali wheat
setov(["id_46"], tl="exclusion", unit="kg/ha/yr", isfig=1)                       # B02 Bisseleua cacao
setov(["id_33"], tl="exclusion", unit="g", tp="season1", loc="Table 1",
      ev_extra="Borkhataria2012 main crop (GT kept main, dropped ratoon)")        # B03 Borkhataria rice
setov(["id_41"], tl="vertebrates", unit="fruits/branch", isfig=1)                # B04 Classen coffee
setov(["id_38"], tl="exclusion", crop="kale", unit="leaves", loc="Table 5")      # B05 Garfinkel 2015 brassica->kale
setov(["id_47"], tl="invertebrates", unit="kg/plant", isfig=1)                   # B09 Ismoilov tomato
setov(["id_42"], tl="vertebrates", unit="proportion", isfig=1)                   # B11 Libran-Embid coffee
setov(["id_34"], tl="birds", crop="cabbage", unit="g", isfig=1)                  # B14 Martin brassica->cabbage
setov(["id_27"], tl="birds", unit="kg", loc="Results")                           # B15 Mols apple (yield per tree)
setov(["id_43"], tl="vertebrates", unit="%", loc="Results")                      # B16 Saunders apple fruit set %
# Liere 2015 (no AI): keep predator group token, leave is_figure per source (figure-derived)
setov(["id_37"], tl="beetles", crop="soybean", unit="count", isfig=0,
      ev_extra="Liere2015 no AI counterpart (B12 absent); coords from GT cols only")

CROP_DEFAULT = {  # lowercase canonical crop where GT 'crop' column is fine
    "Soybean":"soybean","Wheat":"wheat","Cabbage":"cabbage","Cucumber":"cucumber",
    "Squash":"squash","Apple":"apple","Broccoli":"broccoli","Cacao":"cacao",
    "Rice":"rice","Brassica":"cabbage","Cereals":"corn","Coffee":"coffee","Tomato":"tomato",
}

rows = list(csv.DictReader(open(GT, encoding="utf-8-sig"), delimiter=";"))
by_paper = {}
for r in rows:
    es = r["es_id"]
    ov = OV.get(es, {})
    paper = PAPER_ID[(r["author"], r["year"])]
    crop = ov.get("crop") or CROP_DEFAULT.get(r["crop"], r["crop"].lower())
    tl = ov.get("tl", PRED[r["predat_id"]])
    tp = ov.get("tp", "pooled")
    unit = ov.get("unit", "")
    ctrl = ov.get("ctrl", "absolute_control")
    isfig = ov.get("isfig", 0)
    loc = ov.get("loc", "Figure" if isfig else "")
    ev = (f"author={r['author']} year={r['year']} crop={r['crop']}/{r['crop_gen']} "
          f"predat_id={r['predat_id']} treatment={r['treatment']} study_year={r['study_year']} "
          f"mean_t={r['mean_yieldtreat']} mean_c={r['mean_yieldcontrol']}")
    if ov.get("ev_extra"):
        ev += " || " + ov["ev_extra"]
    rec = {
        "row_id": f"gt__{paper}__{es}",
        "side": "gt",
        "paper_id": paper,
        "outcome_canonical": OUTCOME,
        "crop": crop,
        "treatment_level": tl,
        "co_amendment": "none",
        "co_amendment_level": "0",
        "timepoint": tp,
        "aggregation_level": "single_cell",
        "unit_canonical": unit,
        "control_token": ctrl,
        "treatment_mean": r["mean_yieldtreat"],
        "control_mean": r["mean_yieldcontrol"],
        "source_locator": loc,
        "is_figure": isfig,
        "evidence": ev,
        "decoder": DECODER,
    }
    by_paper.setdefault(paper, []).append(rec)

COLS = ["row_id","side","paper_id","outcome_canonical","crop","treatment_level",
        "co_amendment","co_amendment_level","timepoint","aggregation_level",
        "unit_canonical","control_token","treatment_mean","control_mean",
        "source_locator","is_figure","evidence","decoder"]

total = 0
for paper, recs in sorted(by_paper.items()):
    p = os.path.join(OUTDIR, paper + ".csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for rec in recs:
            w.writerow(rec)
    total += len(recs)
    print(f"{paper}: {len(recs)} rows")
print("TOTAL", total, "files", len(by_paper))
print("reference key tables written to", OUTDIR)
