# -*- coding: utf-8 -*-
"""
Build the Li J et al. 2022 published-reference key tables (2_conversion/keys/li2022_v2/gt) from Data Sheet 2.

PROVENANCE LABEL: the original conversion script, with paths parameterised.
  Step 1 is the original script decode_gt_li2022.py (31 May 2026) verbatim except that its input path
  and output folder are now set from the command line. Step 2 is the logic of the original keys_from_jsonl.py
  (31 May 2026) verbatim, wrapped in a function. The intermediate JSONL is written to a temporary folder.
Input : 0_sources/raw/li_j/Data_Sheet_2.XLSX (publisher file; sheet "Supplementary Data 2", header on row 2)
Output: one CSV per paper_id, 1108 rows across 181 files; verified identical on every column to the deposited tables.
Dependencies: Python 3 standard library + openpyxl.
Usage : python build_reference_keys_li_j.py [--raw FILE] [--out FOLDER]
        Defaults: --raw 0_sources/raw/li_j/Data_Sheet_2.XLSX; --out a new temporary folder (printed).
        The script refuses to write into 2_conversion/keys/.
"""
# (original coding line moved to top)
"""
GROUND-TRUTH-side decoder for Li2022 (Data_Sheet_2.xlsx).
Single-sided, outcome-blind. Emits one JSONL per paper_id to jsonl/gt/.

Pinned convention (identical both sides):
  outcome_canonical = 'yield'
  crop              = lowercased crop
  treatment_level   = PBsCategory normalized (seaweed/humic/protein_hydrolysate/chitosan/silicon/plant_extract/phosphite)
  co_amendment      = method (foliar/soil/seed)
  co_amendment_level= dose (canonical numeric, 3 sig figs)
  timepoint         = 'pair' + GT pair index  (guarantees per-contrast uniqueness)
  aggregation_level = 'single_cell'
  unit_canonical    = 'unresolved'  (GT sheet declares NO unit column; values are each study's
                      native published unit. Deriving a unit from the AI side would be cross-side
                      leakage (P4 violation). Honestly flagged unresolved per protocol §5.4.)
  treatment_mean    = TreatmentFreshYield (raw, as published)
  control_mean      = CtrlFreshYield (raw, as published)
  control_token     = 'absolute_control'  (Li2022 controls are untreated baselines; recorded, NOT a key)

paper_id crosswalk: GT studies are keyed by Author/Year/crop/doi. The AI side uses
filename-derived paper_id tokens. The crosswalk below maps GT study index -> AI paper_id
token using PROVENANCE ONLY (author surname stem + year + crop + doi/title). No outcome value
was consulted. GT studies with no confident AI counterpart get a GT-native id 'gt_study<NN>'
(honest NO_MATCH attrition; never force-joined).
"""
import json, os, sys
import openpyxl

BASE = os.path.dirname(os.path.abspath(__file__))
XLSX = None  # set from --raw (was a hard-coded absolute path)
OUT  = None  # set to a temporary JSONL folder (was <script dir>/jsonl/gt)
# os.makedirs(OUT) moved to the command-line block

# GT study index -> AI paper_id token (provenance-only crosswalk).
# Only confident matches (author stem + year + crop + treatment + doi/title) are listed.
STUDY_TO_AI = {
    37:  '006_Alabdulla_2019_Effect of foliar application of humic ac',
    24:  '009_Ali_2019_Biostimulatory activities of Ascophyllum nodosum e',  # Ali O. 2020 tomato A.nodosum (also AI token 'ali')
    170: '027_Chen_2021_Effects of Seaweed Extracts on the Growt',
    68:  '029_Ciepiela_2019_The effect of biostimulants derived from',
    180: '058_Fichhof_2018_Management of Biostimulant and Silicon i',
    78:  '062_Głosek-Sobieraj_2018_The Effect of Growth Regulators and a Bi',
    96:  '067_Grabowska_2012_The Effect of Cultivar and Biostimulant',
    10:  '086_Knapowski_2019_Crop stimulants as a factor determining',  # GT 'Kpowski T 2019 wheat' = Knapowski
    91:  '091_Kocira_2018_Modeling biometric traits, yield and nut',     # adjusted below
    75:  '091_Kocira_2018_Modeling biometric traits, yield and nut',     # Kocira 2018 soybean Fylloton (fpls.2018.00389)
    177: '094_Kowalska_2021_Effect of Different Forms of Silicon on',
    117: '095_Kuisma_1989_The effect of foliar application of seaw',
    56:  '100_Lola-Luz_2014_Effect on yield, total phenolic, total f',
    74:  '105_Mattner_2018_Increased growth response of strawberry',
    73:  '106_Matysiak_2018_Herbicides with natural and synthetic bi',
    82:  '110_Michalak_2016_Evaluation of supercritical extracts of',
    33:  '121_Popescu_2018_Yield, berry quality and physiological r',
    147: '124_Pramanick_2016_Effect of seaweed saps derived from two',
    85:  '125_Procházka_2015_The possibilities of increasing the prod',
    118: '126_Prokkola_2007_Effect of biological sprays on the incid',
    71:  '131_Rahman_2018_Chitosan biopolymer promotes yield and s',
    181: '153_Soppelsa_2018_Use of Biostimulants for Organic Apple P',
    141: '158_Sulakhudin_2019_Application of Coastal Sediments and Fol',
    136: '175_Wilczewski_2018_Response of sugar beet to humic substanc',
    5:   '120_Pohl_2019_The Eggplant Yield and Fruit Composition',
    19:  '1-s2.0-S0304423819306703-main',   # Boselli 2019 grape Corvina soy PH (jscienta2019108784)
    174: '1-s2.0-S0304423820302417-main',   # Bavaresco 2020 grape Trainer PH (scienta.2020.109413)
    89:  'sustainability-11-02171',         # Arif 2019 cotton Moringa (su11072171) exact doi
    149: 'agriculture-10-00618-v2',         # Almadi 2020 olive PH (agriculture10120618) exact doi
    102: 'article1400838000_Azarpour et al',# Azarpour 2012 eggplant humic
    152: 'plants-09-01633',                 # Consentino 2020 celery (plants9121633) exact doi
    129: 'al-tawaha-et-al-2011-foliar-application-of-elicitors-alters-isoflavone-concentrations-and-other-seed-characteristics-of',  # Al-Tawaha soybean chitosan
}
# Kocira soybean amino-acid (088) and seaweed bean (090) need disambiguation among GT 8/9/20/165/75.
# 088 Kocira 2019 soybean amino-acid 'Terra Sorb' single/double spray -> GT 8 (Kocira S 2019 soybean, agronomy9020093).
# 090 Kocira 2020 bean Kelpak seaweed -> GT 20 (Kocira Slawomir 2020 bean, s41598-020-74959-1) or 165. Use 20.
STUDY_TO_AI[8]  = '088_Kocira_2019_Effect of amino acid biostimulant on the'
STUDY_TO_AI[20] = '090_Kocira_2020_Biochemical and economical effect of app'
# Remove the duplicated/incorrect 91 mapping (91 = Ciepiela grass 2013, NOT Kocira). Keep 75 -> 091_Kocira_2018.
del STUDY_TO_AI[91]

PBS_NORMALIZE = {
    'SWE': 'seaweed',
    'HFA': 'humic',
    'PHs': 'protein_hydrolysate',
    'Chi': 'chitosan',
    'Si':  'silicon',
    'PE':  'plant_extract',
    'Phi': 'phosphite',
}

def norm_method(m):
    if m is None: return 'unknown'
    m = str(m).strip().lower()
    if m.startswith('foliar'): return 'foliar'
    if m.startswith('soil'):   return 'soil'
    if m.startswith('seed'):   return 'seed'
    return m

def sig3(x):
    if x is None: return '0'
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if f == 0: return '0'
    from decimal import Decimal
    # round to 3 significant figures, plain decimal string
    import math
    d = 3 - int(math.floor(math.log10(abs(f)))) - 1
    r = round(f, d)
    s = ('%f' % r).rstrip('0').rstrip('.')
    return s if s else '0'

def main():
    wb = openpyxl.load_workbook(XLSX, data_only=True, read_only=True)
    ws = wb['Supplementary Data 2']
    # column indices (0-based) per header row 2
    C = dict(pair=0, study=1, author=2, year=3, doi=4, crop=5,
             pbscat=27, method=29, dose=30, freq=32,
             ctrl=33, ctrlsd=34, trt=35, trtsd=36, cont=37, replicates=14)

    buckets = {}  # paper_id -> list[record]
    per_study_seq = {}
    nrows = 0
    for r in ws.iter_rows(min_row=3, values_only=True):
        if r[C['study']] is None and r[C['author']] is None:
            continue
        nrows += 1
        study = r[C['study']]
        pair  = r[C['pair']]
        author = r[C['author']]
        year = r[C['year']]
        crop = r[C['crop']]
        pbscat = r[C['pbscat']]
        method = r[C['method']]
        dose = r[C['dose']]
        freq = r[C['freq']]
        ctrl = r[C['ctrl']]
        trt  = r[C['trt']]
        ctrlsd = r[C['ctrlsd']]
        trtsd  = r[C['trtsd']]
        cont = r[C['cont']]
        repl = r[C['replicates']]

        paper_id = STUDY_TO_AI.get(study, 'gt_study%02d' % study) if isinstance(study, int) else 'gt_study_%s' % study

        crop_l = (str(crop).strip().lower()) if crop is not None else ''
        tl = PBS_NORMALIZE.get(str(pbscat).strip() if pbscat else '', (str(pbscat).strip().lower() if pbscat else 'unknown'))
        co = norm_method(method)
        co_level = sig3(dose)
        # CONVENTION FIX (triage 2026-05-31): the original decode put a per-row
        # uniqueness token ('pair'+index) into the KEY field `timepoint`. Because
        # timepoint is part of the match key and the AI side independently injected
        # its OWN disjoint per-row serial ('pairN'), this guaranteed 0 matches by
        # construction. Li2022 yield is a single-harvest endpoint with no real time
        # axis, so the canonical-schema convention is `pooled` (see CANONICAL_SCHEMA.md
        # timepoint rule). The pair index is preserved in row_id for audit; it is NOT
        # a match key. This removes the mechanical blocker without consulting any value.
        tp = 'pooled'

        seq = per_study_seq.get(study, 0)
        per_study_seq[study] = seq + 1
        row_id = 'gt__study%s__pair%s' % (study, pair)

        rec = {
            'row_id': row_id,
            'side': 'gt',
            'paper_id': paper_id,
            'outcome_canonical': 'yield',
            'crop': crop_l,
            'treatment_level': tl,
            'co_amendment': co,
            'co_amendment_level': co_level,
            'timepoint': tp,
            'aggregation_level': 'single_cell',
            'unit_canonical': 'unresolved',
            'control_token': 'absolute_control',
            'treatment_mean': (float(trt) if isinstance(trt, (int, float)) else None),
            'control_mean':   (float(ctrl) if isinstance(ctrl, (int, float)) else None),
            'source_locator': 'Supplementary Data 2 row pair=%s study=%s' % (pair, study),
            'is_figure': 0,
            'evidence': ("study=%s; author='%s' %s; crop='%s'; PBsCategory='%s'->%s; method='%s'->%s; "
                         "dose=%s; Frequency=%s; ContinusStudy=%s; CtrlFreshYield=%s; TreatmentFreshYield=%s; "
                         "unit=undeclared_in_GT_sheet->unresolved (no cross-side derivation, P4); "
                         "paper_id=%s" % (
                             study, author, year, crop, pbscat, tl, method, co, dose, freq, cont,
                             ctrl, trt,
                             ('matched_AI_token' if study in STUDY_TO_AI else 'GT_native_no_AI_counterpart'))),
            'decoder': 'claude-opus-4-8/decode_li2022_gt_v2',
            '_aux': {
                'gt_study': study,
                'gt_pair': pair,
                'gt_author': author,
                'gt_year': year,
                'gt_doi': r[C['doi']],
                'pbscategory_raw': pbscat,
                'method_raw': method,
                'dose_raw': dose,
                'frequency_raw': freq,
                'continus_study': cont,
                'replicates': repl,
                'control_sd': (float(ctrlsd) if isinstance(ctrlsd, (int, float)) else None),
                'treatment_sd': (float(trtsd) if isinstance(trtsd, (int, float)) else None),
                'paper_id_basis': ('AI_crosswalk' if study in STUDY_TO_AI else 'gt_native'),
            },
        }
        buckets.setdefault(paper_id, []).append(rec)

    # write one JSONL per paper_id
    total = 0
    matched_papers = 0
    for pid, recs in buckets.items():
        safe = pid.replace('/', '_').replace('\\', '_')
        if len(safe) > 80:
            import hashlib
            safe = safe[:60] + '_' + hashlib.md5(pid.encode('utf-8')).hexdigest()[:8]
        fp = os.path.join(OUT, safe + '.jsonl')
        with open(fp, 'w', encoding='utf-8') as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        total += len(recs)
        if not pid.startswith('gt_study'):
            matched_papers += 1

    summary = ("GT decode Li2022: rows=%d; papers=%d (AI-matched=%d, gt_native=%d); jsonl_total=%d"
               % (nrows, len(buckets), matched_papers, len(buckets) - matched_papers, total))
    with open(os.path.join(OUT, '_DECODE_SUMMARY.txt'), 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
        for pid in sorted(buckets):
            f.write('%-95s rows=%d\n' % (pid, len(buckets[pid])))
    print(summary)



# ---------------------------------------------------------------------------
# Step 2 (original: keys_from_jsonl.py, logic unchanged): JSONL -> canonical CSV
# ---------------------------------------------------------------------------
import csv as _csv
_COLS = ["row_id","side","paper_id","outcome_canonical","crop","treatment_level",
         "co_amendment","co_amendment_level","timepoint","aggregation_level",
         "unit_canonical","control_token","treatment_mean","control_mean",
         "source_locator","is_figure","evidence","decoder"]

def keys_from_jsonl(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    total_w = total_d = 0
    for fn in sorted(os.listdir(indir)):
        if not fn.lower().endswith(".jsonl"):
            continue
        rows, dropped = [], 0
        with open(os.path.join(indir, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    rows.append({c: o.get(c, "") for c in _COLS})
                except Exception:
                    dropped += 1
        out = os.path.join(outdir, os.path.splitext(fn)[0] + ".csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=_COLS)
            w.writeheader(); w.writerows(rows)
        total_w += len(rows); total_d += dropped
    print(f"keys_from_jsonl: wrote {total_w} rows total, dropped {total_d} unparseable.")


# ---------------------------------------------------------------------------
# Command line (added November 2026): paths are arguments, intermediate JSONL goes to a temp dir
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse, tempfile, pathlib, sys
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--raw", default=str(_repo_root() / "0_sources" / "raw" / "li_j" / "Data_Sheet_2.XLSX"), help="published raw file")
    ap.add_argument("--out", default=None, help="output folder for per-paper key CSVs (default: new temp folder)")
    a = ap.parse_args()
    a.out = _resolve_out(a.out, "li_j")
    with tempfile.TemporaryDirectory() as _tmp:
        XLSX = a.raw; OUT = os.path.join(_tmp, 'jsonl_gt'); os.makedirs(OUT, exist_ok=True)
        main()
        keys_from_jsonl(OUT, a.out)
    print("reference key tables written to", a.out)
