#!/usr/bin/env python
"""
Build the Hui et al. 2025 published-reference key tables (2_conversion/keys/hui_v4/gt) from the published Supplementary Data workbook.

PROVENANCE LABEL: a reconstruction written in November 2026 that reproduces the deposited tables.
  Step 1 (workbook -> combined table) is a November 2026 reconstruction: the original script that wrote
  the combined table hui2023_gt_combined.csv (18 Mar 2026) was not found. It was written from the workbook headers and
  reproduces that CSV exactly in all numeric and identifier columns (1593 rows); the only difference is that
  102 'publication' strings keep trailing spaces that the March file does not have. That difference does not
  reach the key tables.
  Step 2 (combined table -> key tables) is the reference-side part of
  the original script build_hui_v4.py (31 May 2026), copied verbatim (functions deaccent, strip_num_prefix,
  canon_from_ai_filename, derive_paper_id, num and the GT-writing loop). The AI-key copying and blinded-view
  parts of that script are omitted because they do not affect the reference tables.
  The original builder decides WHICH publications get a key table from the AI-side key file NAMES
  (paper tokens such as '40_Bharti_2013.csv'); no AI-side value is read. To keep this script self-contained
  the 34 file names are embedded below (AI_KEY_FILENAMES); --ai-keys-dir reproduces the original behaviour.
Input : 0_sources/raw/hui/41467_2025_58397_MOESM5_ESM.xlsx (not redistributed; see
        0_sources/raw/hui/DOWNLOAD.txt), the published workbook (sheets 'Data 2 Soil  application', 'Data 3 Foliar application',
        'Data 4 Soil+Foliar application')
Output: one CSV per paper_id, 546 rows across 26 files; verified identical on every column to the deposited tables.
Dependencies: Python 3 standard library + openpyxl + pandas.
Usage : python build_reference_keys_hui.py [--raw FILE] [--out FOLDER]
        Defaults: --raw 0_sources/raw/hui/41467_2025_58397_MOESM5_ESM.xlsx; --out a new temporary folder (printed).
        The script refuses to write into 2_conversion/keys/.
"""
import argparse, csv, io, os, re, sys, tempfile, unicodedata
import pathlib
from pathlib import Path
import openpyxl
import pandas as pd

# ---- AI-side corpus file names (from the original AI key folder, listing only; contents never read) ----
AI_KEY_FILENAMES = [
    "11_Zhao_2020.csv", "14_37_Liu_2019.csv", "18_Zulfiqar_2020.csv", "21_Wang_2012.csv", "27_Zou_2012.csv",
    "38_Yilmaz_1997.csv", "3_Zhang_2012.csv", "40_Bharti_2013.csv", "42_Curtin_2008.csv", "44_Cakmak_1997.csv",
    "45_Chattha_2017.csv", "46_Ghasal_2017.csv", "49_Dawar_2022.csv", "50_Erdal_2002.csv", "52_Forster_2018.csv",
    "53_Grant_1998.csv", "58_Kalayci_1999.csv", "59_Khoshgoftarmanesh_2013.csv", "5_Yang_2011.csv",
    "61_Kumar_2018.csv", "62_Morshedi_2012.csv", "63_Mosavian_2021.csv", "65_Oliver_1994.csv",
    "66_PahlavanRad_2009.csv", "68_Peck_2008.csv", "69_Ramzan_2020.csv", "70_Rehman_2018.csv",
    "82_Torun_2001.csv", "84_Yilmaz_1998.csv", "Dong_2018.csv", "Li_2013.csv", "Liu_2014.csv",
    "Rashid_2019.csv", "Zhang_2017.csv",
]

# =====================================================================================
# Step 1 (reconstruction, Nov 2026): workbook -> combined table (same columns as hui2023_gt_combined.csv)
# =====================================================================================
SHEETS = {"Data 2 Soil  application": "Soil",
          "Data 3 Foliar application": "Foliar",
          "Data 4 Soil+Foliar application": "Soil+Foliar"}


def combined_rows(xlsx):
    """Rows of the combined table, as CSV text read back with csv.DictReader (the original builder read a CSV).
    pandas writes the table, as the March file evidently was: the 'n' column, which has 14 blanks, is rendered
    as float ('3.0'), and that string is quoted in the key-table 'evidence' field."""
    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    recs = []
    for sheet, app in SHEETS.items():
        rows = list(wb[sheet].iter_rows(values_only=True))
        h = [str(x).strip() if x is not None else "" for x in rows[1]]
        g = h.index("Grain Zn concentration (mg kg-1)")
        nrep = h.index("Numbers of replicates (n)")
        for r in rows[3:]:
            if r[0] is None:
                continue
            c, t = r[g], r[g + 1]
            if c is None or t is None:      # keep only observations with both grain-Zn means
                continue
            recs.append(dict(obs_id=r[0], study_id=r[1], publication=r[2], country=r[3], app_type=app,
                             n=r[nrep], zn_ctrl=c, zn_treat=t, zn_effect_rr=r[g + 2]))
    text = pd.DataFrame(recs).to_csv(index=False)
    return list(csv.DictReader(io.StringIO(text)))

# =====================================================================================
# Step 2 (original build_hui_v4.py, reference side, verbatim logic)
# =====================================================================================
EXCLUDE = {"zhao_2020"}  # mislabeled AI PDF (Mirbolook), see CORPUS_ERRORS.md

CANON_COLS = [
    "row_id", "side", "paper_id", "outcome_canonical", "crop", "treatment_level",
    "co_amendment", "co_amendment_level", "timepoint", "aggregation_level",
    "unit_canonical", "control_token", "treatment_mean", "control_mean",
    "source_locator", "is_figure", "evidence", "decoder",
]


def deaccent(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def strip_num_prefix(stem):
    """'11_Zhao_2020' -> 'Zhao_2020'; '14_37_Liu_2019' -> 'Liu_2019'."""
    return re.sub(r"^[0-9]+(_[0-9]+)*_", "", stem)


def canon_from_ai_filename(fn):
    return strip_num_prefix(fn[:-4]).lower()


def derive_paper_id(publication):
    """paper_id = <first-author-surname>_<year>, lowercased, deaccented, no punctuation."""
    pub = deaccent(publication.strip())
    surname = pub.split(",")[0].strip()
    surname = re.sub(r"[^A-Za-z]", "", surname)
    m = re.search(r"(?:19|20)\d{2}", pub)
    year = m.group(0) if m else "NA"
    return f"{surname.lower()}_{year}"


def num(x):
    try:
        return float(str(x).strip())
    except Exception:
        return ""


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


def main():
    ap = argparse.ArgumentParser(description="Build Hui published-reference key tables")
    ap.add_argument("--raw", default=str(_repo_root() / "0_sources" / "raw" / "hui" / "41467_2025_58397_MOESM5_ESM.xlsx"))
    ap.add_argument("--out", default=None, help="output folder (default: new temp folder)")
    ap.add_argument("--ai-keys-dir", default=None,
                    help="optional: take the corpus paper tokens from the file names in this folder (original behaviour)")
    ap.add_argument("--write-combined", default=None, help="optional: also write the step-1 combined table here")
    a = ap.parse_args()
    if not os.path.exists(a.raw):
        sys.exit("workbook not found: %s (see 0_sources/raw/hui/DOWNLOAD.txt)" % a.raw)
    a.out = _resolve_out(a.out, "hui")

    names = sorted(os.listdir(a.ai_keys_dir)) if a.ai_keys_dir else sorted(AI_KEY_FILENAMES)
    ai_tokens = {canon_from_ai_filename(fn) for fn in names if fn.endswith(".csv")} - EXCLUDE

    rows = combined_rows(a.raw)
    if a.write_combined:
        with open(a.write_combined, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    gt_rows = {}
    for row in rows:
        pub = (row.get("publication") or "").strip()
        if not pub:
            continue
        ct = (row.get("zn_ctrl") or "").strip()
        tt = (row.get("zn_treat") or "").strip()
        if ct == "" or tt == "":
            continue
        gt_rows.setdefault(derive_paper_id(pub), []).append(row)

    os.makedirs(a.out, exist_ok=True)
    n = 0
    for canon in sorted(set(gt_rows) & ai_tokens):
        with open(os.path.join(a.out, canon + ".csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CANON_COLS)
            w.writeheader()
            for i, r in enumerate(gt_rows[canon]):
                w.writerow({
                    "row_id": f"gt__{canon}__r{i}",
                    "side": "gt",
                    "paper_id": canon,
                    "outcome_canonical": "grain_zn",
                    "crop": "wheat",
                    "treatment_level": (r.get("app_type") or "").strip(),
                    "co_amendment": "none",
                    "co_amendment_level": "0",
                    "timepoint": "",
                    "aggregation_level": "single_cell",
                    "unit_canonical": "mg/kg",
                    "control_token": "absolute_control",
                    "treatment_mean": num(r.get("zn_treat")),
                    "control_mean": num(r.get("zn_ctrl")),
                    "source_locator": "",
                    "is_figure": 0,
                    "evidence": f"hui_gt study_id={r.get('study_id')} obs_id={r.get('obs_id')} country={r.get('country')} n={r.get('n')} | pub={(r.get('publication') or '')[:120]}",
                    "decoder": "gt-rekey-publication",
                })
                n += 1
    print(f"wrote {n} reference key rows for {len(set(gt_rows) & ai_tokens)} papers to {a.out}")


if __name__ == "__main__":
    main()
