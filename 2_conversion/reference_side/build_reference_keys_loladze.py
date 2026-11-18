#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build the Loladze 2014 published-reference key tables (2_conversion/keys/loladze_v2/gt) from the published CO2 dataset.

PROVENANCE LABEL: the original conversion script, with paths parameterised.
  Step 1 is the original script decode_gt.py (31 May 2026) verbatim except that its two hard-coded
  path constants are now set from the command line. Step 2 is the logic of the original keys_from_jsonl.py
  (31 May 2026) verbatim, wrapped in a function. The intermediate JSONL is written to a temporary folder.
Input : 0_sources/raw/loladze/CO2 Dataset.xlsx (Dryad; sheet "CO2 Dataset", header on row 2)
Output: one CSV per paper_id, 754 rows across 50 files; verified identical on every column to the deposited tables.
Dependencies: Python 3 standard library + openpyxl.
Usage : python build_reference_keys_loladze.py [--raw FILE] [--out FOLDER]
        Defaults: --raw 0_sources/raw/loladze/CO2 Dataset.xlsx; --out a new temporary folder (printed).
        The script refuses to write into 2_conversion/keys/.
"""
# (original shebang moved to top)
# (original coding line moved to top)
"""
Decode the GROUND-TRUTH side for Loladze CO2 dataset into the canonical
decoded-key schema (one JSONL object per row), per UNBIASED_MATCHING_PROTOCOL
and the pinned CO2 convention.

OUTCOME-BLIND for pairing: coordinates derived from GT's OWN descriptor columns
+ this row's own recorded effect value only. control_token recorded, not a key.

Pinned CO2 convention (identical both sides):
  outcome_canonical = mineral_concentration
  crop              = na
  treatment_level   = element symbol (lowercase)        [blank if no element symbol]
  co_amendment      = tissue mapped to canonical bucket
  co_amendment_level= species_or_cultivar (registry-normalized; blank if single
                      species with no cultivar -> not a distinguishing coordinate)
  timepoint         = 'eco2_<e>_amb_<a>' from eCO2/aCO2 ; 'pooled' if Additional
                      Info documents pooling over time/treatments
  aggregation_level = 'pooled' if pooled else 'single_cell'
  unit_canonical    = 'ratio'
  treatment_mean    = (E-A)/A  (fractional ratio)
  control_mean      = ''
  control_token     = 'ambient_co2'
"""
import openpyxl, json, re, os, collections

GT_PATH = None  # set from --raw (was a hard-coded absolute path)
OUT_DIR = None  # set to a temporary JSONL folder (was a hard-coded absolute path)

# ---------------------------------------------------------------------------
# Reference -> AI paper_id token mapping.
# Mapping is by author-surname + year as encoded in the AI filename tokens.
# The 5 documented mislabels (per project memory) are handled by mapping the
# GT reference whose CONTENT the AI file actually carries, where the AI token
# name conflicts. This is provenance/label mapping only -- no outcome values
# consulted. References with no AI counterpart are skipped (reported as attrition).
# ---------------------------------------------------------------------------
REF_TO_PAPER = {
    # straightforward author+year matches
    'Al-Rawahy et al 2013': '018_Al-Rawahy_2013',
    'Azam et al 2013': '006_Azam_2013',
    'Barnes & Pffirrman 1992': '009_Barnes_1992',
    'Baslam et al 2012': '003_Baslam_2012',
    'Baxter et al 1994': '019_Baxter_1994',
    'Blank et al 2011': '022_Blank_2011',
    'Cambell & Sage 2002': '008_Campbell_2002',
    'Fangmeier et al  2002': '017_Fangmeier_2002',
    'Finzi et al 2001': '004_Finzi_2001',
    'Guo et al 2013': '025_Guo_2011',          # AI '025_Guo_2011' carries Guo 2013 content (Wuyunjing/Yangmai, 570/370)
    'Heagle et al 1993': '039_Heagle_1993',    # AI file empty -> will be NO_MATCH, still decode GT
    'Housman et al 2012': '044_Housman_2012',
    'Huluka et al 1994': '011_Huluka_1994',
    'Johnson et al 1997': '034_Johnson_1997',  # token name; content note below
    'Johnson et al 2003': '033_Johnson_2003',
    'Kanowski 2001': '032_Kanowski_2001',
    'Keutgen & Chen 2001': '013_Keutgen_2001',
    'Khan et al 2012': '048_Khan_2013',        # AI '048_Khan_2013' carries Khan tomato Astra/Eureka 1000/400 content
    'Kuehny et al 1991': '029_Kuehny_1991',
    'Lieffering et al 2004': '014_Lieffering_2004',
    'Luomala et al 2005': '042_Luomala_2005',
    'Mishra et al 2011': '028_Mishra_2011',
    'Mjwara et al 1996': '041_Mjwara_1996',
    'Natali et al 2009': '043_Natali_2009',
    'Newbery 1995': '038_Newbery_1995',
    'Niinemets et al 1999': '005_Niinemets_1999',
    'Niu et al 2013': '051_Niu_2013',
    'Nowak et al 2002': '024_Nowak_2002',
    "O'Neill et al 1987": '058_ONeill_1987',
    'Oksanen et al 2005': '035_Oksanen_2005',
    'Overdieck 1993': '020_Overdieck_1993',
    'Peet et al 1986': '027_Peet_1986',
    'Pfirrmann et al 1996': '040_Pfirrmann_1996',
    'Pleijel & Danielsson 2009': '015_Pleijel_2009',
    'Polley et al 2011': '050_Polley_2011',
    'Porter & Grodzinski 1984': '046_Porter_1984',
    'Rodenkirchen et al 2009': '047_Rodenkirchen_2009',
    'Schenk et al 1997': '036_Schenk_1997',
    'Seneweera & Conroy 1997': '026_Seneweera_1997',
    'Singh et al 2013': '049_Singh_2013',
    'Wilsey et al 1994': '021_Wilsey_1994',
    'Woodin et al 1992': '007_Woodin_1992',
    'Wroblewitz et al 2013': '030_Wroblewitz_2013',
    'Wu et al 2004': '012_Wu_2004',
    'Ziska et al 1997': '002_Ziska_1997',

    # documented mislabels (AI filename author/year != PDF content):
    # AI '001_Ma_2007.pdf' actually = Fernando 2012 grain minerals (Yitpi; Ca,Fe,S,Zn,N).
    #   GT 'Fernando et al 2012a' (Yitpi grain Ca,Fe,S,Zn) + 'Fernando et al 2012c' (Yitpi grain N).
    'Fernando et al 2012a': '001_Ma_2007',
    'Fernando et al 2012c': '001_Ma_2007',
    # AI '016_Fernando_2012a.pdf' carries non-mineral flour/bread variables (Yitpi) -> Fernando 2012b set.
    'Fernando et al 2012b': '016_Fernando_2012a',
    # AI '010_Li_2010.pdf' actually = Hogy et al 2009 (file empty in AI run); decode GT Hogy 2009.
    'Hogy et al 2009': '010_Li_2010',
    # AI '031_Pal_2003.pdf' actually = Pal et al 2004 (Pusa Jayant clover).
    'Pal et al 2004': '031_Pal_2003',
    # AI '037_de_2000.pdf' actually = Haase et al 2008 (barley shoot Fe).
    'Haase et al 2008': '037_de_2000',
}

# Tissue -> canonical co_amendment bucket (matches AI side conventions observed:
# grain, foliar, above_ground, seed, tuber, fruit, shoot, root, needles->foliar?).
# AI used: grain, foliar, above_ground, seed, tuber, fruit, shoot(->foliar in 037? no, 'shoot'),
# Observed AI buckets: grain, foliar, above_ground, seed, tuber, fruit, shoot, root, leaves->foliar.
# GT 'needles' AI mapped to 'foliar' (Pfirrmann needles, Luomala needles) -- verify: AI Pfirrmann tissue?
# Canonical co_amendment closed set observed on AI side:
#   {grain, foliar, above_ground, edible, root, litter}
# Map GT tissue terms to that set (mirrors AI intent: seed->grain, tuber/fruit->edible,
# needles/leaves/blades/frond/seedlings->foliar, shoots/tillers/stems/stover->above_ground).
TISSUE_MAP = {
    'above ground': 'above_ground',
    'foliar': 'foliar',
    'leaves': 'foliar',
    'blades': 'foliar',
    'needles': 'foliar',
    'seedlings': 'foliar',
    'frond': 'foliar',
    'shoots': 'above_ground',
    'shoot': 'above_ground',
    'stems': 'above_ground',
    'tillers': 'above_ground',
    'stover': 'above_ground',
    'seed': 'grain',
    'grain': 'grain',
    'tuber': 'edible',
    'fruit': 'edible',
}

VALID_ELEMENTS = {'ca','k','fe','zn','n','p','mg','mn','cu','c','b','s','si','mo','al','ba',
                  'na','cd','ni','pb','se','cr','br','co','v','sr','cl'}

def norm_token(s):
    """registry-normalize a species/cultivar string to lowercase snake_case."""
    s = s.strip().strip("'").strip()
    s = s.lower()
    s = s.replace("'", " ").replace('"', ' ')
    s = re.sub(r'[^a-z0-9]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    return s

def extract_cultivar(species_raw):
    """Return cultivar token if a quoted cultivar present in the Species string, else None."""
    m = re.search(r"'([^']+)'", species_raw)
    if m:
        return norm_token(m.group(1))
    return None

def species_binomial(species_raw):
    """genus_species token (strip quoted cultivar, strip trailing taxonomic authorities)."""
    s = re.sub(r"'[^']*'", '', species_raw).strip()
    return norm_token(s)


# Pooling markers (whole-note): documented aggregation over time/treatments/levels.
POOL_RE = re.compile(
    r'\b(avg|average|mean[s]?|pooled)\b', re.IGNORECASE)


def parse_additional_info(note):
    """
    Parse the GT 'Additional Info' free-text into distinguishing coordinates.
    Returns (pooled: bool, tp_suffixes: list[str], level_suffixes: list[str]).
    Outcome-blind: uses only the descriptor text.

    Routing:
      time-like (year, season, DOY, harvest-year, 'last (Nth) yr') -> timepoint suffix
      factor-like (fert level, soil, ion source, +/-K, location, O3/NF, cultivar
                   sub-line, leaf age) -> co_amendment_level suffix
    'pooled' whole-note markers (avg/mean over ...) -> pooled.
    """
    note = (note or '').strip()
    if not note:
        return (False, [], [])

    nl = note.lower()

    # Whole-note documented pooling: 'avg over 6 P treatments', 'mean 2008 & 2009',
    # 'avg of three B treatments', 'avg of 10 treatments', 'avg all cohrts...',
    # 'avg of rainfalls', 'avg 3 over soils', 'avg new & old, 1998', '2 clones'.
    pooled_phrases = [
        'avg over', 'avg of', 'average over', 'average of', 'avg all', 'avg 3 over',
        'mean of', 'mean over', 'means over', 'pooled', 'avg new', 'avg.', 'over soils',
        'over years', 'over all',
    ]
    if any(p in nl for p in pooled_phrases):
        return (True, [], [])
    # 'mean <yyyy> & <yyyy>' or 'mean <yyyy>-<yyyy>'  /  'mean 2007-2009'
    if re.search(r'mean\s*\d{4}\s*[&\-,]', nl) or re.search(r'\bmean\b.*\d{4}', nl):
        return (True, [], [])
    # explicit pooling over clones / cohorts
    if re.match(r'^\d+\s+clones$', nl):  # '2 clones'
        return (True, [], [])

    tp_suf = []
    lvl_suf = []
    # split combined notes on commas
    parts = [p.strip() for p in re.split(r',', note) if p.strip()]
    for p in parts:
        pl = p.lower()

        # ---- TIME-LIKE -> timepoint ----
        m = re.fullmatch(r'(19|20)\d{2}', p.strip())               # bare year e.g. 1992
        if m:
            tp_suf.append('y' + p.strip()); continue
        if re.search(r'\bharvest\b', pl):                          # 'Cd=0, 2007 harvest'
            yh = re.search(r'(\d{4})\s*harvest', pl)
            if yh:
                tp_suf.append('h' + yh.group(1))
            # any 'X=0' background goes to level
            mb = re.match(r'([a-z]{1,2})\s*=\s*0', pl)
            if mb:
                lvl_suf.append(mb.group(1) + '0')
            continue
        if pl in ('spring', 'winter', 'summer', 'autumn', 'fall'):
            tp_suf.append(pl); continue
        md = re.search(r'doy\s*(\d+)', pl)                         # 'DOY 247'
        if md:
            tp_suf.append('doy' + md.group(1)); continue
        if 'last' in pl and 'yr' in pl:                            # 'last (3rd) yr'
            mo = re.search(r'\((\d+)', pl)
            tp_suf.append('y' + (mo.group(1) if mo else 'last'));
            if 'unfertil' in pl:
                lvl_suf.append('unfertilized')
            continue
        m2 = re.fullmatch(r'(\d{4})-(\d{2,4})', p.strip())         # '2009-10', '2004-2006'
        if m2:
            tp_suf.append('y' + m2.group(1) + '_' + m2.group(2)); continue
        m3 = re.fullmatch(r'(\d{4,5})-(\d{2})', p.strip())         # '20010-11' (typo)
        if m3:
            tp_suf.append('y' + m3.group(1) + '_' + m3.group(2)); continue

        # ---- FACTOR-LIKE -> co_amendment_level ----
        # P / N fertilizer dose:  'P 60 mg/kg', 'N100', 'N50'
        mp = re.match(r'p\s*(\d+)\s*mg', pl)
        if mp:
            lvl_suf.append('p' + mp.group(1)); continue
        mn = re.match(r'n\s*(\d+)$', pl) or re.match(r'n(\d+)$', pl)
        if mn:
            lvl_suf.append('n' + mn.group(1)); continue
        if pl in ('high n', 'low n', 'high p', 'low p'):
            lvl_suf.append(pl.replace(' ', '')); continue
        if pl in ('basalt', 'rhyolite'):
            lvl_suf.append(pl); continue
        if pl in ('nh4', 'no3'):
            lvl_suf.append(pl); continue
        if pl in ('duke', 'ornl'):    # distinct experimental SITES (Natali FACE sites)
            lvl_suf.append(pl); continue
        if re.match(r'[+\-]\s*k\b', pl):                           # '+K, 1988 needles' / '-K ...'
            lvl_suf.append(('kplus' if p.strip().startswith('+') else 'kminus'))
            yk = re.search(r'(\d{4})', pl)
            if yk:
                tp_suf.append('y' + yk.group(1))
            continue
        if pl in ('nc-r', 'nc-s', 'ncr', 'ncs'):
            lvl_suf.append(pl.replace('-', '')); continue
        if 'unfertil' in pl:
            lvl_suf.append('unfertilized'); continue
        if 'old leaves' in pl or pl == 'old leaves':
            lvl_suf.append('old_leaves'); continue
        if 'inner nm' in pl:
            lvl_suf.append('inner_nm'); continue
        if '1 yr old' in pl or 'yr old' in pl:
            lvl_suf.append('1yr_old'); continue
        if 'needles' in pl:
            # current/old needle descriptor remaining after +/-K handled
            lvl_suf.append('needles'); continue
        if 'o3' in pl or 'ozone' in pl:
            lvl_suf.append('o3'); continue
        if pl.endswith('nf') or ' nf' in pl or pl == 'nf':
            # 'GI, SB, TE loc., NF' -> non-filtered ozone; location set + NF
            lvl_suf.append('nf'); continue
        # Pure facility / chamber-replication / site-code / sampling-design descriptors
        # are NOT biological treatment factors. They are dropped (left blank): they do
        # not recover a value-distinguishing coordinate and the AI side likewise does not
        # encode them, so forcing them into the key would create spurious non-matches.
        # Examples dropped: '8+8 OTC', '8+8 OTC SERC', 'GI, SB, TE loc.', 'JO,GI,GO,SB,...',
        # 'mini-FACE', bare replication counts. Genuine factors are handled above.
        DESIGN_NOISE_HINTS = ('otc', 'serc', 'loc', 'face', 'chamber', 'rep', 'plot', 'block',
                              'gi', 'sb', 'te', 'jo', 'go')
        tok = norm_token(p)
        toks = set(tok.split('_'))
        if toks & set(DESIGN_NOISE_HINTS) or re.fullmatch(r'[\d_]+', tok or 'x'):
            continue  # drop design noise
        # anything else genuinely unrecognized: keep normalized token (audit-visible)
        if tok:
            lvl_suf.append(tok)

    return (False, tp_suf, lvl_suf)


def main():
    wb = openpyxl.load_workbook(GT_PATH, data_only=True)
    ws = wb["CO2 Dataset"]
    hdr = [ws.cell(row=2, column=c).value for c in range(1, ws.max_column + 1)]
    idx = {h: i for i, h in enumerate(hdr)}

    rows = []
    for r in range(3, ws.max_row + 1):
        v = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
        if v[idx['Reference']] is None and v[idx['Species']] is None:
            continue
        rows.append((r, v))

    # Determine, per reference, the set of distinct species -> decide if species is
    # a distinguishing coordinate (multi-species) or not (single species).
    species_by_ref = collections.defaultdict(set)
    for r, v in rows:
        ref = (str(v[idx['Reference']]).strip() if v[idx['Reference']] is not None else None)
        if ref is None:
            continue
        sp = str(v[idx['Species']]).strip() if v[idx['Species']] is not None else ''
        # collapse cultivars of same binomial when deciding multiplicity? No:
        # AI treats distinct cultivars of same species as distinct levels (e.g. Astra/Eureka).
        species_by_ref[ref].add(sp)

    by_paper = collections.defaultdict(list)
    skipped = collections.Counter()
    per_row_counter = collections.defaultdict(int)

    for r, v in rows:
        ref_raw = v[idx['Reference']]
        if ref_raw is None:
            continue
        ref = str(ref_raw).strip()
        paper_id = REF_TO_PAPER.get(ref)
        if paper_id is None:
            skipped[ref] += 1
            continue

        species_raw = str(v[idx['Species']]).strip() if v[idx['Species']] is not None else ''
        tissue_raw = str(v[idx['Tissue']]).strip() if v[idx['Tissue']] is not None else ''
        element_raw = str(v[idx['Element']]).strip() if v[idx['Element']] is not None else ''
        eco2 = v[idx['eCO2']]
        aco2 = v[idx['aCO2']]
        ai_note = v[idx['Additional Info']]
        ai_note = '' if ai_note is None else str(ai_note).strip()
        ea = v[idx['(E-A)/A']]

        # treatment_level: element symbol lowercase if recognized, else blank
        el = element_raw.lower()
        treatment_level = el if el in VALID_ELEMENTS else ''

        # co_amendment: tissue bucket
        co_amendment = TISSUE_MAP.get(tissue_raw.lower(), norm_token(tissue_raw))

        # co_amendment_level: cultivar if present; else species binomial only if the
        # reference has >1 distinct species; else blank (single species not distinguishing).
        cultivar = extract_cultivar(species_raw)
        n_species = len(species_by_ref[ref])
        if cultivar:
            co_amendment_level = cultivar
        elif n_species > 1:
            co_amendment_level = species_binomial(species_raw)
        else:
            co_amendment_level = ''

        # timepoint + aggregation_level via Additional Info parser
        pooled, tp_suf, lvl_suf = parse_additional_info(ai_note)

        def fmt(x):
            if isinstance(x, float) and x.is_integer():
                return str(int(x))
            return str(x)

        if pooled:
            timepoint = 'pooled'
            aggregation_level = 'pooled'
        else:
            if eco2 is not None and aco2 is not None:
                base_tp = 'eco2_%s_amb_%s' % (fmt(eco2), fmt(aco2))
            else:
                base_tp = 'co2_unresolved'
            if tp_suf:
                timepoint = base_tp + ''.join('__' + s for s in tp_suf)
            else:
                timepoint = base_tp
            aggregation_level = 'single_cell'

        # append non-time distinguishing factors to co_amendment_level
        if lvl_suf:
            base_lvl = co_amendment_level
            extra = '__'.join(lvl_suf)
            co_amendment_level = (base_lvl + '__' + extra) if base_lvl else extra

        # treatment_mean = (E-A)/A fractional ratio
        if ea is None:
            tmean = ''
        else:
            try:
                tmean = '%.6f' % float(ea)
            except (TypeError, ValueError):
                tmean = ''

        per_row_counter[paper_id]
        ridx = per_row_counter[paper_id]
        per_row_counter[paper_id] += 1
        row_id = 'gt__%s__r%d' % (paper_id, r)

        # evidence (audit trail) -- descriptor provenance only
        evidence = ("Species=%r->%s | Tissue=%r->%s | Element=%r->%s | eCO2=%s aCO2=%s->%s | "
                    "AdditionalInfo=%r | (E-A)/A=%s | n_species_in_ref=%d | ref=%r"
                    % (species_raw, (co_amendment_level or '(blank)'), tissue_raw, co_amendment,
                       element_raw, (treatment_level or '(blank)'), eco2, aco2, timepoint,
                       ai_note, ea, n_species, ref))

        obj = {
            'row_id': row_id,
            'side': 'gt',
            'paper_id': paper_id,
            'outcome_canonical': 'mineral_concentration',
            'crop': 'na',
            'treatment_level': treatment_level,
            'co_amendment': co_amendment,
            'co_amendment_level': co_amendment_level,
            'timepoint': timepoint,
            'aggregation_level': aggregation_level,
            'unit_canonical': 'ratio',
            'control_token': 'ambient_co2',
            'treatment_mean': tmean,
            'control_mean': '',
            'source_locator': 'Loladze CO2 Dataset row %d' % r,
            'is_figure': 0,
            'evidence': evidence,
            'decoder': 'gt-decode-v2',
        }
        by_paper[paper_id].append(obj)

    # Annotate genuinely-undeterminable within-paper key collisions (completeness rule
    # fired but no descriptor/Additional-Info coordinate recovers the distinction, and
    # BIAS RULES forbid guessing). These remain AMBIGUOUS for the join.
    KEYF = ('treatment_level', 'co_amendment', 'co_amendment_level', 'timepoint',
            'aggregation_level', 'unit_canonical')
    n_undet = 0
    for paper_id, objs in by_paper.items():
        seen = collections.defaultdict(list)
        for o in objs:
            seen[tuple(o[k] for k in KEYF)].append(o)
        for key, grp in seen.items():
            if len(grp) > 1 and len({g['treatment_mean'] for g in grp}) > 1:
                for g in grp:
                    g['evidence'] += (
                        ' || UNDETERMINABLE_COLLISION: %d GT rows share this canonical key '
                        'with differing means; no descriptor/Additional-Info coordinate '
                        'recovers the distinction (not guessed per BIAS RULES) -> AMBIGUOUS'
                        % len(grp))
                    n_undet += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    total = 0
    for paper_id, objs in sorted(by_paper.items()):
        path = os.path.join(OUT_DIR, paper_id + '.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for o in objs:
                f.write(json.dumps(o, ensure_ascii=False) + '\n')
        total += len(objs)

    print("Wrote %d GT rows across %d papers to %s" % (total, len(by_paper), OUT_DIR))
    print("Undeterminable-collision rows flagged AMBIGUOUS: %d" % n_undet)
    print("Papers emitted:", sorted(by_paper.keys()))
    print("\nSKIPPED references (no AI counterpart) -- %d refs, %d rows:" % (len(skipped), sum(skipped.values())))
    for k, c in sorted(skipped.items()):
        print("  %-30s %d" % (k, c))




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
    ap.add_argument("--raw", default=str(_repo_root() / "0_sources" / "raw" / "loladze" / "CO2 Dataset.xlsx"), help="published raw file")
    ap.add_argument("--out", default=None, help="output folder for per-paper key CSVs (default: new temp folder)")
    a = ap.parse_args()
    a.out = _resolve_out(a.out, "loladze")
    with tempfile.TemporaryDirectory() as _tmp:
        GT_PATH = a.raw; OUT_DIR = os.path.join(_tmp, 'jsonl_gt')
        main()
        keys_from_jsonl(OUT_DIR, a.out)
    print("reference key tables written to", a.out)
