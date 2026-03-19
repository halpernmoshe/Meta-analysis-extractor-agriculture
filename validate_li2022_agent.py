"""
Validate Li 2022 biostimulant/yield AGENT extraction against ground truth.

Ground truth: Data_Sheet_2.XLSX
Agent extraction: output/li2022_agent_extraction/*_agent*.json

Usage:
    ./venv/Scripts/python.exe validate_li2022_agent.py
"""
import sys, os, json, math, re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import openpyxl

GT_PATH = r"C:\Users\moshe\Dropbox\Testing metaanalyis program\Li 2022\Data_Sheet_2.XLSX"
AGENT_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\li2022_agent_extraction")

SCALE_FACTORS = [1, 10, 100, 1000, 0.1, 0.01, 0.001, 10000, 0.0001]

YIELD_KEYWORDS = ['yield', 'fresh', 'weight', 'production', 'harvest', 'tuber', 'fruit',
                  'grain', 'seed', 'cane', 'marketable', 'total', 'biomass', 'dry matter',
                  'fw', 'dw', 'fwt', 'dwt']
EXCLUDE_KEYWORDS = ['height', 'chlorophyll', 'sugar content', 'protein content', 'starch',
                    'flavonoid', 'phenolic', 'node', 'spike', 'blight', 'severity', 'leaf area',
                    'root length', 'stem diameter', 'anthocyanin', 'carotenoid', 'vitamin',
                    'color', 'firmness', 'diameter', 'ph ', 'acidity', 'tss']


def is_yield_outcome(name):
    if not name:
        return False
    name = name.lower()
    if any(ex in name for ex in EXCLUDE_KEYWORDS):
        return False
    return any(kw in name for kw in YIELD_KEYWORDS)


def safe_float(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s.startswith('=') or not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def load_ground_truth():
    """Load Li 2022 ground truth from Excel."""
    wb = openpyxl.load_workbook(GT_PATH, read_only=True, data_only=True)
    ws = wb['Supplementary Data 2']
    rows = list(ws.iter_rows(min_row=3, values_only=True))
    wb.close()

    gt_by_study = defaultdict(list)
    for row in rows:
        if row[0] is None:
            continue
        author = str(row[2]).strip() if row[2] else ""
        year = int(row[3]) if row[3] else 0
        ctrl = safe_float(row[33])
        treat = safe_float(row[35])

        if ctrl is None or treat is None:
            continue

        gt_by_study[(author, year)].append({
            'pair': row[0],
            'study': row[1],
            'author': author,
            'year': year,
            'crop': str(row[5]) if row[5] else "",
            'product': str(row[26]) if row[26] else "",
            'n': int(row[14]) if row[14] else None,
            'ctrl_mean': ctrl,
            'treat_mean': treat,
        })

    return gt_by_study


def load_agent_results():
    """Load agent extraction JSON files."""
    papers = {}
    for f in sorted(AGENT_DIR.glob("*_agent*.json")):
        paper_id = f.stem.replace("_agent_v2", "").replace("_agent", "")
        if '_v2' in f.stem or paper_id not in papers:
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                obs_list = data.get('consensus_observations', [])
                # For Li 2022, obs use "outcome" instead of "element"
                yield_obs = []
                for obs in obs_list:
                    outcome = obs.get('outcome', obs.get('element', ''))
                    ctrl = safe_float(obs.get('control_mean'))
                    treat = safe_float(obs.get('treatment_mean'))
                    if ctrl is not None and treat is not None:
                        if is_yield_outcome(outcome) or not outcome:
                            yield_obs.append(obs)

                papers[paper_id] = {
                    'all_obs': obs_list,
                    'yield_obs': yield_obs if yield_obs else [o for o in obs_list
                                                               if safe_float(o.get('control_mean')) is not None
                                                               and safe_float(o.get('treatment_mean')) is not None],
                    'meta': data,
                    'file': str(f),
                }
            except Exception as e:
                print(f"  ERROR loading {f.name}: {e}")

    return papers


def find_best_scale(gt_ctrl, gt_treat, ext_ctrl, ext_treat):
    best_scale = None
    best_error = float('inf')
    for s in SCALE_FACTORS:
        c_err = abs(gt_ctrl * s - ext_ctrl) / max(abs(ext_ctrl), 0.001)
        t_err = abs(gt_treat * s - ext_treat) / max(abs(ext_treat), 0.001)
        err = (c_err + t_err) / 2
        if err < best_error:
            best_error = err
            best_scale = s
    return best_scale, best_error


def match_paper_to_gt(paper_id, paper_meta, gt_by_study):
    """Match agent paper to GT study by author+year."""
    # Try extracting author and year from paper_id
    match = re.match(r'(\d+)_([^_]+(?:[-][^_]+)?)_(\d{4})', paper_id)
    if match:
        author_part = match.group(2).lower().replace('-', '').replace('_', '')
        year_part = int(match.group(3))
    else:
        # Try from metadata
        authors = str(paper_meta.get('authors', '')).split(',')[0].strip()
        year = paper_meta.get('year', 0)
        if authors and year:
            author_part = authors.lower().replace(' ', '').replace('-', '')
            year_part = int(year)
        else:
            return None, None

    best_match_key = None
    best_score = 0

    for (gt_author, gt_year), gt_obs in gt_by_study.items():
        if gt_year != year_part:
            continue
        gt_norm = gt_author.lower().replace(' ', '').replace(',', '').replace('-', '').replace('.', '')

        # Check overlap
        if author_part in gt_norm or gt_norm[:6] in author_part:
            score = len(set(author_part) & set(gt_norm))
            if score > best_score:
                best_score = score
                best_match_key = (gt_author, gt_year)

    if best_match_key:
        return best_match_key, gt_by_study[best_match_key]
    return None, None


def match_observations(ext_obs, gt_obs, tolerance=0.30):
    """Match extracted obs to GT obs by value similarity with scale harmonization."""
    matches = []
    used_gt = set()

    for ext in ext_obs:
        ext_ctrl = safe_float(ext.get('control_mean'))
        ext_treat = safe_float(ext.get('treatment_mean'))
        if ext_ctrl is None or ext_treat is None:
            continue

        best_match = None
        best_err = float('inf')

        for i, gt in enumerate(gt_obs):
            if i in used_gt:
                continue

            scale, err = find_best_scale(gt['ctrl_mean'], gt['treat_mean'], ext_ctrl, ext_treat)
            if err < best_err and err < tolerance:
                best_err = err
                best_match = (i, gt, scale, err)

        if best_match:
            idx, gt, scale, err = best_match
            used_gt.add(idx)

            # Calculate effect sizes
            ext_effect = (ext_treat - ext_ctrl) / ext_ctrl * 100 if ext_ctrl != 0 else 0
            gt_effect = (gt['treat_mean'] - gt['ctrl_mean']) / gt['ctrl_mean'] * 100 if gt['ctrl_mean'] != 0 else 0

            matches.append({
                'ext_ctrl': ext_ctrl,
                'ext_treat': ext_treat,
                'gt_ctrl': gt['ctrl_mean'],
                'gt_treat': gt['treat_mean'],
                'scale': scale,
                'match_err': err,
                'ext_effect': ext_effect,
                'gt_effect': gt_effect,
                'effect_diff': abs(ext_effect - gt_effect),
            })

    return matches


def main():
    print(f"Li 2022 Biostimulant/Yield — AGENT Extraction Validation")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    print("Loading ground truth...")
    gt_by_study = load_ground_truth()
    total_gt_studies = len(gt_by_study)
    total_gt_obs = sum(len(v) for v in gt_by_study.values())
    print(f"  {total_gt_studies} studies, {total_gt_obs} observations\n")

    print("Loading agent extractions...")
    papers = load_agent_results()
    print(f"  {len(papers)} papers loaded\n")

    all_matches = []
    paper_results = []

    for paper_id, paper_data in sorted(papers.items()):
        meta = paper_data['meta']
        gt_key, gt_obs = match_paper_to_gt(paper_id, meta, gt_by_study)

        if not gt_obs:
            print(f"  {paper_id}: {len(paper_data['yield_obs'])} yield obs, NO GT MATCH")
            paper_results.append({"paper_id": paper_id, "ext_obs": len(paper_data['yield_obs']), "gt": 0, "matched": 0})
            continue

        matches = match_observations(paper_data['yield_obs'], gt_obs, tolerance=0.30)
        all_matches.extend(matches)

        n_matched = len(matches)
        n_gt = len(gt_obs)
        n_ext = len(paper_data['yield_obs'])

        if matches:
            mae = sum(m['effect_diff'] for m in matches) / len(matches)
            effects_ext = [m['ext_effect'] for m in matches]
            effects_gt = [m['gt_effect'] for m in matches]

            if len(effects_ext) >= 3:
                mean_e = sum(effects_ext) / len(effects_ext)
                mean_g = sum(effects_gt) / len(effects_gt)
                cov = sum((e - mean_e) * (g - mean_g) for e, g in zip(effects_ext, effects_gt))
                var_e = sum((e - mean_e) ** 2 for e in effects_ext)
                var_g = sum((g - mean_g) ** 2 for g in effects_gt)
                r = cov / math.sqrt(var_e * var_g) if var_e > 0 and var_g > 0 else 0
                r_str = f"r={r:.3f}"
            else:
                r_str = ""
            print(f"  {paper_id}: {n_ext} yield obs, GT={n_gt}, matched={n_matched} MAE={mae:.1f}pp {r_str}")
        else:
            mae = None
            print(f"  {paper_id}: {n_ext} yield obs, GT={n_gt}, matched=0")

        paper_results.append({
            "paper_id": paper_id,
            "gt_study": f"{gt_key[0]} {gt_key[1]}" if gt_key else None,
            "ext_obs": n_ext,
            "gt": n_gt,
            "matched": n_matched,
            "mae_pp": round(mae, 2) if mae is not None else None,
        })

    # Overall
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")

    matched_papers = sum(1 for p in paper_results if p.get("matched", 0) > 0)
    total_matched = len(all_matches)

    print(f"Papers: {len(paper_results)} extracted, {matched_papers} with GT matches")
    print(f"Observations: {total_matched} matched")

    if total_matched > 0:
        overall_mae = sum(m['effect_diff'] for m in all_matches) / total_matched
        effects_ext = [m['ext_effect'] for m in all_matches]
        effects_gt = [m['gt_effect'] for m in all_matches]

        # Direction
        dir_total = sum(1 for e, g in zip(effects_ext, effects_gt) if abs(g) > 1)
        dir_ok = sum(1 for e, g in zip(effects_ext, effects_gt) if abs(g) > 1 and (e > 0) == (g > 0))

        # Pearson r
        if len(effects_ext) >= 3:
            mean_e = sum(effects_ext) / len(effects_ext)
            mean_g = sum(effects_gt) / len(effects_gt)
            cov = sum((e - mean_e) * (g - mean_g) for e, g in zip(effects_ext, effects_gt))
            var_e = sum((e - mean_e) ** 2 for e in effects_ext)
            var_g = sum((g - mean_g) ** 2 for g in effects_gt)
            r = cov / math.sqrt(var_e * var_g) if var_e > 0 and var_g > 0 else 0
        else:
            r = None

        # Within thresholds
        w5 = sum(1 for m in all_matches if m['effect_diff'] <= 5)
        w10 = sum(1 for m in all_matches if m['effect_diff'] <= 10)

        print(f"\nOverall metrics ({total_matched} matched observations):")
        print(f"  Pearson r: {r:.3f}" if r else "  Pearson r: N/A")
        print(f"  MAE: {overall_mae:.2f}pp")
        print(f"  Within 5pp: {w5}/{total_matched} ({w5/total_matched*100:.0f}%)")
        print(f"  Within 10pp: {w10}/{total_matched} ({w10/total_matched*100:.0f}%)")
        print(f"  Direction: {dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)" if dir_total else "  Direction: N/A")
        print(f"  Mean effect - Extracted: {sum(effects_ext)/len(effects_ext):.1f}%, GT: {sum(effects_gt)/len(effects_gt):.1f}%")

    # Save
    out_path = AGENT_DIR / "validation_report_agent.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "total_papers": len(paper_results),
            "matched_papers": matched_papers,
            "total_matches": total_matched,
            "paper_results": paper_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
