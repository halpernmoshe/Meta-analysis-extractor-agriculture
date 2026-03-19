"""
Validate Hui 2023 zinc/wheat AGENT extraction against ground truth.

Ground truth: ground.xlsx (Sheets 2-4: Soil, Foliar, Soil+Foliar application)
Agent extraction: output/hui2023_agent_extraction/*_agent*.json

Matching strategy:
1. Citation-based: map paper filenames to GT publication strings
2. Value-based: match extracted control/treatment means to GT rows
"""
import sys, json, math
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import openpyxl

GT_PATH = r"C:\Users\moshe\Dropbox\Testing metaanalyis program\Hui 2023 source data\Source Data\pdfs\ground.xlsx"
AGENT_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\hui2023_agent_extraction")

# Map agent extraction filenames to GT publication substrings
# Key = part of filename (without _agent.json), Value = list of search terms (AND logic)
PAPER_TO_GT_SEARCH = {
    # new_downloads numbered files
    "03_Zhang_2012": ["Zhang", "2012"],
    "05_Yang_2011": ["Yang", "2011"],
    "11_Zhao_2020": ["Zhao", "2020"],
    "21_Wang_2012": ["Wang", "2012"],
    "38_Yilmaz_1997": ["Yilmaz", "1997"],
    "42_Curtin_2008": ["Curtin", "2008"],
    "44_Cakmak_1997": ["Cakmak", "1997", "Concentration"],
    "46_Ghasal_2017": ["Ghasal"],
    "49_Dawar_2022": ["Dawar"],
    "50_Erdal_2002": ["Erdal"],
    "52_Forster_2018": ["Forster", "2018"],
    "53_Grant_1998": ["Grant", "1998"],
    "58_Kalayci_1999": ["Kalayci"],
    "59_Khoshgoftarmanesh_2013": ["Khoshgoftarmanesh"],
    "61_Kumar_2018": ["Kumar", "2018"],
    "62_Morshedi_2012": ["Morshedi"],
    "63_Mosavian_2021": ["Mosavian"],
    "65_Oliver_1994": ["Oliver", "1994"],
    "66_PahlavanRad_2009": ["Pahlavan"],
    "68_Peck_2008": ["Peck"],
    "69_Ramzan_2020": ["Ramzan"],
    "70_Rehman_2018": ["Rehman", "2018"],
    "82_Torun_2001": ["Torun"],
    "84_Yilmaz_1998": ["Yilmaz", "1998"],
    "Dong_2018": ["Dong", "2018"],
    "Li_2013": ["Li", "2013"],
    "Liu_2014": ["Liu", "2014"],
    "Rashid_2019": ["Rashid"],
    "Zhang_2017": ["Zhang", "2017"],
    # root directory PDFs
    "41598_2018_Article_25247": ["Dapkekar"],
    "agronomy-10-01566-v2": ["Chattha", "Mahmood"],
    "fpls-08-00281": ["Rehman", "Characterizing"],
    "fpls-10-00426": ["Liu, D. Y.", "2019", "Zinc uptake"],
    "HarevstPlus_Zouetal2012": ["Zou, C.Q.", "Rashid"],
    "pse_pse-201308-0003": ["Ghasal"],
    "s11104-015-2758-0": ["Gomez-Coronado"],
    "s11104-016-2815-3": ["Ram, H.", "Rashid"],
}

GT_SHEETS = ["Data 2 Soil  application", "Data 3 Foliar application", "Data 4 Soil+Foliar application"]


def load_gt():
    """Load ground truth from all relevant sheets."""
    SHEET_COLS = {
        "Data 2 Soil  application": {"n": 21, "zn_ctrl": 33, "zn_treat": 34, "zn_effect": 35},
        "Data 3 Foliar application": {"n": 30, "zn_ctrl": 42, "zn_treat": 43, "zn_effect": 44},
        "Data 4 Soil+Foliar application": {"n": 4, "zn_ctrl": 14, "zn_treat": 15, "zn_effect": 16},
    }

    wb = openpyxl.load_workbook(GT_PATH, data_only=True)
    all_gt = []

    for sname in GT_SHEETS:
        ws = wb[sname]
        cols = SHEET_COLS[sname]
        sheet_label = sname.split()[2][:4]

        for row in ws.iter_rows(min_row=4, values_only=True):
            pub = str(row[2]).strip() if row[2] else ""
            ctrl_mean = row[cols["zn_ctrl"]]
            treat_mean = row[cols["zn_treat"]]
            effect = row[cols["zn_effect"]]
            n_val = row[cols["n"]]

            if not pub or ctrl_mean is None or treat_mean is None:
                continue

            try:
                ctrl = float(ctrl_mean)
                treat = float(treat_mean)
                eff = float(effect) if effect is not None else None
                n = int(n_val) if n_val is not None else None
            except (ValueError, TypeError):
                continue

            if ctrl <= 0:
                continue

            all_gt.append({
                "publication": pub,
                "control_mean": ctrl,
                "treatment_mean": treat,
                "effect_size": eff,
                "n": n,
                "sheet": sheet_label,
            })

    wb.close()
    return all_gt


def match_gt_for_paper(paper_id, all_gt):
    """Find GT rows matching a paper by citation search terms."""
    # Try exact key first
    terms = PAPER_TO_GT_SEARCH.get(paper_id)
    if not terms:
        # Try partial match on key
        for key, val in PAPER_TO_GT_SEARCH.items():
            if key.lower() in paper_id.lower() or paper_id.lower() in key.lower():
                terms = val
                break
    if not terms:
        return []

    matched = []
    for gt_row in all_gt:
        pub = gt_row["publication"]
        if all(t.lower() in pub.lower() for t in terms):
            matched.append(gt_row)
    return matched


def pool_extraction(obs_list):
    """Pool extraction observations - only keep Zn grain observations."""
    pooled = []
    for obs in obs_list:
        el = str(obs.get("element", "")).upper()
        if "ZN" not in el and "ZINC" not in el:
            continue

        ctrl = obs.get("control_mean")
        treat = obs.get("treatment_mean")
        if ctrl is None or treat is None:
            continue
        try:
            ctrl = float(ctrl)
            treat = float(treat)
        except (ValueError, TypeError):
            continue
        if ctrl <= 0:
            continue

        ln_rr = math.log(treat / ctrl) if treat > 0 else None
        pct = (treat - ctrl) / ctrl * 100

        pooled.append({
            "control_mean": ctrl,
            "treatment_mean": treat,
            "ln_rr": ln_rr,
            "effect_pct": pct,
            "tissue": obs.get("tissue", ""),
            "treatment_desc": str(obs.get("treatment_description", ""))[:60],
            "n": obs.get("n"),
        })
    return pooled


def value_match(our_obs, gt_rows, tolerance=0.20):
    """Match our observations to GT rows by control/treatment mean similarity."""
    matches = []
    used_gt = set()

    for our in our_obs:
        best_match = None
        best_score = float('inf')

        for i, gt in enumerate(gt_rows):
            if i in used_gt:
                continue

            ctrl_err = abs(our["control_mean"] - gt["control_mean"]) / max(gt["control_mean"], 0.1)
            treat_err = abs(our["treatment_mean"] - gt["treatment_mean"]) / max(gt["treatment_mean"], 0.1)
            combined = (ctrl_err + treat_err) / 2

            if combined < best_score and combined < tolerance:
                best_score = combined
                best_match = (i, gt)

        if best_match:
            idx, gt = best_match
            used_gt.add(idx)
            matches.append((our, gt, best_score))

    return matches


def calc_stats(matches):
    """Calculate validation statistics from matched pairs."""
    if not matches:
        return {}

    n = len(matches)
    our_effects = []
    gt_effects = []
    abs_errors = []

    for our, gt, _ in matches:
        our_pct = our["effect_pct"]
        gt_eff = gt["effect_size"]
        if gt_eff is not None:
            # GT effect_size is ln(RR), convert our % to ln(RR)
            our_ln = our["ln_rr"]
            if our_ln is not None:
                our_effects.append(our_pct)
                gt_pct = (math.exp(gt_eff) - 1) * 100 if abs(gt_eff) < 10 else gt_eff
                gt_effects.append(gt_pct)
                abs_errors.append(abs(our_pct - gt_pct))

    ne = len(our_effects)
    if ne == 0:
        return {"n_matched": n, "n_effect": 0}

    mae_pct = sum(abs_errors) / ne

    # Within thresholds (on % scale)
    w5 = sum(1 for e in abs_errors if e <= 5)
    w10 = sum(1 for e in abs_errors if e <= 10)
    w20 = sum(1 for e in abs_errors if e <= 20)

    # Direction agreement
    dir_total = sum(1 for o, g in zip(our_effects, gt_effects) if abs(g) > 1)
    dir_ok = sum(1 for o, g in zip(our_effects, gt_effects) if abs(g) > 1 and (o > 0) == (g > 0))

    # Pearson r
    if ne >= 3:
        mean_our = sum(our_effects) / ne
        mean_gt = sum(gt_effects) / ne
        cov = sum((o - mean_our) * (g - mean_gt) for o, g in zip(our_effects, gt_effects))
        var_our = sum((o - mean_our) ** 2 for o in our_effects)
        var_gt = sum((g - mean_gt) ** 2 for g in gt_effects)
        r = cov / math.sqrt(var_our * var_gt) if var_our > 0 and var_gt > 0 else 0
    else:
        r = None

    # Mean comparison
    ctrl_errors = [abs(our["control_mean"] - gt["control_mean"]) / gt["control_mean"]
                   for our, gt, _ in matches if gt["control_mean"] > 0]
    treat_errors = [abs(our["treatment_mean"] - gt["treatment_mean"]) / gt["treatment_mean"]
                    for our, gt, _ in matches if gt["treatment_mean"] > 0]

    return {
        "n_matched": n,
        "n_effect": ne,
        "pearson_r": round(r, 3) if r is not None else None,
        "mae_pct": round(mae_pct, 2),
        "within_5pp": f"{w5}/{ne} ({w5/ne*100:.0f}%)",
        "within_10pp": f"{w10}/{ne} ({w10/ne*100:.0f}%)",
        "within_20pp": f"{w20}/{ne} ({w20/ne*100:.0f}%)",
        "direction": f"{dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)" if dir_total else "N/A",
        "ctrl_mean_err": f"{sum(ctrl_errors)/len(ctrl_errors)*100:.1f}%" if ctrl_errors else "N/A",
        "treat_mean_err": f"{sum(treat_errors)/len(treat_errors)*100:.1f}%" if treat_errors else "N/A",
    }


def main():
    print(f"Hui 2023 Zinc/Wheat — AGENT Extraction Validation")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Load ground truth
    print("Loading ground truth...")
    all_gt = load_gt()
    print(f"  Total GT observations: {len(all_gt)} across {len(GT_SHEETS)} sheets\n")

    # Load agent extraction results
    agent_files = {}
    for f in sorted(AGENT_DIR.glob("*_agent*.json")):
        paper_id = f.stem.replace("_agent_v2", "").replace("_agent", "")
        if '_v2' in f.stem or paper_id not in agent_files:
            agent_files[paper_id] = f

    print(f"Agent extraction files: {len(agent_files)} papers\n")

    all_matches = []
    paper_results = []

    for paper_id, fpath in sorted(agent_files.items()):
        try:
            with open(fpath, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  {paper_id}: ERROR loading JSON - {e}")
            continue

        obs_list = data.get("consensus_observations", [])
        our_zn = pool_extraction(obs_list)

        gt_rows = match_gt_for_paper(paper_id, all_gt)

        if not gt_rows:
            print(f"  {paper_id}: {len(our_zn)} Zn obs, NO GT MATCH")
            paper_results.append({"paper_id": paper_id, "our_zn": len(our_zn), "gt": 0, "matched": 0})
            continue

        matches = value_match(our_zn, gt_rows, tolerance=0.20)
        stats = calc_stats(matches)
        all_matches.extend(matches)

        capture = f"{len(matches)}/{len(gt_rows)}" if gt_rows else "N/A"
        r_str = f"r={stats.get('pearson_r', 'N/A')}" if stats.get('pearson_r') is not None else ""
        mae_str = f"MAE={stats.get('mae_pct', 'N/A')}pp" if stats.get('mae_pct') is not None else ""

        print(f"  {paper_id}: {len(our_zn)} Zn obs, GT={len(gt_rows)}, matched={len(matches)} ({capture}) {r_str} {mae_str}")

        paper_results.append({
            "paper_id": paper_id,
            "our_zn": len(our_zn),
            "gt": len(gt_rows),
            "matched": len(matches),
            **stats,
        })

    # Overall statistics
    print(f"\n{'='*70}")
    print(f"OVERALL RESULTS")
    print(f"{'='*70}")

    total_papers = len(paper_results)
    matched_papers = sum(1 for p in paper_results if p.get("matched", 0) > 0)
    total_matches = len(all_matches)
    total_gt = sum(p.get("gt", 0) for p in paper_results)
    total_our = sum(p.get("our_zn", 0) for p in paper_results)

    print(f"Papers: {total_papers} extracted, {matched_papers} with GT matches")
    print(f"Observations: {total_our} extracted Zn, {total_gt} in GT, {total_matches} matched")

    if total_matches > 0:
        overall = calc_stats(all_matches)
        print(f"\nOverall metrics (on {total_matches} matched observations):")
        for k, v in overall.items():
            print(f"  {k}: {v}")

    # Save matched pairs for formal stats
    match_pairs = []
    for our, gt, score in all_matches:
        gt_pct = (math.exp(gt["effect_size"]) - 1) * 100 if gt["effect_size"] is not None and abs(gt["effect_size"]) < 10 else (gt["effect_size"] if gt["effect_size"] else 0)
        match_pairs.append({
            "our_pct": our["effect_pct"],
            "gt_pct": gt_pct,
            "paper": gt.get("publication", "unknown"),
            "our_ctrl": our["control_mean"],
            "our_treat": our["treatment_mean"],
            "gt_ctrl": gt["control_mean"],
            "gt_treat": gt["treatment_mean"],
        })

    # Save results
    out_path = AGENT_DIR / "validation_report_agent.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            "date": datetime.now().isoformat(),
            "total_papers": total_papers,
            "matched_papers": matched_papers,
            "total_matches": total_matches,
            "paper_results": paper_results,
            "overall": calc_stats(all_matches) if all_matches else {},
            "match_pairs": match_pairs,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
