"""
Validate Hui 2023 zinc/wheat extraction against ground truth.

Ground truth: ground.xlsx (Sheets 2-4: Soil, Foliar, Soil+Foliar application)
Our extraction: output/hui2023_extraction/*_consensus.json

Matching strategy:
1. Citation-based: map paper filenames to GT publication strings
2. Value-based: match extracted control/treatment means to GT rows
"""
import sys, json, math, csv
from pathlib import Path
from collections import defaultdict
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import openpyxl

GT_PATH = r"C:\Users\moshe\Dropbox\Testing metaanalyis program\Hui 2023 source data\Source Data\pdfs\ground.xlsx"
RESULTS_DIR = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor\output\hui2023_v2")

# Map our paper filenames to unique substrings in GT Publication column
# Each entry is a list of search terms - ALL must match (AND logic)
PAPER_TO_GT_SEARCH = {
    "41598_2018_Article_25247": ["Dapkekar"],
    "agronomy-10-01566-v2": ["Chattha", "Mahmood"],
    "fpls-08-00281": ["Rehman", "Characterizing"],
    "fpls-10-00426": ["Liu, D. Y.", "2019", "Zinc uptake"],
    "HarevstPlus_Zouetal2012": ["Zou, C.Q., Zhang, Y.Q., Rashid"],
    "pse_pse-201308-0003": ["Ghasal"],
    "s11104-015-2758-0": ["Gomez-Coronado"],
    "s11104-016-2815-3": ["Ram, H., Rashid, A., Zhang, W."],
}

GT_SHEETS = ["Data 2 Soil  application", "Data 3 Foliar application", "Data 4 Soil+Foliar application"]


def load_gt():
    """Load ground truth from all relevant sheets.
    Returns list of dicts: {publication, study_id, obs_id, control_mean, treatment_mean, n, effect_size, sheet}
    """
    # Column positions differ between sheets!
    SHEET_COLS = {
        "Data 2 Soil  application": {"n": 21, "zn_ctrl": 33, "zn_treat": 34, "zn_effect": 35},
        "Data 3 Foliar application": {"n": 30, "zn_ctrl": 42, "zn_treat": 43, "zn_effect": 44},
        "Data 4 Soil+Foliar application": {"n": 4, "zn_ctrl": 14, "zn_treat": 15, "zn_effect": 16},
    }

    wb = openpyxl.load_workbook(GT_PATH, data_only=True)
    all_gt = []

    for sname in GT_SHEETS:
        ws = wb[sname]
        sheet_label = sname.split()[2][:4]  # "Soil", "Foli", "Soil"
        cols = SHEET_COLS[sname]

        for row in ws.iter_rows(min_row=4, values_only=True):
            obs_id = row[0]
            study_id = row[1]
            pub = str(row[2]).strip() if row[2] else ""
            n_val = row[cols["n"]]
            ctrl_mean = row[cols["zn_ctrl"]]
            treat_mean = row[cols["zn_treat"]]
            effect = row[cols["zn_effect"]]

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
                "study_id": study_id,
                "obs_id": obs_id,
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
    terms = PAPER_TO_GT_SEARCH.get(paper_id)
    if not terms:
        return []

    matched = []
    for gt_row in all_gt:
        pub = gt_row["publication"]
        if all(t.lower() in pub.lower() for t in terms):
            matched.append(gt_row)
    return matched


def pool_extraction(obs_list):
    """Pool extraction observations into (control_mean, treatment_mean) pairs.
    Returns list of dicts with control_mean, treatment_mean, effect_pct, ln_rr.
    """
    pooled = []
    for obs in obs_list:
        el = (obs.get("element") or "").upper()
        if "ZN" not in el and "ZINC" not in el:
            continue  # Only Zn observations

        ctrl = obs.get("control_mean")
        treat = obs.get("treatment_mean")
        if ctrl is None or treat is None or ctrl <= 0:
            continue

        ln_rr = math.log(treat / ctrl) if treat > 0 else None
        pct = (treat - ctrl) / ctrl * 100

        pooled.append({
            "control_mean": ctrl,
            "treatment_mean": treat,
            "ln_rr": ln_rr,
            "effect_pct": pct,
            "tissue": obs.get("tissue", ""),
            "treatment_desc": obs.get("treatment_description", "")[:60],
            "n": obs.get("n"),
        })
    return pooled


def value_match(our_obs, gt_rows, tolerance=0.15):
    """Match our observations to GT rows by control/treatment mean similarity.
    Returns list of (our_obs, gt_row, match_quality) tuples.
    """
    matches = []
    used_gt = set()

    for our in our_obs:
        best_match = None
        best_score = float('inf')

        for i, gt in enumerate(gt_rows):
            if i in used_gt:
                continue

            # Relative error for both means
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

    # Effect size comparison (ln response ratio)
    our_effects = []
    gt_effects = []
    abs_errors = []

    for our, gt, _ in matches:
        our_ln = our["ln_rr"]
        gt_ln = gt["effect_size"]
        if our_ln is not None and gt_ln is not None:
            our_effects.append(our_ln)
            gt_effects.append(gt_ln)
            abs_errors.append(abs(our_ln - gt_ln))

    ne = len(our_effects)
    if ne == 0:
        return {"n_matched": n, "n_effect": 0}

    # Mean absolute error on ln_rr
    mae_lnrr = sum(abs_errors) / ne

    # Convert to % effect for easier interpretation
    our_pct = [(math.exp(e) - 1) * 100 for e in our_effects]
    gt_pct = [(math.exp(e) - 1) * 100 for e in gt_effects]
    pct_errors = [abs(o - g) for o, g in zip(our_pct, gt_pct)]
    mae_pct = sum(pct_errors) / ne

    # Within thresholds (on ln_rr scale)
    w005 = sum(1 for e in abs_errors if e <= 0.05)  # ~5% effect
    w010 = sum(1 for e in abs_errors if e <= 0.10)  # ~10% effect
    w020 = sum(1 for e in abs_errors if e <= 0.20)  # ~20% effect

    # Direction agreement (both positive or both negative effect)
    dir_total = sum(1 for o, g in zip(our_effects, gt_effects) if g != 0)
    dir_ok = sum(1 for o, g in zip(our_effects, gt_effects) if g != 0 and (o > 0) == (g > 0))

    # Pearson r on ln_rr
    mean_our = sum(our_effects) / ne
    mean_gt = sum(gt_effects) / ne
    cov = sum((o - mean_our) * (g - mean_gt) for o, g in zip(our_effects, gt_effects))
    var_our = sum((o - mean_our) ** 2 for o in our_effects)
    var_gt = sum((g - mean_gt) ** 2 for g in gt_effects)
    r = cov / math.sqrt(var_our * var_gt) if var_our > 0 and var_gt > 0 else 0

    # Mean comparison (control means)
    ctrl_errors = [abs(our["control_mean"] - gt["control_mean"]) / gt["control_mean"]
                   for our, gt, _ in matches if gt["control_mean"] > 0]
    treat_errors = [abs(our["treatment_mean"] - gt["treatment_mean"]) / gt["treatment_mean"]
                    for our, gt, _ in matches if gt["treatment_mean"] > 0]

    return {
        "n_matched": n,
        "n_effect": ne,
        "pearson_r": round(r, 3),
        "mae_lnrr": round(mae_lnrr, 4),
        "mae_pct": round(mae_pct, 2),
        "within_5pct_lnrr": f"{w005}/{ne} ({w005/ne*100:.0f}%)",
        "within_10pct_lnrr": f"{w010}/{ne} ({w010/ne*100:.0f}%)",
        "within_20pct_lnrr": f"{w020}/{ne} ({w020/ne*100:.0f}%)",
        "direction": f"{dir_ok}/{dir_total} ({dir_ok/dir_total*100:.0f}%)" if dir_total else "N/A",
        "ctrl_mean_error": f"{sum(ctrl_errors)/len(ctrl_errors)*100:.1f}%" if ctrl_errors else "N/A",
        "treat_mean_error": f"{sum(treat_errors)/len(treat_errors)*100:.1f}%" if treat_errors else "N/A",
    }


def main():
    print(f"Hui 2023 Zinc/Wheat Validation")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Load ground truth
    print("Loading ground truth...")
    all_gt = load_gt()
    print(f"  Total GT observations: {len(all_gt)} across {len(GT_SHEETS)} sheets")

    # Count unique studies
    unique_studies = len(set(f"{g['publication'][:50]}_{g['sheet']}" for g in all_gt))
    print(f"  Unique study-sheet combos: {unique_studies}")

    # Load extraction results
    results_files = sorted(RESULTS_DIR.glob("*_consensus.json"))
    print(f"\nExtraction results: {len(results_files)} papers\n")

    all_matches = []
    paper_summaries = []

    for rf in results_files:
        paper_id = rf.stem.replace("_consensus", "")

        with open(rf) as f:
            data = json.load(f)

        obs_list = data.get("consensus_observations", [])
        our_zn = pool_extraction(obs_list)

        # Find GT rows for this paper
        gt_rows = match_gt_for_paper(paper_id, all_gt)

        if not gt_rows:
            print(f"  {paper_id}: NO GT MATCH FOUND")
            paper_summaries.append({
                "paper_id": paper_id,
                "our_obs": len(our_zn),
                "gt_rows": 0,
                "matched": 0,
                "note": "no GT citation match"
            })
            continue

        # Value-match our observations to GT rows
        matches = value_match(our_zn, gt_rows, tolerance=0.20)

        # Stats for this paper
        stats = calc_stats(matches)
        all_matches.extend(matches)

        capture = len(matches)
        total_gt = len(gt_rows)
        rate = capture / total_gt * 100 if total_gt > 0 else 0

        print(f"  {paper_id}:")
        print(f"    Our Zn obs: {len(our_zn)}, GT rows: {total_gt}, Matched: {capture} ({rate:.0f}%)")
        if stats.get("n_effect", 0) > 0:
            print(f"    Pearson r: {stats['pearson_r']}, MAE: {stats['mae_pct']}%")
            print(f"    Direction: {stats['direction']}")
            print(f"    Ctrl mean err: {stats['ctrl_mean_error']}, Treat mean err: {stats['treat_mean_error']}")

        paper_summaries.append({
            "paper_id": paper_id,
            "our_obs": len(our_zn),
            "gt_rows": total_gt,
            "matched": capture,
            "capture_rate": f"{rate:.0f}%",
            "stats": stats,
        })

    # Overall stats
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")

    overall = calc_stats(all_matches)
    total_gt = sum(p["gt_rows"] for p in paper_summaries)
    total_matched = sum(p["matched"] for p in paper_summaries)
    papers_with_gt = sum(1 for p in paper_summaries if p["gt_rows"] > 0)

    print(f"Papers with GT match:    {papers_with_gt}/{len(results_files)}")
    print(f"Observation capture:     {total_matched}/{total_gt} ({total_matched/total_gt*100:.0f}%)" if total_gt else "N/A")
    if overall:
        print(f"Effect size matches:     {overall.get('n_effect', 0)}")
        print(f"Pearson r (ln_rr):       {overall.get('pearson_r', 'N/A')}")
        print(f"MAE (% effect):          {overall.get('mae_pct', 'N/A')}%")
        print(f"Within 5% (ln_rr):       {overall.get('within_5pct_lnrr', 'N/A')}")
        print(f"Within 10% (ln_rr):      {overall.get('within_10pct_lnrr', 'N/A')}")
        print(f"Within 20% (ln_rr):      {overall.get('within_20pct_lnrr', 'N/A')}")
        print(f"Direction agreement:     {overall.get('direction', 'N/A')}")
        print(f"Control mean error:      {overall.get('ctrl_mean_error', 'N/A')}")
        print(f"Treatment mean error:    {overall.get('treat_mean_error', 'N/A')}")

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall": overall,
        "per_paper": paper_summaries,
        "total_gt_obs": total_gt,
        "total_matched": total_matched,
        "all_matches": [
            {
                "our_ctrl": our["control_mean"],
                "our_treat": our["treatment_mean"],
                "our_lnrr": our["ln_rr"],
                "gt_ctrl": gt["control_mean"],
                "gt_treat": gt["treatment_mean"],
                "gt_lnrr": gt["effect_size"],
                "gt_pub": gt["publication"][:60],
                "match_quality": round(score, 4),
            }
            for our, gt, score in all_matches
        ]
    }

    out_path = RESULTS_DIR / "validation_hui2023.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # CSV of matches
    csv_path = RESULTS_DIR / "validation_hui2023_matches.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["our_ctrl", "our_treat", "our_lnrr", "gt_ctrl", "gt_treat", "gt_lnrr", "gt_pub", "match_qual"])
        for our, gt, score in all_matches:
            w.writerow([
                round(our["control_mean"], 2),
                round(our["treatment_mean"], 2),
                round(our["ln_rr"], 4) if our["ln_rr"] else "",
                round(gt["control_mean"], 2),
                round(gt["treatment_mean"], 2),
                round(gt["effect_size"], 4) if gt["effect_size"] else "",
                gt["publication"][:60],
                round(score, 4),
            ])
    print(f"Matches CSV: {csv_path}")


if __name__ == "__main__":
    main()
