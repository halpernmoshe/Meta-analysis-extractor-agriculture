#!/usr/bin/env python3
"""
Build a first within-paper feature layer from existing per-paper validation reports.

This uses the markdown reports in output/loladze_combined_51/per_paper as a
surrogate for within-paper evidence channels already identified in earlier work.
The goal is not perfect section parsing; it is a first machine-readable layer
indicating whether the report references abstract/text/table/figure/conclusion
evidence and what type of paper-structure conflict it highlights.
"""

from __future__ import annotations

import csv
import re
import argparse
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
CODEX_OUTPUT = ROOT / "pipeline_replication" / "codex" / "outputs" / "combined_analysis"
REPORT_DIR = ROOT / "output" / "loladze_combined_51" / "per_paper"

DEFAULT_PAPERS = [
    "020_Overdieck_1993",
    "031_Pal_2003",
    "002_Ziska_1997",
    "003_Baslam_2012",
    "004_Finzi_2001",
    "005_Niinemets_1999",
    "007_Woodin_1992",
    "011_Huluka_1994",
    "016_Fernando_2012a",
    "017_Fangmeier_2002",
    "021_Wilsey_1994",
    "040_Pfirrmann_1996",
    "044_Housman_2012",
    "051_Niu_2013",
]
ACTIVE_PAPERS = list(DEFAULT_PAPERS)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def has_any(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def count_matches(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def match_group(text: str, pattern: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return (m.group(1).strip() if m else "")


def is_zero_match_summary(summary: str) -> bool:
    s = (summary or "").strip().lower()
    return ("zero-match" in s) or bool(re.match(r"^0/\d+", s))


def explicit_ai_error_flag(text: str) -> bool:
    if "not an extraction error" in text or "not a data extraction error" in text:
        return False
    return has_any(
        text,
        [
            "unambiguously an ai extraction error",
            "this is an ai extraction error",
            "dominant error source: bar chart reading imprecision",
            "ai extraction error",
        ],
    )


def summarize_report(paper_id: str) -> dict:
    path = REPORT_DIR / f"{paper_id}_report.md"
    row = {
        "paper_id": paper_id,
        "report_path": str(path),
        "report_available": path.exists(),
    }
    if not path.exists():
        return row

    text = read_text(path)
    lower = text.lower()
    overall_rating = match_group(text, r"^\*\*Overall rating:\*\*\s*([^\n]+)")
    status = match_group(text, r"^\*\*Status:\*\*\s*([^\n]+)")
    match_summary = match_group(text, r"^\*\*Match summary:\*\*\s*([^\n]+)")

    row.update(
        {
            "report_overall_rating_text": overall_rating,
            "report_status_text": status,
            "report_match_summary_text": match_summary,
            "report_overall_excellent": "excellent" in overall_rating.lower(),
            "report_overall_partial": "partial" in overall_rating.lower(),
            "report_status_skip": status.lower().startswith("skip"),
            "report_zero_match": is_zero_match_summary(match_summary),
            "report_wrong_tissue": has_any(
                lower,
                [
                    "wrong tissue",
                    "tissue-mismatched",
                    "tissue mismatch",
                    "wrong-tissue",
                ],
            ),
            "report_treatment_arm_confusion": has_any(
                lower,
                [
                    "treatment-arm confusion",
                    "treatment arm ambiguity",
                    "treatment arm confusion",
                    "ec-only",
                    "ec+eo",
                    "clean-air",
                    "polluted-air",
                    "ozone-stratified",
                ],
            ),
            "report_no_concentration_data": has_any(
                lower,
                [
                    "no concentration data",
                    "non-concentration units",
                    "total-content data",
                    "total content per plant",
                ],
            ),
            "report_gt_issue": has_any(
                lower,
                [
                    "ground-truth data quality issue",
                    "database entry error",
                    "gt data quality issue",
                    "gt dataset",
                ],
            ),
            "report_mentions_abstract": has_any(
                lower,
                [
                    "abstract",
                    "abstract explicitly states",
                    "consistent with the abstract",
                ],
            ),
            "report_mentions_results_text": has_any(
                lower,
                [
                    "results section",
                    "paper text",
                    "stated in text",
                    "the paper's text",
                ],
            ),
            "report_mentions_table_evidence": "table " in lower,
            "report_mentions_figure_evidence": "figure " in lower or "bar chart" in lower,
            "report_mentions_conclusion": has_any(
                lower,
                [
                    "overall assessment",
                    "recommended action",
                    "implication for meta-analysis pipeline",
                    "conclusion",
                ],
            ),
            "report_mentions_gt_crosscheck": has_any(
                lower,
                [
                    "gt row",
                    "ground truth",
                    "loladze gt",
                    "cross-check against figure",
                ],
            ),
            "report_timepoint_conflict": has_any(
                lower,
                [
                    "wrong harvest date",
                    "wrong temporal point",
                    "doy ",
                    "sampling date",
                    "final harvest",
                    "intermediate harvest",
                ],
            ),
            "report_averaging_conflict": has_any(
                lower,
                [
                    "unclipped-only",
                    "clipping treatment averaging",
                    "rainfall-averaged",
                    "average of clipped and unclipped",
                    "systematic methodological divergence",
                    "averaging across clipping",
                    "same ai value is compared to all",
                    "matched observation was selected because it is the first",
                ],
            ),
            "report_factorial_conflict": has_any(
                lower,
                [
                    "factorial design",
                    "co2 x",
                    "o3",
                    "irrigation",
                    "clipping",
                    "k deficiency",
                ],
            ),
            "report_figure_digitization_limit": has_any(
                lower,
                [
                    "figure digitization",
                    "figure-only",
                    "bar chart",
                    "vision-based",
                    "not extractable as absolute values",
                    "only in figure",
                ],
            ),
            "report_explicit_ai_error": explicit_ai_error_flag(lower),
            "report_alignment_artifact": has_any(
                lower,
                [
                    "matching/alignment artifact",
                    "methodological divergence",
                    "gt uses unclipped observations only",
                    "same ai value is compared to all",
                    "averaging across clipping",
                ],
            ),
            "report_coverage_limitation": has_any(
                lower,
                [
                    "coverage limitation",
                    "wrong harvest date",
                    "figure digitization",
                    "not extractable as absolute values",
                    "would require figure digitization",
                ],
            ),
            "report_factorial_arm_selection_issue": has_any(
                lower,
                [
                    "prefer the ec-only",
                    "co2 x ozone factorial",
                    "clean-air (20 nmol o3) condition",
                    "factorial aggregation problem",
                    "diluted by the polluted-air",
                ],
            ),
            "report_channel_count": sum(
                int(v)
                for v in [
                    "abstract" in lower,
                    "results section" in lower or "paper text" in lower,
                    "table " in lower,
                    "figure " in lower or "bar chart" in lower,
                    "overall assessment" in lower or "recommended action" in lower,
                ]
            ),
            "report_abstract_mentions_n": count_matches(lower, r"\babstract\b"),
            "report_results_mentions_n": count_matches(lower, r"results section|paper text|stated in text"),
            "report_table_mentions_n": count_matches(lower, r"\btable\s+\d"),
            "report_figure_mentions_n": count_matches(lower, r"\bfigure\s+\d|bar chart"),
            "report_conclusion_mentions_n": count_matches(lower, r"overall assessment|recommended action|conclusion"),
        }
    )
    row["report_channel_weight_abstract"] = 1 if row["report_mentions_abstract"] else 0
    row["report_channel_weight_results_text"] = 2 if row["report_mentions_results_text"] else 0
    row["report_channel_weight_table"] = 3 if row["report_mentions_table_evidence"] else 0
    row["report_channel_weight_figure"] = 1 if row["report_mentions_figure_evidence"] else 0
    row["report_channel_weight_conclusion"] = 1 if row["report_mentions_conclusion"] else 0
    row["report_channel_weighted_support"] = sum(
        row[k]
        for k in [
            "report_channel_weight_abstract",
            "report_channel_weight_results_text",
            "report_channel_weight_table",
            "report_channel_weight_figure",
            "report_channel_weight_conclusion",
        ]
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-list", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="audit_subset")
    args = parser.parse_args()

    global ACTIVE_PAPERS
    if args.paper_list is not None:
        ACTIVE_PAPERS = [
            line.strip()
            for line in args.paper_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    CODEX_OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = [summarize_report(paper_id) for paper_id in ACTIVE_PAPERS]
    out_path = CODEX_OUTPUT / f"{args.prefix}_within_paper_features_2026-03-27.csv"
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)


if __name__ == "__main__":
    main()
