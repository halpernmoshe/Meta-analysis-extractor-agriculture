#!/usr/bin/env python3
"""Exploratory coverage analysis using published-reference structural burden.

This analysis reconstructs the exact Table 2 cells and their matched status from
the deposited key tables. It never uses outcome values to decide whether cells
match. Structural burden is described with three transparent proxies:

1. number of final published-reference cells contributed by the paper;
2. number of original published-reference rows contributed by the paper; and
3. number of original published-reference rows sharing the final comparison key.

The generic structural slots have dataset-specific meanings, so comparisons are
descriptive and within-dataset. No composite difficulty score or significance
test is used.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = REPO / "runs"

DISPLAY = {
    "Boldorini": "Boldorini et al. 2024",
    "Biochar": "Li X et al. 2024",
    "Hui": "Hui et al. 2025",
    "Loladze": "Loladze 2014",
    "Li2022": "Li J et al. 2022",
}
BASE = {
    "Boldorini": "boldorini/keys",
    "Biochar": "biochar_v2/keys",
    "Hui": "hui_v4/keys",
    "Loladze": "loladze_v2/keys",
    "Li2022": "li2022_v2/keys",
}
EXCLUDE = {
    "Hui": {
        "zhao_2020", "cakmak_1997", "liu_2014", "dong_2018",
        "li_2013", "zhang_2012", "khoshgoftarmanesh_2013", "kumar_2018",
    },
    "Loladze": {
        "johnson_1997", "ma_2007", "rodenkirchen_2009",
        "de_2000", "kuehny_1991", "li_2010",
    },
    "Li2022": {"pramanick_2016", "al-tawaha-et-al-2011"},
    "Biochar": {"jose_2013"},
}
METRIC = {
    "Boldorini": "raw",
    "Biochar": "raw",
    "Hui": "raw",
    "Loladze": "effect",
    "Li2022": "effect",
}
STRUCTURAL_FIELDS = (
    "outcome_canonical",
    "crop",
    "treatment_level",
    "co_amendment",
    "co_amendment_level",
    "timepoint",
    "aggregation_level",
)


def npid(value: object) -> str:
    return re.sub(r"^[\d_]+", "", str(value).strip().lower())


def low(row: dict[str, str], key: str) -> str:
    return str(row.get(key, "")).strip().lower()


def finite_float(value: object) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numtok(row: dict[str, str], key: str) -> str:
    value = finite_float(row.get(key))
    return f"{round(value, 4):g}" if value is not None else low(row, key)


def effect(row: dict[str, str]) -> float | None:
    if low(row, "unit_canonical") == "ratio":
        return finite_float(row.get("treatment_mean"))
    treatment = finite_float(row.get("treatment_mean"))
    control = finite_float(row.get("control_mean"))
    if treatment is None or control in (None, 0):
        return None
    return (treatment - control) / control


def raw_ok(row: dict[str, str]) -> bool:
    return (
        low(row, "unit_canonical") not in {"ratio", "unresolved"}
        and finite_float(row.get("treatment_mean")) is not None
    )


def load_rows(relative_dir: str, excluded: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((RUNS / relative_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if npid(row.get("paper_id", "")) not in excluded:
                    rows.append(row)
    return rows


def key_function(dataset: str):
    if dataset == "Boldorini":
        return lambda row: (
            npid(low(row, "paper_id")),
            low(row, "outcome_canonical"),
            low(row, "crop"),
            low(row, "treatment_level"),
            low(row, "co_amendment"),
            numtok(row, "co_amendment_level"),
            low(row, "timepoint"),
            low(row, "unit_canonical"),
        )
    if dataset == "Biochar":
        return lambda row: (
            npid(low(row, "paper_id")),
            low(row, "outcome_canonical"),
            low(row, "crop"),
            low(row, "treatment_level"),
            low(row, "co_amendment"),
            numtok(row, "co_amendment_level"),
            low(row, "timepoint"),
        )
    if dataset == "Hui":
        return lambda row: (
            npid(low(row, "paper_id")),
            low(row, "outcome_canonical"),
            low(row, "treatment_level"),
        )
    if dataset == "Loladze":
        return lambda row: (
            npid(low(row, "paper_id")),
            low(row, "treatment_level"),
            low(row, "co_amendment"),
            low(row, "co_amendment_level"),
        )
    raise ValueError(f"No ordinary key for {dataset}")


def li2022_crosswalk(ai_rows: list[dict[str, str]]):
    last_name = re.compile(r"[a-z][a-z\-]+")

    def first_last(text: str) -> str | None:
        match = last_name.search(text.lower())
        return match.group(0) if match else None

    index: dict[tuple[str, str | None], str] = {}
    for row in ai_rows:
        paper = npid(low(row, "paper_id"))
        year_match = re.search(r"(19|20)\d{2}", paper)
        author = first_last(paper)
        if author:
            year = year_match.group(0) if year_match else None
            index[(author, year)] = paper
            index.setdefault((author, None), paper)

    def remap(row: dict[str, str]) -> str:
        paper = npid(low(row, "paper_id"))
        if not paper.startswith(("study", "gt_study")):
            return paper
        match = re.search(
            r"author='([^']+)'\s*[, ]*((?:19|20)\d{2})",
            row.get("evidence", ""),
        )
        if not match:
            return paper
        # Source-title audit: do not merge distinct same-author/year GT studies.
        if paper in {"gt_study08", "gt_study146"}:
            return paper
        author = first_last(match.group(1))
        year = match.group(2)
        if not author:
            return paper
        return index.get((author, year)) or index.get((author, None)) or paper

    return remap


def has_metric(row: dict[str, str], metric: str) -> bool:
    return raw_ok(row) if metric == "raw" else effect(row) is not None


def rank(values: list[float]) -> list[float]:
    """Average ranks, starting at 1, for a list that may contain ties."""
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            result[order[position]] = average
        start = end
    return result


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    ss_x = sum((value - mean_x) ** 2 for value in x)
    ss_y = sum((value - mean_y) ** 2 for value in y)
    if ss_x == 0 or ss_y == 0:
        return None
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / math.sqrt(ss_x * ss_y)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(rank(x), rank(y))


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def fmt(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def pctfmt(value: float | None, digits: int = 1) -> str:
    return "NA" if value is None else f"{value * 100:.{digits}f}%"


def build_cells(dataset: str) -> list[dict[str, object]]:
    excluded = {npid(item) for item in EXCLUDE.get(dataset, set())}
    ai_rows = load_rows(f"{BASE[dataset]}/ai", excluded)
    gt_rows = load_rows(f"{BASE[dataset]}/gt", excluded)
    metric = METRIC[dataset]

    if dataset == "Li2022":
        remap = li2022_crosswalk(ai_rows)
        ai_key = lambda row: (npid(low(row, "paper_id")),)
        gt_key = lambda row: (remap(row),)
    else:
        ai_key = gt_key = key_function(dataset)

    ai_cells: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    gt_cells: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in ai_rows:
        ai_cells[ai_key(row)].append(row)
    for row in gt_rows:
        gt_cells[gt_key(row)].append(row)

    paper_cells: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for cell_key in gt_cells:
        paper_cells[cell_key[0]].append(cell_key)
    ai_papers = {cell_key[0] for cell_key in ai_cells}

    output: list[dict[str, object]] = []
    for cell_key, rows in sorted(gt_cells.items(), key=lambda item: str(item[0])):
        paper = cell_key[0]
        all_paper_rows = [row for key in paper_cells[paper] for row in gt_cells[key]]
        ai_metric = any(has_metric(row, metric) for row in ai_cells.get(cell_key, []))
        gt_metric = any(has_metric(row, metric) for row in rows)
        figure_flags = {low(row, "is_figure") for row in rows if low(row, "is_figure") in {"0", "1"}}
        if figure_flags == {"1"}:
            source_format = "figure"
        elif figure_flags == {"0"}:
            source_format = "nonfigure"
        elif figure_flags == {"0", "1"}:
            source_format = "mixed"
        else:
            source_format = "unknown"
        record: dict[str, object] = {
            "dataset": DISPLAY[dataset],
            "dataset_code": dataset,
            "paper_id": paper,
            "cell_key": " | ".join(cell_key),
            "matched": int(cell_key in ai_cells and ai_metric and gt_metric),
            "same_paper_ai_records_present": int(paper in ai_papers),
            "reference_source_format": source_format,
            "reference_rows_in_cell": len(rows),
            "reference_cells_in_paper": len(paper_cells[paper]),
            "reference_rows_in_paper": len(all_paper_rows),
        }
        for field in STRUCTURAL_FIELDS:
            values = {low(row, field) for row in all_paper_rows if low(row, field)}
            record[f"distinct_{field}_in_paper"] = len(values)
        output.append(record)
    return output


def summarize_cells(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for dataset in DISPLAY:
        dataset_cells = [row for row in cells if row["dataset_code"] == dataset]
        matched = [row for row in dataset_cells if row["matched"] == 1]
        unmatched = [row for row in dataset_cells if row["matched"] == 0]
        present = [row for row in dataset_cells if row["same_paper_ai_records_present"] == 1]
        absent = [row for row in dataset_cells if row["same_paper_ai_records_present"] == 0]
        unmatched_present = [row for row in present if row["matched"] == 0]

        def values(rows: list[dict[str, object]], field: str) -> list[float]:
            return [float(row[field]) for row in rows]

        result: dict[str, object] = {
            "dataset": DISPLAY[dataset],
            "reference_cells": len(dataset_cells),
            "matched_cells": len(matched),
            "unmatched_cells": len(unmatched),
            "coverage": len(matched) / len(dataset_cells) if dataset_cells else None,
            "same_paper_present_cells": len(present),
            "same_paper_absent_cells": len(absent),
            "unmatched_same_paper_present_cells": len(unmatched_present),
            "coverage_when_same_paper_present": len(matched) / len(present) if present else None,
        }
        for label, rows in (("matched", matched), ("unmatched", unmatched_present)):
            for field in (
                "reference_cells_in_paper",
                "reference_rows_in_paper",
                "reference_rows_in_cell",
            ):
                field_values = values(rows, field)
                result[f"{label}_{field}_median"] = median(field_values) if field_values else None
                result[f"{label}_{field}_q1"] = percentile(field_values, 0.25)
                result[f"{label}_{field}_q3"] = percentile(field_values, 0.75)
            result[f"{label}_pooled_cell_fraction"] = (
                sum(row["reference_rows_in_cell"] > 1 for row in rows) / len(rows)
                if rows else None
            )
        summary.append(result)
    return summary


def summarize_papers(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in cells:
        grouped[(str(row["dataset_code"]), str(row["paper_id"]))].append(row)

    papers: list[dict[str, object]] = []
    for (dataset, paper), rows in sorted(grouped.items()):
        total = len(rows)
        matched = sum(int(row["matched"]) for row in rows)
        papers.append({
            "dataset": DISPLAY[dataset],
            "dataset_code": dataset,
            "paper_id": paper,
            "same_paper_ai_records_present": int(rows[0]["same_paper_ai_records_present"]),
            "reference_cells": total,
            "reference_rows": int(rows[0]["reference_rows_in_paper"]),
            "matched_cells": matched,
            "match_rate": matched / total,
            "median_reference_rows_per_cell": median(
                float(row["reference_rows_in_cell"]) for row in rows
            ),
        })
    return papers


def paper_associations(papers: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DISPLAY:
        rows = [
            row for row in papers
            if row["dataset_code"] == dataset and row["same_paper_ai_records_present"] == 1
        ]
        match_rates = [float(row["match_rate"]) for row in rows]
        cell_burden = [float(row["reference_cells"]) for row in rows]
        row_burden = [float(row["reference_rows"]) for row in rows]
        output.append({
            "dataset": DISPLAY[dataset],
            "papers": len(rows),
            "spearman_reference_cells_vs_match_rate": spearman(cell_burden, match_rates),
            "spearman_reference_rows_vs_match_rate": spearman(row_burden, match_rates),
        })
    return output


def source_format_summary(cells: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset in DISPLAY:
        dataset_cells = [row for row in cells if row["dataset_code"] == dataset]
        observed_formats = {str(row["reference_source_format"]) for row in dataset_cells}
        if not {"figure", "nonfigure"}.issubset(observed_formats):
            continue
        for source_format in ("figure", "nonfigure", "mixed", "unknown"):
            rows = [
                row for row in dataset_cells
                if row["reference_source_format"] == source_format
                and row["same_paper_ai_records_present"] == 1
            ]
            if not rows:
                continue
            matched = sum(int(row["matched"]) for row in rows)
            output.append({
                "dataset": DISPLAY[dataset],
                "source_format": source_format,
                "same_paper_reference_cells": len(rows),
                "matched_cells": matched,
                "match_rate": matched / len(rows),
            })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    summary: list[dict[str, object]],
    associations: list[dict[str, object]],
) -> None:
    association_by_dataset = {row["dataset"]: row for row in associations}
    lines = [
        "# Exploratory structural-complexity analysis of coverage",
        "",
        "Matched status was reconstructed using the exact outcome-blind structural keys and "
        "dataset-specific pooling rules used for manuscript Table 2. Outcome values were not "
        "used to form matches. Complexity was not scored. The analysis used transparent "
        "structural-burden proxies available in the published-reference key records.",
        "",
        "| Dataset | All reference cells | Same-paper AI records present | Same-paper AI records absent | Matched cells | Overall coverage | Coverage when same-paper AI records present |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['reference_cells']} | "
            f"{row['same_paper_present_cells']} | {row['same_paper_absent_cells']} | "
            f"{row['matched_cells']} | {row['coverage']:.0%} | "
            f"{row['coverage_when_same_paper_present']:.0%} |"
        )

    lines.extend([
        "",
        "Same-paper presence means that the final published crosswalk identified at least one AI "
        "record from that paper. Absence can reflect a paper outside the processed corpus or an "
        "unresolved paper identifier; it is not classified as extraction difficulty.",
        "",
        "The structural-burden comparison below is restricted to reference cells from papers with "
        "same-paper AI records. The unmatched column therefore excludes cells from absent or "
        "unresolved papers.",
        "",
        "| Dataset | Same-paper reference cells | Matched | Unmatched | Within-paper coverage | Cells/paper, matched | Cells/paper, unmatched | Rows/cell, matched | Rows/cell, unmatched | Multirow cells, matched | Multirow cells, unmatched |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {row['same_paper_present_cells']} | {row['matched_cells']} | "
            f"{row['unmatched_same_paper_present_cells']} | {row['coverage_when_same_paper_present']:.0%} | "
            f"{fmt(row['matched_reference_cells_in_paper_median'])} | "
            f"{fmt(row['unmatched_reference_cells_in_paper_median'])} | "
            f"{fmt(row['matched_reference_rows_in_cell_median'])} | "
            f"{fmt(row['unmatched_reference_rows_in_cell_median'])} | "
            f"{pctfmt(row['matched_pooled_cell_fraction'])} | "
            f"{pctfmt(row['unmatched_pooled_cell_fraction'])} |"
        )

    lines.extend([
        "",
        "Values are medians unless shown as percentages. A multirow cell contains more than one "
        "original published-reference row under the final comparison key. This multiplicity does "
        "not imply that the reason for aggregation was documented.",
        "",
        "## Paper-level descriptive associations",
        "",
        "| Dataset | Papers | Spearman rho: reference cells vs match rate | Spearman rho: reference rows vs match rate |",
        "|---|---:|---:|---:|",
    ])
    for row in summary:
        association = association_by_dataset[row["dataset"]]
        lines.append(
            f"| {row['dataset']} | {association['papers']} | "
            f"{fmt(association['spearman_reference_cells_vs_match_rate'])} | "
            f"{fmt(association['spearman_reference_rows_vs_match_rate'])} |"
        )

    lines.extend([
        "",
        "These correlations are restricted to papers with same-paper AI records and are "
        "descriptive only. Papers, cells, and structural slots differ "
        "across datasets; no pooled test, significance test, or composite difficulty score was "
        "used. Reporting quality and extraction uncertainty are not consistently encoded in the "
        "published-reference datasets and are therefore not tested here.",
        "",
    ])
    (HERE / "coverage_structural_complexity_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    cells = [cell for dataset in DISPLAY for cell in build_cells(dataset)]
    summary = summarize_cells(cells)
    papers = summarize_papers(cells)
    associations = paper_associations(papers)
    source_formats = source_format_summary(cells)

    expected = {
        "Boldorini": (47, 9),
        "Biochar": (517, 204),
        "Hui": (36, 33),
        "Loladze": (605, 177),
        "Li2022": (172, 35),
    }
    for row in summary:
        code = next(key for key, value in DISPLAY.items() if value == row["dataset"])
        observed = (row["reference_cells"], row["matched_cells"])
        if observed != expected[code]:
            raise RuntimeError(
                f"Table 2 denominator mismatch for {code}: observed {observed}, "
                f"expected {expected[code]}"
            )

    write_csv(HERE / "coverage_structural_complexity_cells.csv", cells)
    write_csv(HERE / "coverage_structural_complexity_papers.csv", papers)
    write_csv(HERE / "coverage_structural_complexity_summary.csv", summary)
    write_csv(HERE / "coverage_structural_complexity_associations.csv", associations)
    write_csv(HERE / "coverage_source_format_summary.csv", source_formats)
    write_report(summary, associations)
    print((HERE / "coverage_structural_complexity_report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
