#!/usr/bin/env python3
"""
Universal LLM-based adjudication for Pipeline V2.

Replaces keyword-based adjudication (adjudicate_universal.py) with
Claude Opus 4.6 semantic adjudication.

Reads JSONL inputs from codex/outputs/universal_llm_inputs/{topic}/
and writes decisions to codex/outputs/llm_decisions/{topic}/.

Usage:
    python adjudicate_llm_universal.py [topic1 topic2 ...]
    python adjudicate_llm_universal.py --all
    python adjudicate_llm_universal.py legume_rotation --dry-run --max-rows 5

Environment:
    ANTHROPIC_API_KEY must be set (or in ../.env)
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

CODEX_ROOT = Path(__file__).resolve().parent
ROOT = CODEX_ROOT.parent
INPUT_ROOT = CODEX_ROOT / "outputs" / "universal_llm_inputs"
OUTPUT_ROOT = CODEX_ROOT / "outputs" / "llm_decisions"

ALL_TOPICS = [
    "organic_yield_gap",
    "notill_tillage",
    "mycorrhiza_yield",
    "legume_rotation",
    "biochar_crop_yield",
    "intercropping_yield",
]

# ── Prompt Construction ──────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are validating extracted meta-analysis rows after PDF extraction.
Your job is to decide whether each extracted row should be kept for synthesis, \
excluded, flagged for manual review, or treatment/control swapped.

You must use only:
- the topic configuration summary provided
- the extracted row fields provided

Do not invent topic-specific rules beyond what follows from the config.

## Decision Task

For each row, judge:
1. Does the treatment match the configured intervention?
2. Does the control match the configured comparator?
3. Does the outcome match the configured primary outcome?
4. Does the row match the benchmark estimand implied by the config?
5. Is there evidence that treatment and control were swapped?

## Decision Policy

- Choose "keep" if intervention, comparator, and outcome all match at least "partial".
- Choose "exclude" if intervention or comparator clearly does not match ("no").
- Choose "exclude" if the outcome is clearly outside the configured primary outcome ("no").
- Choose "flag" if the row is partly relevant but ambiguous.
- Choose "swap_treatment_control" if intervention/comparator are clearly reversed.
- Use estimand_match="no" when the row measures a different construct than the benchmark, \
even if the topic is similar.
- Be strict: it is better to exclude an ambiguous row than include an off-target one.

## COMMON EXTRACTION ERRORS TO CATCH

These three errors are the most frequent causes of inflated or deflated pooled effects in V1
extraction runs (observed across 3,460 rows, 6 topics). Apply these checks to every row:

**Error 1 — Yield component passed as yield:**
1000-grain weight, hectoliter weight, grain number per spike, grains per panicle, ear/spike
length, tiller number, number of fruits/pods (without weight) are YIELD COMPONENTS, not
harvestable yield per unit area. Set outcome_match="no", normalized_outcome_class="component_yield",
exclusion_reason="yield_component". This error appeared in all 6 V1 topics (>150 affected rows).

**Error 2 — Per-plant value without area conversion:**
If the outcome unit is "g/plant", "kg/plant", or "g/pot" and there is no area-normalised
equivalent available, the row cannot be directly pooled with per-hectare values. Set
normalized_outcome_class="biomass" and flag the row unless area conversion is possible.
This causes systematic upward bias because high-fertilization pot studies dominate.

**Error 3 — Confounded intervention (T/C isolation failure):**
If the treatment arm contains additional inputs (extra NPK, additional irrigation, seaweed
extract, amino acids, microbial inoculant) that the control arm does NOT receive, the
intervention effect cannot be attributed to the focal treatment alone. Set
intervention_match="no" or "partial", exclusion_reason="confounded_intervention". Look
carefully at treatment_description and control_description for asymmetric inputs.

## UNIVERSAL EXTRACTION ERROR PATTERNS — check every row for these

These checks apply to ALL topics regardless of config. They catch systematic extraction errors
that cause inflated or deflated pooled effects when contaminating rows are not removed.

**1. YIELD COMPONENTS ≠ YIELD**
Exclude rows where the outcome is a yield component or yield determinant, not total harvestable
yield per unit area:
- 1000-grain weight / thousand-grain weight / hectoliter weight / test weight
- Grains per spike / panicle / ear; pods per plant; number of fruits (without weight)
- Ear length / spike length / panicle length
Heuristic test: "Could this value be summed across plants to give kg/ha?" — if no, exclude.
Set outcome_match="no", normalized_outcome_class="component_yield",
exclusion_reason="yield_component".

**2. MORPHOLOGICAL TRAITS ≠ YIELD**
Exclude rows where the outcome is a plant growth or structural trait unrelated to harvest:
- Plant height, stem diameter/girth, number of leaves, branching, canopy spread
- Leaf area index (LAI), SPAD value, chlorophyll content/index
- Tiller number or count, root length, root biomass, root dry/fresh weight
Set outcome_match="no", normalized_outcome_class="biomass" (roots) or "other" (structural),
exclusion_reason="morphological_trait".

**3. QUALITY TRAITS ≠ YIELD**
Exclude rows where the outcome is a nutritional or quality measurement, UNLESS the topic
config explicitly lists quality metrics as primary outcomes:
- Nutrient concentration (N%, P%, K%, protein %), oil content %, starch content %
- Mineral element content, dry-matter percentage
Set outcome_match="no", normalized_outcome_class="quality_trait",
exclusion_reason="quality_trait_not_yield".

**4. CONFOUNDED INTERVENTION (T/C isolation failure)**
Flag any row where the treatment arm differs from the control arm in MORE THAN ONE
agronomic factor (e.g., no-till + cover crop vs conventional fallow; mycorrhiza + NPK vs
no-treatment control). The topic intervention must be the ONLY factor distinguishing T from C
for the effect to be attributable to that intervention. If additional inputs are asymmetric:
Set intervention_match="partial" or "no", exclusion_reason="confounded_intervention".
Use "flag" decision if ambiguous; "exclude" if the confounding is clear and severe.

**5. PER-PLANT vs PER-AREA — incompatible units**
Flag rows reporting yield per plant, per pot, or per container without an explicit conversion
factor to a per-area basis (e.g., plants/m², planting density given). Such rows cannot be
directly pooled in a synthesis where most results are in kg/ha or t/ha. They cause systematic
upward bias because high-input pot studies tend to report larger absolute per-plant values.
Set normalized_outcome_class="biomass", decision="flag",
exclusion_reason="per_plant_no_area_conversion" unless density is provided and conversion
is straightforward.

## Output Format

Return ONLY valid JSON matching this schema (no markdown, no commentary):

{
  "row_id": "string",
  "decision": "keep|exclude|flag|swap_treatment_control",
  "intervention_match": "yes|partial|no",
  "comparator_match": "yes|partial|no",
  "outcome_match": "yes|partial|no",
  "estimand_match": "yes|partial|no",
  "needs_tc_swap": false,
  "normalized_outcome_class": "grain_yield|harvest_yield|biomass|quality_trait|component_yield|system_productivity|other",
  "normalized_study_setting": "field|greenhouse|pot|mixed|unknown",
  "exclusion_reason": "string or null",
  "rationale_short": "1-2 sentences explaining the decision"
}
"""


def build_user_message(topic_brief: dict, row: dict, heuristic_flags: dict) -> str:
    """Build the user message for one row adjudication."""
    # Compact topic brief (only essential fields)
    brief_lines = [
        f"Topic: {topic_brief.get('title', 'Unknown')}",
        f"Research question: {topic_brief.get('research_question', '')}",
        f"Intervention: {topic_brief.get('intervention_description', '')}",
        f"Comparator: {topic_brief.get('comparator_description', '')}",
        f"Primary outcome: {topic_brief.get('outcome_description', '')}",
        f"Expected direction: {topic_brief.get('expected_direction', '')}",
        f"Benchmark: {topic_brief.get('benchmark_source', '')}",
        f"Benchmark notes: {topic_brief.get('benchmark_notes', '')}",
    ]
    if topic_brief.get("tc_confusion_warnings"):
        brief_lines.append("T/C confusion warnings:")
        for w in topic_brief["tc_confusion_warnings"][:5]:
            brief_lines.append(f"  - {w}")

    brief_text = "\n".join(brief_lines)

    # Compact row (only relevant fields)
    row_compact = {
        "row_id": row.get("row_id", ""),
        "title": row.get("title", ""),
        "outcome": row.get("outcome_variable", ""),
        "outcome_unit": row.get("outcome_unit", ""),
        "treatment_mean": row.get("treatment_mean"),
        "control_mean": row.get("control_mean"),
        "effect_pct": row.get("effect_pct"),
        "treatment_description": row.get("treatment_description", ""),
        "control_description": row.get("control_description", ""),
        "confidence": row.get("confidence", ""),
        "source_type": row.get("source_type", ""),
        "table_or_figure": row.get("table_or_figure", ""),
        "notes": row.get("notes", ""),
    }
    # Include moderators if present
    mods = row.get("moderators", {})
    if isinstance(mods, dict) and mods:
        row_compact["moderators"] = {
            k: v for k, v in mods.items()
            if v is not None and str(v).strip().lower() not in ("nan", "none", "")
        }

    row_json = json.dumps(row_compact, indent=2, ensure_ascii=False)

    # Heuristic flags as context
    flag_notes = []
    if heuristic_flags.get("missing_means"):
        flag_notes.append("WARNING: Missing treatment or control means")
    if heuristic_flags.get("nonpositive_means"):
        flag_notes.append("WARNING: Non-positive means (cannot compute lnRR)")
    if heuristic_flags.get("low_confidence"):
        flag_notes.append("NOTE: Extraction marked low confidence")
    flag_text = "\n".join(flag_notes) if flag_notes else "No heuristic flags."

    return f"""## Topic Configuration Summary

{brief_text}

## Extracted Row

```json
{row_json}
```

## Heuristic Flags

{flag_text}

Return your JSON decision now."""


# ── API Client ────────────────────────────────────────────────────────────

def get_api_key(provider: str = "anthropic") -> str:
    """Get API key from environment or .env file."""
    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "GOOGLE_API_KEY"
    key = os.environ.get(env_var)
    if key:
        return key

    # Try .env file
    env_path = ROOT.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{env_var}="):
                return line.split("=", 1)[1].strip().strip("'\"")

    raise RuntimeError(
        f"{env_var} not found. Set it in environment or ../.env"
    )


def call_llm(
    system: str,
    user_message: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Call LLM API (Claude or Gemini) and return the text response."""
    if provider == "google":
        return _call_gemini(system, user_message, api_key, model, max_tokens, max_retries)
    return _call_claude(system, user_message, api_key, model, max_tokens, max_retries)


def _call_claude(
    system: str,
    user_message: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Call Claude API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return response.content[0].text
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            print(f"    Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"    API error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


def _call_gemini(
    system: str,
    user_message: str,
    api_key: str,
    model: str = "gemini-2.5-flash",
    max_tokens: int = 1024,
    max_retries: int = 3,
) -> str:
    """Call Google Gemini API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=0.0,
                ),
            )
            return response.text
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = 2 ** attempt * 5
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            elif attempt < max_retries - 1:
                wait = 2 ** attempt * 2
                print(f"    API error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Max retries exceeded")


def parse_llm_response(text: str, row_id: str) -> dict:
    """Parse JSON from Claude response, with fallback for markdown wrapping."""
    text = text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        # Ensure row_id matches
        result["row_id"] = row_id
        return result
    except json.JSONDecodeError as e:
        return {
            "row_id": row_id,
            "decision": "flag",
            "intervention_match": "unknown",
            "comparator_match": "unknown",
            "outcome_match": "unknown",
            "estimand_match": "unknown",
            "needs_tc_swap": False,
            "normalized_outcome_class": "other",
            "normalized_study_setting": "unknown",
            "exclusion_reason": None,
            "rationale_short": f"LLM response parse error: {e}",
            "_parse_error": True,
            "_raw_response": text[:500],
        }


# ── Batch Processing ──────────────────────────────────────────────────────

def process_row(
    topic_brief: dict,
    row: dict,
    heuristic_flags: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    dry_run: bool = False,
) -> dict:
    """Adjudicate one row using Claude."""
    row_id = row.get("row_id", "unknown")

    # Stage A: Hard checks (deterministic, no LLM needed)
    t_mean = row.get("treatment_mean")
    c_mean = row.get("control_mean")

    if t_mean is None or c_mean is None:
        return {
            "row_id": row_id,
            "decision": "exclude",
            "intervention_match": "unknown",
            "comparator_match": "unknown",
            "outcome_match": "unknown",
            "estimand_match": "unknown",
            "needs_tc_swap": False,
            "normalized_outcome_class": "other",
            "normalized_study_setting": "unknown",
            "exclusion_reason": "missing_means",
            "rationale_short": "Treatment or control mean is missing.",
            "_stage": "hard_check",
        }

    try:
        t_mean_f = float(t_mean)
        c_mean_f = float(c_mean)
    except (TypeError, ValueError):
        return {
            "row_id": row_id,
            "decision": "exclude",
            "intervention_match": "unknown",
            "comparator_match": "unknown",
            "outcome_match": "unknown",
            "estimand_match": "unknown",
            "needs_tc_swap": False,
            "normalized_outcome_class": "other",
            "normalized_study_setting": "unknown",
            "exclusion_reason": "non_numeric_means",
            "rationale_short": "Treatment or control mean is not numeric.",
            "_stage": "hard_check",
        }

    if t_mean_f <= 0 and c_mean_f <= 0:
        return {
            "row_id": row_id,
            "decision": "exclude",
            "intervention_match": "unknown",
            "comparator_match": "unknown",
            "outcome_match": "unknown",
            "estimand_match": "unknown",
            "needs_tc_swap": False,
            "normalized_outcome_class": "other",
            "normalized_study_setting": "unknown",
            "exclusion_reason": "both_means_nonpositive",
            "rationale_short": "Both means are non-positive; cannot compute lnRR.",
            "_stage": "hard_check",
        }

    # Stage C: LLM adjudication
    if dry_run:
        return {
            "row_id": row_id,
            "decision": "dry_run",
            "rationale_short": "Dry run mode - no API call made.",
            "_stage": "dry_run",
        }

    user_msg = build_user_message(topic_brief, row, heuristic_flags)
    raw_response = call_llm(
        SYSTEM_PROMPT, user_msg, api_key,
        model=model, provider=provider,
    )
    result = parse_llm_response(raw_response, row_id)
    result["_stage"] = "llm"

    return result


def process_topic(
    topic: str,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    dry_run: bool = False,
    max_rows: int | None = None,
    batch_size: int = 10,
) -> dict:
    """Process all rows for a topic."""
    input_jsonl = INPUT_ROOT / topic / "llm_review_inputs.jsonl"
    if not input_jsonl.exists():
        print(f"  [SKIP] No input JSONL for {topic}")
        return {"topic": topic, "error": "no_input_jsonl"}

    out_dir = OUTPUT_ROOT / topic
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read all rows
    rows_data = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows_data.append(json.loads(line))

    if max_rows:
        rows_data = rows_data[:max_rows]

    total = len(rows_data)
    print(f"  Processing {total} rows for {topic}...")

    decisions = []
    decision_counts = Counter()
    parse_errors = 0

    decisions_jsonl = out_dir / "decisions.jsonl"
    with decisions_jsonl.open("w", encoding="utf-8") as f_out:
        for i, item in enumerate(rows_data):
            topic_brief = item["topic_brief"]
            row = item["row"]
            heuristic_flags = item.get("heuristic_flags", {})

            row_id = row.get("row_id", f"row_{i}")

            try:
                result = process_row(
                    topic_brief, row, heuristic_flags,
                    api_key, model=model, provider=provider,
                    dry_run=dry_run,
                )
            except Exception as e:
                result = {
                    "row_id": row_id,
                    "decision": "flag",
                    "rationale_short": f"Processing error: {e}",
                    "_stage": "error",
                }

            decisions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            decision_counts[result.get("decision", "unknown")] += 1
            if result.get("_parse_error"):
                parse_errors += 1

            # Progress
            if (i + 1) % batch_size == 0 or (i + 1) == total:
                pct = (i + 1) / total * 100
                print(f"    [{i+1}/{total}] ({pct:.0f}%) "
                      f"keep={decision_counts.get('keep', 0)} "
                      f"exclude={decision_counts.get('exclude', 0)} "
                      f"flag={decision_counts.get('flag', 0)}")

            # Rate limit courtesy: small delay between API calls
            if not dry_run and result.get("_stage") == "llm":
                time.sleep(0.3)

    # Write kept rows CSV
    kept = [d for d in decisions if d.get("decision") == "keep"]
    _write_kept_csv(out_dir, rows_data, kept)

    # Write summary
    summary = {
        "topic": topic,
        "total_rows": total,
        "decision_counts": dict(decision_counts),
        "parse_errors": parse_errors,
        "kept_count": len(kept),
        "keep_rate": round(len(kept) / total * 100, 1) if total > 0 else 0,
        "model": model,
        "dry_run": dry_run,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Write summary markdown
    _write_summary_md(out_dir, summary, decisions)

    return summary


def _write_kept_csv(out_dir: Path, rows_data: list, kept_decisions: list):
    """Write kept rows to CSV for downstream synthesis."""
    import csv

    kept_ids = {d["row_id"] for d in kept_decisions}

    # Find the original rows that were kept
    kept_rows = []
    for item in rows_data:
        row = item["row"]
        if row.get("row_id") in kept_ids:
            kept_rows.append(row)

    if not kept_rows:
        return

    # Determine columns from first row
    all_keys = set()
    for row in kept_rows:
        all_keys.update(row.keys())
    # Remove moderators dict (flatten later if needed)
    all_keys.discard("moderators")
    columns = sorted(all_keys)

    csv_path = out_dir / "llm_kept_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in kept_rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _write_summary_md(out_dir: Path, summary: dict, decisions: list):
    """Write a human-readable summary."""
    lines = [
        f"# {summary['topic']} -- LLM Adjudication Summary",
        "",
        f"Model: {summary['model']}",
        f"Total rows: {summary['total_rows']}",
        f"Parse errors: {summary['parse_errors']}",
        "",
        "## Decision Counts",
        "",
    ]
    for dec, cnt in sorted(summary["decision_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"- {dec}: {cnt}")

    lines.extend([
        "",
        f"**Keep rate: {summary['keep_rate']}%**",
        "",
        "## Exclusion Reasons",
        "",
    ])

    exclusion_reasons = Counter()
    for d in decisions:
        if d.get("decision") == "exclude":
            reason = d.get("exclusion_reason") or d.get("rationale_short", "unspecified")
            exclusion_reasons[reason[:80]] += 1

    for reason, cnt in exclusion_reasons.most_common(20):
        lines.append(f"- ({cnt}) {reason}")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based universal adjudication")
    parser.add_argument("topics", nargs="*", help="Topics to process")
    parser.add_argument("--all", action="store_true", help="Process all topics")
    parser.add_argument("--dry-run", action="store_true", help="Skip API calls")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows per topic")
    parser.add_argument("--model", default=None,
                        help="Model to use (default: auto-select based on provider)")
    parser.add_argument("--provider", default="google", choices=["anthropic", "google"],
                        help="LLM provider (default: google)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Progress reporting interval")
    args = parser.parse_args()

    topics = ALL_TOPICS if args.all else (args.topics or ALL_TOPICS)

    # Auto-select model based on provider if not specified
    if args.model is None:
        if args.provider == "google":
            args.model = "gemini-2.5-flash"
        else:
            args.model = "claude-sonnet-4-20250514"

    if not args.dry_run:
        api_key = get_api_key(args.provider)
    else:
        api_key = "dry-run-no-key-needed"

    print(f"Provider: {args.provider}, Model: {args.model}")

    all_summaries = {}
    for topic in topics:
        print(f"\n{'='*60}")
        print(f"  LLM Adjudication: {topic}")
        print(f"{'='*60}")

        summary = process_topic(
            topic,
            api_key=api_key,
            model=args.model,
            provider=args.provider,
            dry_run=args.dry_run,
            max_rows=args.max_rows,
            batch_size=args.batch_size,
        )
        all_summaries[topic] = summary

        if "error" not in summary:
            print(f"  -> {summary['kept_count']}/{summary['total_rows']} kept "
                  f"({summary['keep_rate']}%)")
            print(f"  -> Decisions: {summary['decision_counts']}")

    # Combined summary
    combined_path = OUTPUT_ROOT / "llm_adjudication_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_path.write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print("  ALL TOPICS COMPLETE")
    print(f"{'='*60}")

    # Print combined summary table
    print(f"\n{'Topic':<25} {'Total':>6} {'Kept':>6} {'Rate':>6} {'Excluded':>8}")
    print("-" * 55)
    for topic, s in all_summaries.items():
        if "error" in s:
            print(f"{topic:<25} {'ERROR':>6}")
        else:
            print(f"{topic:<25} {s['total_rows']:>6} {s['kept_count']:>6} "
                  f"{s['keep_rate']:>5.1f}% {s['decision_counts'].get('exclude', 0):>8}")


if __name__ == "__main__":
    main()
