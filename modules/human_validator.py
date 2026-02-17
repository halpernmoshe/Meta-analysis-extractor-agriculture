"""
Human Validation Module

Interactive validation workflow for meta-analysis data extraction.
Designed for ~4 minute reviews per paper with human-in-the-loop verification.

Features:
1. Paper summary (200 words) with experimental design
2. Extracted data display with auto-flagged issues
3. Human-selected verification items
4. PDF page navigation for verification
5. Completeness checks (tables, outcomes, variance)
6. Validation tracking and logging
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
import re
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import sys
import os
import subprocess
import platform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm import LLMClient


# ============================================================================
# PROMPTS
# ============================================================================

SUMMARY_PROMPT = """Read this scientific paper and provide a concise summary for a meta-analysis validator.

PAPER TEXT:
{paper_text}

Write a summary (exactly 150-200 words) with this structure:

**STUDY OVERVIEW** (2-3 sentences)
What was studied, where, and why.

**EXPERIMENTAL DESIGN** (3-4 sentences)
- Study type (field trial, greenhouse, etc.)
- Design (RCBD, split-plot, factorial, etc.)
- Number of replicates
- Treatments and control
- Duration/seasons

**KEY OUTCOMES** (2-3 sentences)
What was measured and main findings.

**DATA SOURCES**
List all tables and figures with data: "Table 1: [description], Table 2: [description], ..."

Return ONLY the summary text, no JSON.
"""

TABLE_INVENTORY_PROMPT = """List ALL tables and figures in this paper.

PAPER TEXT:
{paper_text}

Return JSON:
{{
    "tables": [
        {{"id": "Table 1", "title": "table caption", "page": page_number, "has_outcome_data": true/false}},
        ...
    ],
    "figures": [
        {{"id": "Figure 1", "title": "figure caption", "page": page_number, "has_outcome_data": true/false}},
        ...
    ],
    "methods_info": {{
        "design": "RCBD/split-plot/factorial/etc",
        "replicates": number or null,
        "sample_size_text": "quote about sample size if found"
    }}
}}

Return ONLY the JSON.
"""


# ============================================================================
# PDF UTILITIES
# ============================================================================

def extract_full_text(pdf_path: str, max_pages: int = 30) -> str:
    """Extract full text from PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(max_pages, len(doc))):
            text += f"\n--- Page {i+1} ---\n"
            text += doc[i].get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


def find_page_for_table(pdf_path: str, table_id: str) -> Tuple[int, float]:
    """
    Find the page number containing a table.
    Returns (page_number, confidence).
    """
    try:
        doc = fitz.open(pdf_path)

        # Search patterns
        patterns = [
            table_id,  # "Table 3"
            table_id.replace(" ", ""),  # "Table3"
            table_id.lower(),  # "table 3"
        ]

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().lower()

            for pattern in patterns:
                if pattern.lower() in text:
                    # Check if this looks like the actual table (not just a reference)
                    # Tables usually have their caption near the table
                    if f"{pattern.lower()}" in text and (
                        "mean" in text or "treatment" in text or
                        "control" in text or "±" in text or
                        "sd" in text or "se" in text
                    ):
                        doc.close()
                        return page_num + 1, 0.9  # High confidence

        # Second pass: just find any mention
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().lower()
            if patterns[0].lower() in text:
                doc.close()
                return page_num + 1, 0.6  # Medium confidence

        doc.close()
        return 1, 0.1  # Low confidence, default to page 1

    except Exception as e:
        print(f"Error finding page: {e}")
        return 1, 0.0


def find_page_for_value(pdf_path: str, value: float, context: str = "") -> Tuple[int, float]:
    """
    Find page containing a specific numeric value.
    Returns (page_number, confidence).
    """
    try:
        doc = fitz.open(pdf_path)
        value_str = f"{value:.1f}"
        value_str_alt = f"{value:.2f}"

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            if value_str in text or value_str_alt in text:
                # Bonus confidence if context also found
                confidence = 0.7
                if context and context.lower() in text.lower():
                    confidence = 0.9
                doc.close()
                return page_num + 1, confidence

        doc.close()
        return 1, 0.1

    except Exception as e:
        return 1, 0.0


def open_pdf_to_page(pdf_path: str, page_num: int):
    """Open PDF to specific page using system default viewer."""
    try:
        if platform.system() == 'Windows':
            # On Windows, just open the PDF - page navigation varies by reader
            # Adobe Acrobat supports /A "page=N" but not all readers do
            os.startfile(pdf_path)
            print(f"   [Navigate to page {page_num} in the PDF viewer]")
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', '-a', 'Preview', pdf_path])
            print(f"   [Navigate to page {page_num} in Preview]")
        else:  # Linux
            subprocess.run(['xdg-open', pdf_path])
            print(f"   [Navigate to page {page_num} in the PDF viewer]")
    except Exception as e:
        print(f"Could not open PDF: {e}")
        print(f"Please manually open: {pdf_path} (page {page_num})")


def count_tables_in_pdf(pdf_path: str) -> List[Dict]:
    """Count and list tables found in PDF."""
    try:
        doc = fitz.open(pdf_path)
        tables = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()

            # Find table references
            matches = re.findall(r'Table\s*(\d+)[.\s:]*([^\n]{0,100})', text, re.IGNORECASE)
            for match in matches:
                table_id = f"Table {match[0]}"
                caption = match[1].strip()[:80]

                # Check if we already have this table
                existing = [t for t in tables if t['id'] == table_id]
                if not existing:
                    tables.append({
                        'id': table_id,
                        'caption': caption,
                        'page': page_num + 1
                    })

        doc.close()
        return tables

    except Exception as e:
        print(f"Error counting tables: {e}")
        return []


# ============================================================================
# VALIDATION LOGIC
# ============================================================================

def generate_paper_summary(paper_text: str, llm_client: LLMClient) -> str:
    """Generate 200-word summary of paper."""
    prompt = SUMMARY_PROMPT.format(paper_text=paper_text[:60000])

    try:
        response = llm_client.call(
            prompt=prompt,
            max_tokens=500,
            temperature=0.3
        )
        return response.strip()
    except Exception as e:
        return f"Error generating summary: {e}"


def get_table_inventory(paper_text: str, llm_client: LLMClient) -> Dict:
    """Get inventory of all tables/figures in paper."""
    prompt = TABLE_INVENTORY_PROMPT.format(paper_text=paper_text[:60000])

    try:
        response = llm_client.call(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.1
        )

        # Parse JSON
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```json?\s*', '', response)
            response = re.sub(r'\s*```$', '', response)

        return json.loads(response)
    except Exception as e:
        return {"tables": [], "figures": [], "error": str(e)}


def detect_issues(paper_data: pd.DataFrame) -> List[Dict]:
    """Auto-detect potential issues in extracted data."""
    issues = []

    for idx, row in paper_data.iterrows():
        obs_id = row.get('observation_id', idx)

        # Issue 1: Treatment < Control for positive outcomes (yield, biomass)
        positive_outcomes = ['GWAD', 'CWAD', 'LAID', 'SWAD', 'HWAD', 'TWAD']
        if row.get('outcome_variable') in positive_outcomes:
            if pd.notna(row.get('treatment_mean')) and pd.notna(row.get('control_mean')):
                if row['treatment_mean'] < row['control_mean']:
                    pct_diff = ((row['control_mean'] - row['treatment_mean']) / row['control_mean']) * 100
                    issues.append({
                        'observation_id': obs_id,
                        'type': 'direction',
                        'severity': 'warning' if pct_diff < 20 else 'critical',
                        'message': f"Treatment ({row['treatment_mean']:.1f}) < Control ({row['control_mean']:.1f}) for {row['outcome_variable']}",
                        'row_index': idx
                    })

        # Issue 2: Missing variance
        if pd.isna(row.get('treatment_variance')) and pd.isna(row.get('control_variance')) and pd.isna(row.get('pooled_variance')):
            issues.append({
                'observation_id': obs_id,
                'type': 'missing_variance',
                'severity': 'info',
                'message': f"No variance data for {row.get('outcome_variable', 'unknown')}",
                'row_index': idx
            })

        # Issue 3: Extreme effect sizes
        if pd.notna(row.get('treatment_mean')) and pd.notna(row.get('control_mean')):
            if row['control_mean'] > 0:
                ratio = row['treatment_mean'] / row['control_mean']
                if ratio > 3 or ratio < 0.33:
                    issues.append({
                        'observation_id': obs_id,
                        'type': 'extreme_effect',
                        'severity': 'warning',
                        'message': f"Extreme ratio ({ratio:.2f}x) for {row.get('outcome_variable', 'unknown')}",
                        'row_index': idx
                    })

        # Issue 4: Missing sample size
        if pd.isna(row.get('treatment_n')) and pd.isna(row.get('control_n')):
            issues.append({
                'observation_id': obs_id,
                'type': 'missing_n',
                'severity': 'info',
                'message': f"No sample size for {row.get('outcome_variable', 'unknown')}",
                'row_index': idx
            })

    return issues


def check_table_coverage(extracted_tables: List[str], pdf_tables: List[Dict]) -> Dict:
    """Check which tables were extracted vs available."""
    extracted_set = set(extracted_tables)
    pdf_table_ids = [t['id'] for t in pdf_tables]

    covered = [t for t in pdf_table_ids if t in extracted_set]
    missed = [t for t in pdf_tables if t['id'] not in extracted_set]

    return {
        'total_in_pdf': len(pdf_tables),
        'tables_extracted_from': len(covered),
        'coverage_pct': len(covered) / len(pdf_tables) * 100 if pdf_tables else 0,
        'missed_tables': missed
    }


def check_outcome_coverage(extracted_outcomes: List[str], abstract_outcomes: List[str]) -> Dict:
    """Check which outcomes from abstract were captured."""
    extracted_set = set(o.lower() for o in extracted_outcomes)

    found = []
    missed = []

    for outcome in abstract_outcomes:
        outcome_lower = outcome.lower()
        # Check if any extracted outcome contains this keyword
        if any(outcome_lower in ext or ext in outcome_lower for ext in extracted_set):
            found.append(outcome)
        else:
            missed.append(outcome)

    return {
        'outcomes_in_abstract': len(abstract_outcomes),
        'outcomes_captured': len(found),
        'found': found,
        'missed': missed
    }


def check_variance_coverage(paper_data: pd.DataFrame) -> Dict:
    """Check variance data coverage."""
    total = len(paper_data)

    has_variance = paper_data.apply(
        lambda row: pd.notna(row.get('treatment_variance')) or
                    pd.notna(row.get('control_variance')) or
                    pd.notna(row.get('pooled_variance')),
        axis=1
    ).sum()

    missing_rows = paper_data[
        paper_data.apply(
            lambda row: pd.isna(row.get('treatment_variance')) and
                        pd.isna(row.get('control_variance')) and
                        pd.isna(row.get('pooled_variance')),
            axis=1
        )
    ]

    return {
        'total_observations': total,
        'with_variance': has_variance,
        'coverage_pct': has_variance / total * 100 if total > 0 else 0,
        'missing_indices': list(missing_rows.index)
    }


# ============================================================================
# INTERACTIVE VALIDATION
# ============================================================================

def display_header(text: str, char: str = "="):
    """Display a formatted header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def display_data_table(paper_data: pd.DataFrame, issues: List[Dict]):
    """Display extracted data in a readable format with simple row numbers."""
    issue_rows = {i['row_index'] for i in issues if i['severity'] in ['warning', 'critical']}

    print(f"\n{'#':<4} {'Outcome':<10} {'Treatment':<20} {'Trt Mean':>9} {'Ctrl Mean':>10} {'Variance':>10} {'Source':<10} {'Flag':<4}")
    print("-" * 85)

    for i, (idx, row) in enumerate(paper_data.iterrows()):
        flag = "[!]" if idx in issue_rows else ""
        # Use simple sequential number for display
        row_num = i + 1
        outcome = str(row.get('outcome_variable', ''))[:10]
        treatment = str(row.get('treatment_description', ''))[:20]
        trt_mean = f"{row['treatment_mean']:.1f}" if pd.notna(row.get('treatment_mean')) else "N/A"
        ctrl_mean = f"{row['control_mean']:.1f}" if pd.notna(row.get('control_mean')) else "N/A"
        source = str(row.get('data_source', ''))[:10]

        # Get variance (pooled, treatment, or control - whichever exists)
        var_val = None
        for var_col in ['pooled_variance', 'treatment_variance', 'control_variance']:
            if pd.notna(row.get(var_col)):
                var_val = row[var_col]
                break
        variance = f"{var_val:.2f}" if var_val is not None else "-"

        print(f"{row_num:<4} {outcome:<10} {treatment:<20} {trt_mean:>9} {ctrl_mean:>10} {variance:>10} {source:<10} {flag:<4}")


def display_issues(issues: List[Dict]):
    """Display auto-detected issues."""
    critical = [i for i in issues if i['severity'] == 'critical']
    warnings = [i for i in issues if i['severity'] == 'warning']

    if critical:
        print(f"\n[!!] CRITICAL ISSUES ({len(critical)}):")
        for issue in critical:
            print(f"   Row {issue['observation_id']}: {issue['message']}")

    if warnings:
        print(f"\n[!]  WARNINGS ({len(warnings)}):")
        for issue in warnings[:5]:  # Show first 5
            print(f"   Row {issue['observation_id']}: {issue['message']}")
        if len(warnings) > 5:
            print(f"   ... and {len(warnings) - 5} more")


def get_user_selection(paper_data: pd.DataFrame, issues: List[Dict]) -> Tuple[List[int], str]:
    """Get user's selection of items to verify using simple 1-based row numbers.

    Returns: (list of indices, selection_method)
    selection_method is one of: 'auto', 'all', 'manual', 'skipped'
    """
    print("\n" + "-" * 70)
    print("Which rows to verify? (enter numbers like: 1,4,7)")
    print("  'auto' = flagged items only | 'all' = full review | 'q' = skip paper")

    # Map simple row numbers (1-based) to actual DataFrame indices
    index_map = {i+1: idx for i, idx in enumerate(paper_data.index)}
    max_row = len(paper_data)

    while True:
        selection = input("> ").strip().lower()

        if selection == 'q':
            return [], 'skipped'
        elif selection == 'all':
            return list(paper_data.index), 'all'
        elif selection == 'auto':
            flagged = [i['row_index'] for i in issues if i['severity'] in ['warning', 'critical']]
            if flagged:
                return flagged[:5], 'auto'
            else:
                # No flagged items - pick first 3
                return list(paper_data.index[:3]), 'auto'
        else:
            try:
                # Parse comma-separated numbers (1-based)
                row_nums = [int(x.strip()) for x in selection.split(',')]
                # Convert to actual indices
                valid_indices = []
                invalid_nums = []
                for num in row_nums:
                    if 1 <= num <= max_row:
                        valid_indices.append(index_map[num])
                    else:
                        invalid_nums.append(num)

                if valid_indices:
                    if invalid_nums:
                        print(f"Note: Skipped invalid row numbers: {invalid_nums}")
                    return valid_indices, 'manual'
                else:
                    print(f"Invalid. Enter numbers between 1 and {max_row}.")
            except ValueError:
                print("Invalid input. Enter numbers like '1,4,7' or 'auto'/'all'/'q'.")


def verify_single_item(row: pd.Series, pdf_path: str, row_idx: int, row_num: int = None) -> Dict:
    """Interactive verification of a single observation."""
    display_num = row_num if row_num else row_idx
    result = {
        'observation_id': row.get('observation_id', row_idx),
        'outcome': row.get('outcome_variable'),
        'verified_fields': {},
        'corrections': {},
        'notes': ''
    }

    # Find and open PDF page
    data_source = row.get('data_source', '')
    if 'Table' in str(data_source):
        page_num, confidence = find_page_for_table(pdf_path, data_source)
    else:
        page_num, confidence = find_page_for_value(
            pdf_path,
            row.get('treatment_mean', 0),
            str(row.get('treatment_description', ''))
        )

    outcome = row.get('outcome_variable', 'Unknown')
    print(f"\n{'=' * 60}")
    print(f"  VERIFYING ROW {display_num}: {outcome}")
    print(f"  Source: {data_source} (PDF page {page_num}, {confidence:.0%} confidence)")
    print(f"{'=' * 60}")

    open_pdf_to_page(pdf_path, page_num)

    print(f"\n  EXTRACTED VALUES:")
    print(f"  Treatment: {row.get('treatment_description', 'N/A')[:40]}")
    print(f"    Mean: {row.get('treatment_mean', 'N/A')}")
    print(f"  Control: {row.get('control_description', 'N/A')[:40]}")
    print(f"    Mean: {row.get('control_mean', 'N/A')}")
    if pd.notna(row.get('treatment_variance')) or pd.notna(row.get('control_variance')):
        var_type = row.get('variance_type', '?')
        print(f"  Variance ({var_type}): Trt={row.get('treatment_variance', 'N/A')}, Ctrl={row.get('control_variance', 'N/A')}")

    print(f"\n[If wrong page, navigate to {data_source} in PDF]")

    # Verify treatment mean
    print(f"\n1. Is TREATMENT mean ({row.get('treatment_mean', 'N/A')}) correct?")
    resp = input("   [Y]es / [N]o / [S]kip > ").strip().lower()
    if resp == 'y':
        result['verified_fields']['treatment_mean'] = True
    elif resp == 'n':
        result['verified_fields']['treatment_mean'] = False
        correct = input("   Enter correct value: ").strip()
        if correct:
            try:
                result['corrections']['treatment_mean'] = float(correct)
            except ValueError:
                pass

    # Verify control mean
    print(f"\n2. Is CONTROL mean ({row.get('control_mean', 'N/A')}) correct?")
    resp = input("   [Y]es / [N]o / [S]kip > ").strip().lower()
    if resp == 'y':
        result['verified_fields']['control_mean'] = True
    elif resp == 'n':
        result['verified_fields']['control_mean'] = False
        correct = input("   Enter correct value: ").strip()
        if correct:
            try:
                result['corrections']['control_mean'] = float(correct)
            except ValueError:
                pass

    # Verify variance if present
    if pd.notna(row.get('treatment_variance')) or pd.notna(row.get('control_variance')):
        var_val = row.get('treatment_variance') or row.get('control_variance')
        print(f"\n3. Is VARIANCE ({var_val}) correct? (type: {row.get('variance_type', 'unknown')})")
        resp = input("   [Y]es / [N]o / [M]issing in paper / [S]kip > ").strip().lower()
        if resp == 'y':
            result['verified_fields']['variance'] = True
        elif resp == 'n':
            result['verified_fields']['variance'] = False
            correct = input("   Enter correct value: ").strip()
            if correct:
                try:
                    result['corrections']['variance'] = float(correct)
                except ValueError:
                    pass
        elif resp == 'm':
            result['verified_fields']['variance'] = 'not_in_paper'

    # Any notes
    notes = input("\nAny notes? (press Enter to skip) > ").strip()
    if notes:
        result['notes'] = notes

    return result


def run_completeness_checks(
    paper_data: pd.DataFrame,
    pdf_path: str,
    paper_text: str,
    llm_client: Optional[LLMClient] = None
) -> Dict:
    """Run completeness checks with human input."""

    completeness = {
        'table_coverage': {},
        'outcome_coverage': {},
        'variance_coverage': {},
        'human_flags': []
    }

    display_header("COMPLETENESS CHECKS", "-")

    # 1. Table coverage check
    print("\n[TABLE] TABLE COVERAGE CHECK")
    pdf_tables = count_tables_in_pdf(pdf_path)
    extracted_sources = paper_data['data_source'].dropna().unique().tolist()
    extracted_tables = [s for s in extracted_sources if 'Table' in str(s)]

    coverage = check_table_coverage(extracted_tables, pdf_tables)
    completeness['table_coverage'] = coverage

    print(f"   Tables in PDF: {coverage['total_in_pdf']}")
    print(f"   Tables extracted from: {coverage['tables_extracted_from']}")

    if coverage['missed_tables']:
        print(f"\n   [!]  Unvisited tables:")
        for table in coverage['missed_tables'][:5]:
            print(f"      - {table['id']}: {table.get('caption', '')[:50]}")

        print("\n   Any of these contain relevant outcome data?")
        resp = input("   [Y]es / [N]o / [?]Unsure > ").strip().lower()
        if resp == 'y':
            which = input("   Which tables? (e.g., 'Table 3, Table 5') > ").strip()
            completeness['human_flags'].append({
                'type': 'missed_table',
                'tables': which,
                'action': 're-extract'
            })

    # 2. Variance coverage check
    print("\n[VARIANCE] VARIANCE COVERAGE CHECK")
    var_coverage = check_variance_coverage(paper_data)
    completeness['variance_coverage'] = var_coverage

    print(f"   Observations with variance: {var_coverage['with_variance']}/{var_coverage['total_observations']} ({var_coverage['coverage_pct']:.0f}%)")

    if var_coverage['coverage_pct'] < 100:
        print(f"\n   Missing variance for {var_coverage['total_observations'] - var_coverage['with_variance']} observations.")
        print("   Quick check: Does the paper have variance data (±, SE, SD, LSD) that we missed?")
        resp = input("   [Y]es, missed / [N]o, not in paper / [?]Unsure > ").strip().lower()
        if resp == 'y':
            completeness['human_flags'].append({
                'type': 'missed_variance',
                'action': 'variance_rescue'
            })

    # 3. Sample size check
    print("\n[N] SAMPLE SIZE CHECK")
    has_n = paper_data['treatment_n'].notna().sum() + paper_data['control_n'].notna().sum()
    total_possible = len(paper_data) * 2
    n_coverage = has_n / total_possible * 100 if total_possible > 0 else 0

    print(f"   Sample size coverage: {n_coverage:.0f}%")

    if n_coverage < 50:
        print("   Is sample size (n, replicates) reported in the Methods section?")
        resp = input("   [Y]es / [N]o > ").strip().lower()
        if resp == 'y':
            n_val = input("   What is n? > ").strip()
            if n_val:
                completeness['human_flags'].append({
                    'type': 'missing_n',
                    'value': n_val,
                    'action': 'add_n'
                })

    return completeness


# ============================================================================
# MAIN VALIDATION WORKFLOW
# ============================================================================

class ValidationSession:
    """Manages a validation session with logging."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / 'validation_log.json'
        self.log = self._load_log()

    def _load_log(self) -> Dict:
        if self.log_path.exists():
            with open(self.log_path) as f:
                return json.load(f)
        return {
            'sessions': [],
            'papers_validated': {},
            'summary_stats': {
                'total_papers': 0,
                'total_items_verified': 0,
                'items_correct': 0,
                'items_corrected': 0,
                'common_errors': {}
            }
        }

    def save_log(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)

    def log_paper_validation(self, paper_id: str, result: Dict):
        """Log validation result for a paper."""
        self.log['papers_validated'][paper_id] = result

        # Update summary stats
        self.log['summary_stats']['total_papers'] += 1

        for item in result.get('verified_items', []):
            # Count each FIELD verified, not each item (fixes 300% bug)
            for field, correct in item.get('verified_fields', {}).items():
                self.log['summary_stats']['total_items_verified'] += 1

                if correct == True:
                    self.log['summary_stats']['items_correct'] += 1
                elif correct == False:
                    self.log['summary_stats']['items_corrected'] += 1

                    # Track error type
                    error_type = f"{field}_error"
                    self.log['summary_stats']['common_errors'][error_type] = \
                        self.log['summary_stats']['common_errors'].get(error_type, 0) + 1

        self.save_log()

    def get_papers_to_validate(self, all_papers: List[str]) -> List[str]:
        """Get papers that haven't been validated yet."""
        validated = set(self.log['papers_validated'].keys())
        return [p for p in all_papers if p not in validated]

    def print_summary(self):
        """Print validation summary statistics."""
        stats = self.log['summary_stats']

        display_header("VALIDATION SUMMARY")
        print(f"Papers validated: {stats['total_papers']}")
        print(f"Fields verified: {stats['total_items_verified']}")

        if stats['total_items_verified'] > 0:
            accuracy = stats['items_correct'] / stats['total_items_verified'] * 100
            print(f"Accuracy rate: {accuracy:.1f}%")

        # Show selection method breakdown
        selection_counts = {}
        for paper_id, paper_result in self.log['papers_validated'].items():
            method = paper_result.get('selection_method', 'unknown')
            selection_counts[method] = selection_counts.get(method, 0) + 1

        if selection_counts:
            print(f"\nSelection methods used:")
            for method, count in sorted(selection_counts.items()):
                print(f"  - {method}: {count} papers")

        if stats['common_errors']:
            print(f"\nCommon errors:")
            for error, count in sorted(stats['common_errors'].items(), key=lambda x: -x[1]):
                print(f"  - {error}: {count}")


def validate_paper_interactive(
    paper_id: str,
    pdf_path: str,
    paper_data: pd.DataFrame,
    llm_client: LLMClient,
    session: ValidationSession
) -> Dict:
    """
    Run interactive validation for a single paper.
    Returns validation result dict.
    """

    start_time = datetime.now()

    result = {
        'paper_id': paper_id,
        'validation_date': start_time.isoformat(),
        'validator': os.environ.get('USER', os.environ.get('USERNAME', 'unknown')),
        'selection_method': 'unknown',  # 'auto', 'all', 'manual', 'skipped'
        'verified_items': [],
        'completeness': {},
        'overall_status': 'unknown',
        'notes': ''
    }

    # Extract full text
    print(f"\nLoading paper: {paper_id}...")
    paper_text = extract_full_text(pdf_path)

    if not paper_text:
        print("Error: Could not extract text from PDF")
        result['overall_status'] = 'error'
        return result

    # Generate summary
    display_header(f"VALIDATING: {paper_id}")
    print("\n[PAPER] PAPER SUMMARY")
    print("-" * 70)
    summary = generate_paper_summary(paper_text, llm_client)
    print(summary)

    # Display extracted data
    display_header("EXTRACTED DATA", "-")
    issues = detect_issues(paper_data)
    display_data_table(paper_data, issues)
    display_issues(issues)

    # Get user selection
    selected, selection_method = get_user_selection(paper_data, issues)
    result['selection_method'] = selection_method

    if not selected:
        result['overall_status'] = 'skipped'
        return result

    # Build index to row number mapping
    index_to_rownum = {idx: i+1 for i, idx in enumerate(paper_data.index)}

    # Verify selected items
    for idx in selected:
        if idx in paper_data.index:
            row = paper_data.loc[idx]
            row_num = index_to_rownum.get(idx, idx)
        else:
            # Try to find by observation_id
            matches = paper_data[paper_data['observation_id'] == idx]
            if len(matches) > 0:
                row = matches.iloc[0]
                idx = matches.index[0]
                row_num = index_to_rownum.get(idx, idx)
            else:
                continue

        item_result = verify_single_item(row, pdf_path, idx, row_num)
        result['verified_items'].append(item_result)

    # Run completeness checks
    result['completeness'] = run_completeness_checks(
        paper_data, pdf_path, paper_text, llm_client
    )

    # Final decision
    display_header("FINAL DECISION", "-")
    print("\nOverall assessment:")
    print("  [A] Approve - data looks correct")
    print("  [C] Approve with corrections - applied corrections above")
    print("  [R] Re-extract needed - significant issues found")
    print("  [F] Flag for review - uncertain about quality")

    decision = input("> ").strip().lower()

    status_map = {
        'a': 'approved',
        'c': 'approved_with_corrections',
        'r': 'needs_reextraction',
        'f': 'flagged_for_review'
    }
    result['overall_status'] = status_map.get(decision, 'unknown')

    # Any final notes
    notes = input("\nAny final notes? > ").strip()
    result['notes'] = notes

    # Calculate time spent
    end_time = datetime.now()
    result['time_spent_seconds'] = (end_time - start_time).total_seconds()

    print(f"\n[OK] Validation complete ({result['time_spent_seconds']:.0f} seconds)")

    return result


def validate_all_papers(
    summary_path: str,
    papers_dir: str,
    output_dir: str,
    llm_client: LLMClient,
    limit: int = None
):
    """
    Run interactive validation on all papers.
    """

    # Load data
    df = pd.read_csv(summary_path)
    papers_dir = Path(papers_dir)

    # Initialize session
    session = ValidationSession(output_dir)

    # Get papers to validate
    all_papers = df['paper_id'].unique().tolist()
    pending = session.get_papers_to_validate(all_papers)

    if limit:
        pending = pending[:limit]

    print(f"\nValidation Session")
    print(f"==================")
    print(f"Total papers: {len(all_papers)}")
    print(f"Already validated: {len(all_papers) - len(pending)}")
    print(f"Pending: {len(pending)}")

    if not pending:
        print("\nAll papers have been validated!")
        session.print_summary()
        return

    print(f"\nStarting validation of {len(pending)} papers...")
    print("(Press Ctrl+C at any time to pause and save progress)")

    try:
        for i, paper_id in enumerate(pending):
            print(f"\n{'=' * 70}")
            print(f"Paper {i+1}/{len(pending)}: {paper_id}")
            print(f"{'=' * 70}")

            # Find PDF
            pdf_path = papers_dir / f"{paper_id}.pdf"
            if not pdf_path.exists():
                print(f"PDF not found: {pdf_path}")
                continue

            # Get paper data
            paper_data = df[df['paper_id'] == paper_id].copy()
            paper_data.reset_index(drop=True, inplace=True)

            # Run validation
            result = validate_paper_interactive(
                paper_id,
                str(pdf_path),
                paper_data,
                llm_client,
                session
            )

            # Log result
            session.log_paper_validation(paper_id, result)

            # Continue?
            if i < len(pending) - 1:
                cont = input("\nContinue to next paper? [Y/n] > ").strip().lower()
                if cont == 'n':
                    break

    except KeyboardInterrupt:
        print("\n\nValidation paused. Progress saved.")

    # Print summary
    session.print_summary()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Interactive human validation for meta-analysis data extraction'
    )
    parser.add_argument('--summary', required=True, help='Path to summary.csv')
    parser.add_argument('--papers', required=True, help='Directory containing PDFs')
    parser.add_argument('--output', help='Output directory for validation logs')
    parser.add_argument('--provider', default='google', choices=['google', 'anthropic'])
    parser.add_argument('--api-key', help='API key (or set env var)')
    parser.add_argument('--limit', type=int, help='Limit number of papers to validate')
    parser.add_argument('--paper', help='Validate specific paper ID only')

    args = parser.parse_args()

    # Setup output dir
    if args.output:
        output_dir = args.output
    else:
        output_dir = str(Path(args.summary).parent / 'validation')

    # Setup LLM client
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: No API key. Set --api-key or GOOGLE_API_KEY/ANTHROPIC_API_KEY env var")
        sys.exit(1)

    llm_client = LLMClient(provider=args.provider, api_key=api_key)
    # Use fast model for summaries
    model = 'gemini-2.0-flash' if args.provider == 'google' else 'claude-3-5-haiku-20241022'
    llm_client.default_model = model

    if args.paper:
        # Validate single paper
        df = pd.read_csv(args.summary)
        paper_data = df[df['paper_id'] == args.paper].copy()

        if len(paper_data) == 0:
            print(f"Paper not found: {args.paper}")
            sys.exit(1)

        pdf_path = Path(args.papers) / f"{args.paper}.pdf"
        session = ValidationSession(output_dir)

        result = validate_paper_interactive(
            args.paper,
            str(pdf_path),
            paper_data,
            llm_client,
            session
        )

        session.log_paper_validation(args.paper, result)
        session.print_summary()
    else:
        # Validate all papers
        validate_all_papers(
            args.summary,
            args.papers,
            output_dir,
            llm_client,
            args.limit
        )


if __name__ == '__main__':
    main()
