"""
Abstract Validator Module

Compares claims in paper abstracts against extracted data to detect:
1. Direction mismatches (abstract says increase, data shows decrease)
2. Magnitude discrepancies (abstract says 20% increase, data shows 80% increase)

This catches extraction errors like STUDY_035 where treatment/control were swapped.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import fitz  # PyMuPDF
import re
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm import LLMClient


ABSTRACT_EXTRACTION_PROMPT = """Analyze this scientific paper abstract and extract claims about the treatment effect.

ABSTRACT:
{abstract_text}

Extract the following information about the MAIN treatment effect on the PRIMARY outcome:

1. DIRECTION: Does the treatment increase, decrease, or have no effect on the outcome?
   - "increase" = treatment improves/raises the outcome
   - "decrease" = treatment reduces/lowers the outcome
   - "no_effect" = no significant change
   - "unclear" = direction not stated in abstract

2. MAGNITUDE: What is the approximate effect size mentioned?
   - Extract any percentages (e.g., "20%", "1.5-fold")
   - If a range is given, extract both bounds
   - If no magnitude given, return null

3. OUTCOME: What is the primary outcome variable discussed?
   - e.g., "yield", "grain weight", "biomass", "mineral concentration"

4. TREATMENT: What is the intervention/treatment?
   - e.g., "silicon application", "elevated CO2", "drought stress"

5. CONFIDENCE: How confident are you in this extraction?
   - "high" = clear statement of effect
   - "medium" = implied or indirect statement
   - "low" = very vague or ambiguous

Return as JSON:
{{
    "direction": "increase|decrease|no_effect|unclear",
    "magnitude_low": <number or null>,
    "magnitude_high": <number or null>,
    "magnitude_unit": "percent|fold|absolute|null",
    "outcome": "<outcome variable>",
    "treatment": "<treatment description>",
    "confidence": "high|medium|low",
    "quote": "<relevant quote from abstract>"
}}

Return ONLY the JSON, no other text.
"""


def extract_abstract(pdf_path: str) -> Optional[str]:
    """Extract abstract text from PDF."""
    try:
        doc = fitz.open(pdf_path)

        # Usually abstract is on first 1-2 pages
        text = ""
        for i in range(min(2, len(doc))):
            text += doc[i].get_text()

        doc.close()

        # Try to find abstract section
        # Common patterns: "Abstract", "ABSTRACT", "Summary", "SUMMARY"
        patterns = [
            r'(?i)abstract[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|key\s*words|1\.|background))',
            r'(?i)summary[:\s]*\n?(.*?)(?=\n\s*(?:introduction|keywords|key\s*words|1\.))',
            r'(?i)abstract[:\s]*\n?(.{200,2000})',  # Fallback: grab ~200-2000 chars after "abstract"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 100:  # Reasonable abstract length
                    return abstract[:3000]  # Limit length

        # If no clear abstract found, return first ~2000 chars (often contains abstract)
        return text[:2000]

    except Exception as e:
        print(f"Error extracting abstract: {e}")
        return None


def extract_abstract_claims(abstract_text: str, llm_client: LLMClient) -> Optional[dict]:
    """Use LLM to extract claims from abstract."""
    prompt = ABSTRACT_EXTRACTION_PROMPT.format(abstract_text=abstract_text)

    try:
        response = llm_client.call(
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1
        )

        # Parse JSON response
        response = response.strip()
        if response.startswith("```"):
            response = re.sub(r'^```json?\s*', '', response)
            response = re.sub(r'\s*```$', '', response)

        return json.loads(response)

    except Exception as e:
        print(f"Error extracting claims: {e}")
        return None


def calculate_extracted_effect(paper_data: pd.DataFrame, outcome: str = 'GWAD') -> dict:
    """Calculate effect size from extracted data for a paper."""
    # Filter to outcome
    data = paper_data[paper_data['outcome_variable'] == outcome]

    if len(data) == 0:
        # Try any outcome
        data = paper_data

    if len(data) == 0:
        return {'direction': 'no_data', 'magnitude': None, 'n_obs': 0}

    # Calculate mean effect
    valid = data[
        (data['treatment_mean'].notna()) &
        (data['control_mean'].notna()) &
        (data['treatment_mean'] > 0) &
        (data['control_mean'] > 0)
    ]

    if len(valid) == 0:
        return {'direction': 'no_data', 'magnitude': None, 'n_obs': 0}

    # Calculate log response ratio for each observation
    effects = []
    for _, row in valid.iterrows():
        lnRR = np.log(row['treatment_mean'] / row['control_mean'])
        pct_change = (np.exp(lnRR) - 1) * 100
        effects.append(pct_change)

    mean_effect = np.mean(effects)

    # Determine direction
    if mean_effect > 5:
        direction = 'increase'
    elif mean_effect < -5:
        direction = 'decrease'
    else:
        direction = 'no_effect'

    return {
        'direction': direction,
        'magnitude': mean_effect,
        'n_obs': len(valid),
        'effects': effects
    }


def compare_abstract_vs_extracted(abstract_claims: dict, extracted_effect: dict) -> dict:
    """Compare abstract claims against extracted data."""

    result = {
        'match': True,
        'issues': [],
        'severity': 'none',
        'abstract_direction': abstract_claims.get('direction'),
        'extracted_direction': extracted_effect.get('direction'),
        'abstract_magnitude': None,
        'extracted_magnitude': extracted_effect.get('magnitude')
    }

    # Get abstract magnitude
    if abstract_claims.get('magnitude_low') is not None:
        if abstract_claims.get('magnitude_high') is not None:
            result['abstract_magnitude'] = (
                abstract_claims['magnitude_low'] + abstract_claims['magnitude_high']
            ) / 2
        else:
            result['abstract_magnitude'] = abstract_claims['magnitude_low']

    # Check 1: Direction mismatch
    abstract_dir = abstract_claims.get('direction', 'unclear')
    extracted_dir = extracted_effect.get('direction', 'no_data')

    if abstract_dir != 'unclear' and extracted_dir != 'no_data':
        if abstract_dir == 'increase' and extracted_dir == 'decrease':
            result['match'] = False
            result['issues'].append('DIRECTION MISMATCH: Abstract says INCREASE, data shows DECREASE')
            result['severity'] = 'critical'
        elif abstract_dir == 'decrease' and extracted_dir == 'increase':
            result['match'] = False
            result['issues'].append('DIRECTION MISMATCH: Abstract says DECREASE, data shows INCREASE')
            result['severity'] = 'critical'

    # Check 2: Magnitude discrepancy (if both available)
    if result['abstract_magnitude'] is not None and result['extracted_magnitude'] is not None:
        abs_mag = abs(result['abstract_magnitude'])
        ext_mag = abs(result['extracted_magnitude'])

        # Check if magnitudes are wildly different (more than 3x)
        if abs_mag > 0 and ext_mag > 0:
            ratio = max(abs_mag, ext_mag) / min(abs_mag, ext_mag)
            if ratio > 3:
                result['issues'].append(
                    f'MAGNITUDE DISCREPANCY: Abstract ~{abs_mag:.0f}%, Extracted ~{ext_mag:.0f}% (ratio: {ratio:.1f}x)'
                )
                if result['severity'] == 'none':
                    result['severity'] = 'warning'

    # Check 3: Extreme effects (sanity check)
    if extracted_effect.get('magnitude') is not None:
        mag = abs(extracted_effect['magnitude'])
        if mag > 200:  # More than 200% change is suspicious
            result['issues'].append(f'EXTREME EFFECT: {extracted_effect["magnitude"]:.1f}% change is unusually large')
            if result['severity'] == 'none':
                result['severity'] = 'warning'

    return result


def validate_paper(paper_id: str, pdf_path: str, extracted_data: pd.DataFrame,
                   llm_client: LLMClient, outcome: str = 'GWAD') -> dict:
    """Validate a single paper by comparing abstract to extracted data."""

    result = {
        'paper_id': paper_id,
        'status': 'unknown',
        'abstract_claims': None,
        'extracted_effect': None,
        'comparison': None
    }

    # Step 1: Extract abstract
    abstract = extract_abstract(pdf_path)
    if not abstract:
        result['status'] = 'error'
        result['error'] = 'Could not extract abstract'
        return result

    # Step 2: Extract claims from abstract
    claims = extract_abstract_claims(abstract, llm_client)
    if not claims:
        result['status'] = 'error'
        result['error'] = 'Could not parse abstract claims'
        return result

    result['abstract_claims'] = claims

    # Step 3: Calculate effect from extracted data
    paper_data = extracted_data[extracted_data['paper_id'] == paper_id]
    extracted = calculate_extracted_effect(paper_data, outcome)
    result['extracted_effect'] = extracted

    if extracted['direction'] == 'no_data':
        result['status'] = 'no_data'
        return result

    # Step 4: Compare
    comparison = compare_abstract_vs_extracted(claims, extracted)
    result['comparison'] = comparison

    if comparison['match']:
        result['status'] = 'valid'
    else:
        result['status'] = comparison['severity']

    return result


def validate_all_papers(summary_path: str, papers_dir: str, llm_client: LLMClient,
                        outcome: str = 'GWAD') -> pd.DataFrame:
    """Validate all papers in the extraction."""

    df = pd.read_csv(summary_path)
    papers_dir = Path(papers_dir)

    results = []
    paper_ids = df['paper_id'].unique()

    print(f"\nValidating {len(paper_ids)} papers against abstracts...")
    print("="*70)

    for i, paper_id in enumerate(paper_ids):
        # Find PDF
        pdf_path = papers_dir / f"{paper_id}.pdf"
        if not pdf_path.exists():
            print(f"[{i+1}/{len(paper_ids)}] {paper_id}: PDF not found")
            results.append({
                'paper_id': paper_id,
                'status': 'pdf_not_found',
                'direction_match': None,
                'abstract_direction': None,
                'extracted_direction': None,
                'extracted_magnitude': None,
                'issues': 'PDF not found'
            })
            continue

        # Validate
        result = validate_paper(paper_id, str(pdf_path), df, llm_client, outcome)

        # Format for output
        row = {
            'paper_id': paper_id,
            'status': result['status'],
            'direction_match': None,
            'abstract_direction': None,
            'extracted_direction': None,
            'abstract_magnitude': None,
            'extracted_magnitude': None,
            'issues': ''
        }

        if result.get('abstract_claims'):
            row['abstract_direction'] = result['abstract_claims'].get('direction')
            if result['abstract_claims'].get('magnitude_low'):
                row['abstract_magnitude'] = result['abstract_claims'].get('magnitude_low')

        if result.get('extracted_effect'):
            row['extracted_direction'] = result['extracted_effect'].get('direction')
            row['extracted_magnitude'] = result['extracted_effect'].get('magnitude')

        if result.get('comparison'):
            row['direction_match'] = result['comparison'].get('match')
            row['issues'] = '; '.join(result['comparison'].get('issues', []))

        results.append(row)

        # Print status
        status_symbol = {
            'valid': '✓',
            'critical': '✗ CRITICAL',
            'warning': '⚠ WARNING',
            'no_data': '-',
            'error': '?'
        }.get(result['status'], '?')

        mag_str = f"{row['extracted_magnitude']:+.1f}%" if row['extracted_magnitude'] else "N/A"

        print(f"[{i+1}/{len(paper_ids)}] {paper_id}: {status_symbol} (extracted: {mag_str})")

        if row['issues']:
            print(f"    → {row['issues']}")

    return pd.DataFrame(results)


def main():
    """Run abstract validation on all papers."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate extractions against paper abstracts')
    parser.add_argument('--summary', required=True, help='Path to summary.csv')
    parser.add_argument('--papers', required=True, help='Directory containing PDFs')
    parser.add_argument('--output', help='Output path for validation results')
    parser.add_argument('--provider', default='google', choices=['google', 'anthropic'])
    parser.add_argument('--api-key', help='API key (or set GOOGLE_API_KEY/ANTHROPIC_API_KEY env var)')
    parser.add_argument('--outcome', default='GWAD', help='Outcome variable to validate')

    args = parser.parse_args()

    # Setup LLM client
    api_key = args.api_key or os.environ.get('GOOGLE_API_KEY') or os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: No API key provided. Set --api-key or GOOGLE_API_KEY/ANTHROPIC_API_KEY env var")
        sys.exit(1)

    # Use a fast/cheap model for abstract parsing
    model = 'gemini-2.0-flash' if args.provider == 'google' else 'claude-3-5-haiku-20241022'
    llm_client = LLMClient(provider=args.provider, model=model, api_key=api_key)

    # Run validation
    results = validate_all_papers(
        args.summary,
        args.papers,
        llm_client,
        args.outcome
    )

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    status_counts = results['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    # Highlight critical issues
    critical = results[results['status'] == 'critical']
    if len(critical) > 0:
        print(f"\n⚠️  CRITICAL ISSUES FOUND ({len(critical)} papers):")
        for _, row in critical.iterrows():
            print(f"  - {row['paper_id']}: {row['issues']}")

    warnings = results[results['status'] == 'warning']
    if len(warnings) > 0:
        print(f"\n⚠  WARNINGS ({len(warnings)} papers):")
        for _, row in warnings.iterrows():
            print(f"  - {row['paper_id']}: {row['issues']}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = Path(args.summary).parent / 'abstract_validation.csv'

    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
