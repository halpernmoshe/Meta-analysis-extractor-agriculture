"""
Screening Prompts for Paper Acquisition

LLM prompts for automated title/abstract screening against PICO criteria.
"""

import json
from typing import List, Dict, Optional


SCREENING_SYSTEM_PROMPT = """You are a systematic review screening assistant. Your job is to evaluate
whether scientific papers should be included in a meta-analysis based on PICO criteria.

IMPORTANT RULES:
1. Base decisions ONLY on the title and abstract provided
2. When uncertain, err on the side of INCLUSION (let full-text review decide)
3. Provide clear, specific reasons for exclusion
4. Never include papers without quantitative data potential
5. Reviews and meta-analyses should generally be EXCLUDED (we want primary studies)

RESPONSE FORMAT:
Always respond with valid JSON. Do not include any text outside the JSON object."""


def get_screening_prompt(title: str,
                        abstract: Optional[str],
                        pico: Dict) -> str:
    """
    Build screening prompt for a single paper.

    Args:
        title: Paper title
        abstract: Paper abstract (may be None)
        pico: PICO criteria dict

    Returns:
        Formatted prompt string
    """
    # Format PICO criteria
    pico_text = _format_pico(pico)

    # Handle missing abstract
    abstract_text = abstract if abstract else "[No abstract available - decide based on title only]"

    return f"""SCREEN THIS PAPER FOR META-ANALYSIS INCLUSION

PICO CRITERIA:
{pico_text}

PAPER TO SCREEN:
Title: {title}

Abstract:
{abstract_text}

DECISION TASK:
Evaluate this paper against the PICO criteria above. Consider:
1. Does the population match? (species, study type, etc.)
2. Does the intervention/exposure match?
3. Are the outcomes likely present?
4. Is this a primary study with quantitative data?

Return JSON in this exact format:
{{
    "decision": "include" | "exclude" | "uncertain",
    "confidence": "high" | "medium" | "low",
    "population_match": true | false | "unclear",
    "intervention_match": true | false | "unclear",
    "outcome_potential": true | false | "unclear",
    "quantitative_data": true | false | "unclear",
    "reason": "Brief explanation (max 100 words)",
    "exclusion_category": null | "wrong_population" | "wrong_intervention" |
                          "no_outcomes" | "review_not_primary" | "no_quantitative_data" |
                          "wrong_language" | "duplicate" | "other"
}}

GUIDELINES:
- "include": Paper clearly meets PICO criteria AND likely has extractable quantitative data
- "exclude": Paper clearly does NOT meet criteria OR is a review/commentary/editorial
- "uncertain": Cannot determine from title/abstract alone - needs full text review
- If abstract mentions tables, statistics, figures, or numerical results: quantitative_data = true
- Reviews, meta-analyses, commentaries, and editorials should be EXCLUDED
- Include papers even if only secondary outcomes are mentioned"""


def get_batch_screening_prompt(papers: List[Dict], pico: Dict, batch_size: int = 10) -> str:
    """
    Build prompt for batch screening (more efficient).

    Args:
        papers: List of paper dicts with 'paper_id', 'title', 'abstract'
        pico: PICO criteria dict
        batch_size: Max papers per batch

    Returns:
        Formatted prompt string
    """
    # Format PICO criteria
    pico_text = _format_pico(pico)

    # Format papers
    papers_text = []
    for i, paper in enumerate(papers[:batch_size], 1):
        paper_id = paper.get('paper_id', f'PAPER_{i}')
        title = paper.get('title', 'No title')
        abstract = paper.get('abstract', '')

        # Truncate long abstracts
        if abstract and len(abstract) > 1500:
            abstract = abstract[:1500] + "..."

        papers_text.append(f"""
---
PAPER {i}:
ID: {paper_id}
Title: {title}
Abstract: {abstract if abstract else '[No abstract]'}
---""")

    papers_block = '\n'.join(papers_text)

    return f"""BATCH SCREEN {len(papers[:batch_size])} PAPERS FOR META-ANALYSIS

PICO CRITERIA:
{pico_text}

PAPERS TO SCREEN:
{papers_block}

TASK:
For EACH paper above, evaluate against PICO criteria and provide a decision.

Return a JSON array with one object per paper:
[
    {{
        "paper_id": "P00001",
        "decision": "include" | "exclude" | "uncertain",
        "confidence": "high" | "medium" | "low",
        "population_match": true | false | "unclear",
        "intervention_match": true | false | "unclear",
        "outcome_potential": true | false | "unclear",
        "reason": "Brief explanation",
        "exclusion_category": null | "wrong_population" | "wrong_intervention" | "no_outcomes" | "review_not_primary" | "no_quantitative_data" | "other"
    }},
    ...
]

IMPORTANT:
- Return exactly {len(papers[:batch_size])} objects in the array
- Use the paper_id from each paper
- Be consistent in applying criteria across all papers
- When uncertain, prefer "include" or "uncertain" over "exclude"
- Reviews and meta-analyses should be excluded"""


def _format_pico(pico: Dict) -> str:
    """Format PICO criteria for prompt"""
    lines = []

    # Population
    pop_desc = pico.get('population_description', 'Not specified')
    species = pico.get('crop_species', [])
    study_types = pico.get('study_types', [])

    lines.append("POPULATION:")
    lines.append(f"  - Description: {pop_desc}")
    if species:
        if species == ['All'] or 'All' in species:
            lines.append("  - Species: Any species")
        else:
            lines.append(f"  - Species: {', '.join(species)}")
    if study_types:
        lines.append(f"  - Study types: {', '.join(study_types)}")

    # Intervention
    int_desc = pico.get('intervention_description', 'Not specified')
    int_domain = pico.get('intervention_domain', '')
    int_var = pico.get('intervention_variable', '')

    lines.append("\nINTERVENTION/EXPOSURE:")
    lines.append(f"  - Description: {int_desc}")
    if int_domain:
        lines.append(f"  - Domain: {int_domain}")
    if int_var:
        lines.append(f"  - Variable: {int_var}")

    # Comparison
    control_def = pico.get('control_definition', 'Standard control')
    lines.append("\nCOMPARISON/CONTROL:")
    lines.append(f"  - Control: {control_def}")

    # Outcomes
    primary = pico.get('primary_outcomes', [])
    secondary = pico.get('secondary_outcomes', [])

    lines.append("\nOUTCOMES:")
    if primary:
        lines.append(f"  - Primary: {', '.join(primary)}")
    if secondary:
        lines.append(f"  - Secondary: {', '.join(secondary)}")

    return '\n'.join(lines)


def parse_screening_response(response: str) -> Dict:
    """
    Parse LLM screening response.

    Args:
        response: Raw LLM response

    Returns:
        Parsed dict with screening decision
    """
    try:
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if response.startswith('```'):
            # Remove code block markers
            lines = response.split('\n')
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith('```'):
                    in_block = not in_block
                    continue
                if in_block:
                    json_lines.append(line)
            response = '\n'.join(json_lines)

        # Parse JSON
        result = json.loads(response)

        # Validate required fields
        if isinstance(result, list):
            # Batch response
            return {'batch_results': result}
        elif isinstance(result, dict):
            # Single response
            if 'decision' not in result:
                result['decision'] = 'uncertain'
            return result
        else:
            return {
                'decision': 'uncertain',
                'confidence': 'low',
                'reason': 'Failed to parse response',
                'parse_error': True
            }

    except json.JSONDecodeError as e:
        # Try to extract decision from text
        response_lower = response.lower()
        if 'exclude' in response_lower:
            decision = 'exclude'
        elif 'include' in response_lower:
            decision = 'include'
        else:
            decision = 'uncertain'

        return {
            'decision': decision,
            'confidence': 'low',
            'reason': f'JSON parse error: {str(e)[:100]}',
            'raw_response': response[:500],
            'parse_error': True
        }


def create_pico_from_config(config: Dict) -> Dict:
    """
    Create PICO dict from acquisition config.

    Args:
        config: Acquisition config dict

    Returns:
        PICO dict for screening
    """
    pico_config = config.get('pico_for_screening', {})

    return {
        'population_description': pico_config.get('population_description', 'Not specified'),
        'crop_species': pico_config.get('crop_species', ['All']),
        'study_types': pico_config.get('study_types', ['Field', 'Greenhouse', 'Growth Chamber']),
        'intervention_description': pico_config.get('intervention_description', 'Not specified'),
        'intervention_domain': pico_config.get('intervention_domain', ''),
        'intervention_variable': pico_config.get('intervention_variable', ''),
        'control_definition': pico_config.get('control_definition', 'Standard control'),
        'primary_outcomes': pico_config.get('primary_outcomes', []),
        'secondary_outcomes': pico_config.get('secondary_outcomes', [])
    }
