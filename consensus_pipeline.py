"""
Two-Stage Consensus Pipeline for Meta-Analysis Extraction

Stage 1: Claude "Reconnaissance" - Analyzes paper structure, identifies potential issues
         Now includes Challenge-Aware detection for SCANNED, FIG-ONLY, IMAGE-TABLES
Stage 2: Unified Extraction - Both Claude and Kimi extract with same hints from recon

Includes Zero Trust Variance Verification:
- GRIM test (mean validity)
- P-value triangulation (SE vs SD detection)
- CV bounds checking

Challenge-Aware Routing:
- TEXT: Standard text extraction (default)
- VISION: Papers with FIG-ONLY or IMAGE-TABLES need vision API
- HYBRID: Scanned PDFs benefit from both text and vision
- MANUAL: Flag for human review

Usage:
    python consensus_pipeline.py --input ./papers --output ./results
"""

import os
import json
import re
import csv
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from math import sqrt
import traceback

# API clients
import openai
from dotenv import load_dotenv

# Challenge-aware recon
try:
    from challenge_aware_recon import (
        ChallengeDetector,
        ChallengeAwareRecon,
        ChallengeDetection,
        ExtractionMethod,
        format_hints_for_prompt
    )
    CHALLENGE_AWARE_AVAILABLE = True
except ImportError:
    CHALLENGE_AWARE_AVAILABLE = False
    print("Warning: challenge_aware_recon module not available")

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MetaAnalysisConfig:
    """
    Describes the meta-analysis research question.
    This is passed to both recon and extraction stages.
    """
    name: str = "CO2 Effects on Plant Mineral Concentrations"

    description: str = """
    This meta-analysis investigates how elevated atmospheric CO2 affects
    mineral element concentrations in plants. The key hypothesis is that
    elevated CO2 causes a "dilution effect" - plants grow larger but mineral
    uptake doesn't keep pace, resulting in lower mineral concentrations.
    """

    intervention: str = "Elevated atmospheric CO2 (typically 550-700 ppm)"
    control: str = "Ambient CO2 (typically 350-400 ppm)"

    primary_outcomes: List[str] = field(default_factory=lambda: [
        "Mineral element concentrations: N, P, K, Ca, Mg, Fe, Zn, Mn, Cu, S, B, Mo"
    ])

    expected_direction: str = "negative"  # We expect elevated CO2 to DECREASE mineral concentrations

    typical_effect_size: str = "-5% to -15% for most minerals"

    important_moderators: List[str] = field(default_factory=lambda: [
        "Plant species/cultivar",
        "Tissue type (grain, leaf, root)",
        "Experimental system (FACE, OTC, greenhouse)",
        "Nitrogen fertilization level",
        "Study duration"
    ])

    extraction_priorities: List[str] = field(default_factory=lambda: [
        "1. ⚠️ CRITICAL: Extract EVERY ROW from tables - NO POOLING/AVERAGING across cultivars, years, or treatments",
        "2. If table has 20 rows, you should have ~20 observations. Count them!",
        "3. Prioritize mineral concentration data over biomass data",
        "4. Get variance (SE/SD) and sample size (n) for meta-analysis weighting",
        "5. Record tissue type (grain vs leaf vs root) as it affects interpretation"
    ])

    tc_confusion_warnings: List[str] = field(default_factory=lambda: [
        "Watch for papers that report 'ambient' vs 'elevated' - ambient is control",
        "Some papers report percentage of ambient - 100% = control, >100% = elevated",
        "Water stress treatments (FWC levels) are NOT CO2 treatments",
        "If effect is strongly positive (+50%), verify T/C assignment"
    ])

    @classmethod
    def from_json(cls, json_path: str) -> 'MetaAnalysisConfig':
        """Load config from a JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Map from JSON structure to dataclass fields
        # Support both flat format and nested pico format
        kwargs = {}

        if 'name' in data:
            kwargs['name'] = data['name']
        if 'description' in data:
            kwargs['description'] = data['description']
        if 'intervention' in data:
            kwargs['intervention'] = data['intervention']
        elif 'pico' in data and 'intervention' in data['pico']:
            interv = data['pico']['intervention']
            if isinstance(interv, dict):
                kwargs['intervention'] = interv.get('treatment_variable', '')
            else:
                kwargs['intervention'] = interv
        if 'control' in data:
            kwargs['control'] = data['control']
        elif 'pico' in data and 'comparison' in data['pico']:
            comp = data['pico']['comparison']
            if isinstance(comp, dict):
                kwargs['control'] = comp.get('control_definition', '')
            else:
                kwargs['control'] = comp
        if 'primary_outcomes' in data:
            kwargs['primary_outcomes'] = data['primary_outcomes']
        elif 'pico' in data and 'outcomes' in data['pico']:
            outcomes = data['pico']['outcomes']
            if isinstance(outcomes, dict):
                kwargs['primary_outcomes'] = outcomes.get('primary', [])
            else:
                kwargs['primary_outcomes'] = outcomes
        if 'expected_direction' in data:
            kwargs['expected_direction'] = data['expected_direction']
        if 'typical_effect_size' in data:
            kwargs['typical_effect_size'] = data['typical_effect_size']
        if 'important_moderators' in data:
            kwargs['important_moderators'] = data['important_moderators']
        if 'extraction_priorities' in data:
            kwargs['extraction_priorities'] = data['extraction_priorities']
        if 'tc_confusion_warnings' in data:
            kwargs['tc_confusion_warnings'] = data['tc_confusion_warnings']

        return cls(**kwargs)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ReconResult:
    """Result from Stage 1 reconnaissance pass."""
    paper_id: str
    warnings: List[str]
    variance_type: Optional[str]
    variance_source: Optional[str]
    variance_confidence: str  # high, medium, low
    control_definition: str
    treatment_definition: str
    tables_with_target_data: List[str]
    potential_tc_confusion: Optional[str]
    experimental_design: str
    sample_size_found: Optional[int]
    sample_size_source: Optional[str]
    factorial_structure: Optional[str]
    extraction_guidance: str
    raw_response: str = ""

    # Table targeting fields
    tables_without_target_data: List[str] = field(default_factory=list)
    table_descriptions: Dict[str, str] = field(default_factory=dict)
    figures_with_data: List[str] = field(default_factory=list)
    outcome_variables_detected: Dict[str, List[str]] = field(default_factory=dict)  # e.g. {"Table 3": ["N", "P", "K", "Ca"]}

    # Challenge-aware fields
    extraction_method: str = "text"  # text, vision, hybrid, manual
    extraction_method_reason: str = ""
    challenge_hints: List[str] = field(default_factory=list)
    is_scanned: bool = False
    is_fig_only: bool = False
    has_image_tables: bool = False
    estimated_difficulty: str = "MEDIUM"


@dataclass
class Observation:
    """Single extracted observation."""
    element: str
    tissue: str
    treatment_mean: float
    control_mean: float
    treatment_variance: Optional[float] = None
    control_variance: Optional[float] = None
    variance_type: Optional[str] = None
    n: Optional[int] = None
    unit: str = ""
    data_source: str = ""
    treatment_description: str = ""
    control_description: str = ""
    moderators: Dict[str, str] = field(default_factory=dict)
    confidence: str = "medium"
    notes: str = ""

    # Calculated fields
    effect_pct: Optional[float] = None
    ln_rr: Optional[float] = None

    # Verification flags
    grim_valid: Optional[bool] = None
    cv_reasonable: Optional[bool] = None
    direction_expected: Optional[bool] = None


@dataclass
class ExtractionResult:
    """Result from extraction (from either model)."""
    paper_id: str
    model: str  # "claude" or "kimi"
    observations: List[Observation]
    paper_info: Dict
    extraction_notes: str = ""
    tokens_used: int = 0
    cost_estimate: float = 0.0


@dataclass
class ConsensusResult:
    """Final result after comparing both models."""
    paper_id: str
    recon: ReconResult
    claude_result: Optional[ExtractionResult]
    kimi_result: Optional[ExtractionResult]

    # Consensus metrics
    total_claude_obs: int = 0
    total_kimi_obs: int = 0
    matched_obs: int = 0
    disagreements: List[Dict] = field(default_factory=list)

    # Final merged observations
    consensus_observations: List[Observation] = field(default_factory=list)

    # Vision extraction results (when routing uses VISION or HYBRID)
    vision_result: Optional[ExtractionResult] = None
    kimi_vision_result: Optional[ExtractionResult] = None

    # Gemini tiebreaker results
    gemini_result: Optional[ExtractionResult] = None
    tiebreaker_used: bool = False
    tiebreaker_reason: str = ""

    # Post-processing stats
    post_processing: Dict = field(default_factory=dict)

    # Verification results
    verification_flags: List[Dict] = field(default_factory=list)


# =============================================================================
# VERIFICATION FUNCTIONS (Zero Trust)
# =============================================================================

def grim_test(mean: float, n: int, decimals: int = 2) -> bool:
    """
    GRIM test: Check if mean is mathematically possible for integer data.

    For integer data with sample size n, the mean must be expressible as sum/n.
    Returns True if mean is plausible, False if impossible.
    """
    if n <= 0 or n > 1000:
        return True  # Can't test

    # Calculate what the sum would need to be
    items = mean * n

    # For integer data, sum must be an integer
    return abs(items - round(items)) < (0.5 * 10**-decimals)


def cv_check(mean: float, variance: float, variance_type: str, n: int = None) -> Tuple[bool, float]:
    """
    Check if coefficient of variation is in reasonable range (typically 5-50%).

    Returns (is_reasonable, cv_value)
    """
    if mean == 0 or variance is None:
        return True, 0

    # Convert to SD if needed
    if variance_type == "SE" and n and n > 0:
        sd = variance * sqrt(n)
    else:
        sd = variance

    cv = (sd / abs(mean)) * 100

    # For biological data, CV typically 5-50%
    # Allow wider range (1-100%) to avoid false positives
    is_reasonable = 1 < cv < 100

    return is_reasonable, cv


def p_value_triangulation(
    mean1: float, mean2: float,
    n1: int, n2: int,
    mystery_variance: float,
    reported_p: float = None
) -> Tuple[str, float, str]:
    """
    Use p-value to determine if mystery_variance is SE or SD.

    If we can't use p-value, use CV-based heuristic:
    - If treating as SE gives CV > 100%, it's probably SD
    - If treating as SD gives CV < 1%, it's probably SE

    Returns (likely_type, confidence, reasoning)
    """
    from scipy import stats

    if n1 is None or n2 is None or n1 < 2 or n2 < 2:
        return "unknown", 0.5, "Insufficient sample size info"

    pooled_n = (n1 + n2) / 2
    df = n1 + n2 - 2

    # Calculate what SE and SD would be
    if mystery_variance <= 0:
        return "unknown", 0.5, "Invalid variance value"

    # If reported_p is available, use it
    if reported_p and 0 < reported_p < 1:
        try:
            # Hypothesis A: mystery_variance is SD
            se_if_sd = mystery_variance / sqrt(pooled_n)
            pooled_se_sd = se_if_sd * sqrt(2 / pooled_n)
            if pooled_se_sd > 0:
                t_if_sd = abs(mean1 - mean2) / pooled_se_sd
                p_if_sd = stats.t.sf(abs(t_if_sd), df) * 2
            else:
                p_if_sd = 1

            # Hypothesis B: mystery_variance is SE
            pooled_se_se = mystery_variance * sqrt(2)
            if pooled_se_se > 0:
                t_if_se = abs(mean1 - mean2) / pooled_se_se
                p_if_se = stats.t.sf(abs(t_if_se), df) * 2
            else:
                p_if_se = 1

            # Which matches better?
            error_if_sd = abs(p_if_sd - reported_p)
            error_if_se = abs(p_if_se - reported_p)

            if error_if_sd < error_if_se * 0.5:
                return "SD", 0.8, f"p-value match: SD gives p={p_if_sd:.3f} vs reported {reported_p}"
            elif error_if_se < error_if_sd * 0.5:
                return "SE", 0.8, f"p-value match: SE gives p={p_if_se:.3f} vs reported {reported_p}"
        except:
            pass

    # Fallback: CV-based heuristic
    avg_mean = (abs(mean1) + abs(mean2)) / 2
    if avg_mean > 0:
        # If it's SE, SD would be SE * sqrt(n)
        sd_if_se = mystery_variance * sqrt(pooled_n)
        cv_if_se = (sd_if_se / avg_mean) * 100

        # If it's SD, use directly
        cv_if_sd = (mystery_variance / avg_mean) * 100

        # Most biological CVs are 5-50%
        if cv_if_sd > 100 and 5 < cv_if_se < 50:
            return "SE", 0.6, f"CV heuristic: as SD gives CV={cv_if_sd:.1f}% (too high)"
        elif cv_if_se > 100 and 5 < cv_if_sd < 50:
            return "SD", 0.6, f"CV heuristic: as SE gives CV={cv_if_se:.1f}% (too high)"
        elif 5 < cv_if_sd < 50:
            return "SD", 0.5, f"CV heuristic: as SD gives reasonable CV={cv_if_sd:.1f}%"
        elif 5 < cv_if_se < 50:
            return "SE", 0.5, f"CV heuristic: as SE gives reasonable CV={cv_if_se:.1f}%"

    return "unknown", 0.3, "Could not determine variance type"


# =============================================================================
# GLOBAL VARIANCE SCANNER (deterministic, pre-LLM)
# =============================================================================

# Regex patterns that definitively declare the variance type in a paper
GLOBAL_VARIANCE_PATTERNS = [
    # "Data are presented as mean +/- SE" (broadened verb list)
    (r"(?:data|values|means?)\s+(?:are|were|represent)\s+(?:presented|expressed|shown|reported)\s+(?:as\s+)?(?:the\s+)?(?:means?\s*)?.{0,5}\s*(\bSE[M]?\b|\bSD\b|standard\s+(?:error|deviation))", "definition"),
    # "Values represent means +/- standard deviation" (very common)
    (r"values?\s+represent\s+means?\s*.{0,5}\s*(\bSE[M]?\b|\bSD\b|standard\s+(?:error|deviation))", "definition"),
    # "Error bars indicate SEM"
    (r"error\s+bars?\s+(?:represent|indicate|show|denote)\s+(?:the\s+)?(\bSE[M]?\b|\bSD\b|standard\s+error)", "caption"),
    # "Values are mean(SE)" or "mean(SD)"
    (r"values?\s+(?:are|represent)\s+mean[s]?\s*\((\bSE[M]?\b|\bSD\b)\)", "definition"),
    # "expressed as mean +/- standard error"
    (r"expressed\s+as\s+mean\s*.{0,5}\s*(standard\s+(?:error|deviation)|\bSE[M]?\b|\bSD\b)", "definition"),
    # "Mean +/- SD of N replications"
    (r"means?\s*.{0,5}\s*(\bSE[M]?\b|\bSD\b|standard\s+(?:error|deviation))\s+(?:of|from|for)", "definition"),
    # "each value is the mean +/- SE (n=X)"
    (r"each\s+value\s+(?:is|represents)\s+(?:the\s+)?mean\s*.{0,5}\s*(\bSE[M]?\b|\bSD\b|standard\s+(?:error|deviation))", "definition"),
    # "mean +/- SE, n=X per treatment"
    (r"means?\s*.{0,5}\s*(\bSE[M]?\b|\bSD\b)\s*[,;.]\s*n\s*=", "definition"),
    # "(mean +/- SE)" in parenthetical
    (r"\(\s*means?\s*.{0,5}\s*(\bSE[M]?\b|\bSD\b|standard\s+(?:error|deviation))\s*\)", "definition"),
    # "numbers in parentheses are standard error"
    (r"(?:numbers?|values?)\s+in\s+parentheses\s+(?:are|represent)\s+(?:the\s+)?(standard\s+error|standard\s+deviation|\bSE[M]?\b|\bSD\b)", "definition"),
    # "Data are means+/-SD (n=5)"
    (r"data\s+are\s+means?\s*.{0,3}(\bSD\b|\bSE[M]?\b)\s*\(\s*n\s*=", "definition"),
    # "LSD at P=0.05" (common in agronomy)
    (r"\b(LSD)\s+(?:at|for)\s+[Pp]\s*[=<]\s*0?\.0[15]", "statistics"),
    # "Least significant difference"
    (r"(least\s+significant\s+difference)", "statistics"),
]


def scan_global_variance_type(full_text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Scan paper text for definitive variance type declaration.

    Returns (variance_type, evidence_quote, confidence).
    confidence: "high" = definitive statement, "medium" = contextual, "low" = weak signal
    """
    # Focus on Methods + Statistical Analysis sections if identifiable
    methods_match = re.search(
        r'(?:materials?\s+and\s+)?methods?(.*?)(?:results|discussion)',
        full_text, re.IGNORECASE | re.DOTALL
    )
    # Also search table footnotes and figure captions
    search_regions = []
    if methods_match:
        search_regions.append(("methods", methods_match.group(1)[:10000]))
    # First 25K chars (abstract + methods + start of results)
    search_regions.append(("beginning", full_text[:25000]))
    # Also scan full text for table footnotes (they can be anywhere)
    footnote_patterns = [
        r'(?:note[s]?|footnote|values?\s+(?:are|represent)|data\s+(?:are|represent)|means?\s+.{0,5}\s+(?:SE|SD|standard)).*?(?:\.|$)',
    ]
    for fp in footnote_patterns:
        footnote_matches = re.findall(fp, full_text, re.IGNORECASE)
        for fm in footnote_matches[:10]:
            search_regions.append(("footnote", fm))

    for region_name, search_text in search_regions:
        for pattern, source_type in GLOBAL_VARIANCE_PATTERNS:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                # Get surrounding context for evidence
                start = max(0, match.start() - 40)
                end = min(len(search_text), match.end() + 40)
                evidence = search_text[start:end].strip()

                # Normalize the type
                raw_type = match.group(1).upper()
                if 'ERROR' in raw_type or raw_type in ('SE', 'SEM'):
                    var_type = 'SE'
                elif 'DEVIATION' in raw_type or raw_type == 'SD':
                    var_type = 'SD'
                elif 'LEAST' in raw_type or raw_type == 'LSD':
                    var_type = 'LSD'
                else:
                    continue

                # Confidence based on source
                if source_type == "definition" and region_name in ("methods", "beginning"):
                    confidence = "high"
                elif source_type == "caption" or region_name == "footnote":
                    confidence = "high"
                else:
                    confidence = "medium"

                return var_type, evidence, confidence

    return None, None, "low"


# =============================================================================
# T/C SWAP DETECTION
# =============================================================================

def calculate_swapped_effect(effect: float) -> float:
    """
    Calculate what the effect would be if T and C were swapped.

    Mathematical relationship:
    If true effect E = (T-C)/C, then swapped effect E_swap = (C-T)/T = -E/(1+E)

    Examples:
    - E = -10% (-0.10) → E_swap = 0.10/0.90 = +11.1%
    - E = -50% (-0.50) → E_swap = 0.50/0.50 = +100%
    - E = +10% (+0.10) → E_swap = -0.10/1.10 = -9.1%
    """
    if effect <= -1.0:  # Would cause division by zero or undefined
        return float('inf')
    return -effect / (1 + effect)


def detect_tc_swap(
    effect: float,
    expected_direction: str = "negative",
    typical_range: Tuple[float, float] = (-0.30, 0.10)
) -> Tuple[bool, float, str]:
    """
    Detect if an effect appears to be from swapped T/C values.

    Args:
        effect: The extracted effect as a decimal (e.g., -0.10 for -10%)
        expected_direction: "negative" or "positive"
        typical_range: (min, max) expected effect range as decimals

    Returns:
        (is_likely_swapped, corrected_effect, explanation)
    """
    # Calculate what the "unswapped" effect would be
    unswapped_effect = calculate_swapped_effect(effect)

    if abs(unswapped_effect) > 10:  # Unreasonable value
        return False, effect, "Unswapped effect would be unreasonable"

    # Check 1: Does current effect violate expected direction while unswapped matches?
    direction_violated = (
        (expected_direction == "negative" and effect > 0.05) or
        (expected_direction == "positive" and effect < -0.05)
    )

    unswapped_matches_direction = (
        (expected_direction == "negative" and unswapped_effect < 0) or
        (expected_direction == "positive" and unswapped_effect > 0)
    )

    if direction_violated and unswapped_matches_direction:
        return True, unswapped_effect, f"Effect {effect:+.1%} violates expected direction; unswapped would be {unswapped_effect:+.1%}"

    # Check 2: Is current effect outside typical range but unswapped is inside?
    min_effect, max_effect = typical_range
    current_in_range = min_effect <= effect <= max_effect
    unswapped_in_range = min_effect <= unswapped_effect <= max_effect

    if not current_in_range and unswapped_in_range:
        return True, unswapped_effect, f"Effect {effect:+.1%} outside typical range; unswapped {unswapped_effect:+.1%} is within range"

    return False, effect, "No swap detected"


def check_tc_swap_between_models(
    t1: float, c1: float,
    t2: float, c2: float,
    tolerance: float = 0.05
) -> Tuple[bool, str, Dict]:
    """
    Check if two models might have opposite T/C assignments.

    If Model A has (T=10, C=12) and Model B has (T=12, C=10), one has swapped.

    Returns:
        (swap_detected, which_model_swapped, details)
    """
    # Check if values are swapped: A's T matches B's C and vice versa
    t1_matches_c2 = abs(t1 - c2) / max(abs(t1), 0.001) < tolerance
    c1_matches_t2 = abs(c1 - t2) / max(abs(c1), 0.001) < tolerance

    if t1_matches_c2 and c1_matches_t2:
        # Values are swapped between models
        # Calculate effects for each model
        if c1 > 0 and c2 > 0:
            effect1 = (t1 - c1) / c1
            effect2 = (t2 - c2) / c2

            # The one with positive effect (when negative is expected) is likely swapped
            # Default assumption: negative effect is expected for mineral/CO2 data
            if effect1 > 0 and effect2 < 0:
                return True, "model1", {"effect1": effect1, "effect2": effect2, "recommendation": "Use model2 values"}
            elif effect2 > 0 and effect1 < 0:
                return True, "model2", {"effect1": effect1, "effect2": effect2, "recommendation": "Use model1 values"}
            else:
                return True, "unclear", {"effect1": effect1, "effect2": effect2, "recommendation": "Manual review needed"}

    return False, None, {}


def try_match_with_swap_correction(
    c_obs: 'Observation',
    k_obs: 'Observation',
    tolerance: float = 0.10
) -> Tuple[bool, Optional['Observation'], str]:
    """
    Try to match observations, including checking for T/C swap.

    Returns:
        (matched, corrected_observation, match_type)
        match_type: "exact", "swapped_corrected", or "no_match"
    """
    # First try exact match
    if c_obs.control_mean and k_obs.control_mean and c_obs.control_mean > 0:
        c_diff = abs(c_obs.control_mean - k_obs.control_mean) / max(abs(c_obs.control_mean), 0.001)
        t_diff = abs(c_obs.treatment_mean - k_obs.treatment_mean) / max(abs(c_obs.treatment_mean), 0.001)
        avg_diff = (c_diff + t_diff) / 2

        if avg_diff <= tolerance:
            return True, c_obs, "exact"

    # Check for T/C swap between models
    if c_obs.treatment_mean and c_obs.control_mean and k_obs.treatment_mean and k_obs.control_mean:
        swap_detected, swapped_model, details = check_tc_swap_between_models(
            c_obs.treatment_mean, c_obs.control_mean,
            k_obs.treatment_mean, k_obs.control_mean,
            tolerance
        )

        if swap_detected:
            # Determine which observation to use
            # Use the one with negative effect (expected direction)
            effect_c = (c_obs.treatment_mean - c_obs.control_mean) / c_obs.control_mean if c_obs.control_mean else 0
            effect_k = (k_obs.treatment_mean - k_obs.control_mean) / k_obs.control_mean if k_obs.control_mean else 0

            # Create corrected observation using the one with expected direction
            if effect_c < effect_k:  # Claude has more negative effect
                corrected = Observation(
                    element=c_obs.element,
                    tissue=c_obs.tissue,
                    treatment_mean=c_obs.treatment_mean,
                    control_mean=c_obs.control_mean,
                    treatment_variance=c_obs.treatment_variance or k_obs.treatment_variance,
                    control_variance=c_obs.control_variance or k_obs.control_variance,
                    variance_type=c_obs.variance_type or k_obs.variance_type,
                    n=c_obs.n or k_obs.n,
                    unit=c_obs.unit or k_obs.unit,
                    data_source=c_obs.data_source,
                    treatment_description=c_obs.treatment_description,
                    control_description=c_obs.control_description,
                    moderators={**k_obs.moderators, **c_obs.moderators},
                    confidence="medium",
                    notes=f"SWAP CORRECTED: Kimi had T/C swapped, using Claude values (effect={effect_c:+.1%})"
                )
            else:  # Kimi has more negative effect
                corrected = Observation(
                    element=k_obs.element,
                    tissue=k_obs.tissue,
                    treatment_mean=k_obs.treatment_mean,
                    control_mean=k_obs.control_mean,
                    treatment_variance=k_obs.treatment_variance or c_obs.treatment_variance,
                    control_variance=k_obs.control_variance or c_obs.control_variance,
                    variance_type=k_obs.variance_type or c_obs.variance_type,
                    n=k_obs.n or c_obs.n,
                    unit=k_obs.unit or c_obs.unit,
                    data_source=k_obs.data_source,
                    treatment_description=k_obs.treatment_description,
                    control_description=k_obs.control_description,
                    moderators={**c_obs.moderators, **k_obs.moderators},
                    confidence="medium",
                    notes=f"SWAP CORRECTED: Claude had T/C swapped, using Kimi values (effect={effect_k:+.1%})"
                )

            if corrected.control_mean and corrected.control_mean != 0:
                corrected.effect_pct = ((corrected.treatment_mean - corrected.control_mean) / corrected.control_mean) * 100

            return True, corrected, "swapped_corrected"

    return False, None, "no_match"


def verify_observation(obs: Observation, expected_direction: str = "negative",
                       typical_range: Tuple[float, float] = None) -> Dict:
    """
    Run all verification checks on an observation.

    Args:
        obs: Observation to verify
        expected_direction: "negative" or "positive" expected effect direction
        typical_range: (min, max) expected effect range as decimals.
                       If None, uses a generic biological range based on direction.
    """
    # Guard against non-numeric values from malformed LLM output
    if not isinstance(obs.treatment_mean, (int, float)):
        return {"element": obs.element, "treatment_mean": obs.treatment_mean,
                "control_mean": obs.control_mean, "checks": {},
                "warning": "treatment_mean is not numeric"}
    if not isinstance(obs.control_mean, (int, float)):
        return {"element": obs.element, "treatment_mean": obs.treatment_mean,
                "control_mean": obs.control_mean, "checks": {},
                "warning": "control_mean is not numeric"}

    # Set universal defaults based on expected direction
    if typical_range is None:
        if expected_direction == "negative":
            typical_range = (-0.50, 0.15)  # -50% to +15%
        elif expected_direction == "positive":
            typical_range = (-0.15, 0.50)  # -15% to +50%
        else:
            typical_range = (-0.50, 0.50)  # Any direction

    results = {
        "element": obs.element,
        "treatment_mean": obs.treatment_mean,
        "control_mean": obs.control_mean,
        "checks": {}
    }

    # 1. Effect direction check
    if obs.treatment_mean and obs.control_mean:
        effect = (obs.treatment_mean - obs.control_mean) / obs.control_mean * 100
        results["effect_pct"] = effect

        if expected_direction == "negative":
            results["checks"]["direction"] = {
                "passed": effect < 0,
                "note": f"Effect is {effect:+.1f}%, expected negative"
            }
        elif expected_direction == "positive":
            results["checks"]["direction"] = {
                "passed": effect > 0,
                "note": f"Effect is {effect:+.1f}%, expected positive"
            }

        # Flag extreme effects
        if abs(effect) > 100:
            results["checks"]["magnitude"] = {
                "passed": False,
                "note": f"Effect of {effect:+.1f}% seems extreme"
            }

        # T/C swap detection using mathematical signature
        effect_decimal = effect / 100
        is_swapped, corrected_effect, swap_reason = detect_tc_swap(
            effect_decimal,
            expected_direction=expected_direction,
            typical_range=typical_range
        )
        results["checks"]["tc_swap"] = {
            "passed": not is_swapped,
            "likely_swapped": is_swapped,
            "corrected_effect": corrected_effect * 100 if is_swapped else None,
            "note": swap_reason
        }
        if is_swapped:
            results["swap_warning"] = f"⚠️ LIKELY T/C SWAP: Current effect {effect:+.1f}%, corrected would be {corrected_effect*100:+.1f}%"

    # 2. GRIM test (if n available and means look like they could be from count data)
    try:
        n_val = int(obs.n) if obs.n else 0
    except (ValueError, TypeError):
        n_val = 0
    if n_val > 0 and obs.treatment_mean is not None and obs.control_mean is not None:
        grim_t = grim_test(obs.treatment_mean, n_val)
        grim_c = grim_test(obs.control_mean, n_val)
        results["checks"]["grim"] = {
            "passed": grim_t and grim_c,
            "treatment_valid": grim_t,
            "control_valid": grim_c
        }

    # 3. CV check
    if obs.treatment_variance and obs.variance_type and obs.treatment_mean is not None:
        cv_ok, cv = cv_check(obs.treatment_mean, obs.treatment_variance, obs.variance_type, n_val if n_val > 0 else None)
        results["checks"]["cv"] = {
            "passed": cv_ok,
            "cv_value": cv,
            "note": f"CV = {cv:.1f}%"
        }

    # 4. Sample size plausibility check
    if n_val > 0:
        n_suspicious = n_val > 12
        results["checks"]["sample_size"] = {
            "passed": not n_suspicious,
            "n": n_val,
            "note": f"n={n_val} seems too large for a typical experiment" if n_suspicious else f"n={n_val}"
        }

    # 5. Variance type verification (if we have enough info)
    if obs.treatment_variance and n_val > 0 and obs.treatment_mean and obs.control_mean:
        var_type, conf, reason = p_value_triangulation(
            obs.treatment_mean, obs.control_mean,
            n_val, n_val,
            obs.treatment_variance
        )
        if var_type != "unknown" and obs.variance_type:
            matches = var_type.upper() == obs.variance_type.upper()
            results["checks"]["variance_type"] = {
                "passed": matches,
                "reported": obs.variance_type,
                "calculated": var_type,
                "confidence": conf,
                "reason": reason
            }

    return results


# =============================================================================
# JSON PARSING
# =============================================================================

def robust_json_parse(text: str) -> Optional[Any]:
    """Parse JSON robustly from LLM output."""
    if not text:
        return None

    # Extract from code blocks
    if "```json" in text:
        match = re.search(r'```json\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1)
    elif "```" in text:
        match = re.search(r'```\s*([\s\S]*?)```', text)
        if match:
            text = match.group(1)

    # Find JSON object or array
    for start_char, end_char in [('{', '}'), ('[', ']')]:
        start = text.find(start_char)
        if start == -1:
            continue

        depth = 0
        end = start
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in '[{':
                depth += 1
            elif char in ']}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        json_text = text[start:end]

        # Fix trailing commas
        json_text = re.sub(r',\s*([}\]])', r'\1', json_text)

        # Try to fix unclosed brackets
        if depth > 0:
            json_text = json_text.rstrip().rstrip(',')
            json_text += '}' * (json_text.count('{') - json_text.count('}'))
            json_text += ']' * (json_text.count('[') - json_text.count(']'))

        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            continue

    return None


# =============================================================================
# STAGE 1: RECONNAISSANCE
# =============================================================================

def get_recon_prompt(config: MetaAnalysisConfig) -> str:
    """Generate the reconnaissance prompt."""
    return f"""You are analyzing a scientific paper for meta-analysis data extraction.

META-ANALYSIS CONTEXT:
{config.description}

INTERVENTION: {config.intervention}
CONTROL: {config.control}
EXPECTED EFFECT DIRECTION: {config.expected_direction} ({config.typical_effect_size})

PRIMARY OUTCOMES TO LOOK FOR:
{', '.join(config.primary_outcomes) if config.primary_outcomes else 'All quantitative outcomes'}

Your task is to ANALYZE this paper and identify:
1. Potential issues that could cause extraction errors
2. The variance type used (SE, SD, LSD, etc.)
3. The exact definition of treatment vs control
4. Which tables contain the PRIMARY OUTCOME data (see above) vs other data (biomass, yield, etc.)
5. Any treatment/control confusion risks

CRITICAL WARNINGS TO CHECK:
{chr(10).join(f"- {w}" for w in config.tc_confusion_warnings)}

OUTPUT FORMAT (JSON):
{{
    "paper_summary": {{
        "title": "paper title",
        "species": "plant species studied",
        "experimental_system": "FACE | OTC | greenhouse | growth chamber",
        "co2_levels": {{"control": "ppm", "elevated": "ppm"}}
    }},
    "warnings": [
        "List any potential issues that could cause extraction errors",
        "Flag if T/C assignment is ambiguous",
        "Note any unusual reporting conventions"
    ],
    "variance_detection": {{
        "type": "SE | SD | LSD | CV | CI | none",
        "source": "Quote the exact text that defines variance type",
        "confidence": "high | medium | low"
    }},
    "treatment_control": {{
        "control_definition": "Exact definition of control (e.g., 'Ambient CO2 at 380 ppm')",
        "treatment_definition": "Exact definition of treatment (e.g., 'Elevated CO2 at 700 ppm')",
        "identification_method": "How to identify T vs C in tables",
        "potential_confusion": "Any risk of T/C swap, or null if none"
    }},
    "sample_size": {{
        "n": number or null,
        "source": "Quote where n was found",
        "applies_to": "all observations | specific tables",
        "note": "n = number of INDEPENDENT EXPERIMENTAL REPLICATES per treatment group (e.g., plots, chambers, pots). NOT total plants, NOT total harvests, NOT total samples pooled. Typically 3-10 in plant biology experiments."
    }},
    "data_locations": {{
        "tables_with_target_data": ["Table 1", "Table 3"],
        "tables_without_target_data": ["Table 2"],
        "figures_with_data": ["Figure 2"],
        "table_descriptions": {{
            "Table 1": "brief description of what this table contains",
            "Table 2": "brief description"
        }},
        "outcome_variables_detected": {{
            "Table 1": ["list", "every", "specific", "outcome", "variable", "in", "this", "table"],
            "Table 3": ["list", "every", "specific", "outcome", "variable", "in", "this", "table"]
        }}
    }},
    "experimental_design": {{
        "type": "factorial | simple | split-plot",
        "factors": ["CO2", "cultivar", "year"],
        "total_combinations": number
    }},
    "extraction_guidance": "Specific instructions for extracting from this paper accurately"
}}

Return ONLY the JSON object."""


class ReconPass:
    """Stage 1: Claude reconnaissance pass with challenge-aware detection."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", taxonomy_path: str = None):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/"
        )
        self.model = model

        # Initialize challenge detector if available
        self.challenge_detector = None
        if CHALLENGE_AWARE_AVAILABLE:
            default_taxonomy = "output/challenge_evaluation_2026-02-03/kimi/challenges_kimi.json"
            taxonomy = taxonomy_path or (default_taxonomy if os.path.exists(default_taxonomy) else None)
            self.challenge_detector = ChallengeDetector(taxonomy)

    def _detect_challenges(self, paper_id: str, pdf_text: str) -> Optional[dict]:
        """Detect paper challenges using challenge-aware module."""
        if not self.challenge_detector:
            return None

        try:
            detection = self.challenge_detector.detect(paper_id, pdf_text)
            return {
                'extraction_method': detection.recommended_method.value,
                'extraction_method_reason': detection.routing_reason,
                'challenge_hints': detection.extraction_hints,
                'warnings': detection.warnings,
                'is_scanned': detection.is_scanned,
                'is_fig_only': detection.is_fig_only,
                'has_image_tables': detection.has_image_tables,
                'estimated_difficulty': detection.estimated_difficulty
            }
        except Exception as e:
            print(f"  Challenge detection error: {e}")
            return None

    def analyze(self, pdf_text: str, config: MetaAnalysisConfig, paper_id: str = "unknown") -> ReconResult:
        """Analyze paper and return structured reconnaissance with challenge detection."""

        # First, run challenge detection (fast, pattern-based)
        challenge_info = self._detect_challenges(paper_id, pdf_text)

        prompt = get_recon_prompt(config)

        # Truncate text to fit context
        max_chars = 150000
        text_to_send = pdf_text[:max_chars]

        # Default challenge values
        extraction_method = "text"
        extraction_method_reason = "Standard text extraction"
        challenge_hints = []
        is_scanned = False
        is_fig_only = False
        has_image_tables = False
        estimated_difficulty = "MEDIUM"

        if challenge_info:
            extraction_method = challenge_info.get('extraction_method', 'text')
            extraction_method_reason = challenge_info.get('extraction_method_reason', '')
            challenge_hints = challenge_info.get('challenge_hints', [])
            is_scanned = challenge_info.get('is_scanned', False)
            is_fig_only = challenge_info.get('is_fig_only', False)
            has_image_tables = challenge_info.get('has_image_tables', False)
            estimated_difficulty = challenge_info.get('estimated_difficulty', 'MEDIUM')

        # Run deterministic global variance scan (regex-based, fast)
        regex_var_type, regex_var_evidence, regex_var_conf = scan_global_variance_type(pdf_text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": f"PAPER TEXT:\n\n{text_to_send}\n\n{prompt}"}
                ]
            )

            content = response.choices[0].message.content
            result = robust_json_parse(content)

            if result is None:
                # Merge challenge warnings with parse failure
                warnings = ["Failed to parse recon response"]
                if challenge_info and challenge_info.get('warnings'):
                    warnings.extend(challenge_info['warnings'])

                return ReconResult(
                    paper_id=paper_id,
                    warnings=warnings,
                    variance_type=None,
                    variance_source=None,
                    variance_confidence="low",
                    control_definition="",
                    treatment_definition="",
                    tables_with_target_data=[],
                    potential_tc_confusion=None,
                    experimental_design="",
                    sample_size_found=None,
                    sample_size_source=None,
                    factorial_structure=None,
                    extraction_guidance="",
                    raw_response=content[:2000],
                    extraction_method=extraction_method,
                    extraction_method_reason=extraction_method_reason,
                    challenge_hints=challenge_hints,
                    is_scanned=is_scanned,
                    is_fig_only=is_fig_only,
                    has_image_tables=has_image_tables,
                    estimated_difficulty=estimated_difficulty
                )

            # Extract structured result (guard against LLM returning lists instead of dicts)
            var_info = result.get("variance_detection", {})
            if not isinstance(var_info, dict): var_info = {}
            tc_info = result.get("treatment_control", {})
            if not isinstance(tc_info, dict): tc_info = {}
            sample_info = result.get("sample_size", {})
            if not isinstance(sample_info, dict): sample_info = {}
            data_locs = result.get("data_locations", {})
            if not isinstance(data_locs, dict): data_locs = {}
            design_info = result.get("experimental_design", {})
            if not isinstance(design_info, dict): design_info = {}

            # Merge warnings from LLM recon with challenge detection warnings
            warnings = result.get("warnings", [])
            if challenge_info and challenge_info.get('warnings'):
                warnings.extend(challenge_info['warnings'])

            # Determine variance type: prefer regex (deterministic) over LLM when high confidence
            final_var_type = var_info.get("type")
            final_var_source = var_info.get("source")
            final_var_confidence = var_info.get("confidence", "low")

            if regex_var_type and regex_var_conf in ("high", "medium"):
                llm_var = (var_info.get("type") or "").upper().replace("SEM", "SE")
                if llm_var != regex_var_type:
                    warnings.append(
                        f"Variance type override: LLM said '{llm_var}', "
                        f"regex scan found '{regex_var_type}' ({regex_var_conf}): {regex_var_evidence}"
                    )
                final_var_type = regex_var_type
                final_var_source = regex_var_evidence
                final_var_confidence = regex_var_conf

            return ReconResult(
                paper_id=paper_id,
                warnings=warnings,
                variance_type=final_var_type,
                variance_source=final_var_source,
                variance_confidence=final_var_confidence,
                control_definition=tc_info.get("control_definition", ""),
                treatment_definition=tc_info.get("treatment_definition", ""),
                tables_with_target_data=data_locs.get("tables_with_target_data", []),
                tables_without_target_data=data_locs.get("tables_without_target_data", []),
                table_descriptions=data_locs.get("table_descriptions", {}),
                outcome_variables_detected=data_locs.get("outcome_variables_detected", {}),
                figures_with_data=data_locs.get("figures_with_data", []),
                potential_tc_confusion=tc_info.get("potential_confusion"),
                experimental_design=design_info.get("type", ""),
                sample_size_found=sample_info.get("n"),
                sample_size_source=sample_info.get("source"),
                factorial_structure=json.dumps(design_info.get("factors", [])),
                extraction_guidance=result.get("extraction_guidance", ""),
                raw_response=content[:2000],
                extraction_method=extraction_method,
                extraction_method_reason=extraction_method_reason,
                challenge_hints=challenge_hints,
                is_scanned=is_scanned,
                is_fig_only=is_fig_only,
                has_image_tables=has_image_tables,
                estimated_difficulty=estimated_difficulty
            )

        except Exception as e:
            # Even on error, include challenge info if available
            warnings = [f"Recon error: {str(e)}"]
            if challenge_info and challenge_info.get('warnings'):
                warnings.extend(challenge_info['warnings'])

            return ReconResult(
                paper_id=paper_id,
                warnings=warnings,
                variance_type=None,
                variance_source=None,
                variance_confidence="low",
                control_definition="",
                treatment_definition="",
                tables_with_target_data=[],
                potential_tc_confusion=None,
                experimental_design="",
                sample_size_found=None,
                sample_size_source=None,
                factorial_structure=None,
                extraction_guidance="",
                raw_response="",
                extraction_method=extraction_method,
                extraction_method_reason=extraction_method_reason,
                challenge_hints=challenge_hints,
                is_scanned=is_scanned,
                is_fig_only=is_fig_only,
                has_image_tables=has_image_tables,
                estimated_difficulty=estimated_difficulty
            )


# =============================================================================
# STAGE 2: UNIFIED EXTRACTION
# =============================================================================

def get_unified_extraction_prompt(config: MetaAnalysisConfig, recon: ReconResult) -> str:
    """Generate extraction prompt with recon hints and challenge-specific guidance embedded."""

    # Build warnings section
    warnings_section = ""
    if recon.warnings:
        warnings_section = "WARNINGS FROM PAPER ANALYSIS:\n" + "\n".join(f"- {w}" for w in recon.warnings)

    # Build challenge-specific hints section
    challenge_section = ""
    if recon.challenge_hints:
        challenge_section = "\nCHALLENGE-SPECIFIC EXTRACTION HINTS:\n" + "\n".join(f"- {h}" for h in recon.challenge_hints)

    # Add extraction method warning
    method_warning = ""
    if recon.extraction_method in ("vision", "hybrid"):
        method_warning = f"""
NOTE: This paper has been flagged for {recon.extraction_method.upper()} extraction.
Reason: {recon.extraction_method_reason}
Some data may be in figures or image-based tables that text extraction cannot read.
Extract what you can from the text, and note any tables/figures that appear to be images.
"""
    elif recon.extraction_method == "manual":
        method_warning = """
WARNING: This paper has been flagged for MANUAL review.
Data may be unreliable or inaccessible via automated extraction.
Extract what you can but flag low-confidence observations.
"""

    # Build variance hint
    variance_hint = ""
    if recon.variance_type:
        variance_hint = f"""
VARIANCE TYPE IDENTIFIED: {recon.variance_type}
Source: {recon.variance_source or 'Not specified'}
Confidence: {recon.variance_confidence}
-> Extract variance values as {recon.variance_type}
"""

    # Build T/C guidance
    tc_guidance = ""
    if recon.control_definition or recon.treatment_definition:
        tc_guidance = f"""
TREATMENT vs CONTROL:
- CONTROL: {recon.control_definition or 'Not specified'}
- TREATMENT: {recon.treatment_definition or 'Not specified'}
"""
    if recon.potential_tc_confusion:
        tc_guidance += f"\nPOTENTIAL CONFUSION: {recon.potential_tc_confusion}"

    # Factorial design warning
    factorial_hint = ""
    if recon.experimental_design and 'factorial' in recon.experimental_design.lower():
        factors = []
        if recon.factorial_structure:
            try:
                factors = json.loads(recon.factorial_structure)
            except (json.JSONDecodeError, TypeError):
                pass
        factor_str = ' × '.join(factors) if factors else "multiple factors"
        factorial_hint = f"""
⚠️ FACTORIAL DESIGN DETECTED ({factor_str})
This paper has a factorial design. You MUST extract EACH combination of factors as a SEPARATE observation:
→ For EACH level of each secondary factor, extract the treatment vs control comparison for the primary factor SEPARATELY
→ Record each factor level as a moderator (e.g., cultivar, O3_level, K_treatment, year, tissue_type)
→ Do NOT average across factor levels to get a single "main effect"
→ If the table has 4 cultivars × 3 harvests × 10 elements, you should have ~120 observations, NOT 10
→ The downstream meta-analyst will decide how to aggregate — your job is to preserve maximum granularity"""

    # Sample size hint
    n_hint = "\nSAMPLE SIZE NOTE: n must be the number of independent experimental replicates per treatment group (e.g., plots, chambers, pots). Do NOT use total plants harvested, total samples pooled, or number of harvest dates. In plant biology, n is typically 3-10."
    if recon.sample_size_found:
        n_hint += f"\nSAMPLE SIZE FROM PAPER: n = {recon.sample_size_found} (from: {recon.sample_size_source or 'paper'})"

    # Target tables - strong directive
    tables_hint = ""
    if recon.tables_with_target_data:
        table_list = ', '.join(recon.tables_with_target_data)
        # Add table descriptions if available
        desc_lines = ""
        if recon.table_descriptions:
            desc_lines = "\n  Table contents identified by paper analysis:"
            for tbl, desc in recon.table_descriptions.items():
                marker = " ← TARGET" if tbl in recon.tables_with_target_data else " ← SKIP"
                desc_lines += f"\n    {tbl}: {desc}{marker}"

        # Add skip list
        skip_list = ""
        if recon.tables_without_target_data:
            skip_list = f"\n→ SKIP these tables (no target data): {', '.join(recon.tables_without_target_data)}"

        tables_hint = f"""
MANDATORY TABLE TARGETING:
→ EXTRACT FIRST from these tables: {table_list}
  These tables contain the primary outcome data for this meta-analysis.
→ FULLY extract ALL rows from these tables before looking at other tables.{skip_list}{desc_lines}
→ Other tables may contain secondary data — extract those ONLY AFTER the target tables are complete.
→ If you find yourself extracting biomass, yield, or other non-target outcomes, STOP and check if you are in the right table."""

    # Outcome variables checklist (from recon)
    outcome_vars_hint = ""
    all_detected_vars = set()
    if recon.outcome_variables_detected:
        per_table = []
        for tbl, vars_list in recon.outcome_variables_detected.items():
            if vars_list:
                all_detected_vars.update(vars_list)
                per_table.append(f"  {tbl}: {', '.join(vars_list)}")
        if all_detected_vars:
            outcome_vars_hint = f"""
OUTCOME VARIABLES DETECTED IN THIS PAPER:
{chr(10).join(per_table)}
→ You MUST extract ALL of these variables. Do not skip any.
→ Total unique variables to extract: {len(all_detected_vars)} ({', '.join(sorted(all_detected_vars))})
→ After extraction, verify your count matches. If you extracted fewer variables than listed, go back and find the missing ones."""

    # Build dynamic element example for JSON schema
    if all_detected_vars:
        element_example = ' | '.join(sorted(all_detected_vars))
    else:
        element_example = "all outcome variables reported in the paper"

    # Difficulty indicator
    difficulty_hint = ""
    if recon.estimated_difficulty == "HARD":
        difficulty_hint = "\nDIFFICULTY: HARD - Take extra care with extraction accuracy"

    return f"""You are extracting quantitative data for meta-analysis.

META-ANALYSIS TOPIC: {config.name}
{config.description}

WHAT TO EXTRACT:
- PRIMARY: {', '.join(config.primary_outcomes) if config.primary_outcomes else 'All quantitative outcomes'}
- Include: treatment mean, control mean, variance (SE/SD), sample size (n), units
- Record: tissue type, treatment details, data source (table/figure)

{warnings_section}
{challenge_section}
{method_warning}
{variance_hint}

{tc_guidance}
{factorial_hint}
{n_hint}

{tables_hint}
{outcome_vars_hint}
{difficulty_hint}

EXTRACTION GUIDANCE FROM PAPER ANALYSIS:
{recon.extraction_guidance}

CRITICAL RULES:

⚠️ ABSOLUTELY NO POOLING - THIS IS THE MOST IMPORTANT RULE ⚠️

You MUST extract EVERY SINGLE ROW from every data table separately.
DO NOT average, combine, or pool values across:
- Different years (2017 vs 2018 = 2 separate observations)
- Different cultivars (Cultivar A vs Cultivar B = 2 separate observations)
- Different tissues (leaf vs root = 2 separate observations)
- Different treatment levels (Low N vs High N = 2 separate observations)
- Different sampling times (Week 4 vs Week 8 = 2 separate observations)

EXAMPLE - WRONG (pooled):
  Table has 4 cultivars × 2 years = 8 rows
  You extract: 1 observation with "averaged across cultivars and years" ❌

EXAMPLE - CORRECT (granular):
  Table has 4 cultivars × 2 years = 8 rows
  You extract: 8 separate observations, one per row ✓

If a table has 20 data rows, you should have approximately 20 observations.
If a table has 50 data rows, you should have approximately 50 observations.

Count the rows in each table and ensure your observation count matches!

OTHER RULES:
1. Extract ACTUAL variance numbers (the numeric values), not just the type
2. Record tissue type for every observation
3. Include moderators (cultivar, year, site, treatment level) when available
4. Each unique combination of factors = one observation

EXPECTED EFFECT DIRECTION: {config.expected_direction}
- If you extract effects that are strongly opposite to expected, double-check T/C assignment

OUTPUT FORMAT (JSON):
{{
    "paper_info": {{
        "title": "...",
        "authors": "...",
        "year": YYYY,
        "species": "..."
    }},
    "observations": [
        {{
            "element": "{element_example}",
            "tissue": "grain | leaf | root | shoot | whole plant",
            "treatment_mean": number,
            "control_mean": number,
            "treatment_variance": number or null,
            "control_variance": number or null,
            "variance_type": "SE | SD | LSD | null",
            "n": "number or null — IMPORTANT: n = independent experimental replicates per treatment (plots, chambers, pots), NOT total plants, NOT total harvests, NOT pooled individuals. Typically 3-10.",
            "unit": "mg/kg | g/kg | % | ppm | etc.",
            "data_source": "Table 1 | Figure 2 | etc.",
            "treatment_description": "Elevated CO2 (700 ppm)",
            "control_description": "Ambient CO2 (400 ppm)",
            "moderators": {{
                "cultivar": "...",
                "year": "...",
                "site": "..."
            }},
            "confidence": "high | medium | low",
            "notes": "any extraction notes"
        }}
    ],
    "extraction_notes": "overall notes about extraction quality/issues"
}}

Return ONLY the JSON object."""


class UnifiedExtractor:
    """Extracts data using either Claude or Kimi with unified prompt."""

    # Pricing (per million tokens)
    CLAUDE_PRICE_IN = 3.00
    CLAUDE_PRICE_OUT = 15.00
    KIMI_PRICE_IN = 0.50
    KIMI_PRICE_OUT = 2.80
    GEMINI_FLASH_PRICE_IN = 0.50
    GEMINI_FLASH_PRICE_OUT = 3.00

    def __init__(self):
        self.claude_client = None
        self.kimi_client = None

        # Initialize Claude using native Anthropic SDK
        claude_key = os.environ.get('ANTHROPIC_API_KEY')
        if claude_key:
            try:
                import anthropic
                self.claude_client = anthropic.Anthropic(api_key=claude_key)
            except ImportError:
                print("Warning: anthropic package not installed, using OpenAI wrapper")
                self.claude_client = openai.OpenAI(
                    api_key=claude_key,
                    base_url="https://api.anthropic.com/v1/"
                )

        # Initialize Kimi
        kimi_key = os.environ.get('MOONSHOT_API_KEY')
        if kimi_key:
            self.kimi_client = openai.OpenAI(
                api_key=kimi_key,
                base_url="https://api.moonshot.ai/v1"
            )

        # Initialize Gemini (for tiebreaker)
        self.gemini_client = None
        google_key = os.environ.get('GOOGLE_API_KEY')
        if google_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=google_key)
            except ImportError:
                pass

    def extract_claude(
        self,
        pdf_text: str,
        config: MetaAnalysisConfig,
        recon: ReconResult,
        paper_id: str
    ) -> ExtractionResult:
        """Extract using Claude."""
        if not self.claude_client:
            return ExtractionResult(
                paper_id=paper_id,
                model="claude",
                observations=[],
                paper_info={},
                extraction_notes="Claude API key not available"
            )

        prompt = get_unified_extraction_prompt(config, recon)

        try:
            # Use native Anthropic SDK format with streaming for long requests
            import anthropic
            if isinstance(self.claude_client, anthropic.Anthropic):
                # Use streaming to handle long requests (Anthropic requirement)
                content_parts = []
                with self.claude_client.messages.stream(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32768,
                    messages=[
                        {"role": "user", "content": f"PAPER TEXT:\n\n{pdf_text[:150000]}\n\n{prompt}"}
                    ]
                ) as stream:
                    for text in stream.text_stream:
                        content_parts.append(text)
                content = "".join(content_parts)
                # Get final message for usage stats
                final_message = stream.get_final_message()
                tokens_in = final_message.usage.input_tokens
                tokens_out = final_message.usage.output_tokens
            else:
                # Fallback to OpenAI format
                response = self.claude_client.chat.completions.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=32768,
                    messages=[
                        {"role": "user", "content": f"PAPER TEXT:\n\n{pdf_text[:150000]}\n\n{prompt}"}
                    ]
                )
                content = response.choices[0].message.content
                usage = getattr(response, 'usage', None)
                tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
                tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0
            result = robust_json_parse(content)

            # Calculate cost (tokens already set above based on SDK type)
            cost = (tokens_in * self.CLAUDE_PRICE_IN + tokens_out * self.CLAUDE_PRICE_OUT) / 1_000_000

            if result is None:
                return ExtractionResult(
                    paper_id=paper_id,
                    model="claude",
                    observations=[],
                    paper_info={},
                    extraction_notes=f"JSON parse failed. Raw: {content[:500]}",
                    tokens_used=tokens_in + tokens_out,
                    cost_estimate=cost
                )

            # Parse observations
            observations = []
            for obs_dict in result.get("observations", []):
                obs = Observation(
                    element=obs_dict.get("element", ""),
                    tissue=obs_dict.get("tissue", ""),
                    treatment_mean=obs_dict.get("treatment_mean", 0),
                    control_mean=obs_dict.get("control_mean", 0),
                    treatment_variance=obs_dict.get("treatment_variance"),
                    control_variance=obs_dict.get("control_variance"),
                    variance_type=obs_dict.get("variance_type"),
                    n=obs_dict.get("n"),
                    unit=obs_dict.get("unit", ""),
                    data_source=obs_dict.get("data_source", ""),
                    treatment_description=obs_dict.get("treatment_description", ""),
                    control_description=obs_dict.get("control_description", ""),
                    moderators=obs_dict.get("moderators", {}),
                    confidence=obs_dict.get("confidence", "medium"),
                    notes=obs_dict.get("notes", "")
                )

                # Calculate effect
                if obs.control_mean and obs.control_mean != 0 and obs.treatment_mean is not None:
                    obs.effect_pct = ((obs.treatment_mean - obs.control_mean) / obs.control_mean) * 100

                observations.append(obs)

            return ExtractionResult(
                paper_id=paper_id,
                model="claude",
                observations=observations,
                paper_info=result.get("paper_info", {}),
                extraction_notes=result.get("extraction_notes", ""),
                tokens_used=tokens_in + tokens_out,
                cost_estimate=cost
            )

        except Exception as e:
            return ExtractionResult(
                paper_id=paper_id,
                model="claude",
                observations=[],
                paper_info={},
                extraction_notes=f"Error: {str(e)}"
            )

    def extract_kimi(
        self,
        pdf_text: str,
        config: MetaAnalysisConfig,
        recon: ReconResult,
        paper_id: str
    ) -> ExtractionResult:
        """Extract using Kimi."""
        if not self.kimi_client:
            return ExtractionResult(
                paper_id=paper_id,
                model="kimi",
                observations=[],
                paper_info={},
                extraction_notes="Kimi API key not available"
            )

        prompt = get_unified_extraction_prompt(config, recon)

        try:
            # Use ThreadPoolExecutor with timeout to prevent Kimi API hangs
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
            def _kimi_call():
                return self.kimi_client.chat.completions.create(
                    model="kimi-k2.5",
                    max_tokens=65536,
                    temperature=1.0,
                    messages=[
                        {"role": "system", "content": f"PAPER TEXT:\n\n{pdf_text[:200000]}"},
                        {"role": "user", "content": prompt}
                    ],
                    extra_body={"thinking": {"type": "enabled"}}
                )
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_kimi_call)
                response = future.result(timeout=300)  # 5 minute timeout

            content = response.choices[0].message.content
            result = robust_json_parse(content)

            # Calculate cost
            usage = getattr(response, 'usage', None)
            tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
            tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0
            cost = (tokens_in * self.KIMI_PRICE_IN + tokens_out * self.KIMI_PRICE_OUT) / 1_000_000

            if result is None:
                return ExtractionResult(
                    paper_id=paper_id,
                    model="kimi",
                    observations=[],
                    paper_info={},
                    extraction_notes=f"JSON parse failed. Raw: {content[:500]}",
                    tokens_used=tokens_in + tokens_out,
                    cost_estimate=cost
                )

            # Parse observations
            observations = []
            for obs_dict in result.get("observations", []):
                obs = Observation(
                    element=obs_dict.get("element", ""),
                    tissue=obs_dict.get("tissue", ""),
                    treatment_mean=obs_dict.get("treatment_mean", 0),
                    control_mean=obs_dict.get("control_mean", 0),
                    treatment_variance=obs_dict.get("treatment_variance"),
                    control_variance=obs_dict.get("control_variance"),
                    variance_type=obs_dict.get("variance_type"),
                    n=obs_dict.get("n"),
                    unit=obs_dict.get("unit", ""),
                    data_source=obs_dict.get("data_source", ""),
                    treatment_description=obs_dict.get("treatment_description", ""),
                    control_description=obs_dict.get("control_description", ""),
                    moderators=obs_dict.get("moderators", {}),
                    confidence=obs_dict.get("confidence", "medium"),
                    notes=obs_dict.get("notes", "")
                )

                # Calculate effect
                if obs.control_mean and obs.control_mean != 0 and obs.treatment_mean is not None:
                    obs.effect_pct = ((obs.treatment_mean - obs.control_mean) / obs.control_mean) * 100

                observations.append(obs)

            return ExtractionResult(
                paper_id=paper_id,
                model="kimi",
                observations=observations,
                paper_info=result.get("paper_info", {}),
                extraction_notes=result.get("extraction_notes", ""),
                tokens_used=tokens_in + tokens_out,
                cost_estimate=cost
            )

        except Exception as e:
            error_type = "TIMEOUT" if "TimeoutError" in type(e).__name__ or "timeout" in str(e).lower() else "Error"
            print(f"    Kimi {error_type}: {str(e)[:200]}")
            return ExtractionResult(
                paper_id=paper_id,
                model="kimi",
                observations=[],
                paper_info={},
                extraction_notes=f"{error_type}: {str(e)}"
            )

    def extract_gemini(
        self,
        pdf_text: str,
        config: MetaAnalysisConfig,
        recon: ReconResult,
        paper_id: str
    ) -> ExtractionResult:
        """Extract using Gemini Flash (text-based tiebreaker)."""
        if not self.gemini_client:
            return ExtractionResult(
                paper_id=paper_id,
                model="gemini",
                observations=[],
                paper_info={},
                extraction_notes="Gemini API key not available"
            )

        prompt = get_unified_extraction_prompt(config, recon)

        try:
            from google.genai import types
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

            gen_config = types.GenerateContentConfig(
                max_output_tokens=32768,
                temperature=0.3,
            )

            # Gemini has 1M context — send up to 500K chars
            full_prompt = f"PAPER TEXT:\n\n{pdf_text[:500000]}\n\n{prompt}"

            def _gemini_call():
                return self.gemini_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=full_prompt,
                    config=gen_config,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_gemini_call)
                try:
                    response = future.result(timeout=180)  # 3 minute timeout
                except FuturesTimeout:
                    print(f"    Gemini text call timed out after 180s")
                    return ExtractionResult(
                        paper_id=paper_id, model="gemini",
                        observations=[], paper_info={},
                        extraction_notes="Gemini call timed out"
                    )

            content = response.text if response.text else ""
            result = robust_json_parse(content)

            # Estimate cost from usage metadata
            usage = getattr(response, 'usage_metadata', None)
            tokens_in = getattr(usage, 'prompt_token_count', 0) if usage else 0
            tokens_out = getattr(usage, 'candidates_token_count', 0) if usage else 0
            cost = (tokens_in * self.GEMINI_FLASH_PRICE_IN + tokens_out * self.GEMINI_FLASH_PRICE_OUT) / 1_000_000

            if result is None:
                return ExtractionResult(
                    paper_id=paper_id,
                    model="gemini",
                    observations=[],
                    paper_info={},
                    extraction_notes=f"JSON parse failed. Raw: {content[:500]}",
                    tokens_used=tokens_in + tokens_out,
                    cost_estimate=cost
                )

            # Parse observations (same pattern as extract_claude/extract_kimi)
            observations = []
            for obs_dict in result.get("observations", []):
                obs = Observation(
                    element=obs_dict.get("element", ""),
                    tissue=obs_dict.get("tissue", ""),
                    treatment_mean=obs_dict.get("treatment_mean", 0),
                    control_mean=obs_dict.get("control_mean", 0),
                    treatment_variance=obs_dict.get("treatment_variance"),
                    control_variance=obs_dict.get("control_variance"),
                    variance_type=obs_dict.get("variance_type"),
                    n=obs_dict.get("n"),
                    unit=obs_dict.get("unit", ""),
                    data_source=obs_dict.get("data_source", ""),
                    treatment_description=obs_dict.get("treatment_description", ""),
                    control_description=obs_dict.get("control_description", ""),
                    moderators=obs_dict.get("moderators", {}),
                    confidence=obs_dict.get("confidence", "medium"),
                    notes=obs_dict.get("notes", "")
                )

                # Calculate effect
                if obs.control_mean and obs.control_mean != 0 and obs.treatment_mean is not None:
                    obs.effect_pct = ((obs.treatment_mean - obs.control_mean) / obs.control_mean) * 100

                observations.append(obs)

            return ExtractionResult(
                paper_id=paper_id,
                model="gemini",
                observations=observations,
                paper_info=result.get("paper_info", {}),
                extraction_notes=result.get("extraction_notes", ""),
                tokens_used=tokens_in + tokens_out,
                cost_estimate=cost
            )

        except Exception as e:
            return ExtractionResult(
                paper_id=paper_id,
                model="gemini",
                observations=[],
                paper_info={},
                extraction_notes=f"Error: {str(e)}"
            )

    def extract_vision(
        self,
        pdf_path: str,
        config: MetaAnalysisConfig,
        recon: ReconResult,
        paper_id: str,
        provider: str = "google"
    ) -> ExtractionResult:
        """
        Extract data by sending page images to a vision-capable LLM.

        Used when routing detects VISION or HYBRID papers (FIG-ONLY,
        IMAGE-TABLES, SCANNED). Renders each page as an image and sends
        to the LLM with the same extraction prompt used for text.

        Args:
            pdf_path: Path to PDF file
            config: Meta-analysis configuration (PICO, outcomes, etc.)
            recon: Reconnaissance results (used for prompt construction)
            paper_id: Paper identifier
            provider: "google" (Gemini), "anthropic" (Claude), or "kimi" (Kimi K2.5)
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            return ExtractionResult(
                paper_id=paper_id, model=f"vision_{provider}",
                observations=[], paper_info={},
                extraction_notes="PyMuPDF not installed - cannot render pages"
            )

        # Determine which LLM client and model to use
        if provider == "google":
            google_key = os.environ.get('GOOGLE_API_KEY')
            if not google_key:
                return ExtractionResult(
                    paper_id=paper_id, model="vision_google",
                    observations=[], paper_info={},
                    extraction_notes="GOOGLE_API_KEY not available for vision"
                )
            try:
                from google import genai
                from google.genai import types
                google_client = genai.Client(api_key=google_key)
            except ImportError:
                return ExtractionResult(
                    paper_id=paper_id, model="vision_google",
                    observations=[], paper_info={},
                    extraction_notes="google-genai package not installed"
                )
            vision_model = "gemini-3-flash-preview"
        elif provider == "kimi":
            if not self.kimi_client:
                return ExtractionResult(
                    paper_id=paper_id, model="vision_kimi",
                    observations=[], paper_info={},
                    extraction_notes="Kimi client not available for vision"
                )
        else:
            # Anthropic vision
            if not self.claude_client:
                return ExtractionResult(
                    paper_id=paper_id, model="vision_anthropic",
                    observations=[], paper_info={},
                    extraction_notes="Claude client not available for vision"
                )

        # Build extraction prompt (same as text, but adapted for images)
        base_prompt = get_unified_extraction_prompt(config, recon)
        vision_prefix = """You are looking at page images from a scientific paper.
Extract ALL quantitative data visible in these pages - from tables, figures,
bar charts, line graphs, or any other data presentation.

Read values carefully from:
- Table cells (even if the table is an image)
- Bar chart heights (use y-axis scale)
- Error bars (note their type: SE, SD, etc.)
- Figure legends and captions for context

"""

        # Render PDF pages to images
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        dpi = 200
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        page_images = []
        for page_num in range(num_pages):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            page_images.append(png_bytes)

        doc.close()

        if not page_images:
            return ExtractionResult(
                paper_id=paper_id, model=f"vision_{provider}",
                observations=[], paper_info={},
                extraction_notes="No pages rendered from PDF"
            )

        # For large PDFs, batch pages to stay within context limits
        # Send up to 10 pages at a time (vision models handle ~20 images)
        all_observations = []
        batch_size = 10
        total_cost = 0.0
        total_tokens = 0

        for batch_start in range(0, len(page_images), batch_size):
            batch_end = min(batch_start + batch_size, len(page_images))
            batch_images = page_images[batch_start:batch_end]
            page_range = f"pages {batch_start+1}-{batch_end}"

            prompt = f"{vision_prefix}These are {page_range} of the paper.\n\n{base_prompt}"

            try:
                if provider == "google":
                    import base64
                    # Build content parts: images + prompt
                    contents = []
                    for img_bytes in batch_images:
                        contents.append(
                            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
                        )
                    contents.append(prompt)

                    gen_config = types.GenerateContentConfig(
                        max_output_tokens=32768,
                        temperature=0.3,
                    )

                    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
                    def _vision_call():
                        return google_client.models.generate_content(
                            model=vision_model,
                            contents=contents,
                            config=gen_config,
                        )
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_vision_call)
                        try:
                            response = future.result(timeout=180)  # 3 min timeout
                        except FuturesTimeout:
                            print(f"    Gemini vision call timed out after 180s")
                            content = ""
                            batch_cost = 0
                            continue

                    content = response.text if response.text else ""
                    # Estimate cost (Gemini Flash pricing)
                    batch_cost = 0.01 * len(batch_images)  # rough estimate
                    total_cost += batch_cost

                elif provider == "kimi":
                    # Kimi K2.5 vision (OpenAI-compatible API)
                    import base64
                    image_content = []
                    for img_bytes in batch_images:
                        img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
                        image_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        })
                    image_content.append({"type": "text", "text": prompt})

                    response = self.kimi_client.chat.completions.create(
                        model="kimi-k2.5",
                        max_tokens=32768,
                        messages=[{"role": "user", "content": image_content}],
                        extra_body={"thinking": {"type": "enabled"}}
                    )
                    content = response.choices[0].message.content
                    # Filter out thinking tags
                    if content and "<think>" in content:
                        import re as _re
                        content = _re.sub(r'<think>.*?</think>', '', content, flags=_re.DOTALL).strip()
                    usage = getattr(response, 'usage', None)
                    tokens_in = getattr(usage, 'prompt_tokens', 0) if usage else 0
                    tokens_out = getattr(usage, 'completion_tokens', 0) if usage else 0
                    batch_cost = (tokens_in * self.KIMI_PRICE_IN + tokens_out * self.KIMI_PRICE_OUT) / 1_000_000
                    total_cost += batch_cost
                    total_tokens += tokens_in + tokens_out

                else:
                    # Anthropic vision
                    import anthropic
                    import base64
                    image_content = []
                    for img_bytes in batch_images:
                        img_b64 = base64.standard_b64encode(img_bytes).decode('utf-8')
                        image_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64
                            }
                        })
                    image_content.append({"type": "text", "text": prompt})

                    if isinstance(self.claude_client, anthropic.Anthropic):
                        response = self.claude_client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=32768,
                            messages=[{"role": "user", "content": image_content}]
                        )
                        content = response.content[0].text
                        tokens_in = response.usage.input_tokens
                        tokens_out = response.usage.output_tokens
                        batch_cost = (tokens_in * self.CLAUDE_PRICE_IN + tokens_out * self.CLAUDE_PRICE_OUT) / 1_000_000
                        total_cost += batch_cost
                        total_tokens += tokens_in + tokens_out
                    else:
                        content = ""

                # Parse response
                result = robust_json_parse(content)
                if result and "observations" in result:
                    for obs_dict in result.get("observations", []):
                        obs = Observation(
                            element=obs_dict.get("element", ""),
                            tissue=obs_dict.get("tissue", ""),
                            treatment_mean=obs_dict.get("treatment_mean", 0),
                            control_mean=obs_dict.get("control_mean", 0),
                            treatment_variance=obs_dict.get("treatment_variance"),
                            control_variance=obs_dict.get("control_variance"),
                            variance_type=obs_dict.get("variance_type"),
                            n=obs_dict.get("n"),
                            unit=obs_dict.get("unit", ""),
                            data_source=obs_dict.get("data_source", ""),
                            treatment_description=obs_dict.get("treatment_description", ""),
                            control_description=obs_dict.get("control_description", ""),
                            moderators=obs_dict.get("moderators", {}),
                            confidence=obs_dict.get("confidence", "medium"),
                            notes=obs_dict.get("notes", "")
                        )
                        if obs.control_mean and obs.control_mean != 0 and obs.treatment_mean is not None:
                            obs.effect_pct = ((obs.treatment_mean - obs.control_mean) / obs.control_mean) * 100
                        all_observations.append(obs)

            except Exception as e:
                print(f"    Vision extraction error on {page_range}: {e}")
                continue

        return ExtractionResult(
            paper_id=paper_id,
            model=f"vision_{provider}",
            observations=all_observations,
            paper_info={},
            extraction_notes=f"Vision extraction: {len(all_observations)} obs from {num_pages} pages",
            tokens_used=total_tokens,
            cost_estimate=total_cost
        )


# =============================================================================
# CONSENSUS COMPARISON
# =============================================================================

def normalize_data_source(ds: str) -> str:
    """Normalize data source string for matching (e.g., 'Table 1a' -> 'table1')"""
    if not ds:
        return ""
    # Remove whitespace, parentheses, lowercase, remove section letters
    import re
    normalized = ds.lower().strip()
    normalized = re.sub(r'[()]', '', normalized)  # Remove parentheses
    normalized = re.sub(r'\s+', '', normalized)    # Remove spaces
    normalized = re.sub(r'[a-z]$', '', normalized) # Remove trailing letter like 'a' in 'Table 1a'
    return normalized


# Standard moderator field mappings (different names for same concept)
MODERATOR_ALIASES = {
    # Cultivar variations
    'cultivar': ['cultivar', 'variety', 'cv', 'genotype', 'cultivar_name', 'plant_variety'],
    # Year variations
    'year': ['year', 'experiment_year', 'study_year', 'growing_season', 'season_year'],
    # Site variations
    'site': ['site', 'location', 'study_site', 'experimental_site', 'site_name'],
    # Tissue variations
    'leaf_position': ['leaf_position', 'leaf_type', 'leaf_age', 'leaf_location'],
    # Nitrogen variations
    'nitrogen': ['nitrogen', 'n_level', 'nitrogen_treatment', 'n_treatment', 'nitrogen_level'],
    # AMF variations
    'amf': ['amf', 'mycorrhizal', 'mycorrhizal_status', 'amf_status', 'amf_treatment'],
}


def standardize_moderators(mods: dict) -> dict:
    """
    Standardize moderator field names and values for consistent matching.
    E.g., 'cv. BRM' -> 'brm', 'variety' -> 'cultivar'
    """
    if not mods:
        return {}

    import re
    standardized = {}

    for key, value in mods.items():
        # Standardize key name using aliases
        std_key = key.lower().strip()
        for canonical, aliases in MODERATOR_ALIASES.items():
            if std_key in aliases or std_key == canonical:
                std_key = canonical
                break

        # Standardize value
        if value is None:
            continue

        std_value = str(value).lower().strip()
        # Remove common prefixes
        std_value = re.sub(r'^(cv\.|cultivar|var\.|variety)\s*', '', std_value)
        # Remove quotes and extra whitespace
        std_value = re.sub(r'["\']', '', std_value)
        std_value = re.sub(r'\s+', '_', std_value)

        if std_value:
            standardized[std_key] = std_value

    return standardized


def moderator_key(mods: dict) -> str:
    """Create a hashable key from standardized moderators."""
    std = standardize_moderators(mods)
    # Sort keys for consistent ordering
    return "_".join(f"{k}:{v}" for k, v in sorted(std.items()))


def normalize_tissue(tissue: str) -> str:
    """
    Normalize tissue names for fuzzy matching.
    'outer leaf' -> 'leaf', 'old needles' -> 'needles', 'wheat grain' -> 'grain'
    """
    t = tissue.lower().strip()
    # Remove positional/age qualifiers
    for prefix in ['outer ', 'inner ', 'old ', 'new ', 'young ', 'mature ',
                    'upper ', 'lower ', 'fresh ', 'dried ', 'green ', 'senesced ']:
        t = t.replace(prefix, '')
    # Remove species qualifiers
    for prefix in ['wheat ', 'rice ', 'barley ', 'corn ', 'maize ']:
        t = t.replace(prefix, '')
    return t.strip()


def compare_observations(
    claude_obs: List[Observation],
    kimi_obs: List[Observation],
    tolerance: float = 0.10  # 10% tolerance for matching
) -> Tuple[List[Dict], List[Observation]]:
    """
    Compare observations from both models.
    Returns (disagreements, matched_observations)
    """
    disagreements = []
    matched = []

    # Index Kimi observations by element+tissue+moderators for matching
    # Create multiple indices for hierarchical matching:
    # 1. Full key with moderators (most precise)
    # 2. Element+tissue+source (medium)
    # 3. Element+tissue only (fallback)
    # 4. Element+normalized_tissue (fuzzy tissue match)
    # 5. Element only (last resort, strict value matching)
    kimi_index_full = {}    # element_tissue_source_moderators
    kimi_index_medium = {}  # element_tissue_source
    kimi_index_basic = {}   # element_tissue
    kimi_index_norm = {}    # element_normalized_tissue
    kimi_index_elem = {}    # element_only

    for obs in kimi_obs:
        element_tissue = f"{(obs.element or 'unknown').lower()}_{(obs.tissue or 'grain').lower()}"
        element_norm = f"{(obs.element or 'unknown').lower()}_{normalize_tissue(obs.tissue or 'grain')}"
        element_only = (obs.element or 'unknown').lower()
        source = normalize_data_source(obs.data_source)
        mods = moderator_key(obs.moderators)

        # Full key with moderators
        full_key = f"{element_tissue}_{source}_{mods}"
        if full_key not in kimi_index_full:
            kimi_index_full[full_key] = []
        kimi_index_full[full_key].append(obs)

        # Medium key (element+tissue+source)
        medium_key = f"{element_tissue}_{source}"
        if medium_key not in kimi_index_medium:
            kimi_index_medium[medium_key] = []
        kimi_index_medium[medium_key].append(obs)

        # Basic key (element+tissue)
        if element_tissue not in kimi_index_basic:
            kimi_index_basic[element_tissue] = []
        kimi_index_basic[element_tissue].append(obs)

        # Normalized tissue key
        if element_norm not in kimi_index_norm:
            kimi_index_norm[element_norm] = []
        kimi_index_norm[element_norm].append(obs)

        # Element-only key
        if element_only not in kimi_index_elem:
            kimi_index_elem[element_only] = []
        kimi_index_elem[element_only].append(obs)

    used_kimi = set()

    for c_obs in claude_obs:
        element_tissue = f"{(c_obs.element or 'unknown').lower()}_{(c_obs.tissue or 'grain').lower()}"
        element_norm = f"{(c_obs.element or 'unknown').lower()}_{normalize_tissue(c_obs.tissue or 'grain')}"
        element_only = (c_obs.element or 'unknown').lower()
        source = normalize_data_source(c_obs.data_source)
        mods = moderator_key(c_obs.moderators)

        # Try hierarchical matching: full -> medium -> basic -> norm_tissue -> element
        full_key = f"{element_tissue}_{source}_{mods}"
        medium_key = f"{element_tissue}_{source}"

        candidates = kimi_index_full.get(full_key, [])
        if not candidates:
            candidates = kimi_index_medium.get(medium_key, [])
        if not candidates:
            candidates = kimi_index_basic.get(element_tissue, [])
        if not candidates:
            candidates = kimi_index_norm.get(element_norm, [])
        if not candidates:
            candidates = kimi_index_elem.get(element_only, [])

        best_match = None
        best_diff = float('inf')

        for i, k_obs in enumerate(candidates):
            if id(k_obs) in used_kimi:
                continue

            # Calculate difference
            if (c_obs.control_mean is not None and k_obs.control_mean is not None and
                c_obs.treatment_mean is not None and k_obs.treatment_mean is not None):
                c_diff = abs(c_obs.control_mean - k_obs.control_mean) / max(abs(c_obs.control_mean), 0.001)
                t_diff = abs(c_obs.treatment_mean - k_obs.treatment_mean) / max(abs(c_obs.treatment_mean), 0.001)
                avg_diff = (c_diff + t_diff) / 2

                if avg_diff < best_diff:
                    best_diff = avg_diff
                    best_match = k_obs

        if best_match and best_diff <= tolerance:
            # Match found - direct agreement
            used_kimi.add(id(best_match))

            # Use Claude values but note agreement
            merged = Observation(
                element=c_obs.element,
                tissue=c_obs.tissue,
                treatment_mean=c_obs.treatment_mean,
                control_mean=c_obs.control_mean,
                treatment_variance=c_obs.treatment_variance or best_match.treatment_variance,
                control_variance=c_obs.control_variance or best_match.control_variance,
                variance_type=c_obs.variance_type or best_match.variance_type,
                n=c_obs.n or best_match.n,
                unit=c_obs.unit or best_match.unit,
                data_source=c_obs.data_source,
                treatment_description=c_obs.treatment_description,
                control_description=c_obs.control_description,
                moderators={**best_match.moderators, **c_obs.moderators},
                confidence="high" if best_diff < 0.05 else "medium",
                notes=f"Models agree (diff={best_diff:.1%})"
            )

            if merged.control_mean and merged.control_mean != 0:
                merged.effect_pct = ((merged.treatment_mean - merged.control_mean) / merged.control_mean) * 100

            matched.append(merged)

        elif best_match:
            # No direct match - try T/C swap detection
            swap_matched, corrected_obs, match_type = try_match_with_swap_correction(
                c_obs, best_match, tolerance
            )

            if swap_matched and corrected_obs:
                # T/C swap detected and corrected
                used_kimi.add(id(best_match))
                matched.append(corrected_obs)
                print(f"    ⚠️  SWAP CORRECTED: {c_obs.element} - {corrected_obs.notes}")
            else:
                # Genuine disagreement
                disagreements.append({
                    "type": "value_mismatch",
                    "element": c_obs.element,
                    "tissue": c_obs.tissue,
                    "claude": asdict(c_obs),
                    "kimi": asdict(best_match),
                    "difference": best_diff,
                    "swap_check": "checked_no_swap"
                })
        else:
            # No Kimi candidate found at all
            disagreements.append({
                "type": "claude_only",
                "element": c_obs.element,
                "tissue": c_obs.tissue,
                "claude": asdict(c_obs),
                "kimi": None,
                "difference": None
            })

    # Check for Kimi-only observations
    for k_obs in kimi_obs:
        if id(k_obs) not in used_kimi:
            disagreements.append({
                "type": "kimi_only",
                "element": k_obs.element,
                "tissue": k_obs.tissue,
                "kimi": asdict(k_obs),
                "claude": None
            })

    return disagreements, matched


# =============================================================================
# POST-PROCESSING: DEDUP, NULL MEANS, T/C SWAP CORRECTION
# =============================================================================

def post_process_observations(
    observations: List[Observation],
    expected_direction: str = "negative",
    verbose: bool = False
) -> Tuple[List[Observation], Dict]:
    """
    Post-process consensus observations:
    1. Remove exact duplicates (same element, tissue, control_mean, treatment_mean)
    2. Drop observations with null means
    3. Flag potential T/C swaps (>50% effect opposite to expected direction)
       Note: Does NOT auto-correct — analysis showed some elements legitimately
       go opposite to the expected direction (e.g., Fe, Mn increase under CO2).

    Returns (cleaned_observations, stats_dict).
    """
    stats = {"duplicates_removed": 0, "null_means_removed": 0, "tc_swaps_corrected": 0,
             "original_count": len(observations)}

    if not observations:
        stats["final_count"] = 0
        return observations, stats

    # 1. Remove duplicates (keep first occurrence)
    seen = set()
    deduped = []
    for obs in observations:
        key = (
            (obs.element or "").lower(),
            (obs.tissue or "").lower(),
            round(obs.control_mean, 4) if obs.control_mean is not None else None,
            round(obs.treatment_mean, 4) if obs.treatment_mean is not None else None,
        )
        if key not in seen:
            seen.add(key)
            deduped.append(obs)
        else:
            stats["duplicates_removed"] += 1
    observations = deduped

    # 2. Drop observations with null means
    cleaned = []
    for obs in observations:
        if obs.treatment_mean is None or obs.control_mean is None:
            stats["null_means_removed"] += 1
        else:
            cleaned.append(obs)
    observations = cleaned

    # 3. Flag potential T/C swaps (but do NOT auto-correct)
    # Analysis showed auto-correction is too aggressive: some elements legitimately
    # increase under elevated CO2 (Fe, Mn in some species/conditions), while others
    # decrease (N, P, Zn). The direction is element-specific, not paper-level.
    # Paper-level TC swap detection in validation is more appropriate.
    for obs in observations:
        if obs.control_mean is None or obs.treatment_mean is None or obs.control_mean == 0:
            continue
        effect = (obs.treatment_mean - obs.control_mean) / obs.control_mean

        flagged = False
        if expected_direction == "negative" and effect > 0.50:
            flagged = True
        elif expected_direction == "positive" and effect < -0.50:
            flagged = True

        if flagged:
            obs.notes = ((obs.notes or "") + " [potential T/C swap - verify]").strip()
            stats["tc_swaps_corrected"] += 1  # count as "flagged"

    stats["final_count"] = len(observations)

    if verbose and (stats["duplicates_removed"] or stats["null_means_removed"] or stats["tc_swaps_corrected"]):
        parts = []
        if stats["duplicates_removed"]:
            parts.append(f"{stats['duplicates_removed']} duplicates removed")
        if stats["null_means_removed"]:
            parts.append(f"{stats['null_means_removed']} null-mean obs removed")
        if stats["tc_swaps_corrected"]:
            parts.append(f"{stats['tc_swaps_corrected']} potential T/C swaps flagged")
        print(f"  Post-processing: {', '.join(parts)}")

    return observations, stats


# =============================================================================
# GEMINI TIEBREAKER FUNCTIONS
# =============================================================================

def needs_tiebreaker(
    claude_result: Optional[ExtractionResult],
    kimi_result: Optional[ExtractionResult],
    consensus_obs: List[Observation]
) -> Tuple[bool, str]:
    """
    Determine if a Gemini tiebreaker is needed.

    Triggers when:
    - One model extracted 0 obs, other has data
    - Both extracted >0 but consensus < 30% of max
    Does NOT trigger when both extracted 0.

    Returns (should_run, reason).
    """
    claude_count = len(claude_result.observations) if claude_result else 0
    kimi_count = len(kimi_result.observations) if kimi_result else 0

    # Both empty — nothing to tiebreak
    if claude_count == 0 and kimi_count == 0:
        return False, ""

    # One model got 0, other has data
    if claude_count == 0 and kimi_count > 0:
        return True, f"Claude extracted 0 obs, Kimi extracted {kimi_count}"
    if kimi_count == 0 and claude_count > 0:
        return True, f"Kimi extracted 0 obs, Claude extracted {claude_count}"

    # Both have data but consensus is very low
    max_obs = max(claude_count, kimi_count)
    consensus_count = len(consensus_obs)
    if max_obs > 0 and consensus_count < max_obs * 0.30:
        return True, (f"Low consensus: {consensus_count}/{max_obs} "
                      f"({consensus_count/max_obs:.0%}) — Claude={claude_count}, Kimi={kimi_count}")

    return False, ""


def two_of_three_vote(
    claude_result: Optional[ExtractionResult],
    kimi_result: Optional[ExtractionResult],
    gemini_result: ExtractionResult,
    tolerance: float = 0.10
) -> Tuple[List[Observation], List[Dict], dict]:
    """
    2-of-3 voting across Claude, Kimi, and Gemini.

    Runs pairwise comparisons:
    - Claude vs Gemini
    - Kimi vs Gemini
    Then accepts any observation confirmed by 2+ models.
    Does NOT accept Gemini-only observations.

    Returns (voted_observations, remaining_disagreements, stats).
    """
    claude_obs = claude_result.observations if claude_result else []
    kimi_obs = kimi_result.observations if kimi_result else []
    gemini_obs = gemini_result.observations

    # Pairwise comparisons
    cg_disagree, cg_matched = compare_observations(claude_obs, gemini_obs, tolerance) if claude_obs and gemini_obs else ([], [])
    kg_disagree, kg_matched = compare_observations(kimi_obs, gemini_obs, tolerance) if kimi_obs and gemini_obs else ([], [])

    # Deduplicate across pairwise matches using observation key
    def obs_key(obs: Observation) -> tuple:
        return (
            (obs.element or 'unknown').lower(),
            (obs.tissue or 'grain').lower(),
            round(obs.control_mean, 2) if obs.control_mean else 0,
            round(obs.treatment_mean, 2) if obs.treatment_mean else 0
        )

    seen_keys = set()
    voted_obs = []

    # Add Claude-Gemini matches (Claude confirmed by Gemini)
    for obs in cg_matched:
        key = obs_key(obs)
        if key not in seen_keys:
            seen_keys.add(key)
            obs.confidence = "high"
            obs.notes = (obs.notes or "") + " [Claude+Gemini agree]"
            voted_obs.append(obs)

    # Add Kimi-Gemini matches (Kimi confirmed by Gemini)
    for obs in kg_matched:
        key = obs_key(obs)
        if key not in seen_keys:
            seen_keys.add(key)
            obs.confidence = "high"
            obs.notes = (obs.notes or "") + " [Kimi+Gemini agree]"
            voted_obs.append(obs)

    # Check if any observations were confirmed by all 3 models
    # (present in both cg_matched and kg_matched)
    cg_keys = {obs_key(o) for o in cg_matched}
    kg_keys = {obs_key(o) for o in kg_matched}
    triple_agree = cg_keys & kg_keys

    for obs in voted_obs:
        if obs_key(obs) in triple_agree:
            obs.notes = (obs.notes or "").replace("[Claude+Gemini agree]", "").replace("[Kimi+Gemini agree]", "").strip()
            obs.notes = (obs.notes + " [all 3 models agree]").strip()

    # Remaining disagreements: observations not confirmed by any 2 models
    voted_keys = seen_keys
    remaining_disagreements = []

    for obs in claude_obs:
        key = obs_key(obs)
        if key not in voted_keys:
            remaining_disagreements.append({
                "type": "claude_only_after_vote",
                "element": obs.element,
                "tissue": obs.tissue,
                "claude": asdict(obs),
                "kimi": None,
                "gemini": None
            })

    for obs in kimi_obs:
        key = obs_key(obs)
        if key not in voted_keys:
            remaining_disagreements.append({
                "type": "kimi_only_after_vote",
                "element": obs.element,
                "tissue": obs.tissue,
                "kimi": asdict(obs),
                "claude": None,
                "gemini": None
            })

    stats = {
        "claude_gemini_matches": len(cg_matched),
        "kimi_gemini_matches": len(kg_matched),
        "triple_agreement": len(triple_agree),
        "voted_observations": len(voted_obs),
        "remaining_disagreements": len(remaining_disagreements)
    }

    return voted_obs, remaining_disagreements, stats


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_empty_recon(paper_id: str, warnings: List[str] = None) -> ReconResult:
    """Create an empty ReconResult with default values for all fields."""
    return ReconResult(
        paper_id=paper_id,
        warnings=warnings or [],
        variance_type=None,
        variance_source=None,
        variance_confidence="low",
        control_definition="",
        treatment_definition="",
        tables_with_target_data=[],
        potential_tc_confusion=None,
        experimental_design="",
        sample_size_found=None,
        sample_size_source=None,
        factorial_structure=None,
        extraction_guidance="",
        raw_response="",
        extraction_method="text",
        extraction_method_reason="",
        challenge_hints=[],
        is_scanned=False,
        is_fig_only=False,
        has_image_tables=False,
        estimated_difficulty="MEDIUM"
    )


# =============================================================================
# PDF EXTRACTION
# =============================================================================

def extract_pdf_text(pdf_path: str, kimi_client=None) -> str:
    """Extract text from PDF using Kimi's file API."""
    if kimi_client:
        try:
            with open(pdf_path, "rb") as f:
                file_obj = kimi_client.files.create(file=f, purpose="file-extract")
            content = kimi_client.files.content(file_id=file_obj.id).text
            # Delete file from Kimi storage to avoid hitting 1000-file limit
            try:
                kimi_client.files.delete(file_obj.id)
            except Exception:
                pass
            return content
        except Exception as e:
            print(f"  Kimi PDF extraction failed: {e}")

    # Fallback to PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except ImportError:
        pass

    # Last resort: pdfplumber
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except ImportError:
        return ""


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class ConsensusPipeline:
    """Complete two-stage consensus pipeline."""

    def __init__(self, config: MetaAnalysisConfig = None):
        self.config = config or MetaAnalysisConfig()

        # Initialize clients
        self.extractor = UnifiedExtractor()

        claude_key = os.environ.get('ANTHROPIC_API_KEY')
        self.recon = ReconPass(claude_key) if claude_key else None

        # Stats
        self.total_cost = 0.0
        self.papers_processed = 0

    def process_paper(
        self,
        pdf_path: str,
        run_claude: bool = True,
        run_kimi: bool = True,
        verbose: bool = True
    ) -> ConsensusResult:
        """Process a single paper through the full pipeline."""
        paper_id = Path(pdf_path).stem

        if verbose:
            print(f"\n{'='*70}")
            print(f"PROCESSING: {paper_id}")
            print(f"{'='*70}")

        # Extract PDF text
        if verbose:
            print("  [1/4] Extracting PDF text...")

        pdf_text = extract_pdf_text(pdf_path, self.extractor.kimi_client)

        if not pdf_text:
            return ConsensusResult(
                paper_id=paper_id,
                recon=create_empty_recon(paper_id, ["Failed to extract PDF text"]),
                claude_result=None,
                kimi_result=None
            )

        if verbose:
            print(f"    Extracted {len(pdf_text):,} characters")

        # Stage 1: Reconnaissance (with challenge detection)
        if verbose:
            print("  [2/4] Running Claude reconnaissance...")

        if self.recon:
            recon_result = self.recon.analyze(pdf_text, self.config, paper_id)

            if verbose:
                print(f"    Variance: {recon_result.variance_type} ({recon_result.variance_confidence})")
                print(f"    Tables: {recon_result.tables_with_target_data}")
                # Display challenge-aware routing info
                if recon_result.extraction_method != "text":
                    print(f"    Extraction Method: {recon_result.extraction_method.upper()}")
                    print(f"      Reason: {recon_result.extraction_method_reason}")
                if recon_result.estimated_difficulty == "HARD":
                    print(f"    Difficulty: HARD")
                if recon_result.is_scanned:
                    print("    Flag: SCANNED PDF")
                if recon_result.is_fig_only:
                    print("    Flag: FIG-ONLY (data in figures)")
                if recon_result.has_image_tables:
                    print("    Flag: IMAGE-TABLES")
                if recon_result.warnings:
                    print(f"    Warnings: {len(recon_result.warnings)}")
                    for w in recon_result.warnings[:3]:
                        # Encode to ASCII to avoid Windows console Unicode issues
                        w_safe = w[:80].encode('ascii', 'replace').decode('ascii')
                        print(f"      - {w_safe}...")
        else:
            recon_result = create_empty_recon(paper_id, ["No recon (Claude API not available)"])

        # Post-recon routing fix: if data is in figures but method is still "text",
        # upgrade to "hybrid" so vision extraction can read the figures
        if (recon_result.extraction_method == "text"
                and recon_result.figures_with_data
                and not recon_result.tables_with_target_data):
            recon_result.extraction_method = "hybrid"
            recon_result.extraction_method_reason = (
                f"Auto-upgraded: target data in figures {recon_result.figures_with_data} "
                f"but no target tables found"
            )
            if verbose:
                print(f"    AUTO-UPGRADE to HYBRID: data in {recon_result.figures_with_data}")

        # Stage 2: Extraction - route based on challenge detection
        claude_result = None
        kimi_result = None
        vision_result = None
        use_vision = recon_result.extraction_method in ("vision", "hybrid")
        use_text = recon_result.extraction_method in ("text", "hybrid")

        kimi_vision_result = None

        if use_vision:
            if verbose:
                method = recon_result.extraction_method.upper()
                print(f"  [2.5/4] Running Gemini VISION extraction ({method} mode)...")
            vision_result = self.extractor.extract_vision(
                pdf_path, self.config, recon_result, paper_id, provider="google"
            )
            if verbose:
                print(f"    Gemini vision observations: {len(vision_result.observations)}")
                print(f"    Cost: ${vision_result.cost_estimate:.4f}")
            self.total_cost += vision_result.cost_estimate

            # For pure VISION papers, also run Kimi vision for consensus
            if recon_result.extraction_method == "vision" and run_kimi and self.extractor.kimi_client:
                if verbose:
                    print(f"  [2.6/4] Running Kimi VISION extraction (consensus)...")
                try:
                    kimi_vision_result = self.extractor.extract_vision(
                        pdf_path, self.config, recon_result, paper_id, provider="kimi"
                    )
                    if verbose:
                        print(f"    Kimi vision observations: {len(kimi_vision_result.observations)}")
                        print(f"    Cost: ${kimi_vision_result.cost_estimate:.4f}")
                    self.total_cost += kimi_vision_result.cost_estimate
                except Exception as e:
                    if verbose:
                        print(f"    Kimi vision failed: {e}")

        if use_text and run_claude:
            if verbose:
                print("  [3/4] Running Claude extraction...")
            claude_result = self.extractor.extract_claude(pdf_text, self.config, recon_result, paper_id)
            if verbose:
                print(f"    Observations: {len(claude_result.observations)}")
                print(f"    Cost: ${claude_result.cost_estimate:.4f}")
            self.total_cost += claude_result.cost_estimate

        if use_text and run_kimi:
            if verbose:
                print("  [4/4] Running Kimi extraction...")
            kimi_result = self.extractor.extract_kimi(pdf_text, self.config, recon_result, paper_id)
            if verbose:
                print(f"    Observations: {len(kimi_result.observations)}")
                print(f"    Cost: ${kimi_result.cost_estimate:.4f}")
            self.total_cost += kimi_result.cost_estimate

        # Compare results
        disagreements = []
        consensus_obs = []
        gemini_result = None
        tiebreaker_used = False
        tiebreaker_reason = ""

        if recon_result.extraction_method == "vision":
            # Pure vision mode: use Gemini+Kimi vision consensus
            if vision_result and vision_result.observations:
                if kimi_vision_result and kimi_vision_result.observations:
                    # Best case: Gemini vision + Kimi vision consensus
                    if verbose:
                        print(f"\n  Comparing vision results (Gemini vs Kimi)...")
                    disagreements, consensus_obs = compare_observations(
                        vision_result.observations,
                        kimi_vision_result.observations
                    )
                    if verbose:
                        print(f"    Vision consensus: {len(consensus_obs)} matched")
                        print(f"    Vision disagreements: {len(disagreements)}")
                    # If consensus is very low, trust the model with more observations
                    if len(consensus_obs) < max(len(vision_result.observations),
                                                 len(kimi_vision_result.observations)) * 0.2:
                        if verbose:
                            print(f"    Low vision consensus - using model with more observations")
                        if len(vision_result.observations) >= len(kimi_vision_result.observations):
                            consensus_obs = vision_result.observations
                        else:
                            consensus_obs = kimi_vision_result.observations
                        disagreements = []
                elif claude_result and claude_result.observations:
                    # Fallback: Gemini vision + Claude text
                    disagreements, consensus_obs = compare_observations(
                        vision_result.observations,
                        claude_result.observations
                    )
                    if len(consensus_obs) < len(vision_result.observations) * 0.3:
                        if verbose:
                            print(f"    Low vision-text consensus, using vision as primary")
                        consensus_obs = vision_result.observations
                        disagreements = []
                else:
                    # Single vision model only
                    consensus_obs = vision_result.observations
            elif kimi_vision_result and kimi_vision_result.observations:
                consensus_obs = kimi_vision_result.observations
            elif claude_result:
                consensus_obs = claude_result.observations
            elif kimi_result:
                consensus_obs = kimi_result.observations

        elif recon_result.extraction_method == "hybrid":
            # Hybrid mode: merge text consensus with vision results
            text_consensus = []
            if claude_result and kimi_result:
                disagreements, text_consensus = compare_observations(
                    claude_result.observations,
                    kimi_result.observations
                )
            elif claude_result:
                text_consensus = claude_result.observations
            elif kimi_result:
                text_consensus = kimi_result.observations

            # Tiebreaker for hybrid text portion
            gemini_result = None
            tiebreaker_used = False
            tiebreaker_reason = ""
            should_tiebreak, tb_reason = needs_tiebreaker(claude_result, kimi_result, text_consensus)
            if should_tiebreak and self.extractor.gemini_client:
                tiebreaker_reason = tb_reason
                if verbose:
                    print(f"\n  TIEBREAKER NEEDED (hybrid text): {tb_reason}")
                    print(f"  Running Gemini text extraction...")
                gemini_result = self.extractor.extract_gemini(pdf_text, self.config, recon_result, paper_id)
                if verbose:
                    print(f"    Gemini observations: {len(gemini_result.observations)}")
                    print(f"    Cost: ${gemini_result.cost_estimate:.4f}")
                self.total_cost += gemini_result.cost_estimate

                if gemini_result.observations:
                    tiebreaker_used = True
                    voted_obs, remaining_disagree, vote_stats = two_of_three_vote(
                        claude_result, kimi_result, gemini_result
                    )
                    if voted_obs:
                        text_consensus = voted_obs
                        disagreements = remaining_disagree
                        if verbose:
                            print(f"    2-of-3 vote: {vote_stats['voted_observations']} obs recovered")
                            print(f"    Triple agreement: {vote_stats['triple_agreement']}")

            # Fallback: if text_consensus is still empty but one model has data, use it
            if not text_consensus:
                claude_count = len(claude_result.observations) if claude_result else 0
                kimi_count = len(kimi_result.observations) if kimi_result else 0
                if claude_count > 0 and kimi_count == 0:
                    text_consensus = claude_result.observations
                    for obs in text_consensus:
                        obs.confidence = "low"
                        obs.notes = ((obs.notes or "") + " [single-model fallback: Claude only]").strip()
                    if verbose:
                        print(f"    Fallback: using {claude_count} Claude-only obs (low confidence)")
                elif kimi_count > 0 and claude_count == 0:
                    text_consensus = kimi_result.observations
                    for obs in text_consensus:
                        obs.confidence = "low"
                        obs.notes = ((obs.notes or "") + " [single-model fallback: Kimi only]").strip()
                    if verbose:
                        print(f"    Fallback: using {kimi_count} Kimi-only obs (low confidence)")

            # Merge: start with text consensus, add unique vision observations
            consensus_obs = list(text_consensus)
            if vision_result and vision_result.observations:
                existing_sources = {
                    (o.element.lower(), o.tissue.lower(), round(o.treatment_mean, 1) if o.treatment_mean else 0)
                    for o in consensus_obs
                }
                # Also index existing by element+tissue only (for zero-effect check)
                existing_elements = {
                    (o.element.lower(), o.tissue.lower()) for o in consensus_obs
                }
                # Build index of text disagreements by element+tissue for fallback
                text_disagree_by_el = {}
                for d in disagreements:
                    el = d.get('element', '').lower()
                    tis = d.get('tissue', '').lower()
                    dtype = d.get('type', '')
                    # Get the text observation (prefer kimi_only since Claude misses elements)
                    obs_dict = None
                    if dtype == 'kimi_only' and d.get('kimi'):
                        obs_dict = d['kimi']
                    elif dtype == 'claude_only' and d.get('claude'):
                        obs_dict = d['claude']
                    elif dtype == 'value_mismatch':
                        # For value mismatch, prefer kimi (tends to extract more accurately)
                        obs_dict = d.get('kimi') or d.get('claude')
                    if obs_dict and el:
                        text_disagree_by_el.setdefault((el, tis), []).append(obs_dict)

                added_from_vision = 0
                replaced_zero_vision = 0
                for vobs in vision_result.observations:
                    key = ((vobs.element or 'unknown').lower(), (vobs.tissue or 'grain').lower(),
                           round(vobs.treatment_mean, 1) if vobs.treatment_mean else 0)
                    el_tis_key = ((vobs.element or 'unknown').lower(), (vobs.tissue or 'grain').lower())

                    if key in existing_sources:
                        continue  # Already in text consensus

                    # Check for zero-effect vision artifact (treatment_mean == control_mean)
                    is_zero_effect = (
                        vobs.treatment_mean is not None
                        and vobs.control_mean is not None
                        and vobs.treatment_mean == vobs.control_mean
                    )

                    if is_zero_effect and el_tis_key in text_disagree_by_el:
                        # Vision shows zero-effect but text extraction exists in disagreements
                        # Prefer the text extraction (likely more accurate for close values)
                        text_candidates = text_disagree_by_el[el_tis_key]
                        for tc in text_candidates:
                            t_mean = tc.get('treatment_mean')
                            c_mean = tc.get('control_mean')
                            if t_mean is not None and c_mean is not None and t_mean != c_mean:
                                # Found a text extraction with non-zero effect — use it
                                from dataclasses import fields as dc_fields
                                text_obs = Observation(
                                    element=tc.get('element', vobs.element),
                                    tissue=tc.get('tissue', vobs.tissue),
                                    treatment_mean=t_mean,
                                    control_mean=c_mean,
                                    treatment_variance=tc.get('treatment_variance'),
                                    control_variance=tc.get('control_variance'),
                                    variance_type=tc.get('variance_type'),
                                    n=tc.get('n'),
                                    unit=tc.get('unit', ''),
                                    data_source=tc.get('data_source', ''),
                                    treatment_description=tc.get('treatment_description', ''),
                                    control_description=tc.get('control_description', ''),
                                    moderators=tc.get('moderators', {}),
                                    confidence="low",
                                    notes="[text preferred over zero-effect vision]"
                                )
                                consensus_obs.append(text_obs)
                                existing_sources.add(key)
                                existing_elements.add(el_tis_key)
                                replaced_zero_vision += 1
                                break
                        else:
                            # No non-zero text candidate found, add vision anyway
                            vobs.notes = ((vobs.notes or "") + " [from vision]").strip()
                            consensus_obs.append(vobs)
                            existing_sources.add(key)
                            existing_elements.add(el_tis_key)
                            added_from_vision += 1
                    elif el_tis_key not in existing_elements:
                        # Element+tissue not in consensus at all — add from vision
                        vobs.notes = ((vobs.notes or "") + " [from vision]").strip()
                        consensus_obs.append(vobs)
                        existing_sources.add(key)
                        existing_elements.add(el_tis_key)
                        added_from_vision += 1
                    else:
                        # Element+tissue exists in consensus with different value — skip vision
                        pass

                if verbose and added_from_vision > 0:
                    print(f"    Added {added_from_vision} observations from vision extraction")
                if verbose and replaced_zero_vision > 0:
                    print(f"    Replaced {replaced_zero_vision} zero-effect vision obs with text alternatives")

        else:
            # Standard text mode
            gemini_result = None
            tiebreaker_used = False
            tiebreaker_reason = ""

            if claude_result and kimi_result:
                if verbose:
                    print("\n  Comparing results...")
                disagreements, consensus_obs = compare_observations(
                    claude_result.observations,
                    kimi_result.observations
                )
                if verbose:
                    print(f"    Matched: {len(consensus_obs)}")
                    print(f"    Disagreements: {len(disagreements)}")
            elif claude_result:
                consensus_obs = claude_result.observations
            elif kimi_result:
                consensus_obs = kimi_result.observations

            # Check if tiebreaker is needed
            should_tiebreak, tb_reason = needs_tiebreaker(claude_result, kimi_result, consensus_obs)
            if should_tiebreak and self.extractor.gemini_client:
                tiebreaker_reason = tb_reason
                if verbose:
                    print(f"\n  TIEBREAKER NEEDED: {tb_reason}")
                    print(f"  Running Gemini text extraction...")
                gemini_result = self.extractor.extract_gemini(pdf_text, self.config, recon_result, paper_id)
                if verbose:
                    print(f"    Gemini observations: {len(gemini_result.observations)}")
                    print(f"    Cost: ${gemini_result.cost_estimate:.4f}")
                self.total_cost += gemini_result.cost_estimate

                if gemini_result.observations:
                    tiebreaker_used = True
                    voted_obs, remaining_disagree, vote_stats = two_of_three_vote(
                        claude_result, kimi_result, gemini_result
                    )
                    if voted_obs:
                        consensus_obs = voted_obs
                        disagreements = remaining_disagree
                        if verbose:
                            print(f"    2-of-3 vote: {vote_stats['voted_observations']} obs recovered")
                            print(f"    Triple agreement: {vote_stats['triple_agreement']}")
                            print(f"    Remaining disagreements: {vote_stats['remaining_disagreements']}")

            # Fallback: if consensus is still empty but one model has data, use it
            if not consensus_obs:
                claude_count = len(claude_result.observations) if claude_result else 0
                kimi_count = len(kimi_result.observations) if kimi_result else 0
                if claude_count > 0 and kimi_count == 0:
                    consensus_obs = claude_result.observations
                    for obs in consensus_obs:
                        obs.confidence = "low"
                        obs.notes = ((obs.notes or "") + " [single-model fallback: Claude only]").strip()
                    if verbose:
                        print(f"    Fallback: using {claude_count} Claude-only obs (low confidence)")
                elif kimi_count > 0 and claude_count == 0:
                    consensus_obs = kimi_result.observations
                    for obs in consensus_obs:
                        obs.confidence = "low"
                        obs.notes = ((obs.notes or "") + " [single-model fallback: Kimi only]").strip()
                    if verbose:
                        print(f"    Fallback: using {kimi_count} Kimi-only obs (low confidence)")

        # Post-processing: dedup, drop null means, auto-correct T/C swaps
        consensus_obs, pp_stats = post_process_observations(
            consensus_obs, self.config.expected_direction, verbose=verbose
        )

        # Verify observations
        verification_flags = []
        for obs in consensus_obs:
            ver = verify_observation(obs, self.config.expected_direction)

            # Flag any failed checks
            failed_checks = [k for k, v in ver.get("checks", {}).items()
                          if isinstance(v, dict) and not v.get("passed", True)]

            if failed_checks:
                verification_flags.append({
                    "element": obs.element,
                    "tissue": obs.tissue,
                    "failed_checks": failed_checks,
                    "details": ver
                })

        self.papers_processed += 1

        return ConsensusResult(
            paper_id=paper_id,
            recon=recon_result,
            claude_result=claude_result,
            kimi_result=kimi_result,
            total_claude_obs=len(claude_result.observations) if claude_result else 0,
            total_kimi_obs=len(kimi_result.observations) if kimi_result else 0,
            matched_obs=len(consensus_obs),
            disagreements=disagreements,
            consensus_observations=consensus_obs,
            vision_result=vision_result,
            kimi_vision_result=kimi_vision_result,
            gemini_result=gemini_result,
            tiebreaker_used=tiebreaker_used,
            tiebreaker_reason=tiebreaker_reason,
            post_processing=pp_stats,
            verification_flags=verification_flags
        )

    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        run_claude: bool = True,
        run_kimi: bool = True,
        max_papers: int = None,
        verbose: bool = True
    ) -> List[ConsensusResult]:
        """Process all PDFs in a directory."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Find PDFs
        pdfs = sorted(input_path.glob("*.pdf"))
        if max_papers:
            pdfs = pdfs[:max_papers]

        if verbose:
            print(f"\nFound {len(pdfs)} PDFs to process")
            print(f"Output: {output_path}")
            print(f"Running: Claude={run_claude}, Kimi={run_kimi}")

        results = []
        all_observations = []

        def _check_internet():
            """Quick connectivity check."""
            import socket
            try:
                socket.create_connection(("api.anthropic.com", 443), timeout=5)
                return True
            except OSError:
                return False

        def _wait_for_internet(verbose=True):
            """Block until internet is available, checking every 30s."""
            import time as _time
            if verbose:
                print("\n  *** INTERNET DOWN — pausing until connectivity returns ***")
            while not _check_internet():
                _time.sleep(30)
            if verbose:
                print("  *** Internet restored — resuming ***")

        for i, pdf_path in enumerate(pdfs, 1):
            # Check if already processed
            paper_id = pdf_path.stem
            result_file = output_path / f"{paper_id}_consensus.json"
            if result_file.exists():
                if verbose:
                    print(f"\n[{i}/{len(pdfs)}] SKIPPING {paper_id} (already exists)")
                continue

            if verbose:
                print(f"\n[{i}/{len(pdfs)}] ", end="")

            # Retry loop: if both models return 0 due to connection errors, wait and retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result = self.process_paper(
                        str(pdf_path),
                        run_claude=run_claude,
                        run_kimi=run_kimi,
                        verbose=verbose
                    )

                    # Check for connection-error pattern: both models got 0 obs
                    # with error notes mentioning connection
                    claude_err = (result.claude_result and
                                  not result.claude_result.observations and
                                  "Error" in (result.claude_result.extraction_notes or ""))
                    kimi_err = (result.kimi_result and
                                not result.kimi_result.observations and
                                "Error" in (result.kimi_result.extraction_notes or ""))

                    if claude_err and kimi_err and attempt < max_retries - 1:
                        if not _check_internet():
                            _wait_for_internet(verbose)
                            if verbose:
                                print(f"  Retrying {paper_id} (attempt {attempt + 2}/{max_retries})...")
                            continue  # retry this paper
                    break  # success or non-connection error
                except Exception as e:
                    if attempt < max_retries - 1 and not _check_internet():
                        _wait_for_internet(verbose)
                        if verbose:
                            print(f"  Retrying {paper_id} after exception (attempt {attempt + 2}/{max_retries})...")
                        continue
                    raise

            try:
                results.append(result)

                # Collect observations for CSV
                for obs in result.consensus_observations:
                    obs_dict = asdict(obs)
                    obs_dict['paper_id'] = result.paper_id
                    all_observations.append(obs_dict)

                # Save individual result
                result_file = output_path / f"{result.paper_id}_consensus.json"
                with open(result_file, 'w') as f:
                    result_dict = {
                        'paper_id': result.paper_id,
                        'recon': asdict(result.recon),
                        'claude_obs': len(result.claude_result.observations) if result.claude_result else 0,
                        'kimi_obs': len(result.kimi_result.observations) if result.kimi_result else 0,
                        'gemini_obs': len(result.gemini_result.observations) if result.gemini_result else 0,
                        'matched_obs': result.matched_obs,
                        'tiebreaker_used': result.tiebreaker_used,
                        'tiebreaker_reason': result.tiebreaker_reason,
                        'disagreements': result.disagreements,
                        'consensus_observations': [asdict(o) for o in result.consensus_observations],
                        'post_processing': result.post_processing,
                        'verification_flags': result.verification_flags
                    }
                    json.dump(result_dict, f, indent=2, default=str)

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()

        # Save summary CSV
        if all_observations:
            csv_path = output_path / "consensus_results.csv"
            fieldnames = list(all_observations[0].keys())

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for obs in all_observations:
                    # Flatten moderators dict
                    if isinstance(obs.get('moderators'), dict):
                        obs['moderators'] = json.dumps(obs['moderators'])
                    writer.writerow(obs)

            if verbose:
                print(f"\nSaved {len(all_observations)} observations to {csv_path}")

        # Save summary with challenge stats
        vision_papers = [r for r in results if r.recon and r.recon.extraction_method in ('vision', 'hybrid')]
        manual_papers = [r for r in results if r.recon and r.recon.extraction_method == 'manual']
        hard_papers = [r for r in results if r.recon and r.recon.estimated_difficulty == 'HARD']

        tiebreaker_papers = [r for r in results if r.tiebreaker_used]

        summary = {
            'timestamp': datetime.now().isoformat(),
            'papers_processed': len(results),
            'total_cost': self.total_cost,
            'total_observations': len(all_observations),
            'challenge_stats': {
                'vision_needed': len(vision_papers),
                'manual_review_needed': len(manual_papers),
                'hard_difficulty': len(hard_papers),
                'vision_papers': [r.paper_id for r in vision_papers],
                'manual_papers': [r.paper_id for r in manual_papers]
            },
            'tiebreaker_stats': {
                'tiebreaker_used': len(tiebreaker_papers),
                'tiebreaker_papers': [
                    {
                        'paper_id': r.paper_id,
                        'reason': r.tiebreaker_reason,
                        'gemini_obs': len(r.gemini_result.observations) if r.gemini_result else 0,
                        'final_consensus': r.matched_obs
                    }
                    for r in tiebreaker_papers
                ]
            },
            'papers': [{
                'paper_id': r.paper_id,
                'claude_obs': r.total_claude_obs,
                'kimi_obs': r.total_kimi_obs,
                'gemini_obs': len(r.gemini_result.observations) if r.gemini_result else 0,
                'matched': r.matched_obs,
                'tiebreaker_used': r.tiebreaker_used,
                'disagreements': len(r.disagreements),
                'warnings': len(r.recon.warnings) if r.recon else 0,
                'extraction_method': r.recon.extraction_method if r.recon else 'text',
                'difficulty': r.recon.estimated_difficulty if r.recon else 'MEDIUM'
            } for r in results]
        }

        with open(output_path / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        # Save list of papers needing special handling
        if vision_papers or manual_papers:
            special_path = output_path / "papers_needing_attention.json"
            with open(special_path, 'w') as f:
                json.dump({
                    'vision_or_hybrid': [
                        {
                            'paper_id': r.paper_id,
                            'method': r.recon.extraction_method,
                            'reason': r.recon.extraction_method_reason,
                            'is_scanned': r.recon.is_scanned,
                            'is_fig_only': r.recon.is_fig_only,
                            'has_image_tables': r.recon.has_image_tables
                        }
                        for r in vision_papers
                    ],
                    'manual_review': [
                        {
                            'paper_id': r.paper_id,
                            'reason': r.recon.extraction_method_reason
                        }
                        for r in manual_papers
                    ]
                }, f, indent=2)
            if verbose:
                print(f"\nSaved {len(vision_papers) + len(manual_papers)} papers needing attention to {special_path}")

        if verbose:
            print(f"\n{'='*70}")
            print("PIPELINE COMPLETE")
            print(f"{'='*70}")
            print(f"Papers processed: {len(results)}")
            print(f"Total observations: {len(all_observations)}")
            print(f"Total cost: ${self.total_cost:.4f}")

            # Challenge-aware stats
            if vision_papers:
                print(f"\nPapers needing VISION extraction: {len(vision_papers)}")
                for r in vision_papers[:5]:
                    print(f"  - {r.paper_id}: {r.recon.extraction_method_reason}")
            if manual_papers:
                print(f"\nPapers flagged for MANUAL review: {len(manual_papers)}")
                for r in manual_papers[:5]:
                    print(f"  - {r.paper_id}")
            if hard_papers:
                print(f"\nHARD difficulty papers: {len(hard_papers)}")
            if tiebreaker_papers:
                print(f"\nGemini TIEBREAKER used: {len(tiebreaker_papers)} papers")
                for r in tiebreaker_papers:
                    gemini_count = len(r.gemini_result.observations) if r.gemini_result else 0
                    print(f"  - {r.paper_id}: {r.tiebreaker_reason}")
                    print(f"    Gemini obs: {gemini_count}, Final consensus: {r.matched_obs}")

        return results


# =============================================================================
# GROUND TRUTH COMPARISON
# =============================================================================

def compare_to_ground_truth(
    results: List[ConsensusResult],
    ground_truth_path: str,
    tolerance: float = 0.15
) -> Dict:
    """Compare extraction results to ground truth."""

    # Load ground truth
    gt_data = []
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_data.append(row)

    print(f"\nLoaded {len(gt_data)} ground truth observations")

    # Index by paper
    gt_by_paper = {}
    for row in gt_data:
        paper_id = row['paper_id']
        if paper_id not in gt_by_paper:
            gt_by_paper[paper_id] = []
        gt_by_paper[paper_id].append(row)

    comparison = {
        'papers_in_gt': len(gt_by_paper),
        'papers_matched': 0,
        'total_gt_obs': len(gt_data),
        'matched_obs': 0,
        'mean_accuracy': [],
        'effect_accuracy': [],
        'details': []
    }

    for result in results:
        paper_id = result.paper_id

        # Find matching GT paper
        gt_paper_id = None
        for gp in gt_by_paper.keys():
            if gp in paper_id or paper_id in gp:
                gt_paper_id = gp
                break

        if not gt_paper_id:
            continue

        comparison['papers_matched'] += 1
        gt_obs = gt_by_paper[gt_paper_id]

        paper_detail = {
            'paper_id': paper_id,
            'gt_count': len(gt_obs),
            'extracted_count': len(result.consensus_observations),
            'matches': []
        }

        for gt in gt_obs:
            element = gt['element']
            tissue = gt.get('tissue', '')
            gt_t_mean = float(gt['treatment_mean']) if gt['treatment_mean'] else None
            gt_c_mean = float(gt['control_mean']) if gt['control_mean'] else None
            gt_effect = float(gt['effect_pct']) if gt.get('effect_pct') else None

            # Find best match
            best_match = None
            best_error = float('inf')

            for obs in result.consensus_observations:
                if obs.element.upper() != element.upper():
                    continue
                if tissue and obs.tissue and tissue.lower() not in obs.tissue.lower():
                    continue

                if gt_c_mean and obs.control_mean:
                    c_error = abs(obs.control_mean - gt_c_mean) / max(abs(gt_c_mean), 0.001)
                    t_error = abs(obs.treatment_mean - gt_t_mean) / max(abs(gt_t_mean), 0.001) if gt_t_mean else 0
                    avg_error = (c_error + t_error) / 2

                    if avg_error < best_error:
                        best_error = avg_error
                        best_match = obs

            if best_match and best_error <= tolerance:
                comparison['matched_obs'] += 1
                comparison['mean_accuracy'].append(1 - best_error)

                if gt_effect and best_match.effect_pct:
                    effect_error = abs(best_match.effect_pct - gt_effect) / max(abs(gt_effect), 1)
                    comparison['effect_accuracy'].append(1 - min(effect_error, 1))

                paper_detail['matches'].append({
                    'element': element,
                    'gt_control': gt_c_mean,
                    'extracted_control': best_match.control_mean,
                    'gt_effect': gt_effect,
                    'extracted_effect': best_match.effect_pct,
                    'error': best_error
                })

        comparison['details'].append(paper_detail)

    # Calculate averages
    if comparison['mean_accuracy']:
        comparison['avg_mean_accuracy'] = sum(comparison['mean_accuracy']) / len(comparison['mean_accuracy'])
    if comparison['effect_accuracy']:
        comparison['avg_effect_accuracy'] = sum(comparison['effect_accuracy']) / len(comparison['effect_accuracy'])

    comparison['capture_rate'] = comparison['matched_obs'] / comparison['total_gt_obs'] if comparison['total_gt_obs'] else 0

    return comparison


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    import sys
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    parser = argparse.ArgumentParser(description="Two-Stage Consensus Pipeline for Meta-Analysis")
    parser.add_argument("--input", required=True, help="Input directory with PDFs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--claude-only", action="store_true", help="Run only Claude")
    parser.add_argument("--kimi-only", action="store_true", help="Run only Kimi")
    parser.add_argument("--max-papers", type=int, help="Maximum papers to process")
    parser.add_argument("--ground-truth", help="Path to ground truth CSV for validation")
    parser.add_argument("--config", help="Path to config JSON file for meta-analysis definition")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")

    args = parser.parse_args()

    # Create pipeline with config
    if args.config:
        config = MetaAnalysisConfig.from_json(args.config)
        if not args.quiet:
            print(f"Loaded config: {config.name}")
            print(f"  Primary outcomes: {config.primary_outcomes}")
            print(f"  Expected direction: {config.expected_direction}")
    else:
        config = MetaAnalysisConfig()
    pipeline = ConsensusPipeline(config)

    # Determine which models to run
    run_claude = not args.kimi_only
    run_kimi = not args.claude_only

    # Process
    results = pipeline.process_directory(
        args.input,
        args.output,
        run_claude=run_claude,
        run_kimi=run_kimi,
        max_papers=args.max_papers,
        verbose=not args.quiet
    )

    # Compare to ground truth if provided
    if args.ground_truth:
        print("\n" + "="*70)
        print("GROUND TRUTH COMPARISON")
        print("="*70)

        comparison = compare_to_ground_truth(results, args.ground_truth)

        print(f"Papers in GT: {comparison['papers_in_gt']}")
        print(f"Papers matched: {comparison['papers_matched']}")
        print(f"GT observations: {comparison['total_gt_obs']}")
        print(f"Matched observations: {comparison['matched_obs']}")
        print(f"Capture rate: {comparison['capture_rate']:.1%}")

        if comparison.get('avg_mean_accuracy'):
            print(f"Mean accuracy: {comparison['avg_mean_accuracy']:.1%}")
        if comparison.get('avg_effect_accuracy'):
            print(f"Effect size accuracy: {comparison['avg_effect_accuracy']:.1%}")

        # Save comparison
        with open(Path(args.output) / "ground_truth_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2, default=str)


if __name__ == "__main__":
    main()
