"""
Prompts module for Meta-Analysis Extraction System
"""
from .recon_prompts import RECON_PROMPTS, RECON_SYSTEM_PROMPT
from .extract_prompts import EXTRACT_PROMPTS, EXTRACTION_SYSTEM_PROMPT

__all__ = [
    'RECON_PROMPTS',
    'RECON_SYSTEM_PROMPT',
    'EXTRACT_PROMPTS', 
    'EXTRACTION_SYSTEM_PROMPT'
]
