"""
Core module for Meta-Analysis Extraction System
"""
from .state import (
    Phase,
    PICOSpec,
    PaperRecon,
    Observation,
    ExtractionSchema,
    SessionState,
    Decision,
    OutcomeInfo,
    ModeratorInfo,
    OutcomeField,
    ModeratorField
)

# LLM client requires anthropic - import conditionally
try:
    from .llm import LLMClient, create_llm_client
    _HAS_LLM = True
except ImportError:
    _HAS_LLM = False
    LLMClient = None
    create_llm_client = None

__all__ = [
    'Phase',
    'PICOSpec',
    'PaperRecon',
    'Observation',
    'ExtractionSchema',
    'SessionState',
    'Decision',
    'OutcomeInfo',
    'ModeratorInfo',
    'OutcomeField',
    'ModeratorField',
    'LLMClient',
    'create_llm_client'
]
