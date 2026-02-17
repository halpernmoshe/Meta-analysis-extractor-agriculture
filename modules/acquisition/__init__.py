"""
Paper Acquisition Module

Automated paper discovery, screening, and downloading for meta-analysis.

Pipeline:
    Search (multiple databases) -> Deduplicate -> Screen (LLM) -> Find OA -> Download

Usage:
    from modules.acquisition import SearchModule, AcquisitionState, SearchQuery

    # Quick search
    search = SearchModule(output_dir='./output', unpaywall_email='you@email.com')
    query = SearchQuery(keywords=['elevated CO2', 'mineral concentration'])
    pico = {...}  # PICO criteria
    state = search.run(query, pico)

    # Or use CLI:
    python acquire_papers.py --config config.json --output ./output
"""

from core.acquisition_state import (
    AcquisitionState,
    AcquisitionPhase,
    SearchQuery,
    SearchResult,
    DeduplicatedPaper
)

from .search import SearchModule
from .screen import ScreenModule
from .download import DownloadModule
from .dedup import DeduplicationModule
from .unpaywall import UnpaywallClient, PMCClient

from .sources import (
    BaseSourceAdapter,
    PubMedAdapter,
    SemanticScholarAdapter,
    OpenAlexAdapter
)

__all__ = [
    # State classes
    'AcquisitionState',
    'AcquisitionPhase',
    'SearchQuery',
    'SearchResult',
    'DeduplicatedPaper',

    # Main modules
    'SearchModule',
    'ScreenModule',
    'DownloadModule',
    'DeduplicationModule',

    # OA clients
    'UnpaywallClient',
    'PMCClient',

    # Source adapters
    'BaseSourceAdapter',
    'PubMedAdapter',
    'SemanticScholarAdapter',
    'OpenAlexAdapter'
]
