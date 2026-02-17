"""
Source Adapters for Paper Acquisition

Each adapter connects to a different academic database API.
"""

from .base import BaseSourceAdapter
from .pubmed import PubMedAdapter
from .semantic_scholar import SemanticScholarAdapter
from .openalex import OpenAlexAdapter

__all__ = [
    'BaseSourceAdapter',
    'PubMedAdapter',
    'SemanticScholarAdapter',
    'OpenAlexAdapter'
]
