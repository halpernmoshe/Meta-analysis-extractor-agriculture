"""
Acquisition State Management

Dataclasses for paper acquisition pipeline state, supporting checkpoint/resume.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import json
import uuid
from pathlib import Path


class AcquisitionPhase(Enum):
    """Phases of the acquisition workflow"""
    INIT = "init"
    SEARCH = "search"
    DEDUPLICATE = "deduplicate"
    SCREEN = "screen"
    FIND_OA = "find_oa"
    DOWNLOAD = "download"
    COMPLETE = "complete"


@dataclass
class SearchQuery:
    """Search query specification"""
    keywords: List[str] = field(default_factory=list)
    title_keywords: List[str] = field(default_factory=list)
    abstract_keywords: List[str] = field(default_factory=list)

    # Filters
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None
    publication_types: List[str] = field(default_factory=list)
    journals: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=lambda: ["English"])

    # Raw query string (if provided directly)
    query_string: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'keywords': self.keywords,
            'title_keywords': self.title_keywords,
            'abstract_keywords': self.abstract_keywords,
            'date_from': self.date_from,
            'date_to': self.date_to,
            'publication_types': self.publication_types,
            'journals': self.journals,
            'languages': self.languages,
            'query_string': self.query_string
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SearchQuery':
        return cls(
            keywords=data.get('keywords', []),
            title_keywords=data.get('title_keywords', []),
            abstract_keywords=data.get('abstract_keywords', []),
            date_from=data.get('date_from'),
            date_to=data.get('date_to'),
            publication_types=data.get('publication_types', []),
            journals=data.get('journals', []),
            languages=data.get('languages', ['English']),
            query_string=data.get('query_string')
        )


@dataclass
class SearchResult:
    """Single search result from any source"""
    # Source identification
    source: str  # pubmed, semantic_scholar, openalex, google_scholar
    source_id: str  # Source-specific ID

    # Standard identifiers
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None

    # Core metadata
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None

    # Citation metrics
    citation_count: Optional[int] = None
    reference_count: Optional[int] = None

    # URLs
    url: Optional[str] = None
    pdf_url: Optional[str] = None

    # Additional metadata
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    publication_types: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'source': self.source,
            'source_id': self.source_id,
            'doi': self.doi,
            'pmid': self.pmid,
            'pmcid': self.pmcid,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'journal': self.journal,
            'abstract': self.abstract,
            'citation_count': self.citation_count,
            'reference_count': self.reference_count,
            'url': self.url,
            'pdf_url': self.pdf_url,
            'keywords': self.keywords,
            'mesh_terms': self.mesh_terms,
            'publication_types': self.publication_types
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SearchResult':
        return cls(
            source=data.get('source', ''),
            source_id=data.get('source_id', ''),
            doi=data.get('doi'),
            pmid=data.get('pmid'),
            pmcid=data.get('pmcid'),
            title=data.get('title', ''),
            authors=data.get('authors', []),
            year=data.get('year'),
            journal=data.get('journal'),
            abstract=data.get('abstract'),
            citation_count=data.get('citation_count'),
            reference_count=data.get('reference_count'),
            url=data.get('url'),
            pdf_url=data.get('pdf_url'),
            keywords=data.get('keywords', []),
            mesh_terms=data.get('mesh_terms', []),
            publication_types=data.get('publication_types', [])
        )


@dataclass
class DeduplicatedPaper:
    """Paper after deduplication - merged from multiple sources"""
    paper_id: str  # Generated unique ID (e.g., P00001)

    # Canonical identifiers
    canonical_doi: Optional[str] = None
    pmid: Optional[str] = None
    pmcid: Optional[str] = None

    # Best available metadata
    title: str = ""
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None

    # Source tracking
    found_in_sources: List[str] = field(default_factory=list)
    source_results: Dict[str, dict] = field(default_factory=dict)  # source -> SearchResult dict

    # Screening results
    screening_decision: Optional[str] = None  # include, exclude, uncertain
    screening_confidence: Optional[str] = None  # high, medium, low
    screening_reason: Optional[str] = None
    screening_timestamp: Optional[str] = None

    # PICO matching details
    population_match: Optional[bool] = None
    intervention_match: Optional[bool] = None
    outcome_potential: Optional[bool] = None
    quantitative_data: Optional[bool] = None
    exclusion_category: Optional[str] = None

    # Open access availability
    oa_status: Optional[str] = None  # gold, green, bronze, closed, not_found
    oa_url: Optional[str] = None
    oa_source: Optional[str] = None  # unpaywall, pmc, biorxiv, publisher, search

    # Download status
    download_status: Optional[str] = None  # pending, success, failed, skipped
    download_path: Optional[str] = None
    download_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'paper_id': self.paper_id,
            'canonical_doi': self.canonical_doi,
            'pmid': self.pmid,
            'pmcid': self.pmcid,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'journal': self.journal,
            'abstract': self.abstract,
            'found_in_sources': self.found_in_sources,
            'source_results': self.source_results,
            'screening_decision': self.screening_decision,
            'screening_confidence': self.screening_confidence,
            'screening_reason': self.screening_reason,
            'screening_timestamp': self.screening_timestamp,
            'population_match': self.population_match,
            'intervention_match': self.intervention_match,
            'outcome_potential': self.outcome_potential,
            'quantitative_data': self.quantitative_data,
            'exclusion_category': self.exclusion_category,
            'oa_status': self.oa_status,
            'oa_url': self.oa_url,
            'oa_source': self.oa_source,
            'download_status': self.download_status,
            'download_path': self.download_path,
            'download_error': self.download_error
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DeduplicatedPaper':
        return cls(
            paper_id=data.get('paper_id', ''),
            canonical_doi=data.get('canonical_doi'),
            pmid=data.get('pmid'),
            pmcid=data.get('pmcid'),
            title=data.get('title', ''),
            authors=data.get('authors', []),
            year=data.get('year'),
            journal=data.get('journal'),
            abstract=data.get('abstract'),
            found_in_sources=data.get('found_in_sources', []),
            source_results=data.get('source_results', {}),
            screening_decision=data.get('screening_decision'),
            screening_confidence=data.get('screening_confidence'),
            screening_reason=data.get('screening_reason'),
            screening_timestamp=data.get('screening_timestamp'),
            population_match=data.get('population_match'),
            intervention_match=data.get('intervention_match'),
            outcome_potential=data.get('outcome_potential'),
            quantitative_data=data.get('quantitative_data'),
            exclusion_category=data.get('exclusion_category'),
            oa_status=data.get('oa_status'),
            oa_url=data.get('oa_url'),
            oa_source=data.get('oa_source'),
            download_status=data.get('download_status'),
            download_path=data.get('download_path'),
            download_error=data.get('download_error')
        )


@dataclass
class AcquisitionState:
    """Complete acquisition session state - serializable for checkpoint/resume"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    # Current phase
    current_phase: AcquisitionPhase = AcquisitionPhase.INIT

    # Directories
    output_directory: str = ""
    download_directory: str = ""

    # Search specification
    query: SearchQuery = field(default_factory=SearchQuery)
    pico: Optional[Dict] = None  # PICO criteria for screening

    # Source configuration
    sources_enabled: Dict[str, bool] = field(default_factory=lambda: {
        'pubmed': True,
        'semantic_scholar': True,
        'openalex': True,
        'google_scholar': False
    })
    max_results_per_source: int = 1000

    # Raw search results by source
    raw_results: Dict[str, List[dict]] = field(default_factory=dict)

    # Deduplicated papers
    deduplicated_papers: Dict[str, dict] = field(default_factory=dict)  # paper_id -> paper dict

    # Progress tracking
    search_completed_sources: List[str] = field(default_factory=list)
    screening_completed_ids: List[str] = field(default_factory=list)
    oa_checked_ids: List[str] = field(default_factory=list)
    download_completed_ids: List[str] = field(default_factory=list)

    # Statistics
    total_raw_results: int = 0
    total_after_dedup: int = 0
    total_included: int = 0
    total_excluded: int = 0
    total_uncertain: int = 0
    total_oa_available: int = 0
    total_downloaded: int = 0
    total_download_failed: int = 0

    # Cost tracking
    llm_calls_made: int = 0
    estimated_cost: float = 0.0

    # Error log
    errors: List[Dict] = field(default_factory=list)

    def update_timestamp(self):
        """Update last_updated timestamp"""
        self.last_updated = datetime.now().isoformat()

    def add_error(self, phase: str, message: str, details: Optional[dict] = None):
        """Log an error"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'phase': phase,
            'message': message,
            'details': details
        })

    def get_paper(self, paper_id: str) -> Optional[DeduplicatedPaper]:
        """Get a paper by ID as DeduplicatedPaper object"""
        if paper_id in self.deduplicated_papers:
            return DeduplicatedPaper.from_dict(self.deduplicated_papers[paper_id])
        return None

    def update_paper(self, paper: DeduplicatedPaper):
        """Update a paper in state"""
        self.deduplicated_papers[paper.paper_id] = paper.to_dict()

    def get_papers_for_screening(self) -> List[DeduplicatedPaper]:
        """Get papers that haven't been screened yet"""
        papers = []
        for paper_id, paper_dict in self.deduplicated_papers.items():
            if paper_id not in self.screening_completed_ids:
                papers.append(DeduplicatedPaper.from_dict(paper_dict))
        return papers

    def get_included_papers(self) -> List[DeduplicatedPaper]:
        """Get papers with include decision"""
        papers = []
        for paper_dict in self.deduplicated_papers.values():
            if paper_dict.get('screening_decision') == 'include':
                papers.append(DeduplicatedPaper.from_dict(paper_dict))
        return papers

    def get_papers_for_oa_check(self) -> List[DeduplicatedPaper]:
        """Get included papers that haven't had OA checked"""
        papers = []
        for paper_dict in self.deduplicated_papers.values():
            paper_id = paper_dict.get('paper_id')
            if (paper_dict.get('screening_decision') == 'include' and
                paper_id not in self.oa_checked_ids):
                papers.append(DeduplicatedPaper.from_dict(paper_dict))
        return papers

    def get_papers_for_download(self) -> List[DeduplicatedPaper]:
        """Get papers with OA URL that haven't been downloaded"""
        papers = []
        for paper_dict in self.deduplicated_papers.values():
            paper_id = paper_dict.get('paper_id')
            if (paper_dict.get('oa_url') and
                paper_id not in self.download_completed_ids):
                papers.append(DeduplicatedPaper.from_dict(paper_dict))
        return papers

    def update_statistics(self):
        """Recalculate statistics from current state"""
        self.total_after_dedup = len(self.deduplicated_papers)
        self.total_included = 0
        self.total_excluded = 0
        self.total_uncertain = 0
        self.total_oa_available = 0
        self.total_downloaded = 0
        self.total_download_failed = 0

        for paper_dict in self.deduplicated_papers.values():
            decision = paper_dict.get('screening_decision')
            if decision == 'include':
                self.total_included += 1
            elif decision == 'exclude':
                self.total_excluded += 1
            elif decision == 'uncertain':
                self.total_uncertain += 1

            if paper_dict.get('oa_url'):
                self.total_oa_available += 1

            download_status = paper_dict.get('download_status')
            if download_status == 'success':
                self.total_downloaded += 1
            elif download_status == 'failed':
                self.total_download_failed += 1

    def to_dict(self) -> dict:
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'current_phase': self.current_phase.value,
            'output_directory': self.output_directory,
            'download_directory': self.download_directory,
            'query': self.query.to_dict(),
            'pico': self.pico,
            'sources_enabled': self.sources_enabled,
            'max_results_per_source': self.max_results_per_source,
            'raw_results': self.raw_results,
            'deduplicated_papers': self.deduplicated_papers,
            'search_completed_sources': self.search_completed_sources,
            'screening_completed_ids': self.screening_completed_ids,
            'oa_checked_ids': self.oa_checked_ids,
            'download_completed_ids': self.download_completed_ids,
            'total_raw_results': self.total_raw_results,
            'total_after_dedup': self.total_after_dedup,
            'total_included': self.total_included,
            'total_excluded': self.total_excluded,
            'total_uncertain': self.total_uncertain,
            'total_oa_available': self.total_oa_available,
            'total_downloaded': self.total_downloaded,
            'total_download_failed': self.total_download_failed,
            'llm_calls_made': self.llm_calls_made,
            'estimated_cost': self.estimated_cost,
            'errors': self.errors
        }

    def save(self, filepath: str):
        """Save state to JSON file"""
        self.update_timestamp()
        self.update_statistics()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'AcquisitionState':
        """Load state from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        state = cls(
            session_id=data.get('session_id', str(uuid.uuid4())[:8]),
            created_at=data.get('created_at', datetime.now().isoformat()),
            last_updated=data.get('last_updated', datetime.now().isoformat()),
            current_phase=AcquisitionPhase(data.get('current_phase', 'init')),
            output_directory=data.get('output_directory', ''),
            download_directory=data.get('download_directory', ''),
            query=SearchQuery.from_dict(data.get('query', {})),
            pico=data.get('pico'),
            sources_enabled=data.get('sources_enabled', {
                'pubmed': True,
                'semantic_scholar': True,
                'openalex': True,
                'google_scholar': False
            }),
            max_results_per_source=data.get('max_results_per_source', 1000),
            raw_results=data.get('raw_results', {}),
            deduplicated_papers=data.get('deduplicated_papers', {}),
            search_completed_sources=data.get('search_completed_sources', []),
            screening_completed_ids=data.get('screening_completed_ids', []),
            oa_checked_ids=data.get('oa_checked_ids', []),
            download_completed_ids=data.get('download_completed_ids', []),
            total_raw_results=data.get('total_raw_results', 0),
            total_after_dedup=data.get('total_after_dedup', 0),
            total_included=data.get('total_included', 0),
            total_excluded=data.get('total_excluded', 0),
            total_uncertain=data.get('total_uncertain', 0),
            total_oa_available=data.get('total_oa_available', 0),
            total_downloaded=data.get('total_downloaded', 0),
            total_download_failed=data.get('total_download_failed', 0),
            llm_calls_made=data.get('llm_calls_made', 0),
            estimated_cost=data.get('estimated_cost', 0.0),
            errors=data.get('errors', [])
        )

        return state
