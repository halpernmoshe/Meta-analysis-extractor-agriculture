"""
OpenAlex Source Adapter

Uses OpenAlex API for comprehensive academic search.
API Documentation: https://docs.openalex.org/
"""

import requests
from typing import List, Optional, Callable, Dict

from core.acquisition_state import SearchQuery, SearchResult
from .base import BaseSourceAdapter


class OpenAlexAdapter(BaseSourceAdapter):
    """
    OpenAlex API integration.

    Rate limits:
    - Free tier: 100,000 requests/day
    - Polite pool (with email): Higher priority, faster responses

    OpenAlex is the most comprehensive free academic database,
    replacing Microsoft Academic Graph.
    """

    SOURCE_NAME = "openalex"
    BASE_URL = "https://api.openalex.org"
    DEFAULT_RATE_LIMIT = 10.0  # Very generous limits

    def __init__(self,
                 email: Optional[str] = None,
                 rate_limit: Optional[float] = None):
        """
        Initialize OpenAlex adapter.

        Args:
            email: Email for polite pool (recommended)
            rate_limit: Custom rate limit
        """
        super().__init__(rate_limit=rate_limit)
        self.email = email

    def search(self,
               query: SearchQuery,
               max_results: int = 1000,
               progress_callback: Optional[Callable[[int, int, str], None]] = None
               ) -> List[SearchResult]:
        """
        Search OpenAlex using works endpoint.

        Args:
            query: Search query
            max_results: Maximum results
            progress_callback: Progress callback

        Returns:
            List of SearchResult objects
        """
        if progress_callback:
            progress_callback(0, max_results, "Searching OpenAlex...")

        results = []
        cursor = '*'  # Initial cursor for pagination
        per_page = 200  # Max per request

        params = self._build_params(query)
        params['per-page'] = per_page

        if self.email:
            params['mailto'] = self.email

        while len(results) < max_results and cursor:
            params['cursor'] = cursor

            batch_results, cursor = self._fetch_page(params)

            if not batch_results:
                break

            results.extend(batch_results)

            if progress_callback:
                progress_callback(len(results), max_results,
                                f"Retrieved {len(results)} papers...")

            self._rate_limit_wait()

        if progress_callback:
            progress_callback(len(results), len(results), "OpenAlex search complete")

        return results[:max_results]

    def _build_params(self, query: SearchQuery) -> Dict:
        """Build OpenAlex query parameters"""
        params = {}

        # Text search
        search_terms = []
        search_terms.extend(query.keywords)
        search_terms.extend(query.title_keywords)
        search_terms.extend(query.abstract_keywords)

        if search_terms:
            params['search'] = ' '.join(search_terms)
        elif query.query_string:
            params['search'] = query.query_string

        # Filters
        filters = []

        # Date range
        if query.date_from:
            filters.append(f"from_publication_date:{query.date_from}")
        if query.date_to:
            filters.append(f"to_publication_date:{query.date_to}")

        # Language (OpenAlex uses ISO 639-1 codes)
        if query.languages:
            lang_codes = []
            lang_map = {
                'english': 'en',
                'german': 'de',
                'french': 'fr',
                'spanish': 'es',
                'chinese': 'zh',
                'japanese': 'ja'
            }
            for lang in query.languages:
                code = lang_map.get(lang.lower(), lang.lower()[:2])
                lang_codes.append(code)
            if lang_codes:
                filters.append(f"language:{'|'.join(lang_codes)}")

        # Publication types
        if query.publication_types:
            # OpenAlex uses type field
            type_map = {
                'journal article': 'article',
                'review': 'review',
                'book': 'book',
                'book chapter': 'book-chapter',
                'conference paper': 'proceedings-article'
            }
            types = []
            for pt in query.publication_types:
                oa_type = type_map.get(pt.lower(), pt.lower())
                types.append(oa_type)
            if types:
                filters.append(f"type:{'|'.join(types)}")

        if filters:
            params['filter'] = ','.join(filters)

        # Sort by relevance
        params['sort'] = 'relevance_score:desc'

        return params

    def _fetch_page(self, params: Dict) -> tuple[List[SearchResult], Optional[str]]:
        """Fetch a single page of results"""
        try:
            response = requests.get(
                f"{self.BASE_URL}/works",
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                print(f"OpenAlex error: {response.status_code}")
                return [], None

            data = response.json()
            works = data.get('results', [])

            results = []
            for work in works:
                result = self._parse_work(work)
                if result:
                    results.append(result)

            # Get next cursor
            meta = data.get('meta', {})
            next_cursor = meta.get('next_cursor')

            return results, next_cursor

        except Exception as e:
            print(f"OpenAlex fetch error: {e}")
            return [], None

    def _parse_work(self, work: dict) -> Optional[SearchResult]:
        """Parse OpenAlex work to SearchResult"""
        try:
            # OpenAlex ID
            openalex_id = work.get('id', '')
            if openalex_id:
                openalex_id = openalex_id.replace('https://openalex.org/', '')

            if not openalex_id:
                return None

            # IDs
            ids = work.get('ids') or {}
            doi = ids.get('doi', '').replace('https://doi.org/', '') if ids.get('doi') else None
            pmid = ids.get('pmid', '').replace('https://pubmed.ncbi.nlm.nih.gov/', '') if ids.get('pmid') else None
            pmcid = ids.get('pmcid')

            # Basic metadata
            title = work.get('title', '') or work.get('display_name', '')
            year = work.get('publication_year')

            # Abstract (stored as inverted index)
            abstract = None
            abstract_inverted = work.get('abstract_inverted_index')
            if abstract_inverted:
                abstract = self._reconstruct_abstract(abstract_inverted)

            # Authors
            authors = []
            for authorship in work.get('authorships', []):
                author_info = authorship.get('author', {})
                name = author_info.get('display_name')
                if name:
                    authors.append(name)

            # Journal/Source
            journal = None
            primary_location = work.get('primary_location') or {}
            source = primary_location.get('source') or {}
            journal = source.get('display_name')

            # Citation count
            citation_count = work.get('cited_by_count')

            # URLs
            url = primary_location.get('landing_page_url')
            pdf_url = None

            # Open access info
            oa_info = work.get('open_access') or {}
            if oa_info.get('is_oa'):
                pdf_url = oa_info.get('oa_url')

            # Keywords/Concepts
            keywords = []
            for concept in work.get('concepts', [])[:10]:  # Top 10 concepts
                name = concept.get('display_name')
                if name:
                    keywords.append(name)

            return SearchResult(
                source=self.SOURCE_NAME,
                source_id=openalex_id,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                abstract=abstract,
                citation_count=citation_count,
                url=url,
                pdf_url=pdf_url,
                keywords=keywords
            )

        except Exception as e:
            print(f"Error parsing OpenAlex work: {e}")
            return None

    def _reconstruct_abstract(self, inverted_index: Dict) -> str:
        """
        Reconstruct abstract from OpenAlex inverted index format.

        OpenAlex stores abstracts as {word: [positions]} for compression.
        """
        if not inverted_index:
            return ""

        try:
            # Find max position
            max_pos = 0
            for positions in inverted_index.values():
                if positions:
                    max_pos = max(max_pos, max(positions))

            # Build word list
            words = [''] * (max_pos + 1)
            for word, positions in inverted_index.items():
                for pos in positions:
                    words[pos] = word

            # Join and clean
            abstract = ' '.join(words)
            # Clean up extra spaces
            abstract = ' '.join(abstract.split())

            return abstract

        except Exception:
            return ""

    def get_work_by_id(self, work_id: str) -> Optional[SearchResult]:
        """
        Get a specific work by OpenAlex ID or DOI.

        Args:
            work_id: OpenAlex ID (W1234567890) or DOI

        Returns:
            SearchResult or None
        """
        # Handle DOI input
        if work_id.startswith('10.') or 'doi.org' in work_id:
            work_id = work_id.replace('https://doi.org/', '')
            work_id = f"https://doi.org/{work_id}"

        params = {}
        if self.email:
            params['mailto'] = self.email

        try:
            self._rate_limit_wait()

            response = requests.get(
                f"{self.BASE_URL}/works/{work_id}",
                params=params,
                timeout=30
            )

            if response.status_code != 200:
                return None

            work = response.json()
            return self._parse_work(work)

        except Exception as e:
            print(f"Error fetching work {work_id}: {e}")
            return None
