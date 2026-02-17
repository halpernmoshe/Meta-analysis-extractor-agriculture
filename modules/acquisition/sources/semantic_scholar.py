"""
Semantic Scholar Source Adapter

Uses Semantic Scholar Academic Graph API.
API Documentation: https://api.semanticscholar.org/api-docs/
"""

import requests
from typing import List, Optional, Callable
import time

from core.acquisition_state import SearchQuery, SearchResult
from .base import BaseSourceAdapter


class SemanticScholarAdapter(BaseSourceAdapter):
    """
    Semantic Scholar API integration.

    Rate limits:
    - Without API key: 100 requests per 5 minutes
    - With API key: Higher limits (apply at https://www.semanticscholar.org/product/api)
    """

    SOURCE_NAME = "semantic_scholar"
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    DEFAULT_RATE_LIMIT = 0.33  # ~20 requests per minute (conservative)

    def __init__(self,
                 api_key: Optional[str] = None,
                 rate_limit: Optional[float] = None):
        """
        Initialize Semantic Scholar adapter.

        Args:
            api_key: API key for higher rate limits (optional)
            rate_limit: Custom rate limit
        """
        super().__init__(rate_limit=rate_limit)
        self.api_key = api_key

    def search(self,
               query: SearchQuery,
               max_results: int = 1000,
               progress_callback: Optional[Callable[[int, int, str], None]] = None
               ) -> List[SearchResult]:
        """
        Search Semantic Scholar.

        Args:
            query: Search query
            max_results: Maximum results
            progress_callback: Progress callback

        Returns:
            List of SearchResult objects
        """
        query_str = self._build_query_string(query)

        if not query_str:
            return []

        if progress_callback:
            progress_callback(0, max_results, f"Searching Semantic Scholar: {query_str[:50]}...")

        results = []
        offset = 0
        limit = 100  # API max per request

        while len(results) < max_results:
            batch_results = self._search_batch(
                query_str,
                offset=offset,
                limit=min(limit, max_results - len(results)),
                year_from=query.date_from[:4] if query.date_from else None,
                year_to=query.date_to[:4] if query.date_to else None
            )

            if not batch_results:
                break

            results.extend(batch_results)
            offset += limit

            if progress_callback:
                progress_callback(len(results), max_results,
                                f"Retrieved {len(results)} papers...")

            self._rate_limit_wait()

        if progress_callback:
            progress_callback(len(results), len(results), "Semantic Scholar search complete")

        return results[:max_results]

    def _search_batch(self,
                      query: str,
                      offset: int = 0,
                      limit: int = 100,
                      year_from: Optional[str] = None,
                      year_to: Optional[str] = None) -> List[SearchResult]:
        """Execute a single search batch"""

        # Fields to request
        fields = [
            'paperId', 'externalIds', 'title', 'abstract',
            'year', 'venue', 'authors', 'citationCount',
            'referenceCount', 'openAccessPdf', 'url'
        ]

        params = {
            'query': query,
            'offset': offset,
            'limit': limit,
            'fields': ','.join(fields)
        }

        # Add year filter if specified
        if year_from or year_to:
            year_filter = ""
            if year_from:
                year_filter = f"{year_from}-"
            else:
                year_filter = "1900-"
            if year_to:
                year_filter += year_to
            else:
                year_filter += "2100"
            params['year'] = year_filter

        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key

        try:
            response = requests.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=headers,
                timeout=30
            )

            if response.status_code == 429:
                # Rate limited - wait and return empty
                print("Semantic Scholar rate limited, waiting...")
                time.sleep(60)
                return []

            if response.status_code != 200:
                print(f"Semantic Scholar error: {response.status_code}")
                return []

            data = response.json()
            papers = data.get('data', [])

            results = []
            for paper in papers:
                result = self._parse_paper(paper)
                if result:
                    results.append(result)

            return results

        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []

    def _parse_paper(self, paper: dict) -> Optional[SearchResult]:
        """Parse Semantic Scholar paper to SearchResult"""
        try:
            paper_id = paper.get('paperId', '')
            if not paper_id:
                return None

            # External IDs
            external_ids = paper.get('externalIds') or {}
            doi = external_ids.get('DOI')
            pmid = external_ids.get('PubMed')
            pmcid = external_ids.get('PubMedCentral')

            # Basic metadata
            title = paper.get('title', '')
            abstract = paper.get('abstract')
            year = paper.get('year')
            journal = paper.get('venue')

            # Authors
            authors = []
            for author in paper.get('authors', []):
                name = author.get('name')
                if name:
                    authors.append(name)

            # Citation count
            citation_count = paper.get('citationCount')
            reference_count = paper.get('referenceCount')

            # URLs
            url = paper.get('url')
            pdf_url = None
            oa_pdf = paper.get('openAccessPdf')
            if oa_pdf:
                pdf_url = oa_pdf.get('url')

            return SearchResult(
                source=self.SOURCE_NAME,
                source_id=paper_id,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                abstract=abstract,
                citation_count=citation_count,
                reference_count=reference_count,
                url=url,
                pdf_url=pdf_url
            )

        except Exception as e:
            print(f"Error parsing Semantic Scholar paper: {e}")
            return None

    def get_paper_by_id(self, paper_id: str) -> Optional[SearchResult]:
        """
        Get a specific paper by its Semantic Scholar ID or DOI.

        Args:
            paper_id: Paper ID or DOI (prefix with "DOI:" for DOI)

        Returns:
            SearchResult or None
        """
        fields = [
            'paperId', 'externalIds', 'title', 'abstract',
            'year', 'venue', 'authors', 'citationCount',
            'referenceCount', 'openAccessPdf', 'url'
        ]

        headers = {}
        if self.api_key:
            headers['x-api-key'] = self.api_key

        try:
            self._rate_limit_wait()

            response = requests.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={'fields': ','.join(fields)},
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                return None

            paper = response.json()
            return self._parse_paper(paper)

        except Exception as e:
            print(f"Error fetching paper {paper_id}: {e}")
            return None
