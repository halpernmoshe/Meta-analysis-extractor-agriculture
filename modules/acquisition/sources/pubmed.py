"""
PubMed/PMC Source Adapter

Uses NCBI E-utilities API for searching PubMed.
API Documentation: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Callable
import time

from core.acquisition_state import SearchQuery, SearchResult
from .base import BaseSourceAdapter


class PubMedAdapter(BaseSourceAdapter):
    """
    PubMed E-utilities integration.

    Rate limits:
    - Without API key: 3 requests/second
    - With API key: 10 requests/second

    Get API key: https://www.ncbi.nlm.nih.gov/account/settings/
    """

    SOURCE_NAME = "pubmed"
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    DEFAULT_RATE_LIMIT = 3.0  # Conservative default

    def __init__(self,
                 api_key: Optional[str] = None,
                 email: Optional[str] = None,
                 rate_limit: Optional[float] = None):
        """
        Initialize PubMed adapter.

        Args:
            api_key: NCBI API key (optional, increases rate limit)
            email: Email address (required by NCBI for identification)
            rate_limit: Custom rate limit (default based on API key)
        """
        # Set rate limit based on API key
        if rate_limit is None:
            rate_limit = 10.0 if api_key else 3.0

        super().__init__(rate_limit=rate_limit)

        self.api_key = api_key
        self.email = email

    def search(self,
               query: SearchQuery,
               max_results: int = 1000,
               progress_callback: Optional[Callable[[int, int, str], None]] = None
               ) -> List[SearchResult]:
        """
        Search PubMed using ESearch + EFetch.

        Args:
            query: Search query
            max_results: Maximum results to return
            progress_callback: Progress callback

        Returns:
            List of SearchResult objects
        """
        # Build PubMed query
        query_str = self._build_pubmed_query(query)

        if progress_callback:
            progress_callback(0, max_results, f"Searching PubMed: {query_str[:50]}...")

        # Step 1: ESearch to get PMIDs
        pmids = self._esearch(query_str, max_results)

        if not pmids:
            return []

        if progress_callback:
            progress_callback(0, len(pmids), f"Found {len(pmids)} papers, fetching metadata...")

        # Step 2: EFetch to get metadata (in batches of 200)
        results = []
        batch_size = 200

        for i, batch in enumerate(self._batch(pmids, batch_size)):
            if progress_callback:
                progress_callback(len(results), len(pmids),
                                f"Fetching batch {i+1}...")

            records = self._efetch(batch)
            batch_results = self._parse_records(records)
            results.extend(batch_results)

            self._rate_limit_wait()

        if progress_callback:
            progress_callback(len(results), len(results), "PubMed search complete")

        return results

    def _build_pubmed_query(self, query: SearchQuery) -> str:
        """Build PubMed query string with field tags"""
        parts = []

        # Title/Abstract keywords
        if query.keywords:
            terms = " OR ".join(f'"{kw}"[Title/Abstract]' for kw in query.keywords)
            parts.append(f"({terms})")

        # Title-specific keywords
        if query.title_keywords:
            terms = " OR ".join(f'"{kw}"[Title]' for kw in query.title_keywords)
            parts.append(f"({terms})")

        # Date range
        if query.date_from or query.date_to:
            date_from = query.date_from.replace('-', '/') if query.date_from else "1900/01/01"
            date_to = query.date_to.replace('-', '/') if query.date_to else "3000/12/31"
            parts.append(f'("{date_from}"[Date - Publication] : "{date_to}"[Date - Publication])')

        # Language filter
        if query.languages:
            lang_terms = " OR ".join(f'"{lang}"[Language]' for lang in query.languages)
            parts.append(f"({lang_terms})")

        # Publication types
        if query.publication_types:
            type_terms = " OR ".join(f'"{pt}"[Publication Type]' for pt in query.publication_types)
            parts.append(f"({type_terms})")

        # Journal filter
        if query.journals:
            journal_terms = " OR ".join(f'"{j}"[Journal]' for j in query.journals)
            parts.append(f"({journal_terms})")

        # If no parts, use raw query or keywords
        if not parts:
            if query.query_string:
                return query.query_string
            return ' '.join(query.keywords) if query.keywords else ""

        return " AND ".join(parts)

    def _esearch(self, query: str, max_results: int) -> List[str]:
        """Execute ESearch and return PMIDs"""
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance',
            'usehistory': 'n'
        }

        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email

        self._rate_limit_wait()

        try:
            response = requests.get(
                f"{self.BASE_URL}/esearch.fcgi",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

            result = data.get('esearchresult', {})
            pmids = result.get('idlist', [])

            return pmids

        except Exception as e:
            print(f"PubMed ESearch error: {e}")
            return []

    def _efetch(self, pmids: List[str]) -> str:
        """Fetch full records for PMIDs (XML format)"""
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract'
        }

        if self.api_key:
            params['api_key'] = self.api_key
        if self.email:
            params['email'] = self.email

        try:
            response = requests.get(
                f"{self.BASE_URL}/efetch.fcgi",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            return response.text

        except Exception as e:
            print(f"PubMed EFetch error: {e}")
            return ""

    def _parse_records(self, xml_text: str) -> List[SearchResult]:
        """Parse PubMed XML records to SearchResult objects"""
        results = []

        if not xml_text:
            return results

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall('.//PubmedArticle'):
                result = self._parse_article(article)
                if result:
                    results.append(result)

        except ET.ParseError as e:
            print(f"PubMed XML parse error: {e}")

        return results

    def _parse_article(self, article: ET.Element) -> Optional[SearchResult]:
        """Parse single PubMed article to SearchResult"""
        try:
            medline = article.find('.//MedlineCitation')
            if medline is None:
                return None

            # PMID
            pmid_elem = medline.find('.//PMID')
            pmid = pmid_elem.text if pmid_elem is not None else None

            if not pmid:
                return None

            # Article element
            article_elem = medline.find('.//Article')
            if article_elem is None:
                return None

            # Title
            title_elem = article_elem.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_parts = []
            for abstract_text in article_elem.findall('.//AbstractText'):
                if abstract_text.text:
                    label = abstract_text.get('Label', '')
                    text = abstract_text.text
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            abstract = ' '.join(abstract_parts) if abstract_parts else None

            # Authors
            authors = []
            for author in article_elem.findall('.//Author'):
                last_name = author.find('LastName')
                fore_name = author.find('ForeName')
                if last_name is not None and last_name.text:
                    name = last_name.text
                    if fore_name is not None and fore_name.text:
                        name = f"{fore_name.text} {name}"
                    authors.append(name)

            # Year
            year = None
            pub_date = article_elem.find('.//PubDate')
            if pub_date is not None:
                year_elem = pub_date.find('Year')
                if year_elem is not None and year_elem.text:
                    try:
                        year = int(year_elem.text)
                    except ValueError:
                        pass

            # Journal
            journal = None
            journal_elem = article_elem.find('.//Journal/Title')
            if journal_elem is not None:
                journal = journal_elem.text

            # DOI
            doi = None
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'doi':
                    doi = article_id.text
                    break

            # PMCID
            pmcid = None
            for article_id in article.findall('.//ArticleId'):
                if article_id.get('IdType') == 'pmc':
                    pmcid = article_id.text
                    break

            # MeSH terms
            mesh_terms = []
            for mesh in medline.findall('.//MeshHeading/DescriptorName'):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            # Keywords
            keywords = []
            for kw in medline.findall('.//KeywordList/Keyword'):
                if kw.text:
                    keywords.append(kw.text)

            # Publication types
            pub_types = []
            for pt in article_elem.findall('.//PublicationTypeList/PublicationType'):
                if pt.text:
                    pub_types.append(pt.text)

            # Build URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

            return SearchResult(
                source=self.SOURCE_NAME,
                source_id=pmid,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                title=title,
                authors=authors,
                year=year,
                journal=journal,
                abstract=abstract,
                url=url,
                keywords=keywords,
                mesh_terms=mesh_terms,
                publication_types=pub_types
            )

        except Exception as e:
            print(f"Error parsing PubMed article: {e}")
            return None
