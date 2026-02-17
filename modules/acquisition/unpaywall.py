"""
Unpaywall Client

Find open access versions of papers using the Unpaywall API.
API Documentation: https://unpaywall.org/products/api
"""

import requests
from typing import Dict, List, Optional, Callable
import time


class UnpaywallClient:
    """
    Unpaywall API client for finding open access versions.

    Unpaywall indexes open access locations for papers with DOIs,
    including Green OA (repositories), Gold OA (publisher), and
    Bronze OA (free to read but not openly licensed).

    Rate limit: 100,000 requests/day with email identification.
    """

    BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(self, email: str, rate_limit: float = 10.0):
        """
        Initialize Unpaywall client.

        Args:
            email: Email address (required for API access)
            rate_limit: Requests per second
        """
        if not email:
            raise ValueError("Email is required for Unpaywall API")
        self.email = email
        self.rate_limit = rate_limit
        self._last_request_time = 0

    def _rate_limit_wait(self):
        """Wait to respect rate limit"""
        if self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def find_oa(self, doi: str) -> Optional[Dict]:
        """
        Find open access version for a DOI.

        Args:
            doi: DOI of the paper

        Returns:
            Dict with OA information:
                - is_oa: bool - Whether paper is open access
                - oa_status: str - gold, green, bronze, hybrid, closed
                - best_oa_url: str - URL to best OA PDF
                - best_oa_location: dict - Full location info
                - oa_locations: list - All OA locations
            Returns None on error
        """
        if not doi:
            return {'is_oa': False, 'oa_status': 'no_doi', 'error': 'No DOI provided'}

        # Clean DOI
        doi = self._normalize_doi(doi)

        self._rate_limit_wait()

        try:
            response = requests.get(
                f"{self.BASE_URL}/{doi}",
                params={'email': self.email},
                timeout=10
            )

            if response.status_code == 404:
                return {
                    'is_oa': False,
                    'oa_status': 'not_found',
                    'error': 'DOI not found in Unpaywall'
                }

            if response.status_code == 422:
                return {
                    'is_oa': False,
                    'oa_status': 'invalid_doi',
                    'error': 'Invalid DOI format'
                }

            if response.status_code != 200:
                return {
                    'is_oa': False,
                    'oa_status': 'error',
                    'error': f'API error: {response.status_code}'
                }

            data = response.json()

            # Extract best OA location
            best_location = data.get('best_oa_location') or {}

            return {
                'is_oa': data.get('is_oa', False),
                'oa_status': data.get('oa_status', 'closed'),
                'best_oa_url': best_location.get('url_for_pdf'),
                'best_oa_location': best_location,
                'oa_locations': data.get('oa_locations', []),
                'journal_is_oa': data.get('journal_is_oa', False),
                'publisher': data.get('publisher'),
                'title': data.get('title'),
                'doi_url': data.get('doi_url')
            }

        except requests.Timeout:
            return {
                'is_oa': False,
                'oa_status': 'timeout',
                'error': 'Request timed out'
            }
        except Exception as e:
            return {
                'is_oa': False,
                'oa_status': 'error',
                'error': str(e)
            }

    def find_oa_batch(self,
                      dois: List[str],
                      progress_callback: Optional[Callable[[int, int, str], None]] = None
                      ) -> Dict[str, Dict]:
        """
        Find OA for multiple DOIs.

        Args:
            dois: List of DOIs
            progress_callback: Progress callback(current, total, message)

        Returns:
            Dict mapping DOI to OA result
        """
        results = {}

        for i, doi in enumerate(dois):
            if progress_callback:
                progress_callback(i, len(dois), f"Checking {doi[:30]}...")

            results[doi] = self.find_oa(doi)

        if progress_callback:
            progress_callback(len(dois), len(dois), "OA check complete")

        return results

    def get_best_pdf_url(self, doi: str) -> Optional[str]:
        """
        Get the best PDF URL for a DOI.

        Convenience method that returns just the URL.

        Args:
            doi: DOI of the paper

        Returns:
            PDF URL string or None
        """
        result = self.find_oa(doi)
        if result and result.get('is_oa'):
            return result.get('best_oa_url')
        return None

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI string"""
        if not doi:
            return ""
        doi = doi.strip()
        # Remove common prefixes
        prefixes = [
            'https://doi.org/',
            'http://doi.org/',
            'http://dx.doi.org/',
            'doi:',
            'DOI:'
        ]
        for prefix in prefixes:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break
        return doi

    @staticmethod
    def get_oa_type_description(oa_status: str) -> str:
        """
        Get human-readable description of OA status.

        Args:
            oa_status: Status from Unpaywall

        Returns:
            Description string
        """
        descriptions = {
            'gold': 'Published open access (Gold OA)',
            'green': 'Repository version (Green OA)',
            'bronze': 'Free to read on publisher site',
            'hybrid': 'OA article in subscription journal',
            'closed': 'Not open access',
            'not_found': 'DOI not found in Unpaywall',
            'no_doi': 'No DOI provided',
            'error': 'Error checking OA status'
        }
        return descriptions.get(oa_status, f'Unknown status: {oa_status}')


class PMCClient:
    """
    PubMed Central client for checking PMC availability.

    PMC provides free full-text articles. If a paper has a PMCID,
    it's likely available for free download.
    """

    def __init__(self, rate_limit: float = 3.0):
        self.rate_limit = rate_limit
        self._last_request_time = 0

    def _rate_limit_wait(self):
        if self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def get_pdf_url(self, pmcid: str) -> Optional[str]:
        """
        Get PDF URL for a PMC ID.

        Args:
            pmcid: PMC ID (e.g., PMC1234567)

        Returns:
            PDF URL or None
        """
        if not pmcid:
            return None

        # Normalize PMCID
        pmcid = pmcid.upper().strip()
        if not pmcid.startswith('PMC'):
            pmcid = f'PMC{pmcid}'

        # PMC PDF URL pattern
        # Note: Not all PMC articles have PDFs available
        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"

        # Check if accessible
        self._rate_limit_wait()

        try:
            response = requests.head(
                pdf_url,
                allow_redirects=True,
                timeout=10
            )

            if response.status_code == 200:
                return pdf_url

        except Exception:
            pass

        return None

    def get_xml_url(self, pmcid: str) -> str:
        """
        Get XML full-text URL for a PMC ID.

        XML is always available for PMC articles.

        Args:
            pmcid: PMC ID

        Returns:
            XML URL
        """
        pmcid = pmcid.upper().strip()
        if not pmcid.startswith('PMC'):
            pmcid = f'PMC{pmcid}'

        return f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}"
