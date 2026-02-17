"""
Download Module

PDF download coordinator for open access papers.
"""

import requests
from pathlib import Path
from typing import List, Dict, Optional, Callable
import re
import time
from bs4 import BeautifulSoup

from core.acquisition_state import DeduplicatedPaper
from .unpaywall import UnpaywallClient, PMCClient


class SciHubClient:
    """
    Sci-Hub client for downloading papers by DOI.
    """

    # Sci-Hub mirrors to try
    MIRRORS = [
        "https://sci-hub.se",
        "https://sci-hub.st",
        "https://sci-hub.ru",
    ]

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    def __init__(self):
        self.working_mirror = None
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.USER_AGENT})

    def get_pdf_url(self, doi: str) -> Optional[str]:
        """
        Get PDF URL from Sci-Hub for a DOI.

        Args:
            doi: The DOI to look up

        Returns:
            PDF URL or None
        """
        if not doi:
            return None

        # Clean DOI
        doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')

        # Try working mirror first, then others
        mirrors = [self.working_mirror] if self.working_mirror else []
        mirrors.extend([m for m in self.MIRRORS if m != self.working_mirror])

        for mirror in mirrors:
            if not mirror:
                continue
            try:
                pdf_url = self._try_mirror(mirror, doi)
                if pdf_url:
                    self.working_mirror = mirror
                    return pdf_url
            except Exception:
                continue

        return None

    def _try_mirror(self, mirror: str, doi: str) -> Optional[str]:
        """Try to get PDF URL from a specific mirror"""
        url = f"{mirror}/{doi}"

        try:
            response = self.session.get(url, timeout=30, allow_redirects=True)

            if response.status_code != 200:
                return None

            # Parse the page to find PDF URL
            soup = BeautifulSoup(response.text, 'html.parser')

            # Method 1: Look for embed/iframe with PDF
            for tag in soup.find_all(['embed', 'iframe']):
                src = tag.get('src', '')
                if src and ('.pdf' in src.lower() or '/pdf/' in src.lower()):
                    return self._normalize_url(src, mirror)

            # Method 2: Look for direct PDF link in buttons
            for tag in soup.find_all('button', onclick=True):
                onclick = tag.get('onclick', '')
                if 'location.href' in onclick:
                    match = re.search(r"location\.href='([^']+)'", onclick)
                    if match:
                        return self._normalize_url(match.group(1), mirror)

            # Method 3: Look for #pdf id
            pdf_div = soup.find(id='pdf')
            if pdf_div:
                embed = pdf_div.find('embed')
                if embed and embed.get('src'):
                    return self._normalize_url(embed['src'], mirror)

            # Method 4: Look for any PDF link
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.pdf') or '/pdf/' in href:
                    return self._normalize_url(href, mirror)

        except Exception:
            pass

        return None

    def _normalize_url(self, url: str, mirror: str) -> str:
        """Normalize URL to absolute"""
        if url.startswith('//'):
            return 'https:' + url
        elif url.startswith('/'):
            return mirror + url
        elif not url.startswith('http'):
            return mirror + '/' + url
        return url


class DownloadModule:
    """
    PDF download coordinator for open access papers.

    Priority order for finding PDFs:
    1. PDF URL from search results (Semantic Scholar, OpenAlex)
    2. PMC (if PMCID available)
    3. Unpaywall (finds OA versions)
    4. bioRxiv/medRxiv (for preprints)
    """

    # User agent for requests
    USER_AGENT = "MetaAnalysisBot/1.0 (scientific research; python-requests)"

    # Minimum PDF file size (bytes)
    MIN_PDF_SIZE = 10 * 1024  # 10 KB

    def __init__(self,
                 download_dir: str,
                 unpaywall_email: str,
                 rate_limit: float = 1.0,
                 use_scihub: bool = False):
        """
        Initialize download module.

        Args:
            download_dir: Directory to save PDFs
            unpaywall_email: Email for Unpaywall API
            rate_limit: Downloads per second
            use_scihub: Whether to use Sci-Hub as fallback
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.unpaywall = UnpaywallClient(unpaywall_email) if unpaywall_email else None
        self.pmc = PMCClient()
        self.scihub = SciHubClient() if use_scihub else None
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

    def find_and_download(self,
                         papers: List[DeduplicatedPaper],
                         progress_callback: Optional[Callable[[int, int, str], None]] = None,
                         skip_existing: bool = True
                         ) -> Dict[str, str]:
        """
        Find OA URLs and download PDFs for papers.

        Args:
            papers: Papers to download
            progress_callback: Progress callback
            skip_existing: Skip papers that already have downloaded PDFs

        Returns:
            Dict mapping paper_id to download path or error message
        """
        results = {}

        for i, paper in enumerate(papers):
            if progress_callback:
                progress_callback(i, len(papers),
                                f"Processing {paper.paper_id}: {paper.title[:40]}...")

            # Skip if already downloaded
            if skip_existing and paper.download_status == 'success' and paper.download_path:
                if Path(paper.download_path).exists():
                    results[paper.paper_id] = paper.download_path
                    continue

            try:
                # Step 1: Find OA URL
                oa_url = self._find_oa_url(paper)

                if not oa_url:
                    paper.oa_status = 'closed'
                    paper.download_status = 'skipped'
                    results[paper.paper_id] = 'No open access version found'
                    continue

                paper.oa_url = oa_url

                # Step 2: Download PDF
                filename = self._generate_filename(paper)
                filepath = self.download_dir / filename

                success = self._download_pdf(oa_url, filepath)

                if success:
                    paper.download_status = 'success'
                    paper.download_path = str(filepath)
                    results[paper.paper_id] = str(filepath)
                else:
                    paper.download_status = 'failed'
                    paper.download_error = 'Download failed'
                    results[paper.paper_id] = 'Download failed'

            except Exception as e:
                paper.download_status = 'failed'
                paper.download_error = str(e)
                results[paper.paper_id] = f'Error: {str(e)[:100]}'

            self._rate_limit_wait()

        if progress_callback:
            progress_callback(len(papers), len(papers), "Download complete")

        return results

    def find_oa_urls(self,
                    papers: List[DeduplicatedPaper],
                    progress_callback: Optional[Callable[[int, int, str], None]] = None
                    ) -> List[DeduplicatedPaper]:
        """
        Find OA URLs for papers (without downloading).

        Args:
            papers: Papers to check
            progress_callback: Progress callback

        Returns:
            Papers with OA status updated
        """
        for i, paper in enumerate(papers):
            if progress_callback:
                progress_callback(i, len(papers),
                                f"Checking OA for {paper.paper_id}...")

            try:
                oa_url = self._find_oa_url(paper)

                if oa_url:
                    paper.oa_url = oa_url
                else:
                    paper.oa_status = 'closed'

            except Exception as e:
                paper.oa_status = 'error'
                paper.download_error = str(e)

        if progress_callback:
            progress_callback(len(papers), len(papers), "OA check complete")

        return papers

    def _find_oa_url(self, paper: DeduplicatedPaper) -> Optional[str]:
        """Find best OA URL for paper"""

        # Priority 1: Check if already have PDF URL from search
        for source_result in paper.source_results.values():
            if isinstance(source_result, dict):
                pdf_url = source_result.get('pdf_url')
                if pdf_url and self._verify_url(pdf_url):
                    paper.oa_status = 'found_in_search'
                    paper.oa_source = source_result.get('source', 'search')
                    return pdf_url

        # Priority 2: PMC via PMCID
        if paper.pmcid:
            pmc_url = self.pmc.get_pdf_url(paper.pmcid)
            if pmc_url:
                paper.oa_status = 'green'
                paper.oa_source = 'pmc'
                return pmc_url

        # Priority 3: Unpaywall
        if self.unpaywall and paper.canonical_doi:
            oa_info = self.unpaywall.find_oa(paper.canonical_doi)
            if oa_info and oa_info.get('is_oa'):
                paper.oa_status = oa_info.get('oa_status', 'unknown')
                paper.oa_source = 'unpaywall'
                pdf_url = oa_info.get('best_oa_url')
                if pdf_url:
                    return pdf_url

        # Priority 4: Check bioRxiv/medRxiv by DOI pattern
        if paper.canonical_doi:
            doi_lower = paper.canonical_doi.lower()
            if 'biorxiv' in doi_lower or 'medrxiv' in doi_lower:
                preprint_url = self._get_preprint_pdf(paper.canonical_doi)
                if preprint_url:
                    paper.oa_status = 'green'
                    paper.oa_source = 'preprint'
                    return preprint_url

        # Priority 5: Sci-Hub (fallback)
        if self.scihub and paper.canonical_doi:
            scihub_url = self.scihub.get_pdf_url(paper.canonical_doi)
            if scihub_url:
                paper.oa_status = 'scihub'
                paper.oa_source = 'scihub'
                return scihub_url

        return None

    def _verify_url(self, url: str) -> bool:
        """Verify URL is accessible"""
        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=10,
                headers={'User-Agent': self.USER_AGENT}
            )
            return response.status_code == 200
        except Exception:
            return False

    def _get_preprint_pdf(self, doi: str) -> Optional[str]:
        """Get PDF URL from bioRxiv/medRxiv"""
        # bioRxiv/medRxiv DOI pattern: 10.1101/YYYY.MM.DD.XXXXXX
        # PDF URL: https://www.biorxiv.org/content/10.1101/YYYY.MM.DD.XXXXXX.full.pdf

        if not doi:
            return None

        doi = doi.replace('https://doi.org/', '')

        if doi.startswith('10.1101/'):
            # Try bioRxiv first
            pdf_url = f"https://www.biorxiv.org/content/{doi}.full.pdf"
            if self._verify_url(pdf_url):
                return pdf_url

            # Try medRxiv
            pdf_url = f"https://www.medrxiv.org/content/{doi}.full.pdf"
            if self._verify_url(pdf_url):
                return pdf_url

        return None

    def _download_pdf(self, url: str, filepath: Path) -> bool:
        """Download PDF from URL"""
        try:
            response = requests.get(
                url,
                timeout=120,  # 2 minutes for large PDFs
                headers={'User-Agent': self.USER_AGENT},
                stream=True
            )

            if response.status_code != 200:
                return False

            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'html' in content_type:
                # Got HTML instead of PDF (common for paywalled content)
                return False

            # Download to file
            content = response.content

            # Verify it's a PDF
            if not content.startswith(b'%PDF'):
                return False

            # Check minimum size
            if len(content) < self.MIN_PDF_SIZE:
                return False

            # Write file
            with open(filepath, 'wb') as f:
                f.write(content)

            # Verify written file
            if filepath.stat().st_size < self.MIN_PDF_SIZE:
                filepath.unlink()
                return False

            return True

        except Exception as e:
            print(f"Download error for {url}: {e}")
            return False

    def _generate_filename(self, paper: DeduplicatedPaper) -> str:
        """Generate clean filename for paper"""
        # Get first author last name
        first_author = "Unknown"
        if paper.authors:
            parts = paper.authors[0].split()
            if parts:
                first_author = parts[-1]  # Last name

        # Clean author name
        first_author = re.sub(r'[^\w]', '', first_author)

        # Year
        year = paper.year or 'XXXX'

        # Clean title for filename (first few words)
        title = paper.title or 'Untitled'
        title_clean = re.sub(r'[^\w\s]', '', title)
        title_words = '_'.join(title_clean.split()[:5])

        # Build filename
        filename = f"{first_author}_{year}_{title_words}.pdf"

        # Clean any remaining problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)

        # Limit length
        if len(filename) > 150:
            filename = filename[:147] + ".pdf"

        return filename

    def download_single(self, paper: DeduplicatedPaper) -> Optional[str]:
        """
        Download a single paper.

        Args:
            paper: Paper to download

        Returns:
            Download path or None
        """
        results = self.find_and_download([paper])
        result = results.get(paper.paper_id)

        if result and not result.startswith('Error') and not result.startswith('No open'):
            return result
        return None

    def get_download_summary(self, papers: List[DeduplicatedPaper]) -> Dict:
        """
        Get summary of download results.

        Args:
            papers: Papers that were processed

        Returns:
            Summary dict
        """
        total = len(papers)
        success = sum(1 for p in papers if p.download_status == 'success')
        failed = sum(1 for p in papers if p.download_status == 'failed')
        skipped = sum(1 for p in papers if p.download_status == 'skipped')
        pending = sum(1 for p in papers if p.download_status is None or p.download_status == 'pending')

        # OA status breakdown
        oa_statuses = {}
        for p in papers:
            status = p.oa_status or 'unknown'
            oa_statuses[status] = oa_statuses.get(status, 0) + 1

        # OA source breakdown
        oa_sources = {}
        for p in papers:
            if p.oa_source:
                oa_sources[p.oa_source] = oa_sources.get(p.oa_source, 0) + 1

        return {
            'total': total,
            'success': success,
            'failed': failed,
            'skipped': skipped,
            'pending': pending,
            'success_rate': success / total if total > 0 else 0,
            'oa_statuses': oa_statuses,
            'oa_sources': oa_sources
        }
