#!/usr/bin/env python3
"""
pdf_downloader.py - Download open-access PDFs for meta-analysis papers.

Tries multiple sources in priority order to find and download PDFs:
1. OpenAlex OA URL (if provided in search results)
2. Unpaywall API (free, requires email)
3. CORE API (core.ac.uk, ~200M OA papers, free API key)
4. CrossRef fulltext links (publisher-provided PDF URLs)
5. Fatcat / Internet Archive (~500M works, no key needed)
6. OpenAIRE (~100M works, no key needed)
7. Semantic Scholar API (open access PDF field)
8. PubMed Central via DOI→PMCID lookup (NCBI ID Converter)
9. PubMed Central (for papers with PMC IDs in input)
10. bioRxiv/medRxiv (preprint versions of published papers)
11. Publisher-specific URL patterns (MDPI, Frontiers, PLOS, Elsevier, Springer, Wiley, T&F, Cambridge)
12. Europe PMC (europepmc.org)
13. DOI redirect + landing page HTML parsing
14. Title-based Semantic Scholar search (fallback when DOI missing)

Usage:
    # From search_pipeline.py output:
    python pdf_downloader.py --input search_results.csv --output ./papers/ --email user@uni.edu

    # Specify column names:
    python pdf_downloader.py --input results.csv --output ./papers/ --doi-column doi --title-column title

    # Single DOI:
    python pdf_downloader.py --doi "10.7554/eLife.02245" --output ./papers/

    # From JSON (search_pipeline.py also outputs JSON):
    python pdf_downloader.py --input search_results.json --output ./papers/ --email user@uni.edu
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import requests
import urllib.parse

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pdf_downloader")

PDF_MAGIC = b"%PDF"
MIN_PDF_SIZE = 10_000  # 10 KB - smaller files are usually error pages
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DownloadResult:
    """Result of a single paper download attempt."""
    paper_id: str = ""
    doi: str = ""
    title: str = ""
    status: str = ""        # downloaded, already_exists, no_oa_available, download_failed, not_pdf
    filepath: str = ""      # Path to downloaded file, or empty
    source: str = ""        # openalex_oa, unpaywall, semantic_scholar, pmc, doi_redirect
    attempts: list = field(default_factory=list)   # Sources tried
    error: str = ""         # Error message if failed


# ---------------------------------------------------------------------------
# PDFDownloader
# ---------------------------------------------------------------------------

class PDFDownloader:
    """Download open-access PDFs from multiple sources."""

    # Semantic Scholar: 100 requests per 5 minutes = 1 per 3 seconds
    # Unpaywall: 100K/day, ~1/sec recommended
    # OpenAlex: 10/sec in polite pool

    def __init__(
        self,
        output_dir: str,
        email: str = "user@example.com",
        core_api_key: Optional[str] = None,
        delay: float = 1.0,
        max_retries: int = 2,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.email = email
        self.core_api_key = core_api_key or os.environ.get("CORE_API_KEY", "")
        self.delay = delay
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": f"MetaAnalysisPipeline/1.0 (mailto:{email})",
            "Accept": "application/pdf, application/json, */*",
        })
        self.download_log: list[DownloadResult] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def download_paper(self, paper: dict) -> DownloadResult:
        """
        Try to download PDF for a single paper.

        Args:
            paper: dict with keys: doi, title, year, authors.
                   Optional: oa_url (from OpenAlex), pmcid, openalex_id.

        Returns:
            DownloadResult with status and file path.
        """
        doi = (paper.get("doi") or "").strip()
        title = (paper.get("title") or "").strip()
        year = paper.get("year", 0)
        authors = (paper.get("authors") or "").strip()
        oa_url = (paper.get("oa_url") or "").strip()
        pmcid = (paper.get("pmcid") or paper.get("pmc_id") or "").strip()
        paper_id = paper.get("openalex_id") or doi or title[:60]

        result = DownloadResult(
            paper_id=paper_id,
            doi=doi,
            title=title,
        )

        if not doi and not oa_url:
            result.status = "download_failed"
            result.error = "No DOI or OA URL available"
            log.warning("Skipping '%s': no DOI or OA URL", title[:60])
            self.download_log.append(result)
            return result

        # Build target filename
        first_author = self._extract_first_author(authors)
        filename = self._sanitize_filename(title, year, first_author)
        filepath = self.output_dir / filename

        # Check if already downloaded
        if self._file_already_exists(first_author, year):
            result.status = "already_exists"
            result.filepath = str(self._find_existing(first_author, year))
            result.source = "cache"
            log.info("Already exists: %s", result.filepath)
            self.download_log.append(result)
            return result

        # Try sources in priority order
        sources = [
            ("openalex_oa", lambda fp: self._try_oa_url(oa_url, fp) if oa_url else False),
            ("unpaywall", lambda fp: self._try_unpaywall(doi, fp) if doi else False),
            ("core", lambda fp: self._try_core(doi, fp) if self.core_api_key and doi else False),
            ("crossref", lambda fp: self._try_crossref(doi, fp) if doi else False),
            ("fatcat", lambda fp: self._try_fatcat(doi, fp) if doi else False),
            ("openaire", lambda fp: self._try_openaire(doi, fp) if doi else False),
            ("semantic_scholar", lambda fp: self._try_semantic_scholar(doi, fp) if doi else False),
            ("pmc_lookup", lambda fp: self._try_pmc_from_doi(doi, fp) if doi and not pmcid else False),
            ("pmc", lambda fp: self._try_pmc(pmcid, fp) if pmcid else False),
            ("biorxiv", lambda fp: self._try_biorxiv(doi, title, fp) if doi or title else False),
            ("publisher_specific", lambda fp: self._try_publisher_specific(doi, fp) if doi else False),
            ("europe_pmc", lambda fp: self._try_europe_pmc(doi, fp) if doi else False),
            ("doi_redirect", lambda fp: self._try_doi_redirect(doi, fp) if doi else False),
            ("title_search", lambda fp: self._try_title_search(title, fp) if title and not doi else False),
        ]

        for source_name, try_fn in sources:
            result.attempts.append(source_name)
            log.info("Trying %s for '%s'...", source_name, title[:50])
            try:
                success = try_fn(filepath)
                if success:
                    result.status = "downloaded"
                    result.filepath = str(filepath)
                    result.source = source_name
                    log.info("Downloaded via %s: %s", source_name, filepath.name)
                    self.download_log.append(result)
                    return result
            except Exception as e:
                log.debug("Source %s failed: %s", source_name, e)

            time.sleep(self.delay)

        # No source worked
        result.status = "no_oa_available"
        result.error = f"Tried {len(result.attempts)} sources, none had open-access PDF"
        log.warning("No OA PDF found for '%s' (DOI: %s)", title[:50], doi)
        self.download_log.append(result)
        return result

    def download_batch(
        self,
        papers: list[dict],
        skip_existing: bool = True,
    ) -> list[DownloadResult]:
        """
        Download PDFs for multiple papers.

        Args:
            papers: List of paper dicts (from search_pipeline.py output).
            skip_existing: If True, skip papers whose files already exist.

        Returns:
            List of DownloadResult for each paper.
        """
        results = []
        total = len(papers)
        downloaded = 0
        skipped = 0
        failed = 0

        log.info("Starting batch download of %d papers to %s", total, self.output_dir)

        for i, paper in enumerate(papers, 1):
            log.info("--- [%d/%d] %s ---", i, total, (paper.get("title") or "")[:60])
            result = self.download_paper(paper)
            results.append(result)

            if result.status == "downloaded":
                downloaded += 1
            elif result.status == "already_exists":
                skipped += 1
            else:
                failed += 1

            # Progress update every 10 papers
            if i % 10 == 0:
                log.info(
                    "Progress: %d/%d (downloaded=%d, skipped=%d, failed=%d)",
                    i, total, downloaded, skipped, failed,
                )

        log.info(
            "Batch complete: %d downloaded, %d already existed, %d failed out of %d total",
            downloaded, skipped, failed, total,
        )
        return results

    def save_download_report(self, path: str):
        """Save CSV report of all download attempts and results."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "paper_id", "doi", "title", "status", "filepath",
            "source", "attempts", "error",
        ]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.download_log:
                row = asdict(result)
                row["attempts"] = "; ".join(row["attempts"])
                writer.writerow(row)

        log.info("Download report saved to %s (%d entries)", out_path, len(self.download_log))

    def verify_downloads(
        self,
        papers: list[dict],
        results: list[DownloadResult],
        output_path: Optional[str] = None,
    ) -> dict:
        """Verify downloaded PDFs match the intended papers.

        Extracts text from the first page of each PDF and checks if the
        paper's title appears in the text. Flags mismatches for review.

        Parameters
        ----------
        papers : list[dict]
            Original paper dicts with ``title`` and ``doi``.
        results : list[DownloadResult]
            Download results from ``download_batch()``.
        output_path : str or None
            Path to save verification report CSV.

        Returns
        -------
        dict
            ``{verified, mismatched, unreadable, skipped, details}``.
        """
        try:
            import fitz  # PyMuPDF
            has_pymupdf = True
        except ImportError:
            has_pymupdf = False
            log.warning(
                "PyMuPDF (fitz) not installed. Skipping PDF content verification. "
                "Install with: pip install PyMuPDF"
            )

        # Build lookup from result paper_id -> paper dict
        paper_lookup = {}
        for p in papers:
            pid = p.get("openalex_id") or p.get("doi") or p.get("title", "")[:60]
            paper_lookup[pid] = p

        verified = 0
        mismatched = 0
        unreadable = 0
        skipped = 0
        details = []

        for result in results:
            if result.status != "downloaded" or not result.filepath:
                skipped += 1
                continue

            filepath = Path(result.filepath)
            if not filepath.exists():
                skipped += 1
                continue

            paper = paper_lookup.get(result.paper_id, {})
            expected_title = (paper.get("title") or result.title or "").strip()

            if not has_pymupdf or not expected_title:
                skipped += 1
                continue

            try:
                doc = fitz.open(str(filepath))
                if len(doc) == 0:
                    unreadable += 1
                    details.append({
                        "paper_id": result.paper_id,
                        "doi": result.doi,
                        "expected_title": expected_title,
                        "status": "unreadable",
                        "reason": "Empty PDF (0 pages)",
                        "filepath": str(filepath),
                    })
                    doc.close()
                    continue

                # Extract first page text
                first_page_text = doc[0].get_text().lower()
                doc.close()

                # Normalize expected title for comparison
                norm_title = re.sub(r'[^\w\s]', '', expected_title.lower())
                title_words = norm_title.split()

                # Check if enough title words appear in first page
                if len(title_words) <= 3:
                    # Short title: require all words
                    matched_words = sum(1 for w in title_words if w in first_page_text)
                    match_ratio = matched_words / max(len(title_words), 1)
                else:
                    # Longer title: require 60%+ of significant words (4+ chars)
                    sig_words = [w for w in title_words if len(w) >= 4]
                    if not sig_words:
                        sig_words = title_words
                    matched_words = sum(1 for w in sig_words if w in first_page_text)
                    match_ratio = matched_words / max(len(sig_words), 1)

                if match_ratio >= 0.6:
                    verified += 1
                    details.append({
                        "paper_id": result.paper_id,
                        "doi": result.doi,
                        "expected_title": expected_title[:80],
                        "status": "verified",
                        "match_ratio": f"{match_ratio:.0%}",
                        "filepath": str(filepath),
                    })
                else:
                    mismatched += 1
                    # Extract what looks like the actual title (first ~200 chars)
                    actual_start = first_page_text[:200].strip()
                    details.append({
                        "paper_id": result.paper_id,
                        "doi": result.doi,
                        "expected_title": expected_title[:80],
                        "status": "MISMATCH",
                        "match_ratio": f"{match_ratio:.0%}",
                        "actual_text_start": actual_start[:100],
                        "filepath": str(filepath),
                    })
                    log.warning(
                        "PDF MISMATCH: '%s' (%.0f%% title match) - %s",
                        expected_title[:50], match_ratio * 100, filepath.name,
                    )

            except Exception as e:
                unreadable += 1
                details.append({
                    "paper_id": result.paper_id,
                    "doi": result.doi,
                    "expected_title": expected_title[:80],
                    "status": "unreadable",
                    "reason": str(e)[:100],
                    "filepath": str(filepath),
                })

        summary = {
            "verified": verified,
            "mismatched": mismatched,
            "unreadable": unreadable,
            "skipped": skipped,
            "total_checked": verified + mismatched + unreadable,
            "verification_rate": (
                f"{verified / max(verified + mismatched, 1):.0%}"
            ),
        }

        log.info(
            "PDF verification: %d verified, %d mismatched, %d unreadable, %d skipped",
            verified, mismatched, unreadable, skipped,
        )

        # Save verification report
        if output_path and details:
            out_path = Path(output_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = list(details[0].keys())
            # Union of all keys across details
            for d in details:
                for k in d:
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(details)
            log.info("Verification report saved to %s", out_path)

        return {**summary, "details": details}

    # -----------------------------------------------------------------------
    # Source-specific download methods
    # -----------------------------------------------------------------------

    def _try_oa_url(self, url: str, filepath: Path) -> bool:
        """Try downloading from an OpenAlex OA URL."""
        if not url:
            return False
        return self._download_file(url, filepath)

    def _try_unpaywall(self, doi: str, filepath: Path) -> bool:
        """Query Unpaywall API for PDF URL, then download."""
        if not doi:
            return False

        api_url = f"https://api.unpaywall.org/v2/{doi}?email={self.email}"
        try:
            resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.debug("Unpaywall returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()

            # Try best_oa_location first
            best = data.get("best_oa_location") or {}
            pdf_url = best.get("url_for_pdf") or best.get("url")

            # Fall back to any OA location with a PDF URL
            if not pdf_url:
                for loc in data.get("oa_locations") or []:
                    pdf_url = loc.get("url_for_pdf")
                    if pdf_url:
                        break

            if not pdf_url:
                log.debug("Unpaywall has no PDF URL for %s", doi)
                return False

            return self._download_file(pdf_url, filepath)

        except (requests.RequestException, json.JSONDecodeError) as e:
            log.debug("Unpaywall error for %s: %s", doi, e)
            return False

    def _try_core(self, doi: str, filepath: Path) -> bool:
        """Query CORE API (core.ac.uk) for open-access PDF URL, then download.

        CORE aggregates ~200M open-access papers from repositories worldwide.
        Free API key required (register at https://core.ac.uk/services/api).
        """
        if not doi or not self.core_api_key:
            return False

        api_url = "https://api.core.ac.uk/v3/search/works"
        headers = {
            "Authorization": f"Bearer {self.core_api_key}",
            "Accept": "application/json",
        }
        params = {
            "q": f"doi:{doi}",
            "limit": 1,
        }
        try:
            resp = self.session.get(
                api_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code == 429:
                log.debug("CORE rate limited, waiting 5s...")
                time.sleep(5)
                resp = self.session.get(
                    api_url, headers=headers, params=params, timeout=REQUEST_TIMEOUT,
                )

            if resp.status_code != 200:
                log.debug("CORE returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()
            results = data.get("results") or []
            if not results:
                log.debug("CORE has no results for %s", doi)
                return False

            paper = results[0]
            pdf_url = paper.get("downloadUrl") or ""

            # Also check sourceFulltextUrls
            if not pdf_url:
                for url in paper.get("sourceFulltextUrls") or []:
                    if url:
                        pdf_url = url
                        break

            if not pdf_url:
                log.debug("CORE has no download URL for %s", doi)
                return False

            return self._download_file(pdf_url, filepath)

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            log.debug("CORE error for %s: %s", doi, e)
            return False

    def _try_fatcat(self, doi: str, filepath: Path) -> bool:
        """Query Internet Archive's Fatcat API for archived PDFs.

        Fatcat indexes ~500M+ scholarly works with file metadata from
        Internet Archive, web archives, and institutional repositories.
        No API key required.
        """
        if not doi:
            return False

        api_url = f"https://api.fatcat.wiki/v0/release/lookup?doi={doi}&expand=files&hide=abstracts,refs"
        try:
            resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.debug("Fatcat returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()
            files = data.get("files") or []

            for f in files:
                urls = f.get("urls") or []
                for u in urls:
                    url = u.get("url", "")
                    if url and ("archive.org" in url or url.endswith(".pdf")):
                        if self._download_file(url, filepath):
                            return True
                # Try any URL if archive.org-specific ones didn't work
                for u in urls:
                    url = u.get("url", "")
                    if url:
                        if self._download_file(url, filepath):
                            return True

            log.debug("Fatcat has no downloadable files for %s", doi)
            return False

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            log.debug("Fatcat error for %s: %s", doi, e)
            return False

    def _try_openaire(self, doi: str, filepath: Path) -> bool:
        """Query OpenAIRE for open-access PDFs.

        OpenAIRE indexes ~100M+ research products from European and
        global repositories. No API key required.
        """
        if not doi:
            return False

        api_url = "https://api.openaire.eu/search/publications"
        params = {
            "doi": doi,
            "format": "json",
            "size": 1,
        }
        try:
            resp = self.session.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.debug("OpenAIRE returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()
            # Navigate the OpenAIRE response structure
            response = data.get("response") or {}
            results = response.get("results") or {}
            result_list = results.get("result") or []

            if not result_list:
                log.debug("OpenAIRE has no results for %s", doi)
                return False

            # Get the first result's metadata
            result = result_list[0]
            metadata = result.get("metadata") or {}
            oaf = metadata.get("oaf:entity") or {}
            oaf_result = oaf.get("oaf:result") or {}

            # Look for web resources (PDF URLs)
            children = oaf_result.get("children") or {}
            instances = children.get("instance") or []
            if isinstance(instances, dict):
                instances = [instances]

            for inst in instances:
                # Check for open access
                access = inst.get("accessright", {})
                if isinstance(access, dict):
                    access_code = access.get("@classid", "")
                else:
                    access_code = str(access)

                web_urls = inst.get("webresource") or []
                if isinstance(web_urls, dict):
                    web_urls = [web_urls]

                for wr in web_urls:
                    url = wr.get("url", "")
                    if url:
                        if self._download_file(url, filepath):
                            return True

            log.debug("OpenAIRE has no downloadable PDF for %s", doi)
            return False

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            log.debug("OpenAIRE error for %s: %s", doi, e)
            return False

    def _try_semantic_scholar(self, doi: str, filepath: Path) -> bool:
        """Query Semantic Scholar for open access PDF."""
        if not doi:
            return False

        api_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=openAccessPdf"
        try:
            resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                log.debug("Semantic Scholar rate limited, waiting 5s...")
                time.sleep(5)
                resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)

            if resp.status_code != 200:
                log.debug("Semantic Scholar returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()
            oa_pdf = data.get("openAccessPdf") or {}
            pdf_url = oa_pdf.get("url")

            if not pdf_url:
                log.debug("Semantic Scholar has no OA PDF for %s", doi)
                return False

            return self._download_file(pdf_url, filepath)

        except (requests.RequestException, json.JSONDecodeError) as e:
            log.debug("Semantic Scholar error for %s: %s", doi, e)
            return False

    def _try_pmc(self, pmcid: str, filepath: Path) -> bool:
        """Download from PubMed Central."""
        if not pmcid:
            return False

        # Normalize PMC ID
        pmcid = pmcid.strip().upper()
        if not pmcid.startswith("PMC"):
            pmcid = f"PMC{pmcid}"

        pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        return self._download_file(pdf_url, filepath)

    def _try_doi_redirect(self, doi: str, filepath: Path) -> bool:
        """
        Follow DOI redirect and check if it resolves to a PDF.
        Some publishers serve the PDF directly at the DOI URL.
        """
        if not doi:
            return False

        doi_url = f"https://doi.org/{doi}"
        try:
            # Use HEAD first to check Content-Type without downloading
            resp = self.session.head(
                doi_url, timeout=REQUEST_TIMEOUT, allow_redirects=True,
            )
            content_type = resp.headers.get("Content-Type", "")

            if "pdf" in content_type.lower():
                # DOI resolves directly to PDF
                return self._download_file(resp.url, filepath)

            # Some publishers put PDF at predictable URLs
            final_url = resp.url
            # Try appending /pdf or .pdf to the resolved URL
            pdf_variants = []
            if not final_url.endswith("/"):
                pdf_variants.append(final_url + ".pdf")
            pdf_variants.append(final_url.rstrip("/") + "/pdf")

            for variant_url in pdf_variants:
                head = self.session.head(
                    variant_url, timeout=REQUEST_TIMEOUT, allow_redirects=True,
                )
                if head.status_code == 200 and "pdf" in head.headers.get("Content-Type", "").lower():
                    return self._download_file(variant_url, filepath)

            # Last resort: parse landing page HTML for PDF links
            return self._try_links_from_page(final_url, filepath)

        except requests.RequestException as e:
            log.debug("DOI redirect error for %s: %s", doi, e)
            return False

    # -----------------------------------------------------------------------
    # Publisher-specific download strategies
    # -----------------------------------------------------------------------

    def _try_publisher_specific(self, doi: str, filepath: Path) -> bool:
        """Try publisher-specific URL patterns based on DOI prefix."""
        if not doi or "/" not in doi:
            return False

        prefix = doi.split("/")[0]

        # Map DOI prefixes to publisher download strategies
        strategies = {
            "10.3390": self._try_mdpi,       # MDPI (always OA)
            "10.3389": self._try_frontiers,   # Frontiers (always OA)
            "10.1371": self._try_plos,        # PLOS (always OA)
            "10.1016": self._try_elsevier,    # Elsevier / ScienceDirect
            "10.1007": self._try_springer,    # Springer / Nature
            "10.1038": self._try_springer,    # Nature
            "10.1002": self._try_wiley,       # Wiley
            "10.1111": self._try_wiley,       # Wiley (old)
            "10.1080": self._try_taylor_francis,  # Taylor & Francis
            "10.1017": self._try_cambridge,   # Cambridge UP
            "10.2134": self._try_wiley,       # ACSESS (now on Wiley)
            "10.2136": self._try_wiley,       # ACSESS
            "10.1094": self._try_landing_page_pdf,  # APS
            "10.1614": self._try_cambridge,   # WSSA (some on Cambridge)
        }

        strategy = strategies.get(prefix)
        if strategy:
            return strategy(doi, filepath)

        return False

    def _resolve_doi_url(self, doi: str) -> str | None:
        """Follow DOI redirect to get the landing page URL."""
        try:
            resp = self.session.head(
                f"https://doi.org/{doi}", timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            return resp.url
        except requests.RequestException:
            try:
                resp = self.session.get(
                    f"https://doi.org/{doi}", timeout=REQUEST_TIMEOUT,
                    allow_redirects=True, stream=True,
                )
                return resp.url
            except requests.RequestException:
                return None

    def _extract_pdf_links_from_html(self, resp: requests.Response) -> list[str]:
        """Parse HTML response looking for PDF download links."""
        if not HAS_BS4 or not resp or not resp.text:
            return []
        try:
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception:
            return []
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = (a.get_text() or "").lower()
            if any(kw in href.lower() for kw in [".pdf", "/pdf", "pdfft", "pdfdirect"]):
                links.append(href)
            elif any(kw in text for kw in ["download pdf", "full text pdf", "download article"]):
                links.append(href)
        return links

    def _try_links_from_page(self, landing_url: str, filepath: Path) -> bool:
        """Download landing page, parse for PDF links, try each."""
        try:
            resp = self.session.get(landing_url, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            return False
        if not resp or resp.status_code != 200:
            return False
        # Check if the landing page itself is a PDF
        if self._is_valid_pdf(resp.content):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(resp.content)
            return True
        links = self._extract_pdf_links_from_html(resp)
        for link in links[:5]:  # cap at 5 to avoid runaway
            if not link.startswith("http"):
                link = urllib.parse.urljoin(landing_url, link)
            if self._download_file(link, filepath):
                return True
        return False

    def _try_mdpi(self, doi: str, filepath: Path) -> bool:
        """MDPI (10.3390) - always open access, /pdf suffix."""
        landing = self._resolve_doi_url(doi)
        if not landing:
            return False
        if "mdpi.com" in landing:
            pdf_url = landing.rstrip("/") + "/pdf"
            if self._download_file(pdf_url, filepath):
                return True
        return self._try_links_from_page(landing, filepath)

    def _try_frontiers(self, doi: str, filepath: Path) -> bool:
        """Frontiers (10.3389) - always open access."""
        landing = self._resolve_doi_url(doi)
        if not landing:
            return False
        if "frontiersin.org" in landing:
            pdf_url = landing.replace("/full", "/pdf").replace("/abstract", "/pdf")
            if not pdf_url.endswith("/pdf"):
                pdf_url = pdf_url.rstrip("/") + "/pdf"
            if self._download_file(pdf_url, filepath):
                return True
        return False

    def _try_plos(self, doi: str, filepath: Path) -> bool:
        """PLOS (10.1371) - always open access."""
        url = f"https://journals.plos.org/plosone/article/file?id={doi}&type=printable"
        return self._download_file(url, filepath)

    def _try_elsevier(self, doi: str, filepath: Path) -> bool:
        """Elsevier (10.1016) - try ScienceDirect PDF patterns."""
        landing = self._resolve_doi_url(doi)
        if not landing:
            return False

        # Extract PII from URL
        pii_match = re.search(r'/pii/([A-Z0-9]+)', landing)
        if pii_match:
            pii = pii_match.group(1)
            for suffix in ["/pdfft", "/pdf"]:
                pdf_url = f"https://www.sciencedirect.com/science/article/pii/{pii}{suffix}"
                if self._download_file(pdf_url, filepath):
                    return True

        return self._try_links_from_page(landing, filepath)

    def _try_springer(self, doi: str, filepath: Path) -> bool:
        """Springer/Nature (10.1007, 10.1038) - content/pdf pattern."""
        pdf_url = f"https://link.springer.com/content/pdf/{doi}.pdf"
        if self._download_file(pdf_url, filepath):
            return True

        landing = self._resolve_doi_url(doi)
        if landing and "link.springer.com" in landing:
            pdf_url2 = landing.replace("/article/", "/content/pdf/") + ".pdf"
            if self._download_file(pdf_url2, filepath):
                return True

        if landing:
            return self._try_links_from_page(landing, filepath)
        return False

    def _try_wiley(self, doi: str, filepath: Path) -> bool:
        """Wiley (10.1002, 10.1111, 10.2134, 10.2136) - pdfdirect pattern."""
        pdf_url = f"https://onlinelibrary.wiley.com/doi/pdfdirect/{doi}?download=true"
        if self._download_file(pdf_url, filepath):
            return True

        pdf_url2 = f"https://onlinelibrary.wiley.com/doi/epdf/{doi}"
        if self._download_file(pdf_url2, filepath):
            return True

        landing = self._resolve_doi_url(doi)
        if landing and "wiley" in landing:
            return self._try_links_from_page(landing, filepath)
        return False

    def _try_taylor_francis(self, doi: str, filepath: Path) -> bool:
        """Taylor & Francis (10.1080)."""
        pdf_url = f"https://www.tandfonline.com/doi/pdf/{doi}?download=true"
        if self._download_file(pdf_url, filepath):
            return True

        landing = self._resolve_doi_url(doi)
        if landing:
            return self._try_links_from_page(landing, filepath)
        return False

    def _try_cambridge(self, doi: str, filepath: Path) -> bool:
        """Cambridge University Press (10.1017)."""
        landing = self._resolve_doi_url(doi)
        if not landing:
            return False
        if "cambridge.org" in landing:
            pdf_url = landing.rstrip("/") + "/pdf"
            if self._download_file(pdf_url, filepath):
                return True
        return self._try_links_from_page(landing, filepath)

    def _try_landing_page_pdf(self, doi: str, filepath: Path) -> bool:
        """Generic: resolve DOI and parse landing page for PDF links."""
        landing = self._resolve_doi_url(doi)
        if not landing:
            return False
        return self._try_links_from_page(landing, filepath)

    def _try_europe_pmc(self, doi: str, filepath: Path) -> bool:
        """Try Europe PMC for open-access PDF."""
        if not doi:
            return False

        api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:{doi}&format=json"
        try:
            resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                return False

            data = resp.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                return False

            result = results[0]
            pmcid = result.get("pmcid")

            if result.get("hasPDF") == "Y" and pmcid:
                pdf_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                if self._download_file(pdf_url, filepath):
                    return True

            if pmcid:
                pdf_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
                if self._download_file(pdf_url, filepath):
                    return True

        except (requests.RequestException, json.JSONDecodeError, KeyError):
            pass
        return False

    # -----------------------------------------------------------------------
    # New sources (added 2026-03-25)
    # -----------------------------------------------------------------------

    def _try_crossref(self, doi: str, filepath: Path) -> bool:
        """Query CrossRef API for fulltext PDF links.

        CrossRef stores publisher-provided fulltext links for many papers.
        These often include direct PDF URLs even for paywalled papers
        that have OA versions via author-deposit or green OA.
        No API key required; email enables polite pool.
        """
        if not doi:
            return False

        api_url = f"https://api.crossref.org/works/{doi}"
        headers = {"User-Agent": f"MetaAnalysisPipeline/1.0 (mailto:{self.email})"}
        try:
            resp = self.session.get(api_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.debug("CrossRef returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()
            message = data.get("message", {})

            # Check for fulltext links
            links = message.get("link", [])
            for link in links:
                url = link.get("URL", "")
                content_type = link.get("content-type", "")
                if "pdf" in content_type.lower() and url:
                    if self._download_file(url, filepath):
                        return True

            # Check for license links that indicate OA
            licenses = message.get("license", [])
            is_oa = any(
                "creativecommons" in (lic.get("URL", "") or "").lower()
                for lic in licenses
            )

            # If OA and we have a resource link, try it
            if is_oa:
                resource = message.get("resource", {})
                primary = resource.get("primary", {})
                primary_url = primary.get("URL", "")
                if primary_url:
                    return self._try_links_from_page(primary_url, filepath)

            return False

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            log.debug("CrossRef error for %s: %s", doi, e)
            return False

    def _try_pmc_from_doi(self, doi: str, filepath: Path) -> bool:
        """Look up PMC ID from DOI via NCBI ID Converter, then download from PMC.

        Many papers are deposited in PMC but the input data doesn't include
        the PMC ID. This converts DOI → PMCID using the NCBI converter API.
        """
        if not doi:
            return False

        api_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {
            "ids": doi,
            "format": "json",
            "tool": "meta_analysis_pipeline",
            "email": self.email,
        }
        try:
            resp = self.session.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                log.debug("NCBI ID Converter returned %d for %s", resp.status_code, doi)
                return False

            data = resp.json()
            records = data.get("records", [])
            if not records:
                return False

            pmcid = records[0].get("pmcid", "")
            if not pmcid:
                log.debug("No PMC ID found for DOI %s", doi)
                return False

            log.debug("Found PMC ID %s for DOI %s", pmcid, doi)
            return self._try_pmc(pmcid, filepath)

        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            log.debug("NCBI ID Converter error for %s: %s", doi, e)
            return False

    def _try_biorxiv(self, doi: str, title: str, filepath: Path) -> bool:
        """Try bioRxiv/medRxiv for preprint versions.

        Many published papers have earlier preprint versions on bioRxiv/medRxiv.
        The bioRxiv API can find these from the published DOI.
        """
        if not doi:
            return False

        # Check if this is directly a bioRxiv/medRxiv DOI
        if doi.startswith("10.1101/"):
            pdf_url = f"https://www.biorxiv.org/content/{doi}v1.full.pdf"
            if self._download_file(pdf_url, filepath):
                return True
            pdf_url = f"https://www.medrxiv.org/content/{doi}v1.full.pdf"
            if self._download_file(pdf_url, filepath):
                return True

        # Try bioRxiv API to find preprint version of a published paper
        api_url = f"https://api.biorxiv.org/details/biorxiv/{doi}"
        try:
            resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                collection = data.get("collection", [])
                if collection:
                    biorxiv_doi = collection[0].get("biorxiv_doi", "")
                    if biorxiv_doi:
                        pdf_url = f"https://www.biorxiv.org/content/10.1101/{biorxiv_doi}v1.full.pdf"
                        if self._download_file(pdf_url, filepath):
                            return True
        except (requests.RequestException, json.JSONDecodeError):
            pass

        # Try medRxiv too
        api_url = f"https://api.biorxiv.org/details/medrxiv/{doi}"
        try:
            resp = self.session.get(api_url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                collection = data.get("collection", [])
                if collection:
                    medrxiv_doi = collection[0].get("biorxiv_doi", "")
                    if medrxiv_doi:
                        pdf_url = f"https://www.medrxiv.org/content/10.1101/{medrxiv_doi}v1.full.pdf"
                        if self._download_file(pdf_url, filepath):
                            return True
        except (requests.RequestException, json.JSONDecodeError):
            pass

        return False

    def _try_title_search(self, title: str, filepath: Path) -> bool:
        """Search Semantic Scholar by title as last resort when DOI is missing.

        This catches papers that have OA PDFs but weren't found because
        the DOI was missing or malformed in the search results.
        """
        if not title or len(title) < 10:
            return False

        # Clean title for search
        clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
        if len(clean_title) < 10:
            return False

        api_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": clean_title[:200],
            "limit": 3,
            "fields": "openAccessPdf,title",
        }
        try:
            resp = self.session.get(api_url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(5)
                resp = self.session.get(api_url, params=params, timeout=REQUEST_TIMEOUT)

            if resp.status_code != 200:
                return False

            data = resp.json()
            papers = data.get("data", [])

            for paper in papers:
                # Verify title similarity before downloading
                found_title = (paper.get("title") or "").lower()
                query_title = title.lower()
                # Simple word overlap check
                query_words = set(query_title.split())
                found_words = set(found_title.split())
                if len(query_words) > 0:
                    overlap = len(query_words & found_words) / len(query_words)
                    if overlap < 0.5:
                        continue

                oa_pdf = paper.get("openAccessPdf") or {}
                pdf_url = oa_pdf.get("url")
                if pdf_url:
                    if self._download_file(pdf_url, filepath):
                        return True

            return False

        except (requests.RequestException, json.JSONDecodeError) as e:
            log.debug("Title search error for '%s': %s", title[:40], e)
            return False

    # -----------------------------------------------------------------------
    # File handling utilities
    # -----------------------------------------------------------------------

    def _download_file(self, url: str, filepath: Path) -> bool:
        """
        Download a file from URL. Verify it is a real PDF.

        Returns True if download succeeded and file is a valid PDF.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(
                    url, timeout=REQUEST_TIMEOUT, allow_redirects=True, stream=True,
                )
                if resp.status_code != 200:
                    log.debug("HTTP %d from %s (attempt %d)", resp.status_code, url[:80], attempt)
                    continue

                # Read content
                content = resp.content

                # Verify it is actually a PDF
                if not self._is_valid_pdf(content):
                    log.debug(
                        "Not a valid PDF from %s (size=%d, starts with %r)",
                        url[:80], len(content), content[:10],
                    )
                    return False  # Don't retry - the URL serves non-PDF content

                # Write to disk
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_bytes(content)
                log.debug("Saved %d bytes to %s", len(content), filepath.name)
                return True

            except requests.Timeout:
                log.debug("Timeout downloading %s (attempt %d/%d)", url[:80], attempt, self.max_retries)
            except requests.RequestException as e:
                log.debug("Download error %s (attempt %d/%d): %s", url[:80], attempt, self.max_retries, e)

            if attempt < self.max_retries:
                time.sleep(2 * attempt)  # Exponential-ish backoff

        return False

    def _is_valid_pdf(self, content: bytes) -> bool:
        """Check that content is a real PDF (magic bytes + minimum size)."""
        if len(content) < MIN_PDF_SIZE:
            return False
        if not content[:4] == PDF_MAGIC:
            return False
        return True

    def _sanitize_filename(self, title: str, year: int, first_author: str) -> str:
        """
        Generate filename like 'AuthorLastName_Year_ShortTitle.pdf'.

        Takes the first 5 words of the title after removing special characters.
        """
        # Clean author
        author_clean = re.sub(r"[^\w]", "", first_author) if first_author else "Unknown"

        # Clean title: take first 5 words
        title_words = re.sub(r"[^\w\s]", "", title).split()
        short_title = "_".join(title_words[:5]) if title_words else "untitled"

        # Build filename
        year_str = str(year) if year else "XXXX"
        filename = f"{author_clean}_{year_str}_{short_title}.pdf"

        # Final sanitization: remove anything that is not alphanumeric, underscore, hyphen, or dot
        filename = re.sub(r"[^\w.\-]", "_", filename)
        # Collapse multiple underscores
        filename = re.sub(r"_+", "_", filename)

        return filename

    def _extract_first_author(self, authors: str) -> str:
        """Extract the last name of the first author from an author string."""
        if not authors:
            return "Unknown"

        # Handle "LastName, FirstName; ..." or "FirstName LastName, ..." formats
        # Split on semicolons or " and " or " & "
        first = re.split(r"[;&]|\band\b", authors, maxsplit=1)[0].strip()

        if not first:
            return "Unknown"

        # If "Last, First" format
        if "," in first:
            return first.split(",")[0].strip()

        # If "First Last" format, take the last word
        parts = first.split()
        return parts[-1] if parts else "Unknown"

    def _file_already_exists(self, first_author: str, year: int) -> bool:
        """Check if a PDF matching this author+year already exists."""
        if not first_author or first_author == "Unknown" or not year:
            return False
        pattern = f"{first_author}_{year}_*.pdf"
        matches = list(self.output_dir.glob(pattern))
        return len(matches) > 0

    def _find_existing(self, first_author: str, year: int) -> Optional[Path]:
        """Find existing file matching author+year."""
        if not first_author or not year:
            return None
        pattern = f"{first_author}_{year}_*.pdf"
        matches = list(self.output_dir.glob(pattern))
        return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Input loading helpers
# ---------------------------------------------------------------------------

def load_papers_from_csv(
    csv_path: str,
    doi_column: str = "doi",
    title_column: str = "title",
    year_column: str = "year",
    authors_column: str = "authors",
) -> list[dict]:
    """
    Load paper records from a CSV file.

    Flexibly maps column names. Looks for oa_url, pmcid columns if present.
    """
    papers = []
    path = Path(csv_path)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        available_cols = reader.fieldnames or []

        # Auto-detect column names (case-insensitive matching)
        col_map = {}
        for target, default in [
            ("doi", doi_column),
            ("title", title_column),
            ("year", year_column),
            ("authors", authors_column),
        ]:
            # Try exact match, then case-insensitive
            if default in available_cols:
                col_map[target] = default
            else:
                for col in available_cols:
                    if col.lower() == default.lower():
                        col_map[target] = col
                        break

        # Also look for optional columns
        for optional in ["oa_url", "pmcid", "pmc_id", "openalex_id"]:
            for col in available_cols:
                if col.lower() == optional.lower():
                    col_map[optional] = col
                    break

        for row in reader:
            paper = {}
            for target, col_name in col_map.items():
                paper[target] = row.get(col_name, "")

            # Parse year as int
            try:
                paper["year"] = int(paper.get("year", 0))
            except (ValueError, TypeError):
                paper["year"] = 0

            papers.append(paper)

    log.info("Loaded %d papers from %s", len(papers), csv_path)
    return papers


def load_papers_from_json(json_path: str) -> list[dict]:
    """
    Load paper records from a JSON file.

    Handles both a list of paper dicts and the search_pipeline.py output
    format (which has papers nested under various keys).
    """
    path = Path(json_path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If it is already a list of papers
    if isinstance(data, list):
        papers = data
    elif isinstance(data, dict):
        # search_pipeline.py output: look for 'papers', 'results', or 'records' key
        for key in ["papers", "results", "records", "all_papers"]:
            if key in data and isinstance(data[key], list):
                papers = data[key]
                break
        else:
            # Maybe the dict itself has doi/title keys (single paper)
            if "doi" in data or "title" in data:
                papers = [data]
            else:
                log.error("Could not find paper list in JSON. Top-level keys: %s", list(data.keys()))
                return []
    else:
        log.error("Unexpected JSON structure: %s", type(data))
        return []

    # Normalize year to int
    for p in papers:
        try:
            p["year"] = int(p.get("year", 0))
        except (ValueError, TypeError):
            p["year"] = 0

    log.info("Loaded %d papers from %s", len(papers), json_path)
    return papers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download open-access PDFs for meta-analysis papers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From search_pipeline.py CSV output:
  python pdf_downloader.py --input search_results.csv --output ./papers/ --email user@uni.edu

  # From JSON:
  python pdf_downloader.py --input results.json --output ./papers/ --email user@uni.edu

  # Single DOI:
  python pdf_downloader.py --doi "10.7554/eLife.02245" --output ./papers/

  # Custom column names:
  python pdf_downloader.py --input data.csv --output ./papers/ --doi-column DOI --title-column Title
        """,
    )
    parser.add_argument(
        "--input", "-i",
        help="Path to CSV or JSON file with paper records (from search_pipeline.py)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Directory to save downloaded PDFs",
    )
    parser.add_argument(
        "--email", "-e",
        default="user@example.com",
        help="Email for API polite pool (Unpaywall, OpenAlex). Default: user@example.com",
    )
    parser.add_argument(
        "--doi",
        help="Download a single paper by DOI",
    )
    parser.add_argument(
        "--doi-column",
        default="doi",
        help="Name of the DOI column in input CSV (default: doi)",
    )
    parser.add_argument(
        "--title-column",
        default="title",
        help="Name of the title column in input CSV (default: title)",
    )
    parser.add_argument(
        "--core-api-key",
        default=None,
        help="CORE API key for additional OA PDF discovery (free at https://core.ac.uk/services/api). Falls back to CORE_API_KEY env var.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between download attempts (default: 1.0)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retries per download attempt (default: 2)",
    )
    parser.add_argument(
        "--report",
        help="Path to save download report CSV (default: {output}/download_report.csv)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input and not args.doi:
        parser.error("Provide either --input (CSV/JSON file) or --doi (single DOI)")

    # Initialize downloader
    downloader = PDFDownloader(
        output_dir=args.output,
        email=args.email,
        core_api_key=args.core_api_key,
        delay=args.delay,
        max_retries=args.max_retries,
    )

    if args.doi:
        # Single DOI mode
        paper = {"doi": args.doi, "title": args.doi, "year": 0, "authors": ""}
        result = downloader.download_paper(paper)
        if result.status == "downloaded":
            print(f"Downloaded: {result.filepath}")
        else:
            print(f"Failed: {result.status} - {result.error}")
            sys.exit(1)
    else:
        # Batch mode from file
        input_path = Path(args.input)
        if not input_path.exists():
            log.error("Input file not found: %s", input_path)
            sys.exit(1)

        # Load papers based on file extension
        if input_path.suffix.lower() == ".json":
            papers = load_papers_from_json(str(input_path))
        else:
            papers = load_papers_from_csv(
                str(input_path),
                doi_column=args.doi_column,
                title_column=args.title_column,
            )

        if not papers:
            log.error("No papers loaded from %s", input_path)
            sys.exit(1)

        # Download all
        results = downloader.download_batch(papers)

        # Summary
        statuses = {}
        for r in results:
            statuses[r.status] = statuses.get(r.status, 0) + 1

        print("\n--- Download Summary ---")
        for status, count in sorted(statuses.items()):
            print(f"  {status}: {count}")
        print(f"  Total: {len(results)}")

        # Source breakdown for successful downloads
        sources = {}
        for r in results:
            if r.status == "downloaded":
                sources[r.source] = sources.get(r.source, 0) + 1
        if sources:
            print("\nPDFs by source:")
            for source, count in sorted(sources.items(), key=lambda x: -x[1]):
                print(f"  {source}: {count}")

    # Save report
    report_path = args.report or str(Path(args.output) / "download_report.csv")
    downloader.save_download_report(report_path)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
