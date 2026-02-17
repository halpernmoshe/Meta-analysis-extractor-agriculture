"""
Search Module - Main Coordinator

Coordinates the full paper acquisition pipeline:
Search -> Deduplicate -> Screen -> Find OA -> Download
"""

from typing import List, Dict, Optional, Callable
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.acquisition_state import (
    AcquisitionState, AcquisitionPhase,
    SearchQuery, SearchResult, DeduplicatedPaper
)
from .sources import PubMedAdapter, SemanticScholarAdapter, OpenAlexAdapter
from .dedup import DeduplicationModule
from .screen import ScreenModule
from .download import DownloadModule


class SearchModule:
    """
    Main coordinator for paper acquisition pipeline.

    Orchestrates search, deduplication, screening, and download.
    Supports checkpoint/resume for long-running operations.
    """

    CHECKPOINT_INTERVAL = 50

    def __init__(self,
                 output_dir: str,
                 download_dir: str = None,
                 llm_client = None,
                 unpaywall_email: str = None,
                 pubmed_email: str = None,
                 pubmed_api_key: str = None,
                 semantic_scholar_api_key: str = None):
        """
        Initialize search module.

        Args:
            output_dir: Directory for state and output files
            download_dir: Directory for downloaded PDFs (default: output_dir/pdfs)
            llm_client: LLMClient for screening (optional)
            unpaywall_email: Email for Unpaywall API
            pubmed_email: Email for PubMed API
            pubmed_api_key: NCBI API key (optional)
            semantic_scholar_api_key: Semantic Scholar API key (optional)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.download_dir = Path(download_dir) if download_dir else self.output_dir / 'pdfs'
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.state_file = self.output_dir / 'acquisition_state.json'

        # Initialize source adapters
        self.sources = {
            'pubmed': PubMedAdapter(
                api_key=pubmed_api_key,
                email=pubmed_email or unpaywall_email
            ),
            'semantic_scholar': SemanticScholarAdapter(
                api_key=semantic_scholar_api_key
            ),
            'openalex': OpenAlexAdapter(
                email=unpaywall_email or pubmed_email
            )
        }

        # Initialize modules
        self.dedup = DeduplicationModule()
        self.screener = ScreenModule(llm_client) if llm_client else None
        self.downloader = DownloadModule(
            str(self.download_dir),
            unpaywall_email or pubmed_email
        ) if (unpaywall_email or pubmed_email) else None

        # State
        self.state: Optional[AcquisitionState] = None

    def run(self,
           query: SearchQuery,
           pico: Dict,
           sources_enabled: Dict[str, bool] = None,
           max_results_per_source: int = 1000,
           skip_screening: bool = False,
           skip_download: bool = False,
           resume: bool = True,
           progress_callback: Optional[Callable[[str, int, int, str], None]] = None
           ) -> AcquisitionState:
        """
        Run the full acquisition pipeline.

        Args:
            query: Search query specification
            pico: PICO criteria for screening
            sources_enabled: Which sources to use
            max_results_per_source: Max results per source
            skip_screening: Skip LLM screening phase
            skip_download: Skip PDF download phase
            resume: Resume from checkpoint if available
            progress_callback: Callback(phase, current, total, message)

        Returns:
            Final AcquisitionState
        """
        # Check for existing state
        if resume and self.state_file.exists():
            self.state = AcquisitionState.load(str(self.state_file))
            print(f"Resuming from phase: {self.state.current_phase.value}")
        else:
            self.state = AcquisitionState()
            self.state.output_directory = str(self.output_dir)
            self.state.download_directory = str(self.download_dir)
            self.state.query = query
            self.state.pico = pico

        # Update source settings
        if sources_enabled:
            self.state.sources_enabled = sources_enabled
        self.state.max_results_per_source = max_results_per_source

        # Run phases
        try:
            # Phase 1: Search
            if self.state.current_phase in [AcquisitionPhase.INIT, AcquisitionPhase.SEARCH]:
                self._phase_search(progress_callback)
                self._save_checkpoint()

            # Phase 2: Deduplicate
            if self.state.current_phase == AcquisitionPhase.SEARCH:
                self.state.current_phase = AcquisitionPhase.DEDUPLICATE

            if self.state.current_phase == AcquisitionPhase.DEDUPLICATE:
                self._phase_deduplicate(progress_callback)
                self._save_checkpoint()

            # Phase 3: Screen
            if not skip_screening and self.screener:
                if self.state.current_phase == AcquisitionPhase.DEDUPLICATE:
                    self.state.current_phase = AcquisitionPhase.SCREEN

                if self.state.current_phase == AcquisitionPhase.SCREEN:
                    self._phase_screen(progress_callback)
                    self._save_checkpoint()

            # Phase 4: Find OA
            if self.state.current_phase in [AcquisitionPhase.DEDUPLICATE, AcquisitionPhase.SCREEN]:
                self.state.current_phase = AcquisitionPhase.FIND_OA

            if self.state.current_phase == AcquisitionPhase.FIND_OA and self.downloader:
                self._phase_find_oa(progress_callback)
                self._save_checkpoint()

            # Phase 5: Download
            if not skip_download and self.downloader:
                if self.state.current_phase == AcquisitionPhase.FIND_OA:
                    self.state.current_phase = AcquisitionPhase.DOWNLOAD

                if self.state.current_phase == AcquisitionPhase.DOWNLOAD:
                    self._phase_download(progress_callback)
                    self._save_checkpoint()

            # Complete
            self.state.current_phase = AcquisitionPhase.COMPLETE
            self._save_checkpoint()

            # Generate reports
            self._generate_reports()

        except KeyboardInterrupt:
            print("\nInterrupted. Saving checkpoint...")
            self._save_checkpoint()
            raise

        except Exception as e:
            print(f"\nError: {e}")
            self.state.add_error(self.state.current_phase.value, str(e))
            self._save_checkpoint()
            raise

        return self.state

    def _phase_search(self, progress_callback: Optional[Callable]):
        """Execute search phase"""
        self.state.current_phase = AcquisitionPhase.SEARCH

        for source_name, enabled in self.state.sources_enabled.items():
            if not enabled:
                continue

            # Skip if already completed
            if source_name in self.state.search_completed_sources:
                continue

            adapter = self.sources.get(source_name)
            if not adapter:
                continue

            if progress_callback:
                progress_callback('search', 0, 100, f"Searching {source_name}...")

            try:
                results = adapter.search(
                    self.state.query,
                    max_results=self.state.max_results_per_source,
                    progress_callback=lambda c, t, m: progress_callback(
                        'search', c, t, f"[{source_name}] {m}"
                    ) if progress_callback else None
                )

                # Store results
                self.state.raw_results[source_name] = [r.to_dict() for r in results]
                self.state.search_completed_sources.append(source_name)
                self.state.total_raw_results += len(results)

                print(f"  {source_name}: {len(results)} results")

            except Exception as e:
                self.state.add_error('search', f"{source_name}: {str(e)}")
                print(f"  {source_name}: Error - {e}")

        if progress_callback:
            progress_callback('search', 100, 100,
                            f"Search complete: {self.state.total_raw_results} total results")

    def _phase_deduplicate(self, progress_callback: Optional[Callable]):
        """Execute deduplication phase"""
        if progress_callback:
            progress_callback('dedup', 0, 100, "Deduplicating results...")

        # Convert stored dicts back to SearchResult objects
        results_by_source = {}
        for source, result_dicts in self.state.raw_results.items():
            results_by_source[source] = [SearchResult.from_dict(d) for d in result_dicts]

        # Deduplicate
        deduplicated = self.dedup.deduplicate(
            results_by_source,
            progress_callback=lambda c, t, m: progress_callback('dedup', c, t, m)
                if progress_callback else None
        )

        # Store as dicts
        self.state.deduplicated_papers = {p.paper_id: p.to_dict() for p in deduplicated}
        self.state.total_after_dedup = len(deduplicated)

        print(f"  Deduplicated: {self.state.total_raw_results} -> {self.state.total_after_dedup}")

        if progress_callback:
            progress_callback('dedup', 100, 100,
                            f"Deduplicated: {self.state.total_after_dedup} unique papers")

    def _phase_screen(self, progress_callback: Optional[Callable]):
        """Execute screening phase"""
        if not self.screener:
            return

        # Get papers to screen
        papers = self.state.get_papers_for_screening()

        if not papers:
            print("  No papers to screen")
            return

        if progress_callback:
            progress_callback('screen', 0, len(papers), "Starting screening...")

        # Screen papers
        screened = self.screener.screen_papers(
            papers,
            self.state.pico,
            progress_callback=lambda c, t, m: progress_callback('screen', c, t, m)
                if progress_callback else None,
            checkpoint_callback=self._save_checkpoint
        )

        # Update state
        for paper in screened:
            self.state.update_paper(paper)
            self.state.screening_completed_ids.append(paper.paper_id)

        # Update counts
        self.state.update_statistics()

        summary = self.screener.get_screening_summary(screened)
        print(f"  Screening: {summary['included']} included, {summary['excluded']} excluded, "
              f"{summary['uncertain']} uncertain")

        if progress_callback:
            progress_callback('screen', len(papers), len(papers), "Screening complete")

    def _phase_find_oa(self, progress_callback: Optional[Callable]):
        """Execute OA finding phase"""
        if not self.downloader:
            return

        # Get papers to check (included or uncertain)
        papers = self.state.get_papers_for_oa_check()

        if not papers:
            # Also check papers that weren't screened
            papers = [
                DeduplicatedPaper.from_dict(p)
                for p in self.state.deduplicated_papers.values()
                if p.get('paper_id') not in self.state.oa_checked_ids
            ]

        if not papers:
            print("  No papers to check OA")
            return

        if progress_callback:
            progress_callback('find_oa', 0, len(papers), "Finding open access versions...")

        # Find OA URLs
        checked = self.downloader.find_oa_urls(
            papers,
            progress_callback=lambda c, t, m: progress_callback('find_oa', c, t, m)
                if progress_callback else None
        )

        # Update state
        for paper in checked:
            self.state.update_paper(paper)
            self.state.oa_checked_ids.append(paper.paper_id)

        # Count OA available
        self.state.total_oa_available = sum(
            1 for p in self.state.deduplicated_papers.values()
            if p.get('oa_url')
        )

        print(f"  OA available: {self.state.total_oa_available}")

        if progress_callback:
            progress_callback('find_oa', len(papers), len(papers), "OA check complete")

    def _phase_download(self, progress_callback: Optional[Callable]):
        """Execute download phase"""
        if not self.downloader:
            return

        # Get papers to download
        papers = self.state.get_papers_for_download()

        if not papers:
            print("  No papers to download")
            return

        if progress_callback:
            progress_callback('download', 0, len(papers), "Downloading PDFs...")

        # Download
        results = self.downloader.find_and_download(
            papers,
            progress_callback=lambda c, t, m: progress_callback('download', c, t, m)
                if progress_callback else None
        )

        # Update state
        for paper in papers:
            self.state.update_paper(paper)
            self.state.download_completed_ids.append(paper.paper_id)

        # Update counts
        self.state.update_statistics()

        print(f"  Downloaded: {self.state.total_downloaded}, "
              f"Failed: {self.state.total_download_failed}")

        if progress_callback:
            progress_callback('download', len(papers), len(papers), "Download complete")

    def _save_checkpoint(self):
        """Save current state to checkpoint file"""
        self.state.save(str(self.state_file))

    def _generate_reports(self):
        """Generate final reports"""
        # CSV of all papers
        csv_path = self.output_dir / 'acquired_papers.csv'
        self._write_papers_csv(csv_path)

        # CSV of papers not found (for manual acquisition)
        not_found_path = self.output_dir / 'papers_not_found.csv'
        self._write_not_found_csv(not_found_path)

        # Summary report
        report_path = self.output_dir / 'acquisition_report.md'
        self._write_report(report_path)

    def _write_papers_csv(self, path: Path):
        """Write CSV of all acquired papers"""
        import csv

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'paper_id', 'doi', 'pmid', 'title', 'authors', 'year', 'journal',
                'screening_decision', 'screening_reason', 'oa_status', 'download_status',
                'download_path', 'sources'
            ])

            # Data
            for paper_dict in self.state.deduplicated_papers.values():
                authors = '; '.join(paper_dict.get('authors', [])[:3])
                if len(paper_dict.get('authors', [])) > 3:
                    authors += ' et al.'

                writer.writerow([
                    paper_dict.get('paper_id'),
                    paper_dict.get('canonical_doi'),
                    paper_dict.get('pmid'),
                    paper_dict.get('title'),
                    authors,
                    paper_dict.get('year'),
                    paper_dict.get('journal'),
                    paper_dict.get('screening_decision'),
                    paper_dict.get('screening_reason'),
                    paper_dict.get('oa_status'),
                    paper_dict.get('download_status'),
                    paper_dict.get('download_path'),
                    ', '.join(paper_dict.get('found_in_sources', []))
                ])

    def _write_not_found_csv(self, path: Path):
        """Write CSV of papers without OA access"""
        import csv

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['paper_id', 'doi', 'pmid', 'title', 'year', 'journal', 'reason'])

            for paper_dict in self.state.deduplicated_papers.values():
                # Only papers that were included but couldn't be downloaded
                if (paper_dict.get('screening_decision') in ['include', 'uncertain', None] and
                    paper_dict.get('download_status') != 'success'):

                    writer.writerow([
                        paper_dict.get('paper_id'),
                        paper_dict.get('canonical_doi'),
                        paper_dict.get('pmid'),
                        paper_dict.get('title'),
                        paper_dict.get('year'),
                        paper_dict.get('journal'),
                        paper_dict.get('download_error') or paper_dict.get('oa_status') or 'No OA found'
                    ])

    def _write_report(self, path: Path):
        """Write acquisition report"""
        self.state.update_statistics()

        report = f"""# Paper Acquisition Report

## Summary
- **Session ID**: {self.state.session_id}
- **Created**: {self.state.created_at}
- **Completed**: {self.state.last_updated}

## Search Results
- **Sources searched**: {', '.join(self.state.search_completed_sources)}
- **Raw results**: {self.state.total_raw_results}
- **After deduplication**: {self.state.total_after_dedup}

## Screening Results
- **Included**: {self.state.total_included}
- **Excluded**: {self.state.total_excluded}
- **Uncertain**: {self.state.total_uncertain}
- **Inclusion rate**: {self.state.total_included / self.state.total_after_dedup * 100:.1f}%

## Download Results
- **OA available**: {self.state.total_oa_available}
- **Downloaded**: {self.state.total_downloaded}
- **Failed**: {self.state.total_download_failed}
- **Download rate**: {self.state.total_downloaded / max(self.state.total_oa_available, 1) * 100:.1f}%

## Query
```
Keywords: {', '.join(self.state.query.keywords)}
Date range: {self.state.query.date_from} to {self.state.query.date_to}
```

## Output Files
- `acquired_papers.csv` - All papers with metadata and status
- `papers_not_found.csv` - Papers needing manual acquisition
- `pdfs/` - Downloaded PDF files
- `acquisition_state.json` - Full session state
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)
