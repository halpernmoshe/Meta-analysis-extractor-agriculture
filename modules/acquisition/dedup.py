"""
Deduplication Module

Cross-source deduplication with metadata merging.
"""

from typing import List, Dict, Optional, Callable
from difflib import SequenceMatcher
import re

from core.acquisition_state import SearchResult, DeduplicatedPaper


class DeduplicationModule:
    """
    Cross-source deduplication with metadata merging.

    Strategy:
    1. Exact DOI match (highest confidence)
    2. Exact PMID match (high confidence)
    3. Title similarity + author overlap (fuzzy matching)
    """

    # Source priority for metadata selection (higher = better)
    SOURCE_PRIORITY = {
        'semantic_scholar': 1,
        'openalex': 2,
        'pubmed': 3,
        'google_scholar': 0
    }

    def __init__(self,
                 title_threshold: float = 0.9,
                 author_threshold: float = 0.5):
        """
        Initialize deduplication module.

        Args:
            title_threshold: Minimum title similarity for fuzzy matching (0-1)
            author_threshold: Minimum author overlap for fuzzy matching (0-1)
        """
        self.title_threshold = title_threshold
        self.author_threshold = author_threshold

    def deduplicate(self,
                   results_by_source: Dict[str, List[SearchResult]],
                   progress_callback: Optional[Callable[[int, int, str], None]] = None
                   ) -> List[DeduplicatedPaper]:
        """
        Deduplicate results from multiple sources.

        Args:
            results_by_source: Dict mapping source name to list of SearchResults
            progress_callback: Progress callback

        Returns:
            List of DeduplicatedPaper with merged metadata
        """
        if progress_callback:
            progress_callback(0, 100, "Starting deduplication...")

        # Flatten all results
        all_results: List[SearchResult] = []
        for source, results in results_by_source.items():
            all_results.extend(results)

        if not all_results:
            return []

        if progress_callback:
            progress_callback(10, 100, f"Processing {len(all_results)} total results...")

        # Build lookup indices
        doi_index: Dict[str, List[SearchResult]] = {}
        pmid_index: Dict[str, List[SearchResult]] = {}

        for result in all_results:
            if result.doi:
                doi_clean = self._normalize_doi(result.doi)
                if doi_clean:
                    doi_index.setdefault(doi_clean, []).append(result)

            if result.pmid:
                pmid_index.setdefault(result.pmid, []).append(result)

        if progress_callback:
            progress_callback(30, 100, "Clustering duplicates...")

        # Cluster duplicates
        clusters = self._cluster_duplicates(all_results, doi_index, pmid_index)

        if progress_callback:
            progress_callback(70, 100, f"Merging {len(clusters)} unique papers...")

        # Merge clusters into DeduplicatedPaper
        deduplicated = []
        for i, cluster in enumerate(clusters):
            paper = self._merge_cluster(cluster, i)
            deduplicated.append(paper)

        if progress_callback:
            progress_callback(100, 100,
                            f"Deduplication complete: {len(all_results)} -> {len(deduplicated)}")

        return deduplicated

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI for comparison"""
        if not doi:
            return ""
        doi = doi.lower().strip()
        # Remove common prefixes
        doi = doi.replace('https://doi.org/', '')
        doi = doi.replace('http://doi.org/', '')
        doi = doi.replace('http://dx.doi.org/', '')
        doi = doi.replace('doi:', '')
        return doi

    def _normalize_title(self, title: str) -> str:
        """Normalize title for comparison"""
        if not title:
            return ""
        # Lowercase, remove punctuation, normalize whitespace
        title = title.lower()
        title = re.sub(r'[^\w\s]', ' ', title)
        title = ' '.join(title.split())
        return title

    def _title_similarity(self, t1: str, t2: str) -> float:
        """Calculate title similarity ratio"""
        t1_norm = self._normalize_title(t1)
        t2_norm = self._normalize_title(t2)

        if not t1_norm or not t2_norm:
            return 0.0

        return SequenceMatcher(None, t1_norm, t2_norm).ratio()

    def _author_overlap(self, authors1: List[str], authors2: List[str]) -> float:
        """Calculate author overlap ratio (Jaccard similarity of last names)"""
        if not authors1 or not authors2:
            return 0.0

        def get_last_name(name: str) -> str:
            """Extract last name from full name"""
            if not name:
                return ""
            parts = name.strip().split()
            return parts[-1].lower() if parts else ""

        names1 = {get_last_name(a) for a in authors1 if a}
        names2 = {get_last_name(a) for a in authors2 if a}

        # Remove empty strings
        names1.discard("")
        names2.discard("")

        if not names1 or not names2:
            return 0.0

        intersection = names1 & names2
        union = names1 | names2

        return len(intersection) / len(union) if union else 0.0

    def _cluster_duplicates(self,
                           results: List[SearchResult],
                           doi_index: Dict[str, List[SearchResult]],
                           pmid_index: Dict[str, List[SearchResult]]
                           ) -> List[List[SearchResult]]:
        """
        Group duplicate results into clusters using Union-Find.
        """
        # Union-Find data structure
        parent = {id(r): id(r) for r in results}
        rank = {id(r): 0 for r in results}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Union by rank
                if rank[px] < rank[py]:
                    px, py = py, px
                parent[py] = px
                if rank[px] == rank[py]:
                    rank[px] += 1

        # Merge by DOI (exact match)
        for doi, group in doi_index.items():
            if len(group) > 1:
                first = group[0]
                for r in group[1:]:
                    union(id(first), id(r))

        # Merge by PMID (exact match)
        for pmid, group in pmid_index.items():
            if len(group) > 1:
                first = group[0]
                for r in group[1:]:
                    union(id(first), id(r))

        # Fuzzy title matching for results without DOI/PMID
        # This is O(n^2) so we limit it to unmatched results
        unmatched = [r for r in results if not r.doi and not r.pmid]

        if len(unmatched) <= 1000:  # Only do fuzzy matching for reasonable sizes
            for i, r1 in enumerate(unmatched):
                for r2 in unmatched[i+1:]:
                    # Skip if already in same cluster
                    if find(id(r1)) == find(id(r2)):
                        continue

                    # Check title similarity
                    title_sim = self._title_similarity(r1.title, r2.title)
                    if title_sim >= self.title_threshold:
                        # Also check author overlap
                        author_sim = self._author_overlap(r1.authors, r2.authors)
                        if author_sim >= self.author_threshold:
                            union(id(r1), id(r2))

        # Build clusters from union-find
        clusters_dict: Dict[int, List[SearchResult]] = {}
        for r in results:
            root = find(id(r))
            clusters_dict.setdefault(root, []).append(r)

        return list(clusters_dict.values())

    def _merge_cluster(self, cluster: List[SearchResult], index: int) -> DeduplicatedPaper:
        """
        Merge a cluster of duplicates into single DeduplicatedPaper.

        Prioritizes metadata from higher-quality sources.
        """
        # Sort by source priority (higher priority first)
        cluster.sort(key=lambda r: self.SOURCE_PRIORITY.get(r.source, 0), reverse=True)

        # Collect all identifiers
        dois = {self._normalize_doi(r.doi) for r in cluster if r.doi}
        dois.discard("")
        pmids = {r.pmid for r in cluster if r.pmid}
        pmcids = {r.pmcid for r in cluster if r.pmcid}

        # Get best metadata (from highest priority source)
        best = cluster[0]

        # Get best abstract (longest non-empty)
        best_abstract = None
        for r in cluster:
            if r.abstract:
                if not best_abstract or len(r.abstract) > len(best_abstract):
                    best_abstract = r.abstract

        # Get most authors
        best_authors = best.authors
        for r in cluster:
            if len(r.authors) > len(best_authors):
                best_authors = r.authors

        # Get highest citation count
        best_citations = None
        for r in cluster:
            if r.citation_count is not None:
                if best_citations is None or r.citation_count > best_citations:
                    best_citations = r.citation_count

        # Collect all sources
        sources = list(set(r.source for r in cluster))

        # Store source results as dicts
        source_results = {r.source: r.to_dict() for r in cluster}

        # Generate paper ID
        paper_id = f"P{index+1:05d}"

        return DeduplicatedPaper(
            paper_id=paper_id,
            canonical_doi=next(iter(dois), None),
            pmid=next(iter(pmids), None),
            pmcid=next(iter(pmcids), None),
            title=best.title or "",
            authors=best_authors,
            year=best.year,
            journal=best.journal,
            abstract=best_abstract,
            found_in_sources=sources,
            source_results=source_results
        )

    def merge_additional_results(self,
                                existing_papers: List[DeduplicatedPaper],
                                new_results: List[SearchResult]
                                ) -> List[DeduplicatedPaper]:
        """
        Merge new results into existing deduplicated papers.

        Useful for incremental search/updates.

        Args:
            existing_papers: Previously deduplicated papers
            new_results: New search results to merge

        Returns:
            Updated list with new papers added and duplicates merged
        """
        # Build index of existing papers
        doi_to_paper: Dict[str, DeduplicatedPaper] = {}
        pmid_to_paper: Dict[str, DeduplicatedPaper] = {}

        for paper in existing_papers:
            if paper.canonical_doi:
                doi_to_paper[self._normalize_doi(paper.canonical_doi)] = paper
            if paper.pmid:
                pmid_to_paper[paper.pmid] = paper

        # Process new results
        new_papers = []
        next_id = len(existing_papers) + 1

        for result in new_results:
            # Check if matches existing paper
            matched_paper = None

            if result.doi:
                doi_norm = self._normalize_doi(result.doi)
                if doi_norm in doi_to_paper:
                    matched_paper = doi_to_paper[doi_norm]

            if not matched_paper and result.pmid:
                if result.pmid in pmid_to_paper:
                    matched_paper = pmid_to_paper[result.pmid]

            if matched_paper:
                # Update existing paper with new source
                if result.source not in matched_paper.found_in_sources:
                    matched_paper.found_in_sources.append(result.source)
                    matched_paper.source_results[result.source] = result.to_dict()
            else:
                # Create new paper
                new_paper = DeduplicatedPaper(
                    paper_id=f"P{next_id:05d}",
                    canonical_doi=result.doi,
                    pmid=result.pmid,
                    pmcid=result.pmcid,
                    title=result.title,
                    authors=result.authors,
                    year=result.year,
                    journal=result.journal,
                    abstract=result.abstract,
                    found_in_sources=[result.source],
                    source_results={result.source: result.to_dict()}
                )
                new_papers.append(new_paper)

                # Update indices
                if result.doi:
                    doi_to_paper[self._normalize_doi(result.doi)] = new_paper
                if result.pmid:
                    pmid_to_paper[result.pmid] = new_paper

                next_id += 1

        return existing_papers + new_papers
