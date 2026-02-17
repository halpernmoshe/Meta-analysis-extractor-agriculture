"""
Screening Module

LLM-based screening of papers against PICO criteria.
"""

from typing import List, Dict, Optional, Callable
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.acquisition_state import DeduplicatedPaper
from prompts.screening_prompts import (
    SCREENING_SYSTEM_PROMPT,
    get_screening_prompt,
    get_batch_screening_prompt,
    parse_screening_response
)


class ScreenModule:
    """
    LLM-based screening of papers against PICO criteria.

    Supports both single-paper and batch screening for efficiency.
    """

    # Default batch size for batch screening
    DEFAULT_BATCH_SIZE = 10

    # Cost estimate per 1000 papers (Gemini Flash)
    ESTIMATED_COST_PER_1000 = 0.50

    def __init__(self,
                 llm_client,
                 batch_size: int = None,
                 checkpoint_interval: int = 50):
        """
        Initialize screening module.

        Args:
            llm_client: LLMClient instance from core.llm
            batch_size: Papers per batch (default 10)
            checkpoint_interval: Save checkpoint every N papers
        """
        self.llm = llm_client
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.checkpoint_interval = checkpoint_interval

    def screen_papers(self,
                     papers: List[DeduplicatedPaper],
                     pico: Dict,
                     progress_callback: Optional[Callable[[int, int, str], None]] = None,
                     checkpoint_callback: Optional[Callable[[], None]] = None,
                     use_batch: bool = True
                     ) -> List[DeduplicatedPaper]:
        """
        Screen papers against PICO criteria.

        Args:
            papers: Papers to screen
            pico: PICO criteria dict
            progress_callback: Progress callback
            checkpoint_callback: Called every checkpoint_interval papers
            use_batch: Use batch screening (more efficient)

        Returns:
            Papers with screening decisions populated
        """
        if not papers:
            return []

        if progress_callback:
            progress_callback(0, len(papers), "Starting screening...")

        if use_batch:
            return self._screen_batch(papers, pico, progress_callback, checkpoint_callback)
        else:
            return self._screen_single(papers, pico, progress_callback, checkpoint_callback)

    def _screen_batch(self,
                     papers: List[DeduplicatedPaper],
                     pico: Dict,
                     progress_callback: Optional[Callable],
                     checkpoint_callback: Optional[Callable]
                     ) -> List[DeduplicatedPaper]:
        """Screen papers in batches"""
        processed = 0

        for i in range(0, len(papers), self.batch_size):
            batch = papers[i:i + self.batch_size]

            if progress_callback:
                progress_callback(processed, len(papers),
                                f"Screening batch {i // self.batch_size + 1}...")

            # Build batch data
            batch_data = []
            for paper in batch:
                batch_data.append({
                    'paper_id': paper.paper_id,
                    'title': paper.title,
                    'abstract': paper.abstract
                })

            # Get batch screening prompt
            prompt = get_batch_screening_prompt(batch_data, pico, self.batch_size)

            # Call LLM
            try:
                response = self.llm.call_recon(prompt, SCREENING_SYSTEM_PROMPT)
                result = parse_screening_response(response)

                # Process batch results
                if 'batch_results' in result:
                    self._apply_batch_results(batch, result['batch_results'])
                else:
                    # Fallback: apply single result to all
                    for paper in batch:
                        self._apply_screening_result(paper, result)

            except Exception as e:
                # On error, mark all as uncertain
                for paper in batch:
                    paper.screening_decision = 'uncertain'
                    paper.screening_confidence = 'low'
                    paper.screening_reason = f'Screening error: {str(e)[:100]}'
                    paper.screening_timestamp = datetime.now().isoformat()

            processed += len(batch)

            # Checkpoint
            if checkpoint_callback and processed % self.checkpoint_interval < self.batch_size:
                checkpoint_callback()

        if progress_callback:
            progress_callback(len(papers), len(papers), "Screening complete")

        return papers

    def _screen_single(self,
                      papers: List[DeduplicatedPaper],
                      pico: Dict,
                      progress_callback: Optional[Callable],
                      checkpoint_callback: Optional[Callable]
                      ) -> List[DeduplicatedPaper]:
        """Screen papers one at a time"""
        for i, paper in enumerate(papers):
            if progress_callback:
                progress_callback(i, len(papers), f"Screening paper {i+1}/{len(papers)}...")

            # Get screening prompt
            prompt = get_screening_prompt(paper.title, paper.abstract, pico)

            # Call LLM
            try:
                response = self.llm.call_recon(prompt, SCREENING_SYSTEM_PROMPT)
                result = parse_screening_response(response)
                self._apply_screening_result(paper, result)

            except Exception as e:
                paper.screening_decision = 'uncertain'
                paper.screening_confidence = 'low'
                paper.screening_reason = f'Screening error: {str(e)[:100]}'
                paper.screening_timestamp = datetime.now().isoformat()

            # Checkpoint
            if checkpoint_callback and (i + 1) % self.checkpoint_interval == 0:
                checkpoint_callback()

        if progress_callback:
            progress_callback(len(papers), len(papers), "Screening complete")

        return papers

    def _apply_batch_results(self, papers: List[DeduplicatedPaper], results: List[Dict]):
        """Apply batch results to papers"""
        # Create lookup by paper_id
        result_lookup = {r.get('paper_id'): r for r in results}

        for paper in papers:
            result = result_lookup.get(paper.paper_id)
            if result:
                self._apply_screening_result(paper, result)
            else:
                # No result for this paper
                paper.screening_decision = 'uncertain'
                paper.screening_confidence = 'low'
                paper.screening_reason = 'No screening result returned'
                paper.screening_timestamp = datetime.now().isoformat()

    def _apply_screening_result(self, paper: DeduplicatedPaper, result: Dict):
        """Apply screening result to a paper"""
        paper.screening_decision = result.get('decision', 'uncertain')
        paper.screening_confidence = result.get('confidence', 'low')
        paper.screening_reason = result.get('reason', '')
        paper.screening_timestamp = datetime.now().isoformat()

        # PICO matching
        paper.population_match = result.get('population_match')
        paper.intervention_match = result.get('intervention_match')
        paper.outcome_potential = result.get('outcome_potential')
        paper.quantitative_data = result.get('quantitative_data')
        paper.exclusion_category = result.get('exclusion_category')

    def estimate_cost(self, num_papers: int) -> float:
        """
        Estimate screening cost.

        Args:
            num_papers: Number of papers to screen

        Returns:
            Estimated cost in USD
        """
        return (num_papers / 1000) * self.ESTIMATED_COST_PER_1000

    def get_screening_summary(self, papers: List[DeduplicatedPaper]) -> Dict:
        """
        Get summary of screening results.

        Args:
            papers: Screened papers

        Returns:
            Summary dict
        """
        total = len(papers)
        included = sum(1 for p in papers if p.screening_decision == 'include')
        excluded = sum(1 for p in papers if p.screening_decision == 'exclude')
        uncertain = sum(1 for p in papers if p.screening_decision == 'uncertain')
        not_screened = sum(1 for p in papers if p.screening_decision is None)

        # Exclusion reasons
        exclusion_reasons = {}
        for p in papers:
            if p.exclusion_category:
                exclusion_reasons[p.exclusion_category] = \
                    exclusion_reasons.get(p.exclusion_category, 0) + 1

        return {
            'total': total,
            'included': included,
            'excluded': excluded,
            'uncertain': uncertain,
            'not_screened': not_screened,
            'inclusion_rate': included / total if total > 0 else 0,
            'exclusion_reasons': exclusion_reasons
        }
