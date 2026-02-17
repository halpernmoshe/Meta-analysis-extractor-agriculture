"""
Base Source Adapter

Abstract base class for all academic database adapters.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable
import time

from core.acquisition_state import SearchQuery, SearchResult


class BaseSourceAdapter(ABC):
    """
    Abstract base class for academic database adapters.

    All source adapters must implement the search() method.
    """

    # Source identifier (override in subclasses)
    SOURCE_NAME: str = "base"

    # Rate limiting (requests per second)
    DEFAULT_RATE_LIMIT: float = 1.0

    def __init__(self, rate_limit: Optional[float] = None):
        """
        Initialize adapter.

        Args:
            rate_limit: Requests per second (default varies by source)
        """
        self.rate_limit = rate_limit or self.DEFAULT_RATE_LIMIT
        self._last_request_time: float = 0

    def _rate_limit_wait(self):
        """Wait to respect rate limit"""
        if self.rate_limit > 0:
            min_interval = 1.0 / self.rate_limit
            elapsed = time.time() - self._last_request_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    @abstractmethod
    def search(self,
               query: SearchQuery,
               max_results: int = 1000,
               progress_callback: Optional[Callable[[int, int, str], None]] = None
               ) -> List[SearchResult]:
        """
        Search the database and return results.

        Args:
            query: Search query specification
            max_results: Maximum number of results to return
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of SearchResult objects
        """
        pass

    def _build_query_string(self, query: SearchQuery) -> str:
        """
        Build a query string from SearchQuery.

        Override in subclasses for source-specific query syntax.

        Args:
            query: SearchQuery object

        Returns:
            Query string for the API
        """
        if query.query_string:
            return query.query_string

        # Default: join all keywords with spaces
        terms = []
        terms.extend(query.keywords)
        terms.extend(query.title_keywords)
        terms.extend(query.abstract_keywords)

        return ' '.join(terms)

    @staticmethod
    def _batch(items: List, batch_size: int):
        """Yield batches of items"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def test_connection(self) -> bool:
        """
        Test if the API is accessible.

        Returns:
            True if connection successful
        """
        try:
            # Minimal test query
            test_query = SearchQuery(keywords=["test"])
            results = self.search(test_query, max_results=1)
            return True
        except Exception:
            return False
