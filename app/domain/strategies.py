import abc
import numpy as np
from typing import List, Tuple, Any

class MatchingStrategy(abc.ABC):
    """Abstract interface representing a face matching algorithm."""

    @abc.abstractmethod
    def find_matches(
        self,
        query_embedding: np.ndarray,
        candidates: List[Tuple[Any, np.ndarray]],
        threshold: float
    ) -> List[Tuple[Any, float]]:
        """Compares a query face embedding vector against a list of candidates.

        Args:
            query_embedding: A 512-dimension face embedding array.
            candidates: A list of tuples, each containing (identifier, embedding_vector).
            threshold: Minimum score threshold for a valid match.

        Returns:
            A list of tuples (identifier, score) ordered by similarity descending,
            containing only matches that meet or exceed the threshold.
        """
        pass
