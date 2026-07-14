import numpy as np
from typing import List, Tuple, Any
from app.domain.strategies import MatchingStrategy

class CosineMatchingStrategy(MatchingStrategy):
    """Cosine similarity implementation of face matching strategy.
    
    Assumes that the query embedding and candidate embeddings are 
    L2-normalized unit vectors, in which case Cosine Similarity is
    exactly the Dot Product.
    """

    def find_matches(
        self,
        query_embedding: np.ndarray,
        candidates: List[Tuple[Any, np.ndarray]],
        threshold: float
    ) -> List[Tuple[Any, float]]:
        """Finds matching candidate embeddings that exceed similarity threshold.

        Args:
            query_embedding: Target 512-dimension face embedding.
            candidates: List of tuples (identifier, embedding_vector).
            threshold: Minimum cosine similarity score [0.0 - 1.0].

        Returns:
            List of tuples (identifier, score) ordered by score descending.
        """
        if not candidates:
            return []

        # Extract identifiers and embeddings list
        identifiers = [c[0] for c in candidates]
        embeddings = [c[1] for c in candidates]

        # Stack candidate vectors into a single 2D NumPy array: Shape (N, 512)
        vectors_arr = np.vstack(embeddings)

        # Vectorized dot product matching
        # query_embedding shape: (512,), vectors_arr shape: (N, 512) -> results in shape (N,)
        similarities = np.dot(vectors_arr, query_embedding)

        # Filter candidates above the similarity threshold
        matches = []
        for idx, score in enumerate(similarities):
            # Float conversion makes it JSON serializable if needed
            val = float(score)
            if val >= threshold:
                matches.append((identifiers[idx], val))

        # Sort matches by score descending
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches
