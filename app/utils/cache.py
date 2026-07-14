import threading
import uuid
from typing import List, Tuple, Optional
import numpy as np
from app.infrastructure.ai.matching import CosineMatchingStrategy

class FaceEmbeddingCache:
    """Thread-safe in-memory cache for fast face embedding lookups.
    
    Loads all student embeddings from PostgreSQL into RAM as a matrix
    so that real-time video frames can be matched via NumPy vectorization
    without hit overhead on the database.
    """
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceEmbeddingCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.lock = threading.Lock()
        # Stacked list of candidate face embedding vectors (512-dimension)
        self.embeddings: List[np.ndarray] = []
        # Parallel list of metadata mapping to embeddings: (student_uuid, student_name, student_code)
        self.metadata: List[Tuple[uuid.UUID, str, str]] = []
        self.matcher = CosineMatchingStrategy()
        self._initialized = True

    def clear(self) -> None:
        """Clears all cached embeddings and metadata."""
        with self.lock:
            self.embeddings.clear()
            self.metadata.clear()

    def load_from_db(self, db_session) -> None:
        """Reloads the entire embedding cache from the database.

        Args:
            db_session: An active SQLAlchemy Session.
        """
        from app.infrastructure.database.models import FaceEmbeddingDB, StudentDB
        
        with self.lock:
            self.embeddings.clear()
            self.metadata.clear()
            
            # Join face_embeddings with students to grab names and IDs
            results = db_session.query(FaceEmbeddingDB, StudentDB).join(
                StudentDB, FaceEmbeddingDB.student_id == StudentDB.id
            ).all()
            
            for emb_db, student_db in results:
                emb_vector = np.array(emb_db.embedding, dtype=np.float32)
                # Ensure it is normalized to unit length
                norm = np.linalg.norm(emb_vector)
                if norm > 0:
                    emb_vector = emb_vector / norm
                
                self.embeddings.append(emb_vector)
                self.metadata.append((student_db.id, student_db.name, student_db.student_id))

    def add(self, student_uuid: uuid.UUID, name: str, student_id: str, embedding: np.ndarray) -> None:
        """Adds a single face embedding to the cache.
        
        Allows incremental cache updates without reloading the entire DB.
        """
        emb_vector = np.array(embedding, dtype=np.float32).copy()
        norm = np.linalg.norm(emb_vector)
        if norm > 0:
            emb_vector = emb_vector / norm

        with self.lock:
            self.embeddings.append(emb_vector)
            self.metadata.append((student_uuid, name, student_id))

    def remove(self, student_uuid: uuid.UUID) -> None:
        """Removes all embeddings belonging to a student from the cache."""
        with self.lock:
            # Filter and keep only indices that do not match the student uuid
            indices_to_keep = [i for i, meta in enumerate(self.metadata) if meta[0] != student_uuid]
            self.embeddings = [self.embeddings[i] for i in indices_to_keep]
            self.metadata = [self.metadata[i] for i in indices_to_keep]

    def search(
        self,
        query_embedding: np.ndarray,
        threshold: float = 0.60
    ) -> Optional[Tuple[uuid.UUID, str, str, float]]:
        """Searches the cache for the best matching student face.

        Args:
            query_embedding: The face embedding vector from the frame.
            threshold: Cosine similarity threshold for a valid match.

        Returns:
            Tuple of (student_uuid, name, student_id, score) if a match
            exists above the threshold, otherwise None.
        """
        # Take a snapshot of candidates under lock to compute similarity safely
        with self.lock:
            if not self.embeddings:
                return None
            
            # Format candidates: list of (index, embedding_vector)
            candidates = [(i, emb) for i, emb in enumerate(self.embeddings)]

        # Run vector similarity matching (released lock to keep CPU computation non-blocking for cache writes)
        matches = self.matcher.find_matches(query_embedding, candidates, threshold)
        if not matches:
            return None

        # Best match is the first element
        best_idx, score = matches[0]

        # Re-lock to extract matching metadata safely
        with self.lock:
            if best_idx < len(self.metadata):
                student_uuid, name, student_id = self.metadata[best_idx]
                return student_uuid, name, student_id, score
            
        return None

    @property
    def size(self) -> int:
        with self.lock:
            return len(self.embeddings)
