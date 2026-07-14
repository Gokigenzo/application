import abc
import numpy as np
from app.infrastructure.ai.insightface_manager import InsightFaceManager
from app.core.exceptions import AIException

class FaceEmbedder(abc.ABC):
    """Abstract interface for face embedding extraction."""

    @abc.abstractmethod
    def compute_embedding(self, aligned_image: np.ndarray) -> np.ndarray:
        """Computes a 512-dimension face embedding for an aligned face image.

        Args:
            aligned_image: Aligned BGR image (112x112 pixels).

        Returns:
            512-dimensional normalized face embedding NumPy array.
        """
        pass

class ArcFaceEmbedder(FaceEmbedder):
    """ArcFace face embedder implementation using the shared InsightFace manager."""

    def __init__(self):
        self.manager = InsightFaceManager()
        if not self.manager.is_initialized:
            self.manager.initialize()

        # Find the recognition model key inside the shared models
        self.model = None
        for key, model in self.manager.app.models.items():
            if key == "recognition" or "recognize" in key or getattr(model, "taskname", "") == "recognition":
                self.model = model
                break

        if self.model is None:
            # Fallback scan
            for model in self.manager.app.models.values():
                if hasattr(model, "get") and hasattr(model, "input_size"):
                    self.model = model
                    break

        if self.model is None:
            raise AIException("ArcFace recognition model could not be found in InsightFace shared manager.")

    def compute_embedding(self, aligned_image: np.ndarray) -> np.ndarray:
        """Inferences the ArcFace model on an aligned 112x112 face image."""
        try:
            # Ensure shape is 112x112
            if aligned_image.shape[:2] != (112, 112):
                raise ValueError(f"Aligned face image must be 112x112. Got: {aligned_image.shape[:2]}")

            # Run ONNX model inference
            emb = self.model.get(aligned_image)
            
            # Extract output embedding array
            if isinstance(emb, list):
                emb = emb[0]
            emb = np.squeeze(emb)
            
            # L2 Normalization (unit length) for cosine similarity calculation
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
            return emb
        except Exception as e:
            raise AIException(f"Error during face embedding inference: {str(e)}")
