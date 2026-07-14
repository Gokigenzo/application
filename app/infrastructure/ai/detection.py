import abc
from dataclasses import dataclass
import numpy as np
from typing import List
from app.infrastructure.ai.insightface_manager import InsightFaceManager
from app.core.exceptions import AIException

@dataclass
class DetectionResult:
    """Standardized face detection output."""
    bbox: np.ndarray        # [x1, y1, x2, y2] bounding box
    score: float            # Detection confidence score
    keypoints: np.ndarray   # [5, 2] landmarks (eyes, nose, mouth corners)

class FaceDetector(abc.ABC):
    """Abstract interface for face detection."""

    @abc.abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Detects faces in an image.

        Args:
            image: BGR numpy array.

        Returns:
            List of DetectionResult.
        """
        pass

class SCRFDDetector(FaceDetector):
    """SCRFD detector implementation using shared InsightFace manager."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.manager = InsightFaceManager()
        if not self.manager.is_initialized:
            self.manager.initialize()

    def detect(self, image: np.ndarray) -> List[DetectionResult]:
        """Inferences the SCRFD model using the shared model manager."""
        try:
            # We call the shared FaceAnalysis app
            faces = self.manager.app.get(image)
            results = []
            for face in faces:
                if face.det_score >= self.threshold:
                    results.append(
                        DetectionResult(
                            bbox=face.bbox,
                            score=float(face.det_score),
                            keypoints=face.kps
                        )
                    )
            return results
        except Exception as e:
            raise AIException(f"Error during face detection inference: {str(e)}")
