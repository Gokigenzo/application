import threading
from insightface.app import FaceAnalysis
from app.core.configs import get_settings
from app.core.exceptions import AIException

class InsightFaceManager:
    """Thread-safe Singleton manager for shared InsightFace models."""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(InsightFaceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        # Already initialized by __new__ structure, but this is standard PEP8
        pass

    def initialize(self) -> None:
        """Initializes the FaceAnalysis model zoo on the CPU."""
        if self._initialized:
            return
        
        settings = get_settings()
        try:
            # We initialize both detection and recognition modules in a single pass
            self.app = FaceAnalysis(
                name=settings.FACE_RECOGNITION_MODEL_NAME,
                allowed_modules=["detection", "recognition"],
                root="~/.insightface"
            )
            # ctx_id < 0 runs model inference on CPU
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
            self._initialized = True
        except Exception as e:
            raise AIException(f"Failed to initialize shared InsightFace FaceAnalysis models: {str(e)}")

    @property
    def is_initialized(self) -> bool:
        return self._initialized
