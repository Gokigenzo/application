import os
import time
import threading
from collections import deque
from typing import List, Dict, Any, Deque
import psutil
from loguru import logger

class MonitorService:
    """Singleton performance monitor for compiling telemetry and metrics.
    
    Tracks CPU, RAM, disk usage, model inference speeds (detection, 
    embedding), database lookup times, frame rates (FPS), and retrieves 
    recent log files.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MonitorService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.lock = threading.Lock()
        
        # Deque buffers to store rolling telemetry averages
        self.detection_latencies: Deque[float] = deque(maxlen=100)
        self.embedding_latencies: Deque[float] = deque(maxlen=100)
        self.lookup_latencies: Deque[float] = deque(maxlen=100)
        self.fps_readings: Deque[float] = deque(maxlen=100)
        
        self._initialized = True

    def log_detection_latency(self, val_sec: float) -> None:
        """Records a face detection runtime entry in seconds."""
        with self.lock:
            self.detection_latencies.append(val_sec)

    def log_embedding_latency(self, val_sec: float) -> None:
        """Records a face embedding runtime entry in seconds."""
        with self.lock:
            self.embedding_latencies.append(val_sec)

    def log_lookup_latency(self, val_sec: float) -> None:
        """Records a vector search lookup runtime entry in seconds."""
        with self.lock:
            self.lookup_latencies.append(val_sec)

    def log_fps(self, fps: float) -> None:
        """Records a camera feed FPS reading."""
        with self.lock:
            self.fps_readings.append(fps)

    def get_system_metrics(self) -> Dict[str, Any]:
        """Queries CPU usage, RAM utilization, and disk space percentage."""
        try:
            # interval=None gets non-blocking CPU usage since last call
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            return {
                "cpu_usage_pct": float(cpu_percent),
                "memory_used_mb": float(memory.used / (1024 * 1024)),
                "memory_total_mb": float(memory.total / (1024 * 1024)),
                "memory_usage_pct": float(memory.percent),
                "disk_usage_pct": float(disk.percent),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"Failed to query system performance metrics: {str(e)}")
            return {
                "cpu_usage_pct": 0.0,
                "memory_used_mb": 0.0,
                "memory_total_mb": 0.0,
                "memory_usage_pct": 0.0,
                "disk_usage_pct": 0.0,
                "timestamp": time.time()
            }

    def get_ai_metrics(self) -> Dict[str, Any]:
        """Computes rolling averages for AI processing latency and frame rate."""
        with self.lock:
            avg_det = sum(self.detection_latencies) / len(self.detection_latencies) if self.detection_latencies else 0.0
            avg_emb = sum(self.embedding_latencies) / len(self.embedding_latencies) if self.embedding_latencies else 0.0
            avg_lookup = sum(self.lookup_latencies) / len(self.lookup_latencies) if self.lookup_latencies else 0.0
            avg_fps = sum(self.fps_readings) / len(self.fps_readings) if self.fps_readings else 0.0
            
            return {
                "avg_detection_ms": avg_det * 1000.0,   # Convert to milliseconds for readouts
                "avg_embedding_ms": avg_emb * 1000.0,
                "avg_lookup_ms": avg_lookup * 1000.0,
                "avg_fps": avg_fps,
                "history_length": len(self.fps_readings)
            }

    def get_recent_logs(self, max_lines: int = 150) -> List[str]:
        """Reads the end of the Loguru log file for viewing in the UI admin dashboard."""
        log_file = "logs/app.log"
        if not os.path.exists(log_file):
            return ["Log file has not been initialized yet."]
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                return [line.strip() for line in lines[-max_lines:]]
        except Exception as e:
            return [f"Failed to read application logs: {str(e)}"]
