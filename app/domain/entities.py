from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid
import numpy as np

@dataclass
class Student:
    """Represents a student in the system."""
    student_id: str
    name: str
    email: Optional[str] = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class FaceEmbedding:
    """Represents a student's face embedding vector (512 dimensions)."""
    student_id: uuid.UUID
    embedding: np.ndarray
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        # Validate embedding shape
        if not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding, dtype=np.float32)
        if self.embedding.shape != (512,):
            raise ValueError(f"Face embedding must be a 512-dimensional vector. Got shape: {self.embedding.shape}")

@dataclass
class Session:
    """Represents an active or historical class attendance session."""
    name: str
    teacher_name: str
    start_time: datetime
    end_time: datetime
    is_active: bool = True
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class AttendanceRecord:
    """Represents a single attendance event where a student is marked present in a session."""
    session_id: uuid.UUID
    student_id: uuid.UUID
    status: str = "PRESENT"
    marked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: uuid.UUID = field(default_factory=uuid.uuid4)
