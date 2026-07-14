import abc
import uuid
from typing import List, Optional
from app.domain.entities import Student, FaceEmbedding, Session, AttendanceRecord

class StudentRepository(abc.ABC):
    """Abstract interface for student data persistence."""

    @abc.abstractmethod
    def add(self, student: Student) -> Student:
        """Adds a new student to storage."""
        pass

    @abc.abstractmethod
    def get_by_id(self, id_: uuid.UUID) -> Optional[Student]:
        """Retrieves a student by their unique internal UUID."""
        pass

    @abc.abstractmethod
    def get_by_student_id(self, student_id: str) -> Optional[Student]:
        """Retrieves a student by their registration/student code."""
        pass

    @abc.abstractmethod
    def list_all(self) -> List[Student]:
        """Lists all registered students."""
        pass

    @abc.abstractmethod
    def delete(self, id_: uuid.UUID) -> bool:
        """Deletes a student record by UUID."""
        pass


class FaceEmbeddingRepository(abc.ABC):
    """Abstract interface for face embedding vector persistence."""

    @abc.abstractmethod
    def add(self, embedding: FaceEmbedding) -> FaceEmbedding:
        """Adds a face embedding vector linked to a student."""
        pass

    @abc.abstractmethod
    def get_by_student_id(self, student_id: uuid.UUID) -> List[FaceEmbedding]:
        """Retrieves all embeddings stored for a given student UUID."""
        pass

    @abc.abstractmethod
    def list_all(self) -> List[FaceEmbedding]:
        """Lists all registered embeddings in the database."""
        pass

    @abc.abstractmethod
    def delete(self, id_: uuid.UUID) -> bool:
        """Deletes a specific face embedding vector."""
        pass


class SessionRepository(abc.ABC):
    """Abstract interface for session management persistence."""

    @abc.abstractmethod
    def add(self, session: Session) -> Session:
        """Adds a new class session."""
        pass

    @abc.abstractmethod
    def get_by_id(self, id_: uuid.UUID) -> Optional[Session]:
        """Retrieves a class session by its UUID."""
        pass

    @abc.abstractmethod
    def get_active_sessions(self) -> List[Session]:
        """Retrieves all currently open/active sessions."""
        pass

    @abc.abstractmethod
    def list_all(self) -> List[Session]:
        """Lists all class sessions (active and past)."""
        pass

    @abc.abstractmethod
    def update(self, session: Session) -> Session:
        """Updates session attributes (e.g. status, end_time)."""
        pass


class AttendanceRepository(abc.ABC):
    """Abstract interface for attendance record persistence."""

    @abc.abstractmethod
    def add(self, record: AttendanceRecord) -> AttendanceRecord:
        """Saves a new attendance record."""
        pass

    @abc.abstractmethod
    def get_record(self, session_id: uuid.UUID, student_id: uuid.UUID) -> Optional[AttendanceRecord]:
        """Checks if a student already has an attendance record for a session."""
        pass

    @abc.abstractmethod
    def get_by_session(self, session_id: uuid.UUID) -> List[AttendanceRecord]:
        """Retrieves all attendance records for a specific session."""
        pass

    @abc.abstractmethod
    def get_by_student(self, student_id: uuid.UUID) -> List[AttendanceRecord]:
        """Retrieves all attendance records for a specific student."""
        pass

    @abc.abstractmethod
    def list_all(self) -> List[AttendanceRecord]:
        """Lists all attendance records."""
        pass
