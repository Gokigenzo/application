import uuid
from sqlalchemy import Column, String, DateTime, ForeignKey, Boolean, text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.core.database import Base

class StudentDB(Base):
    """SQLAlchemy model for students."""
    __tablename__ = "students"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))


class FaceEmbeddingDB(Base):
    """SQLAlchemy model for face embeddings (pgvector)."""
    __tablename__ = "face_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    embedding = Column(Vector(512), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))


class SessionDB(Base):
    """SQLAlchemy model for class sessions."""
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False)
    teacher_name = Column(String(100), nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=text("now()"))


class AttendanceRecordDB(Base):
    """SQLAlchemy model for student attendance records."""
    __tablename__ = "attendance_records"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(UUID(as_uuid=True), ForeignKey("students.id", ondelete="CASCADE"), nullable=False, index=True)
    marked_at = Column(DateTime(timezone=True), server_default=text("now()"))
    status = Column(String(20), nullable=False, default="PRESENT")

    __table_args__ = (
        UniqueConstraint("session_id", "student_id", name="uq_session_student_attendance"),
    )
