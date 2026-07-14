import threading
import streamlit as st
from sqlalchemy import text
from app.core.database import SessionLocal, engine, Base
from app.infrastructure.database.repositories import (
    SQLAlchemyStudentRepository,
    SQLAlchemyFaceEmbeddingRepository,
    SQLAlchemySessionRepository,
    SQLAlchemyAttendanceRepository
)
from app.infrastructure.ai.detection import SCRFDDetector
from app.infrastructure.ai.embedding import ArcFaceEmbedder
from app.infrastructure.ai.tracking import ByteTracker
from app.services.student_service import StudentService
from app.services.session_service import SessionService
from app.services.attendance_service import AttendanceService
from app.services.ai_service import AIService
from app.services.monitor_service import MonitorService
from app.utils.cache import FaceEmbeddingCache

_init_lock = threading.Lock()

def init_db() -> None:
    """Ensures pgvector extension is enabled and database schemas exist in Supabase."""
    # Base.metadata.create_all can raise errors if pgvector is not enabled first
    # So we execute CREATE EXTENSION if allowed.
    with SessionLocal() as session:
        try:
            session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            session.commit()
        except Exception as e:
            # Fallback if user permissions restrict CREATE EXTENSION (pgvector is pre-enabled on Supabase by default)
            session.rollback()
            
    # Create all tables (students, face_embeddings, sessions, attendance_records)
    Base.metadata.create_all(bind=engine)

def get_services() -> dict:
    """Gets or initializes the service layer container (Singleton per Streamlit session)."""
    with _init_lock:
        if "services" not in st.session_state:
            # 1. Prepare database structures
            init_db()

            # 2. Instantiate SQL session
            db = SessionLocal()

            # 3. Instantiate concrete repositories
            student_repo = SQLAlchemyStudentRepository(db)
            emb_repo = SQLAlchemyFaceEmbeddingRepository(db)
            sess_repo = SQLAlchemySessionRepository(db)
            att_repo = SQLAlchemyAttendanceRepository(db)

            # 4. Instantiate AI processors
            detector = SCRFDDetector(threshold=0.5)
            embedder = ArcFaceEmbedder()
            tracker = ByteTracker(track_thresh=0.5, match_thresh=0.8, track_buffer=30)

            # 5. Load/Initialize in-memory vector cache
            cache = FaceEmbeddingCache()
            cache.load_from_db(db)

            # 6. Instantiate application services
            student_service = StudentService(student_repo, emb_repo, detector, embedder)
            session_service = SessionService(sess_repo)
            attendance_service = AttendanceService(att_repo, sess_repo, student_repo)
            ai_service = AIService(detector, embedder, tracker)
            monitor_service = MonitorService()

            st.session_state.services = {
                "db": db,
                "student_service": student_service,
                "session_service": session_service,
                "attendance_service": attendance_service,
                "ai_service": ai_service,
                "monitor_service": monitor_service,
                "cache": cache
            }

    return st.session_state.services
