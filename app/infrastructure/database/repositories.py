from sqlalchemy.orm import Session as SqlSession
from typing import List, Optional
import uuid
import numpy as np

from app.domain.repositories import (
    StudentRepository, FaceEmbeddingRepository, SessionRepository, AttendanceRepository
)
from app.domain.entities import Student, FaceEmbedding, Session, AttendanceRecord
from app.infrastructure.database.models import StudentDB, FaceEmbeddingDB, SessionDB, AttendanceRecordDB

class SQLAlchemyStudentRepository(StudentRepository):
    """SQLAlchemy implementation of StudentRepository."""

    def __init__(self, db: SqlSession):
        self.db = db

    def add(self, student: Student) -> Student:
        db_student = StudentDB(
            id=student.id,
            student_id=student.student_id,
            name=student.name,
            email=student.email,
            created_at=student.created_at
        )
        self.db.add(db_student)
        self.db.flush()
        return student

    def get_by_id(self, id_: uuid.UUID) -> Optional[Student]:
        db_student = self.db.query(StudentDB).filter(StudentDB.id == id_).first()
        if not db_student:
            return None
        return Student(
            id=db_student.id,
            student_id=db_student.student_id,
            name=db_student.name,
            email=db_student.email,
            created_at=db_student.created_at
        )

    def get_by_student_id(self, student_id: str) -> Optional[Student]:
        db_student = self.db.query(StudentDB).filter(StudentDB.student_id == student_id).first()
        if not db_student:
            return None
        return Student(
            id=db_student.id,
            student_id=db_student.student_id,
            name=db_student.name,
            email=db_student.email,
            created_at=db_student.created_at
        )

    def list_all(self) -> List[Student]:
        db_students = self.db.query(StudentDB).all()
        return [
            Student(
                id=db_s.id,
                student_id=db_s.student_id,
                name=db_s.name,
                email=db_s.email,
                created_at=db_s.created_at
            ) for db_s in db_students
        ]

    def delete(self, id_: uuid.UUID) -> bool:
        db_student = self.db.query(StudentDB).filter(StudentDB.id == id_).first()
        if db_student:
            self.db.delete(db_student)
            self.db.flush()
            return True
        return False


class SQLAlchemyFaceEmbeddingRepository(FaceEmbeddingRepository):
    """SQLAlchemy implementation of FaceEmbeddingRepository using pgvector."""

    def __init__(self, db: SqlSession):
        self.db = db

    def add(self, embedding: FaceEmbedding) -> FaceEmbedding:
        db_emb = FaceEmbeddingDB(
            id=embedding.id,
            student_id=embedding.student_id,
            embedding=embedding.embedding.tolist(),  # Convert NumPy array to list for pgvector
            created_at=embedding.created_at
        )
        self.db.add(db_emb)
        self.db.flush()
        return embedding

    def get_by_student_id(self, student_id: uuid.UUID) -> List[FaceEmbedding]:
        db_embs = self.db.query(FaceEmbeddingDB).filter(FaceEmbeddingDB.student_id == student_id).all()
        return [
            FaceEmbedding(
                id=db_e.id,
                student_id=db_e.student_id,
                embedding=np.array(db_e.embedding, dtype=np.float32),
                created_at=db_e.created_at
            ) for db_e in db_embs
        ]

    def list_all(self) -> List[FaceEmbedding]:
        db_embs = self.db.query(FaceEmbeddingDB).all()
        return [
            FaceEmbedding(
                id=db_e.id,
                student_id=db_e.student_id,
                embedding=np.array(db_e.embedding, dtype=np.float32),
                created_at=db_e.created_at
            ) for db_e in db_embs
        ]

    def delete(self, id_: uuid.UUID) -> bool:
        db_emb = self.db.query(FaceEmbeddingDB).filter(FaceEmbeddingDB.id == id_).first()
        if db_emb:
            self.db.delete(db_emb)
            self.db.flush()
            return True
        return False


class SQLAlchemySessionRepository(SessionRepository):
    """SQLAlchemy implementation of SessionRepository."""

    def __init__(self, db: SqlSession):
        self.db = db

    def add(self, session: Session) -> Session:
        db_sess = SessionDB(
            id=session.id,
            name=session.name,
            teacher_name=session.teacher_name,
            start_time=session.start_time,
            end_time=session.end_time,
            is_active=session.is_active,
            created_at=session.created_at
        )
        self.db.add(db_sess)
        self.db.flush()
        return session

    def get_by_id(self, id_: uuid.UUID) -> Optional[Session]:
        db_sess = self.db.query(SessionDB).filter(SessionDB.id == id_).first()
        if not db_sess:
            return None
        return Session(
            id=db_sess.id,
            name=db_sess.name,
            teacher_name=db_sess.teacher_name,
            start_time=db_sess.start_time,
            end_time=db_sess.end_time,
            is_active=db_sess.is_active,
            created_at=db_sess.created_at
        )

    def get_active_sessions(self) -> List[Session]:
        db_sessions = self.db.query(SessionDB).filter(SessionDB.is_active == True).all()
        return [
            Session(
                id=db_s.id,
                name=db_s.name,
                teacher_name=db_s.teacher_name,
                start_time=db_s.start_time,
                end_time=db_s.end_time,
                is_active=db_s.is_active,
                created_at=db_s.created_at
            ) for db_s in db_sessions
        ]

    def list_all(self) -> List[Session]:
        db_sessions = self.db.query(SessionDB).all()
        return [
            Session(
                id=db_s.id,
                name=db_s.name,
                teacher_name=db_s.teacher_name,
                start_time=db_s.start_time,
                end_time=db_s.end_time,
                is_active=db_s.is_active,
                created_at=db_s.created_at
            ) for db_s in db_sessions
        ]

    def update(self, session: Session) -> Session:
        db_sess = self.db.query(SessionDB).filter(SessionDB.id == session.id).first()
        if db_sess:
            db_sess.name = session.name
            db_sess.teacher_name = session.teacher_name
            db_sess.start_time = session.start_time
            db_sess.end_time = session.end_time
            db_sess.is_active = session.is_active
            self.db.flush()
        return session


class SQLAlchemyAttendanceRepository(AttendanceRepository):
    """SQLAlchemy implementation of AttendanceRepository."""

    def __init__(self, db: SqlSession):
        self.db = db

    def add(self, record: AttendanceRecord) -> AttendanceRecord:
        db_record = AttendanceRecordDB(
            id=record.id,
            session_id=record.session_id,
            student_id=record.student_id,
            marked_at=record.marked_at,
            status=record.status
        )
        self.db.add(db_record)
        self.db.flush()
        return record

    def get_record(self, session_id: uuid.UUID, student_id: uuid.UUID) -> Optional[AttendanceRecord]:
        db_rec = self.db.query(AttendanceRecordDB).filter(
            AttendanceRecordDB.session_id == session_id,
            AttendanceRecordDB.student_id == student_id
        ).first()
        if not db_rec:
            return None
        return AttendanceRecord(
            id=db_rec.id,
            session_id=db_rec.session_id,
            student_id=db_rec.student_id,
            marked_at=db_rec.marked_at,
            status=db_rec.status
        )

    def get_by_session(self, session_id: uuid.UUID) -> List[AttendanceRecord]:
        db_recs = self.db.query(AttendanceRecordDB).filter(AttendanceRecordDB.session_id == session_id).all()
        return [
            AttendanceRecord(
                id=db_r.id,
                session_id=db_r.session_id,
                student_id=db_r.student_id,
                marked_at=db_r.marked_at,
                status=db_r.status
            ) for db_r in db_recs
        ]

    def get_by_student(self, student_id: uuid.UUID) -> List[AttendanceRecord]:
        db_recs = self.db.query(AttendanceRecordDB).filter(AttendanceRecordDB.student_id == student_id).all()
        return [
            AttendanceRecord(
                id=db_r.id,
                session_id=db_r.session_id,
                student_id=db_r.student_id,
                marked_at=db_r.marked_at,
                status=db_r.status
            ) for db_r in db_recs
        ]

    def list_all(self) -> List[AttendanceRecord]:
        db_recs = self.db.query(AttendanceRecordDB).all()
        return [
            AttendanceRecord(
                id=db_r.id,
                session_id=db_r.session_id,
                student_id=db_r.student_id,
                marked_at=db_r.marked_at,
                status=db_r.status
            ) for db_r in db_recs
        ]
