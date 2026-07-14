import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from app.domain.entities import Student, Session, AttendanceRecord
from app.domain.repositories import StudentRepository, SessionRepository, AttendanceRepository
from app.services.attendance_service import AttendanceService
from app.core.exceptions import AttendanceException

# Mock implementations of repositories to isolate service testing from PostgreSQL
class MockStudentRepository(StudentRepository):
    def __init__(self):
        self.db = {}

    def add(self, student: Student) -> Student:
        self.db[student.id] = student
        return student

    def get_by_id(self, id_: uuid.UUID) -> Optional[Student]:
        return self.db.get(id_)

    def get_by_student_id(self, student_id: str) -> Optional[Student]:
        for s in self.db.values():
            if s.student_id == student_id:
                return s
        return None

    def list_all(self) -> List[Student]:
        return list(self.db.values())

    def delete(self, id_: uuid.UUID) -> bool:
        return self.db.pop(id_, None) is not None


class MockSessionRepository(SessionRepository):
    def __init__(self):
        self.db = {}

    def add(self, session: Session) -> Session:
        self.db[session.id] = session
        return session

    def get_by_id(self, id_: uuid.UUID) -> Optional[Session]:
        return self.db.get(id_)

    def get_active_sessions(self) -> List[Session]:
        return [s for s in self.db.values() if s.is_active]

    def list_all(self) -> List[Session]:
        return list(self.db.values())

    def update(self, session: Session) -> Session:
        self.db[session.id] = session
        return session


class MockAttendanceRepository(AttendanceRepository):
    def __init__(self):
        self.db = {}

    def add(self, record: AttendanceRecord) -> AttendanceRecord:
        self.db[(record.session_id, record.student_id)] = record
        return record

    def get_record(self, session_id: uuid.UUID, student_id: uuid.UUID) -> Optional[AttendanceRecord]:
        return self.db.get((session_id, student_id))

    def get_by_session(self, session_id: uuid.UUID) -> List[AttendanceRecord]:
        return [r for r in self.db.values() if r.session_id == session_id]

    def get_by_student(self, student_id: uuid.UUID) -> List[AttendanceRecord]:
        return [r for r in self.db.values() if r.student_id == student_id]

    def list_all(self) -> List[AttendanceRecord]:
        return list(self.db.values())


def test_attendance_marking_logic() -> None:
    """Verifies that duplicate student registrations are blocked and session status is honored."""
    s_repo = MockStudentRepository()
    sess_repo = MockSessionRepository()
    att_repo = MockAttendanceRepository()
    
    service = AttendanceService(att_repo, sess_repo, s_repo)

    # Setup dummy student
    student = Student(student_id="STD001", name="Alice Cooper")
    s_repo.add(student)

    # Setup dummy class session
    session = Session(
        name="Chemistry 101",
        teacher_name="Prof. Snape",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc) + timedelta(hours=1),
        is_active=True
    )
    sess_repo.add(session)

    # 1. Assert normal check-in records successfully
    res1 = service.record_attendance(session.id, student.id)
    assert res1 == "RECORDED"
    assert len(att_repo.get_by_session(session.id)) == 1

    # 2. Assert duplicate check-in is blocked (Anti-duplicate policy)
    res2 = service.record_attendance(session.id, student.id)
    assert res2 == "ALREADY_ATTENDED"
    assert len(att_repo.get_by_session(session.id)) == 1  # Still exactly 1 record

    # 3. Assert check-in fails on inactive sessions
    session.is_active = False
    sess_repo.update(session)

    other_student = Student(student_id="STD002", name="Bob Marley")
    s_repo.add(other_student)

    res3 = service.record_attendance(session.id, other_student.id)
    assert res3 == "SESSION_INACTIVE"
    assert len(att_repo.get_by_session(session.id)) == 1  # Second student rejected
