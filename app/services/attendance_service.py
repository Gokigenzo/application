from typing import List, Dict, Any
import uuid
from datetime import datetime, timezone
from loguru import logger

from app.domain.entities import AttendanceRecord
from app.domain.repositories import AttendanceRepository, SessionRepository, StudentRepository
from app.core.exceptions import AttendanceException

class AttendanceService:
    """Manages attendance verification, duplicate checking, and logging."""

    def __init__(
        self,
        attendance_repo: AttendanceRepository,
        session_repo: SessionRepository,
        student_repo: StudentRepository
    ):
        self.attendance_repo = attendance_repo
        self.session_repo = session_repo
        self.student_repo = student_repo

    def record_attendance(
        self,
        session_id: uuid.UUID,
        student_id: uuid.UUID,
        status: str = "PRESENT"
    ) -> str:
        """Records student attendance in a session, preventing duplicates.

        Args:
            session_id: Target class Session UUID.
            student_id: Present Student UUID.
            status: Attendance status flag (default "PRESENT").

        Returns:
            Status result code: "RECORDED", "ALREADY_ATTENDED", or "SESSION_INACTIVE".
        """
        # 1. Verify target session is active
        session = self.session_repo.get_by_id(session_id)
        if not session:
            raise AttendanceException("Class session not found.", "SESSION_NOT_FOUND")
        if not session.is_active:
            logger.warning(f"Refused attendance for student '{student_id}': Session '{session_id}' is closed.")
            return "SESSION_INACTIVE"

        # 2. Check for duplicate record (Idempotency check)
        existing = self.attendance_repo.get_record(session_id, student_id)
        if existing:
            return "ALREADY_ATTENDED"

        # 3. Create and persist attendance record
        record = AttendanceRecord(
            session_id=session_id,
            student_id=student_id,
            status=status,
            marked_at=datetime.now(timezone.utc)
        )
        
        try:
            self.attendance_repo.add(record)
            logger.info(f"Student '{student_id}' marked PRESENT in Session '{session_id}'.")
            return "RECORDED"
        except Exception as e:
            # Catch unique database constraints in case of rare multi-threaded race conditions
            logger.warning(f"Concurrency match caught. Marking as duplicate: {str(e)}")
            return "ALREADY_ATTENDED"

    def get_attendance_sheet(self, session_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Returns details of all students marked present for a session."""
        records = self.attendance_repo.get_by_session(session_id)
        sheet = []
        for rec in records:
            student = self.student_repo.get_by_id(rec.student_id)
            if student:
                sheet.append({
                    "record_id": rec.id,
                    "student_uuid": student.id,
                    "student_id": student.student_id,
                    "student_name": student.name,
                    "student_email": student.email,
                    "marked_at": rec.marked_at,
                    "status": rec.status
                })
        # Sort by marked timestamp descending
        sheet.sort(key=lambda x: x["marked_at"], reverse=True)
        return sheet
