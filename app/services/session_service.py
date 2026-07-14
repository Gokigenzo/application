from typing import List, Optional
import uuid
from datetime import datetime
from loguru import logger

from app.domain.entities import Session
from app.domain.repositories import SessionRepository
from app.core.exceptions import SessionException

class SessionService:
    """Manages class lecture and attendance sessions."""

    def __init__(self, session_repo: SessionRepository):
        self.session_repo = session_repo

    def create_session(
        self,
        name: str,
        teacher_name: str,
        start_time: datetime,
        end_time: datetime
    ) -> Session:
        """Creates a new active attendance session.

        Args:
            name: Title of the class (e.g. Mathematics 101).
            teacher_name: Name of the presiding instructor.
            start_time: Expected start timestamp.
            end_time: Expected end timestamp.

        Returns:
            The created Session domain entity.
        """
        if start_time >= end_time:
            raise SessionException("Session start time must precede its end time.", "INVALID_TIME_RANGE")

        session = Session(
            name=name,
            teacher_name=teacher_name,
            start_time=start_time,
            end_time=end_time,
            is_active=True
        )
        self.session_repo.add(session)
        logger.info(f"New session created: '{name}' by Teacher: '{teacher_name}' (ID: {session.id})")
        return session

    def list_sessions(self) -> List[Session]:
        """Lists all class sessions (active and archived)."""
        return self.session_repo.list_all()

    def get_active_sessions(self) -> List[Session]:
        """Retrieves all sessions currently accepting attendance registration."""
        return self.session_repo.get_active_sessions()

    def get_session_by_id(self, id_: uuid.UUID) -> Optional[Session]:
        """Finds session details by its UUID."""
        return self.session_repo.get_by_id(id_)

    def close_session(self, id_: uuid.UUID) -> Optional[Session]:
        """Deactivates a session, halting further attendance registrations."""
        session = self.session_repo.get_by_id(id_)
        if session:
            session.is_active = False
            self.session_repo.update(session)
            logger.info(f"Closed session '{session.name}' (ID: {id_}).")
        return session
