class AttendanceSystemException(Exception):
    """Base exception for the attendance system."""
    def __init__(self, message: str, error_code: str = "SYSTEM_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code

class DatabaseException(AttendanceSystemException):
    """Exception raised for database operations failures."""
    def __init__(self, message: str, error_code: str = "DATABASE_ERROR"):
        super().__init__(message, error_code)

class AIException(AttendanceSystemException):
    """Exception raised for AI model/processing operations failures."""
    def __init__(self, message: str, error_code: str = "AI_ERROR"):
        super().__init__(message, error_code)

class AttendanceException(AttendanceSystemException):
    """Exception raised for attendance logic violations (e.g. duplicate)."""
    def __init__(self, message: str, error_code: str = "ATTENDANCE_ERROR"):
        super().__init__(message, error_code)

class StudentNotFoundException(DatabaseException):
    """Exception raised when student is not found."""
    def __init__(self, student_id: str):
        super().__init__(f"Student with ID '{student_id}' not found.", "STUDENT_NOT_FOUND")

class SessionException(AttendanceSystemException):
    """Exception raised for Session management issues."""
    def __init__(self, message: str, error_code: str = "SESSION_ERROR"):
        super().__init__(message, error_code)
