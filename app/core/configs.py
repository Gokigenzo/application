from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application configuration settings read from environment variables and .env file."""
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/postgres"
    SIMILARITY_THRESHOLD: float = 0.60
    FACE_RECOGNITION_MODEL_NAME: str = "buffalo_l"
    DETECTION_THRESHOLD: float = 0.5
    TRACKING_BUFFER_FRAMES: int = 30
    TRACKING_THRESHOLD: float = 0.5
    TEMPORAL_VOTING_MIN_VOTES: int = 4
    TEMPORAL_VOTING_BUFFER_SIZE: int = 7
    CAMERA_STARTUP_TIMEOUT_SEC: float = 2.0
    APP_TITLE: str = "AI Face Attendance System"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    """Returns a singleton settings instance."""
    return Settings()
