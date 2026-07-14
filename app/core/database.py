import contextlib
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, scoped_session
from app.core.configs import get_settings

settings = get_settings()

# Create SQLAlchemy Engine
engine = create_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

# Thread-safe Session Factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

# Declarative Base for models
Base = declarative_base()

@contextlib.contextmanager
def get_db():
    """Context manager for obtaining a database session.
    
    Ensures that sessions are properly committed, rolled back on exception,
    and closed at the end of the request context.
    """
    session = db_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
