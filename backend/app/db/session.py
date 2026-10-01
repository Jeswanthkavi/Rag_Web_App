"""Database engine, session factory, connection pooling, and FastAPI session dependency."""

from typing import Generator, Optional, Dict, Any
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import get_settings, DatabaseSettings


def create_db_engine(
    db_url: str,
    db_settings: Optional[DatabaseSettings] = None,
) -> Engine:
    """Creates a configured SQLAlchemy Engine with connection pooling."""
    settings = db_settings or get_settings().database
    
    # Handle SQLite for in-memory testing if ever requested, otherwise standard pooling
    if db_url.startswith("sqlite"):
        return create_engine(
            db_url,
            echo=settings.db_echo,
            connect_args={"check_same_thread": False},
        )
        
    return create_engine(
        db_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=settings.db_pool_recycle,
        pool_pre_ping=True,
        echo=settings.db_echo,
    )


def create_session_factory(target_engine: Engine) -> sessionmaker[Session]:
    """Creates a configured sessionmaker bound to the given engine."""
    return sessionmaker(
        bind=target_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


# App-level Engine & SessionLocal
engine: Engine = create_db_engine(get_settings().database.database_url)
SessionLocal: sessionmaker[Session] = create_session_factory(engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for yielding database sessions with safe lifecycle management."""
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def check_db_health(target_engine_or_session: Optional[Any] = None) -> Dict[str, Any]:
    """Probes the database connection and returns health metrics."""
    target = target_engine_or_session or engine
    
    try:
        if isinstance(target, Session):
            result = target.execute(text("SELECT 1 AS is_alive, DATABASE() AS db_name, VERSION() AS version")).fetchone()
        else:
            with target.connect() as conn:
                result = conn.execute(text("SELECT 1 AS is_alive, DATABASE() AS db_name, VERSION() AS version")).fetchone()
                
        return {
            "status": "healthy",
            "is_alive": bool(result[0]) if result else False,
            "database_name": str(result[1]) if result and len(result) > 1 else None,
            "version": str(result[2]) if result and len(result) > 2 else None,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "is_alive": False,
            "error": str(e),
        }
