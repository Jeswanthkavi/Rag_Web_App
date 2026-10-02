"""Pytest configuration and database fixtures for Advanced RAG test suite."""

import pytest
from typing import Generator
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from alembic.config import Config
from alembic import command
from backend.app.core.config import get_settings
from backend.app.db.base import Base
from backend.app.db.session import create_db_engine, create_session_factory
# Ensure all models are imported
from backend.app.models import (
    User,
    Document,
    DocumentChunk,
    Conversation,
    Message,
)


@pytest.fixture(scope="session")
def test_engine() -> Generator[Engine, None, None]:
    """Provides a SQLAlchemy Engine bound exclusively to the dedicated test database."""
    settings = get_settings()
    test_db_url = settings.database.test_database_url or "mysql+pymysql://root:@localhost:3306/advanced_rag_test"
    
    # Safety assertion: prevent tests from running against the primary production database
    assert "test" in test_db_url.lower(), f"Safety violation: Refusing to run tests against non-test DB: {test_db_url}"
    
    engine = create_db_engine(test_db_url)
    
    # Run Alembic migration to ensure test database matches latest migration revision
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.set_main_option("sqlalchemy.url", test_db_url)
    
    from sqlalchemy import text
    with engine.connect() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        # Drop all known tables plus alembic_version
        result = conn.execute(text("SHOW TABLES;"))
        tables = [row[0] for row in result]
        for tbl in tables:
            conn.execute(text(f"DROP TABLE IF EXISTS `{tbl}`;"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        conn.commit()
        
    command.upgrade(alembic_cfg, "head")
    
    yield engine
    
    # Clean up test schema tables after entire test session completes
    try:
        command.downgrade(alembic_cfg, "base")
    except Exception:
        with engine.connect() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(text(f"DROP TABLE IF EXISTS {table.name};"))
            conn.execute(text("DROP TABLE IF EXISTS alembic_version;"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()
    engine.dispose()


@pytest.fixture(scope="session")
def test_session_factory(test_engine: Engine) -> sessionmaker[Session]:
    """Provides a session factory bound to the test engine."""
    return create_session_factory(test_engine)


@pytest.fixture(scope="function")
def db_session(test_session_factory: sessionmaker[Session], test_engine: Engine) -> Generator[Session, None, None]:
    """Provides an isolated database session per test function with automatic cleanup."""
    session = test_session_factory()
    try:
        yield session
    finally:
        session.rollback()
        # Clean data between test runs to guarantee isolation
        with test_engine.connect() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())
            conn.commit()
        session.close()
