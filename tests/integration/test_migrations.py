"""Integration tests for Alembic migration lifecycle (Phase 4)."""

import os
import pytest
import pymysql
import urllib.parse
from alembic.config import Config
from alembic import command
from backend.app.core.config import get_settings


def get_test_db_connection():
    """Connects to advanced_rag_test database for direct table/index inspection."""
    settings = get_settings()
    test_url = settings.database.test_database_url
    assert "test" in test_url.lower(), f"Safety check: refusing non-test DB: {test_url}"
    
    cleaned = test_url.split("://", 1)[1] if "://" in test_url else test_url
    parsed = urllib.parse.urlparse("//" + cleaned)
    return pymysql.connect(
        host=parsed.hostname or "localhost",
        user=parsed.username or "root",
        password=parsed.password or "",
        port=parsed.port or 3306,
        database=parsed.path.lstrip("/"),
        charset="utf8mb4",
    )


def get_alembic_config(db_url: str) -> Config:
    """Instantiates Alembic Config pointing to alembic.ini with dynamic target URL."""
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


class TestAlembicMigrationLifecycle:
    """Verifies complete upgrade -> downgrade -> upgrade lifecycle and schema integrity."""

    def test_migration_upgrade_and_downgrade_cycle(self):
        settings = get_settings()
        test_db_url = settings.database.test_database_url
        alembic_cfg = get_alembic_config(test_db_url)

        # 1. Downgrade to base first in case test tables exist
        try:
            command.downgrade(alembic_cfg, "base")
        except Exception:
            pass

        # 2. Upgrade to head
        command.upgrade(alembic_cfg, "head")

        # 3. Verify all 5 application tables and alembic_version exist
        conn = get_test_db_connection()
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            tables = {row[0] for row in cur.fetchall()}
            expected_tables = {"users", "documents", "document_chunks", "conversations", "messages", "alembic_version"}
            assert expected_tables.issubset(tables), f"Missing tables. Found: {tables}"

            # Verify columns and indexes in users table
            cur.execute("DESCRIBE users;")
            user_cols = {row[0] for row in cur.fetchall()}
            assert {"id", "email", "hashed_password", "full_name", "is_active", "is_superuser", "created_at", "updated_at"}.issubset(user_cols)

            # Verify composite index on document_chunks
            cur.execute("SHOW INDEX FROM document_chunks WHERE Key_name = 'ix_document_chunk_seq';")
            idx_rows = cur.fetchall()
            assert len(idx_rows) >= 2, "Expected composite index on (document_id, chunk_index)"

            # Verify current revision stored in alembic_version
            cur.execute("SELECT version_num FROM alembic_version;")
            version = cur.fetchone()[0]
            assert version == "d998d17cd59d"

        conn.close()

        # 4. Downgrade back to base
        command.downgrade(alembic_cfg, "base")

        # 5. Verify all tables are removed
        conn = get_test_db_connection()
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            tables = {row[0] for row in cur.fetchall()}
            assert "users" not in tables
            assert "documents" not in tables
            assert "document_chunks" not in tables
            assert "conversations" not in tables
            assert "messages" not in tables
        conn.close()

        # 6. Re-upgrade to head to leave test DB in migrated state
        command.upgrade(alembic_cfg, "head")

        # 7. Confirm restored
        conn = get_test_db_connection()
        with conn.cursor() as cur:
            cur.execute("SHOW TABLES;")
            tables = {row[0] for row in cur.fetchall()}
            assert expected_tables.issubset(tables)
        conn.close()
