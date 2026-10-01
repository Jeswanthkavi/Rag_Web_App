"""Alembic environment configuration for Advanced RAG Platform."""

import logging
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

# Import application settings and declarative models
from backend.app.core.config import get_settings
from backend.app.models import (
    Base,
    User,
    Document,
    DocumentStatus,
    DocumentChunk,
    Conversation,
    Message,
    MessageRole,
)

# Alembic Config object
config = context.config

# Setup logging if config file exists
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

logger = logging.getLogger("alembic.env")

# Target metadata for autogenerate support
target_metadata = Base.metadata


def get_target_url() -> str:
    """Dynamically resolves the target database URL from CLI arguments or environment settings."""
    # Check if a custom DB url was passed via alembic -x db_url=...
    cmd_line_url = context.get_x_argument(as_dictionary=True).get("db_url")
    if cmd_line_url:
        return cmd_line_url

    # Check if already set on config object
    cfg_url = config.get_main_option("sqlalchemy.url")
    if cfg_url and cfg_url.strip():
        return cfg_url.strip()

    # Fall back to application database URL from environment
    settings = get_settings()
    return settings.database.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_target_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    target_url = get_target_url()
    
    # Configure the URL dynamically
    configuration = config.get_section(config.config_ini_section, {}) or {}
    configuration["sqlalchemy.url"] = target_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
