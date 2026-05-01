"""Async Alembic environment for Smart Travel Planner."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from backend.app.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _repo_root() -> Path:
    """``backend/alembic/env.py`` → repo root (parent of ``backend``)."""
    return Path(__file__).resolve().parents[2]


def get_url() -> str:
    """
    Resolve DB URL: shell env → ``python-dotenv`` on repo ``.env`` → ``Settings``.
    ``alembic`` does not load ``.env`` into ``os.environ`` by itself.
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        try:
            from dotenv import load_dotenv

            load_dotenv(_repo_root() / ".env", override=False)
        except Exception:
            pass
        url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        from backend.app.core.config import get_settings

        url = str(get_settings().database_url).strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL is required for migrations. Add it to .env at the repo root "
            "or set it in the shell (PowerShell: $env:DATABASE_URL = 'postgresql+asyncpg://...')."
        )
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=get_url().replace("+asyncpg", ""),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()
    connectable = async_engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
