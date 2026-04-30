"""Async SQLAlchemy engine and session factory."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.app.core.config import Settings, get_settings
from backend.app.db.models import Base

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def init_engine(settings: Settings | None = None) -> None:
    global _engine, _session_factory
    s = settings or get_settings()
    if _engine is not None:
        return
    _engine = create_async_engine(s.database_url, echo=s.debug, pool_pre_ping=True)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False, class_=AsyncSession)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


async def create_tables() -> None:
    init_engine()
    assert _engine is not None
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    init_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        yield session
