"""FastAPI dependencies — settings, DB, services, optional user identity."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import get_async_session
from backend.app.services.flights_service import FlightsService, get_flights_service
from backend.app.services.fx_service import FxService, get_fx_service
from backend.app.services.rag_service import RAGService
from backend.app.services.weather_service import WeatherService, get_weather_service


async def settings_dep() -> Settings:
    return get_settings()


async def db_session_dep() -> AsyncGenerator[AsyncSession, None]:
    async for s in get_async_session():
        yield s


def weather_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> WeatherService:
    return get_weather_service(
        api_key=settings.weather_api_key,
        cache_ttl_seconds=settings.weather_cache_ttl_seconds,
    )


def flights_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> FlightsService:
    return get_flights_service(
        amadeus_api_key=settings.amadeus_api_key,
        amadeus_api_secret=settings.amadeus_api_secret,
        cache_ttl_seconds=settings.flights_cache_ttl_seconds,
    )


def fx_dep(settings: Annotated[Settings, Depends(settings_dep)]) -> FxService:
    return get_fx_service(
        api_key=settings.exchange_rate_api_key,
        base_url=settings.fx_base_url,
        cache_ttl_seconds=settings.fx_cache_ttl_seconds,
    )


def rag_dep() -> RAGService:
    return RAGService.get_instance()


async def current_user_sub(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    """Anonymous demo identity via header (replace with JWT in production)."""
    return x_user_id or "anonymous"
