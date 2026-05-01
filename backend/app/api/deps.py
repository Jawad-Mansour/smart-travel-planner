"""FastAPI dependencies — settings, DB, services, JWT identity."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import Settings, get_settings
from backend.app.core.security import decode_token_safe
from backend.app.db.models import User
from backend.app.db.session import get_async_session
from backend.app.services.flights_service import FlightsService, get_flights_service
from backend.app.services.fx_service import FxService, get_fx_service
from backend.app.services.rag_service import RAGService, get_instance
from backend.app.services.weather_service import WeatherService, get_weather_service

_http_bearer = HTTPBearer(auto_error=False)


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
    return get_instance()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_http_bearer)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
) -> User:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    payload = decode_token_safe(settings, credentials.credentials)
    if not payload or payload.get("typ") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject"
        )
    try:
        user_id = UUID(str(sub))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject"
        )
    user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


async def current_user_sub(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_http_bearer)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    """
    Prefer JWT ``sub`` (user id) when ``Authorization: Bearer`` is valid;
    otherwise fall back to ``X-User-Id`` or ``anonymous`` (legacy demo / curl).
    """
    if credentials is not None:
        payload = decode_token_safe(settings, credentials.credentials)
        if payload and payload.get("typ") == "access" and payload.get("sub"):
            return str(payload["sub"])
    return x_user_id or "anonymous"
