"""FastAPI entrypoint for Smart Travel Planner."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.routes.auth import router as auth_router
from backend.app.api.routes.chat import router as chat_router
from backend.app.api.routes.sessions import router as sessions_router
from backend.app.api.routes.travel import router as travel_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.db.session import create_tables, dispose_engine, init_engine
from backend.app.services.flights_service import clear_flights_service_cache, get_flights_service
from backend.app.services.fx_service import clear_fx_service_cache, get_fx_service
from backend.app.services.rag_service import get_instance as get_rag_service
from backend.app.services.weather_service import clear_weather_service_cache, get_weather_service

_settings = get_settings()


async def _shutdown_cached_http_clients() -> None:
    """Close singleton HTTP clients and clear factories (idempotent)."""
    s = _settings
    try:
        w = get_weather_service(
            api_key=s.weather_api_key,
            cache_ttl_seconds=s.weather_cache_ttl_seconds,
        )
        await w.aclose()
    except Exception:
        pass
    try:
        f = get_flights_service(
            amadeus_api_key=s.amadeus_api_key,
            amadeus_api_secret=s.amadeus_api_secret,
            cache_ttl_seconds=s.flights_cache_ttl_seconds,
        )
        await f.aclose()
    except Exception:
        pass
    try:
        x = get_fx_service(
            api_key=s.exchange_rate_api_key,
            base_url=s.fx_base_url,
            cache_ttl_seconds=s.fx_cache_ttl_seconds,
        )
        await x.aclose()
    except Exception:
        pass
    clear_weather_service_cache()
    clear_flights_service_cache()
    clear_fx_service_cache()


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging(json_logs=not _settings.debug)
    init_engine(_settings)
    await create_tables()
    await get_rag_service().startup()
    yield
    await _shutdown_cached_http_clients()
    try:
        await get_rag_service().shutdown()
    except Exception:
        pass
    await dispose_engine()


app = FastAPI(title=_settings.app_name, lifespan=lifespan)

_origins = [o.strip() for o in _settings.cors_allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/api")
app.include_router(sessions_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(travel_router, prefix="/api")


@app.exception_handler(Exception)
async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors (HTTPException and validation errors use more specific handlers)."""
    log = structlog.get_logger("api")
    log.exception("unhandled_exception", path=str(request.url.path))
    detail = str(exc) if _settings.debug else "Internal server error"
    return JSONResponse(status_code=500, content={"detail": detail})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
async def public_meta() -> dict[str, Any]:
    """Static dataset stats for the UI (aligned with ingest notebook)."""
    return {
        "rag": {
            "destination_count": 25,
            "parent_chunks": 1184,
            "child_chunks": 20047,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dims": 384,
        },
        "ml_classifier": {
            "destinations_trained": 155,
            "styles": ["Adventure", "Culture", "Budget", "Luxury", "Family", "Relaxation"],
            "test_f1": 0.894,
        },
    }
