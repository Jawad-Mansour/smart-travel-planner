"""Single ``Settings`` object for the backend (pydantic-settings, ``extra='forbid'``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """
    All configuration is loaded here — avoid ``os.getenv`` scattered across modules.
    """

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    app_name: str = Field(default="Smart Travel Planner", alias="APP_NAME")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    debug: bool = Field(default=False, alias="DEBUG")

    secret_key: str = Field(default="change-me-in-production-use-long-random", alias="SECRET_KEY")
    jwt_secret_key: str = Field(default="change-me-jwt-secret", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expiry_minutes: int = Field(default=60, alias="JWT_EXPIRY_MINUTES")
    jwt_access_expire_minutes: int = Field(default=30, alias="JWT_ACCESS_EXPIRE_MINUTES")
    jwt_refresh_expire_days: int = Field(default=14, alias="JWT_REFRESH_EXPIRE_DAYS")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/travel_planner",
        alias="DATABASE_URL",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_cheap_model: str = Field(default="gpt-4o-mini", alias="OPENAI_CHEAP_MODEL")
    openai_strong_model: str = Field(
        default="gpt-4o",
        alias="OPENAI_STRONG_MODEL",
        description="Strong model for final travel synthesis (structured multi-destination answers).",
    )

    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    weather_api_key: str = Field(default="", alias="WEATHER_API_KEY")
    weather_cache_ttl_seconds: int = Field(default=600, alias="WEATHER_CACHE_TTL_SECONDS")

    amadeus_api_key: str | None = Field(default=None, alias="AMADEUS_API_KEY")
    amadeus_api_secret: str | None = Field(default=None, alias="AMADEUS_API_SECRET")
    flights_cache_ttl_seconds: int = Field(default=3600, alias="FLIGHTS_CACHE_TTL_SECONDS")

    exchange_rate_api_key: str | None = Field(default=None, alias="EXCHANGE_RATE_API_KEY")
    fx_cache_ttl_seconds: int = Field(default=3600, alias="FX_CACHE_TTL_SECONDS")
    fx_base_url: str = Field(default="https://open.er-api.com/v6/latest", alias="FX_BASE_URL")

    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str | None = Field(default="smart-travel-planner", alias="LANGCHAIN_PROJECT")
    langsmith_tracing: bool | None = Field(default=None, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str | None = Field(default=None, alias="LANGSMITH_ENDPOINT")
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str | None = Field(default=None, alias="LANGSMITH_PROJECT")

    frontend_url: str = Field(default="http://localhost:5173", alias="FRONTEND_URL")
    cors_allowed_origins: str = Field(
        default=(
            "http://localhost:5173,http://127.0.0.1:5173,"
            "http://localhost,http://127.0.0.1,"
            "http://localhost:80,http://127.0.0.1:80,"
            "http://localhost:8080,http://127.0.0.1:8080,"
            "http://localhost:3000,http://127.0.0.1:3000"
        ),
        alias="CORS_ALLOWED_ORIGINS",
        description="Comma-separated browser origins (local dev + Docker UI)",
    )

    ml_models_dir: Path = Field(
        default=_PROJECT_ROOT / "backend" / "ml" / "models",
        alias="ML_MODELS_DIR",
    )
    ml_destinations_csv: Path = Field(
        default=_PROJECT_ROOT / "backend" / "ml" / "data" / "destinations_raw.csv",
        alias="ML_DESTINATIONS_CSV",
    )

    default_flight_origin: str = Field(default="NYC", alias="DEFAULT_FLIGHT_ORIGIN")

    discord_webhook_url: str | None = Field(default=None, alias="DISCORD_WEBHOOK_URL")
    slack_webhook_url: str | None = Field(default=None, alias="SLACK_WEBHOOK_URL")

    # Optional: email the signed-in user when a full plan is ready (SMTP).
    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from: str = Field(default="", alias="SMTP_FROM")
    smtp_use_tls: bool = Field(default=True, alias="SMTP_USE_TLS")
    smtp_use_ssl: bool = Field(default=False, alias="SMTP_USE_SSL")
    rag_relevance_threshold: float = Field(default=0.48, alias="RAG_RELEVANCE_THRESHOLD")
    rag_gibberish_raw_cap: float = Field(default=0.4, alias="RAG_GIBBERISH_RAW_CAP")

    @field_validator("database_url", mode="before")
    @classmethod
    def ensure_async_database_driver(cls, v: str) -> str:
        """
        Normalize sync Postgres URL to asyncpg for SQLAlchemy async engine.
        """
        s = str(v or "").strip()
        if s.startswith("postgresql://"):
            return s.replace("postgresql://", "postgresql+asyncpg://", 1)
        return s


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
