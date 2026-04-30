"""
Phase 12: OpenWeatherMap weather service.

Uses async httpx, tenacity retries, TTL caching (default 10 minutes), and Pydantic
models at the boundary. Free-tier forecast covers up to 5 days at 3-hour steps;
longer trips return the available window with dates labeled.

Singleton access: use ``get_weather_service(...)`` (``functools.lru_cache``) from
app lifespan or FastAPI dependencies, passing settings-derived primitives only.
Call ``await service.aclose()`` on shutdown to dispose the HTTP client.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any, Literal

import httpx
import structlog
from cachetools import TTLCache
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

DEFAULT_OPENWEATHER_API_ROOT = "https://api.openweathermap.org/data/2.5"
DEFAULT_OPENWEATHER_GEO_ROOT = "https://api.openweathermap.org/geo/1.0"


def _retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class WeatherPeriod(BaseModel):
    """One calendar day (UTC) of aggregated forecast hints."""

    date: date
    temp_min_c: float = Field(..., description="Daily minimum air temperature (°C)")
    temp_max_c: float = Field(..., description="Daily maximum air temperature (°C)")
    conditions_summary: str = Field(
        ...,
        description="Short human-readable summary (dominant daytime condition)",
    )
    precipitation_probability_max: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Max POP across 3-hour slots that day, if provided by API",
    )
    wind_speed_max_m_s: float | None = Field(
        None,
        ge=0.0,
        description="Peak wind speed for the day (m/s)",
    )


class WeatherForecastResult(BaseModel):
    """Successful forecast bundle returned to callers / tools."""

    location_query: str
    resolved_name: str
    latitude: float
    longitude: float
    unit_system: Literal["metric"] = "metric"
    daily: list[WeatherPeriod] = Field(default_factory=list)
    source: Literal["openweathermap"] = "openweathermap"
    note: str | None = Field(
        None,
        description="e.g. free-tier horizon limits",
    )


class WeatherServiceFailure(BaseModel):
    """Structured failure (no stack traces to the agent)."""

    ok: Literal[False] = False
    error: str
    detail: str | None = None


class WeatherServiceResponse(BaseModel):
    """Union-style API wrapper for forecast success or structured failure."""

    ok: bool
    forecast: WeatherForecastResult | None = None
    failure: WeatherServiceFailure | None = None

    @classmethod
    def success(cls, forecast: WeatherForecastResult) -> WeatherServiceResponse:
        return cls(ok=True, forecast=forecast, failure=None)

    @classmethod
    def failure(cls, error: str, detail: str | None = None) -> WeatherServiceResponse:
        return cls(ok=False, forecast=None, failure=WeatherServiceFailure(error=error, detail=detail))


class GeocodeHit(BaseModel):
    """Minimal OpenWeather geocoding row."""

    model_config = ConfigDict(extra="ignore")

    name: str
    lat: float
    lon: float
    country: str | None = None


class _ForecastListItem(BaseModel):
    """Subset of one OpenWeather ``list[]`` entry."""

    model_config = ConfigDict(extra="ignore")

    dt: int
    main: dict[str, Any]
    weather: list[dict[str, Any]]
    wind: dict[str, Any] | None = None
    pop: float | None = None


class WeatherService:
    """
    Async OpenWeatherMap client with TTL cache and retries.

    Parameters are injected explicitly (typically from :class:`Settings` in ``deps``).
    """

    def __init__(
        self,
        *,
        api_key: str,
        cache_ttl_seconds: int = 600,
        request_timeout_seconds: float = 10.0,
        api_root: str = DEFAULT_OPENWEATHER_API_ROOT,
        geo_root: str = DEFAULT_OPENWEATHER_GEO_ROOT,
        max_cache_entries: int = 512,
    ) -> None:
        self._api_key = api_key
        self._cache_ttl = cache_ttl_seconds
        self._timeout = request_timeout_seconds
        self._api_root = str(api_root).rstrip("/")
        self._geo_root = str(geo_root).rstrip("/")
        self._cache: TTLCache[str, WeatherForecastResult] = TTLCache(
            maxsize=max_cache_entries,
            ttl=cache_ttl_seconds,
        )
        self._cache_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=request_timeout_seconds)
        self._closed = False

    async def aclose(self) -> None:
        """Close the shared HTTP client (call from app lifespan shutdown)."""
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    def _require_client(self) -> httpx.AsyncClient:
        if self._closed:
            msg = "WeatherService HTTP client is closed"
            raise RuntimeError(msg)
        return self._client

    @staticmethod
    def _cache_key(
        *,
        lat: float,
        lon: float,
        start: date | None,
        end: date | None,
    ) -> str:
        s = start.isoformat() if start else "none"
        e = end.isoformat() if end else "none"
        return f"{lat:.5f}|{lon:.5f}|{s}|{e}|metric"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_retryable_http),
    )
    async def _request_json(self, method: str, url: str, *, params: dict[str, Any]) -> dict[str, Any]:
        client = self._require_client()
        response = await client.request(method, url, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status >= 500:
                raise
            body_preview = exc.response.text[:500]
            detail = f"HTTP {status}: {body_preview}"
            raise RuntimeError(detail) from exc
        data = response.json()
        if not isinstance(data, dict):
            msg = "OpenWeather response JSON was not an object"
            raise TypeError(msg)
        return data

    async def geocode(self, city: str, *, country_code: str | None = None) -> GeocodeHit | None:
        """Resolve a place name to coordinates (first hit)."""
        if not self._api_key.strip():
            return None
        q = city.strip()
        if country_code:
            q = f"{q},{country_code.strip()}"
        params: dict[str, Any] = {"q": q, "limit": 1, "appid": self._api_key}
        url = f"{self._geo_root}/direct"
        data = await self._request_json("GET", url, params=params)
        if not isinstance(data, list) or not data:
            return None
        row = data[0]
        try:
            return GeocodeHit.model_validate(row)
        except Exception:
            logger.warning("weather.geocode.parse_error", row_keys=list(row.keys()) if isinstance(row, dict) else None)
            return None

    def _aggregate_daily(
        self,
        items: list[_ForecastListItem],
        *,
        start: date | None,
        end: date | None,
    ) -> list[WeatherPeriod]:
        """Aggregate 3-hour slots into UTC calendar days, optionally clipped to [start, end]."""
        by_day: dict[date, list[_ForecastListItem]] = {}
        for raw in items:
            dt_utc = datetime.fromtimestamp(raw.dt, tz=timezone.utc).date()
            by_day.setdefault(dt_utc, []).append(raw)

        days = sorted(by_day.keys())
        if start is not None:
            days = [d for d in days if d >= start]
        if end is not None:
            days = [d for d in days if d <= end]

        periods: list[WeatherPeriod] = []
        for d in days:
            slots = by_day[d]
            temps_min: list[float] = []
            temps_max: list[float] = []
            pops: list[float] = []
            winds: list[float] = []
            descriptions: list[str] = []
            for s in slots:
                main = s.main
                temps_min.append(float(main.get("temp_min", main.get("temp", 0.0))))
                temps_max.append(float(main.get("temp_max", main.get("temp", 0.0))))
                if s.pop is not None:
                    pops.append(float(s.pop))
                if s.wind and "speed" in s.wind:
                    winds.append(float(s.wind["speed"]))
                if s.weather:
                    descriptions.append(str(s.weather[0].get("description", "unknown")))
            summary = max(set(descriptions), key=descriptions.count) if descriptions else "unknown"
            periods.append(
                WeatherPeriod(
                    date=d,
                    temp_min_c=min(temps_min),
                    temp_max_c=max(temps_max),
                    conditions_summary=summary,
                    precipitation_probability_max=max(pops) if pops else None,
                    wind_speed_max_m_s=max(winds) if winds else None,
                )
            )
        return periods

    async def forecast_for_place(
        self,
        *,
        city: str,
        country_code: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> WeatherServiceResponse:
        """
        Fetch a multi-day forecast for ``city`` (optional ISO country hint).

        ``start_date`` / ``end_date`` clip the returned daily rows (UTC dates).
        OpenWeather free ``forecast`` horizon is 5 days from request time; requests
        beyond that window return available days plus an explanatory ``note``.
        """
        query = city.strip()
        if not query:
            return WeatherServiceResponse.failure("City name is required")

        if not self._api_key.strip():
            logger.warning("weather.missing_api_key")
            return WeatherServiceResponse.failure(
                "Weather API key not configured",
                detail="Set WEATHER_API_KEY for OpenWeatherMap access.",
            )

        geo = await self.geocode(query, country_code=country_code)
        if geo is None:
            return WeatherServiceResponse.failure(
                "Could not resolve location",
                detail=f"No geocoding results for {query!r}",
            )

        key = self._cache_key(lat=geo.lat, lon=geo.lon, start=start_date, end=end_date)
        async with self._cache_lock:
            cached = self._cache.get(key)
        if cached is not None:
            logger.debug("weather.cache.hit", key=key)
            return WeatherServiceResponse.success(cached)

        params: dict[str, Any] = {
            "lat": geo.lat,
            "lon": geo.lon,
            "appid": self._api_key,
            "units": "metric",
        }
        url = f"{self._api_root}/forecast"
        try:
            payload = await self._request_json("GET", url, params=params)
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            logger.exception("weather.forecast.transport_error", error=str(exc))
            return WeatherServiceResponse.failure(
                "Weather service temporarily unavailable",
                detail="Upstream transport error after retries.",
            )
        except RuntimeError as exc:
            logger.warning("weather.forecast.client_error", detail=str(exc))
            return WeatherServiceResponse.failure("Weather request failed", detail=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("weather.forecast.unexpected_error", error=str(exc))
            return WeatherServiceResponse.failure("Unexpected weather error", detail=str(exc))

        raw_list = payload.get("list", [])
        items: list[_ForecastListItem] = []
        for row in raw_list:
            try:
                items.append(_ForecastListItem.model_validate(row))
            except Exception:
                continue

        daily = self._aggregate_daily(items, start=start_date, end=end_date)
        pretty_name = geo.name
        if geo.country:
            pretty_name = f"{geo.name}, {geo.country}"

        note: str | None = None
        if start_date and end_date:
            span = (end_date - start_date).days + 1
            if span > 5:
                note = (
                    "OpenWeather free forecast covers about five days from now; "
                    "longer trips show the earliest available window only."
                )

        result = WeatherForecastResult(
            location_query=query,
            resolved_name=pretty_name,
            latitude=geo.lat,
            longitude=geo.lon,
            daily=daily,
            note=note,
        )

        async with self._cache_lock:
            self._cache[key] = result

        logger.info(
            "weather.forecast.ok",
            location=pretty_name,
            days=len(daily),
            cache_ttl_s=self._cache_ttl,
        )
        return WeatherServiceResponse.success(result)


@lru_cache(maxsize=16)
def get_weather_service(
    *,
    api_key: str,
    cache_ttl_seconds: int = 600,
    request_timeout_seconds: float = 10.0,
) -> WeatherService:
    """
    Process-wide cached factory (hashable primitive settings only).

    Wire from ``Settings`` in FastAPI dependencies or lifespan; pass explicit values
    so tests can inject keys without mutating global environment state.
    """
    return WeatherService(
        api_key=api_key,
        cache_ttl_seconds=cache_ttl_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )


def clear_weather_service_cache() -> None:
    """Clear the LRU factory (tests)."""
    get_weather_service.cache_clear()


__all__ = [
    "WeatherForecastResult",
    "WeatherPeriod",
    "WeatherService",
    "WeatherServiceFailure",
    "WeatherServiceResponse",
    "clear_weather_service_cache",
    "get_weather_service",
]
