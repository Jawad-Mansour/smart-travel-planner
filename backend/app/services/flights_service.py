"""
Phase 13: Flight price estimates via Amadeus self-service API when configured,
otherwise deterministic mock estimates suitable for demos.

async httpx, tenacity retries, TTL cache (default 30 minutes), Pydantic boundaries.
"""

from __future__ import annotations

import asyncio
from datetime import date
from functools import lru_cache
from typing import Any, Literal

import httpx
import structlog
from cachetools import TTLCache
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
_AMADEUS_FLIGHT_OFFERS = "https://test.api.amadeus.com/v2/shopping/flight-offers"


def _retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class FlightEstimate(BaseModel):
    """Round-trip price estimate in USD for agent synthesis."""

    origin_display: str = Field(..., description="Departure city/airport label")
    destination_display: str = Field(..., description="Arrival city/airport label")
    currency: Literal["USD"] = "USD"
    round_trip_price_usd_estimate: float = Field(..., ge=0.0)
    source: Literal["amadeus", "mock"] = "mock"
    note: str | None = Field(None, description="Data freshness or assumptions")


class FlightLookupFailure(BaseModel):
    ok: Literal[False] = False
    error: str
    detail: str | None = None


class FlightLookupResult(BaseModel):
    ok: bool
    estimate: FlightEstimate | None = None
    failure: FlightLookupFailure | None = None

    @classmethod
    def success(cls, estimate: FlightEstimate) -> FlightLookupResult:
        return cls(ok=True, estimate=estimate, failure=None)

    @classmethod
    def failure(cls, error: str, detail: str | None = None) -> FlightLookupResult:
        return cls(ok=False, estimate=None, failure=FlightLookupFailure(error=error, detail=detail))


def _mock_estimate(origin_city: str, destination_city: str) -> FlightEstimate:
    """Deterministic pseudo-prices from city names (demo when Amadeus keys absent)."""
    o = origin_city.strip().lower()
    d = destination_city.strip().lower()
    base = 380.0 + (hash((o, d)) % 450)
    longhaul = sum(
        1 for x in ("bangkok", "tokyo", "sydney", "kathmandu", "maldives", "bali", "dubai")
        if x in d or x in o
    )
    price = base + longhaul * 220.0
    return FlightEstimate(
        origin_display=origin_city.strip(),
        destination_display=destination_city.strip(),
        round_trip_price_usd_estimate=round(price, 2),
        source="mock",
        note="Mock estimate (set AMADEUS_* keys for live Amadeus test API quotes).",
    )


class FlightsService:
    """Async flights helper with TTL cache and optional Amadeus OAuth + flight-offers."""

    def __init__(
        self,
        *,
        amadeus_api_key: str | None,
        amadeus_api_secret: str | None,
        cache_ttl_seconds: int = 1800,
        request_timeout_seconds: float = 15.0,
        max_cache_entries: int = 512,
    ) -> None:
        self._key = (amadeus_api_key or "").strip()
        self._secret = (amadeus_api_secret or "").strip()
        self._cache_ttl = cache_ttl_seconds
        self._timeout = request_timeout_seconds
        self._cache: TTLCache[str, FlightEstimate] = TTLCache(maxsize=max_cache_entries, ttl=cache_ttl_seconds)
        self._cache_lock = asyncio.Lock()
        self._token_lock = asyncio.Lock()
        self._access_token: str | None = None
        self._client = httpx.AsyncClient(timeout=request_timeout_seconds)
        self._closed = False

    async def aclose(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    def _require_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("FlightsService HTTP client is closed")
        return self._client

    def _cache_key(
        self,
        origin: str,
        destination: str,
        departure_date: date | None,
        return_date: date | None,
    ) -> str:
        return "|".join(
            (
                origin.strip().lower(),
                destination.strip().lower(),
                departure_date.isoformat() if departure_date else "",
                return_date.isoformat() if return_date else "",
            )
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        retry=retry_if_exception(_retryable_http),
    )
    async def _post_form(self, url: str, data: dict[str, str]) -> dict[str, Any]:
        client = self._require_client()
        r = await client.post(url, data=data)
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, dict):
            raise TypeError("JSON response was not an object")
        return out

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=12),
        retry=retry_if_exception(_retryable_http),
    )
    async def _get_json(self, url: str, *, headers: dict[str, str], params: dict[str, Any]) -> dict[str, Any]:
        client = self._require_client()
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        out = r.json()
        if not isinstance(out, dict):
            raise TypeError("JSON response was not an object")
        return out

    async def _oauth_token(self) -> str | None:
        if self._access_token:
            return self._access_token
        async with self._token_lock:
            if self._access_token:
                return self._access_token
            if not self._key or not self._secret:
                return None
            payload = await self._post_form(
                _AMADEUS_AUTH_URL,
                {
                    "grant_type": "client_credentials",
                    "client_id": self._key,
                    "client_secret": self._secret,
                },
            )
            tok = payload.get("access_token")
            if isinstance(tok, str):
                self._access_token = tok
                return tok
            return None

    async def estimate_round_trip(
        self,
        *,
        origin_city: str,
        destination_city: str,
        departure_date: date | None = None,
        return_date: date | None = None,
        adults: int = 1,
    ) -> FlightLookupResult:
        """
        Return a single aggregate USD estimate for display (not booking).

        Uses Amadeus test flight-offers when credentials exist; otherwise mock.
        """
        if not origin_city.strip() or not destination_city.strip():
            return FlightLookupResult.failure("Origin and destination are required")

        key = self._cache_key(origin_city, destination_city, departure_date, return_date)
        async with self._cache_lock:
            hit = self._cache.get(key)
        if hit:
            logger.debug("flights.cache.hit", key=key)
            return FlightLookupResult.success(hit)

        if not self._key or not self._secret:
            est = _mock_estimate(origin_city, destination_city)
            async with self._cache_lock:
                self._cache[key] = est
            logger.info("flights.mock_estimate", origin=origin_city, destination=destination_city)
            return FlightLookupResult.success(est)

        token = await self._oauth_token()
        if not token:
            est = _mock_estimate(origin_city, destination_city)
            async with self._cache_lock:
                self._cache[key] = est
            return FlightLookupResult.success(est)

        def _iata(s: str) -> str | None:
            t = s.strip().upper()
            return t if len(t) == 3 and t.isalpha() else None

        origin_code = _iata(origin_city)
        dest_code = _iata(destination_city)
        # Amadeus Flight Offers requires IATA codes and a departure date.
        if not origin_code or not dest_code or departure_date is None:
            est = _mock_estimate(origin_city, destination_city)
            est.note = (
                "Mock estimate: provide 3-letter IATA origin/destination and dates for Amadeus quotes."
            )
            async with self._cache_lock:
                self._cache[key] = est
            return FlightLookupResult.success(est)

        params: dict[str, Any] = {
            "originLocationCode": origin_code,
            "destinationLocationCode": dest_code,
            "departureDate": departure_date.isoformat(),
            "adults": adults,
            "currencyCode": "USD",
            "max": 5,
        }
        if return_date is not None:
            params["returnDate"] = return_date.isoformat()

        try:
            data = await self._get_json(
                _AMADEUS_FLIGHT_OFFERS,
                headers={"Authorization": f"Bearer {token}"},
                params=params,
            )
        except httpx.HTTPStatusError as exc:
            logger.warning("flights.amadeus_http_error", status=exc.response.status_code)
            est = _mock_estimate(origin_city, destination_city)
            est.source = "mock"
            est.note = "Amadeus request failed; showing mock estimate. Use IATA airport codes for live quotes."
            async with self._cache_lock:
                self._cache[key] = est
            return FlightLookupResult.success(est)
        except Exception as exc:
            logger.exception("flights.amadeus_error", error=str(exc))
            return FlightLookupResult.failure("Flight search failed", detail=str(exc))

        offers = data.get("data", [])
        if not isinstance(offers, list) or not offers:
            est = _mock_estimate(origin_city, destination_city)
            est.note = "No offers returned; mock estimate shown."
            async with self._cache_lock:
                self._cache[key] = est
            return FlightLookupResult.success(est)

        prices: list[float] = []
        for off in offers[:5]:
            try:
                price = off["price"]["grandTotal"]
                prices.append(float(price))
            except (KeyError, TypeError, ValueError):
                continue
        if not prices:
            est = _mock_estimate(origin_city, destination_city)
            est.note = "Could not parse Amadeus prices; mock estimate shown."
            async with self._cache_lock:
                self._cache[key] = est
            return FlightLookupResult.success(est)

        avg = sum(prices) / len(prices)
        est = FlightEstimate(
            origin_display=origin_city.strip(),
            destination_display=destination_city.strip(),
            round_trip_price_usd_estimate=round(avg, 2),
            source="amadeus",
            note="Amadeus test API aggregate (grandTotal), sample of returned offers.",
        )
        async with self._cache_lock:
            self._cache[key] = est
        logger.info(
            "flights.amadeus_ok",
            origin=origin_city,
            destination=destination_city,
            price=est.round_trip_price_usd_estimate,
        )
        return FlightLookupResult.success(est)


@lru_cache(maxsize=8)
def get_flights_service(
    *,
    amadeus_api_key: str | None,
    amadeus_api_secret: str | None,
    cache_ttl_seconds: int = 1800,
    request_timeout_seconds: float = 15.0,
) -> FlightsService:
    return FlightsService(
        amadeus_api_key=amadeus_api_key,
        amadeus_api_secret=amadeus_api_secret,
        cache_ttl_seconds=cache_ttl_seconds,
        request_timeout_seconds=request_timeout_seconds,
    )


def clear_flights_service_cache() -> None:
    get_flights_service.cache_clear()


__all__ = [
    "FlightEstimate",
    "FlightLookupFailure",
    "FlightLookupResult",
    "FlightsService",
    "clear_flights_service_cache",
    "get_flights_service",
]
