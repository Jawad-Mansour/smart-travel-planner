"""
Phase 14: Exchange rates via ExchangeRate-API compatible endpoint (async httpx).
TTL cache default 1 hour. Used for daily budget conversion to user's currency.

See https://www.exchangerate-api.com/ — default base URL is the open client endpoint.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import httpx
import structlog
from cachetools import TTLCache
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

DEFAULT_FX_BASE_URL = "https://open.er-api.com/v6/latest"


def _retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class ExchangeRatesSnapshot(BaseModel):
    """FX snapshot keyed by currency code (uppercase)."""

    base_code: str = Field(..., description="Base currency, typically USD")
    rates: dict[str, float] = Field(..., description="1 base = rates[code] in that currency")
    source: str = Field(default="exchangerate-api")
    note: str | None = None


class FxFailure(BaseModel):
    ok: bool = False
    error: str
    detail: str | None = None


class FxResult(BaseModel):
    ok: bool
    snapshot: ExchangeRatesSnapshot | None = None
    failure: FxFailure | None = None

    @classmethod
    def success(cls, snapshot: ExchangeRatesSnapshot) -> FxResult:
        return cls(ok=True, snapshot=snapshot, failure=None)

    @classmethod
    def fail(cls, error: str, detail: str | None = None) -> FxResult:
        return cls(ok=False, snapshot=None, failure=FxFailure(error=error, detail=detail))


class FxService:
    """Async FX client with hourly TTL cache."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_FX_BASE_URL,
        cache_ttl_seconds: int = 3600,
        request_timeout_seconds: float = 10.0,
        default_base_currency: str = "USD",
        max_cache_entries: int = 32,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._cache_ttl = cache_ttl_seconds
        self._timeout = request_timeout_seconds
        self._base_ccy = default_base_currency.upper()
        self._cache: TTLCache[str, ExchangeRatesSnapshot] = TTLCache(
            maxsize=max_cache_entries,
            ttl=cache_ttl_seconds,
        )
        self._cache_lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=request_timeout_seconds)
        self._closed = False

    async def aclose(self) -> None:
        if not self._closed:
            await self._client.aclose()
            self._closed = True

    def _require_client(self) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("FxService HTTP client is closed")
        return self._client

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_retryable_http),
    )
    async def _get_json(self, url: str) -> dict[str, Any]:
        client = self._require_client()
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise TypeError("FX JSON was not an object")
        return data

    async def latest_rates(self, *, base_currency: str | None = None) -> FxResult:
        """Fetch latest rates with ``base_currency`` (default from settings)."""
        base = (base_currency or self._base_ccy).upper()
        cache_key = f"latest:{base}"
        async with self._cache_lock:
            hit = self._cache.get(cache_key)
        if hit:
            logger.debug("fx.cache.hit", key=cache_key)
            return FxResult.success(hit)

        # Open.er-api.com v6: https://open.er-api.com/v6/latest/USD (no key)
        # ExchangeRate-API paid: https://v6.exchangerate-api.com/v6/{KEY}/latest/USD
        if self._api_key:
            url = f"https://v6.exchangerate-api.com/v6/{self._api_key}/latest/{base}"
        else:
            url = f"{self._base_url}/{base}"

        try:
            data = await self._get_json(url)
        except httpx.HTTPStatusError as exc:
            logger.warning("fx.http_error", status=exc.response.status_code)
            return FxResult.fail("Exchange rate request failed", detail=str(exc))
        except Exception as exc:
            logger.exception("fx.error", error=str(exc))
            return FxResult.fail("Exchange rate service error", detail=str(exc))

        # open.er-api shape: { "result": "success", "base_code": "USD", "rates": { ... } }
        rates_raw = data.get("rates")
        if not isinstance(rates_raw, dict):
            return FxResult.fail("Unexpected FX payload", detail="missing rates")

        rates: dict[str, float] = {}
        for k, v in rates_raw.items():
            try:
                rates[str(k).upper()] = float(v)
            except (TypeError, ValueError):
                continue

        base_code = str(data.get("base_code", base)).upper()
        snap = ExchangeRatesSnapshot(base_code=base_code, rates=rates, source="exchangerate-api")
        async with self._cache_lock:
            self._cache[cache_key] = snap
        logger.info("fx.latest_ok", base=base_code, n_rates=len(rates))
        return FxResult.success(snap)

    async def convert_usd_to(self, amount_usd: float, target_currency: str) -> FxResult:
        """Convert an amount expressed in USD to ``target_currency`` (e.g. EUR)."""
        if amount_usd < 0:
            return FxResult.fail("Amount must be non-negative")
        tgt = target_currency.strip().upper()
        root = await self.latest_rates(base_currency="USD")
        if not root.ok or root.snapshot is None:
            return root
        rate = root.snapshot.rates.get(tgt)
        if rate is None:
            return FxResult.fail(f"No rate for {tgt}", detail="Currency not in snapshot")
        converted = amount_usd * rate
        snap = ExchangeRatesSnapshot(
            base_code="USD",
            rates={tgt: rate},
            note=f"{amount_usd:.2f} USD ≈ {converted:.2f} {tgt}",
        )
        return FxResult.success(snap)


@lru_cache(maxsize=8)
def get_fx_service(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_FX_BASE_URL,
    cache_ttl_seconds: int = 3600,
    request_timeout_seconds: float = 10.0,
    default_base_currency: str = "USD",
) -> FxService:
    return FxService(
        api_key=api_key,
        base_url=base_url,
        cache_ttl_seconds=cache_ttl_seconds,
        request_timeout_seconds=request_timeout_seconds,
        default_base_currency=default_base_currency,
    )


def clear_fx_service_cache() -> None:
    get_fx_service.cache_clear()


__all__ = [
    "DEFAULT_FX_BASE_URL",
    "ExchangeRatesSnapshot",
    "FxFailure",
    "FxResult",
    "FxService",
    "clear_fx_service_cache",
    "get_fx_service",
]
