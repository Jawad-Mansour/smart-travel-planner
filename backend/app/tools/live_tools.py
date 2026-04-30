"""
Phase 15: Async wrappers around weather, flights, and FX services with structured envelopes.
"""

from __future__ import annotations

import time
from datetime import date

import structlog

from backend.app.schemas.tools import ToolEnvelope, ToolError
from backend.app.services.flights_service import FlightsService
from backend.app.services.fx_service import FxService
from backend.app.services.weather_service import WeatherService

logger = structlog.get_logger(__name__)


async def weather_forecast_tool(
    weather: WeatherService,
    *,
    city: str,
    country_code: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> ToolEnvelope:
    tool_name = "weather_forecast"
    t0 = time.perf_counter()
    try:
        res = await weather.forecast_for_place(
            city=city,
            country_code=country_code,
            start_date=start_date,
            end_date=end_date,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        if not res.ok or res.forecast is None:
            err = res.failure.error if res.failure else "unknown"
            detail = res.failure.detail if res.failure else None
            return ToolEnvelope(
                ok=False,
                tool=tool_name,
                payload=None,
                error=ToolError(tool=tool_name, error=err, detail=detail),
            )
        payload = {
            "forecast": res.forecast.model_dump(mode="json"),
            "duration_ms": ms,
        }
        return ToolEnvelope(ok=True, tool=tool_name, payload=payload, error=None)
    except Exception as exc:
        logger.exception("tool.weather_forecast.error")
        return ToolEnvelope(
            ok=False,
            tool=tool_name,
            payload=None,
            error=ToolError(tool=tool_name, error="weather_failed", detail=str(exc)),
        )


async def flight_estimate_tool(
    flights: FlightsService,
    *,
    origin_city: str,
    destination_city: str,
    departure_date: date | None = None,
    return_date: date | None = None,
) -> ToolEnvelope:
    tool_name = "flight_estimate"
    t0 = time.perf_counter()
    try:
        res = await flights.estimate_round_trip(
            origin_city=origin_city,
            destination_city=destination_city,
            departure_date=departure_date,
            return_date=return_date,
        )
        ms = int((time.perf_counter() - t0) * 1000)
        if not res.ok or res.estimate is None:
            err = res.failure.error if res.failure else "unknown"
            detail = res.failure.detail if res.failure else None
            return ToolEnvelope(
                ok=False,
                tool=tool_name,
                payload=None,
                error=ToolError(tool=tool_name, error=err, detail=detail),
            )
        payload = {"estimate": res.estimate.model_dump(mode="json"), "duration_ms": ms}
        return ToolEnvelope(ok=True, tool=tool_name, payload=payload, error=None)
    except Exception as exc:
        logger.exception("tool.flight_estimate.error")
        return ToolEnvelope(
            ok=False,
            tool=tool_name,
            payload=None,
            error=ToolError(tool=tool_name, error="flight_failed", detail=str(exc)),
        )


async def fx_latest_tool(
    fx: FxService,
    *,
    target_currency: str = "EUR",
) -> ToolEnvelope:
    tool_name = "fx_rates"
    t0 = time.perf_counter()
    try:
        res = await fx.latest_rates()
        ms = int((time.perf_counter() - t0) * 1000)
        if not res.ok or res.snapshot is None:
            err = res.failure.error if res.failure else "unknown"
            detail = res.failure.detail if res.failure else None
            return ToolEnvelope(
                ok=False,
                tool=tool_name,
                payload=None,
                error=ToolError(tool=tool_name, error=err, detail=detail),
            )
        tgt = target_currency.strip().upper()
        rate = res.snapshot.rates.get(tgt)
        payload = {
            "base": res.snapshot.base_code,
            "sample_rate_to": {tgt: rate} if rate is not None else {},
            "rates_truncated": dict(list(res.snapshot.rates.items())[:12]),
            "duration_ms": ms,
        }
        return ToolEnvelope(ok=True, tool=tool_name, payload=payload, error=None)
    except Exception as exc:
        logger.exception("tool.fx_rates.error")
        return ToolEnvelope(
            ok=False,
            tool=tool_name,
            payload=None,
            error=ToolError(tool=tool_name, error="fx_failed", detail=str(exc)),
        )


__all__ = [
    "flight_estimate_tool",
    "fx_latest_tool",
    "weather_forecast_tool",
]
