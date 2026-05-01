"""Critical-path tests with mocked external APIs where practical."""

from __future__ import annotations

import pytest

from backend.app.schemas.intent import IntentResult
from backend.app.services.weather_service import WeatherService


def test_intent_critical_missing_covers_duration_budget_activities() -> None:
    intent = IntentResult(
        duration_days=None,
        budget_usd=1200.0,
        activities=["hiking"],
        timing_or_season="October",
    )
    miss = intent.critical_missing()
    assert "duration" in miss
    assert "budget" not in miss
    assert "activities" not in miss
    assert "preferred_month" not in miss


def test_intent_critical_missing_includes_preferred_month_when_timing_absent() -> None:
    intent = IntentResult(
        duration_days=7,
        budget_usd=1500.0,
        activities=["hiking"],
        timing_or_season=None,
    )
    assert "preferred_month" in intent.critical_missing()


@pytest.mark.asyncio
async def test_weather_service_returns_structured_failure_without_api_key() -> None:
    svc = WeatherService(api_key="", cache_ttl_seconds=60)
    try:
        res = await svc.forecast_for_place(city="Paris")
        assert res.ok is False
        assert res.failure is not None
        assert "not configured" in res.failure.error.lower()
    finally:
        await svc.aclose()


def test_tool_allowlist_contains_core_tools() -> None:
    from backend.app.tools import TOOL_ALLOWLIST

    assert "rag_search" in TOOL_ALLOWLIST
    assert "classify_destinations" in TOOL_ALLOWLIST
    assert "weather_forecast" in TOOL_ALLOWLIST
