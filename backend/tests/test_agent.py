"""Critical-path tests with mocked external APIs where practical."""

from __future__ import annotations

import pytest

from backend.app.core.agent import _rag_destination_for_search, _rag_digest_for_synthesis
from backend.app.core.country_flags import resolve_destination_flag
from backend.app.schemas.intent import IntentResult, merge_context_patch_into_intent
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


def test_rag_destination_none_when_multi_city_compare() -> None:
    intent = IntentResult(
        duration_days=6,
        budget_usd=2200,
        activities=["sightseeing"],
        timing_or_season="March",
        destination_hint="Lisbon",
        comparison_places=["Lisbon", "Porto", "Seville"],
    )
    assert _rag_destination_for_search(intent) is None


def test_rag_destination_first_city_when_comma_in_hint() -> None:
    intent = IntentResult(
        duration_days=5,
        budget_usd=1000,
        activities=["food"],
        destination_hint="Lisbon, Portugal",
    )
    assert _rag_destination_for_search(intent) == "Lisbon"


def test_rag_destination_none_when_comma_separates_two_cities() -> None:
    intent = IntentResult(
        duration_days=5,
        budget_usd=1000,
        activities=["food"],
        destination_hint="Lisbon, Porto",
    )
    assert _rag_destination_for_search(intent) is None


def test_rag_digest_includes_chunk_content_when_rag_ok() -> None:
    digest = _rag_digest_for_synthesis(
        {
            "rag": {
                "ok": True,
                "payload": {
                    "chunks": [
                        {
                            "destination": "Lisbon",
                            "heading": "Get around",
                            "content": "Trams on route 28 are iconic but crowded in summer.",
                        }
                    ]
                },
            }
        }
    )
    assert "Lisbon" in digest
    assert "Trams" in digest


def test_resolve_destination_flag_uses_country_when_model_sent_globe() -> None:
    assert resolve_destination_flag("🌍", "France") == "🇫🇷"
    assert resolve_destination_flag("", "Thailand") == "🇹🇭"
    assert resolve_destination_flag("FR", "") == "🇫🇷"


def test_merge_context_patch_fills_budget_and_prunes_stale_missing_fields() -> None:
    intent = IntentResult(
        duration_days=7,
        budget_usd=None,
        activities=["beach"],
        timing_or_season="July",
        missing_fields=["budget", "duration", "activities"],
    )
    merged = merge_context_patch_into_intent(intent, {"budget_usd": 1500, "duration_days": 10})
    assert merged.budget_usd == 1500.0
    assert merged.duration_days == 10
    assert "budget" not in merged.missing_fields
    assert "duration" not in merged.missing_fields


def test_merge_context_patch_without_patch_still_prunes_stale_missing() -> None:
    intent = IntentResult(
        duration_days=5,
        budget_usd=800.0,
        activities=["museums"],
        timing_or_season="fall",
        missing_fields=["budget", "duration"],
    )
    merged = merge_context_patch_into_intent(intent, None)
    assert merged.missing_fields == []


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
