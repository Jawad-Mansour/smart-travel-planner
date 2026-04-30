"""
Travel agent tools: allowlist plus public tool entrypoints for tests and wiring.
"""

from __future__ import annotations

from backend.app.tools.classifier_tool import classify_destinations
from backend.app.tools.live_tools import flight_estimate_tool, fx_latest_tool, weather_forecast_tool
from backend.app.tools.rag_tool import rag_destination_detail, rag_search

TOOL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "rag_search",
        "rag_destination_detail",
        "classify_destinations",
        "weather_forecast",
        "flight_estimate",
        "fx_rates",
    }
)

__all__ = [
    "TOOL_ALLOWLIST",
    "classify_destinations",
    "flight_estimate_tool",
    "fx_latest_tool",
    "rag_destination_detail",
    "rag_search",
    "weather_forecast_tool",
]
