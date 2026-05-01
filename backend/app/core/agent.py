"""
Phases 17–22: LangGraph agent — intent extraction, clarification, parallel tools, synthesis.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, AsyncIterator, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

try:
    from langsmith import traceable
except Exception:  # pragma: no cover - optional tracing fallback

    def traceable(*_: Any, **__: Any):
        def _decorator(fn):
            return fn

        return _decorator


from backend.app.core.config import Settings
from backend.app.core.country_flags import resolve_destination_flag
from backend.app.schemas.intent import IntentResult, merge_context_patch_into_intent
from backend.app.services.flights_service import get_flights_service
from backend.app.services.fx_service import get_fx_service
from backend.app.services.intent_extractor import IntentExtractor
from backend.app.services.rag_service import get_instance
from backend.app.services.weather_service import get_weather_service
from backend.app.tools.classifier_tool import classify_destinations
from backend.app.tools.live_tools import flight_estimate_tool, fx_latest_tool, weather_forecast_tool
from backend.app.tools.rag_tool import rag_search

logger = structlog.get_logger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are an expert travel planner. Your response MUST follow this EXACT structure.

CRITICAL FORMATTING RULES (DO NOT BREAK):

1. Always suggest 3-5 distinct destinations (numbered ### 1 through ### 3–5). NEVER output only 1 or 2 full options for open-ended trip planning.

2. Open with:
## Recommended Destinations for Your Trip
Then ONE short intro paragraph that restates the user's actual numbers and preferences (budget USD, duration in days, month/season, activities, temperature, tourist density, etc.).

3. For EACH destination you MUST include, in order:
   - A heading line: "### [Number]. [Destination Name], [Country] [emoji flag for country]"
   - flag_emoji must be the Unicode regional-indicator pair for that country (e.g. 🇯🇵 for Japan, 🇫🇷 for France). Never use 🌍 when country is a real sovereign state.
   - "**Why it matches YOUR preferences:**" with 3-4 bullet points that EXPLICITLY quote the user's constraints:
     * Their total budget and/or per-day budget ("fits your $X total / $Y per day" or "would exceed your $Y/day ⚠️")
     * Their duration ("for your N-day trip")
     * Their activities ("matches your interest in …")
     * Their temperature/timing ("warm in [month] — matches your request")
     * Their tourist density ("quieter than …" / "busier but …")
   - "**Estimated costs for YOUR trip:**" with bullets for:
     * Daily budget range → state clearly if it fits or exceeds their implied per-day budget
     * Flight (use origin from tool data; label ✈️; say round-trip estimate)
     * Accommodation per night range
     * Total for N days + flight summary; use ✅ if within budget, ⚠️ if over
   - "**Weather in [specific month or timing from user intent]:**" with:
     * Conditions (rain, sun, monsoon, snow, etc.) — be honest; use 🌞🌧️⛄🌡️ where helpful
     * Temperature range if known from tools; otherwise estimate with uncertainty stated
     * Packing or timing tips if relevant (e.g. monsoon → pack rain gear)
   - "**Best for:**" one concise line (who this option suits)

4. Between destinations insert a line with only: ---

5. Use emojis throughout for scanability: country flags 🇳🇵🇨🇭🇨🇦🇫🇷🇯🇵🇹🇭🇮🇩🇪🇸🇮🇹🇵🇹🇦🇹, weather 🌞🌧️⛄🌡️, money 💰💸, hiking 🥾🏔️🌲, food 🍽️🍜, flight ✈️.

6. End with:
## My Recommendation
Pick a winner, explain why in 2-4 sentences, and name trade-offs (e.g. if another option needs a higher budget).

7. Be HONEST: label budget stress with ⚠️ and good fits with ✅. Say clearly if weather is suboptimal (monsoon, cold snaps, wildfire season, etc.).

8. NEVER use generic filler bullets like "beautiful scenery" without tying them to THIS user's stated preferences and numbers.
8.1 NEVER use boilerplate phrases such as:
- "Selected from classifier results aligned with your trip preferences."
- "Included because it has the strongest available tool-backed fit."
- "Travelers seeking value-for-money options matching your request."
Every bullet must be destination-specific and user-specific.

9. Use tool JSON for costs, weather, and classifier facts. When **plain-text RAG excerpts** appear in the user message before the JSON (retrieved guide passages), you MUST weave at least one concrete, accurate detail from those excerpts into each destination (\"Why it matches\" or \"Best for\"—paraphrase faithfully; do not invent quotes). If no RAG excerpts are present or RAG returned no chunks, say briefly that color comes from live tools and general knowledge. If a tool failed, note it briefly and still give reasoned estimates marked as approximate.

SPECIAL CASE — user names ONE primary place (e.g. "hiking in Kathmandu" with no "where should I go"):
- "### 1." is that place with FULL sections above.
- "### 2." through "### 4" or "### 5" MUST be well-chosen alternatives (nearby or same vibe) with the SAME section headings and honesty rules, so the user still receives 3-5 comparable options.

SPECIAL CASE — user asks to compare two places:
- Give each named place a full "###" block first if applicable, then add 1-3 additional destinations to reach 3-5 total, each with the same structure.
"""

SYNTHESIS_JSON_SCHEMA: dict[str, Any] = {
    "name": "travel_recommendation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "intro": {"type": "string"},
            "month_label": {"type": "string"},
            "destinations": {
                "type": "array",
                "minItems": 3,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "country": {"type": "string"},
                        "flag_emoji": {
                            "type": "string",
                            "description": "Regional-indicator flag for country (e.g. 🇨🇦), not 🌍",
                        },
                        "why_matches": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 5,
                            "items": {"type": "string"},
                        },
                        "daily_budget_line": {"type": "string"},
                        "flight_line": {"type": "string"},
                        "accommodation_line": {"type": "string"},
                        "total_line": {"type": "string"},
                        "weather_line": {"type": "string"},
                        "best_for": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "country",
                        "flag_emoji",
                        "why_matches",
                        "daily_budget_line",
                        "flight_line",
                        "accommodation_line",
                        "total_line",
                        "weather_line",
                        "best_for",
                    ],
                },
            },
            "recommendation_title": {"type": "string"},
            "recommendation_body": {"type": "string"},
        },
        "required": [
            "intro",
            "month_label",
            "destinations",
            "recommendation_title",
            "recommendation_body",
        ],
    },
}


def _normalize_line(text: str) -> str:
    """Collapse whitespace while preserving readable punctuation."""
    return " ".join((text or "").strip().split())


def _clean_weather_line(text: str) -> str:
    t = _normalize_line(text)
    lower = t.lower()
    if lower.startswith("weather in "):
        parts = t.split(":", 1)
        if len(parts) == 2:
            return _normalize_line(parts[1])
    return t


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        canon = re.sub(r"\s+", " ", item.strip().lower())
        canon = re.sub(r"[^a-z0-9 $:/+%().,-]", "", canon)
        if not canon or canon in seen:
            continue
        seen.add(canon)
        out.append(item)
    return out


def _clean_recommendation_title(text: str) -> str:
    t = _normalize_line(text)
    if t.lower() in {"my recommendation", "recommendation"}:
        return "Top pick"
    return t


def _enforce_daily_budget_line(line: str, user_budget_per_day: float | None) -> str:
    base = _normalize_line(line)
    if user_budget_per_day is None:
        return base
    if "fits your $" in base.lower() or "exceed" in base.lower():
        return base
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", base)]
    ref = round(user_budget_per_day, 2)
    if nums:
        avg = sum(nums[:2]) / min(len(nums[:2]), 2)
        ok = avg <= user_budget_per_day
        verdict = "Fits" if ok else "Exceeds"
        mark = "✅" if ok else "⚠️"
        return f"{base} -> {verdict} your ${ref}/day {mark}"
    return f"{base} -> Compare against your ${ref}/day"


def _fallback_destinations_from_tools(
    tool_results: dict[str, Any],
    intent: IntentResult,
    month_label: str,
    user_budget_per_day: float | None,
) -> list[dict[str, Any]]:
    """
    Build deterministic destination fallbacks from classifier + live tool envelopes.
    This prevents placeholder destinations when synthesis JSON omits destinations.
    """
    classifier = (
        tool_results.get("classifier") if isinstance(tool_results.get("classifier"), dict) else {}
    )
    rows = classifier.get("payload", {}).get("destinations", []) if classifier else []
    if not isinstance(rows, list):
        rows = []

    flights_rows = (
        tool_results.get("flights", []) if isinstance(tool_results.get("flights"), list) else []
    )
    weather_rows = (
        tool_results.get("weather", []) if isinstance(tool_results.get("weather"), list) else []
    )

    by_city_flight: dict[str, str] = {}
    for item in flights_rows:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        est = payload.get("estimate") if isinstance(payload.get("estimate"), dict) else {}
        city = str(est.get("destination_display") or est.get("destination_city") or "").strip()
        usd = est.get("round_trip_price_usd_estimate")
        if usd is None:
            usd = est.get("estimated_usd")
        if city:
            by_city_flight[city.lower()] = (
                f"✈️ Flight: Estimated round-trip from NYC ${int(float(usd))}"
                if isinstance(usd, (int, float))
                else "✈️ Flight estimate unavailable"
            )

    by_city_weather: dict[str, str] = {}
    for item in weather_rows:
        if not isinstance(item, dict):
            continue
        city = str(item.get("city") or "").strip()
        env = item.get("envelope") if isinstance(item.get("envelope"), dict) else {}
        payload = env.get("payload") if isinstance(env.get("payload"), dict) else {}
        forecast = payload.get("forecast") if isinstance(payload.get("forecast"), dict) else {}
        summary = ""
        daily = forecast.get("daily") if isinstance(forecast.get("daily"), list) else []
        if daily and isinstance(daily[0], dict):
            first = daily[0]
            cond = str(first.get("conditions_summary") or "").strip()
            tmin = first.get("temp_min_c")
            tmax = first.get("temp_max_c")
            if cond and isinstance(tmin, (int, float)) and isinstance(tmax, (int, float)):
                summary = f"{cond.capitalize()}, around {int(tmin)}-{int(tmax)}°C"
            elif cond:
                summary = cond.capitalize()
        if not summary:
            err = env.get("error") if isinstance(env.get("error"), dict) else {}
            detail = str(err.get("detail") or "").strip()
            if detail:
                summary = f"Forecast unavailable ({detail[:90]})"
        if city:
            by_city_weather[city.lower()] = (
                summary or f"Weather estimate for {month_label} unavailable"
            )

    out: list[dict[str, Any]] = []
    traveler = (intent.traveler_style or "traveler").strip()
    activities = (
        ", ".join(intent.activities[:2]) if intent.activities else "your planned activities"
    )
    temp_pref = intent.temperature_preference or "your preferred climate"
    crowd_pref = intent.tourist_density or "balanced crowd levels"
    for row in rows[:5]:
        if not isinstance(row, dict):
            continue
        city = str(row.get("destination_city") or "").strip()
        country = str(row.get("country") or "").strip() or "Unknown"
        daily = row.get("cost_per_day_avg_usd")
        daily_num = float(daily) if isinstance(daily, (int, float)) else None
        hotel_est = daily_num * 0.55 if daily_num is not None else None

        daily_line = (
            f"Daily budget range: ${int(max(30, daily_num - 20))}-${int(daily_num + 20)}"
            if daily_num is not None
            else "Daily budget: estimate unavailable"
        )
        if user_budget_per_day is not None and daily_num is not None:
            mark = "✅" if daily_num <= user_budget_per_day else "⚠️"
            verdict = "fits" if daily_num <= user_budget_per_day else "exceeds"
            daily_line = f"{daily_line} ({verdict} your ${round(user_budget_per_day,2)}/day {mark})"

        out.append(
            {
                "name": city or "Recommended destination",
                "country": country,
                "flag_emoji": resolve_destination_flag(None, country),
                "why_matches": [
                    f"Matches your {activities} focus for this trip.",
                    f"Fits your {month_label} timing and preference for {temp_pref} conditions.",
                    f"Suitable for a {traveler} and preference for {crowd_pref}.",
                ],
                "daily_budget_line": daily_line,
                "flight_line": by_city_flight.get(city.lower(), "✈️ Flight estimate unavailable"),
                "accommodation_line": (
                    f"Accommodation per night: ~${int(max(25, hotel_est - 15))}-${int(hotel_est + 20)}"
                    if hotel_est is not None
                    else "Accommodation estimate unavailable"
                ),
                "total_line": "Total trip estimate depends on exact dates and booking window.",
                "weather_line": by_city_weather.get(city.lower(), "Weather estimate unavailable"),
                "best_for": f"{traveler.capitalize()} travelers prioritizing {activities}",
            }
        )
    return out


def _render_structured_markdown(
    payload: dict[str, Any],
    user_budget_per_day: float | None,
    fallback_destinations: list[dict[str, Any]] | None = None,
) -> str:
    """Render strict markdown layout from structured synthesis JSON."""
    intro = _normalize_line(str(payload.get("intro") or ""))
    month_label = _normalize_line(str(payload.get("month_label") or "your travel dates"))
    rec_title = _clean_recommendation_title(str(payload.get("recommendation_title") or "Top pick"))
    rec_body = _normalize_line(str(payload.get("recommendation_body") or ""))
    destinations = (
        payload.get("destinations") if isinstance(payload.get("destinations"), list) else []
    )
    destinations = destinations[:5]
    fallback_destinations = fallback_destinations or []
    if len(destinations) < 3:
        for fb in fallback_destinations:
            if len(destinations) >= 5:
                break
            destinations.append(fb)
    while len(destinations) < 3:
        idx = len(destinations) + 1
        destinations.append(
            {
                "name": f"Alternative {idx}",
                "country": "TBD",
                "flag_emoji": "🌍",
                "why_matches": [
                    "Matches your requested activities and trip style.",
                    "Budget-fit details are estimated from available tool data.",
                    "Timing and weather are aligned to your travel window.",
                ],
                "daily_budget_line": "Daily budget: estimate pending → verify against your target budget ⚠️",
                "flight_line": "✈️ Flight: estimate pending",
                "accommodation_line": "Accommodation: estimate pending",
                "total_line": "Total trip estimate: pending data ⚠️",
                "weather_line": "Conditions estimate pending (tool data unavailable)",
                "best_for": "Travelers open to flexible options while data refreshes",
            }
        )

    lines: list[str] = []
    lines.append("## Recommended Destinations for Your Trip")
    lines.append("")
    lines.append(intro or "Based on your preferences, here are suitable options:")
    lines.append("")

    for i, raw in enumerate(destinations, start=1):
        d = raw if isinstance(raw, dict) else {}
        name = _normalize_line(str(d.get("name") or f"Option {i}"))
        country = _normalize_line(str(d.get("country") or ""))
        flag_raw = str(d.get("flag_emoji") or "").strip() or None
        flag = resolve_destination_flag(flag_raw, country)
        daily = _enforce_daily_budget_line(
            str(d.get("daily_budget_line") or "Daily budget: estimate unavailable"),
            user_budget_per_day,
        )
        flight = _normalize_line(str(d.get("flight_line") or "Flight estimate unavailable"))
        accom = _normalize_line(
            str(d.get("accommodation_line") or "Accommodation estimate unavailable")
        )
        total = _normalize_line(str(d.get("total_line") or "Total estimate unavailable"))
        weather = _clean_weather_line(str(d.get("weather_line") or "Weather estimate unavailable"))
        best_for = _normalize_line(str(d.get("best_for") or "General travelers"))
        why_raw = d.get("why_matches") if isinstance(d.get("why_matches"), list) else []
        why_items = [_normalize_line(str(x)) for x in why_raw if _normalize_line(str(x))]
        why_items = [
            x
            for x in why_items
            if "why it matches your preferences" not in x.lower()
            and "matches your preferences" not in x.lower()
        ]
        why_items = _dedupe_preserve_order(why_items)
        if len(why_items) < 3:
            why_items = (why_items + ["Matches your budget and activity preferences."] * 3)[:3]

        lines.append("---")
        lines.append("")
        lines.append(f"### {i}. {name}, {country} {flag}".rstrip())
        lines.append("")
        lines.append("**Why it matches YOUR preferences:**")
        for item in why_items[:5]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Estimated costs for YOUR trip:**")
        lines.append(f"- {daily}")
        lines.append(f"- {flight}")
        lines.append(f"- {accom}")
        lines.append(f"- {total}")
        lines.append("")
        lines.append(f"**Weather in {month_label}:** {weather}")
        lines.append("")
        lines.append(f"**Best for:** {best_for}")
        lines.append("")

    lines.append("## My Recommendation")
    lines.append("")
    if not rec_body and destinations:
        d0 = destinations[0] if isinstance(destinations[0], dict) else {}
        top_name = _normalize_line(str(d0.get("name") or "Top option"))
        top_country = _normalize_line(str(d0.get("country") or ""))
        rec_body = f"{top_name}, {top_country} is the strongest overall fit for your constraints."
    lines.append(f"**{rec_title}** {rec_body}".strip())
    return "\n".join(lines).strip()


def _compact_rag_query(user_query: str, intent: IntentResult) -> str:
    """Intent-enriched, length-capped query so embeddings are not dominated by long chat headers."""
    u = (user_query or "").strip()
    if len(u) > 1200:
        u = f"{u[:1197]}…"
    bits: list[str] = []
    if intent.destination_hint:
        bits.append(str(intent.destination_hint).strip())
    if intent.activities:
        bits.append(", ".join(intent.activities))
    if intent.timing_or_season:
        bits.append(str(intent.timing_or_season).strip())
    if intent.comparison_places:
        bits.append(" vs ".join(intent.comparison_places))
    glue = " · ".join(bits)
    if glue and u:
        return f"{u}\n{glue}"
    return u or glue or "travel planning"


_RAG_COUNTRY_TAIL: frozenset[str] = frozenset(
    {
        "portugal",
        "spain",
        "france",
        "italy",
        "greece",
        "germany",
        "austria",
        "netherlands",
        "belgium",
        "switzerland",
        "poland",
        "czech republic",
        "czechia",
        "uk",
        "united kingdom",
        "usa",
        "u.s.a.",
        "united states",
        "canada",
        "mexico",
        "japan",
        "china",
        "thailand",
        "vietnam",
        "indonesia",
        "australia",
        "new zealand",
        "brazil",
        "argentina",
        "chile",
        "peru",
        "colombia",
        "south africa",
        "egypt",
        "morocco",
        "turkey",
        "croatia",
        "norway",
        "sweden",
        "denmark",
        "ireland",
        "finland",
        "hungary",
        "romania",
        "bulgaria",
        "slovenia",
        "slovakia",
        "estonia",
        "latvia",
        "iceland",
        "malta",
        "cyprus",
        "luxembourg",
        "monaco",
        "andorra",
        "liechtenstein",
        "scotland",
        "england",
        "wales",
        "ireland",
    }
)


def _rag_destination_for_search(intent: IntentResult) -> str | None:
    """
    When set, RAG vector search is filtered to one guide's ``metadata.destination``.
    Use ``None`` for multi-city / compare queries so we search the whole corpus
    (many cities in the repo are not ingested; scoping to e.g. Porto would return zero rows).
    """
    places = [str(p).strip() for p in (intent.comparison_places or []) if str(p).strip()]
    if len(places) >= 2:
        return None
    hint = (intent.destination_hint or "").strip()
    if not hint:
        return None
    low = hint.lower()
    padded = f" {low} "
    if " vs " in low or " versus " in low or " compare " in padded or " compared " in padded:
        return None
    if " between " in padded and len(places) >= 2:
        return None

    parts = [p.strip() for p in hint.split(",") if p.strip()]
    if len(parts) >= 2:
        second = parts[1].lower().strip(".")
        if second in _RAG_COUNTRY_TAIL or second.endswith(" republic"):
            return parts[0].split("(")[0].strip() or None
        return None

    first = hint.split(",")[0].strip().split("(")[0].strip()
    return first or None


def _rag_digest_for_synthesis(
    tool_results: dict[str, Any], *, max_chars: int = 6500, max_blocks: int = 10
) -> str:
    """Plain-text digest so the synthesis model reliably attends to RAG, not only buried JSON."""
    raw = tool_results.get("rag")
    if not isinstance(raw, dict) or raw.get("ok") is not True:
        return ""
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return ""
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        return ""
    intro = (
        "RAG travel-guide excerpts (use concrete facts in your bullets; paraphrase faithfully):\n"
    )
    blocks: list[str] = []
    used = len(intro)
    sep = "\n\n---\n\n"
    for row in chunks[:max_blocks]:
        if not isinstance(row, dict):
            continue
        dest = str(row.get("destination") or "").strip() or "Unknown"
        head = str(row.get("heading") or "").strip() or "Section"
        body = str(row.get("content") or "").strip()
        if not body:
            continue
        excerpt = body[:900] + ("…" if len(body) > 900 else "")
        block = f"[{dest}] {head}\n{excerpt}"
        gap = len(sep) if blocks else 0
        if used + gap + len(block) > max_chars:
            break
        blocks.append(block)
        used += gap + len(block)
    if not blocks:
        return ""
    return intro + sep.join(blocks)


def _configure_langsmith_env(settings: Settings) -> None:
    """
    Ensure LangSmith / LangChain tracing env vars are visible to ``langsmith.traceable``
    (Settings may load from ``.env`` without populating ``os.environ``).
    """
    tracing_on = bool(settings.langchain_tracing_v2 or settings.langsmith_tracing)
    if tracing_on:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
    key = settings.langsmith_api_key or settings.langchain_api_key
    if key:
        os.environ["LANGSMITH_API_KEY"] = key
    proj = settings.langsmith_project or settings.langchain_project
    if proj:
        os.environ["LANGSMITH_PROJECT"] = proj
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    if settings.langchain_project:
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


class TravelAgentGraph:
    """Compiles LangGraph workflow for travel planning."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        _kw: dict[str, Any] = {"api_key": settings.openai_api_key or "dummy"}
        if settings.openai_base_url and str(settings.openai_base_url).strip():
            _kw["base_url"] = str(settings.openai_base_url).strip().rstrip("/")
        self._client = AsyncOpenAI(**_kw)

    def compile(self):
        g = StateGraph(dict)
        g.add_node("extract_intent", self._extract_intent)
        g.add_node("clarify", self._clarify)
        g.add_node("orchestrate_tools", self._orchestrate_tools)
        g.add_node("synthesize", self._synthesize)

        g.add_edge(START, "extract_intent")
        g.add_conditional_edges(
            "extract_intent",
            self._route_after_intent,
            {"clarify": "clarify", "tools": "orchestrate_tools"},
        )
        g.add_edge("clarify", END)
        g.add_edge("orchestrate_tools", "synthesize")
        g.add_edge("synthesize", END)
        return g.compile()

    async def _extract_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        user_query = str(state.get("user_query") or "").strip()
        ext = IntentExtractor(self.settings)
        intent, meta = await ext.extract(user_query)
        raw_patch = state.get("context_patch")
        patch = raw_patch if isinstance(raw_patch, dict) and raw_patch else None
        intent = merge_context_patch_into_intent(intent, patch)
        usage = state.get("usage_parts", []) + [{"step": "intent_extraction", **meta}]
        return {"user_query": user_query, "intent": intent.model_dump(), "usage_parts": usage}

    def _route_after_intent(self, state: dict[str, Any]) -> Literal["clarify", "tools"]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        # Do not OR raw missing_fields — the extractor often leaves stale labels.
        missing = set(intent.critical_missing())
        critical = {"duration", "budget", "activities", "preferred_month"}
        if critical & missing:
            return "clarify"
        return "tools"

    async def _clarify(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        user_query = str(state.get("user_query") or "").strip()
        miss = sorted(set(intent.critical_missing()))
        sys_msg = (
            "You help travelers plan trips. Ask ONE concise clarifying question listing "
            "what you still need (duration in days, total budget USD, main activities)."
        )
        user_msg = f"Missing fields: {', '.join(miss)}. User said: {user_query}"
        if not self.settings.openai_api_key.strip():
            text = (
                "To suggest destinations, I need your trip length (days), approximate total "
                "budget (USD), and what you'd like to do (e.g. hiking, beaches, food)."
            )
            return {
                "user_query": user_query,
                "intent": state.get("intent"),
                "clarification": text,
                "answer": text,
            }

        completion = await self._client.chat.completions.create(
            model=self.settings.openai_cheap_model,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=180,
        )
        text = completion.choices[0].message.content or ""
        usage = state.get("usage_parts", []) + [
            {
                "step": "clarify",
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": completion.usage.completion_tokens
                if completion.usage
                else None,
            }
        ]
        return {
            "user_query": user_query,
            "intent": state.get("intent"),
            "clarification": text,
            "answer": text,
            "usage_parts": usage,
        }

    async def _orchestrate_tools(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        rag = get_instance()
        await rag.startup()
        weather = get_weather_service(
            api_key=self.settings.weather_api_key,
            cache_ttl_seconds=self.settings.weather_cache_ttl_seconds,
        )
        flights = get_flights_service(
            amadeus_api_key=self.settings.amadeus_api_key,
            amadeus_api_secret=self.settings.amadeus_api_secret,
            cache_ttl_seconds=self.settings.flights_cache_ttl_seconds,
        )
        fx = get_fx_service(
            api_key=self.settings.exchange_rate_api_key,
            base_url=self.settings.fx_base_url,
            cache_ttl_seconds=self.settings.fx_cache_ttl_seconds,
        )

        q = str(state.get("user_query") or "").strip()
        if not q:
            parts = [", ".join(intent.activities or []), intent.destination_hint or ""]
            q = " ".join([p for p in parts if p]).strip() or "travel recommendations"

        rag_query = _compact_rag_query(q, intent)

        t0 = time.perf_counter()
        cls_task = classify_destinations(
            self.settings,
            activities=intent.activities or ["general travel"],
            duration_days=intent.duration_days,
            budget_usd=intent.budget_usd,
            destination_hint=intent.destination_hint,
            top_k=8,
        )
        rag_dest = _rag_destination_for_search(intent)
        rag_task = rag_search(rag, query=rag_query, destination=rag_dest, top_k=5)
        fx_task = fx_latest_tool(fx, target_currency="EUR")

        cls_env, rag_env, fx_env = await asyncio.gather(cls_task, rag_task, fx_task)
        bundle: dict[str, Any] = {
            "classifier": cls_env.model_dump(),
            "rag": rag_env.model_dump(),
            "fx": fx_env.model_dump(),
            "weather": [],
            "flights": [],
        }

        cities: list[str] = []
        if cls_env.ok and cls_env.payload:
            for row in cls_env.payload.get("destinations", [])[:5]:
                c = row.get("destination_city")
                if isinstance(c, str):
                    cities.append(c)

        if intent.destination_hint:
            cities.insert(0, intent.destination_hint)

        dedup: list[str] = []
        for c in cities:
            if c not in dedup:
                dedup.append(c)
        cities = dedup[:4]

        async def _one_city(city: str) -> tuple[str, Any]:
            w = await weather_forecast_tool(weather, city=city)
            return city, w.model_dump()

        weather_results = await asyncio.gather(*[_one_city(c) for c in cities[:3]])
        bundle["weather"] = [{"city": c, "envelope": env} for c, env in weather_results]

        origin = self.settings.default_flight_origin
        flight_pairs: list[Any] = []
        for c in cities[:3]:
            flight_pairs.append(
                await flight_estimate_tool(
                    flights,
                    origin_city=origin,
                    destination_city=c,
                    departure_date=None,
                    return_date=None,
                )
            )
        bundle["flights"] = [f.model_dump() for f in flight_pairs]

        dt_ms = int((time.perf_counter() - t0) * 1000)
        bundle["orchestration_ms"] = dt_ms
        logger.info("agent.tools.done", ms=dt_ms, n_cities=len(cities))
        return {"user_query": q, "intent": state.get("intent"), "tool_results": bundle}

    async def _synthesize(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        user_query = str(state.get("user_query") or "").strip()
        tool_results = (
            state.get("tool_results") if isinstance(state.get("tool_results"), dict) else {}
        )
        rag_digest = _rag_digest_for_synthesis(tool_results)
        tools_blob = json.dumps(tool_results, default=str)
        max_tools = 22000 if rag_digest else 28000
        if len(tools_blob) > max_tools:
            tools_blob = tools_blob[:max_tools] + "\n…[tool JSON truncated]"

        per_day: float | None = None
        if intent.budget_usd is not None and intent.duration_days and intent.duration_days > 0:
            per_day = round(float(intent.budget_usd) / float(intent.duration_days), 2)

        personalization_block = f"""USER QUERY (verbatim): {user_query}

NUMBERS TO USE EXACTLY IN YOUR PROSE (do not invent different totals):
- duration_days: {intent.duration_days}
- budget_usd (trip total): {intent.budget_usd}
- implied_budget_per_day_usd: {per_day}
- temperature_preference: {intent.temperature_preference}
- tourist_density: {intent.tourist_density}
- activities: {intent.activities}
- timing_or_season: {intent.timing_or_season}
- destination_hint: {intent.destination_hint}
- comparison_places: {intent.comparison_places}
- must_haves: {intent.must_haves}
- avoid: {intent.avoid}
- traveler_style: {intent.traveler_style}
"""

        rag_section = f"{rag_digest}\n\n" if rag_digest else ""
        user_blob = f"""{personalization_block}
Structured intent (JSON):
{json.dumps(intent.model_dump(), default=str)}

{rag_section}Tool outputs (JSON) — ground your cost and weather claims here:
{tools_blob}
"""

        if not self.settings.openai_api_key.strip():
            reply = (
                "Configure OPENAI_API_KEY to synthesize a full multi-destination answer. "
                "Tool results were collected successfully."
            )
            return {"user_query": user_query, "intent": state.get("intent"), "answer": reply}

        completion = await self._client.chat.completions.create(
            model=self.settings.openai_strong_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_blob},
            ],
            response_format={"type": "json_schema", "json_schema": SYNTHESIS_JSON_SCHEMA},
            max_tokens=4096,
            temperature=0.55,
        )
        raw_json = completion.choices[0].message.content or "{}"
        try:
            structured = json.loads(raw_json)
            month_label = str(
                structured.get("month_label") or intent.timing_or_season or "your travel dates"
            )
            fallback_destinations = _fallback_destinations_from_tools(
                tool_results=tool_results,
                intent=intent,
                month_label=month_label,
                user_budget_per_day=per_day,
            )
            reply = _render_structured_markdown(structured, per_day, fallback_destinations)
        except Exception:
            logger.exception("agent.synthesis.json_parse_failed")
            reply = (
                "## Recommended Destinations for Your Trip\n\n"
                "I had trouble formatting the final recommendation cleanly. "
                "Please retry your request.\n\n"
                "## My Recommendation\n\n"
                "**Retry requested** I can generate the full 3-5 destination plan on the next attempt."
            )
        usage = state.get("usage_parts", []) + [
            {
                "step": "synthesize",
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": completion.usage.completion_tokens
                if completion.usage
                else None,
                "model": self.settings.openai_strong_model,
            }
        ]
        return {
            "user_query": user_query,
            "intent": state.get("intent"),
            "answer": reply,
            "usage_parts": usage,
            "tool_results": state.get("tool_results", {}),
        }


@traceable(name="run_travel_agent", run_type="chain")
async def run_travel_agent(
    settings: Settings,
    user_query: str,
    *,
    context_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute compiled graph and return final state."""
    _configure_langsmith_env(settings)
    agent = TravelAgentGraph(settings)
    graph = agent.compile()
    result = await graph.ainvoke(
        {
            "user_query": user_query.strip(),
            "usage_parts": [],
            "tool_results": {},
            "context_patch": context_patch if context_patch else {},
        }
    )
    return result


async def stream_answer_fallback(settings: Settings, user_query: str) -> AsyncIterator[str]:
    """Yield answer as chunks for SSE (non-streaming API aggregated into lines)."""
    out = await run_travel_agent(settings, user_query)
    answer = out.get("answer") or ""
    for line in answer.splitlines():
        yield line + "\n"


def persist_tool_envelopes_stub(*_: Any, **__: Any) -> None:
    """Reserved for DB persistence hooks from routes."""
    return None
