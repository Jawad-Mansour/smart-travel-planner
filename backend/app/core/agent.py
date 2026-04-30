"""
Phases 17–22: LangGraph agent — intent extraction, clarification, parallel tools, synthesis.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Literal

import structlog
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from backend.app.core.config import Settings
from backend.app.schemas.intent import IntentResult
from backend.app.services.flights_service import get_flights_service
from backend.app.services.fx_service import get_fx_service
from backend.app.services.intent_extractor import IntentExtractor
from backend.app.services.rag_service import RAGService
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

9. Use tool JSON below for costs, weather, and destination facts. If a tool failed, say so briefly and still give reasoned estimates marked as approximate.

SPECIAL CASE — user names ONE primary place (e.g. "hiking in Kathmandu" with no "where should I go"):
- "### 1." is that place with FULL sections above.
- "### 2." through "### 4" or "### 5" MUST be well-chosen alternatives (nearby or same vibe) with the SAME section headings and honesty rules, so the user still receives 3-5 comparable options.

SPECIAL CASE — user asks to compare two places:
- Give each named place a full "###" block first if applicable, then add 1-3 additional destinations to reach 3-5 total, each with the same structure.
"""


class TravelAgentGraph:
    """Compiles LangGraph workflow for travel planning."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "dummy")

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
        ext = IntentExtractor(self.settings)
        intent, meta = await ext.extract(state["user_query"])
        usage = state.get("usage_parts", []) + [{"step": "intent_extraction", **meta}]
        return {"intent": intent.model_dump(), "usage_parts": usage}

    def _route_after_intent(self, state: dict[str, Any]) -> Literal["clarify", "tools"]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        missing = set(intent.critical_missing()) | set(intent.missing_fields)
        critical = {"duration", "budget", "activities"}
        if critical & missing:
            return "clarify"
        return "tools"

    async def _clarify(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        miss = list(set(intent.critical_missing()) | set(intent.missing_fields))
        sys_msg = (
            "You help travelers plan trips. Ask ONE concise clarifying question listing "
            "what you still need (duration in days, total budget USD, main activities)."
        )
        user_msg = f"Missing fields: {miss}. User said: {state['user_query']}"
        if not self.settings.openai_api_key.strip():
            text = (
                "To suggest destinations, I need your trip length (days), approximate total "
                "budget (USD), and what you'd like to do (e.g. hiking, beaches, food)."
            )
            return {"clarification": text, "answer": text}

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
                "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
            }
        ]
        return {"clarification": text, "answer": text, "usage_parts": usage}

    async def _orchestrate_tools(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        rag = RAGService.get_instance()
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

        q = state["user_query"]
        activities_text = ", ".join(intent.activities) if intent.activities else q[:400]

        t0 = time.perf_counter()
        cls_task = classify_destinations(
            self.settings,
            activities=intent.activities or ["general travel"],
            duration_days=intent.duration_days,
            budget_usd=intent.budget_usd,
            destination_hint=intent.destination_hint,
            top_k=8,
        )
        rag_task = rag_search(rag, query=q, destination=intent.destination_hint, top_k=5)
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
        return {"tool_results": bundle}

    async def _synthesize(self, state: dict[str, Any]) -> dict[str, Any]:
        intent = IntentResult.model_validate(state.get("intent", {}))
        tools_blob = json.dumps(state.get("tool_results", {}), default=str)[:28000]

        per_day: float | None = None
        if intent.budget_usd is not None and intent.duration_days and intent.duration_days > 0:
            per_day = round(float(intent.budget_usd) / float(intent.duration_days), 2)

        personalization_block = f"""USER QUERY (verbatim): {state["user_query"]}

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

        user_blob = f"""{personalization_block}
Structured intent (JSON):
{json.dumps(intent.model_dump(), default=str)}

Tool outputs (JSON) — ground your cost and weather claims here:
{tools_blob}
"""

        if not self.settings.openai_api_key.strip():
            reply = (
                "Configure OPENAI_API_KEY to synthesize a full multi-destination answer. "
                "Tool results were collected successfully."
            )
            return {"answer": reply}

        completion = await self._client.chat.completions.create(
            model=self.settings.openai_strong_model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_blob},
            ],
            max_tokens=4096,
            temperature=0.55,
        )
        reply = completion.choices[0].message.content or ""
        usage = state.get("usage_parts", []) + [
            {
                "step": "synthesize",
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
                "model": self.settings.openai_strong_model,
            }
        ]
        return {"answer": reply, "usage_parts": usage}


async def run_travel_agent(settings: Settings, user_query: str) -> dict[str, Any]:
    """Execute compiled graph and return final state."""
    agent = TravelAgentGraph(settings)
    graph = agent.compile()
    result = await graph.ainvoke({"user_query": user_query.strip(), "usage_parts": [], "tool_results": {}})
    return result


async def stream_answer_fallback(
    settings: Settings, user_query: str
) -> AsyncIterator[str]:
    """Yield answer as chunks for SSE (non-streaming API aggregated into lines)."""
    out = await run_travel_agent(settings, user_query)
    answer = out.get("answer") or ""
    for line in answer.splitlines():
        yield line + "\n"


def persist_tool_envelopes_stub(*_: Any, **__: Any) -> None:
    """Reserved for DB persistence hooks from routes."""
    return None
