"""
Phase 16: Cheap-model structured intent extraction with token accounting.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from openai import AsyncOpenAI

from backend.app.core.config import Settings
from backend.app.schemas.intent import IntentResult

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You extract structured travel planning fields from the user's message.
Infer reasonable values when clearly implied. Capture ALL preferences you can detect:
- Trip length in days, total budget USD, timing/month/season, activities, temperature and crowd preferences
- Named destinations (destination_hint), places to compare (comparison_places), must-haves and avoidances
- Traveler style (solo, couple, family, luxury, backpacker, etc.)

Use missing_fields only for critical gaps you truly cannot infer:
- duration (days) when not inferable
- budget (total USD) when not inferable
- activities when none are stated or implied

Respond with JSON only using keys:
duration_days (int or null), budget_usd (number or null),
temperature_preference ("warm"|"cool"|"mild"|"any" or null),
tourist_density ("quiet"|"moderate"|"busy"|"any" or null),
activities (array of strings),
destination_hint (string or null),
timing_or_season (string or null),
comparison_places (array of strings),
must_haves (array of strings),
avoid (array of strings),
traveler_style (string or null),
missing_fields (array of strings).
"""


class IntentExtractor:
    """Async OpenAI wrapper for IntentResult extraction."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = AsyncOpenAI(api_key=settings.openai_api_key or "dummy")

    async def extract(self, user_query: str) -> tuple[IntentResult, dict[str, Any]]:
        """
        Returns (intent, usage_metadata) where usage_metadata includes token counts when available.
        """
        if not self._settings.openai_api_key.strip():
            logger.warning("intent_extractor.no_api_key")
            ir = IntentResult(
                missing_fields=["duration", "budget", "activities"],
                activities=[],
            )
            return ir, {
                "source": "fallback_no_api_key",
                "total_tokens": 0,
                "note": "Set OPENAI_API_KEY for LLM-based extraction.",
            }

        try:
            completion = await self._client.chat.completions.create(
                model=self._settings.openai_cheap_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_query.strip()},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
        except Exception as exc:
            logger.exception("intent_extractor.openai_error", error=str(exc))
            ir = IntentResult(
                missing_fields=["duration", "budget", "activities"],
                activities=[],
            )
            return ir, {
                "source": "fallback_openai_error",
                "total_tokens": 0,
                "detail": str(exc),
            }

        raw = completion.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            validated = IntentResult.model_validate(data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("intent_extractor.parse_failed", detail=str(exc))
            validated = IntentResult(
                missing_fields=["duration", "budget", "activities"],
                activities=[],
            )

        usage = completion.usage
        meta: dict[str, Any] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "model": self._settings.openai_cheap_model,
        }
        logger.info("intent_extractor.ok", **{k: v for k, v in meta.items() if v is not None})
        return validated, meta


def get_intent_extractor(settings: Settings) -> IntentExtractor:
    return IntentExtractor(settings)
