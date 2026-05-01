"""Pydantic models for structured travel intent extraction."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IntentResult(BaseModel):
    """Structured output from the cheap extraction model."""

    duration_days: int | None = Field(default=None, ge=1, le=365)
    budget_usd: float | None = Field(default=None, ge=0)
    temperature_preference: Literal["warm", "cool", "mild", "any"] | None = None
    tourist_density: Literal["quiet", "moderate", "busy", "any"] | None = None
    activities: list[str] = Field(default_factory=list)
    destination_hint: str | None = None
    timing_or_season: str | None = Field(
        default=None,
        description="Month, season, or holiday window (e.g. July, summer, Christmas week)",
    )
    comparison_places: list[str] = Field(
        default_factory=list,
        description="Named places user wants compared (e.g. Bali, Thailand)",
    )
    must_haves: list[str] = Field(
        default_factory=list,
        description="Non-negotiables inferred from the message",
    )
    avoid: list[str] = Field(
        default_factory=list,
        description="Things to avoid (e.g. crowds, heat, long flights)",
    )
    traveler_style: str | None = Field(
        default=None,
        description="Short label e.g. solo backpacker, family with kids, luxury",
    )
    missing_fields: list[str] = Field(
        default_factory=list,
        description="Critical gaps: duration, budget, activities",
    )

    def critical_missing(self) -> list[str]:
        """Fields the agent should clarify before heavy tool orchestration."""
        needed: list[str] = []
        if self.duration_days is None:
            needed.append("duration")
        if self.budget_usd is None:
            needed.append("budget")
        if not self.activities:
            needed.append("activities")
        if not (self.timing_or_season or "").strip():
            needed.append("preferred_month")
        return needed


def merge_context_patch_into_intent(
    intent: IntentResult, patch: dict[str, Any] | None
) -> IntentResult:
    """
    Apply structured values from the Quick details form (or API context_patch).

    Always prunes ``missing_fields`` against ``critical_missing()`` so the extractor
    cannot keep stale labels (e.g. "budget" after ``budget_usd`` is set).
    """
    if patch:
        data = intent.model_dump()

        def _set_budget(v: Any) -> None:
            try:
                f = float(v)
                if f >= 0:
                    data["budget_usd"] = f
            except (TypeError, ValueError):
                pass

        if patch.get("budget_usd") is not None:
            _set_budget(patch.get("budget_usd"))
        elif patch.get("budget") is not None:
            _set_budget(patch.get("budget"))

        def _set_duration(v: Any) -> None:
            try:
                n = int(v)
                if 1 <= n <= 365:
                    data["duration_days"] = n
            except (TypeError, ValueError):
                pass

        if patch.get("duration_days") is not None:
            _set_duration(patch.get("duration_days"))
        elif patch.get("length_days") is not None:
            _set_duration(patch.get("length_days"))

        acts = patch.get("activities")
        if acts is not None:
            if isinstance(acts, list):
                merged = [str(x).strip() for x in acts if str(x).strip()]
                if merged:
                    data["activities"] = merged
            elif isinstance(acts, str) and acts.strip():
                data["activities"] = [s.strip() for s in acts.split(",") if s.strip()]

        timing = patch.get("timing_or_season") or patch.get("preferred_month")
        if timing is not None and str(timing).strip():
            data["timing_or_season"] = str(timing).strip()

        out = IntentResult.model_validate(data)
    else:
        out = intent

    crit = set(out.critical_missing())
    pruned = sorted({x for x in out.missing_fields if x in crit})
    return out.model_copy(update={"missing_fields": pruned})
