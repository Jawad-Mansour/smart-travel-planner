"""Pydantic models for structured travel intent extraction."""

from __future__ import annotations

from typing import Literal

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
        return needed
