"""Shared tool boundary schemas (structured successes and failures)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    """Structured tool failure consumed by the agent (does not raise)."""

    tool: str
    error: str
    detail: str | None = None


class ToolEnvelope(BaseModel):
    """Either payload data or a structured error."""

    ok: bool
    tool: str
    payload: dict[str, Any] | None = None
    error: ToolError | None = None
