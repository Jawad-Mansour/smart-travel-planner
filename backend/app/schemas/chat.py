"""Chat API schemas."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32000)
    session_id: UUID | None = None
    """Optional structured answers from the onboarding / missing-info flow."""
    context_patch: dict[str, Any] | None = None


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=512)


class SessionOut(BaseModel):
    id: UUID
    title: str
    created_at: str | None
    updated_at: str | None


class MessageOut(BaseModel):
    id: UUID
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: str | None
    meta: dict[str, Any] | None = None
