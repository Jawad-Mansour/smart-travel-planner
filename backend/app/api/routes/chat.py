"""SSE chat stream with persistence and optional outbound webhooks."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.chat_markdown_split import split_travel_answer_segments
from backend.app.api.deps import db_session_dep, get_current_user, settings_dep
from backend.app.api.routes.sessions import get_owned_session
from backend.app.core.agent import run_travel_agent
from backend.app.core.config import Settings
from backend.app.db.models import ChatMessage, ChatSession, ToolCallLog, User
from backend.app.schemas.chat import ChatStreamRequest
from backend.app.schemas.intent import IntentResult
from backend.app.services.webhook_service import (
    notify_discord_plan_ready,
    notify_email_plan_ready,
    notify_slack_plan_ready,
    plan_ready_email_configured,
)

router = APIRouter(prefix="/chat", tags=["chat"])
_log = structlog.get_logger(__name__)

SEGMENT_PAUSE_SEC = 0.35


def _compose_query(message: str, patch: dict[str, Any] | None) -> str:
    m = message.strip()
    if not patch:
        return m
    lines = [m, "", "[The traveler also provided the following missing details:]"]
    if v := patch.get("budget_usd") or patch.get("budget"):
        lines.append(f"- Trip budget (USD total): {v}")
    if v := patch.get("duration_days") or patch.get("length_days"):
        lines.append(f"- Trip length (days): {v}")
    if v := patch.get("timing_or_season") or patch.get("preferred_month"):
        lines.append(f"- Preferred timing / month: {v}")
    if v := patch.get("activities"):
        if isinstance(v, list):
            lines.append(f"- Interests / activities: {', '.join(str(x) for x in v)}")
        else:
            lines.append(f"- Interests / activities: {v}")
    return "\n".join(lines)


def _sse(obj: dict[str, Any]) -> str:
    return "data: " + json.dumps(obj, default=str) + "\n\n"


def _chunks(text: str, size: int = 18) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _rag_embedding_preview(tools: dict[str, Any]) -> list[float]:
    raw = tools.get("rag")
    if not isinstance(raw, dict):
        return []
    payload = raw.get("payload")
    if not isinstance(payload, dict):
        return []
    vec = payload.get("query_embedding_preview")
    if not isinstance(vec, list):
        return []
    out: list[float] = []
    for x in vec[:10]:
        try:
            out.append(round(float(x), 5))
        except (TypeError, ValueError):
            continue
    return out


async def _persist_tool_logs(
    db: AsyncSession,
    session_row: ChatSession,
    message_row: ChatMessage,
    tool_results: dict[str, Any],
) -> None:
    for key, val in tool_results.items():
        if key == "orchestration_ms":
            continue
        db.add(
            ToolCallLog(
                session_id=session_row.id,
                message_id=message_row.id,
                tool_name=str(key),
                input_json={"tool": str(key)},
                output_json=val if isinstance(val, (dict, list)) else {"value": val},
                duration_ms=None,
            )
        )


@router.post("/stream")
async def chat_stream(
    body: ChatStreamRequest,
    background_tasks: BackgroundTasks,
    db: Annotated[AsyncSession, Depends(db_session_dep)],
    user: Annotated[User, Depends(get_current_user)],
    settings: Annotated[Settings, Depends(settings_dep)],
) -> StreamingResponse:
    query = _compose_query(body.message, body.context_patch)

    async def event_gen() -> AsyncIterator[str]:
        try:
            session_row: ChatSession | None = None
            if body.session_id is not None:
                session_row = await get_owned_session(db, user.id, body.session_id)
                if session_row is None:
                    yield _sse({"type": "error", "detail": "session_not_found"})
                    return
            else:
                raw_title = (body.message.strip()[:80] or "New trip").replace("\n", " ")
                session_row = ChatSession(user_id=user.id, title=raw_title)
                db.add(session_row)
                await db.flush()

            assert session_row is not None
            user_msg = ChatMessage(
                session_id=session_row.id, role="user", content=body.message.strip()
            )
            db.add(user_msg)
            session_row.updated_at = datetime.now(tz=UTC)
            await db.commit()
            await db.refresh(session_row)
            await db.refresh(user_msg)

            yield _sse({"type": "session", "session_id": str(session_row.id)})

            t_agent = time.perf_counter()
            result = await run_travel_agent(settings, query)
            elapsed_agent = round(time.perf_counter() - t_agent, 2)
            answer = str(result.get("answer") or "")
            raw_intent = result.get("intent")
            try:
                intent_obj = (
                    IntentResult.model_validate(raw_intent)
                    if isinstance(raw_intent, dict)
                    else IntentResult()
                )
            except ValidationError:
                intent_obj = IntentResult()
            missing = sorted(set(intent_obj.critical_missing()) | set(intent_obj.missing_fields))
            is_clarification = bool(result.get("clarification"))
            needs_clarification = bool(missing) and is_clarification

            yield _sse(
                {
                    "type": "meta",
                    "needs_clarification": needs_clarification,
                    "missing_fields": missing if needs_clarification else [],
                    "intent": raw_intent,
                }
            )

            tools = (
                result.get("tool_results") if isinstance(result.get("tool_results"), dict) else {}
            )
            emb_preview = _rag_embedding_preview(tools)

            seg_list: list[str] | None = None
            if not needs_clarification and answer.strip():
                seg_list = split_travel_answer_segments(answer)

            if seg_list and len(seg_list) >= 2:
                for i, seg in enumerate(seg_list):
                    if i:
                        await asyncio.sleep(SEGMENT_PAUSE_SEC)
                    yield _sse({"type": "segment", "content": seg})
            else:
                for part in _chunks(answer):
                    yield _sse({"type": "delta", "content": part})
                    await asyncio.sleep(0)

            assistant = ChatMessage(
                session_id=session_row.id,
                role="assistant",
                content=answer,
                meta_json={
                    "usage": result.get("usage_parts"),
                    "needs_clarification": needs_clarification,
                    "missing_fields": missing if needs_clarification else [],
                },
            )
            db.add(assistant)
            session_row.updated_at = datetime.now(tz=UTC)
            await db.flush()
            await _persist_tool_logs(db, session_row, assistant, tools)
            await db.commit()
            await db.refresh(assistant)

            has_hook = bool(
                (settings.discord_webhook_url or "").strip()
                or (settings.slack_webhook_url or "").strip()
                or plan_ready_email_configured(settings)
            )
            if not has_hook:
                webhook_status = "not_configured"
            elif needs_clarification or not answer.strip():
                webhook_status = "skipped_clarification"
            else:
                webhook_status = "queued"

            yield _sse(
                {
                    "type": "done",
                    "message_id": str(assistant.id),
                    "session_id": str(session_row.id),
                    "needs_clarification": needs_clarification,
                    "missing_fields": missing if needs_clarification else [],
                    "elapsed_seconds": elapsed_agent,
                    "intent": raw_intent,
                    "tool_results": tools,
                    "usage_parts": result.get("usage_parts"),
                    "webhook_status": webhook_status,
                    "query_embedding_preview": emb_preview,
                }
            )

            if not needs_clarification and answer.strip():
                background_tasks.add_task(
                    notify_discord_plan_ready,
                    settings,
                    user_email=user.email,
                    session_title=session_row.title,
                    answer_preview=answer[:1200],
                )
                background_tasks.add_task(
                    notify_slack_plan_ready,
                    settings,
                    user_email=user.email,
                    session_title=session_row.title,
                    answer_preview=answer[:1200],
                )
                background_tasks.add_task(
                    notify_email_plan_ready,
                    settings,
                    user_email=user.email,
                    session_title=session_row.title,
                    answer_preview=answer[:1200],
                )
        except Exception as exc:
            _log.exception("chat_stream_failed", error=str(exc))
            try:
                await db.rollback()
            except Exception:
                pass
            yield _sse(
                {
                    "type": "error",
                    "detail": "Could not complete the reply. Please try again.",
                }
            )

    return StreamingResponse(event_gen(), media_type="text/event-stream")
