"""Travel planning API — SSE stream + history."""

from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import (
    current_user_sub,
    db_session_dep,
    settings_dep,
)
from backend.app.core.agent import run_travel_agent
from backend.app.core.config import Settings
from backend.app.db.models import AgentRun, ToolCall
from backend.app.schemas.intent import IntentResult

router = APIRouter(prefix="/travel", tags=["travel"])


@router.post("/plan")
async def plan_travel(
    body: dict[str, Any],
    settings: Annotated[Settings, Depends(settings_dep)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    user_sub: Annotated[str, Depends(current_user_sub)],
) -> StreamingResponse:
    """SSE stream with final JSON payload (demo-friendly single-shot)."""
    q = str(body.get("query") or body.get("message") or "").strip()
    if not q:

        async def err_gen():
            yield "data: " + json.dumps({"error": "query required"}) + "\n\n"

        return StreamingResponse(err_gen(), media_type="text/event-stream")

    async def event_gen():
        result = await run_travel_agent(settings, q)
        answer = result.get("answer") or ""
        usage = result.get("usage_parts") or []
        raw_intent = result.get("intent")
        intent_obj = (
            IntentResult.model_validate(raw_intent)
            if isinstance(raw_intent, dict)
            else IntentResult()
        )
        missing = sorted(set(intent_obj.critical_missing()) | set(intent_obj.missing_fields))
        is_clarification = bool(result.get("clarification"))
        payload = {
            "answer": answer,
            "usage": usage,
            "intent": raw_intent,
            "needs_clarification": bool(missing) and is_clarification,
            "missing_fields": missing if is_clarification else [],
        }
        yield "data: " + json.dumps(payload, default=str) + "\n\n"

        tools_blob = (
            result.get("tool_results") if isinstance(result.get("tool_results"), dict) else {}
        )
        orch_ms = tools_blob.get("orchestration_ms") if isinstance(tools_blob, dict) else None

        run = AgentRun(
            user_sub=user_sub,
            query=q,
            intent_json=result.get("intent"),
            answer=answer,
            usage_json={"parts": usage, "tool_summary": True},
        )
        session.add(run)
        await session.flush()

        session.add(
            ToolCall(
                run_id=run.id,
                tool_name="orchestrate_parallel",
                input_json={"query": q},
                output_json=tools_blob,
                duration_ms=orch_ms if isinstance(orch_ms, int) else None,
            )
        )
        await session.commit()

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/history")
async def travel_history(
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    user_sub: Annotated[str, Depends(current_user_sub)],
    limit: int = 20,
) -> dict[str, Any]:
    from sqlalchemy import select

    stmt = (
        select(AgentRun)
        .where(AgentRun.user_sub == user_sub)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return {
        "items": [
            {
                "id": str(r.id),
                "query": r.query,
                "answer": r.answer,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    }
