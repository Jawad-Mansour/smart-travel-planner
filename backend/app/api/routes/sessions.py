"""Chat sessions CRUD for the authenticated user."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import db_session_dep, get_current_user
from backend.app.db.models import ChatMessage, ChatSession, User
from backend.app.schemas.chat import MessageOut, SessionCreate, SessionOut

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    limit: int = 50,
) -> list[SessionOut]:
    stmt = (
        select(ChatSession)
        .where(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc())
        .limit(min(limit, 100))
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [_session_out(r) for r in rows]


@router.post("", response_model=SessionOut, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: SessionCreate,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
) -> SessionOut:
    title = (body.title or "New trip").strip()[:512] or "New trip"
    row = ChatSession(user_id=user.id, title=title)
    session.add(row)
    try:
        await session.commit()
        await session.refresh(row)
    except SQLAlchemyError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable",
        )
    return _session_out(row)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
) -> Response:
    cs = await get_owned_session(session, user.id, session_id)
    if cs is None:
        raise HTTPException(status_code=404, detail="Session not found")
    await session.execute(delete(ChatSession).where(ChatSession.id == cs.id))
    try:
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{session_id}/messages", response_model=list[MessageOut])
async def list_messages(
    session_id: UUID,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
) -> list[MessageOut]:
    cs = await get_owned_session(session, user.id, session_id)
    if cs is None:
        raise HTTPException(status_code=404, detail="Session not found")
    stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == cs.id)
        .order_by(ChatMessage.created_at.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [_msg_out(m) for m in rows]


async def get_owned_session(
    db: AsyncSession, user_id: UUID, session_id: UUID
) -> ChatSession | None:
    row = (
        await db.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
    ).scalar_one_or_none()
    return row


def _session_out(r: ChatSession) -> SessionOut:
    return SessionOut(
        id=r.id,
        title=r.title,
        created_at=r.created_at.isoformat() if r.created_at else None,
        updated_at=r.updated_at.isoformat() if r.updated_at else None,
    )


def _msg_out(m: ChatMessage) -> MessageOut:
    role = m.role if m.role in ("user", "assistant", "system") else "assistant"
    return MessageOut(
        id=m.id,
        role=role,  # type: ignore[arg-type]
        content=m.content,
        created_at=m.created_at.isoformat() if m.created_at else None,
        meta=m.meta_json,
    )
