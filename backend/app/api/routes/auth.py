"""Register, login, refresh, onboarding."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.deps import db_session_dep, get_current_user, settings_dep
from backend.app.core.config import Settings
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token_safe,
    hash_password,
    verify_password,
)
from backend.app.db.models import User
from backend.app.schemas.auth import (
    AuthResponse,
    OnboardingPatch,
    RefreshRequest,
    TokenPair,
    UserCreate,
    UserLogin,
    UserPublic,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=AuthResponse)
async def register(
    body: UserCreate,
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
) -> AuthResponse:
    existing = (
        await session.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    user = User(
        email=str(body.email).lower().strip(),
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
        onboarding_completed=False,
    )
    session.add(user)
    try:
        await session.commit()
        await session.refresh(user)
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")
    except SQLAlchemyError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable",
        )
    tokens = TokenPair(
        access_token=create_access_token(settings=settings, user_id=user.id, email=user.email),
        refresh_token=create_refresh_token(settings=settings, user_id=user.id),
    )
    return AuthResponse(tokens=tokens, user=_public(user))


@router.post("/login", response_model=AuthResponse)
async def login(
    body: UserLogin,
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
) -> AuthResponse:
    user = (
        await session.execute(select(User).where(User.email == str(body.email).lower().strip()))
    ).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    tokens = TokenPair(
        access_token=create_access_token(settings=settings, user_id=user.id, email=user.email),
        refresh_token=create_refresh_token(settings=settings, user_id=user.id),
    )
    return AuthResponse(tokens=tokens, user=_public(user))


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    body: RefreshRequest,
    session: Annotated[AsyncSession, Depends(db_session_dep)],
    settings: Annotated[Settings, Depends(settings_dep)],
) -> TokenPair:
    payload = decode_token_safe(settings, body.refresh_token)
    if payload is None or payload.get("typ") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    try:
        user_id = UUID(str(sub))
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        )
    user = (await session.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return TokenPair(
        access_token=create_access_token(settings=settings, user_id=user.id, email=user.email),
        refresh_token=create_refresh_token(settings=settings, user_id=user.id),
    )


@router.get("/me", response_model=UserPublic)
async def me(user: Annotated[User, Depends(get_current_user)]) -> UserPublic:
    return _public(user)


@router.patch("/me/onboarding", response_model=UserPublic)
async def patch_onboarding(
    body: OnboardingPatch,
    user: Annotated[User, Depends(get_current_user)],
    session: Annotated[AsyncSession, Depends(db_session_dep)],
) -> UserPublic:
    user.onboarding_completed = body.onboarding_completed
    try:
        await session.commit()
        await session.refresh(user)
    except SQLAlchemyError:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database temporarily unavailable",
        )
    return _public(user)


def _public(user: User) -> UserPublic:
    return UserPublic(
        id=user.id,
        email=user.email,  # type: ignore[arg-type]
        full_name=user.full_name,
        onboarding_completed=user.onboarding_completed,
    )
