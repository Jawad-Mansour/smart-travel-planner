"""Password hashing and JWT helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import bcrypt
from jose import JWTError, jwt

from backend.app.core.config import Settings


def hash_password(password: str) -> str:
    """Hash with bcrypt (native library; avoids passlib + bcrypt 5.x incompatibility)."""
    pw = password.encode("utf-8")
    if len(pw) > 72:
        pw = pw[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        p = plain.encode("utf-8")
        if len(p) > 72:
            p = p[:72]
        return bcrypt.checkpw(p, hashed.encode("ascii"))
    except (ValueError, TypeError):
        return False


def create_access_token(
    *,
    settings: Settings,
    user_id: UUID,
    email: str,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    expire = datetime.now(tz=UTC) + timedelta(minutes=settings.jwt_access_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "typ": "access",
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        payload.update(extra_claims)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(*, settings: Settings, user_id: UUID) -> str:
    expire = datetime.now(tz=UTC) + timedelta(days=settings.jwt_refresh_expire_days)
    payload = {"sub": str(user_id), "typ": "refresh", "exp": int(expire.timestamp())}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(settings: Settings, token: str) -> dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def decode_token_safe(settings: Settings, token: str) -> dict[str, Any] | None:
    try:
        return decode_token(settings, token)
    except JWTError:
        return None
