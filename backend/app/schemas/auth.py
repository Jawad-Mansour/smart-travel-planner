"""Auth request/response schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserPublic(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str | None
    onboarding_completed: bool


class AuthResponse(BaseModel):
    tokens: TokenPair
    user: UserPublic


class RefreshRequest(BaseModel):
    refresh_token: str


class OnboardingPatch(BaseModel):
    onboarding_completed: bool = True
