"""Outbound notification helpers (CI-safe, no network)."""

from __future__ import annotations

from backend.app.services.webhook_service import smtp_plan_email_configured


def test_smtp_plan_email_not_configured_when_empty() -> None:
    assert smtp_plan_email_configured("", "") is False
    assert smtp_plan_email_configured(" ", "noreply@x.com") is False


def test_smtp_plan_email_configured_when_host_and_from_present() -> None:
    assert smtp_plan_email_configured("smtp.example.com", "trips@example.com") is True
