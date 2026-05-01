"""Outbound notifications when a plan is ready — failures never break the user response."""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any

import httpx
import structlog
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from backend.app.core.config import Settings

logger = structlog.get_logger(__name__)


def plan_ready_email_configured(settings: Settings) -> bool:
    return bool((settings.smtp_host or "").strip() and (settings.smtp_from or "").strip())


async def notify_slack_plan_ready(
    settings: Settings,
    *,
    user_email: str,
    session_title: str,
    answer_preview: str,
) -> None:
    url = (settings.slack_webhook_url or "").strip()
    if not url:
        return
    preview = (answer_preview or "").strip().replace("\n", " ")[:1500]
    payload = {
        "text": f"*Travel plan ready* — {session_title}\nUser: {user_email}\n\n{preview}",
    }
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
    except Exception as exc:
        logger.warning("webhook.slack_failed", error=str(exc))


async def notify_discord_plan_ready(
    settings: Settings,
    *,
    user_email: str,
    session_title: str,
    answer_preview: str,
) -> None:
    url = (settings.discord_webhook_url or "").strip()
    if not url:
        return
    preview = (answer_preview or "").strip().replace("\n", " ")[:1800]
    payload: dict[str, Any] = {
        "content": None,
        "embeds": [
            {
                "title": "Travel plan ready",
                "description": f"**{session_title}**\n\n{preview}",
                "fields": [{"name": "User", "value": user_email, "inline": True}],
                "color": 3447003,
            }
        ],
    }
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(url, json=payload)
                    r.raise_for_status()
    except Exception as exc:
        logger.warning("webhook.discord_failed", error=str(exc))


def _send_plan_ready_email_sync(
    *,
    host: str,
    port: int,
    user: str,
    password: str,
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
    use_tls: bool,
    use_ssl: bool,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    ctx = ssl.create_default_context()

    if use_ssl:
        with smtplib.SMTP_SSL(host, port, context=ctx) as smtp:
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return

    with smtplib.SMTP(host, port) as smtp:
        if use_tls:
            smtp.starttls(context=ctx)
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)


async def notify_email_plan_ready(
    settings: Settings,
    *,
    user_email: str,
    session_title: str,
    answer_preview: str,
) -> None:
    if not plan_ready_email_configured(settings):
        return
    host = (settings.smtp_host or "").strip()
    from_addr = (settings.smtp_from or "").strip()
    to_addr = (user_email or "").strip()
    if not to_addr:
        return

    preview = (answer_preview or "").strip()
    subject = f"Your travel plan is ready — {session_title}"
    body = (
        f"Hi,\n\n"
        f'Your itinerary for "{session_title}" is ready in Smart Travel Planner.\n\n'
        f"---\n{preview}\n---\n\n"
        f"(This message was sent because SMTP is configured on the server.)\n"
    )

    try:
        await asyncio.to_thread(
            _send_plan_ready_email_sync,
            host=host,
            port=int(settings.smtp_port),
            user=(settings.smtp_user or "").strip(),
            password=(settings.smtp_password or "").strip(),
            from_addr=from_addr,
            to_addr=to_addr,
            subject=subject,
            body=body[:20000],
            use_tls=settings.smtp_use_tls and not settings.smtp_use_ssl,
            use_ssl=settings.smtp_use_ssl,
        )
    except Exception as exc:
        logger.warning("webhook.email_failed", error=str(exc), to=to_addr)
