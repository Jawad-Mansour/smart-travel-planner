"""JSON-friendly structlog configuration for request and agent logging."""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(json_logs: bool = True) -> None:
    """Call once from FastAPI lifespan or ``main``."""
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors = shared + [structlog.processors.JSONRenderer()]
    else:
        processors = shared + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
