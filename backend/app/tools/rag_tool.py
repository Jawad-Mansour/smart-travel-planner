"""
Phase 15: Async RAG tool wrapping :class:`backend.app.services.rag_service.RAGService`.
"""

from __future__ import annotations

import time

import structlog

from backend.app.schemas.tools import ToolEnvelope, ToolError
from backend.app.services.rag_service import RAGService

logger = structlog.get_logger(__name__)


async def rag_search(
    rag: RAGService,
    *,
    query: str,
    destination: str | None = None,
    top_k: int = 5,
) -> ToolEnvelope:
    """
    Retrieve parent-level passages for ``query``, optionally scoped to ``destination``.
    """
    t0 = time.perf_counter()
    tool_name = "rag_search"
    try:
        if destination:
            rows = await rag.search_simple(
                query.strip(), destination=destination.strip(), top_k=top_k
            )
        else:
            rows = await rag.search_all_destinations(query.strip(), top_k=top_k)
        ms = int((time.perf_counter() - t0) * 1000)
        emb_head: list[float] = []
        try:
            emb_head = await rag.embed_text_prefix(query.strip())
        except Exception:
            emb_head = []
        payload = {
            "chunks": rows,
            "duration_ms": ms,
            "query": query.strip(),
            "destination": destination,
            "query_embedding_preview": emb_head,
        }
        logger.info("tool.rag_search.ok", n=len(rows), ms=ms)
        return ToolEnvelope(ok=True, tool=tool_name, payload=payload, error=None)
    except Exception as exc:
        logger.exception("tool.rag_search.error")
        return ToolEnvelope(
            ok=False,
            tool=tool_name,
            payload=None,
            error=ToolError(tool=tool_name, error="rag_search_failed", detail=str(exc)),
        )


async def rag_destination_detail(
    rag: RAGService,
    *,
    query: str,
    destination: str,
    top_k: int = 6,
) -> ToolEnvelope:
    """Focused retrieval for a named destination (higher recall for follow-ups)."""
    return await rag_search(rag, query=query, destination=destination, top_k=top_k)
