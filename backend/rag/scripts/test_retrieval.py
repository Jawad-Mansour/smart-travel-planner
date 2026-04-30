"""
Phase 11: RAG retrieval smoke tests + parent-chunk verification.

Child chunks are searched only; parents (full sections) are returned.
Relevance threshold comes from RAG_RELEVANCE_THRESHOLD in .env or service default.

Structured regex suite: python backend/rag/scripts/relevance_test.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.rag_service import get_instance

# Minimum parent section body length (characters)
PARENT_BODY_MIN_CHARS = 40

TEST_QUERIES: list[tuple[str, str | None]] = [
    ("hiking", None),
    ("hiking", "Queenstown"),
    ("hiking in Queenstown", "Queenstown"),
    ("beaches", "Maldives"),
    ("temples", "Kathmandu"),
    ("food", "Bangkok"),
    ("visa", None),
    ("best hiking trails in Queenstown", "Queenstown"),
    ("budget accommodation in Paris", "Paris"),
    ("street food in Bangkok", "Bangkok"),
    ("best months to visit tokyo", "Tokyo"),
    ("how to get around with public transport", "Paris"),
    ("best hiking activities nearby", "Queenstown"),
    ("cultural places and temples", "Kathmandu"),
    ("nightlife and food districts", "Bangkok"),
    ("romantic sunset viewpoints", "Santorini"),
    (
        "I have two weeks in July and around $1500. I want somewhere warm, "
        "not too touristy, and I like hiking",
        None,
    ),
    (
        "Tell me about the best food and nightlife in Bangkok for a solo traveler",
        "Bangkok",
    ),
    (
        "What are the must-see temples in Kathmandu and how much time do I need?",
        "Kathmandu",
    ),
    (
        "I'm planning a honeymoon in December. Looking for luxury beach resorts with good food",
        "Maldives",
    ),
    (
        "What's the weather like in Iceland during summer? Can I see the midnight sun?",
        "Reykjavik",
    ),
    ("where should I go for adventure", None),
    ("best time to visit", None),
    ("temples in Kathmandu", "Kathmandu"),
    ("nightlife in Bangkok", "Bangkok"),
    ("visa requirements", None),
    ("best beaches", None),
    ("", None),
    ("xyzabc123 gibberish query that should return nothing", None),
    ("where should I go", None),
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def verify_parent_payload(
    results: list[dict[str, Any]],
    query_label: str,
    relevance_threshold: float,
) -> list[str]:
    """Ensure returned rows look like parent sections; scores above configured threshold."""
    errors: list[str] = []
    for i, r in enumerate(results):
        ctype = r.get("chunk_type")
        if ctype is not None and ctype != "parent":
            errors.append(f"{query_label}[{i}]: chunk_type expected parent, got {ctype!r}")
        content = str(r.get("content", ""))
        if len(content) < PARENT_BODY_MIN_CHARS:
            errors.append(
                f"{query_label}[{i}]: parent body too short ({len(content)} < {PARENT_BODY_MIN_CHARS})"
            )
        score = r.get("retrieval_score")
        if score is not None and float(score) <= relevance_threshold + 1e-9:
            errors.append(
                f"{query_label}[{i}]: retrieval_score {score} should be > {relevance_threshold}"
            )
    return errors


async def run_critical_checks(
    service: Any,
    logger: logging.Logger,
    relevance_threshold: float,
) -> list[str]:
    """Sanity checks — soft where corpus/embeddings vary."""
    errs: list[str] = []

    r1 = await service.search_all_destinations("hiking", top_k=5)
    if r1:
        dests = {str(x.get("destination", "")).lower() for x in r1}
        dests.discard("")
        if len(dests) < 2:
            logger.warning(
                "Multi-dest 'hiking': fewer than 2 distinct destinations in top results (%s) — "
                "often normal if threshold is strict or few destinations embed strongly.",
                dests,
            )
        errs.extend(verify_parent_payload(r1, "multi_hiking", relevance_threshold))

    r2 = await service.search("hiking in Queenstown", destination="Queenstown", top_k=3)
    if r2:
        heads = [(str(x.get("heading", "")).lower(), str(x.get("destination", ""))) for x in r2]
        if all("introduction" in h[0] or h[0] == "general" for h in heads):
            logger.warning(
                "Queenstown hiking: top results are only Introduction — "
                "check chunk data / threshold / re-embed."
            )
        if any(d.strip().lower() != "queenstown" for _, d in r2):
            errs.append(
                f"Queenstown filter leaked other destinations: "
                f"{[(x.get('heading'), x.get('destination')) for x in r2]}"
            )
        errs.extend(verify_parent_payload(r2, "hiking_queenstown", relevance_threshold))

    r3 = await service.search("temples in Kathmandu", destination="Kathmandu", top_k=3)
    if r3:
        blob = " ".join(str(x.get("content", "")) for x in r3).lower()
        if not any(
            k in blob
            for k in (
                "pashupatinath",
                "boudhanath",
                "swayambhu",
                "durbar",
                "temple",
                "stupa",
                "shrine",
                "monastery",
                "heritage",
                "pagoda",
                "unesco",
                "hanuman",
            )
        ):
            logger.warning(
                "Kathmandu temples: no expected keywords in top parent text — "
                "Wikivoyage wording may differ; not a hard failure."
            )
        errs.extend(verify_parent_payload(r3, "temples_kathmandu", relevance_threshold))

    # Short no-signal query — should return empty via gibberish path
    r4 = await service.search("qq", destination=None, top_k=3)
    if r4:
        errs.append(f"Gibberish short query should return 0 parents, got {len(r4)}")

    r5 = await service.search_all_destinations("beaches", top_k=5)
    if r5:
        errs.extend(verify_parent_payload(r5, "beaches_multi", relevance_threshold))

    return errs


async def run() -> None:
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("PHASE 11: RAG RETRIEVAL (parents only)")
    logger.info("=" * 60)

    service = get_instance()
    await service.startup()
    relevance_threshold = float(service.settings.relevance_threshold)

    verification_errors: list[str] = []

    try:
        results_summary: list[dict[str, str | int | bool | None]] = []

        for query, destination in TEST_QUERIES:
            if not query.strip():
                logger.info("Skipping empty query")
                continue

            logger.info("%s", "-" * 50)
            logger.info("Query: %r", query)
            if destination:
                logger.info("Filter: destination=%s", destination)

            try:
                results = await service.search(query=query, destination=destination, top_k=4)
                logger.info("Retrieved %s parent chunk(s)", len(results))
                for i, r in enumerate(results):
                    preview = str(r.get("content", ""))[:220].replace("\n", " ")
                    score = r.get("retrieval_score", "n/a")
                    ctype = r.get("chunk_type", "parent")
                    logger.info(
                        "  [%s] type=%s | %s (%s) score=%s",
                        i + 1,
                        ctype,
                        r.get("heading"),
                        r.get("destination"),
                        score,
                    )
                    logger.info("      %s...", preview)

                verification_errors.extend(
                    verify_parent_payload(results, query[:40], relevance_threshold)
                )

                results_summary.append(
                    {
                        "query": query[:80],
                        "destination": destination,
                        "results_count": len(results),
                        "success": True,
                    }
                )
            except Exception as exc:
                logger.error("Query failed: %s", exc)
                results_summary.append(
                    {
                        "query": query[:80],
                        "destination": destination,
                        "error": str(exc),
                        "success": False,
                    }
                )

        verification_errors.extend(await run_critical_checks(service, logger, relevance_threshold))

        success_count = sum(1 for r in results_summary if r.get("success", False))
        logger.info("")
        logger.info("=" * 60)
        logger.info(
            "SUMMARY: %s/%s queries completed without error",
            success_count,
            len(results_summary),
        )
        if verification_errors:
            for msg in verification_errors:
                logger.error("VERIFICATION: %s", msg)
            raise AssertionError(
                f"Parent verification failed ({len(verification_errors)} issue(s))"
            )
        logger.info("Parent verification: all checks passed")
        logger.info("Regex suite: python backend/rag/scripts/relevance_test.py")
        logger.info("=" * 60)

    finally:
        await service.shutdown()

    logger.info("Smoke testing complete")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Failed")
        raise
