"""
Structured relevance checks for RAG retrieval (beyond "got HTTP 200").
Each case defines regex patterns expected to appear in returned parent chunks.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.rag_service import get_instance


@dataclass(frozen=True)
class RelevanceCase:
    """Single test case: query + optional filter + content rules."""

    name: str
    query: str
    destination: str | None
    # At least one of these regexes must match (unless only must_match_all is used)
    must_match_any: tuple[str, ...] = ()
    # Each regex here must match somewhere in the combined blob
    must_match_all: tuple[str, ...] = ()
    min_results: int = 1
    expect_empty: bool = False
    # If all of these match, result is considered off-topic (optional guard)
    forbid_if_all_match: tuple[str, ...] = ()


def _blob(results: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for r in results:
        parts.append(str(r.get("heading", "")))
        parts.append(str(r.get("destination", "")))
        parts.append(str(r.get("content", "")))
    return " ".join(parts).lower()


def evaluate_case(case: RelevanceCase, results: list[dict[str, Any]]) -> tuple[bool, str]:
    if case.expect_empty:
        if len(results) == 0:
            return True, "No results (expected)"
        return False, f"Expected empty but got {len(results)} hits"

    if len(results) < case.min_results:
        return False, f"min_results={case.min_results} but got {len(results)}"

    text = _blob(results)
    for pat in case.must_match_all:
        if not re.search(pat, text, re.IGNORECASE):
            return False, f"Missing required pattern {pat!r}; preview={text[:220]!r}"
    if case.must_match_any:
        if not any(re.search(pat, text, re.IGNORECASE) for pat in case.must_match_any):
            return False, f"No pattern in {case.must_match_any}; preview={text[:220]!r}"
    if not case.must_match_any and not case.must_match_all and not case.expect_empty:
        return False, "Case must define must_match_any, must_match_all, or expect_empty"

    if case.forbid_if_all_match:
        if all(re.search(pat, text, re.IGNORECASE) for pat in case.forbid_if_all_match):
            return False, f"Forbidden combination matched: {case.forbid_if_all_match}"

    scores = [r.get("retrieval_score") for r in results if r.get("retrieval_score") is not None]
    score_note = f" scores={scores}" if scores else ""
    return True, f"OK ({len(results)} parents){score_note}"


# Regexes are intentionally broad to survive varied Wikivoyage wording.
CASES: tuple[RelevanceCase, ...] = (
    RelevanceCase("short_hiking_qtown", "hiking", "Queenstown", (r"hik|trail|trek|walk|gondola|route|skyline|ben",)),
    RelevanceCase("short_beaches_maldives", "beaches", "Maldives", (r"beach|coast|sand|atoll|island|lagoon|swim|dive",)),
    RelevanceCase("short_temples_kathmandu", "temples", "Kathmandu", (r"temple|shrine|durbar|stupa|monastery|heritage|unesco",)),
    RelevanceCase(
        "medium_hiking_qtown",
        "best hiking trails in Queenstown",
        "Queenstown",
        (r"hik|trail|trek|walk|gondola|route|skyline|ben|mount",),
    ),
    RelevanceCase(
        "medium_budget_paris",
        "budget accommodation in Paris",
        "Paris",
        (r"hostel|hotel|sleep|budget|cheap|afford|guest",),
    ),
    RelevanceCase(
        "medium_street_food_bangkok",
        "street food in Bangkok",
        "Bangkok",
        (r"food|eat|market|stall|cuisine|restaurant|dish|snack",),
    ),
    RelevanceCase(
        "long_warm_hiking",
        "I have two weeks in July and $1500, want warm weather and hiking, not touristy",
        None,
        must_match_any=(
            r"hik|trail|trek|mount|walk|outdoor|warm|tropical|season|climate|weather|summer|dry|rain|month",
        ),
        min_results=1,
    ),
    RelevanceCase(
        "ambiguous_adventure",
        "where should I go for adventure",
        None,
        (r"adventure|hik|trek|trail|outdoor|ski|climb|raft|dive|activity",),
        min_results=1,
    ),
    RelevanceCase(
        "ambiguous_best_time",
        "best time to visit",
        None,
        (r"month|season|climate|weather|dry|rain|summer|winter|spring|autumn|fall",),
        min_results=1,
    ),
    RelevanceCase(
        "filtered_temples_kathmandu",
        "temples in Kathmandu",
        "Kathmandu",
        (r"temple|shrine|durbar|stupa|heritage|see|monastery",),
    ),
    RelevanceCase(
        "filtered_nightlife_bangkok",
        "nightlife in Bangkok",
        "Bangkok",
        (r"night|bar|club|pub|drink|party|entertain|evening",),
    ),
    RelevanceCase(
        "unfiltered_visa",
        "visa requirements",
        None,
        (r"visa|passport|entry|permit|immigration|arriv",),
    ),
    RelevanceCase(
        "unfiltered_beaches",
        "best beaches",
        None,
        (r"beach|coast|island|sand|lagoon|swim|atoll",),
    ),
    RelevanceCase(
        "gibberish_empty",
        "qq",
        None,
        (),
        min_results=0,
        expect_empty=True,
    ),
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


async def run_all() -> int:
    configure_logging()
    logger = logging.getLogger(__name__)
    service = get_instance()
    await service.startup()
    failed = 0
    try:
        logger.info("Running %s relevance cases", len(CASES))
        for case in CASES:
            try:
                results = await service.search(query=case.query, destination=case.destination, top_k=4)
            except Exception as exc:
                failed += 1
                logger.error("CASE FAIL %s: exception %s", case.name, exc)
                continue
            ok, reason = evaluate_case(case, results)
            if ok:
                logger.info("CASE PASS %s: %s", case.name, reason)
            else:
                failed += 1
                logger.error("CASE FAIL %s: %s", case.name, reason)
                if results:
                    for i, r in enumerate(results[:2]):
                        prev = str(r.get("content", ""))[:160].replace("\n", " ")
                        logger.error("  [%s] %s | %s", i + 1, r.get("heading"), prev)
    finally:
        await service.shutdown()
    return failed


def main() -> None:
    failures = asyncio.run(run_all())
    if failures:
        logging.getLogger(__name__).error("Relevance tests failed: %s case(s)", failures)
        raise SystemExit(1)
    logging.getLogger(__name__).info("All relevance cases passed")


if __name__ == "__main__":
    main()
