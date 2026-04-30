"""
Run many query types against RAG (smoke). Uses project-root import path (host or Docker).

Run from repo root:
  python backend/rag/scripts/test_all_queries.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.rag_service import get_instance


async def test() -> None:
    service = get_instance()
    await service.startup()

    queries: list[tuple[str, str | None]] = [
        ("hiking", None),
        ("beaches", None),
        ("temples", None),
        ("hiking in Queenstown", "Queenstown"),
        ("temples in Kathmandu", "Kathmandu"),
        ("nightlife in Bangkok", "Bangkok"),
        ("food in Paris", "Paris"),
        ("visa requirements", None),
        ("best beaches", None),
        ("budget accommodation", None),
        ("where should I go for adventure", None),
        ("best time to visit", None),
        ("recommend a place for honeymoon", None),
        (
            "I have two weeks in July and $1500, want warm weather and hiking",
            None,
        ),
        ("Tell me about temples and culture in Nepal", "Kathmandu"),
        ("xyzabc123 gibberish", None),
    ]

    print("=" * 70)
    print("RAG RETRIEVAL TEST - ALL QUERY TYPES")
    print("=" * 70)

    for query, dest in queries:
        if not query.strip():
            continue

        print(f'\nQuery: "{query[:60]}"')
        if dest:
            print(f"   Filter: {dest}")

        results = await service.search(query, destination=dest, top_k=3)

        if results:
            print(f"   OK: {len(results)} parent(s)")
            for r in results[:2]:
                preview = str(r["content"])[:100].replace("\n", " ")
                print(f"      -> {r['destination']} | {r['heading']}: {preview}...")
        else:
            note = (
                " (likely gibberish or below relevance threshold)"
                if "gibberish" in query.lower() or "xyzabc" in query.lower()
                else ""
            )
            print(f"   No results{note}")

    await service.shutdown()
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test())
