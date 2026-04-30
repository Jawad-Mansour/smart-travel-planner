"""
Interactive smoke test for one realistic query (host or Docker).

Run from repo root:
  python backend/rag/scripts/real_user_test.py
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

    query = (
        "I have 2 weeks in July with $1500, want warm weather and hiking"
    )
    results = await service.search(query, destination=None, top_k=4)

    print("=" * 70)
    print("REAL USER QUERY TEST")
    print("=" * 70)
    print(f"Query: {query}")
    print()

    for i, r in enumerate(results):
        print(f"{i + 1}. {r['destination']} | {r['heading']} | score={r.get('retrieval_score')}")
        content = str(r.get("content", ""))
        print(f"   {content[:200]}...")
        print()

    await service.shutdown()


if __name__ == "__main__":
    asyncio.run(test())
