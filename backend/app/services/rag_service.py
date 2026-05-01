"""
Phase 11: RAG Retrieval Service.
Child chunks = vector search only. Parent chunks = full sections returned to the Agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import asyncpg
import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Default cosine relevance (1 - distance). ~0.48 works better with all-MiniLM-L6-v2 than 0.6
# (many valid queries score 0.35–0.55). Stricter mode: set RAG_RELEVANCE_THRESHOLD=0.6 in .env
RELEVANCE_THRESHOLD_DEFAULT: float = 0.48
# If top raw vector scores are all below this, treat query as off-topic / gibberish
GIBBERISH_RAW_SCORE_CAP: float = 0.4
# Keyword fallback score — must stay above relevance_threshold
KEYWORD_MATCH_SCORE: float = 0.52

SHORT_QUERY_EXPANSIONS: dict[str, str] = {
    "hiking": "hiking trails trekking outdoors walking routes mountains nature",
    "hike": "hiking trails trekking outdoors walking routes",
    "beaches": "beach sand coast ocean swimming seaside resorts shoreline",
    "beach": "beaches coast ocean sand swimming seaside",
    "temples": "temples shrines monuments religious sites culture heritage",
    "temple": "temples shrines culture heritage worship",
    "food": "food restaurants street food cuisine eating markets local dishes",
    "visa": "visa entry passport border immigration requirements permit",
    "nightlife": "nightlife bars clubs evening entertainment drinks going out",
    "transport": "public transport buses trains metro getting around tickets",
    "budget": "budget cheap affordable hostels inexpensive low cost economy",
    "accommodation": "hotels hostels lodging guesthouses places to stay sleep",
    "street": "street food markets vendors local eating stalls",
}

OUTDOOR_QUERY_TERMS = frozenset(
    {"hike", "hiking", "trail", "trek", "trekking", "outdoor", "adventure", "climb", "ski", "walk"}
)

OUTDOOR_HEADING_TERMS = (
    "do",
    "hike",
    "trail",
    "outdoor",
    "activity",
    "sport",
    "trek",
    "walk",
    "climb",
    "ski",
    "park",
    "adventure",
)

# Substrings (lowercase) — if none match, query is unlikely to be travel-related
TRAVEL_SIGNAL_SUBSTRINGS: frozenset[str] = frozenset(
    {
        "hiking",
        "hike",
        "trek",
        "trail",
        "mountain",
        "beach",
        "temple",
        "culture",
        "museum",
        "food",
        "nightlife",
        "visa",
        "weather",
        "budget",
        "hotel",
        "hostel",
        "flight",
        "attraction",
        "adventure",
        "relax",
        "luxury",
        "family",
        "city",
        "nature",
        "wildlife",
        "safari",
        "snorkel",
        "dive",
        "ski",
        "climb",
        "yoga",
        "spa",
        "shopping",
        "festival",
        "history",
        "architecture",
        "travel",
        "trip",
        "visit",
        "vacation",
        "holiday",
        "tour",
        "itinerary",
        "accommodation",
        "restaurant",
        "cafe",
        "bar",
        "club",
        "park",
        "lake",
        "island",
        "coast",
        "snorkeling",
        "diving",
        "trekking",
        "walking",
        "cycling",
        "train",
        "bus",
        "airport",
        "passport",
        "immigration",
        "season",
        "climate",
        "summer",
        "winter",
        "spring",
        "autumn",
        "fall",
        "month",
        "week",
        "day trip",
        "sightsee",
        "monument",
        "cathedral",
        "palace",
        "market",
        "street food",
        "backpack",
        "resort",
        "cruise",
        "hiking in",
        "things to do",
        "places to",
        "where to",
        "best time",
        "how to get",
        "cost",
        "price",
        "cheap",
        "expensive",
        "romantic",
        "honeymoon",
        "solo",
        "kids",
        "elderly",
        "accessible",
        # Common destinations (substring match)
        "queenstown",
        "kathmandu",
        "paris",
        "tokyo",
        "bangkok",
        "maldives",
        "santorini",
        "reykjavik",
        "istanbul",
        "berlin",
        "amsterdam",
        "barcelona",
        "lisbon",
        "prague",
        "vienna",
        "budapest",
        "krakow",
        "edinburgh",
        "phuket",
        "bali",
        "dubai",
        "rome",
        "sydney",
        "cusco",
        "cape town",
        "new york",
    }
)


class RAGSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/smart_travel",
        alias="DATABASE_URL",
    )
    relevance_threshold: float = Field(
        default=RELEVANCE_THRESHOLD_DEFAULT, alias="RAG_RELEVANCE_THRESHOLD"
    )
    gibberish_raw_score_cap: float = Field(
        default=GIBBERISH_RAW_SCORE_CAP, alias="RAG_GIBBERISH_RAW_CAP"
    )
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")


@dataclass
class RetrievedChunk:
    """Child chunk from vector or keyword search (never returned to Agent)."""

    id: int
    parent_id: int | None
    content: str
    heading: str
    destination: str
    score: float
    source: str


class RAGService:
    """
    Singleton RAG: search CHILD embeddings, return PARENT sections only.
    """

    _instance: Optional["RAGService"] = None
    _initialized: bool = False

    def __new__(cls) -> "RAGService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def startup(self) -> None:
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)
        self.settings = RAGSettings()

        self.pool = await self._create_pool_with_retry(self.settings.database_url)
        self.logger.info("Database connection pool created")

        self.model = self._load_model()
        self.logger.info("Embedding model loaded")

        self._initialized = True

    async def shutdown(self) -> None:
        if hasattr(self, "pool") and self.pool:
            await self.pool.close()
            self.logger.info("Database pool closed")

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model() -> SentenceTransformer:
        return SentenceTransformer("all-MiniLM-L6-v2")

    async def _create_pool_with_retry(self, database_url: str) -> asyncpg.Pool:
        # asyncpg expects "postgresql://" (or "postgres://"), not SQLAlchemy dialect DSNs.
        dsn = database_url
        if dsn.startswith("postgresql+asyncpg://"):
            dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(8),
            wait=wait_fixed(3),
            retry=retry_if_exception_type(
                (ConnectionError, TimeoutError, OSError, asyncpg.PostgresError)
            ),
            reraise=True,
        ):
            with attempt:
                return await asyncpg.create_pool(
                    dsn,
                    min_size=1,
                    max_size=10,
                    command_timeout=30,
                    timeout=20,
                )
        raise RuntimeError("Retry loop exhausted while creating PostgreSQL connection pool")

    async def _embed(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model.encode(text, normalize_embeddings=True).tolist(),
        )

    async def embed_text_prefix(
        self, text: str, *, max_chars: int = 2000, head_dims: int = 10
    ) -> list[float]:
        """Public helper for analytics UI: first ``head_dims`` values of the query embedding."""
        await self.startup()
        snippet = (text or "").strip()[:max_chars]
        if not snippet:
            return []
        vec = await self._embed(snippet)
        return [float(x) for x in vec[:head_dims]]

    def _query_has_travel_signals(self, query: str) -> bool:
        ql = query.lower()
        return any(sig in ql for sig in TRAVEL_SIGNAL_SUBSTRINGS)

    def _is_gibberish(self, query: str, raw_vector_top_scores: list[float]) -> bool:
        """
        Heuristic gibberish / off-topic detection.
        - No travel-like substring AND query shorter than 5 characters (spec)
        - Top raw vector scores all weak (semantic mismatch / noise)
        """
        q = query.strip()
        if len(q) < 2:
            return True
        if not self._query_has_travel_signals(q) and len(q) < 5:
            self.logger.info(
                "Gibberish: no travel signals and query shorter than 5 characters",
                extra={"query": q},
            )
            return True
        cap = self.settings.gibberish_raw_score_cap
        if len(raw_vector_top_scores) >= 3 and max(raw_vector_top_scores) < cap:
            self.logger.info(
                "Gibberish: top raw vector scores below cap",
                extra={"max_raw": max(raw_vector_top_scores), "cap": cap},
            )
            return True
        return False

    def _expand_short_query(self, query: str) -> str:
        words = [w.strip(".,!?;:\"'()[]") for w in query.strip().split() if w.strip()]
        if len(words) >= 3:
            return query.strip()
        extra: list[str] = []
        for w in words:
            key = w.lower()
            if key in SHORT_QUERY_EXPANSIONS:
                extra.append(SHORT_QUERY_EXPANSIONS[key])
        if not extra:
            return query.strip()
        return f"{query.strip()} {' '.join(extra)}"

    def _is_ambiguous_for_rewrite(self, query: str) -> bool:
        t = query.lower().strip()
        hooks = (
            "where should i go",
            "where should i",
            "best time to visit",
            "what should i do",
            "recommend a destination",
            "recommend somewhere",
        )
        return any(h in t for h in hooks)

    async def _rewrite_query_llm(self, query: str) -> str:
        key = self.settings.openai_api_key
        if not key or not key.strip():
            return query
        if not self._is_ambiguous_for_rewrite(query):
            return query
        url = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's travel search into one concise English sentence "
                        "optimized for semantic search over Wikivoyage travel guide snippets. "
                        "Preserve place names and concrete intent (activities, budget, season). "
                        "Output only the rewritten query, no quotes or preamble."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": 100,
            "temperature": 0.2,
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    content=json.dumps(payload),
                )
                response.raise_for_status()
                data = response.json()
                text = data["choices"][0]["message"]["content"]
                if isinstance(text, str) and text.strip():
                    self.logger.info(
                        "LLM query rewrite applied",
                        extra={"original": query[:120], "rewritten": text.strip()[:200]},
                    )
                    return text.strip()
        except Exception:
            self.logger.exception("LLM query rewrite failed; using original query")
        return query

    def _tokenize_for_keywords(self, query: str) -> list[str]:
        raw = re.findall(r"[A-Za-z]{3,}", query.lower())
        stop = frozenset(
            {
                "the",
                "and",
                "for",
                "are",
                "but",
                "not",
                "you",
                "all",
                "can",
                "her",
                "was",
                "one",
                "our",
                "out",
                "day",
                "get",
                "has",
                "him",
                "his",
                "how",
                "its",
                "may",
                "new",
                "now",
                "old",
                "see",
                "two",
                "way",
                "who",
                "boy",
                "did",
                "let",
                "put",
                "say",
                "she",
                "too",
                "use",
                "want",
                "with",
                "have",
                "this",
                "that",
                "from",
                "they",
                "been",
                "into",
                "than",
                "then",
                "them",
                "some",
                "what",
                "when",
                "where",
                "which",
                "while",
                "will",
                "your",
                "about",
                "after",
                "before",
                "during",
                "visit",
                "best",
                "time",
                "like",
                "looking",
                "planning",
                "need",
                "know",
                "tell",
            }
        )
        return [t for t in raw if t not in stop][:10]

    def _heading_query_boost(self, query: str, heading: str) -> float:
        q = query.lower()
        h = heading.lower()
        mult = 1.0
        if any(w in q for w in OUTDOOR_QUERY_TERMS):
            if any(t in h for t in OUTDOOR_HEADING_TERMS):
                mult *= 1.15
        if any(w in q for w in ("beach", "beaches", "coast", "swim")):
            if any(t in h for t in ("beach", "do", "see", "island", "coast", "swim")):
                mult *= 1.12
        if any(w in q for w in ("temple", "temples", "shrine", "culture")):
            if any(
                t in h for t in ("see", "temple", "culture", "museum", "heritage", "understand")
            ):
                mult *= 1.12
        if any(w in q for w in ("food", "eat", "restaurant", "street food", "cuisine")):
            if any(t in h for t in ("eat", "drink", "food", "restaurant", "cuisine")):
                mult *= 1.15
        if any(w in q for w in ("visa", "passport", "entry")):
            if any(t in h for t in ("get in", "visa", "entry", "arrive", "respect")):
                mult *= 1.12
        if any(w in q for w in ("nightlife", "bar", "club", "party")):
            if any(t in h for t in ("drink", "night", "club", "bar", "entertain")):
                mult *= 1.15
        if any(w in q for w in ("budget", "cheap", "hostel")):
            if any(t in h for t in ("sleep", "budget", "hostel", "cheap", "money")):
                mult *= 1.12
        return min(mult, 1.25)

    def _apply_heading_boosts(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        for c in chunks:
            boost = self._heading_query_boost(query, c.heading)
            new_score = min(c.score * boost, 1.0)
            out.append(
                RetrievedChunk(
                    id=c.id,
                    parent_id=c.parent_id,
                    content=c.content,
                    heading=c.heading,
                    destination=c.destination,
                    score=new_score,
                    source=c.source,
                )
            )
        return out

    def _penalize_introduction_for_intent(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Demote Introduction when the query asks for specific section content."""
        q = query.lower()
        specific = any(
            x in q
            for x in (
                "hik",
                "trail",
                "trek",
                "beach",
                "temple",
                "night",
                "food",
                "visa",
                "budget",
                "museum",
                "ski",
                "dive",
                "culture",
                "hostel",
                "hotel",
                "transport",
                "weather",
            )
        )
        if not specific:
            return chunks
        out: list[RetrievedChunk] = []
        for c in chunks:
            h = (c.heading or "").strip().lower()
            factor = 1.0
            if h in ("introduction", "general") or h.startswith("introduction"):
                factor = 0.52
            new_score = min(c.score * factor, 1.0)
            out.append(
                RetrievedChunk(
                    id=c.id,
                    parent_id=c.parent_id,
                    content=c.content,
                    heading=c.heading,
                    destination=c.destination,
                    score=new_score,
                    source=c.source,
                )
            )
        return out

    async def _vector_search(
        self,
        embedding: list[float],
        destination: Optional[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        embedding_str = "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"

        if destination:
            sql = """
                SELECT
                    c.id,
                    c.parent_chunk_id,
                    c.content,
                    c.heading,
                    c.metadata->>'destination' AS destination,
                    (1 - (c.embedding <=> $1::vector))::float8 AS similarity
                FROM chunks c
                WHERE c.chunk_type = 'child'
                  AND c.embedding IS NOT NULL
                  AND LOWER(c.metadata->>'destination') = LOWER($2::text)
                ORDER BY c.embedding <=> $1::vector
                LIMIT $3
            """
            rows = await self.pool.fetch(sql, embedding_str, destination, limit)
        else:
            sql = """
                SELECT
                    c.id,
                    c.parent_chunk_id,
                    c.content,
                    c.heading,
                    c.metadata->>'destination' AS destination,
                    (1 - (c.embedding <=> $1::vector))::float8 AS similarity
                FROM chunks c
                WHERE c.chunk_type = 'child'
                  AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> $1::vector
                LIMIT $2
            """
            rows = await self.pool.fetch(sql, embedding_str, limit)

        return [
            RetrievedChunk(
                id=int(row["id"]),
                parent_id=int(row["parent_chunk_id"])
                if row["parent_chunk_id"] is not None
                else None,
                content=str(row["content"]),
                heading=str(row["heading"] or "General"),
                destination=str(row["destination"] or "Unknown"),
                score=float(row["similarity"]),
                source="vector",
            )
            for row in rows
        ]

    async def _keyword_search(
        self,
        tokens: list[str],
        destination: Optional[str],
        limit: int,
    ) -> list[RetrievedChunk]:
        if not tokens or self.pool is None:
            return []
        tokens = [t for t in tokens if len(t) >= 3][:8]
        if not tokens:
            return []

        if destination:
            sql = """
                SELECT
                    c.id,
                    c.parent_chunk_id,
                    c.content,
                    c.heading,
                    c.metadata->>'destination' AS destination,
                    $4::float8 AS similarity
                FROM chunks c
                WHERE c.chunk_type = 'child'
                  AND LOWER(c.metadata->>'destination') = LOWER($2::text)
                  AND EXISTS (
                    SELECT 1
                    FROM unnest($1::text[]) AS kw
                    WHERE c.content ILIKE ('%' || kw || '%')
                  )
                LIMIT $3
            """
            rows = await self.pool.fetch(sql, tokens, destination, limit, KEYWORD_MATCH_SCORE)
        else:
            sql = """
                SELECT
                    c.id,
                    c.parent_chunk_id,
                    c.content,
                    c.heading,
                    c.metadata->>'destination' AS destination,
                    $3::float8 AS similarity
                FROM chunks c
                WHERE c.chunk_type = 'child'
                  AND EXISTS (
                    SELECT 1
                    FROM unnest($1::text[]) AS kw
                    WHERE c.content ILIKE ('%' || kw || '%')
                  )
                LIMIT $2
            """
            rows = await self.pool.fetch(sql, tokens, limit, KEYWORD_MATCH_SCORE)

        return [
            RetrievedChunk(
                id=int(row["id"]),
                parent_id=int(row["parent_chunk_id"])
                if row["parent_chunk_id"] is not None
                else None,
                content=str(row["content"]),
                heading=str(row["heading"] or "General"),
                destination=str(row["destination"] or "Unknown"),
                score=float(row["similarity"]),
                source="keyword",
            )
            for row in rows
        ]

    async def _fetch_parents(self, parent_ids: list[int]) -> list[dict[str, Any]]:
        if not parent_ids:
            return []
        sql = """
            SELECT
                c.id,
                c.content,
                c.heading,
                c.metadata->>'destination' AS destination,
                d.source_url
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.id = ANY($1::bigint[])
              AND c.chunk_type = 'parent'
        """
        rows = await self.pool.fetch(sql, parent_ids)
        return [
            {
                "id": int(row["id"]),
                "content": str(row["content"]),
                "heading": str(row["heading"] or "General"),
                "destination": str(row["destination"] or "Unknown"),
                "source_url": str(row["source_url"] or ""),
            }
            for row in rows
        ]

    def _apply_mmr(
        self,
        candidates: list[RetrievedChunk],
        lambda_param: float = 0.5,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        selected: list[RetrievedChunk] = []
        remaining = candidates.copy()
        picks = min(max(len(candidates), 1), max(top_k * 2, top_k))

        for _ in range(picks):
            if not remaining:
                break
            best_idx = 0
            best_score = float("-inf")
            for i, chunk in enumerate(remaining):
                relevance = chunk.score
                diversity_penalty = 0.0
                for sel in selected:
                    if chunk.parent_id and chunk.parent_id == sel.parent_id:
                        diversity_penalty = max(diversity_penalty, 0.85)
                    elif chunk.heading == sel.heading and chunk.destination == sel.destination:
                        diversity_penalty = max(diversity_penalty, 0.55)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            selected.append(remaining.pop(best_idx))
        return selected

    def _merge_dedupe_chunks(
        self,
        primary: list[RetrievedChunk],
        secondary: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        by_id: dict[int, RetrievedChunk] = {}
        for c in primary + secondary:
            existing = by_id.get(c.id)
            if existing is None or c.score > existing.score:
                by_id[c.id] = c
        return sorted(by_id.values(), key=lambda x: x.score, reverse=True)

    def _filter_by_relevance(
        self,
        candidates: list[RetrievedChunk],
        threshold: float,
        query: str,
    ) -> list[RetrievedChunk]:
        kept: list[RetrievedChunk] = []
        for c in candidates:
            if c.score > threshold:
                kept.append(c)
            else:
                self.logger.info(
                    "Skipping child chunk below relevance threshold",
                    extra={
                        "query_preview": query[:80],
                        "chunk_id": c.id,
                        "parent_id": c.parent_id,
                        "score": round(c.score, 4),
                        "threshold": threshold,
                        "source": c.source,
                    },
                )
        return kept

    def _order_parent_ids_diverse_destinations(
        self,
        diversified: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[int], dict[int, float]]:
        """
        Prefer distinct destinations (multi-destination retrieval).
        Same parent_id appears at most once.
        """
        parent_best: dict[int, tuple[float, str]] = {}
        for ch in diversified:
            if ch.parent_id is None:
                continue
            pid = int(ch.parent_id)
            dest = (ch.destination or "Unknown").strip()
            prev = parent_best.get(pid)
            if prev is None or ch.score > prev[0]:
                parent_best[pid] = (ch.score, dest)

        ranked_pids = sorted(parent_best.keys(), key=lambda p: parent_best[p][0], reverse=True)
        parent_scores = {pid: parent_best[pid][0] for pid in ranked_pids}

        picked: list[int] = []
        seen_dest: set[str] = set()
        for pid in ranked_pids:
            dest_l = parent_best[pid][1].lower()
            if dest_l in seen_dest:
                continue
            seen_dest.add(dest_l)
            picked.append(pid)
            if len(picked) >= top_k:
                return picked, parent_scores

        for pid in ranked_pids:
            if pid not in picked:
                picked.append(pid)
            if len(picked) >= top_k:
                break
        return picked[:top_k], parent_scores

    def _order_parent_ids_single_destination(
        self,
        diversified: list[RetrievedChunk],
        top_k: int,
    ) -> tuple[list[int], dict[int, float]]:
        """Same city: dedupe by parent_id, preserve best child score per parent."""
        parent_scores: dict[int, float] = {}
        order: list[int] = []
        seen: set[int] = set()
        for ch in diversified:
            if ch.parent_id is None:
                continue
            pid = int(ch.parent_id)
            parent_scores[pid] = max(parent_scores.get(pid, 0.0), ch.score)
            if pid not in seen:
                seen.add(pid)
                order.append(pid)
            if len(order) >= top_k * 2:
                break
        return order[: top_k * 2], parent_scores

    async def search(
        self,
        query: str,
        destination: Optional[str] = None,
        top_k: int = 5,
        lambda_param: float = 0.5,
        *,
        multi_destination_diversity: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """
        Search child embeddings; return parent sections only (full context for Agent).

        When ``destination`` is None, results prefer distinct destinations unless
        ``multi_destination_diversity=False``.
        """
        if not query or not query.strip():
            return []

        await self.startup()
        threshold = self.settings.relevance_threshold

        rewritten = await self._rewrite_query_llm(query.strip())
        expanded = self._expand_short_query(rewritten)
        if expanded != rewritten.strip():
            self.logger.info(
                "Short-query expansion",
                extra={"original": rewritten[:120], "expanded": expanded[:200]},
            )

        query_embedding = await self._embed(expanded)
        vector_limit = 60
        vec_raw = await self._vector_search(query_embedding, destination, limit=vector_limit)
        raw_scores = [c.score for c in vec_raw[:8]]

        if self._is_gibberish(expanded, raw_scores):
            return []

        candidates = self._apply_heading_boosts(expanded, vec_raw)
        candidates = self._penalize_introduction_for_intent(expanded, candidates)

        above = self._filter_by_relevance(candidates, threshold, expanded)

        if len(above) < 2:
            vector_kept = len(above)
            tokens = self._tokenize_for_keywords(expanded)
            kw = await self._keyword_search(tokens, destination, limit=40)
            kw = self._penalize_introduction_for_intent(
                expanded, self._apply_heading_boosts(expanded, kw)
            )
            merged = self._merge_dedupe_chunks(above, kw)
            above = self._filter_by_relevance(merged, threshold, expanded)
            self.logger.info(
                "Keyword fallback",
                extra={
                    "vector_above_threshold": vector_kept,
                    "after_merge": len(above),
                    "tokens": tokens[:6],
                },
            )

        if not above:
            self.logger.warning(
                "No child chunks above relevance threshold",
                extra={"query": query[:80], "destination": destination, "threshold": threshold},
            )
            return []

        diversified = self._apply_mmr(above, lambda_param=lambda_param, top_k=top_k)

        use_diversity = (
            multi_destination_diversity
            if multi_destination_diversity is not None
            else (destination is None)
        )
        if use_diversity:
            parent_ids_ordered, parent_scores = self._order_parent_ids_diverse_destinations(
                diversified, top_k
            )
        else:
            parent_ids_ordered, parent_scores = self._order_parent_ids_single_destination(
                diversified, top_k
            )

        parents = await self._fetch_parents(parent_ids_ordered)
        by_id = {p["id"]: p for p in parents}
        ordered: list[dict[str, Any]] = []
        for pid in parent_ids_ordered:
            if pid in by_id and len(ordered) < top_k:
                row = dict(by_id[pid])
                row["chunk_type"] = "parent"
                row["retrieval_score"] = round(parent_scores.get(pid, 0.0), 4)
                ordered.append(row)
        return ordered

    async def search_all_destinations(
        self,
        query: str,
        top_k: int = 5,
        lambda_param: float = 0.5,
    ) -> list[dict[str, Any]]:
        """
        Multi-destination retrieval: search all cities, return top parents from
        distinct destinations where possible.
        """
        return await self.search(
            query,
            destination=None,
            top_k=top_k,
            lambda_param=lambda_param,
            multi_destination_diversity=True,
        )

    async def search_simple(
        self,
        query: str,
        destination: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return await self.search(query, destination=destination, top_k=top_k, lambda_param=1.0)


@lru_cache(maxsize=1)
def get_instance() -> RAGService:
    return RAGService()
