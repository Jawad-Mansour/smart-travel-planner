"""
Phase 10B: Generate embeddings and store in PostgreSQL.
Uses sentence-transformers locally (free, offline, 384-dim vectors).
"""

from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import asyncpg
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CHUNKS_PATH = PROJECT_ROOT / "backend" / "rag" / "data" / "chunks" / "chunks.json"


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/smart_travel",
        alias="DATABASE_URL",
    )


def normalize_asyncpg_dsn(database_url: str) -> str:
    """
    Convert SQLAlchemy async DSN to an asyncpg-compatible DSN.
    """
    dsn = str(database_url or "").strip()
    if dsn.startswith("postgresql+asyncpg://"):
        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    return dsn


class ChunkRecord(BaseModel):
    id: int
    type: str
    destination: str
    heading: str
    content: str
    url: str
    parent_id: int | None = None


class ChunkPayload(BaseModel):
    chunks: list[ChunkRecord]


# ============================================================
# LOGGING
# ============================================================


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# MODEL LOADING (CACHED)
# ============================================================


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    """Load sentence transformer model (cached, loaded once)."""
    logger = logging.getLogger(__name__)
    logger.info("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    logger.info("Model loaded successfully")
    return model


# ============================================================
# DATABASE OPERATIONS
# ============================================================


def vector_to_pg_text(embedding: list[float]) -> str:
    """Convert embedding list to PostgreSQL vector string format."""
    return "[" + ",".join(f"{value:.8f}" for value in embedding) + "]"


async def get_or_create_document(
    conn: asyncpg.Connection, destination: str, source_url: str
) -> int:
    """Get existing document ID or create new one."""
    existing_id = await conn.fetchval(
        """
        SELECT id
        FROM documents
        WHERE destination_name = $1 AND source_url = $2
        ORDER BY id
        LIMIT 1
        """,
        destination,
        source_url,
    )
    if existing_id is not None:
        return int(existing_id)

    document_id = await conn.fetchval(
        """
        INSERT INTO documents (destination_name, source_url, source_type)
        VALUES ($1, $2, 'wikivoyage')
        RETURNING id
        """,
        destination,
        source_url,
    )
    return int(document_id)


async def fetch_existing_source_chunk_ids(conn: asyncpg.Connection) -> set[int]:
    """Get IDs of chunks already stored (to skip duplicates)."""
    rows = await conn.fetch(
        """
        SELECT (metadata->>'source_chunk_id') AS source_chunk_id
        FROM chunks
        WHERE metadata ? 'source_chunk_id'
        """
    )
    output: set[int] = set()
    for row in rows:
        raw = row["source_chunk_id"]
        if raw is None:
            continue
        try:
            output.add(int(raw))
        except ValueError:
            continue
    return output


async def store_chunks(chunks: list[ChunkRecord], settings: DatabaseSettings) -> None:
    """Store parent and child chunks with embeddings."""
    logger = logging.getLogger(__name__)
    conn = await connect_with_retry(settings.database_url)

    try:
        existing_source_ids = await fetch_existing_source_chunk_ids(conn)
        parent_map: dict[int, int] = {}
        model = load_model()

        parent_chunks = [chunk for chunk in chunks if chunk.type == "parent"]
        child_chunks = [chunk for chunk in chunks if chunk.type == "child"]

        # Store parent chunks first
        logger.info(f"Storing {len(parent_chunks)} parent chunks...")
        for parent in tqdm(parent_chunks, desc="Parent chunks"):
            document_id = await get_or_create_document(conn, parent.destination, parent.url)
            metadata = {"source_chunk_id": parent.id, "destination": parent.destination}

            row_id = await conn.fetchval(
                """
                INSERT INTO chunks (
                    document_id, parent_chunk_id, chunk_type, content, content_length,
                    heading, embedding, metadata
                )
                VALUES ($1, NULL, 'parent', $2, $3, $4, NULL, $5::jsonb)
                ON CONFLICT ((metadata->>'source_chunk_id')) WHERE metadata ? 'source_chunk_id'
                DO UPDATE SET content = EXCLUDED.content
                RETURNING id
                """,
                document_id,
                parent.content,
                len(parent.content),
                parent.heading,
                json.dumps(metadata),
            )
            parent_map[parent.id] = int(row_id)

        # Generate embeddings and store child chunks
        logger.info(f"Generating embeddings for {len(child_chunks)} child chunks...")
        records_to_insert: list[tuple[Any, ...]] = []
        skipped = 0

        for child in tqdm(child_chunks, desc="Child chunks"):
            if child.id in existing_source_ids:
                skipped += 1
                continue
            if child.parent_id is None or child.parent_id not in parent_map:
                continue

            document_id = await get_or_create_document(conn, child.destination, child.url)
            embedding = model.encode(child.content, normalize_embeddings=True).tolist()
            metadata = {"source_chunk_id": child.id, "destination": child.destination}
            records_to_insert.append(
                (
                    document_id,
                    parent_map[child.parent_id],
                    child.content,
                    len(child.content),
                    child.heading,
                    vector_to_pg_text(embedding),
                    json.dumps(metadata),
                )
            )

        if records_to_insert:
            await conn.executemany(
                """
                INSERT INTO chunks (
                    document_id, parent_chunk_id, chunk_type, content, content_length,
                    heading, embedding, metadata
                )
                VALUES ($1, $2, 'child', $3, $4, $5, $6::vector, $7::jsonb)
                ON CONFLICT ((metadata->>'source_chunk_id')) WHERE metadata ? 'source_chunk_id'
                DO NOTHING
                """,
                records_to_insert,
            )

        logger.info(
            f"Stored {len(records_to_insert)} new child chunks (skipped {skipped} existing)"
        )

    finally:
        await conn.close()


async def connect_with_retry(database_url: str) -> asyncpg.Connection:
    dsn = normalize_asyncpg_dsn(database_url)
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(8),
        wait=wait_fixed(3),
        retry=retry_if_exception_type(
            (ConnectionError, TimeoutError, OSError, asyncpg.PostgresError)
        ),
        reraise=True,
    ):
        with attempt:
            return await asyncpg.connect(dsn, timeout=20)
    raise RuntimeError("Retry loop exhausted while connecting to PostgreSQL")


# ============================================================
# MAIN
# ============================================================


async def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("PHASE 10B: EMBEDDINGS + VECTOR STORE")
    logger.info("=" * 50)

    try:
        # Check if chunks.json exists
        if not CHUNKS_PATH.exists():
            logger.error(f"Chunks file not found: {CHUNKS_PATH}")
            logger.info("Run chunk_documents.py first")
            return

        settings = DatabaseSettings()
        payload = ChunkPayload.model_validate_json(CHUNKS_PATH.read_text(encoding="utf-8"))

        if not payload.chunks:
            logger.warning("No chunks found in file")
            return

        parent_count = sum(1 for c in payload.chunks if c.type == "parent")
        child_count = sum(1 for c in payload.chunks if c.type == "child")
        logger.info(f"Loaded {parent_count} parents, {child_count} children")

        await store_chunks(payload.chunks, settings)

        logger.info("=" * 50)
        logger.info("✅ Embedding and storage complete")
        logger.info("=" * 50)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in chunks file: {e}")
        raise
    except Exception:
        logger.exception("Fatal error during embedding and storage")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Failed")
