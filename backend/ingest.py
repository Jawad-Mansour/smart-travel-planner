"""
Ingest destination markdown files: chunk (500 chars, overlap 50), embed with
``sentence-transformers/all-MiniLM-L6-v2``, store in Postgres pgvector (``documents`` + ``chunks``).

Run after migrations and with ``DATABASE_URL`` set::

    python -m backend.ingest
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

import asyncpg
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from backend.app.core.config import get_settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "backend" / "rag" / "data" / "clean"
CHUNK_SIZE = 500
OVERLAP = 50


def _normalize_dsn(url: str) -> str:
    u = str(url).strip()
    if u.startswith("postgresql+asyncpg://"):
        return u.replace("postgresql+asyncpg://", "postgresql://", 1)
    return u


def _vector_to_pg(embedding: list[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


def _chunk_text(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", text).strip()
    if not t:
        return []
    out: list[str] = []
    i = 0
    while i < len(t):
        out.append(t[i : i + CHUNK_SIZE])
        i += CHUNK_SIZE - OVERLAP
    return out


async def _ensure_document(conn: asyncpg.Connection, destination: str, source_url: str) -> int:
    row = await conn.fetchval(
        """
        SELECT id FROM documents
        WHERE destination_name = $1 AND source_url = $2
        ORDER BY id LIMIT 1
        """,
        destination,
        source_url,
    )
    if row is not None:
        return int(row)
    return int(
        await conn.fetchval(
            """
            INSERT INTO documents (destination_name, source_url, source_type)
            VALUES ($1, $2, 'ingest')
            RETURNING id
            """,
            destination,
            source_url,
        )
    )


async def ingest() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("ingest")
    settings = get_settings()
    if not CLEAN_DIR.is_dir():
        raise FileNotFoundError(f"Missing clean markdown dir: {CLEAN_DIR}")

    dsn = _normalize_dsn(settings.database_url)
    conn = await asyncpg.connect(dsn, timeout=30)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        md_files = sorted(CLEAN_DIR.glob("*.md"))
        log.info("Found %s markdown files", len(md_files))
        sid = 0
        for path in tqdm(md_files, desc="files"):
            slug = path.stem.replace("_", " ").title()
            text = path.read_text(encoding="utf-8", errors="ignore")
            destination = slug
            source_url = f"file:///{path.as_posix()}"
            doc_id = await _ensure_document(conn, destination, source_url)
            parent_meta = json.dumps({"source": "ingest_parent", "slug": path.stem})
            parent_id = await conn.fetchval(
                """
                INSERT INTO chunks (
                    document_id, parent_chunk_id, chunk_type, content, content_length,
                    heading, embedding, metadata
                )
                VALUES ($1, NULL, 'parent', $2, $3, $4, NULL, $5::jsonb)
                RETURNING id
                """,
                doc_id,
                text[:8000],
                min(len(text), 8000),
                path.stem,
                parent_meta,
            )
            chunks = _chunk_text(text)
            rows: list[tuple[Any, ...]] = []
            for part in chunks:
                sid += 1
                emb = model.encode(part, normalize_embeddings=True).tolist()
                meta = json.dumps(
                    {"source_chunk_id": sid, "destination": destination, "slug": path.stem}
                )
                rows.append(
                    (
                        doc_id,
                        int(parent_id),
                        part,
                        len(part),
                        path.stem,
                        _vector_to_pg(emb),
                        meta,
                    )
                )
            if rows:
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
                    rows,
                )
        log.info("Ingest complete.")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(ingest())
