"""
Phase 10A: PostgreSQL + pgvector database setup.
Creates database, enables pgvector, creates tables and indexes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from urllib.parse import urlparse

import asyncpg
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/smart_travel",
        alias="DATABASE_URL",
    )


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# DATABASE SETUP
# ============================================================

def build_admin_url(target_url: str) -> tuple[str, str]:
    """Build admin connection URL (to 'postgres' default DB) and extract target DB name."""
    parsed = urlparse(target_url)
    db_name = parsed.path.lstrip("/")
    # Connect to default 'postgres' database to create new DB
    admin_url = parsed._replace(path="/postgres").geturl()
    return admin_url, db_name


async def create_database_if_missing(settings: DatabaseSettings) -> None:
    """Create target database if it doesn't exist."""
    logger = logging.getLogger(__name__)
    admin_url, db_name = build_admin_url(settings.database_url)
    
    try:
        admin_conn = await connect_with_retry(admin_url)
    except Exception as e:
        logger.error(f"Cannot connect to admin database: {e}")
        logger.info("\nMake sure PostgreSQL is running:")
        logger.info("  - Docker: docker run --name postgres-rag -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:pg16")
        logger.info("  - Windows: net start postgresql-x64-15")
        logger.info("  - Mac: brew services start postgresql")
        logger.info("  - Linux: sudo service postgresql start")
        raise
    
    try:
        exists = await admin_conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", db_name)
        if exists:
            logger.info(f"Database '{db_name}' already exists")
            return
        await admin_conn.execute(f'CREATE DATABASE "{db_name}"')
        logger.info(f"Created database '{db_name}'")
    finally:
        await admin_conn.close()


async def create_schema(settings: DatabaseSettings) -> None:
    """Create tables, indexes, and enable extensions."""
    logger = logging.getLogger(__name__)
    
    try:
        conn = await connect_with_retry(settings.database_url)
    except Exception as e:
        logger.error(f"Cannot connect to target database: {e}")
        raise
    
    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("✅ pgvector extension enabled")
        
        # Documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id BIGSERIAL PRIMARY KEY,
                destination_id TEXT,
                destination_name TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'wikivoyage',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        logger.info("✅ Created 'documents' table")
        
        # Chunks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                parent_chunk_id BIGINT REFERENCES chunks(id) ON DELETE CASCADE,
                chunk_type TEXT NOT NULL CHECK (chunk_type IN ('parent', 'child')),
                content TEXT NOT NULL,
                content_length INT NOT NULL,
                heading TEXT,
                embedding VECTOR(384),
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        logger.info("✅ Created 'chunks' table")
        
        # Index on document_id + chunk_type
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_type 
            ON chunks(document_id, chunk_type)
        """)
        logger.info("✅ Created index on (document_id, chunk_type)")
        
        # HNSW index for fast vector similarity search
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        logger.info("✅ Created HNSW index on embedding column")
        
        # Unique constraint to prevent duplicate ingestion
        await conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_source_chunk_id
            ON chunks ((metadata->>'source_chunk_id'))
            WHERE metadata ? 'source_chunk_id'
        """)
        logger.info("✅ Created unique index on source_chunk_id")
        
        logger.info("=" * 50)
        logger.info("✅ Database schema setup complete")
        logger.info("=" * 50)
        
    finally:
        await conn.close()


async def connect_with_retry(database_url: str) -> asyncpg.Connection:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(8),
        wait=wait_fixed(3),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError, asyncpg.PostgresError)),
        reraise=True,
    ):
        with attempt:
            return await asyncpg.connect(database_url, timeout=20)
    raise RuntimeError("Retry loop exhausted while connecting to PostgreSQL")


# ============================================================
# MAIN
# ============================================================

async def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("PHASE 10A: DATABASE SETUP")
    logger.info("=" * 50)
    
    try:
        settings = DatabaseSettings()
        # Hide password in logs
        safe_url = settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url
        logger.info(f"Target database: {safe_url}")
        
        await create_database_if_missing(settings)
        await create_schema(settings)
        
    except asyncpg.exceptions.InvalidPasswordError:
        logger.error("❌ Database authentication failed. Check DATABASE_URL in .env")
        logger.info("Expected format: postgresql://username:password@localhost:5432/smart_travel")
        raise
    except asyncpg.exceptions.CannotConnectNowError:
        logger.error("❌ Cannot connect to PostgreSQL. Is it running?")
        raise
    except Exception as e:
        logger.exception("Fatal error during database setup")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Failed")