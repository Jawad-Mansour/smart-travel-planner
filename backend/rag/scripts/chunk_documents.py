"""
Phase 9: Parent-child chunking.
Splits documents by headings (##, ###) into section parents,
then splits sections into sentence children.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CLEAN_DIR = PROJECT_ROOT / "backend" / "rag" / "data" / "clean"
META_DIR = PROJECT_ROOT / "backend" / "rag" / "data" / "metadata"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "rag" / "data" / "chunks" / "chunks.json"

# Destinations where Wikivoyage "Do" / outdoor sections should be chunked first
# so sentence children from hiking/activities appear early in the corpus.
OUTDOOR_ADVENTURE_SLUGS: frozenset[str] = frozenset(
    {
        "queenstown",
        "cusco",
        "cape_town",
        "kathmandu",
        "reykjavik",
        "bali",
        "sydney",
        "tokyo",
        "dubai",
        "santorini",
    }
)


# ============================================================
# MODELS
# ============================================================


class ChunkRecord(BaseModel):
    """
    Parent-child chunking: parents are full sections (type parent, no parent_id).
    Children are sentences (type child) with parent_id pointing at the parent's id
    in chunks.json (the database uses separate surrogate keys for rows).
    """

    id: int
    type: str
    destination: str
    heading: str
    content: str
    url: str
    parent_id: int | None = None


class ChunkOutput(BaseModel):
    chunks: list[ChunkRecord] = Field(default_factory=list)


# ============================================================
# LOGGING
# ============================================================


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# CHUNKING UTILITIES
# ============================================================


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.
    Handles long sentences (>500 chars) by splitting on clauses.
    """
    rough_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences: list[str] = []

    for sentence in rough_sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue

        if len(cleaned) <= 500:
            sentences.append(cleaned)
            continue

        # Long sentence fallback: split by clauses (semicolon/comma)
        clauses = [clause.strip() for clause in re.split(r"[;,]\s+", cleaned) if clause.strip()]
        if clauses:
            sentences.extend(clauses)
        else:
            sentences.append(cleaned[:500])
            remainder = cleaned[500:].strip()
            if remainder:
                sentences.append(remainder)

    return sentences


def parse_sections(text: str) -> list[tuple[str, str]]:
    """
    Parse markdown text into sections based on headings.
    Returns list of (heading, content) tuples.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    heading = "Introduction"
    current_lines: list[str] = []

    for line in lines:
        heading_match = re.match(r"^(##|###)\s+(.+)$", line.strip())
        if heading_match:
            if current_lines:
                sections.append((heading, current_lines.copy()))
                current_lines.clear()
            heading = heading_match.group(2).strip()
            continue
        current_lines.append(line)

    if current_lines:
        sections.append((heading, current_lines))

    # Join lines and filter empty sections
    normalized: list[tuple[str, str]] = []
    for section_heading, section_lines in sections:
        content = "\n".join(section_lines).strip()
        if content:
            normalized.append((section_heading, content))

    return normalized


def _section_outdoor_priority(heading: str) -> int:
    """Lower = higher priority for outdoor/adventure destinations."""
    h = heading.lower()
    if any(
        t in h
        for t in (
            "do",
            "hike",
            "trail",
            "outdoor",
            "activity",
            "sport",
            "trek",
            "climb",
            "ski",
            "walk",
            "park",
            "adventure",
            "cycling",
            "running",
        )
    ):
        return 0
    if any(t in h for t in ("see", "go next", "itiner", "buy", "sleep", "eat", "drink")):
        return 1
    return 2


def prioritize_sections_for_slug(
    sections: list[tuple[str, str]],
    destination_slug: str,
) -> list[tuple[str, str]]:
    if destination_slug.lower() not in OUTDOOR_ADVENTURE_SLUGS:
        return sections
    indexed = list(enumerate(sections))
    indexed.sort(key=lambda pair: (_section_outdoor_priority(pair[1][0]), pair[0]))
    return [sec for _, sec in indexed]


def build_chunks_for_destination(clean_file: Path, next_id: int) -> tuple[list[ChunkRecord], int]:
    """
    Build parent and child chunks for a single destination.
    Returns (chunks_list, next_available_id).
    """
    destination_slug = clean_file.stem
    destination = destination_slug.replace("_", " ").title()

    # Load metadata for URL
    metadata_path = META_DIR / f"{destination_slug}.json"
    url = ""
    if metadata_path.exists():
        data: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        url = str(data.get("url", ""))

    text = clean_file.read_text(encoding="utf-8")
    sections = parse_sections(text)
    sections = prioritize_sections_for_slug(sections, destination_slug)
    chunks: list[ChunkRecord] = []

    for heading, section_content in sections:
        # Create parent chunk (full section)
        parent_chunk = ChunkRecord(
            id=next_id,
            type="parent",
            destination=destination,
            heading=heading,
            content=section_content,
            url=url,
        )
        chunks.append(parent_chunk)
        parent_id = next_id
        next_id += 1

        # Create child chunks (sentences)
        for sentence in split_sentences(section_content):
            child_chunk = ChunkRecord(
                id=next_id,
                type="child",
                parent_id=parent_id,
                destination=destination,
                heading=heading,
                content=sentence,
                url=url,
            )
            chunks.append(child_chunk)
            next_id += 1

    return chunks, next_id


# ============================================================
# MAIN
# ============================================================


def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 50)
    logger.info("PHASE 9: CHUNKING DOCUMENTS")
    logger.info("=" * 50)

    try:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        all_chunks: list[ChunkRecord] = []
        current_id = 1

        clean_files = sorted(CLEAN_DIR.glob("*.md"))
        if not clean_files:
            logger.warning(f"No .md files found in {CLEAN_DIR}")
            logger.info("Run collect_content.py first")
            return

        logger.info(f"Found {len(clean_files)} cleaned documents")

        for clean_file in clean_files:
            destination_chunks, current_id = build_chunks_for_destination(clean_file, current_id)
            all_chunks.extend(destination_chunks)
            parent_count = sum(1 for c in destination_chunks if c.type == "parent")
            child_count = sum(1 for c in destination_chunks if c.type == "child")
            logger.info(
                f"Chunked {clean_file.name}: {parent_count} parents, {child_count} children"
            )

        payload = ChunkOutput(chunks=all_chunks)
        OUTPUT_PATH.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

        total_parents = sum(1 for c in all_chunks if c.type == "parent")
        total_children = sum(1 for c in all_chunks if c.type == "child")

        logger.info("=" * 50)
        logger.info("✅ Chunking complete")
        logger.info(f"   Total parents: {total_parents}")
        logger.info(f"   Total children: {total_children}")
        logger.info(f"   Output: {OUTPUT_PATH}")
        logger.info("=" * 50)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.info("Run collect_content.py first")
        raise
    except Exception:
        logger.exception("Fatal error during chunking")
        raise


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Failed")
