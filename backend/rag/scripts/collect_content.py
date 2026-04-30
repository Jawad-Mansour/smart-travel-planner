"""
Phase 8: Automated content collection from Wikivoyage.
Async scraping with retries, concurrency control, and structured logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# ============================================================
# CONFIGURATION
# ============================================================

DESTINATIONS: list[str] = [
    "Kathmandu",
    "Paris",
    "Tokyo",
    "Cape Town",
    "Maldives",
    "Reykjavik",
    "Queenstown",
    "Bali",
    "Rome",
    "Bangkok",
    "New_York",
    "Dubai",
    "Cusco",
    "Santorini",
    "Sydney",
    "Istanbul",
    "Berlin",
    "Amsterdam",
    "Barcelona",
    "Lisbon",
    "Prague",
    "Vienna",
    "Budapest",
    "Krakow",
    "Edinburgh",
]

BASE_URL: str = "https://en.wikivoyage.org/wiki/{destination}"
MAX_CONCURRENT_REQUESTS: int = 5

# Get project root (4 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
RAW_DIR = PROJECT_ROOT / "backend" / "rag" / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "backend" / "rag" / "data" / "clean"
META_DIR = PROJECT_ROOT / "backend" / "rag" / "data" / "metadata"


@dataclass(slots=True)
class FetchResult:
    destination: str
    url: str
    status: int
    fetched_at: str


# ============================================================
# LOGGING
# ============================================================

def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ============================================================
# UTILITIES
# ============================================================

def normalize_destination(destination: str) -> str:
    """Convert destination name to URL-friendly format."""
    return destination.strip().replace(" ", "_")


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for directory in (RAW_DIR, CLEAN_DIR, META_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def extract_clean_markdown(html: str) -> str:
    """Extract readable markdown-like text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # Find main content
    content = soup.select_one("#mw-content-text")
    if content is None:
        content = soup.body or soup

    lines: list[str] = []
    for node in content.find_all(["h2", "h3", "p", "li"]):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if node.name == "h2":
            lines.append(f"## {text}")
        elif node.name == "h3":
            lines.append(f"### {text}")
        elif node.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    return "\n".join(lines).strip()


# ============================================================
# FETCHING
# ============================================================

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(httpx.HTTPError),
)
async def fetch_page(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Fetch a page with retry logic."""
    response = await client.get(url)
    response.raise_for_status()
    return response


def save_result(destination: str, html: str, clean_text: str, metadata: dict[str, Any]) -> None:
    """Save fetched content to files."""
    slug = normalize_destination(destination).lower()
    (RAW_DIR / f"{slug}.html").write_text(html, encoding="utf-8")
    (CLEAN_DIR / f"{slug}.md").write_text(clean_text, encoding="utf-8")
    (META_DIR / f"{slug}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


async def fetch_destination(
    client: httpx.AsyncClient,
    destination: str,
    semaphore: asyncio.Semaphore,
) -> FetchResult:
    """Fetch a single destination with rate limiting."""
    async with semaphore:
        destination_slug = normalize_destination(destination)
        url = BASE_URL.format(destination=destination_slug)
        fetched_at = datetime.now(UTC).isoformat()
        logger = logging.getLogger(__name__)
        
        try:
            response = await fetch_page(client, url)
            clean_text = extract_clean_markdown(response.text)
            metadata: dict[str, Any] = {
                "destination": destination,
                "url": url,
                "fetched_at": fetched_at,
                "status": response.status_code,
            }
            save_result(destination, response.text, clean_text, metadata)
            logger.info("Fetched destination page", extra={"destination": destination, "url": url})
            return FetchResult(
                destination=destination, 
                url=url, 
                status=response.status_code, 
                fetched_at=fetched_at
            )
        except httpx.HTTPError:
            metadata = {
                "destination": destination,
                "url": url,
                "fetched_at": fetched_at,
                "status": 0,
                "error": "fetch_failed_after_retries",
            }
            save_result(destination, "", "", metadata)
            logger.exception("Failed destination fetch", extra={"destination": destination, "url": url})
            return FetchResult(
                destination=destination, 
                url=url, 
                status=0, 
                fetched_at=fetched_at
            )


# ============================================================
# MAIN
# ============================================================

async def run() -> None:
    """Main entry point."""
    configure_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("PHASE 8: CONTENT COLLECTION")
    logger.info("=" * 50)
    
    try:
        ensure_directories()
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        headers = {
            "User-Agent": "SmartTravelPlannerBot/1.0 (+https://github.com/Jawad-Mansour/smart-travel-planner)",
        }
        timeout = httpx.Timeout(20.0, connect=10.0)
        
        logger.info(f"Fetching {len(DESTINATIONS)} destinations from Wikivoyage...")
        
        async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
            tasks = [fetch_destination(client, destination, semaphore) for destination in DESTINATIONS]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = sum(1 for r in results if isinstance(r, FetchResult) and r.status == 200)
            fail_count = len(results) - success_count
            
            logger.info(f"Collection complete: {success_count} succeeded, {fail_count} failed")
            
            if fail_count > 0:
                logger.warning(f"Failed destinations: {fail_count}")
            
    except Exception as e:
        logger.exception("Fatal error during content collection")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logging.getLogger(__name__).warning("Interrupted by user")
    except Exception:
        logging.getLogger(__name__).exception("Failed")