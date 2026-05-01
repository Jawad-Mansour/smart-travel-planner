"""Split rendered travel-plan markdown into sequential chat segments."""

from __future__ import annotations

import re


def split_travel_answer_segments(answer: str) -> list[str] | None:
    """
    Split a full synthesis markdown reply into separate bubbles:
    intro + first destination, each subsequent destination, then recommendation.

    Returns None if the text doesn't match the structured travel format (single bubble).
    """
    text = (answer or "").strip()
    if not text:
        return None
    if "### 1." not in text:
        return None

    rec_match = re.search(r"\n## My Recommendation\s*\n", text)
    tail = ""
    main = text
    if rec_match:
        main = text[: rec_match.start()].rstrip()
        tail = text[rec_match.start() :].strip()

    parts = re.split(r"\n---\n\n", main)
    if len(parts) < 2:
        return None

    intro = parts[0].strip()
    dest_parts = [p.strip() for p in parts[1:] if p.strip()]
    if not dest_parts:
        return None

    segments: list[str] = []
    first = dest_parts[0]
    segments.append(f"{intro}\n\n{first}".strip())
    for extra in dest_parts[1:]:
        segments.append(extra)
    if tail:
        segments.append(tail)
    return segments if len(segments) >= 2 else None
