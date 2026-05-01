"""Markdown segmentation for multi-bubble chat UX."""

from backend.app.api.chat_markdown_split import split_travel_answer_segments


def test_split_travel_answer_segments_basic() -> None:
    md = """## Recommended Destinations for Your Trip

Intro line here.

---

### 1. Paris, France 🇫🇷

**Why**

---

### 2. Rome, Italy 🇮🇹

**Why**

## My Recommendation

**Top pick** Choose Paris."""

    parts = split_travel_answer_segments(md)
    assert parts is not None
    assert len(parts) >= 3
    assert "### 1." in parts[0]
    assert "### 2." in parts[1]
    assert "My Recommendation" in parts[-1]


def test_split_returns_none_without_numbered_destinations() -> None:
    assert split_travel_answer_segments("Just some text without headings.") is None
