"""Wake phrase matching for BoBe."""

from __future__ import annotations

WAKE_PHRASE = "hey jarvis"


def normalize_transcript(text: str) -> str:
    """Normalize ASR text for wake phrase comparison."""
    stripped = text.strip().strip(" \t\n\r,.:;!?-")
    return " ".join(stripped.casefold().split())


def matches_wake_phrase(text: str, *, phrase: str = WAKE_PHRASE) -> bool:
    """Return whether a transcript contains the wake phrase."""
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    wake = phrase.casefold()
    if normalized.startswith(wake):
        return True
    if normalized.startswith("jarvis"):
        return True
    return wake in normalized
