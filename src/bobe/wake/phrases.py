"""Wake phrase matching for BoBe."""

from __future__ import annotations

WAKE_PHRASE = "hey jarvis"

DEFAULT_SLEEP_PHRASES: tuple[str, ...] = ("go to sleep", "κοιμήσου")
SLEEP_PHRASE_ASR_VARIANTS: tuple[str, ...] = ("got to sleep",)

# Substrings that must not trigger wake (common Whisper false positives / homophones).
FALSE_WAKE_SUBSTRINGS: tuple[str, ...] = (
    "hey service",
    "the service",
    "customer service",
    " church service",
)


def normalize_transcript(text: str) -> str:
    """Normalize ASR text for wake phrase comparison."""
    stripped = text.strip().strip(" \t\n\r,.:;!?-")
    return " ".join(stripped.casefold().split())


def is_false_wake_transcript(text: str) -> bool:
    """Return True when the transcript is a known non-wake homophone."""
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    return any(substring in normalized for substring in FALSE_WAKE_SUBSTRINGS)


def matches_wake_phrase(text: str, *, phrase: str = WAKE_PHRASE) -> bool:
    """Return whether a transcript contains the wake phrase."""
    if is_false_wake_transcript(text):
        return False
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    wake = phrase.casefold()
    wake_name = wake.split()[-1] if wake else ""
    if normalized.startswith(wake):
        return True
    if wake_name and normalized.startswith(wake_name):
        return True
    return wake in normalized


def matches_sleep_phrase(
    text: str,
    phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES,
) -> bool:
    """Return whether a transcript asks BoBe to go back to sleep."""
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    candidates = (*phrases, *SLEEP_PHRASE_ASR_VARIANTS)
    return any(phrase.casefold() in normalized for phrase in candidates if phrase.strip())
