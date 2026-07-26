"""Wake phrase matching for BoBe."""

from __future__ import annotations
import unicodedata


WAKE_PHRASE = "hey bobe"

DEFAULT_SLEEP_PHRASES: tuple[str, ...] = ("go to sleep", "κοιμήσου")
SLEEP_PHRASE_ASR_VARIANTS: tuple[str, ...] = ("got to sleep",)

# Common Whisper near-misses for the wake name (keep tight — avoid everyday phrases).
WAKE_PHRASE_ASR_VARIANTS: tuple[str, ...] = (
    "hey bobby",
    "hey boby",
    "hey bobbie",
)

# Substrings that must not trigger wake (common Whisper false positives / homophones).
FALSE_WAKE_SUBSTRINGS: tuple[str, ...] = (
    "hey service",
    "the service",
    "customer service",
    " church service",
)


def normalize_transcript(text: str) -> str:
    """Normalize ASR text for wake phrase comparison."""
    # Recompose first: decomposed (NFD) accents are combining marks that the
    # punctuation filter below would turn into spaces, splitting words apart.
    text = unicodedata.normalize("NFKC", text)
    # Strip punctuation inside tokens too — Whisper often emits "Hey, Bobby."
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text.casefold())
    return " ".join(cleaned.split())


def is_false_wake_transcript(text: str) -> bool:
    """Return True when the transcript is a known non-wake homophone."""
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    return any(substring in normalized for substring in FALSE_WAKE_SUBSTRINGS)


def matches_wake_phrase(text: str, *, phrase: str = WAKE_PHRASE) -> bool:
    """Return whether a transcript contains the wake phrase.

    Requires the full phrase (e.g. ``hey bobe``), not the bare name. Whisper's
    initial prompt often echoes just ``Bobe.`` / ``Jarvis.`` and that must not wake.
    """
    if is_false_wake_transcript(text):
        return False
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    # Normalize the phrase side too — a punctuated custom phrase ("hey, bobe")
    # could otherwise never match the punctuation-stripped transcript.
    wake = normalize_transcript(phrase)
    if not wake:
        return False
    if normalized == wake or normalized.startswith(wake + " ") or f" {wake} " in f" {normalized} ":
        return True
    # Near-miss variants apply only for the default BoBe phrase.
    if wake == WAKE_PHRASE:
        for variant in WAKE_PHRASE_ASR_VARIANTS:
            candidate = normalize_transcript(variant)
            if not candidate:
                continue
            if (
                normalized == candidate
                or normalized.startswith(candidate + " ")
                or f" {candidate} " in f" {normalized} "
            ):
                return True
    return False


def matches_sleep_phrase(
    text: str,
    phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES,
) -> bool:
    """Return whether a transcript asks BoBe to go back to sleep."""
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    # Normalize the phrase side too so punctuated/decomposed custom phrases
    # can match the normalized transcript.
    candidates = (normalize_transcript(phrase) for phrase in (*phrases, *SLEEP_PHRASE_ASR_VARIANTS))
    return any(candidate in normalized for candidate in candidates if candidate)
