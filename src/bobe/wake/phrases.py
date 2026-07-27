"""Wake phrase matching for BoBe."""

from __future__ import annotations
import unicodedata


WAKE_PHRASE = "hey bobe"

DEFAULT_SLEEP_PHRASES: tuple[str, ...] = ("go to sleep", "κοιμήσου")
SLEEP_PHRASE_ASR_VARIANTS: tuple[str, ...] = ("got to sleep",)

# Common Whisper near-misses for the wake name (keep tight — avoid everyday phrases).
# "hey bob"/"hey bove" added after benchmarking: every Whisper size (including
# medium.en) predominantly transcribes "hey bobe" as "Hey Bob"; without these
# variants most genuine wakes are dropped. Trade-off: saying "hey Bob" to a
# person named Bob within earshot now wakes the robot.
WAKE_PHRASE_ASR_VARIANTS: tuple[str, ...] = (
    "hey bobby",
    "hey boby",
    "hey bobbie",
    "hey bob",
    "hey bove",
)

# Substrings that must not trigger wake (common Whisper false positives / homophones).
FALSE_WAKE_SUBSTRINGS: tuple[str, ...] = (
    "hey service",
    "the service",
    "customer service",
    " church service",
)

# Filler words allowed around a sleep phrase without turning it into ordinary
# conversation ("okay bobe, please go to sleep now" sleeps; "my toddler won't
# go to sleep, any tips?" must not).
SLEEP_COMMAND_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "please",
        "now",
        "ok",
        "okay",
        "hey",
        "bobe",
        "thanks",
        "thank",
        "you",
        "and",
        "παρακαλώ",
        "τώρα",
        "εντάξει",
    }
)
# Beyond this many non-phrase words, treat the transcript as conversation.
SLEEP_COMMAND_MAX_EXTRA_WORDS: int = 4


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
    """Return whether a transcript contains a sleep phrase anywhere.

    Loose substring containment — prefer :func:`matches_sleep_command` for
    deciding whether to actually put BoBe to sleep, since sleep phrases are
    ordinary English n-grams that occur naturally inside sentences.
    """
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    # Normalize the phrase side too so punctuated/decomposed custom phrases
    # can match the normalized transcript.
    candidates = (normalize_transcript(phrase) for phrase in (*phrases, *SLEEP_PHRASE_ASR_VARIANTS))
    return any(candidate in normalized for candidate in candidates if candidate)


def matches_sleep_command(
    text: str,
    phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES,
) -> bool:
    """Return True when a transcript is essentially just a sleep phrase.

    The configured sleep phrases are ordinary English n-grams ("go to sleep")
    that occur naturally inside sentences, so substring containment over full
    conversational transcripts would put the robot to sleep mid-conversation.
    A transcript only counts as a sleep command when, after normalization, it
    is the phrase itself surrounded by nothing but a few filler words. The
    Whisper mishear variants ("got to sleep") are accepted under the same
    strict rule — as near-exact commands, never as substrings of a longer
    sentence.
    """
    normalized = normalize_transcript(text)
    if not normalized:
        return False
    words = normalized.split()
    for phrase in (*phrases, *SLEEP_PHRASE_ASR_VARIANTS):
        # Normalize the phrase side too so punctuated/decomposed custom
        # phrases can match the normalized transcript.
        candidate = normalize_transcript(phrase)
        if not candidate:
            continue
        phrase_words = candidate.split()
        extra_count = len(words) - len(phrase_words)
        if extra_count < 0 or extra_count > SLEEP_COMMAND_MAX_EXTRA_WORDS:
            continue
        for start in range(extra_count + 1):
            if words[start : start + len(phrase_words)] != phrase_words:
                continue
            extras = words[:start] + words[start + len(phrase_words) :]
            if all(word in SLEEP_COMMAND_FILLER_WORDS for word in extras):
                return True
    return False
