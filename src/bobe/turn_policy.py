"""Pure turn-policy helpers for BoBe wake-word handling."""

from __future__ import annotations
import re
from dataclasses import dataclass

from bobe.claude import DEFAULT_WAKE_WORD, should_respond_to_wake_word


_LEADING_FILLERS = {"hey", "hi", "hello", "ok", "okay", "yo"}
_DEFAULT_WAKE_WORD_ALIASES = ("Bob", "Μπομπ", "Μπόμπ")


@dataclass(frozen=True)
class TurnDecision:
    """Decision for a transcribed user turn."""

    should_respond: bool
    request_text: str
    reason: str


def decide_turn(text: str, wake_word: str = DEFAULT_WAKE_WORD) -> TurnDecision:
    """Decide whether BoBe should respond to transcribed speech."""
    normalized = _normalize_space(text)
    if not normalized:
        return TurnDecision(False, "", "empty")

    match = _wake_word_match(normalized, wake_word)
    if match is None:
        return TurnDecision(False, normalized, "wake_word_missing")

    request_text = _remove_wake_word(normalized, match)
    reason = "wake_word_only" if not request_text else "wake_word_present"
    return TurnDecision(True, request_text, reason)


def _wake_word_match(text: str, wake_word: str) -> re.Match[str] | None:
    for word in _wake_word_aliases(wake_word):
        if not should_respond_to_wake_word(text, word):
            continue
        return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text, re.IGNORECASE)
    return None


def _wake_word_aliases(wake_word: str) -> tuple[str, ...]:
    word = wake_word.strip()
    if not word:
        return ()
    if word.casefold() == DEFAULT_WAKE_WORD.casefold():
        return _DEFAULT_WAKE_WORD_ALIASES
    return (word,)


def _remove_wake_word(text: str, match: re.Match[str]) -> str:
    before = _strip_turn_punctuation(text[: match.start()])
    after = _strip_turn_punctuation(text[match.end() :])

    if _only_leading_fillers(before):
        return after
    if before and after:
        return _normalize_space(f"{before} {after}")
    return before or after


def _only_leading_fillers(text: str) -> bool:
    if not text:
        return True
    return all(part.lower() in _LEADING_FILLERS for part in text.split())


def _strip_turn_punctuation(text: str) -> str:
    return _normalize_space(text.strip(" \t\n\r,.:;!?-"))


def _normalize_space(text: str) -> str:
    return " ".join(text.strip().split())
