# ruff: noqa: D103
from bobe.wake.phrases import matches_wake_phrase, normalize_transcript


def test_normalize_transcript():
    assert normalize_transcript("  Hey Jarvis!  ") == "hey jarvis"


def test_matches_wake_phrase_exact():
    assert matches_wake_phrase("hey jarvis")


def test_matches_wake_phrase_with_command():
    assert matches_wake_phrase("hey jarvis what's the weather")


def test_matches_wake_phrase_jarvis_prefix():
    assert matches_wake_phrase("jarvis turn on the lights")


def test_rejects_unrelated_speech():
    assert not matches_wake_phrase("good morning")
    assert not matches_wake_phrase("hey there")
