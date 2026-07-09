# ruff: noqa: D103
from bobe.wake.phrases import matches_wake_phrase, matches_sleep_phrase, normalize_transcript


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


def test_rejects_false_wake_homophones():
    assert not matches_wake_phrase("hey service")
    assert not matches_wake_phrase("hey service please")
    assert not matches_wake_phrase("customer service desk")


def test_matches_sleep_phrase():
    assert matches_sleep_phrase("go to sleep")
    assert matches_sleep_phrase("please go to sleep now")
    assert matches_sleep_phrase("got to sleep")
    assert matches_sleep_phrase("κοιμήσου")
    assert not matches_sleep_phrase("hey jarvis")
    assert not matches_sleep_phrase("")
