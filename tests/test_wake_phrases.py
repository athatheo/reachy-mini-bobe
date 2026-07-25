# ruff: noqa: D103
from bobe.wake.phrases import matches_wake_phrase, matches_sleep_phrase, normalize_transcript


def test_normalize_transcript():
    assert normalize_transcript("  Hey Bobe!  ") == "hey bobe"
    assert normalize_transcript("Hey, Bobby.") == "hey bobby"


def test_matches_wake_phrase_exact():
    assert matches_wake_phrase("hey bobe")


def test_matches_wake_phrase_with_command():
    assert matches_wake_phrase("hey bobe what's the weather")


def test_matches_whisper_comma_bobby():
    assert matches_wake_phrase("Hey, Bobby.")


def test_rejects_bare_name_prompt_echo():
    # Whisper initial_prompt often emits just the name — must not wake.
    assert not matches_wake_phrase("bobe")
    assert not matches_wake_phrase("Bobe.")
    assert not matches_wake_phrase("jarvis")


def test_matches_wake_phrase_asr_variants():
    assert matches_wake_phrase("hey bobby")
    assert matches_wake_phrase("hey boby what's up")


def test_rejects_unrelated_speech():
    assert not matches_wake_phrase("good morning")
    assert not matches_wake_phrase("hey there")
    assert not matches_wake_phrase("I'm going to ask you.")


def test_rejects_false_wake_homophones():
    assert not matches_wake_phrase("hey service")
    assert not matches_wake_phrase("hey service please")
    assert not matches_wake_phrase("customer service desk")


def test_custom_phrase_does_not_use_bobe_variants():
    assert not matches_wake_phrase("hey bobby", phrase="hey jarvis")


def test_matches_sleep_phrase():
    assert matches_sleep_phrase("go to sleep")
    assert matches_sleep_phrase("please go to sleep now")
    assert matches_sleep_phrase("got to sleep")
    assert matches_sleep_phrase("κοιμήσου")
    assert not matches_sleep_phrase("hey bobe")
    assert not matches_sleep_phrase("")
