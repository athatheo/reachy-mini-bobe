# ruff: noqa: D103

from bobe.turn_policy import decide_turn


def test_decide_turn_ignores_empty_text():
    decision = decide_turn("   ")

    assert not decision.should_respond
    assert decision.request_text == ""
    assert decision.reason == "empty"


def test_decide_turn_requires_standalone_wake_word():
    decision = decide_turn("bobcat facts are interesting")

    assert not decision.should_respond
    assert decision.request_text == "bobcat facts are interesting"
    assert decision.reason == "wake_word_missing"


def test_decide_turn_removes_leading_wake_word():
    decision = decide_turn("Bob, what's next?")

    assert decision.should_respond
    assert decision.request_text == "what's next"
    assert decision.reason == "wake_word_present"


def test_decide_turn_removes_greeting_before_wake_word():
    decision = decide_turn("hey bob can you help me")

    assert decision.should_respond
    assert decision.request_text == "can you help me"


def test_decide_turn_preserves_non_filler_words_around_wake_word():
    decision = decide_turn("please Bob check my schedule")

    assert decision.should_respond
    assert decision.request_text == "please check my schedule"


def test_decide_turn_handles_wake_word_only():
    decision = decide_turn("Bob!")

    assert decision.should_respond
    assert decision.request_text == ""
    assert decision.reason == "wake_word_only"


def test_decide_turn_accepts_greek_wake_word_alias():
    decision = decide_turn("Μπομπ, τι μπορείς να κάνεις;")

    assert decision.should_respond
    assert decision.request_text == "τι μπορείς να κάνεις"


def test_decide_turn_rejects_greek_wake_word_inside_longer_word():
    decision = decide_turn("Μπομπάκι τι κάνεις;")

    assert not decision.should_respond
