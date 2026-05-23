# ruff: noqa: D103

import pytest

from bobe.emotion_policy import classify_emotion


@pytest.mark.parametrize(
    ("text", "emotion"),
    [
        ("Great, I can help with that.", "happy"),
        ("I am sorry, that did not work.", "sad"),
        ("Wow, that is unexpected.", "surprised"),
        ("I am curious about that.", "curious"),
        ("Let me check before answering.", "thinking"),
    ],
)
def test_classify_emotion_maps_clear_signals(text, emotion):
    decision = classify_emotion(text)

    assert decision.emotion == emotion
    assert decision.should_play_emotion


def test_classify_emotion_defaults_to_neutral():
    decision = classify_emotion("The next event is at three.")

    assert decision.emotion == "neutral"
    assert decision.reason == "no_clear_signal"
    assert not decision.should_play_emotion


def test_classify_emotion_handles_empty_text():
    decision = classify_emotion("")

    assert decision.emotion == "neutral"
    assert decision.reason == "empty"
    assert not decision.should_play_emotion
