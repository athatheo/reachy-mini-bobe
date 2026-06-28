# ruff: noqa: D103
import numpy as np
import pytest

from bobe.wake_daemon.config import load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine

_TEST_ENV = {"BOBE_WAKE_TOKEN": "test-token"}


def test_load_wake_daemon_config_requires_token():
    with pytest.raises(ValueError, match="BOBE_WAKE_TOKEN"):
        load_wake_daemon_config({})


def _session(config=None, *, monkeypatch=None, transcribe=None):
    runtime = config or load_wake_daemon_config(_TEST_ENV)
    engine = WhisperWakeEngine(runtime)
    session = engine.session(runtime)
    if monkeypatch is not None and transcribe is not None:
        monkeypatch.setattr(engine, "transcribe", lambda pcm, *, config=None: transcribe(pcm))
    return session


def test_load_wake_daemon_config_defaults():
    config = load_wake_daemon_config(_TEST_ENV)

    assert config.phrase == "hey jarvis"
    assert config.whisper_model == "base.en"
    assert config.whisper_initial_prompt == "Jarvis."
    assert config.whisper_hotwords is None
    assert config.port == 8765


def test_whisper_prompt_helpers():
    from bobe.wake_daemon.config import whisper_hotwords_from_phrase, whisper_initial_prompt_from_phrase

    assert whisper_initial_prompt_from_phrase("hey jarvis") == "Jarvis."
    assert whisper_hotwords_from_phrase("hey jarvis") == "Hey Jarvis Jarvis"


def test_whisper_prompt_env_override():
    config = load_wake_daemon_config(
        {
            **_TEST_ENV,
            "WHISPER_INITIAL_PROMPT": "Jarvis.",
            "WHISPER_HOTWORDS": "Jarvis",
        }
    )
    assert config.whisper_initial_prompt == "Jarvis."
    assert config.whisper_hotwords == "Jarvis"


def test_whisper_engine_detects_wake_phrase(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "hey jarvis")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = session.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe

    assert event is not None
    assert event["type"] == "wake"
    assert event["phrase"] == "hey jarvis"


def test_whisper_engine_ignores_unrelated_speech(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "good morning")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    events = [session.feed(pcm[offset : offset + 1600]) for offset in range(0, pcm.size, 1600)]

    assert all(event is None for event in events)


def test_whisper_session_listen_modes_are_isolated():
    config = load_wake_daemon_config(_TEST_ENV)
    engine = WhisperWakeEngine(config)
    session_a = engine.session(config)
    session_b = engine.session(config)

    session_a.set_listen_mode("sleep")
    session_b.set_listen_mode("wake")

    assert session_a.debug_state()["listen_mode"] == "sleep"
    assert session_b.debug_state()["listen_mode"] == "wake"

    pcm = np.zeros(1600, dtype=np.int16)
    pcm[:] = 5000
    assert session_a.feed(pcm) is None
    assert session_b.feed(pcm) is None
    assert session_b.debug_state()["in_speech"] is True
    assert session_a.debug_state()["in_speech"] is True


def test_whisper_engine_detects_sleep_phrase(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "go to sleep")
    session.set_listen_mode("sleep")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = session.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe

    assert event is not None
    assert event["type"] == "sleep"


def test_whisper_engine_loads_model_once(monkeypatch):
    config = load_wake_daemon_config(_TEST_ENV)
    engine = WhisperWakeEngine(config)
    load_calls = {"count": 0}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return ([], None)

    def fake_load():
        if engine._model is not None:
            return engine._model
        load_calls["count"] += 1
        engine._model = FakeModel()
        return engine._model

    monkeypatch.setattr(engine, "_load_model", fake_load)

    pcm = np.zeros(1600, dtype=np.int16)
    engine.transcribe(pcm)
    engine.transcribe(pcm)

    assert load_calls["count"] == 1


def test_wake_daemon_app_starts_with_empty_engine_pool():
    from bobe.wake_daemon.server import create_app

    config = load_wake_daemon_config(_TEST_ENV)
    app = create_app(config)
    assert app.state.wake_engines == {}
