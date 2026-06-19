# ruff: noqa: D103
import numpy as np

from bobe.wake_daemon.config import load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine


def test_load_wake_daemon_config_defaults():
    config = load_wake_daemon_config({})

    assert config.phrase == "hey jarvis"
    assert config.whisper_model == "tiny.en"
    assert config.port == 8765


def test_whisper_engine_detects_wake_phrase(monkeypatch):
    config = load_wake_daemon_config({})
    engine = WhisperWakeEngine(config)
    engine.config = config
    monkeypatch.setattr(engine, "_transcribe", lambda _audio: "hey jarvis")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = engine.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe

    assert event is not None
    assert event["type"] == "wake"
    assert event["phrase"] == "hey jarvis"


def test_whisper_engine_ignores_unrelated_speech(monkeypatch):
    config = load_wake_daemon_config({})
    engine = WhisperWakeEngine(config)
    monkeypatch.setattr(engine, "_transcribe", lambda _audio: "good morning")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    events = [engine.feed(pcm[offset : offset + 1600]) for offset in range(0, pcm.size, 1600)]

    assert all(event is None for event in events)
