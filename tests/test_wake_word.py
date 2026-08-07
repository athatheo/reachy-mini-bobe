# ruff: noqa: D101,D102,D103,D107

import pytest

from bobe.wake_word import (
    WakeSession,
    AudioRingBuffer,
    load_wake_config,
    wake_detector_error,
    create_wake_detector,
)
from bobe.wake.remote_client import RemoteWakeClient


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---- WakeSession ----


def test_wake_session_starts_asleep_and_wakes_on_request():
    session = WakeSession()

    assert not session.awake
    assert not session.consume_wake_request()

    session.request_wake()
    assert session.consume_wake_request()
    assert not session.consume_wake_request()

    session.wake()
    assert session.awake


def test_wake_session_ignores_wake_request_while_awake():
    session = WakeSession()
    session.wake()

    session.request_wake()
    assert not session.consume_wake_request()


def test_wake_session_expires_after_timeout_and_touch_resets_it():
    clock = FakeClock()
    session = WakeSession(timeout_s=300.0, clock=clock)
    session.wake()

    clock.advance(299.0)
    assert not session.expired()

    session.touch()
    clock.advance(299.0)
    assert not session.expired()

    clock.advance(2.0)
    assert session.expired()

    session.sleep()
    assert not session.expired()
    assert not session.awake


# ---- AudioRingBuffer ----


def test_ring_buffer_keeps_only_the_most_recent_audio():
    import numpy as np

    buffer = AudioRingBuffer(seconds=1.0, sample_rate=100)

    for value in range(20):
        buffer.append(np.full(10, value, dtype=np.int16))

    tail = buffer.drain_tail(seconds=10.0)
    assert tail.size == 100
    assert tail.min() == 10
    assert tail.max() == 19


def test_ring_buffer_drain_tail_slices_and_clears():
    import numpy as np

    buffer = AudioRingBuffer(seconds=2.0, sample_rate=100)
    buffer.append(np.arange(200, dtype=np.int16))

    tail = buffer.drain_tail(seconds=1.0)
    assert tail.size == 100
    assert tail[0] == 100


# ---- load_wake_config / create_wake_detector ----


def test_load_wake_config_defaults():
    config = load_wake_config({})

    assert config.backend == "remote"
    assert config.gain == 1.75
    assert config.timeout_s == 300.0
    assert config.phrase == "hey bobe"
    assert "go to sleep" in config.sleep_phrases


def test_load_wake_config_env_overrides():
    config = load_wake_config(
        {
            "BOBE_WAKE_BACKEND": "remote",
            "BOBE_WAKE_REMOTE_URL": "ws://mac-mini.local:8765/v1/stream",
            "BOBE_WAKE_TOKEN": "secret",
            "BOBE_WAKE_GAIN": "3.5",
            "BOBE_WAKE_TIMEOUT_S": "60",
            "BOBE_WAKE_PHRASE": "Hey BoBe",
            "BOBE_SLEEP_PHRASE": "time for bed",
        }
    )

    assert config.backend == "remote"
    assert config.remote_url == "ws://mac-mini.local:8765/v1/stream"
    assert config.remote_token == "secret"
    assert config.gain == 3.5
    assert config.timeout_s == 60.0
    assert config.phrase == "hey bobe"
    assert config.sleep_phrases[0] == "time for bed"


def test_load_wake_config_remote_backend():
    config = load_wake_config(
        {
            "BOBE_WAKE_BACKEND": "remote",
            "BOBE_WAKE_REMOTE_URL": "ws://mac-mini.local:8765/v1/stream",
            "BOBE_WAKE_TOKEN": "secret",
        }
    )

    assert config.backend == "remote"
    assert config.remote_url == "ws://mac-mini.local:8765/v1/stream"
    assert config.remote_token == "secret"


def test_create_wake_detector_remote_requires_url():
    config = load_wake_config({"BOBE_WAKE_BACKEND": "remote"})
    assert create_wake_detector(lambda: None, config) is None


def test_create_wake_detector_remote_requires_token():
    config = load_wake_config(
        {
            "BOBE_WAKE_BACKEND": "remote",
            "BOBE_WAKE_REMOTE_URL": "ws://mac-mini.local:8765/v1/stream",
        }
    )
    assert create_wake_detector(lambda: None, config) is None


def test_wake_session_sleep_request():
    session = WakeSession(timeout_s=60.0)
    session.wake()
    session.request_sleep()
    assert session.consume_sleep_request()
    assert not session.consume_sleep_request()


def test_create_wake_detector_remote_returns_client():
    config = load_wake_config(
        {
            "BOBE_WAKE_BACKEND": "remote",
            "BOBE_WAKE_REMOTE_URL": "ws://mac-mini.local:8765/v1/stream",
            "BOBE_WAKE_TOKEN": "secret",
            "BOBE_WAKE_PHRASE": "hey bobe",
        }
    )
    detector = create_wake_detector(lambda: None, config)
    assert isinstance(detector, RemoteWakeClient)
    assert detector.phrase == "hey bobe"


@pytest.mark.parametrize("backend", ["heed", "openwakeword"])
def test_create_wake_detector_rejects_deprecated_backends(backend):
    config = load_wake_config({"BOBE_WAKE_BACKEND": backend})
    assert create_wake_detector(lambda: None, config) is None


@pytest.mark.parametrize(
    ("env", "expected_substring"),
    [
        ({"BOBE_WAKE_BACKEND": "remote"}, "BOBE_WAKE_REMOTE_URL"),
        (
            {
                "BOBE_WAKE_BACKEND": "remote",
                "BOBE_WAKE_REMOTE_URL": "ws://mac-mini.local:8765/v1/stream",
            },
            "BOBE_WAKE_TOKEN",
        ),
        ({"BOBE_WAKE_BACKEND": "heed"}, "no longer supported"),
        ({"BOBE_WAKE_BACKEND": "bogus"}, "Unknown wake backend"),
    ],
)
def test_wake_detector_error(env, expected_substring):
    config = load_wake_config(env)
    error = wake_detector_error(config)
    assert error is not None
    assert expected_substring in error


def test_wake_detector_error_none_when_remote_configured():
    config = load_wake_config(
        {
            "BOBE_WAKE_BACKEND": "remote",
            "BOBE_WAKE_REMOTE_URL": "ws://mac-mini.local:8765/v1/stream",
            "BOBE_WAKE_TOKEN": "secret",
        }
    )
    assert wake_detector_error(config) is None


def test_wake_session_sleep_requested_property():
    session = WakeSession()
    assert not session.sleep_requested
    session.request_sleep()
    assert session.sleep_requested


# ---- speech downlink queue ----


def test_wake_gate_queues_and_drains_speech_clips():
    import numpy as np

    from bobe.wake_word import WakeGate, WakeConfig

    gate = WakeGate(
        input_sample_rate=24000,
        config=WakeConfig(remote_url="ws://mac:8765/v1/stream", remote_token="secret"),
        detector_factory=lambda **kwargs: None,
    )
    pcm = np.ones(100, dtype=np.int16)

    gate.request_speech(pcm, 24000)
    gate.request_speech(np.zeros(0, dtype=np.int16), 24000)  # empty: ignored
    gate.request_speech(pcm, 0)  # bad rate: ignored

    clips = gate.drain_speech()
    assert len(clips) == 1
    assert np.array_equal(clips[0][0], pcm)
    assert clips[0][1] == 24000
    assert gate.drain_speech() == []


def test_wake_gate_supports_legacy_detector_factories():
    from bobe.wake_word import WakeGate, WakeConfig

    calls = []

    def legacy_factory(on_wake, config, on_sleep=None, on_announce=None):
        calls.append(True)
        return None

    gate = WakeGate(
        input_sample_rate=24000,
        config=WakeConfig(remote_url="ws://mac:8765/v1/stream", remote_token="secret"),
        detector_factory=legacy_factory,
    )

    assert calls == [True]
    assert gate.detector is None


def test_create_wake_detector_passes_speech_callback():
    from bobe.wake_word import WakeConfig, create_wake_detector

    config = WakeConfig(remote_url="ws://mac:8765/v1/stream", remote_token="secret")
    received = []
    detector = create_wake_detector(
        lambda: None,
        config,
        on_speech=lambda pcm, rate: received.append((pcm, rate)),
    )

    assert detector is not None
    import numpy as np

    pcm = np.arange(4, dtype=np.int16)
    detector._on_speak(pcm, 24000)
    assert len(received) == 1
