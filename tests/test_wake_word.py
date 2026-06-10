# ruff: noqa: D101,D102,D103,D107

import time

import numpy as np

from bobe.wake_word import (
    DETECTOR_FRAME_SAMPLES,
    WakeSession,
    AudioRingBuffer,
    WakeWordDetector,
    is_sleep_phrase,
    load_wake_config,
)


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
    buffer = AudioRingBuffer(seconds=1.0, sample_rate=100)

    # 200 samples in 10-sample chunks; capacity is 100 samples.
    for value in range(20):
        buffer.append(np.full(10, value, dtype=np.int16))

    tail = buffer.drain_tail(seconds=10.0)
    assert tail.size == 100
    assert tail.min() == 10  # the oldest half was discarded locally
    assert tail.max() == 19


def test_ring_buffer_drain_tail_slices_and_clears():
    buffer = AudioRingBuffer(seconds=2.0, sample_rate=100)
    buffer.append(np.arange(200, dtype=np.int16))

    tail = buffer.drain_tail(seconds=0.5)
    assert tail.size == 50
    assert tail[0] == 150

    assert buffer.drain_tail(seconds=1.0).size == 0


# ---- Sleep phrase ----


def test_sleep_phrase_matches_english_and_greek():
    assert is_sleep_phrase("Go to sleep!")
    assert is_sleep_phrase("okay bobe, go to sleep now")
    assert is_sleep_phrase("Κοιμήσου")
    assert not is_sleep_phrase("let's talk about sleep schedules")
    assert not is_sleep_phrase("")


# ---- Config ----


def test_load_wake_config_defaults():
    config = load_wake_config({})

    assert config.enabled
    assert config.model_name == "hey_jarvis"
    assert config.threshold == 0.35
    assert config.gain == 2.0
    assert config.timeout_s == 300.0
    assert "go to sleep" in config.sleep_phrases


def test_load_wake_config_env_overrides():
    config = load_wake_config(
        {
            "BOBE_WAKE_DISABLED": "1",
            "BOBE_WAKE_MODEL": "alexa",
            "BOBE_WAKE_THRESHOLD": "0.7",
            "BOBE_WAKE_GAIN": "3.5",
            "BOBE_WAKE_TIMEOUT_S": "60",
            "BOBE_SLEEP_PHRASE": "time for bed",
        }
    )

    assert not config.enabled
    assert config.model_name == "alexa"
    assert config.threshold == 0.7
    assert config.gain == 3.5
    assert config.timeout_s == 60.0
    assert config.sleep_phrases[0] == "time for bed"


# ---- Detector ----


class FakeWakeModel:
    def __init__(self, fire_on_chunk: int) -> None:
        self.fire_on_chunk = fire_on_chunk
        self.chunks_seen = 0
        self.reset_called = False

    def predict(self, chunk):
        assert len(chunk) == DETECTOR_FRAME_SAMPLES
        self.chunks_seen += 1
        return {"hey_jarvis": 0.9 if self.chunks_seen >= self.fire_on_chunk else 0.0}

    def reset(self) -> None:
        self.reset_called = True


def test_detector_fires_callback_when_threshold_crossed(monkeypatch):
    fired = []
    detector = WakeWordDetector(on_wake=lambda: fired.append(True), threshold=0.5)
    fake_model = FakeWakeModel(fire_on_chunk=2)
    monkeypatch.setattr(detector, "_load_model", lambda: fake_model)

    detector.start()
    try:
        for _ in range(4):
            detector.feed(np.zeros(DETECTOR_FRAME_SAMPLES, dtype=np.int16))
        deadline = time.time() + 3.0
        while not fired and time.time() < deadline:
            time.sleep(0.05)
    finally:
        detector.stop()

    assert fired
    assert fake_model.reset_called


def test_detector_applies_gain_to_quiet_audio(monkeypatch):
    seen_max = []

    class RecordingModel:
        def predict(self, chunk):
            seen_max.append(int(np.abs(chunk).max()))
            return {"hey_jarvis": 0.0}

        def reset(self):
            pass

    detector = WakeWordDetector(on_wake=lambda: None, threshold=0.5, gain=2.0)
    monkeypatch.setattr(detector, "_load_model", lambda: RecordingModel())

    detector.start()
    try:
        detector.feed(np.full(DETECTOR_FRAME_SAMPLES, 1000, dtype=np.int16))
        deadline = time.time() + 3.0
        while not seen_max and time.time() < deadline:
            time.sleep(0.05)
    finally:
        detector.stop()

    assert seen_max and seen_max[0] == 2000


def test_detector_accumulates_partial_frames(monkeypatch):
    fired = []
    detector = WakeWordDetector(on_wake=lambda: fired.append(True), threshold=0.5)
    fake_model = FakeWakeModel(fire_on_chunk=1)
    monkeypatch.setattr(detector, "_load_model", lambda: fake_model)

    detector.start()
    try:
        # Feed half-frames; the detector must assemble full 80ms chunks itself.
        for _ in range(3):
            detector.feed(np.zeros(DETECTOR_FRAME_SAMPLES // 2, dtype=np.int16))
        deadline = time.time() + 3.0
        while not fired and time.time() < deadline:
            time.sleep(0.05)
    finally:
        detector.stop()

    assert fired
