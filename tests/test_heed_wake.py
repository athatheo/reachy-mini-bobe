# ruff: noqa: D101,D102,D103

"""Tests for Heed wake-word inference."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from bobe.heed_wake import HeedWakeWordDetector, load_heed_meta, default_heed_model_dir


POSITIVE_SAMPLE = Path(__file__).resolve().parents[1] / "wake_training" / "hey_jarvis" / "positive" / "seed_pos_000.wav"


def test_load_heed_meta_from_bundled_model():
    meta = load_heed_meta(default_heed_model_dir())

    assert meta.phrase == "hey jarvis"
    assert meta.threshold > 0.5
    assert meta.consecutive_frames >= 1


def test_heed_detector_triggers_on_positive_sample():
    if not POSITIVE_SAMPLE.is_file():
        pytest.skip("training positive sample not available")

    audio, sr = sf.read(POSITIVE_SAMPLE, dtype="float32")
    assert sr == 16000
    pcm = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)

    fired = {"value": False}

    detector = HeedWakeWordDetector(lambda: fired.__setitem__("value", True), threshold=0.5)
    detector.start()
    try:
        hop = 1600
        for start in range(0, pcm.size, hop):
            detector.feed(pcm[start : start + hop])
        deadline = __import__("time").time() + 5.0
        while __import__("time").time() < deadline and not fired["value"]:
            __import__("time").sleep(0.05)
    finally:
        detector.stop()

    assert fired["value"]
    assert detector.debug_state()["score_peak"] > 0.5
