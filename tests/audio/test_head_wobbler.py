"""Regression tests for the audio-driven head wobble behaviour."""

import math
import time
import base64
import threading
from typing import Any, List, Tuple
from collections.abc import Callable

import numpy as np

from reachy_mini.utils import create_head_pose
from reachy_mini.motion.move import Move
from bobe.moves import (
    SECONDARY_ROTATION_LIMIT_RAD,
    SECONDARY_TRANSLATION_LIMIT_M,
    MovementManager,
)
from bobe.audio.head_wobbler import HeadWobbler
from bobe.dance_emotion_moves import GotoQueueMove


NEUTRAL_OFFSETS = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _make_audio_chunk(duration_s: float = 0.3, frequency_hz: float = 220.0) -> str:
    """Generate a base64-encoded mono PCM16 sine wave."""
    sample_rate = 24000
    sample_count = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, sample_count, endpoint=False)
    wave = 0.6 * np.sin(2 * math.pi * frequency_hz * t)
    pcm = np.clip(wave * np.iinfo(np.int16).max, -32768, 32767).astype(np.int16)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _wait_for(predicate: Callable[[], bool], timeout: float = 0.6) -> bool:
    """Poll `predicate` until true or timeout."""
    end_time = time.time() + timeout
    while time.time() < end_time:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def _start_wobbler() -> Tuple[HeadWobbler, List[Tuple[float, Tuple[float, float, float, float, float, float]]]]:
    captured: List[Tuple[float, Tuple[float, float, float, float, float, float]]] = []

    def capture(offsets: Tuple[float, float, float, float, float, float]) -> None:
        captured.append((time.time(), offsets))

    wobbler = HeadWobbler(set_speech_offsets=capture)
    wobbler.start()
    return wobbler, captured


def test_reset_drops_pending_offsets() -> None:
    """Reset should stop wobble output derived from pre-reset audio."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        pre_reset_count = len(captured)
        wobbler.reset()
        time.sleep(0.3)
        new_offsets = [offsets for _, offsets in captured[pre_reset_count:]]
        assert all(offsets == NEUTRAL_OFFSETS for offsets in new_offsets), (
            "offsets continued after reset without new audio"
        )
    finally:
        wobbler.stop()


def test_reset_pushes_neutral_offsets() -> None:
    """Reset must zero the APPLIED offsets so the head is not left mid-sway."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        wobbler.reset()
        offsets = [captured_offsets for _, captured_offsets in captured]
        assert NEUTRAL_OFFSETS in offsets, "reset did not push a neutral offset snapshot"
        assert offsets[-1] == NEUTRAL_OFFSETS, "head left with non-neutral offsets after reset"
    finally:
        wobbler.stop()


def test_stop_returns_promptly_while_pacing_audio() -> None:
    """stop() must not block while the worker paces out a long queued chunk."""
    wobbler, captured = _start_wobbler()
    stopped = False
    try:
        wobbler.feed(_make_audio_chunk(duration_s=2.0))
        assert _wait_for(lambda: len(captured) > 0, timeout=1.5), "wobbler did not start pacing"

        start = time.monotonic()
        wobbler.stop()
        stopped = True
        assert time.monotonic() - start < 1.0, "stop() blocked while pacing a queued chunk"
    finally:
        if not stopped:
            wobbler.stop()


def test_reset_allows_future_offsets() -> None:
    """After reset, fresh audio must still produce wobble offsets."""
    wobbler, captured = _start_wobbler()
    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert _wait_for(lambda: len(captured) > 0), "wobbler did not emit initial offsets"

        wobbler.reset()
        pre_second_count = len(captured)

        wobbler.feed(_make_audio_chunk(duration_s=0.35, frequency_hz=440.0))
        assert _wait_for(lambda: len(captured) > pre_second_count), "no offsets after reset"
        assert wobbler._thread is not None and wobbler._thread.is_alive()
    finally:
        wobbler.stop()


def test_reset_during_inflight_chunk_keeps_worker(monkeypatch: Any) -> None:
    """Simulate reset during chunk processing to ensure the worker survives."""
    wobbler, captured = _start_wobbler()
    ready = threading.Event()
    release = threading.Event()

    original_feed = wobbler.sway.feed

    def blocking_feed(pcm, sr):  # type: ignore[no-untyped-def]
        ready.set()
        release.wait(timeout=2.0)
        return original_feed(pcm, sr)

    monkeypatch.setattr(wobbler.sway, "feed", blocking_feed)

    try:
        wobbler.feed(_make_audio_chunk(duration_s=0.35))
        assert ready.wait(timeout=1.0), "worker thread did not dequeue audio"

        wobbler.reset()
        release.set()

        # Allow the worker to finish processing the first chunk (which should be discarded)
        time.sleep(0.1)

        assert wobbler._thread is not None and wobbler._thread.is_alive(), "worker thread died after reset"

        pre_second = len(captured)
        wobbler.feed(_make_audio_chunk(duration_s=0.35, frequency_hz=440.0))
        assert _wait_for(lambda: len(captured) > pre_second), "no offsets emitted after in-flight reset"
        assert wobbler._thread.is_alive()
    finally:
        wobbler.stop()


# --- MovementManager: consumer of the wobbler's speech offsets ---


class _FakeRobot:
    """Minimal ReachyMini stand-in for MovementManager tests."""

    def __init__(self) -> None:
        self.set_target_calls = 0
        self.last_target: Tuple[Any, Any, Any] | None = None
        self.goto_calls = 0

    def set_target(self, head: Any = None, antennas: Any = None, body_yaw: Any = None) -> None:
        self.set_target_calls += 1
        self.last_target = (head, antennas, body_yaw)

    def goto_target(self, head: Any = None, antennas: Any = None, duration: Any = None, body_yaw: Any = None) -> None:
        self.goto_calls += 1

    def get_current_joint_positions(self) -> Tuple[Any, Any]:
        return np.zeros(7), np.zeros(2)

    def get_current_head_pose(self) -> Any:
        return np.eye(4)


class _ExplodingMove(Move):  # type: ignore[misc]
    """Move whose evaluation always raises, simulating a poisoned tick."""

    @property
    def duration(self) -> float:
        return 1.0

    def evaluate(self, t: float) -> Any:
        raise RuntimeError("boom")


def test_movement_manager_survives_tick_errors() -> None:
    """A raising move must not kill the 100 Hz control loop (finding #24)."""
    robot = _FakeRobot()
    manager = MovementManager(robot)
    manager.start()
    try:
        assert _wait_for(lambda: robot.set_target_calls > 0), "control loop never issued a command"

        manager.queue_move(_ExplodingMove())
        assert _wait_for(lambda: manager.state.current_move is None and manager._command_queue.empty(), timeout=1.0)

        before = robot.set_target_calls
        assert _wait_for(lambda: robot.set_target_calls > before + 5, timeout=1.0), (
            "control loop stopped issuing commands after a tick error"
        )
        assert manager._thread is not None and manager._thread.is_alive()

        # A follow-up move still runs to completion.
        goto = GotoQueueMove(
            target_head_pose=create_head_pose(0, 0, 0, 0, 0, 0, degrees=True),
            start_head_pose=np.eye(4),
            target_body_yaw=0.5,
            start_body_yaw=0.0,
            duration=0.05,
        )
        manager.queue_move(goto)
        assert _wait_for(
            lambda: robot.last_target is not None and robot.last_target[2] > 0.4,
            timeout=1.0,
        ), "follow-up move did not execute after a tick error"
    finally:
        manager.stop()


def test_secondary_offsets_are_clamped_and_sanitized() -> None:
    """Fused speech+face offsets are bounded and NaN-safe before composition."""
    manager = MovementManager(_FakeRobot())
    manager.state.speech_offsets = (1.0, -1.0, float("nan"), 2.0, -2.0, 1.0)
    manager.state.face_tracking_offsets = NEUTRAL_OFFSETS

    head_pose, antennas, body_yaw = manager._get_secondary_pose()

    expected = create_head_pose(
        x=SECONDARY_TRANSLATION_LIMIT_M,
        y=-SECONDARY_TRANSLATION_LIMIT_M,
        z=0.0,
        roll=SECONDARY_ROTATION_LIMIT_RAD,
        pitch=-SECONDARY_ROTATION_LIMIT_RAD,
        yaw=SECONDARY_ROTATION_LIMIT_RAD,
        degrees=False,
        mm=False,
    )
    assert np.all(np.isfinite(head_pose))
    assert np.allclose(head_pose, expected)
    assert antennas == (0.0, 0.0)
    assert body_yaw == 0.0


def test_listening_debounce_coalesces_to_latest_state() -> None:
    """Burst toggles inside the debounce window are deferred, not dropped (finding #50)."""
    manager = MovementManager(_FakeRobot())
    clock = {"t": 1000.0}
    manager._now = lambda: clock["t"]  # type: ignore[method-assign]
    manager._last_listening_toggle_time = clock["t"]

    clock["t"] += 1.0
    manager._request_listening(True)
    assert manager._is_listening is True

    # Opposite toggle arriving within the debounce window is deferred...
    clock["t"] += 0.05
    manager._request_listening(False)
    manager._flush_pending_listening()
    assert manager._is_listening is True, "toggle applied inside the debounce window"

    # ...and applied once the window expires (instead of being dropped).
    clock["t"] += 0.2
    manager._flush_pending_listening()
    assert manager._is_listening is False, "deferred toggle was dropped"

    # Several queued toggles coalesce to the most recent requested state.
    clock["t"] += 0.05
    manager._request_listening(True)
    manager._request_listening(False)
    manager._request_listening(True)
    clock["t"] += 0.2
    manager._flush_pending_listening()
    assert manager._is_listening is True


def test_set_listening_producer_does_not_drop_return_toggle() -> None:
    """The producer-side dedupe must track requested (not lagging applied) state."""
    manager = MovementManager(_FakeRobot())

    manager.set_listening(True)
    manager.set_listening(False)

    commands = []
    while not manager._command_queue.empty():
        commands.append(manager._command_queue.get_nowait())
    assert commands == [("set_listening", True), ("set_listening", False)]
