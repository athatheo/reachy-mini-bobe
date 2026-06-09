"""Local wake-word detection and session gating for BoBe.

While asleep, microphone audio stays on the robot: frames are fed to a local
openWakeWord model and a short ring buffer, and nothing is sent upstream.
After the wake phrase is detected, audio streams to the realtime backend until
the inactivity timeout or the sleep phrase puts BoBe back to sleep.
"""

from __future__ import annotations
import os
import time
import queue
import logging
import threading
from typing import Mapping, Callable
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)


WAKE_SAMPLE_RATE = 16000
DETECTOR_FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, openWakeWord's native frame size

DEFAULT_WAKE_MODEL = "hey_jarvis"
DEFAULT_WAKE_THRESHOLD = 0.5
DEFAULT_WAKE_TIMEOUT_S = 300.0
DEFAULT_SLEEP_PHRASES = ("go to sleep", "κοιμήσου")
DEFAULT_BUFFER_SECONDS = 3.0
DEFAULT_FLUSH_SECONDS = 1.6


@dataclass(frozen=True)
class WakeConfig:
    """Environment-driven configuration for wake-word gating."""

    enabled: bool = True
    model_name: str = DEFAULT_WAKE_MODEL
    threshold: float = DEFAULT_WAKE_THRESHOLD
    timeout_s: float = DEFAULT_WAKE_TIMEOUT_S
    sleep_phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES


def load_wake_config(env: Mapping[str, str] | None = None) -> WakeConfig:
    """Load wake-word settings from environment variables."""
    source = os.environ if env is None else env

    def _float(name: str, default: float) -> float:
        try:
            return float(source.get(name, default))
        except (TypeError, ValueError):
            return default

    sleep_phrases = list(DEFAULT_SLEEP_PHRASES)
    custom_phrase = (source.get("BOBE_SLEEP_PHRASE") or "").strip()
    if custom_phrase and custom_phrase.casefold() not in {p.casefold() for p in sleep_phrases}:
        sleep_phrases.insert(0, custom_phrase)

    return WakeConfig(
        enabled=(source.get("BOBE_WAKE_DISABLED", "0").strip().lower() not in {"1", "true", "yes", "on"}),
        model_name=(source.get("BOBE_WAKE_MODEL") or DEFAULT_WAKE_MODEL).strip() or DEFAULT_WAKE_MODEL,
        threshold=_float("BOBE_WAKE_THRESHOLD", DEFAULT_WAKE_THRESHOLD),
        timeout_s=max(1.0, _float("BOBE_WAKE_TIMEOUT_S", DEFAULT_WAKE_TIMEOUT_S)),
        sleep_phrases=tuple(sleep_phrases),
    )


def is_sleep_phrase(text: str, phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES) -> bool:
    """Return whether a transcript asks BoBe to go back to sleep."""
    normalized = " ".join(text.strip().strip(" \t\n\r,.:;!?-").casefold().split())
    if not normalized:
        return False
    return any(phrase.casefold() in normalized for phrase in phrases if phrase.strip())


class AudioRingBuffer:
    """Fixed-duration mono int16 ring buffer holding pre-wake audio locally."""

    def __init__(self, seconds: float = DEFAULT_BUFFER_SECONDS, sample_rate: int = WAKE_SAMPLE_RATE) -> None:
        """Initialize an empty buffer holding at most ``seconds`` of audio."""
        self._sample_rate = sample_rate
        self._max_samples = max(1, int(seconds * sample_rate))
        self._chunks: deque[NDArray[np.int16]] = deque()
        self._total_samples = 0
        self._lock = threading.Lock()

    def append(self, frame: NDArray[np.int16]) -> None:
        """Append a mono frame, dropping the oldest audio beyond capacity."""
        if frame.size == 0:
            return
        with self._lock:
            self._chunks.append(frame)
            self._total_samples += frame.size
            while self._total_samples > self._max_samples and len(self._chunks) > 1:
                dropped = self._chunks.popleft()
                self._total_samples -= dropped.size

    def drain_tail(self, seconds: float) -> NDArray[np.int16]:
        """Return up to the last ``seconds`` of audio and clear the buffer."""
        with self._lock:
            chunks = list(self._chunks)
            self._chunks.clear()
            self._total_samples = 0
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        samples = np.concatenate(chunks)
        wanted = int(seconds * self._sample_rate)
        return samples[-wanted:] if 0 < wanted < samples.size else samples


class WakeSession:
    """Thread-safe asleep/awake state with an inactivity timeout."""

    def __init__(
        self,
        timeout_s: float = DEFAULT_WAKE_TIMEOUT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize an asleep session with the given timeout."""
        self._timeout_s = timeout_s
        self._clock = clock
        self._lock = threading.Lock()
        self._awake = False
        self._last_activity = clock()
        self._wake_requested = False

    @property
    def awake(self) -> bool:
        """Return whether audio is currently allowed to stream upstream."""
        with self._lock:
            return self._awake

    def request_wake(self) -> None:
        """Flag a wake request from the detector thread."""
        with self._lock:
            self._wake_requested = True

    def consume_wake_request(self) -> bool:
        """Return True once per pending wake request while asleep."""
        with self._lock:
            requested = self._wake_requested and not self._awake
            self._wake_requested = False
            return requested

    def wake(self) -> None:
        """Enter the awake state and reset the inactivity timer."""
        with self._lock:
            self._awake = True
            self._last_activity = self._clock()

    def sleep(self) -> None:
        """Return to the asleep (local-only) state."""
        with self._lock:
            self._awake = False
            self._wake_requested = False

    def touch(self) -> None:
        """Record session activity, resetting the inactivity timer."""
        with self._lock:
            self._last_activity = self._clock()

    def expired(self) -> bool:
        """Return whether the awake session passed the inactivity timeout."""
        with self._lock:
            return self._awake and (self._clock() - self._last_activity) >= self._timeout_s


class WakeWordDetector:
    """Background thread running openWakeWord on locally buffered mic frames."""

    def __init__(
        self,
        on_wake: Callable[[], None],
        *,
        model_name: str = DEFAULT_WAKE_MODEL,
        threshold: float = DEFAULT_WAKE_THRESHOLD,
    ) -> None:
        """Initialize the detector; the model loads lazily in its own thread."""
        self._on_wake = on_wake
        self._model_name = model_name
        self._threshold = threshold
        self._queue: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the detection thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="wake-word-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the detection thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def feed(self, frame: NDArray[np.int16]) -> None:
        """Queue a 16kHz mono int16 frame; drops frames when backlogged."""
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def _load_model(self) -> object | None:
        try:
            from openwakeword.model import Model
        except Exception:
            logger.exception("openwakeword is not available; wake-word detection disabled")
            return None

        try:
            model: object = Model(wakeword_models=[self._model_name])
            return model
        except Exception:
            logger.info("Wake model %r missing, downloading openWakeWord models...", self._model_name)
            try:
                import openwakeword

                openwakeword.utils.download_models()
                model = Model(wakeword_models=[self._model_name])
                return model
            except Exception:
                logger.exception("Failed to load wake model %r; wake-word detection disabled", self._model_name)
                return None

    def _run(self) -> None:
        model = self._load_model()
        if model is None:
            return
        logger.info("Wake-word detector listening for %r (threshold=%.2f)", self._model_name, self._threshold)

        pending = np.zeros(0, dtype=np.int16)
        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            pending = np.concatenate([pending, frame.reshape(-1)])
            while pending.size >= DETECTOR_FRAME_SAMPLES:
                chunk = pending[:DETECTOR_FRAME_SAMPLES]
                pending = pending[DETECTOR_FRAME_SAMPLES:]
                try:
                    scores = model.predict(chunk)  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("Wake-word inference failed; stopping detector")
                    return
                score = max(scores.values()) if scores else 0.0
                if score >= self._threshold:
                    logger.info("Wake word detected (score=%.2f)", score)
                    self._reset_model(model)
                    pending = np.zeros(0, dtype=np.int16)
                    self._drain_queue()
                    self._on_wake()
                    break

    def _reset_model(self, model: object) -> None:
        try:
            model.reset()  # type: ignore[attr-defined]
        except Exception:
            logger.debug("Wake model reset unavailable", exc_info=True)

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
