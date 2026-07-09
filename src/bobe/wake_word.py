"""Wake-word session gating for BoBe.

While asleep, microphone audio stays on the robot and is streamed to the Mac
wake daemon (Whisper), which listens for the wake phrase. Nothing is sent to
OpenAI until wake. After wake, audio streams to the realtime backend and the
daemon switches to sleep-phrase detection until timeout, local sleep, or the
OpenAI transcript fallback.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from typing import Mapping, Callable
from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from bobe.env_utils import parse_float
from bobe.wake.constants import WAKE_SAMPLE_RATE
from bobe.wake.phrases import DEFAULT_SLEEP_PHRASES, WAKE_PHRASE
from bobe.wake.remote_client import RemoteWakeClient


logger = logging.getLogger(__name__)

DEFAULT_WAKE_BACKEND = "remote"
DEFAULT_WAKE_GAIN = 1.75
DEFAULT_WAKE_TIMEOUT_S = 300.0
DEFAULT_BUFFER_SECONDS = 3.0
DEFAULT_FLUSH_SECONDS = 1.6
_DEPRECATED_BACKENDS = frozenset({"heed", "openwakeword"})


@dataclass(frozen=True)
class WakeConfig:
    """Environment-driven configuration for wake-word gating."""

    backend: str = DEFAULT_WAKE_BACKEND
    gain: float = DEFAULT_WAKE_GAIN
    timeout_s: float = DEFAULT_WAKE_TIMEOUT_S
    phrase: str = WAKE_PHRASE
    sleep_phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES
    remote_url: str | None = None
    remote_token: str | None = None


def load_wake_config(env: Mapping[str, str] | None = None) -> WakeConfig:
    """Load wake-word settings from environment variables."""
    source = os.environ if env is None else env

    def _float(name: str, default: float) -> float:
        return parse_float(source.get(name), default)

    sleep_phrases = list(DEFAULT_SLEEP_PHRASES)
    custom_phrase = (source.get("BOBE_SLEEP_PHRASE") or "").strip()
    if custom_phrase and custom_phrase.casefold() not in {p.casefold() for p in sleep_phrases}:
        sleep_phrases.insert(0, custom_phrase)

    backend = (source.get("BOBE_WAKE_BACKEND") or DEFAULT_WAKE_BACKEND).strip().lower()

    return WakeConfig(
        backend=backend,
        gain=max(1.0, _float("BOBE_WAKE_GAIN", DEFAULT_WAKE_GAIN)),
        timeout_s=max(1.0, _float("BOBE_WAKE_TIMEOUT_S", DEFAULT_WAKE_TIMEOUT_S)),
        phrase=(source.get("BOBE_WAKE_PHRASE") or WAKE_PHRASE).strip().casefold() or WAKE_PHRASE,
        sleep_phrases=tuple(sleep_phrases),
        remote_url=(source.get("BOBE_WAKE_REMOTE_URL") or "").strip() or None,
        remote_token=(source.get("BOBE_WAKE_TOKEN") or "").strip() or None,
    )


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
        self._sleep_requested = False

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

    def request_sleep(self) -> None:
        """Flag a sleep request from the detector thread."""
        with self._lock:
            self._sleep_requested = True

    def consume_sleep_request(self) -> bool:
        """Return True once per pending sleep request while awake."""
        with self._lock:
            requested = self._sleep_requested and self._awake
            self._sleep_requested = False
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
            self._sleep_requested = False

    def touch(self) -> None:
        """Record session activity, resetting the inactivity timer."""
        with self._lock:
            self._last_activity = self._clock()

    def expired(self) -> bool:
        """Return whether the awake session passed the inactivity timeout."""
        with self._lock:
            return self._awake and (self._clock() - self._last_activity) >= self._timeout_s


def wake_detector_error(config: WakeConfig) -> str | None:
    """Return a user-visible error when wake detection cannot start."""
    backend = config.backend
    if backend in _DEPRECATED_BACKENDS:
        return (
            f"BOBE_WAKE_BACKEND={backend!r} is no longer supported; "
            "use remote with BOBE_WAKE_REMOTE_URL"
        )
    if backend == "remote":
        if not config.remote_url:
            return "BOBE_WAKE_REMOTE_URL is required when BOBE_WAKE_BACKEND=remote"
        return None
    return f"Unknown wake backend {backend!r}; wake-word detection disabled"


def create_wake_detector(
    on_wake: Callable[[], None],
    config: WakeConfig,
    *,
    on_sleep: Callable[[], None] | None = None,
) -> RemoteWakeClient | None:
    """Instantiate the configured wake-word backend."""
    error = wake_detector_error(config)
    if error is not None:
        logger.error(error)
        return None

    assert config.remote_url is not None  # guaranteed by wake_detector_error
    return RemoteWakeClient(
        on_wake,
        url=config.remote_url,
        token=config.remote_token,
        gain=config.gain,
        phrase=config.phrase,
        on_sleep=on_sleep,
        sleep_phrases=config.sleep_phrases,
    )
