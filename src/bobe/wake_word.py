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
from typing import Any, Literal, Mapping, Callable
from collections import deque
from dataclasses import dataclass

import numpy as np
from fastrtc import audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from bobe.env_utils import parse_float
from bobe.wake.phrases import WAKE_PHRASE, DEFAULT_SLEEP_PHRASES
from bobe.wake.constants import WAKE_SAMPLE_RATE
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

    def restore(self, samples: NDArray[np.int16]) -> None:
        """Put drained audio back at the FRONT of the buffer (oldest position).

        Used when flushing drained audio upstream fails: frames captured while
        the flush was in flight have already been appended, so the restored
        tail must precede them to keep the audio time-ordered.
        """
        if samples.size == 0:
            return
        with self._lock:
            self._chunks.appendleft(samples)
            self._total_samples += samples.size
            while self._total_samples > self._max_samples and len(self._chunks) > 1:
                dropped = self._chunks.popleft()
                self._total_samples -= dropped.size


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

    @property
    def sleep_requested(self) -> bool:
        """Return whether a sleep request is pending (thread-safe)."""
        with self._lock:
            return self._sleep_requested

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
            # A sleep detected while we were still asleep must not survive
            # into the fresh awake state and put us straight back to sleep
            # (mirrors sleep() clearing _wake_requested).
            self._sleep_requested = False

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
        if not config.remote_token:
            # The daemon refuses to start without a token and closes token-less
            # handshakes with 1008, so a missing token can never connect —
            # surface it here instead of reconnect-looping forever.
            return "BOBE_WAKE_TOKEN is required when BOBE_WAKE_BACKEND=remote"
        return None
    return f"Unknown wake backend {backend!r}; wake-word detection disabled"


def create_wake_detector(
    on_wake: Callable[[], None],
    config: WakeConfig,
    *,
    on_sleep: Callable[[], None] | None = None,
    on_announce: Callable[[str], None] | None = None,
    on_speech: Callable[[NDArray[np.int16], int], None] | None = None,
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
        on_announce=on_announce,
        on_speak=on_speech,
        sleep_phrases=config.sleep_phrases,
    )


def _to_wake_rate(audio_frame: NDArray[Any], input_sample_rate: int) -> NDArray[np.int16]:
    """Convert a mono frame to the 16 kHz int16 format the wake backends expect."""
    mono = audio_frame.reshape(-1)
    if input_sample_rate != WAKE_SAMPLE_RATE:
        mono = resample(mono, int(len(mono) * WAKE_SAMPLE_RATE / input_sample_rate))
    if np.issubdtype(mono.dtype, np.integer):
        return mono.astype(np.int16, copy=False)
    converted: NDArray[np.int16] = audio_to_int16(np.asarray(mono, dtype=np.float32))
    return converted


class WakeGate:
    """Own the local wake-gating building blocks: config, session, buffer, detector.

    Pure wake-word plumbing with no OpenAI knowledge: the realtime handler
    drives this gate from its mic loop and keeps the transition orchestration
    (connection management, chimes, antenna cues) to itself. ``sleep()`` and
    ``listen_for_wake()`` are deliberately separate methods: the handler flips
    the session asleep BEFORE its response-cancel/buffer-clear awaits and
    switches the detector back to wake-listening AFTER them, so that audio is
    gated throughout the transition.
    """

    def __init__(
        self,
        input_sample_rate: int,
        config: WakeConfig | None = None,
        detector_factory: Callable[..., RemoteWakeClient | None] = create_wake_detector,
    ) -> None:
        """Assemble the gating subsystem from environment-driven configuration."""
        self.config = load_wake_config() if config is None else config
        self.session = WakeSession(timeout_s=self.config.timeout_s)
        self.buffer = AudioRingBuffer(sample_rate=input_sample_rate)
        # Announcements and speech clips arrive on the detector thread; the
        # handler's mic loop drains them on the event loop via
        # drain_announcements() / drain_speech().
        self._announce_lock = threading.Lock()
        self._announcements: list[str] = []
        self._speech_clips: list[tuple[NDArray[np.int16], int]] = []
        detector_kwargs: dict[str, Any] = {
            "on_wake": self.session.request_wake,
            "config": self.config,
            "on_sleep": self.session.request_sleep,
            "on_announce": self.request_announce,
        }
        try:
            self.detector = detector_factory(on_speech=self.request_speech, **detector_kwargs)
        except TypeError:
            # Older factories/test stubs predate the speech downlink.
            self.detector = detector_factory(**detector_kwargs)
        self.enabled = self.detector is not None
        self.error: str | None
        if self.enabled:
            self.error = None
        else:
            self.error = wake_detector_error(self.config) or "Wake-word detector unavailable"
            logger.error("Wake-word gating disabled: %s", self.error)
            # Fallback: without a detector, stay in an always-on session so mic + replies work.
            self.session.wake()
            logger.warning("Wake-word detection unavailable; running in always-on mode until wake is configured")

    @property
    def awake(self) -> bool:
        """Return whether audio is currently allowed to stream upstream."""
        return self.session.awake

    def poll(self) -> Literal["wake", "sleep", "expired"] | None:
        """Consume at most one pending gating event, in mic-loop priority order."""
        if self.session.consume_wake_request():
            return "wake"
        elif self.session.consume_sleep_request():
            return "sleep"
        elif self.session.expired():
            return "expired"
        return None

    def request_announce(self, text: str) -> None:
        """Queue an announcement (thread-safe; called from the detector thread)."""
        if not text:
            return
        with self._announce_lock:
            self._announcements.append(text)

    def drain_announcements(self) -> list[str]:
        """Return and clear queued announcements (consumed by the mic loop)."""
        with self._announce_lock:
            pending = self._announcements
            self._announcements = []
        return pending

    def request_speech(self, pcm: NDArray[np.int16], rate: int) -> None:
        """Queue a speech clip for playback (thread-safe; detector thread)."""
        if pcm.size == 0 or rate <= 0:
            return
        with self._announce_lock:
            self._speech_clips.append((pcm, rate))

    def drain_speech(self) -> list[tuple[NDArray[np.int16], int]]:
        """Return and clear queued speech clips (consumed by the mic loop)."""
        with self._announce_lock:
            pending = self._speech_clips
            self._speech_clips = []
        return pending

    def feed(self, audio_frame: NDArray[Any], input_sample_rate: int) -> None:
        """Forward mic audio to the local wake detector, restarting it if needed."""
        if self.detector is None:
            return
        if not self.detector.is_running():
            logger.warning("Wake detector thread not running; restarting")
            self.detector.start()
        self.detector.feed(_to_wake_rate(audio_frame, input_sample_rate))

    def buffer_frame(self, frame: NDArray[np.int16]) -> None:
        """Keep a mono mic frame local in the pre-wake ring buffer."""
        self.buffer.append(frame)

    def drain_tail(self, seconds: float) -> NDArray[np.int16]:
        """Return up to the last ``seconds`` of buffered audio and clear the buffer."""
        return self.buffer.drain_tail(seconds)

    def restore(self, samples: NDArray[np.int16]) -> None:
        """Put drained audio back at the front of the buffer after a failed flush."""
        self.buffer.restore(samples)

    def enter_awake(self, *, converse: bool = False) -> None:
        """Open the gate: detector listens for the sleep phrase, session wakes.

        With ``converse=True`` (Hermes voice backend) the daemon additionally
        captures full awake utterances for the agent.
        """
        if self.detector is not None:
            if converse and hasattr(self.detector, "listen_for_converse"):
                self.detector.listen_for_converse()
            else:
                self.detector.listen_for_sleep()
        self.session.wake()

    def sleep(self) -> None:
        """Flip the session asleep (call ``listen_for_wake`` separately, after any awaits)."""
        self.session.sleep()

    def listen_for_wake(self) -> None:
        """Switch the detector back to wake-phrase listening, restarting it if needed."""
        detector = self.detector
        if detector is None:
            return
        detector.listen_for_wake()
        if not detector.is_running():
            logger.warning("Wake detector thread not running; restarting")
            detector.start()

    def requeue_wake(self) -> None:
        """Re-queue a wake request (used when a wake transition fails and backs off)."""
        self.session.request_wake()

    def touch(self) -> None:
        """Record session activity, resetting the inactivity timer."""
        self.session.touch()

    def start(self) -> None:
        """Start the detector thread, if a detector is configured."""
        if self.detector is not None:
            self.detector.start()

    def stop(self) -> None:
        """Stop the detector thread, if a detector is configured."""
        if self.detector is not None:
            self.detector.stop()

    def debug_state(self) -> Any:
        """Return the detector's debug state (None when no detector exists)."""
        return self.detector.debug_state() if self.detector is not None else None
