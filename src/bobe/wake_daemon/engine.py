"""faster-whisper wake phrase detection for streamed PCM audio."""

from __future__ import annotations
import time
import logging
from typing import Any, Literal
from collections import deque
from dataclasses import field, dataclass

import numpy as np

from bobe.wake.phrases import DEFAULT_SLEEP_PHRASES, matches_wake_phrase, matches_sleep_phrase
from bobe.wake.constants import WAKE_SAMPLE_RATE
from bobe.wake_daemon.config import WakeDaemonConfig


logger = logging.getLogger(__name__)

PARTIAL_TRANSCRIBE_INTERVAL_S = 0.45
STATS_PARTIAL_MIN_SAMPLES = int(0.35 * WAKE_SAMPLE_RATE)
TRANSCRIPT_HISTORY_MAX = 30
ListenMode = Literal["wake", "sleep"]


def whisper_engine_key(config: WakeDaemonConfig) -> tuple[str, str, str, str | None, str | None]:
    """Hashable key for sharing one loaded Whisper model across connections."""
    return (
        config.whisper_model,
        config.whisper_device,
        config.whisper_compute_type,
        config.whisper_initial_prompt,
        config.whisper_hotwords,
    )


@dataclass
class WhisperWakeEngine:
    """Shared faster-whisper model holder; use sessions for per-connection state."""

    config: WakeDaemonConfig
    _model: object | None = field(default=None, init=False, repr=False)

    def session(self, config: WakeDaemonConfig | None = None) -> WhisperWakeSession:
        """Create an isolated VAD/transcript session backed by this shared model."""
        return WhisperWakeSession(engine=self, config=config or self.config)

    def transcribe(self, pcm_i16: np.ndarray, *, config: WakeDaemonConfig | None = None) -> str:
        """Run Whisper on PCM audio using prompt settings from config."""
        runtime = config or self.config
        model = self._load_model()
        audio = pcm_i16.astype(np.float32) / 32768.0
        segments, _info = model.transcribe(  # type: ignore[attr-defined]
            audio,
            language="en",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=runtime.whisper_initial_prompt or None,
            hotwords=runtime.whisper_hotwords or None,
        )
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("faster-whisper is not installed; install with: uv sync --extra wake-daemon") from exc

        logger.info(
            "Loading Whisper model %r (device=%r, compute_type=%r)",
            self.config.whisper_model,
            self.config.whisper_device,
            self.config.whisper_compute_type,
        )
        self._model = WhisperModel(
            self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type,
        )
        return self._model


@dataclass
class WhisperWakeSession:
    """Per-connection speech buffer and phrase detection state."""

    engine: WhisperWakeEngine
    config: WakeDaemonConfig
    _listen_mode: ListenMode = field(default="wake", init=False)
    _sleep_phrases: tuple[str, ...] = field(default=DEFAULT_SLEEP_PHRASES, init=False)
    _in_speech: bool = field(default=False, init=False)
    _speech_samples: list[np.ndarray] = field(default_factory=list, init=False)
    _silence_samples: int = field(default=0, init=False)
    _last_wake_at: float = field(default=-1e9, init=False)
    _last_sleep_at: float = field(default=-1e9, init=False)
    _last_transcript: str = field(default="", init=False)
    _partial_transcript: str = field(default="", init=False)
    _last_partial_at: float = field(default=0.0, init=False)
    _last_latency_ms: float = field(default=0.0, init=False)
    _last_rms: float = field(default=0.0, init=False)
    _transcript_history: deque[dict[str, str | float | bool]] = field(
        default_factory=lambda: deque(maxlen=TRANSCRIPT_HISTORY_MAX),
        init=False,
    )

    @property
    def phrase(self) -> str:
        """Wake phrase this session listens for."""
        return self.config.phrase

    def set_listen_mode(
        self,
        mode: ListenMode,
        *,
        sleep_phrases: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        """Switch between wake-phrase and sleep-phrase detection."""
        self._listen_mode = mode
        if sleep_phrases:
            self._sleep_phrases = tuple(sleep_phrases)
        self.reset()

    def reset(self) -> None:
        """Clear the in-progress utterance buffer and partial transcript."""
        self._in_speech = False
        self._speech_samples.clear()
        self._silence_samples = 0
        self._partial_transcript = ""

    def _set_partial(self, text: str) -> None:
        """Track live Whisper text in the rolling stream shown by the UI."""
        cleaned = text.strip()
        self._partial_transcript = cleaned
        if not cleaned:
            if self._transcript_history and self._transcript_history[-1].get("partial"):
                self._transcript_history.pop()
            return
        entry: dict[str, str | float | bool] = {"text": cleaned, "partial": True, "ts": round(time.time(), 3)}
        if self._transcript_history and self._transcript_history[-1].get("partial"):
            self._transcript_history[-1] = entry
        else:
            self._transcript_history.append(entry)

    def _append_final(self, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        if self._transcript_history and self._transcript_history[-1].get("partial"):
            self._transcript_history.pop()
        self._transcript_history.append(
            {
                "text": cleaned,
                "partial": False,
                "ts": round(time.time(), 3),
            }
        )

    def debug_state(self) -> dict[str, Any]:
        """Snapshot of session state for stats messages and diagnostics."""
        return {
            "engine": "faster-whisper",
            "model": self.config.whisper_model,
            "phrase": self.config.phrase,
            "listen_mode": self._listen_mode,
            "paused": self._listen_mode == "sleep",
            "in_speech": self._in_speech,
            "transcript_last": self._last_transcript,
            "transcript_partial": self._partial_transcript,
            "transcript_stream": list(self._transcript_history)[-10:],
            "latency_ms_last": round(self._last_latency_ms, 1),
            "rms_last": round(self._last_rms, 1),
        }

    def _maybe_emit_sleep(self, transcript: str, *, latency_ms: float) -> dict[str, Any] | None:
        if self._listen_mode != "sleep":
            return None
        if not matches_sleep_phrase(transcript, self._sleep_phrases):
            return None
        if time.monotonic() - self._last_sleep_at < self.config.refractory_s:
            return None
        self._last_sleep_at = time.monotonic()
        logger.info("Sleep phrase detected (transcript=%r, latency_ms=%.1f)", transcript, latency_ms)
        return {
            "type": "sleep",
            "transcript": transcript,
            "latency_ms": latency_ms,
        }

    def _maybe_emit_wake(self, transcript: str, *, latency_ms: float) -> dict[str, Any] | None:
        if self._listen_mode != "wake":
            return None
        if time.monotonic() - self._last_wake_at < self.config.refractory_s:
            return None
        if not matches_wake_phrase(transcript, phrase=self.config.phrase):
            return None
        self._last_wake_at = time.monotonic()
        logger.info("Wake phrase detected (transcript=%r, latency_ms=%.1f)", transcript, latency_ms)
        return {
            "type": "wake",
            "phrase": self.config.phrase,
            "transcript": transcript,
            "latency_ms": latency_ms,
        }

    def feed(self, pcm_i16: np.ndarray) -> dict[str, Any] | None:
        """Consume PCM samples and return a wake/sleep event when detected."""
        chunk = pcm_i16.reshape(-1).astype(np.int16, copy=False)
        if chunk.size == 0:
            return None

        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        self._last_rms = rms
        voiced = rms >= self.config.speech_rms

        if voiced:
            self._in_speech = True
            self._silence_samples = 0
            self._speech_samples.append(chunk)
        elif self._in_speech:
            self._speech_samples.append(chunk)
            self._silence_samples += chunk.size

        utterance_samples = sum(part.size for part in self._speech_samples)
        min_speech_samples = int(self.config.min_speech_ms * WAKE_SAMPLE_RATE / 1000)
        if self._in_speech and utterance_samples >= STATS_PARTIAL_MIN_SAMPLES:
            now = time.monotonic()
            if now - self._last_partial_at >= PARTIAL_TRANSCRIBE_INTERVAL_S:
                started = time.monotonic()
                partial = self.engine.transcribe(np.concatenate(self._speech_samples), config=self.config)
                latency_ms = (time.monotonic() - started) * 1000.0
                self._set_partial(partial)
                self._last_partial_at = now
                if partial and self._listen_mode == "sleep":
                    event = self._maybe_emit_sleep(partial, latency_ms=latency_ms)
                    if event is not None:
                        self.reset()
                        return event

        if not self._in_speech:
            return None

        utterance_samples = sum(part.size for part in self._speech_samples)
        max_samples = int(self.config.max_utterance_s * WAKE_SAMPLE_RATE)
        end_silence_samples = int(self.config.end_silence_ms * WAKE_SAMPLE_RATE / 1000)
        should_finalize = utterance_samples >= max_samples or self._silence_samples >= end_silence_samples
        if not should_finalize:
            return None

        utterance = np.concatenate(self._speech_samples) if self._speech_samples else np.zeros(0, dtype=np.int16)
        self.reset()
        if utterance.size < min_speech_samples:
            return None

        started = time.monotonic()
        transcript = self.engine.transcribe(utterance, config=self.config)
        latency_ms = (time.monotonic() - started) * 1000.0
        self._last_transcript = transcript
        self._partial_transcript = ""
        self._last_latency_ms = latency_ms
        self._append_final(transcript)

        if not transcript:
            return None
        if self._listen_mode == "sleep":
            return self._maybe_emit_sleep(transcript, latency_ms=latency_ms)
        return self._maybe_emit_wake(transcript, latency_ms=latency_ms)
