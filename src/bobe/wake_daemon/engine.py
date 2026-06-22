"""faster-whisper wake phrase detection for streamed PCM audio."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from bobe.wake.phrases import matches_wake_phrase
from bobe.wake_daemon.config import WakeDaemonConfig


logger = logging.getLogger(__name__)

WAKE_SAMPLE_RATE = 16000
PARTIAL_TRANSCRIBE_INTERVAL_S = 0.7
TRANSCRIPT_HISTORY_MAX = 30


@dataclass
class WhisperWakeEngine:
    """Segment speech with simple RMS VAD and confirm wake phrases via Whisper."""

    config: WakeDaemonConfig
    _model: object | None = field(default=None, init=False, repr=False)
    _paused: bool = field(default=False, init=False)
    _in_speech: bool = field(default=False, init=False)
    _speech_samples: list[np.ndarray] = field(default_factory=list, init=False)
    _silence_samples: int = field(default=0, init=False)
    _last_wake_at: float = field(default=-1e9, init=False)
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
        return self.config.phrase

    def pause(self) -> None:
        self._paused = True
        self.reset()

    def resume(self) -> None:
        self._paused = False
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._speech_samples.clear()
        self._silence_samples = 0

    def debug_state(self) -> dict[str, float | int | str | bool]:
        return {
            "engine": "faster-whisper",
            "model": self.config.whisper_model,
            "phrase": self.config.phrase,
            "paused": self._paused,
            "in_speech": self._in_speech,
            "transcript_last": self._last_transcript,
            "transcript_partial": self._partial_transcript,
            "transcript_stream": list(self._transcript_history)[-10:],
            "latency_ms_last": round(self._last_latency_ms, 1),
            "rms_last": round(self._last_rms, 1),
        }

    def feed(self, pcm_i16: np.ndarray) -> dict[str, object] | None:
        """Consume PCM samples and return a wake event dict when detected."""
        if self._paused:
            return None

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
        if self._in_speech and utterance_samples >= min_speech_samples:
            now = time.monotonic()
            if now - self._last_partial_at >= PARTIAL_TRANSCRIBE_INTERVAL_S:
                partial = self._transcribe(np.concatenate(self._speech_samples))
                self._partial_transcript = partial
                self._last_partial_at = now

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
        transcript = self._transcribe(utterance)
        latency_ms = (time.monotonic() - started) * 1000.0
        self._last_transcript = transcript
        self._partial_transcript = ""
        self._last_latency_ms = latency_ms
        if transcript:
            self._transcript_history.append(
                {
                    "text": transcript,
                    "partial": False,
                    "ts": round(time.time(), 3),
                }
            )

        if not transcript:
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

    def _transcribe(self, pcm_i16: np.ndarray) -> str:
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
        )
        parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(parts).strip()
