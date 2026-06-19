"""Environment-driven configuration for the Mac wake daemon."""

from __future__ import annotations

import os
from dataclasses import dataclass

from bobe.wake.phrases import WAKE_PHRASE


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_WHISPER_MODEL = "tiny.en"
DEFAULT_WHISPER_DEVICE = "auto"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_END_SILENCE_MS = 200
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MAX_UTTERANCE_S = 3.0
DEFAULT_SPEECH_RMS = 450.0
DEFAULT_REFRACTORY_S = 2.5


@dataclass(frozen=True)
class WakeDaemonConfig:
    """Runtime settings for bobe-wake-daemon."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str | None = None
    phrase: str = WAKE_PHRASE
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_device: str = DEFAULT_WHISPER_DEVICE
    whisper_compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE
    end_silence_ms: int = DEFAULT_END_SILENCE_MS
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    max_utterance_s: float = DEFAULT_MAX_UTTERANCE_S
    speech_rms: float = DEFAULT_SPEECH_RMS
    refractory_s: float = DEFAULT_REFRACTORY_S


def load_wake_daemon_config(env: dict[str, str] | None = None) -> WakeDaemonConfig:
    """Load daemon settings from environment variables."""
    source = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        try:
            return int(source.get(name, default))
        except (TypeError, ValueError):
            return default

    def _float(name: str, default: float) -> float:
        try:
            return float(source.get(name, default))
        except (TypeError, ValueError):
            return default

    token = (source.get("BOBE_WAKE_TOKEN") or "").strip() or None
    phrase = (source.get("BOBE_WAKE_PHRASE") or WAKE_PHRASE).strip().casefold() or WAKE_PHRASE

    return WakeDaemonConfig(
        host=(source.get("WAKE_DAEMON_HOST") or DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=_int("WAKE_DAEMON_PORT", DEFAULT_PORT),
        token=token,
        phrase=phrase,
        whisper_model=(source.get("WHISPER_MODEL") or DEFAULT_WHISPER_MODEL).strip() or DEFAULT_WHISPER_MODEL,
        whisper_device=(source.get("WHISPER_DEVICE") or DEFAULT_WHISPER_DEVICE).strip() or DEFAULT_WHISPER_DEVICE,
        whisper_compute_type=(
            source.get("WHISPER_COMPUTE_TYPE") or DEFAULT_WHISPER_COMPUTE_TYPE
        ).strip()
        or DEFAULT_WHISPER_COMPUTE_TYPE,
        end_silence_ms=max(50, _int("VAD_END_SILENCE_MS", DEFAULT_END_SILENCE_MS)),
        min_speech_ms=max(100, _int("VAD_MIN_SPEECH_MS", DEFAULT_MIN_SPEECH_MS)),
        max_utterance_s=max(0.5, _float("VAD_MAX_UTTERANCE_S", DEFAULT_MAX_UTTERANCE_S)),
        speech_rms=max(50.0, _float("VAD_SPEECH_RMS", DEFAULT_SPEECH_RMS)),
        refractory_s=max(0.5, _float("WAKE_REFRACTORY_S", DEFAULT_REFRACTORY_S)),
    )
