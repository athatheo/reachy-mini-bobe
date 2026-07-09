"""Environment-driven configuration for the Mac wake daemon."""

from __future__ import annotations
import os
from dataclasses import dataclass

from bobe.env_utils import parse_int, parse_bool, parse_float
from bobe.wake.phrases import WAKE_PHRASE


DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_WHISPER_MODEL = "base.en"
DEFAULT_WHISPER_DEVICE = "auto"
DEFAULT_WHISPER_COMPUTE_TYPE = "int8"
DEFAULT_END_SILENCE_MS = 200
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MAX_UTTERANCE_S = 3.0
DEFAULT_SPEECH_RMS = 450.0
DEFAULT_REFRACTORY_S = 2.5
DEFAULT_CLAUDE_CODE_WORKDIR = "~/repos/bobe-claude-code-workspace"
DEFAULT_CLAUDE_CODE_BIN = "claude"
DEFAULT_CLAUDE_CODE_LAUNCH_COOLDOWN_S = 30.0
DEFAULT_CLAUDE_CODE_COMMAND_TIMEOUT_S = 300.0
DEFAULT_CLAUDE_CODE_OUTPUT_LIMIT_CHARS = 6000
DEFAULT_CLAUDE_CODE_PERMISSION_MODE = "default"


def whisper_initial_prompt_from_phrase(phrase: str) -> str:
    """Build a mild Whisper initial prompt (wake name only — less bias than full phrase)."""
    words = phrase.strip().casefold().split()
    if not words:
        return ""
    return f"{words[-1].capitalize()}."


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
    whisper_initial_prompt: str | None = None
    whisper_hotwords: str | None = None
    end_silence_ms: int = DEFAULT_END_SILENCE_MS
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    max_utterance_s: float = DEFAULT_MAX_UTTERANCE_S
    speech_rms: float = DEFAULT_SPEECH_RMS
    refractory_s: float = DEFAULT_REFRACTORY_S
    claude_code_launch_enabled: bool = False
    claude_code_launch_token: str | None = None
    claude_code_workdir: str = DEFAULT_CLAUDE_CODE_WORKDIR
    claude_code_bin: str = DEFAULT_CLAUDE_CODE_BIN
    claude_code_launch_cooldown_s: float = DEFAULT_CLAUDE_CODE_LAUNCH_COOLDOWN_S
    claude_code_command_timeout_s: float = DEFAULT_CLAUDE_CODE_COMMAND_TIMEOUT_S
    claude_code_output_limit_chars: int = DEFAULT_CLAUDE_CODE_OUTPUT_LIMIT_CHARS
    claude_code_permission_mode: str = DEFAULT_CLAUDE_CODE_PERMISSION_MODE


def load_wake_daemon_config(env: dict[str, str] | None = None) -> WakeDaemonConfig:
    """Load daemon settings from environment variables."""
    source = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        return parse_int(source.get(name), default)

    def _float(name: str, default: float) -> float:
        return parse_float(source.get(name), default)

    def _bool(name: str, default: bool = False) -> bool:
        value = parse_bool(source.get(name))
        return default if value is None else value

    token = (source.get("BOBE_WAKE_TOKEN") or "").strip()
    if not token:
        raise ValueError("BOBE_WAKE_TOKEN must be set to a non-empty value")
    phrase = (source.get("BOBE_WAKE_PHRASE") or WAKE_PHRASE).strip().casefold() or WAKE_PHRASE

    def _optional(name: str) -> str | None:
        if name not in source:
            return None
        value = (source.get(name) or "").strip()
        return value or None

    if "WHISPER_INITIAL_PROMPT" in source:
        initial_prompt = _optional("WHISPER_INITIAL_PROMPT")
    else:
        initial_prompt = whisper_initial_prompt_from_phrase(phrase)
    hotwords = _optional("WHISPER_HOTWORDS")

    return WakeDaemonConfig(
        host=(source.get("WAKE_DAEMON_HOST") or DEFAULT_HOST).strip() or DEFAULT_HOST,
        port=_int("WAKE_DAEMON_PORT", DEFAULT_PORT),
        token=token,
        phrase=phrase,
        whisper_model=(source.get("WHISPER_MODEL") or DEFAULT_WHISPER_MODEL).strip() or DEFAULT_WHISPER_MODEL,
        whisper_device=(source.get("WHISPER_DEVICE") or DEFAULT_WHISPER_DEVICE).strip() or DEFAULT_WHISPER_DEVICE,
        whisper_compute_type=(source.get("WHISPER_COMPUTE_TYPE") or DEFAULT_WHISPER_COMPUTE_TYPE).strip()
        or DEFAULT_WHISPER_COMPUTE_TYPE,
        whisper_initial_prompt=initial_prompt,
        whisper_hotwords=hotwords,
        end_silence_ms=max(50, _int("VAD_END_SILENCE_MS", DEFAULT_END_SILENCE_MS)),
        min_speech_ms=max(100, _int("VAD_MIN_SPEECH_MS", DEFAULT_MIN_SPEECH_MS)),
        max_utterance_s=max(0.5, _float("VAD_MAX_UTTERANCE_S", DEFAULT_MAX_UTTERANCE_S)),
        speech_rms=max(50.0, _float("VAD_SPEECH_RMS", DEFAULT_SPEECH_RMS)),
        refractory_s=max(0.5, _float("WAKE_REFRACTORY_S", DEFAULT_REFRACTORY_S)),
        claude_code_launch_enabled=_bool("BOBE_CLAUDE_CODE_LAUNCH_ENABLED"),
        claude_code_launch_token=_optional("BOBE_CLAUDE_CODE_LAUNCH_TOKEN"),
        claude_code_workdir=source.get("BOBE_CLAUDE_CODE_WORKDIR", DEFAULT_CLAUDE_CODE_WORKDIR).strip()
        or DEFAULT_CLAUDE_CODE_WORKDIR,
        claude_code_bin=source.get("BOBE_CLAUDE_CODE_BIN", DEFAULT_CLAUDE_CODE_BIN).strip() or DEFAULT_CLAUDE_CODE_BIN,
        claude_code_launch_cooldown_s=max(
            0.0,
            _float("BOBE_CLAUDE_CODE_LAUNCH_COOLDOWN_S", DEFAULT_CLAUDE_CODE_LAUNCH_COOLDOWN_S),
        ),
        claude_code_command_timeout_s=max(
            1.0,
            _float("BOBE_CLAUDE_CODE_COMMAND_TIMEOUT_S", DEFAULT_CLAUDE_CODE_COMMAND_TIMEOUT_S),
        ),
        claude_code_output_limit_chars=max(
            500,
            _int("BOBE_CLAUDE_CODE_OUTPUT_LIMIT_CHARS", DEFAULT_CLAUDE_CODE_OUTPUT_LIMIT_CHARS),
        ),
        claude_code_permission_mode=(
            source.get("BOBE_CLAUDE_CODE_PERMISSION_MODE", DEFAULT_CLAUDE_CODE_PERMISSION_MODE).strip()
            or DEFAULT_CLAUDE_CODE_PERMISSION_MODE
        ),
    )
