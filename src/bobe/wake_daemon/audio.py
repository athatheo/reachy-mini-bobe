"""Audio decoding for the wake daemon's robot-speech downlink.

Hermes TTS providers hand the daemon compressed audio (MP3 from most
providers, OGG/Opus from voice-bubble platforms, occasionally WAV). The robot
speaker path wants mono s16le PCM at its output rate, so everything funnels
through one decoder. ffmpeg does the real work; a stdlib WAV fallback keeps
the daemon usable (and the tests hermetic) when ffmpeg is absent.
"""

from __future__ import annotations
import io
import wave
import shutil
import logging
import subprocess

import numpy as np
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

ROBOT_SPEECH_RATE = 24000
# One decoded minute at 24 kHz mono int16 is ~2.9 MB; cap inbound compressed
# payloads well below anything a reasonable TTS reply produces.
MAX_AUDIO_BYTES = 20 * 1024 * 1024
_FFMPEG_TIMEOUT_S = 30.0


class AudioDecodeError(RuntimeError):
    """Raised when an audio payload cannot be decoded to PCM."""


# LaunchAgents run with a minimal PATH that omits Homebrew's bin directories,
# so PATH lookup alone would miss the ffmpeg that is actually installed.
_FFMPEG_FALLBACK_PATHS = ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg")


def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    for candidate in _FFMPEG_FALLBACK_PATHS:
        if shutil.which(candidate):
            return candidate
    return None


def _decode_with_ffmpeg(ffmpeg: str, data: bytes, rate: int) -> NDArray[np.int16]:
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(rate),
        "pipe:1",
    ]
    try:
        result = subprocess.run(
            cmd,
            input=data,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError("ffmpeg timed out decoding audio") from exc
    if result.returncode != 0 or not result.stdout:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise AudioDecodeError(f"ffmpeg could not decode audio: {stderr[-300:] or 'no output'}")
    return np.frombuffer(result.stdout, dtype=np.int16)


def _decode_wav_stdlib(data: bytes, rate: int) -> NDArray[np.int16]:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            src_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except Exception as exc:
        raise AudioDecodeError(f"not a decodable WAV file: {exc}") from exc
    if width != 2:
        raise AudioDecodeError(f"unsupported WAV sample width: {width}")
    pcm = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0].copy()
    if src_rate != rate and pcm.size:
        from scipy.signal import resample

        pcm = np.asarray(
            np.clip(resample(pcm, int(pcm.size * rate / src_rate)), -32768, 32767),
            dtype=np.int16,
        )
    return pcm


def decode_audio_to_pcm(data: bytes, rate: int = ROBOT_SPEECH_RATE) -> NDArray[np.int16]:
    """Decode any common audio container to mono s16le PCM at ``rate``.

    Raises:
        AudioDecodeError: If the payload is empty, oversized, or undecodable.

    """
    if not data:
        raise AudioDecodeError("empty audio payload")
    if len(data) > MAX_AUDIO_BYTES:
        raise AudioDecodeError(f"audio payload too large ({len(data)} bytes)")
    ffmpeg = _find_ffmpeg()
    if ffmpeg is not None:
        return _decode_with_ffmpeg(ffmpeg, data, rate)
    logger.warning("ffmpeg not found; falling back to WAV-only decoding")
    return _decode_wav_stdlib(data, rate)
