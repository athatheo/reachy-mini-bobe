"""Heed wake-word streaming inference (ONNX + numpy/scipy, no torch at runtime)."""

# ruff: noqa: D102,D107

from __future__ import annotations
import base64
import json
import math
import time
import queue
import logging
import threading
from typing import Callable
from pathlib import Path
from functools import lru_cache
from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.signal
from numpy.typing import NDArray


logger = logging.getLogger(__name__)

WAKE_SAMPLE_RATE = 16000
STEP_SAMPLES = 1600  # 100 ms inference hop (matches Heed export guidance)
DEBUG_WINDOW_SECONDS = 10.0


@dataclass(frozen=True)
class HeedWakeMeta:
    """Parsed wake.json contract for a deployed Heed model."""

    phrase: str
    threshold: float
    sample_rate: int
    window_samples: int
    n_mels: int
    n_fft: int
    win_length: int
    hop_length: int
    window_frames: int
    mel_fmin_hz: float
    mel_fmax_hz: float
    consecutive_frames: int
    refractory_seconds: float
    rms_threshold_dbfs: float
    voice_band_min_fraction: float
    voice_band_lo_hz: float
    voice_band_hi_hz: float


def default_heed_model_dir() -> Path:
    """Return the bundled Hey Jarvis Heed export directory."""
    return Path(__file__).resolve().parent / "wake_models" / "hey_jarvis"


def resolve_wake_model_path(model_dir: Path) -> Path:
    """Return the ONNX model path (raw, bundled, or base64 export for HF Spaces)."""
    for name in ("wake.onnx", "wake_model", "wake_model.b64"):
        candidate = model_dir / name
        if candidate.is_file():
            return candidate
    return model_dir / "wake.onnx"


def load_heed_meta(model_dir: Path) -> HeedWakeMeta:
    """Load and validate wake.json from a Heed export directory."""
    meta_path = model_dir / "wake.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing Heed metadata: {meta_path}")
    raw = json.loads(meta_path.read_text())
    trigger = raw.get("trigger") or {}
    gate = raw.get("energy_gate") or {}
    return HeedWakeMeta(
        phrase=str(raw.get("phrase") or ""),
        threshold=float(raw["threshold"]),
        sample_rate=int(raw.get("sample_rate", WAKE_SAMPLE_RATE)),
        window_samples=int(raw.get("audio_window_samples", WAKE_SAMPLE_RATE)),
        n_mels=int(raw.get("n_mels", 40)),
        n_fft=int(raw.get("n_fft", 512)),
        win_length=int(raw.get("win_length", 400)),
        hop_length=int(raw.get("hop_length", 160)),
        window_frames=int(raw.get("window_frames", 100)),
        mel_fmin_hz=float(raw.get("mel_fmin_hz", 0.0)),
        mel_fmax_hz=float(raw.get("mel_fmax_hz", 8000.0)),
        consecutive_frames=int(trigger.get("consecutive_frames", 2)),
        refractory_seconds=float(trigger.get("refractory_seconds", 0.7)),
        rms_threshold_dbfs=float(gate.get("rms_threshold_dbfs", -55.0)),
        voice_band_min_fraction=float(gate.get("voice_band_min_fraction", 0.15)),
        voice_band_lo_hz=float(gate.get("voice_band_lo_hz", 100.0)),
        voice_band_hi_hz=float(gate.get("voice_band_hi_hz", 7000.0)),
    )


class _StreamingHighpass:
    """Stateful causal HPF + mains notches (ported from heed.audio)."""

    def __init__(self, *, cutoff_hz: float = 100.0, sample_rate: int = WAKE_SAMPLE_RATE, order: int = 8) -> None:
        self._sos = [_hpf_coefficients(cutoff_hz, sample_rate, order)]
        self._sos += [_notch_coefficients(freq, sample_rate) for freq in (50.0, 60.0)]
        self._zi_unit = [scipy.signal.sosfilt_zi(sos) for sos in self._sos]
        self._zi: list[np.ndarray | None] = [None] * len(self._sos)

    def reset(self) -> None:
        self._zi = [None] * len(self._sos)

    def __call__(self, chunk: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.asarray(chunk, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)
        for i, sos in enumerate(self._sos):
            if self._zi[i] is None:
                self._zi[i] = self._zi_unit[i] * x[0]
            x, self._zi[i] = scipy.signal.sosfilt(sos, x, zi=self._zi[i])
        return np.ascontiguousarray(x.astype(np.float32))


_HPF_CACHE: dict[tuple[float, int, int], np.ndarray] = {}
_NOTCH_CACHE: dict[tuple[float, int], np.ndarray] = {}


def _hpf_coefficients(cutoff_hz: float, sample_rate: int, order: int) -> np.ndarray:
    key = (float(cutoff_hz), int(sample_rate), int(order))
    if key not in _HPF_CACHE:
        _HPF_CACHE[key] = scipy.signal.butter(order, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return _HPF_CACHE[key]


def _notch_coefficients(freq_hz: float, sample_rate: int) -> np.ndarray:
    key = (float(freq_hz), int(sample_rate))
    if key not in _NOTCH_CACHE:
        b, a = scipy.signal.iirnotch(freq_hz, 30.0, fs=sample_rate)
        _NOTCH_CACHE[key] = scipy.signal.tf2sos(b, a)
    return _NOTCH_CACHE[key]


@lru_cache(maxsize=8)
def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int, fmin: float, fmax: float) -> NDArray[np.float32]:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    freqs = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(n_mels):
        left, center, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        rising = (freqs - left) / (center - left + 1e-9)
        falling = (right - freqs) / (right - center + 1e-9)
        fb[m] = np.clip(np.minimum(rising, falling), 0.0, None)
    return fb


def _peak_normalize(audio: NDArray[np.float32], target_dbfs: float = -3.0) -> NDArray[np.float32]:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio
    target = 10.0 ** (target_dbfs / 20.0)
    return (audio * (target / peak)).astype(np.float32)


def _log_mel(audio: NDArray[np.float32], meta: HeedWakeMeta) -> NDArray[np.float32]:
    window = np.hanning(meta.win_length).astype(np.float32)
    _, _, spec = scipy.signal.stft(
        audio,
        fs=meta.sample_rate,
        window=window,
        nperseg=meta.win_length,
        noverlap=meta.win_length - meta.hop_length,
        nfft=meta.n_fft,
        boundary="zeros",
        padded=True,
    )
    power = (np.abs(spec) ** 2).astype(np.float32)
    mel_fb = _mel_filterbank(meta.n_mels, meta.n_fft, meta.sample_rate, meta.mel_fmin_hz, meta.mel_fmax_hz)
    mel = mel_fb @ power
    out = np.log(np.clip(mel, 1e-9, None))
    out = out - out.mean(axis=1, keepdims=True)
    wanted_frames = meta.window_frames + 1
    if out.shape[1] < wanted_frames:
        out = np.pad(out, ((0, 0), (0, wanted_frames - out.shape[1])))
    elif out.shape[1] > wanted_frames:
        out = out[:, :wanted_frames]
    return out.astype(np.float32)


def _energy_gate(audio: NDArray[np.float32], meta: HeedWakeMeta) -> tuple[bool, float]:
    if audio.size < 64:
        return False, 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    rms_dbfs = 20.0 * math.log10(max(rms, 1e-9))
    if rms_dbfs < meta.rms_threshold_dbfs:
        return False, 0.0
    window = np.hanning(audio.size).astype(np.float32)
    spec = np.fft.rfft(audio * window)
    mag = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(audio.size, d=1.0 / meta.sample_rate)
    total = float(np.sum(mag)) + 1e-12
    voice_mask = (freqs >= meta.voice_band_lo_hz) & (freqs <= meta.voice_band_hi_hz)
    band_frac = float(np.sum(mag[voice_mask]) / total)
    return band_frac >= meta.voice_band_min_fraction, band_frac


class HeedWakeWordDetector:
    """Background Heed wake-word detector using exported wake.onnx + wake.json."""

    def __init__(
        self,
        on_wake: Callable[[], None],
        *,
        model_dir: Path | str | None = None,
        threshold: float | None = None,
        gain: float = 1.0,
    ) -> None:
        self._on_wake = on_wake
        self._model_dir = Path(model_dir) if model_dir is not None else default_heed_model_dir()
        self._meta = load_heed_meta(self._model_dir)
        self._threshold = float(self._meta.threshold if threshold is None else threshold)
        self._gain = gain
        self._queue: queue.Queue[NDArray[np.int16]] = queue.Queue(maxsize=64)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._recent_stats: deque[tuple[float, float, float]] = deque()

    @property
    def phrase(self) -> str:
        return self._meta.phrase

    def is_running(self) -> bool:
        """Return whether the background detection thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="heed-wake-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def feed(self, frame: NDArray[np.int16]) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass

    def debug_state(self) -> dict[str, float | int | str]:
        now = time.monotonic()
        with self._stats_lock:
            while self._recent_stats and now - self._recent_stats[0][0] > DEBUG_WINDOW_SECONDS:
                self._recent_stats.popleft()
            entries = list(self._recent_stats)
        scores = [score for _, score, _ in entries]
        levels = [rms for _, _, rms in entries]
        return {
            "backend": "heed",
            "phrase": self._meta.phrase,
            "threshold": self._threshold,
            "gain": self._gain,
            "frames_window": len(entries),
            "score_last": round(scores[-1], 4) if scores else 0.0,
            "score_peak": round(max(scores), 4) if scores else 0.0,
            "rms_peak": round(max(levels), 1) if levels else 0.0,
            "rms_last": round(levels[-1], 1) if levels else 0.0,
            "thread_alive": self.is_running(),
        }

    def _record_stats(self, score: float, rms: float) -> None:
        now = time.monotonic()
        with self._stats_lock:
            self._recent_stats.append((now, score, rms))
            while self._recent_stats and now - self._recent_stats[0][0] > DEBUG_WINDOW_SECONDS:
                self._recent_stats.popleft()

    def _load_session(self) -> object | None:
        try:
            import onnxruntime as ort
        except Exception:
            logger.exception("onnxruntime is not available; Heed wake-word detection disabled")
            return None
        model_path = resolve_wake_model_path(self._model_dir)
        if not model_path.is_file():
            logger.error("Missing Heed model at %s", model_path)
            return None
        try:
            if model_path.suffix == ".b64":
                model_bytes = base64.b64decode(model_path.read_bytes())
                session: object = ort.InferenceSession(
                    model_bytes, providers=["CPUExecutionProvider"]
                )
            else:
                session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
            return session
        except Exception:
            logger.exception("Failed to load Heed ONNX model from %s", model_path)
            return None

    def _run(self) -> None:
        session = self._load_session()
        if session is None:
            return
        logger.info(
            "Heed wake-word detector listening for %r (threshold=%.3f, gain=%.1fx)",
            self._meta.phrase,
            self._threshold,
            self._gain,
        )

        meta = self._meta
        buffer = np.zeros(meta.window_samples, dtype=np.float32)
        hpf = _StreamingHighpass(sample_rate=meta.sample_rate)
        pending = np.zeros(0, dtype=np.int16)
        above = 0
        last_trigger = -1e9

        while not self._stop_event.is_set():
            try:
                frame = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            pending = np.concatenate([pending, frame.reshape(-1)])
            while pending.size >= STEP_SAMPLES:
                chunk_i16 = pending[:STEP_SAMPLES]
                pending = pending[STEP_SAMPLES:]
                rms = float(np.sqrt(np.mean(chunk_i16.astype(np.float64) ** 2)))
                chunk = chunk_i16.astype(np.float32) / 32768.0
                if self._gain != 1.0:
                    chunk = np.clip(chunk * self._gain, -1.0, 1.0)
                filtered = hpf(chunk)
                n = filtered.size
                if n >= meta.window_samples:
                    buffer = filtered[-meta.window_samples :]
                else:
                    buffer = np.concatenate([buffer[n:], filtered])

                passed, _band_frac = _energy_gate(buffer, meta)
                if not passed:
                    self._record_stats(0.0, rms)
                    above = 0
                    continue

                mel = _log_mel(_peak_normalize(buffer), meta)
                mel_input = mel[np.newaxis, :, :]
                try:
                    logit = session.run(None, {"mel": mel_input})[0]  # type: ignore[attr-defined]
                except Exception:
                    logger.exception("Heed wake-word inference failed; stopping detector")
                    return
                prob = float(1.0 / (1.0 + np.exp(-float(np.asarray(logit).reshape(-1)[0]))))
                self._record_stats(prob, rms)

                if prob > self._threshold:
                    above += 1
                else:
                    above = 0
                now = time.monotonic()
                if above >= meta.consecutive_frames and now - last_trigger > meta.refractory_seconds:
                    logger.info("Heed wake word detected (score=%.3f)", prob)
                    last_trigger = now
                    above = 0
                    pending = np.zeros(0, dtype=np.int16)
                    hpf.reset()
                    buffer = np.zeros(meta.window_samples, dtype=np.float32)
                    self._drain_queue()
                    self._on_wake()
                    break

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
