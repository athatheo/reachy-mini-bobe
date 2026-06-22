"""Remote wake-word client that streams PCM to a Mac-side daemon."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bobe.wake.phrases import WAKE_PHRASE
from bobe.wake.protocol import hello_message, pause_message, resume_message


logger = logging.getLogger(__name__)

WAKE_SAMPLE_RATE = 16000
DEBUG_WINDOW_SECONDS = 10.0

RECONNECT_BASE_S = 0.5
RECONNECT_MAX_S = 10.0


class RemoteWakeClient:
    """Stream mic PCM to a remote wake daemon and receive wake events."""

    def __init__(
        self,
        on_wake: Any,
        *,
        url: str,
        token: str | None = None,
        gain: float = 1.0,
        sample_rate: int = WAKE_SAMPLE_RATE,
    ) -> None:
        self._on_wake = on_wake
        self._url = url
        self._token = (token or "").strip() or None
        self._gain = gain
        self._sample_rate = sample_rate
        self._audio_queue: queue.Queue[NDArray[np.int16] | None] = queue.Queue(maxsize=128)
        self._control_queue: queue.Queue[str] = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._paused = False
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._recent_stats: deque[tuple[float, float, str]] = deque()
        self._event_log: deque[dict[str, float | int | str | bool]] = deque(maxlen=40)
        self._remote_stats: dict[str, float | int | str | bool] = {}
        self._daemon_engine = ""
        self._connected = False
        self._last_transcript = ""
        self._last_partial_logged = ""
        self._transcript_stream: list[dict[str, float | int | str | bool]] = []

    @property
    def phrase(self) -> str:
        return WAKE_PHRASE

    def is_running(self) -> bool:
        """Return whether the background client thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="remote-wake-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def feed(self, frame: NDArray[np.int16]) -> None:
        if self._paused:
            return
        try:
            self._audio_queue.put_nowait(frame.reshape(-1).astype(np.int16, copy=False))
        except queue.Full:
            pass

    def pause(self) -> None:
        """Pause upstream streaming while BoBe is awake."""
        self._paused = True
        self._control_queue.put("pause")

    def resume(self) -> None:
        """Resume upstream streaming after BoBe goes back to sleep."""
        self._paused = False
        self._control_queue.put("resume")

    def debug_state(self) -> dict[str, float | int | str | bool | list[dict[str, float | int | str | bool]] | dict[str, float | int | str | bool]]:
        now = time.monotonic()
        with self._stats_lock:
            while self._recent_stats and now - self._recent_stats[0][0] > DEBUG_WINDOW_SECONDS:
                self._recent_stats.popleft()
            entries = list(self._recent_stats)
            events = list(self._event_log)
            remote_stats = dict(self._remote_stats)
            daemon_engine = self._daemon_engine
        rms_values = [rms for _, rms, _ in entries]
        return {
            "backend": "remote",
            "phrase": WAKE_PHRASE,
            "url": self._url,
            "gain": self._gain,
            "frames_window": len(entries),
            "rms_peak": round(max(rms_values), 1) if rms_values else 0.0,
            "rms_last": round(rms_values[-1], 1) if rms_values else 0.0,
            "transcript_last": self._last_transcript,
            "transcript_partial": self._remote_stats.get("partial", ""),
            "transcript_stream": list(self._transcript_stream)[-12:],
            "connected": self._connected,
            "paused": self._paused,
            "thread_alive": self.is_running(),
            "daemon_engine": daemon_engine,
            "remote_stats": remote_stats,
            "events": events[-20:],
        }

    def _log_event(self, level: str, message: str, **fields: float | int | str | bool) -> None:
        entry: dict[str, float | int | str | bool] = {
            "ts": round(time.time(), 3),
            "level": level,
            "message": message,
        }
        entry.update(fields)
        with self._stats_lock:
            self._event_log.append(entry)

    def _apply_remote_stats(self, payload: dict[str, Any]) -> None:
        stats: dict[str, float | int | str | bool] = {}
        for key in (
            "transcript",
            "partial",
            "rms",
            "in_speech",
            "paused",
            "latency_ms",
            "latency_ms_last",
            "engine",
            "model",
        ):
            if key in payload and payload[key] is not None:
                stats[key] = payload[key]  # type: ignore[assignment]
        transcript = str(payload.get("transcript") or "")
        partial = str(payload.get("partial") or "")
        stream = payload.get("transcript_stream")
        with self._stats_lock:
            self._remote_stats.update(stats)
            if isinstance(stream, list):
                self._transcript_stream = [entry for entry in stream if isinstance(entry, dict)][-12:]
            if partial:
                self._last_transcript = partial
                if partial != self._last_partial_logged:
                    self._last_partial_logged = partial
                    self._log_event("transcript", partial, partial=True)
            elif transcript:
                self._last_transcript = transcript
                self._last_partial_logged = ""
                self._log_event("transcript", transcript, partial=False)

    def _record_stats(self, rms: float, transcript: str) -> None:
        now = time.monotonic()
        with self._stats_lock:
            self._recent_stats.append((now, rms, transcript))
            while self._recent_stats and now - self._recent_stats[0][0] > DEBUG_WINDOW_SECONDS:
                self._recent_stats.popleft()
            if transcript:
                self._last_transcript = transcript

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:
            logger.exception("Remote wake client stopped with an error")

    async def _main(self) -> None:
        try:
            import websockets
        except ImportError:
            logger.exception("websockets is not available; remote wake-word detection disabled")
            return

        backoff = RECONNECT_BASE_S
        while not self._stop_event.is_set():
            try:
                async with websockets.connect(self._url, open_timeout=5.0, ping_interval=20.0) as ws:
                    await ws.send(json.dumps(hello_message(sample_rate=self._sample_rate, token=self._token)))
                    self._connected = True
                    backoff = RECONNECT_BASE_S
                    logger.info("Remote wake client connected to %s", self._url)
                    self._log_event("info", f"Connected to {self._url}")
                    await self._session(ws)
            except Exception as exc:
                self._connected = False
                if self._stop_event.is_set():
                    break
                self._log_event("warn", f"Connection failed: {exc}")
                logger.warning("Remote wake connection failed (%s); retrying in %.1fs", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, RECONNECT_MAX_S)
            finally:
                self._connected = False

    async def _session(self, ws: Any) -> None:
        sender = asyncio.create_task(self._send_loop(ws), name="remote-wake-send")
        receiver = asyncio.create_task(self._recv_loop(ws), name="remote-wake-recv")
        try:
            await asyncio.gather(sender, receiver)
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _send_loop(self, ws: Any) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            while True:
                try:
                    control = self._control_queue.get_nowait()
                except queue.Empty:
                    break
                if control == "pause":
                    await ws.send(json.dumps(pause_message()))
                    self._log_event("info", "Stream paused (BoBe awake)")
                elif control == "resume":
                    await ws.send(json.dumps(resume_message()))
                    self._log_event("info", "Stream resumed (BoBe asleep)")

            try:
                frame = await loop.run_in_executor(
                    None,
                    lambda: self._audio_queue.get(timeout=0.05),
                )
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue
            if frame is None or self._stop_event.is_set():
                break
            if self._paused:
                continue
            if self._gain != 1.0:
                boosted = np.clip(frame.astype(np.int32) * self._gain, -32768, 32767).astype(np.int16)
            else:
                boosted = frame
            rms = float(np.sqrt(np.mean(boosted.astype(np.float64) ** 2)))
            self._record_stats(rms, self._last_transcript)
            await ws.send(boosted.tobytes())

    async def _recv_loop(self, ws: Any) -> None:
        async for message in ws:
            if self._stop_event.is_set():
                break
            if isinstance(message, bytes):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            msg_type = payload.get("type")
            if msg_type == "ready":
                engine = str(payload.get("engine") or "")
                phrase = str(payload.get("phrase") or WAKE_PHRASE)
                with self._stats_lock:
                    self._daemon_engine = engine
                self._log_event("info", f"Daemon ready ({engine})", phrase=phrase)
                logger.info(
                    "Remote wake daemon ready (engine=%r, phrase=%r)",
                    payload.get("engine"),
                    payload.get("phrase"),
                )
            elif msg_type == "stats":
                self._apply_remote_stats(payload)
            elif msg_type == "wake":
                transcript = str(payload.get("transcript") or "")
                latency_ms = payload.get("latency_ms")
                self._apply_remote_stats(payload)
                self._log_event(
                    "wake",
                    f"Wake detected: {transcript!r}",
                    latency_ms=float(latency_ms) if latency_ms is not None else 0.0,
                )
                logger.info(
                    "Remote wake word detected (transcript=%r, latency_ms=%s)",
                    transcript,
                    latency_ms,
                )
                self._on_wake()
