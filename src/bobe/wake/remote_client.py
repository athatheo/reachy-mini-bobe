"""Remote wake-word client that streams PCM to a Mac-side daemon."""

# ruff: noqa: D102,D107

from __future__ import annotations
import json
import time
import queue
import asyncio
import logging
import threading
from typing import Any
from collections import deque

import numpy as np
from numpy.typing import NDArray

from bobe.wake.phrases import WAKE_PHRASE, DEFAULT_SLEEP_PHRASES, matches_wake_phrase, matches_sleep_command
from bobe.wake.protocol import hello_message, listen_message
from bobe.wake.constants import WAKE_SAMPLE_RATE, DEBUG_WINDOW_SECONDS


logger = logging.getLogger(__name__)

RECONNECT_BASE_S = 0.5
RECONNECT_MAX_S = 10.0
# A 1008 (policy violation) close means the daemon rejected our handshake —
# usually a bad/missing BOBE_WAKE_TOKEN. Fast retries can never succeed.
AUTH_RETRY_S = 60.0


def _close_code(exc: BaseException) -> int | None:
    """Extract the websocket close code from a ConnectionClosed exception."""
    rcvd = getattr(exc, "rcvd", None)
    code = getattr(rcvd, "code", None)
    if code is None:
        code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


class RemoteWakeClient:
    """Stream mic PCM to a remote wake daemon and receive wake events."""

    def __init__(
        self,
        on_wake: Any,
        *,
        url: str,
        token: str | None = None,
        gain: float = 1.0,
        phrase: str = WAKE_PHRASE,
        sample_rate: int = WAKE_SAMPLE_RATE,
        on_sleep: Any | None = None,
        sleep_phrases: tuple[str, ...] = DEFAULT_SLEEP_PHRASES,
    ) -> None:
        self._on_wake = on_wake
        self._on_sleep = on_sleep
        self._phrase = phrase.strip().casefold() or WAKE_PHRASE
        self._sleep_phrases = sleep_phrases
        self._url = url
        self._token = (token or "").strip() or None
        self._gain = gain
        self._sample_rate = sample_rate
        self._audio_queue: queue.Queue[NDArray[np.int16] | None] = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        # Desired listen mode, guarded by its lock. The send loop resolves the
        # mode at send time (latest-wins), so concurrent mode changes and
        # reconnect replays can never deliver a stale mode to the daemon.
        self._mode_lock = threading.Lock()
        self._listen_mode = "wake"
        self._mode_send_pending = threading.Event()
        self._thread: threading.Thread | None = None
        self._stats_lock = threading.Lock()
        self._recent_stats: deque[tuple[float, float, str]] = deque()
        self._event_log: deque[dict[str, float | int | str | bool]] = deque(maxlen=40)
        self._remote_stats: dict[str, float | int | str | bool] = {}
        self._daemon_engine = ""
        self._connected = False
        self._auth_error: str | None = None
        self._last_transcript = ""
        self._transcript_stream: list[dict[str, float | int | str | bool]] = []
        self._display_lines: list[str] = []

    @property
    def phrase(self) -> str:
        return self._phrase

    def is_running(self) -> bool:
        """Return whether the background client thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running():
            return
        self._thread = None
        self._stop_event.clear()
        # Remove stale frames and stop sentinels left over from a previous run.
        self._drain_audio_queue()
        self._thread = threading.Thread(target=self._run, name="remote-wake-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._audio_queue.put_nowait(None)
        except queue.Full:
            pass
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
            if not thread.is_alive():
                self._thread = None
            else:
                logger.warning("Remote wake client thread did not stop within timeout")

    def feed(self, frame: NDArray[np.int16]) -> None:
        try:
            self._audio_queue.put_nowait(frame.reshape(-1).astype(np.int16, copy=False))
        except queue.Full:
            pass

    def listen_for_sleep(self) -> None:
        """Listen for sleep phrases while BoBe is awake."""
        self._set_listen_mode("sleep")

    def listen_for_wake(self) -> None:
        """Listen for wake phrases while BoBe is asleep."""
        self._set_listen_mode("wake")

    def _set_listen_mode(self, mode: str) -> None:
        with self._mode_lock:
            self._listen_mode = mode
        # Listen mode is idempotent state — only the latest request matters.
        # The flag just says "a send is due"; the payload is built from the
        # current desired mode when the send loop gets to it.
        self._mode_send_pending.set()

    def _current_listen_mode(self) -> str:
        with self._mode_lock:
            return self._listen_mode

    def _drain_audio_queue(self) -> None:
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def debug_state(self) -> dict[str, Any]:
        now = time.monotonic()
        listen_mode = self._current_listen_mode()
        with self._stats_lock:
            while self._recent_stats and now - self._recent_stats[0][0] > DEBUG_WINDOW_SECONDS:
                self._recent_stats.popleft()
            entries = list(self._recent_stats)
            events = list(self._event_log)
            remote_stats = dict(self._remote_stats)
            daemon_engine = self._daemon_engine
            transcript_stream = list(self._transcript_stream)
            display_lines = list(self._display_lines)
        rms_values = [rms for _, rms, _ in entries]
        partial = str(remote_stats.get("partial") or "")
        return {
            "backend": "remote",
            "phrase": self._phrase,
            "url": self._url,
            "gain": self._gain,
            "frames_window": len(entries),
            "rms_peak": round(max(rms_values), 1) if rms_values else 0.0,
            "rms_last": round(rms_values[-1], 1) if rms_values else 0.0,
            "transcript_last": self._last_transcript,
            "transcript_partial": partial,
            "transcript_stream": transcript_stream[-12:],
            "transcript_display": display_lines[-20:],
            "connected": self._connected,
            "auth_error": self._auth_error,
            "listen_mode": listen_mode,
            "paused": listen_mode == "sleep",
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
                stats[key] = payload[key]
        transcript = str(payload.get("transcript") or "")
        partial = str(payload.get("partial") or "")
        stream = payload.get("transcript_stream")
        with self._stats_lock:
            self._remote_stats.update(stats)
            if isinstance(stream, list):
                self._transcript_stream = [entry for entry in stream if isinstance(entry, dict)][-12:]
            if partial:
                self._last_transcript = partial
                line = f"[live] {partial}"
                if not self._display_lines or self._display_lines[-1] != line:
                    if self._display_lines and self._display_lines[-1].startswith("[live] "):
                        self._display_lines[-1] = line
                    else:
                        self._display_lines.append(line)
                    if len(self._display_lines) > 40:
                        self._display_lines = self._display_lines[-40:]
            elif transcript:
                self._last_transcript = transcript
                line = f"[final] {transcript}"
                if self._display_lines and self._display_lines[-1].startswith("[live] "):
                    self._display_lines[-1] = line
                elif not self._display_lines or self._display_lines[-1] != line:
                    self._display_lines.append(line)
                    if len(self._display_lines) > 40:
                        self._display_lines = self._display_lines[-40:]

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
                    backoff = RECONNECT_BASE_S
                    await self._run_connection(ws)
            except Exception as exc:
                self._connected = False
                if self._stop_event.is_set():
                    break
                if _close_code(exc) == 1008:
                    self._auth_error = (
                        "Wake daemon rejected the handshake (close code 1008); check BOBE_WAKE_TOKEN"
                    )
                    self._log_event("error", self._auth_error)
                    logger.error("%s. Retrying in %.0fs.", self._auth_error, AUTH_RETRY_S)
                    await self._sleep_unless_stopped(AUTH_RETRY_S)
                    continue
                self._log_event("warn", f"Connection failed: {exc}")
                logger.warning("Remote wake connection failed (%s); retrying in %.1fs", exc, backoff)
                await self._sleep_unless_stopped(backoff)
                backoff = min(backoff * 2.0, RECONNECT_MAX_S)
            finally:
                self._connected = False

    async def _run_connection(self, ws: Any) -> None:
        await ws.send(
            json.dumps(hello_message(sample_rate=self._sample_rate, token=self._token, phrase=self._phrase))
        )
        # Drop mic audio queued while disconnected: replaying it into the fresh
        # (no-refractory) daemon session would fire delayed ghost wakes.
        self._drain_audio_queue()
        # The daemon starts every connection in wake mode, so always replay the
        # current listen mode — a reconnect while BoBe is awake must restore
        # sleep-phrase detection. Only the flag is set here; the send loop
        # reads the desired mode at send time, so a concurrent mode change
        # cannot be clobbered by this replay.
        self._mode_send_pending.set()
        self._connected = True
        logger.info("Remote wake client connected to %s", self._url)
        self._log_event("info", f"Connected to {self._url}")
        await self._session(ws)

    async def _sleep_unless_stopped(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            await asyncio.sleep(min(0.25, remaining))

    async def _session(self, ws: Any) -> None:
        sender = asyncio.create_task(self._send_loop(ws), name="remote-wake-send")
        receiver = asyncio.create_task(self._recv_loop(ws), name="remote-wake-recv")
        try:
            # Either loop finishing must end the session: after the stop
            # sentinel the recv loop would otherwise sit in `async for`
            # forever and stop() would always hit its join timeout.
            done, _pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc
        finally:
            sender.cancel()
            receiver.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    async def _send_loop(self, ws: Any) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            if self._mode_send_pending.is_set():
                # Clear before reading the mode: a concurrent change after the
                # read re-flags the send, so the daemon always converges to
                # the latest requested mode.
                self._mode_send_pending.clear()
                mode = self._current_listen_mode()
                payload = listen_message(
                    mode=mode,
                    sleep_phrases=self._sleep_phrases if mode == "sleep" else None,
                )
                try:
                    await ws.send(json.dumps(payload))
                except Exception:
                    # Keep the mode change for the reconnect: re-flag it so it
                    # is not silently lost when the socket is already dead.
                    self._mode_send_pending.set()
                    raise
                if mode == "sleep":
                    self._log_event("info", "Listening for sleep phrases (BoBe awake)")
                else:
                    self._log_event("info", "Listening for wake phrase (BoBe asleep)")
                # Re-check for an even newer mode before blocking on audio.
                continue

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
            if self._gain != 1.0:
                boosted = np.clip(frame.astype(np.int32) * self._gain, -32768, 32767).astype(np.int16)
            else:
                boosted = frame
            rms = float(np.sqrt(np.mean(boosted.astype(np.float64) ** 2)))
            self._record_stats(rms, self._last_transcript)
            await ws.send(boosted.tobytes())

    def _handle_wake_payload(self, payload: dict[str, Any]) -> None:
        transcript = str(payload.get("transcript") or "")
        latency_ms = payload.get("latency_ms")
        self._apply_remote_stats(payload)
        if not matches_wake_phrase(transcript, phrase=self.phrase):
            self._log_event(
                "warn",
                f"Ignored wake without phrase match: {transcript!r}",
                latency_ms=float(latency_ms) if latency_ms is not None else 0.0,
            )
            logger.warning(
                "Ignored remote wake event without phrase match (transcript=%r)",
                transcript,
            )
            return
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

    def _handle_sleep_payload(self, payload: dict[str, Any]) -> None:
        transcript = str(payload.get("transcript") or "")
        latency_ms = payload.get("latency_ms")
        self._apply_remote_stats(payload)
        # Strict command re-validation: a substring match would put BoBe to
        # sleep mid-conversation ("my toddler won't go to sleep, any tips?").
        if not matches_sleep_command(transcript, self._sleep_phrases):
            self._log_event(
                "warn",
                f"Ignored sleep without phrase match: {transcript!r}",
                latency_ms=float(latency_ms) if latency_ms is not None else 0.0,
            )
            logger.warning(
                "Ignored remote sleep event without phrase match (transcript=%r)",
                transcript,
            )
            return
        self._log_event(
            "sleep",
            f"Sleep detected: {transcript!r}",
            latency_ms=float(latency_ms) if latency_ms is not None else 0.0,
        )
        logger.info(
            "Remote sleep phrase detected (transcript=%r, latency_ms=%s)",
            transcript,
            latency_ms,
        )
        if self._on_sleep is not None:
            self._on_sleep()

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
                phrase = str(payload.get("phrase") or self._phrase)
                self._auth_error = None
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
                self._handle_wake_payload(payload)
            elif msg_type == "sleep":
                self._handle_sleep_payload(payload)
