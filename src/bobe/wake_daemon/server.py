"""WebSocket + HTTP server for the Mac wake daemon."""

from __future__ import annotations
import hmac
import time
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import replace

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bobe.wake.phrases import matches_wake_phrase
from bobe.wake.protocol import (
    MSG_HELLO,
    MSG_LISTEN,
    CLOSE_POLICY_VIOLATION,
    CLOSE_UNSUPPORTED_DATA,
    parse_json,
    wake_message,
    ready_message,
    sleep_message,
    stats_message,
)
from bobe.wake.constants import WAKE_SAMPLE_RATE
from bobe.wake_daemon.config import WakeDaemonConfig, load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine, WhisperWakeSession, warn_if_phrases_unsupported


logger = logging.getLogger(__name__)


def create_app(config: WakeDaemonConfig | None = None) -> FastAPI:
    """Build the wake daemon FastAPI application."""
    runtime = config or load_wake_daemon_config()
    if not (runtime.token or "").strip():
        raise ValueError("BOBE_WAKE_TOKEN must be set to a non-empty value")
    warn_if_phrases_unsupported(runtime)

    def shared_engine() -> WhisperWakeEngine:
        # Lazily create the one shared engine; the engine's own load lock
        # makes the actual model load single-flight across threads.
        engine: WhisperWakeEngine | None = app.state.wake_engine
        if engine is None:
            engine = WhisperWakeEngine(runtime)
            app.state.wake_engine = engine
        return engine

    def preload_whisper_model() -> None:
        try:
            shared_engine().preload()
        except Exception:
            logger.exception("Whisper model preload failed; the first utterance will retry the load")

    @asynccontextmanager
    async def lifespan(started_app: FastAPI):
        # Preload in a background thread so the first wake never pays the
        # model-load cost; the engine's load lock makes early feeds wait for
        # this thread instead of loading a second model.
        preload_thread = threading.Thread(target=preload_whisper_model, name="whisper-preload", daemon=True)
        started_app.state.whisper_preload_thread = preload_thread
        preload_thread.start()
        yield

    app = FastAPI(title="BoBe Wake Daemon", version="0.1.0", lifespan=lifespan)
    app.state.wake_engine = None

    @app.get("/status")
    def status() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "phrase": runtime.phrase,
                "engine": "faster-whisper",
                "model": runtime.whisper_model,
            }
        )

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        client_phrase = runtime.phrase
        last_stats_at = 0.0
        session: WhisperWakeSession | None = None

        def apply_listen(payload: dict[str, object]) -> None:
            if session is None:
                return
            mode = str(payload.get("mode") or "wake").casefold()
            if mode not in {"wake", "sleep"}:
                return
            raw_phrases = payload.get("sleep_phrases")
            sleep_phrases = None
            if isinstance(raw_phrases, list):
                sleep_phrases = tuple(str(item) for item in raw_phrases if str(item).strip())
            session.set_listen_mode(mode, sleep_phrases=sleep_phrases)  # type: ignore[arg-type]
            logger.info("Wake stream listen mode set to %r", mode)

        try:
            hello_raw = await websocket.receive_text()
            hello = parse_json(hello_raw)
            if hello is None or hello.get("type") != MSG_HELLO:
                await websocket.close(code=CLOSE_UNSUPPORTED_DATA)
                return
            if hello.get("sample_rate") != WAKE_SAMPLE_RATE:
                logger.warning(
                    "Rejected wake stream with invalid sample_rate=%r (expected %d)",
                    hello.get("sample_rate"),
                    WAKE_SAMPLE_RATE,
                )
                await websocket.close(code=CLOSE_UNSUPPORTED_DATA)
                return
            hello_token = str(hello.get("token") or "").strip()
            if not hello_token or not hmac.compare_digest(hello_token, runtime.token or ""):
                logger.warning("Rejected wake stream with missing or invalid token")
                await websocket.close(code=CLOSE_POLICY_VIOLATION)
                return
            client_phrase = str(hello.get("phrase") or runtime.phrase).casefold()
            # Match against the daemon env phrase so Mac-side config wins when the
            # robot app still has a stale BOBE_WAKE_PHRASE (load_wake_daemon_config
            # guarantees a non-empty phrase).
            match_phrase = runtime.phrase.casefold()
            session = shared_engine().session(replace(runtime, phrase=match_phrase))
            await websocket.send_json(ready_message(engine="faster-whisper", phrase=match_phrase))

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message and message["text"] is not None:
                    payload = parse_json(message["text"])
                    if payload is None:
                        continue
                    msg_type = payload.get("type")
                    if msg_type == MSG_LISTEN:
                        apply_listen(payload)
                    continue

                data = message.get("bytes")
                if not data or session is None:
                    continue

                pcm = np.frombuffer(data, dtype=np.int16)
                event = await asyncio.to_thread(session.feed, pcm)
                if event is not None:
                    transcript = str(event["transcript"])
                    latency_ms = float(event["latency_ms"])
                    event_type = str(event.get("type") or "wake")
                    if event_type == "sleep":
                        logger.info("Sleep phrase detected (transcript=%r, latency_ms=%.1f)", transcript, latency_ms)
                        await websocket.send_json(
                            sleep_message(
                                transcript=transcript,
                                latency_ms=latency_ms,
                            )
                        )
                    else:
                        wake_transcript = transcript
                        if client_phrase and client_phrase != match_phrase:
                            if not matches_wake_phrase(transcript, phrase=client_phrase):
                                # Old robot builds still filter wake events with their
                                # local phrase; rewrite so the event is accepted.
                                wake_transcript = client_phrase
                        logger.info(
                            "Wake phrase detected (transcript=%r, client_transcript=%r, latency_ms=%.1f)",
                            transcript,
                            wake_transcript,
                            latency_ms,
                        )
                        await websocket.send_json(
                            wake_message(
                                transcript=wake_transcript,
                                latency_ms=latency_ms,
                                phrase=match_phrase,
                            )
                        )

                now = time.monotonic()
                debug = session.debug_state()
                interval = 0.15 if debug.get("in_speech") else 1.0
                if now - last_stats_at >= interval:
                    await websocket.send_json(
                        stats_message(
                            transcript=debug.get("transcript_last", ""),
                            partial=debug.get("transcript_partial", ""),
                            transcript_stream=debug.get("transcript_stream", []),
                            rms=debug.get("rms_last", 0.0),
                            in_speech=debug.get("in_speech", False),
                            listen_mode=debug.get("listen_mode", "wake"),
                            paused=debug.get("listen_mode") == "sleep",
                            latency_ms_last=debug.get("latency_ms_last", 0.0),
                            engine="faster-whisper",
                            model=runtime.whisper_model,
                        )
                    )
                    last_stats_at = now
        except WebSocketDisconnect:
            logger.info("Wake stream disconnected")
        except Exception:
            logger.exception("Wake stream failed")
            raise

    return app
