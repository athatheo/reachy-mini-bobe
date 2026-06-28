"""WebSocket + HTTP server for the Mac wake daemon."""

from __future__ import annotations
import time
import asyncio
import logging
from dataclasses import replace

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bobe.wake.protocol import parse_json, wake_message, sleep_message, ready_message, stats_message
from bobe.wake.constants import WAKE_SAMPLE_RATE
from bobe.wake_daemon.config import WakeDaemonConfig, load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine, WhisperWakeSession, whisper_engine_key


logger = logging.getLogger(__name__)


def create_app(config: WakeDaemonConfig | None = None) -> FastAPI:
    """Build the wake daemon FastAPI application."""
    runtime = config or load_wake_daemon_config()
    if not (runtime.token or "").strip():
        raise ValueError("BOBE_WAKE_TOKEN must be set to a non-empty value")
    app = FastAPI(title="BoBe Wake Daemon", version="0.1.0")
    engines: dict[tuple[str, str, str, str | None, str | None], WhisperWakeEngine] = {}
    app.state.wake_engines = engines

    def shared_engine() -> WhisperWakeEngine:
        key = whisper_engine_key(runtime)
        engine = engines.get(key)
        if engine is None:
            engine = WhisperWakeEngine(runtime)
            engines[key] = engine
        return engine

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
            if hello is None or hello.get("type") != "hello":
                await websocket.close(code=1003)
                return
            if hello.get("sample_rate") != WAKE_SAMPLE_RATE:
                logger.warning(
                    "Rejected wake stream with invalid sample_rate=%r (expected %d)",
                    hello.get("sample_rate"),
                    WAKE_SAMPLE_RATE,
                )
                await websocket.close(code=1003)
                return
            hello_token = str(hello.get("token") or "").strip()
            if hello_token != runtime.token:
                logger.warning("Rejected wake stream with missing or invalid token")
                await websocket.close(code=1008)
                return
            client_phrase = str(hello.get("phrase") or runtime.phrase).casefold()
            session = shared_engine().session(replace(runtime, phrase=client_phrase))
            await websocket.send_json(ready_message(engine="faster-whisper", phrase=client_phrase))

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message and message["text"] is not None:
                    payload = parse_json(message["text"])
                    if payload is None:
                        continue
                    msg_type = payload.get("type")
                    if msg_type == "listen":
                        apply_listen(payload)
                    elif msg_type == "pause":
                        apply_listen({"mode": "sleep"})
                    elif msg_type == "resume":
                        apply_listen({"mode": "wake"})
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
                        logger.info("Wake phrase detected (transcript=%r, latency_ms=%.1f)", transcript, latency_ms)
                        await websocket.send_json(
                            wake_message(
                                transcript=transcript,
                                latency_ms=latency_ms,
                                phrase=str(event["phrase"]),
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
