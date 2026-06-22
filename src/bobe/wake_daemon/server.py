"""WebSocket + HTTP server for the Mac wake daemon."""

from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bobe.wake.protocol import parse_json, ready_message, stats_message, wake_message
from bobe.wake_daemon.config import WakeDaemonConfig, load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine


logger = logging.getLogger(__name__)


def create_app(config: WakeDaemonConfig | None = None) -> FastAPI:
    """Build the wake daemon FastAPI application."""
    runtime = config or load_wake_daemon_config()
    app = FastAPI(title="BoBe Wake Daemon", version="0.1.0")
    engine = WhisperWakeEngine(runtime)

    last_stats_at = 0.0

    @app.get("/status")
    def status() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "phrase": runtime.phrase,
                "engine": "faster-whisper",
                "model": runtime.whisper_model,
                "debug": engine.debug_state(),
            }
        )

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        nonlocal last_stats_at
        await websocket.accept()
        client_phrase = runtime.phrase
        paused = False

        try:
            hello_raw = await websocket.receive_text()
            hello = parse_json(hello_raw)
            if hello is None or hello.get("type") != "hello":
                await websocket.close(code=1003)
                return
            if runtime.token and hello.get("token") != runtime.token:
                logger.warning("Rejected wake stream with invalid token")
                await websocket.close(code=1008)
                return
            client_phrase = str(hello.get("phrase") or runtime.phrase).casefold()
            await websocket.send_json(ready_message(engine="faster-whisper", phrase=client_phrase))
            engine.resume()

            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break

                if "text" in message and message["text"] is not None:
                    payload = parse_json(message["text"])
                    if payload is None:
                        continue
                    msg_type = payload.get("type")
                    if msg_type == "pause":
                        paused = True
                        engine.pause()
                    elif msg_type == "resume":
                        paused = False
                        engine.resume()
                    continue

                data = message.get("bytes")
                if not data or paused:
                    continue

                pcm = np.frombuffer(data, dtype=np.int16)
                event = engine.feed(pcm)
                if event is not None:
                    transcript = str(event["transcript"])
                    latency_ms = float(event["latency_ms"])
                    logger.info("Wake phrase detected (transcript=%r, latency_ms=%.1f)", transcript, latency_ms)
                    await websocket.send_json(
                        wake_message(
                            transcript=transcript,
                            latency_ms=latency_ms,
                            phrase=str(event["phrase"]),
                        )
                    )

                now = time.monotonic()
                debug = engine.debug_state()
                interval = 0.25 if debug.get("in_speech") else 1.0
                if now - last_stats_at >= interval:
                    await websocket.send_json(
                        stats_message(
                            transcript=debug.get("transcript_last", ""),
                            partial=debug.get("transcript_partial", ""),
                            transcript_stream=debug.get("transcript_stream", []),
                            rms=debug.get("rms_last", 0.0),
                            in_speech=debug.get("in_speech", False),
                            paused=paused,
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
        finally:
            engine.pause()

    return app
