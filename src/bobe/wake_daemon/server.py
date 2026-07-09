"""WebSocket + HTTP server for the Mac wake daemon."""

from __future__ import annotations
import hmac
import time
import asyncio
import logging
from dataclasses import replace

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bobe.wake.protocol import parse_json, wake_message, ready_message, sleep_message, stats_message
from bobe.wake.constants import WAKE_SAMPLE_RATE
from bobe.wake_daemon.config import WakeDaemonConfig, load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine, WhisperWakeSession, whisper_engine_key
from bobe.wake_daemon.launcher import ClaudeCodeLauncher
from bobe.wake_daemon.claude_session import ClaudeCodeSessionManager


logger = logging.getLogger(__name__)

_CLAUDE_CODE_ERROR_STATUS = {
    "busy": 409,
    "cooldown": 429,
    "disabled": 403,
    "empty_command": 400,
    "invalid_config": 400,
}


def create_app(config: WakeDaemonConfig | None = None) -> FastAPI:
    """Build the wake daemon FastAPI application."""
    runtime = config or load_wake_daemon_config()
    if not (runtime.token or "").strip():
        raise ValueError("BOBE_WAKE_TOKEN must be set to a non-empty value")
    app = FastAPI(title="BoBe Wake Daemon", version="0.1.0")
    engines: dict[tuple[str, str, str, str | None, str | None], WhisperWakeEngine] = {}
    app.state.wake_engines = engines
    app.state.claude_code_launcher = ClaudeCodeLauncher(runtime)
    app.state.claude_code_session_manager = ClaudeCodeSessionManager(runtime)

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

    def require_claude_code_control(request: Request) -> JSONResponse | None:
        if not runtime.claude_code_launch_enabled:
            return JSONResponse({"ok": False, "error": "disabled"}, status_code=403)

        expected_token = (runtime.claude_code_launch_token or "").strip()
        if not expected_token:
            return JSONResponse({"ok": False, "error": "missing_launch_token"}, status_code=503)

        provided_token = (request.headers.get("x-bobe-launch-token") or "").strip()
        if not provided_token or not hmac.compare_digest(provided_token, expected_token):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        return None

    def response_for_result(result: dict[str, object]) -> JSONResponse:
        if result.get("ok"):
            return JSONResponse(result)

        error = str(result.get("error") or "")
        status_code = _CLAUDE_CODE_ERROR_STATUS.get(error, 500)
        return JSONResponse(result, status_code=status_code)

    @app.post("/v1/launch/claude-code")
    async def launch_claude_code(request: Request) -> JSONResponse:
        auth_error = require_claude_code_control(request)
        if auth_error is not None:
            return auth_error
        launcher: ClaudeCodeLauncher = app.state.claude_code_launcher
        result = await asyncio.to_thread(launcher.launch)
        return response_for_result(result)

    @app.post("/v1/claude-code/session/start")
    async def start_claude_code_session(request: Request) -> JSONResponse:
        auth_error = require_claude_code_control(request)
        if auth_error is not None:
            return auth_error
        manager: ClaudeCodeSessionManager = app.state.claude_code_session_manager
        result = await asyncio.to_thread(manager.start)
        return response_for_result(result)

    @app.post("/v1/claude-code/session/send")
    async def send_claude_code_command(request: Request) -> JSONResponse:
        auth_error = require_claude_code_control(request)
        if auth_error is not None:
            return auth_error
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        command = str(payload.get("command") or "").strip() if isinstance(payload, dict) else ""
        manager: ClaudeCodeSessionManager = app.state.claude_code_session_manager
        result = await asyncio.to_thread(manager.send, command)
        return response_for_result(result)

    @app.get("/v1/claude-code/session/status")
    async def claude_code_session_status(request: Request) -> JSONResponse:
        auth_error = require_claude_code_control(request)
        if auth_error is not None:
            return auth_error
        manager: ClaudeCodeSessionManager = app.state.claude_code_session_manager
        return JSONResponse(manager.status())

    @app.post("/v1/claude-code/session/stop")
    async def stop_claude_code_session(request: Request) -> JSONResponse:
        auth_error = require_claude_code_control(request)
        if auth_error is not None:
            return auth_error
        manager: ClaudeCodeSessionManager = app.state.claude_code_session_manager
        result = await asyncio.to_thread(manager.stop)
        return response_for_result(result)

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
