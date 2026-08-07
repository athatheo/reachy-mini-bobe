"""WebSocket + HTTP server for the Mac wake daemon."""

from __future__ import annotations
import hmac
import time
import uuid
import base64
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import replace

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from bobe.wake.phrases import matches_wake_phrase
from bobe.wake.protocol import (
    MSG_HELLO,
    MSG_LISTEN,
    CLOSE_POLICY_VIOLATION,
    CLOSE_UNSUPPORTED_DATA,
    parse_json,
    wake_message,
    emote_message,
    ready_message,
    sleep_message,
    speak_message,
    stats_message,
    announce_message,
)
from bobe.wake.constants import WAKE_SAMPLE_RATE
from bobe.wake_daemon.audio import ROBOT_SPEECH_RATE, AudioDecodeError, decode_audio_to_pcm
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
    # Authenticated robot wake streams, used to push announcements robot-ward.
    app.state.stream_connections = set()
    # Robot utterances awaiting pickup by the Hermes bobe plugin (long-poll
    # consumer). Bounded so an absent consumer can never grow memory; a full
    # queue drops the OLDEST utterance — the newest one is what the user just
    # said and still expects an answer to.
    app.state.utterances = asyncio.Queue(maxsize=16)

    def enqueue_utterance(text: str) -> None:
        """Queue an utterance for the agent, evicting the oldest when full."""
        while True:
            try:
                app.state.utterances.put_nowait({"text": text, "ts": round(time.time(), 3)})
                return
            except asyncio.QueueFull:
                try:
                    dropped = app.state.utterances.get_nowait()
                    logger.warning("Utterance queue full; dropped %r", dropped.get("text"))
                except asyncio.QueueEmpty:
                    pass

    app.state.enqueue_utterance = enqueue_utterance
    # Half-duplex echo guard: while a relayed speech clip is playing on the
    # robot there is no echo cancellation, so converse-mode capture would
    # transcribe the robot's own voice. /v1/speak advances this deadline by
    # the clip duration plus a grace second; converse capture drops frames
    # until it passes.
    app.state.speak_suppress_until = 0.0

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

    @app.post("/v1/announce")
    async def announce(request: Request) -> JSONResponse:
        provided_token = (request.headers.get("x-bobe-wake-token") or "").strip()
        if not provided_token or not hmac.compare_digest(provided_token, runtime.token or ""):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

        try:
            payload = await request.json()
        except Exception:
            payload = {}
        message = str(payload.get("message") or "").strip() if isinstance(payload, dict) else ""
        if not message:
            return JSONResponse({"ok": False, "error": "empty_message"}, status_code=400)

        connections = list(app.state.stream_connections)
        if not connections:
            return JSONResponse({"ok": False, "error": "no_robot_connected"}, status_code=409)

        delivered = 0
        for connection in connections:
            try:
                await connection.send_json(announce_message(text=message))
                delivered += 1
            except Exception:
                logger.warning("Failed to deliver announcement to a robot stream", exc_info=True)
        if delivered == 0:
            return JSONResponse({"ok": False, "error": "delivery_failed"}, status_code=502)
        return JSONResponse({"ok": True, "delivered": delivered})

    @app.post("/v1/speak")
    async def speak(request: Request) -> JSONResponse:
        """Relay pre-synthesized speech audio (e.g. Hermes TTS) to the robot.

        Accepts either raw audio bytes (``Content-Type: audio/*``) or JSON
        ``{"audio_b64": ...}``. Any common container is decoded to mono s16le
        PCM at the robot's output rate and chunked over the wake WebSocket.
        """
        provided_token = (request.headers.get("x-bobe-wake-token") or "").strip()
        if not provided_token or not hmac.compare_digest(provided_token, runtime.token or ""):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

        content_type = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
        if content_type.startswith("audio/") or content_type == "application/octet-stream":
            audio_bytes = await request.body()
        else:
            try:
                payload = await request.json()
            except Exception:
                payload = {}
            audio_b64 = payload.get("audio_b64") if isinstance(payload, dict) else None
            if not isinstance(audio_b64, str) or not audio_b64:
                return JSONResponse({"ok": False, "error": "missing_audio"}, status_code=400)
            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                return JSONResponse({"ok": False, "error": "bad_audio_b64"}, status_code=400)

        try:
            pcm = await asyncio.to_thread(decode_audio_to_pcm, audio_bytes, ROBOT_SPEECH_RATE)
        except AudioDecodeError as exc:
            logger.warning("Rejected /v1/speak audio: %s", exc)
            return JSONResponse({"ok": False, "error": f"undecodable_audio: {exc}"}, status_code=400)
        if pcm.size == 0:
            return JSONResponse({"ok": False, "error": "empty_audio"}, status_code=400)

        connections = list(app.state.stream_connections)
        if not connections:
            return JSONResponse({"ok": False, "error": "no_robot_connected"}, status_code=409)

        clip_id = uuid.uuid4().hex[:12]
        chunk_samples = ROBOT_SPEECH_RATE  # 1 s per chunk keeps messages small
        delivered = 0
        for connection in connections:
            try:
                for seq, start in enumerate(range(0, pcm.size, chunk_samples)):
                    chunk = pcm[start : start + chunk_samples]
                    await connection.send_json(
                        speak_message(
                            clip_id=clip_id,
                            seq=seq,
                            pcm_b64=base64.b64encode(chunk.tobytes()).decode("ascii"),
                            rate=ROBOT_SPEECH_RATE,
                            last=start + chunk_samples >= pcm.size,
                        )
                    )
                delivered += 1
            except Exception:
                logger.warning("Failed to deliver speech clip to a robot stream", exc_info=True)
        if delivered == 0:
            return JSONResponse({"ok": False, "error": "delivery_failed"}, status_code=502)
        seconds = round(pcm.size / ROBOT_SPEECH_RATE, 2)
        # The robot primes ~1 s of buffer before real-time playback; suppress
        # converse capture for the clip length plus that lead and a grace tail.
        app.state.speak_suppress_until = time.monotonic() + seconds + 2.0
        logger.info("Relayed speech clip %s (%.1fs) to %d robot stream(s)", clip_id, seconds, delivered)
        return JSONResponse({"ok": True, "delivered": delivered, "seconds": seconds})

    @app.post("/v1/emote")
    async def emote(request: Request) -> JSONResponse:
        """Relay an emotion-move request (from the Hermes plugin) to the robot."""
        provided_token = (request.headers.get("x-bobe-wake-token") or "").strip()
        if not provided_token or not hmac.compare_digest(provided_token, runtime.token or ""):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        emotion = str(payload.get("emotion") or "").strip() if isinstance(payload, dict) else ""
        if not emotion:
            return JSONResponse({"ok": False, "error": "empty_emotion"}, status_code=400)

        connections = list(app.state.stream_connections)
        if not connections:
            return JSONResponse({"ok": False, "error": "no_robot_connected"}, status_code=409)
        delivered = 0
        for connection in connections:
            try:
                await connection.send_json(emote_message(emotion=emotion))
                delivered += 1
            except Exception:
                logger.warning("Failed to deliver emote to a robot stream", exc_info=True)
        if delivered == 0:
            return JSONResponse({"ok": False, "error": "delivery_failed"}, status_code=502)
        logger.info("Relayed emote %r to %d robot stream(s)", emotion, delivered)
        return JSONResponse({"ok": True, "delivered": delivered})

    @app.get("/v1/utterances")
    async def utterances(request: Request) -> JSONResponse:
        """Long-poll robot utterances (consumed by the Hermes bobe plugin).

        Waits up to ``wait`` seconds (default 25, capped at 60) for the first
        utterance, then drains everything queued so multi-utterance bursts
        arrive together.
        """
        provided_token = (request.headers.get("x-bobe-wake-token") or "").strip()
        if not provided_token or not hmac.compare_digest(provided_token, runtime.token or ""):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)

        try:
            wait_s = min(max(float(request.query_params.get("wait", 25.0)), 0.0), 60.0)
        except (TypeError, ValueError):
            wait_s = 25.0

        def drain() -> list[dict[str, object]]:
            drained: list[dict[str, object]] = []
            while True:
                try:
                    drained.append(app.state.utterances.get_nowait())
                except asyncio.QueueEmpty:
                    return drained

        # Drain non-blocking first: waiting on get() would miss items that are
        # already queued when ``wait`` is 0.
        events = drain()
        if not events and wait_s > 0:
            try:
                events.append(await asyncio.wait_for(app.state.utterances.get(), timeout=wait_s))
                events.extend(drain())
            except asyncio.TimeoutError:
                pass
        return JSONResponse({"ok": True, "events": events})

    @app.post("/v1/utterances")
    async def inject_utterance(request: Request) -> JSONResponse:
        """Inject an utterance as if the robot heard it (testing / text lane)."""
        provided_token = (request.headers.get("x-bobe-wake-token") or "").strip()
        if not provided_token or not hmac.compare_digest(provided_token, runtime.token or ""):
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
        if not text:
            return JSONResponse({"ok": False, "error": "empty_text"}, status_code=400)
        enqueue_utterance(text)
        return JSONResponse({"ok": True})

    @app.websocket("/v1/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        client_phrase = runtime.phrase
        last_stats_at = 0.0
        session: WhisperWakeSession | None = None
        listen_mode = "wake"
        suppressed = False

        def apply_listen(payload: dict[str, object]) -> None:
            nonlocal listen_mode
            if session is None:
                return
            mode = str(payload.get("mode") or "wake").casefold()
            if mode not in {"wake", "sleep", "converse"}:
                return
            raw_phrases = payload.get("sleep_phrases")
            sleep_phrases = None
            if isinstance(raw_phrases, list):
                sleep_phrases = tuple(str(item) for item in raw_phrases if str(item).strip())
            session.set_listen_mode(mode, sleep_phrases=sleep_phrases)  # type: ignore[arg-type]
            listen_mode = mode
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
            app.state.stream_connections.add(websocket)

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

                # Echo guard: drop converse-mode audio while a relayed clip is
                # playing on the robot, and clear any half-captured utterance
                # when suppression ends so clip tails never reach the agent.
                if listen_mode == "converse":
                    if time.monotonic() < app.state.speak_suppress_until:
                        if not suppressed:
                            suppressed = True
                            session.reset()
                        continue
                    if suppressed:
                        suppressed = False
                        session.reset()

                pcm = np.frombuffer(data, dtype=np.int16)
                event = await asyncio.to_thread(session.feed, pcm)
                if event is not None:
                    transcript = str(event["transcript"])
                    latency_ms = float(event["latency_ms"])
                    event_type = str(event.get("type") or "wake")
                    if event_type == "utterance":
                        enqueue_utterance(transcript)
                        continue
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
        finally:
            app.state.stream_connections.discard(websocket)

    return app
