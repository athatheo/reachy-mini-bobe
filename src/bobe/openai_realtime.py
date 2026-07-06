import json
import time
import uuid
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path

import cv2
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from bobe.config import config
from bobe.prompts import get_session_voice, get_realtime_session_instructions
from bobe.wake_word import (
    WAKE_SAMPLE_RATE,
    DEFAULT_FLUSH_SECONDS,
    WakeSession,
    AudioRingBuffer,
    is_sleep_phrase,
    load_wake_config,
    create_wake_detector,
    wake_detector_error,
)
from bobe.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
)
from bobe.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000

# Cost tracking from usage data (pricing as of Feb 2026 https://openai.com/api/pricing/)
AUDIO_INPUT_COST_PER_1M = 32.0
AUDIO_OUTPUT_COST_PER_1M = 64.0
TEXT_INPUT_COST_PER_1M = 4.0
TEXT_OUTPUT_COST_PER_1M = 16.0
IMAGE_INPUT_COST_PER_1M = 5.0

_RESPONSE_DONE_TIMEOUT: Final[float] = 30.0
# Ignore server VAD briefly after assistant audio so speaker echo does not freeze motors.
_ASSISTANT_VAD_GUARD_S: Final[float] = 0.4
# World-frame head translation applied when falling asleep (millimeters, vertical).
_SLEEP_HEAD_Z_OFFSET_MM: Final[float] = 30.0


class RealtimeSessionError(Exception):
    """Retryable failure while establishing or updating a realtime session."""


class PartialTranscriptDebouncer:
    """Debounce partial ASR transcripts before emitting them to the UI."""

    # ruff: noqa: D102, D107

    def __init__(
        self,
        output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]",
        delay: float = 0.5,
    ) -> None:
        self._output_queue = output_queue
        self._delay = delay
        self._task: asyncio.Task[None] | None = None
        self._sequence = 0

    async def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def schedule(self, transcript: str) -> None:
        await self.cancel()
        self._sequence += 1
        sequence = self._sequence
        self._task = asyncio.create_task(self._emit(transcript, sequence))

    async def _emit(self, transcript: str, sequence: int) -> None:
        try:
            await asyncio.sleep(self._delay)
            if self._sequence == sequence:
                await self._output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise


def _compute_response_cost(usage: Any) -> float:
    """Compute dollar cost from a response usage object."""
    inp = getattr(usage, "input_token_details", None)
    out = getattr(usage, "output_token_details", None)
    cost = 0.0
    if inp:
        cost += (getattr(inp, "audio_tokens", 0) or 0) * AUDIO_INPUT_COST_PER_1M / 1e6
        cost += (getattr(inp, "text_tokens", 0) or 0) * TEXT_INPUT_COST_PER_1M / 1e6
        cost += (getattr(inp, "image_tokens", 0) or 0) * IMAGE_INPUT_COST_PER_1M / 1e6
    if out:
        cost += (getattr(out, "audio_tokens", 0) or 0) * AUDIO_OUTPUT_COST_PER_1M / 1e6
        cost += (getattr(out, "text_tokens", 0) or 0) * TEXT_OUTPUT_COST_PER_1M / 1e6
    return cost


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        # Override typing of the sample rates to match OpenAI's requirements
        self.output_sample_rate: Literal[24000] = self.output_sample_rate
        self.input_sample_rate: Literal[24000] = self.input_sample_rate

        self.deps = deps

        # Override type annotations for OpenAI strict typing (only for values used in API)
        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        # Track how the API key was provided (env vs textbox) and its value
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None

        self._partial_debouncer = PartialTranscriptDebouncer(self.output_queue)

        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Cost tracking
        self.cumulative_cost: float = 0.0

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._last_response_rejected: bool = False
        self._last_assistant_audio_at: float = 0.0
        self._last_reconnect_attempt_at: float = 0.0
        self._latest_user_transcript: str = ""
        self._sleep_pending: bool = False
        self._realtime_session_task: asyncio.Task[None] | None = None

        # Local wake-word gating: while asleep, mic audio never leaves the robot.
        self.wake_config = load_wake_config()
        self.wake_session = WakeSession(timeout_s=self.wake_config.timeout_s)
        # Diagnostic mode: keep scoring the mic but never wake or stream upstream.
        self.wake_test_mode = False
        self.wake_test_detections = 0
        self._wake_buffer = AudioRingBuffer(sample_rate=self.input_sample_rate)
        self._wake_detector = create_wake_detector(
            on_wake=self.wake_session.request_wake,
            config=self.wake_config,
            on_sleep=self.wake_session.request_sleep,
        )
        self.wake_gating_enabled = self._wake_detector is not None
        if self.wake_gating_enabled:
            self.wake_error = None
        else:
            self.wake_error = wake_detector_error(self.wake_config) or "Wake-word detector unavailable"
            logger.error("Wake-word gating disabled: %s", self.wake_error)
            # Fallback: without a detector, stay in an always-on session so mic + replies work.
            self.wake_session.wake()
            logger.warning(
                "Wake-word detection unavailable; running in always-on mode until wake is configured"
            )

    def _session_accepts_responses(self) -> bool:
        """Return whether assistant responses should play (wake gating or always-on fallback)."""
        if not self.wake_gating_enabled:
            return not self._sleep_pending
        return self.wake_session.awake and not self._sleep_pending

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    def _should_ignore_server_vad(self) -> bool:
        """Return True when server VAD is likely picking up BoBe's own speaker output."""
        if not self._response_done_event.is_set():
            return True
        return (time.monotonic() - self._last_assistant_audio_at) < _ASSISTANT_VAD_GUARD_S

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from bobe.config import config as _config
            from bobe.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_realtime_session_instructions()
                voice = get_session_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session={
                            "type": "realtime",
                            "instructions": instructions,
                            "audio": {"output": {"voice": voice}},
                        },
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        openai_api_key = config.OPENAI_API_KEY
        if self.gradio_mode and not openai_api_key:
            # api key was not found in .env or in the environment variables
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args[3]) > 0 else None
            if textbox_api_key is not None:
                openai_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                openai_api_key = config.OPENAI_API_KEY
        else:
            if not openai_api_key or not openai_api_key.strip():
                # In headless console mode, LocalStream now blocks startup until the key is provided.
                # However, unit tests may invoke this handler directly with a stubbed client.
                # To keep tests hermetic without requiring a real key, fall back to a placeholder.
                logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                openai_api_key = "DUMMY"

        self.client = AsyncOpenAI(api_key=openai_api_key)

        if self._wake_detector is not None:
            self._wake_detector.start()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self._realtime_session_task = asyncio.create_task(
                    self._run_realtime_session(),
                    name="openai-realtime-session",
                )
                await self._realtime_session_task
                # Normal exit from the session, stop retrying
                return
            except (ConnectionClosedError, RealtimeSessionError) as e:
                # Abrupt close or session setup failure → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                logger.error(
                    "Realtime session unavailable after %d attempts; keeping settings UI alive until stop",
                    max_attempts,
                )
                try:
                    while True:
                        await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    return
                return
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _ensure_openai_connection(self, timeout: float = 5.0) -> bool:
        """Wait for an active Realtime connection, restarting the session if needed."""
        if self.connection is not None:
            return True
        if getattr(self, "client", None) is None:
            logger.warning("OpenAI client not initialized; cannot connect")
            return False

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        if self.connection is not None:
            return True

        logger.warning("No OpenAI Realtime connection; restarting session")
        await self._restart_session()
        if self.connection is not None:
            return True

        try:
            await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.error("OpenAI Realtime connection unavailable after restart")
            return False
        return self.connection is not None

    async def _await_or_cancel_session_task(self, *, timeout: float = 2.0) -> None:
        """Wait for the current session task to finish, or cancel it if it stalls."""
        task = self._realtime_session_task
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Cancelling stale realtime session task after timeout")
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _invalidate_realtime_connection(self) -> None:
        """Drop the active connection so the session task can exit and restart."""
        conn = self.connection
        self.connection = None
        try:
            self._connected_event.clear()
        except Exception:
            pass
        if conn is not None:
            try:
                await conn.close()
            except Exception:
                pass
        await self._await_or_cancel_session_task()

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            await self._invalidate_realtime_connection()

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return

            if self._realtime_session_task is not None and not self._realtime_session_task.done():
                try:
                    await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                    logger.info("Realtime session reconnected by existing task.")
                except asyncio.TimeoutError:
                    logger.warning("Existing realtime session task did not reconnect in time.")
                    await self._await_or_cancel_session_task()
                if self.connection is not None:
                    return

            self._realtime_session_task = asyncio.create_task(
                self._run_realtime_session(),
                name="openai-realtime-restart",
            )
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _safe_response_create(self, **kwargs: Any) -> None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        await self._pending_responses.put(kwargs)

    async def _response_sender_loop(self) -> None:
        """Dedicated worker that sends ``response.create()`` calls serially.

        This logic was designed to comply with the response.create() docstring specification for event ordering:
        https://github.com/openai/openai-python/blob/3e0c05b84a2056870abf3bd6a5e7849020209cc3/src/openai/resources/realtime/realtime.py#L649C1-L651C30

        For each queued request the worker:
        1. Waits until no response is active (_response_done_event).
        2. Sends response.create().
        3. Waits for the response cycle to complete (response.done).
        4. If the server rejected with active_response, retries from step 1.
        """
        while self.connection:
            try:
                kwargs = await self._pending_responses.get()
            except asyncio.CancelledError:
                return

            sent = False
            max_retries = 5
            attempts = 0
            while not sent and self.connection and attempts < max_retries:
                try:
                    await asyncio.wait_for(self._response_done_event.wait(), timeout=_RESPONSE_DONE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for previous response to finish; forcing ahead")
                    self._response_done_event.set()

                if not self.connection:
                    break

                self._last_response_rejected = False
                try:
                    await self.connection.response.create(**kwargs)
                except Exception as e:
                    logger.debug("_response_sender_loop: send failed: %s", e)
                    self._response_done_event.set()
                    break

                try:
                    await asyncio.wait_for(self._response_done_event.wait(), timeout=_RESPONSE_DONE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.done; assuming response completed")
                    self._response_done_event.set()
                    break

                # Check if we were rejected
                if self._last_response_rejected:
                    attempts += 1
                    if attempts >= max_retries:
                        logger.debug("response.create rejected %d times; giving up", attempts)
                        break
                    logger.debug("response.create was rejected; retrying (%d/%d)", attempts, max_retries)
                    continue

                sent = True

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        if bg_tool.error is not None:
            logger.error("Tool '%s' (id=%s) failed with error: %s", bg_tool.tool_name, bg_tool.id, bg_tool.error)
            tool_result = {"error": bg_tool.error}
        elif bg_tool.result is not None:
            tool_result = bg_tool.result
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                bg_tool.tool_name, bg_tool.id,
            )
            logger.debug("Tool '%s' full result: %s", bg_tool.tool_name, tool_result)
        else:
            logger.warning("Tool '%s' (id=%s) returned no result and no error", bg_tool.tool_name, bg_tool.id)
            tool_result = {"error": "No result returned from tool execution"}

        # Connection may have closed while tool was running
        if not self.connection:
            logger.warning("Connection closed during tool '%s' (id=%s) execution; cannot send result back", bg_tool.tool_name, bg_tool.id)
            return

        try:
            # Send the tool result back
            if isinstance(bg_tool.id, str):
                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": bg_tool.id,
                        "output": json.dumps(tool_result),
                    },
                )

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result),
                        # Gradio UI metadata.status accept only "pending" and "done". Do not accept bg.tool.status values.
                        "metadata": {
                            "title": f"🛠️ Used tool {bg_tool.tool_name}",
                            "status": "done",
                        },
                    },
                ),
            )

            if bg_tool.tool_name == "camera" and "b64_im" in tool_result:
                # use raw base64, don't json.dumps (which adds quotes)
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_im}",
                            },
                        ],
                    },
                )
                logger.info("Added camera image to conversation")

                if self.deps.camera_worker is not None:
                    np_img = self.deps.camera_worker.get_latest_frame()
                    if np_img is not None:
                        # Camera frames are BGR from OpenCV; convert so Gradio displays correct colors.
                        rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = None
                    img = gr.Image(value=rgb_frame)

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": img,
                            },
                        ),
                    )

            await self._safe_response_create(
                response={
                    "instructions": (
                        "Use the tool result just returned and answer concisely in speech. "
                        "Speak only English or Greek."
                    ),
                },
            )

            # Re-synchronize the head wobble after a tool call that may have taken some time
            if self.deps.head_wobbler is not None:
                self.deps.head_wobbler.reset()

        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._response_done_event.set()


    def _transcript_requests_sleep(self, transcript: str | None) -> bool:
        """Return True when a transcript contains a configured sleep phrase."""
        return bool(
            transcript
            and is_sleep_phrase(transcript, self.wake_config.sleep_phrases),
        )

    async def _preempt_sleep_response(self) -> None:
        """Cancel an in-flight or imminent server VAD response for a sleep command."""
        self._sleep_pending = True
        if not self.connection:
            return
        try:
            await self.connection.response.cancel()
        except Exception:
            pass
        try:
            await self.connection.input_audio_buffer.clear()
        except Exception:
            pass

    async def _maybe_sleep_from_transcript(self, transcript: str | None) -> bool:
        """Return True after transitioning to sleep on a sleep phrase."""
        if self.wake_gating_enabled and not self.wake_session.awake:
            return False
        if not self._transcript_requests_sleep(transcript):
            return False
        await self._preempt_sleep_response()
        await self._transition_to_sleep("sleep phrase")
        return True

    def _record_user_transcript(self, transcript: str | None) -> None:
        """Track the latest user transcript for early sleep detection."""
        if transcript:
            self._latest_user_transcript = transcript
            if (
                self.wake_session.awake
                and self._transcript_requests_sleep(transcript)
            ):
                self._sleep_pending = True

    async def _handle_completed_user_transcript(self, transcript: str) -> None:
        """Record a completed user transcript and watch for the sleep phrase.

        Responses are created automatically by server VAD; creating another one
        here would answer the same question twice. Local Whisper sleep detection
        is primary; this path is a fallback when the daemon misses the phrase.
        """
        if self.wake_gating_enabled and not self.wake_session.awake:
            logger.debug("Ignoring transcript while asleep: %r", transcript)
            return

        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
        if await self._maybe_sleep_from_transcript(transcript):
            return
        self.wake_session.touch()

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            try:
                await conn.session.update(
                    session={
                        "type": "realtime",
                        "instructions": get_realtime_session_instructions(),
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.input_sample_rate,
                                },
                                "transcription": {"model": "gpt-4o-transcribe"},
                                "turn_detection": {
                                    "type": "server_vad",
                                    "interrupt_response": True,
                                },
                            },
                            "output": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.output_sample_rate,
                                },
                                "voice": get_session_voice(),
                            },
                        },
                        "tools": get_tool_specs(),  # type: ignore[typeddict-item]
                        "tool_choice": "auto",
                    },
                )
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    get_session_voice(),
                )
                # If we reached here, the session update succeeded which implies the API key worked.
                # Persist the key to a newly created .env (copied from .env.example) if needed.
                self._persist_api_key_if_needed()
            except Exception:
                logger.exception("Realtime session.update failed; retrying session")
                raise RealtimeSessionError("session.update failed")

            logger.info("Realtime session updated successfully")

            # Manage event received from the openai server
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass


            response_sender_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start the response sender worker
                response_sender_task = asyncio.create_task(
                    self._response_sender_loop(), name="response-sender"
                )

                async for event in self.connection:
                    logger.debug(f"OpenAI event: {event.type}")
                    if event.type == "input_audio_buffer.speech_started":
                        if self._should_ignore_server_vad():
                            logger.debug("Ignoring speech_started during assistant output")
                        else:
                            self._latest_user_transcript = ""
                            self._sleep_pending = False
                            if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                                self._clear_queue()
                            if self.deps.head_wobbler is not None:
                                self.deps.head_wobbler.reset()
                            self.deps.movement_manager.set_listening(True)
                            self.wake_session.touch()
                            logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        if not self._should_ignore_server_vad():
                            self.deps.movement_manager.set_listening(False)
                            if await self._maybe_sleep_from_transcript(self._latest_user_transcript):
                                continue
                            logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type in (
                        "response.audio.done",  # GA
                        "response.output_audio.done",  # GA alias
                        "response.audio.completed",  # legacy (for safety)
                        "response.completed",  # text-only completion
                    ):
                        logger.debug("response completed")

                    if event.type == "response.created":
                        if (
                            not self._session_accepts_responses()
                            or await self._maybe_sleep_from_transcript(self._latest_user_transcript)
                        ):
                            if self.connection:
                                try:
                                    await self.connection.response.cancel()
                                except Exception:
                                    pass
                            continue
                        self._response_done_event.clear()
                        logger.debug("Response created (active)")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        self._response_done_event.set()
                        logger.debug("Response done")

                        response = getattr(event, "response", None)
                        usage = getattr(response, "usage", None) if response else None
                        if usage:
                            cost = _compute_response_cost(usage)
                            self.cumulative_cost += cost
                            logger.debug("Cost: $%.4f | Cumulative: $%.4f", cost, self.cumulative_cost)
                        else:
                            logger.warning("No usage data available for cost tracking")

                    # Handle partial transcription (user speaking in real-time)
                    if event.type == "conversation.item.input_audio_transcription.partial":
                        logger.debug(f"User partial transcript: {event.transcript}")
                        transcript = getattr(event, "transcript", None)
                        self._record_user_transcript(transcript)
                        if self._sleep_pending and self.wake_session.awake:
                            await self._preempt_sleep_response()

                        if await self._maybe_sleep_from_transcript(transcript):
                            continue

                        await self._partial_debouncer.schedule(transcript)

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        logger.debug(f"User transcript: {event.transcript}")
                        self._record_user_transcript(getattr(event, "transcript", None))

                        await self._partial_debouncer.cancel()
                        await self._handle_completed_user_transcript(event.transcript)

                    # Handle assistant transcription
                    if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                        if not self._session_accepts_responses():
                            continue
                        logger.debug(f"Assistant transcript: {event.transcript}")
                        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                    # Handle audio delta
                    if event.type in ("response.audio.delta", "response.output_audio.delta"):
                        if not self._session_accepts_responses():
                            continue
                        self._last_assistant_audio_at = time.monotonic()
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed(event.delta)
                        self.wake_session.touch()
                        delta_audio = np.frombuffer(base64.b64decode(event.delta), dtype=np.int16)
                        await self.output_queue.put(
                            (
                                self.output_sample_rate,
                                delta_audio.reshape(1, -1),
                            ),
                        )

                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, args=%s",
                            tool_name, call_id, args_json_str,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name, type(tool_name).__name__,
                                args_json_str, type(args_json_str).__name__,
                                call_id,
                            )
                            continue

                        bg_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=args_json_str,
                                deps=self.deps,
                            ),
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. The tool is now running. Tool ID: {bg_tool.tool_id}",
                                },
                            ),
                        )

                        # No extra response here: the model's own turn already announces
                        # the tool; a second "notify" response doubled the speech.
                        logger.info("Started background tool: %s (id=%s, call_id=%s)", tool_name, bg_tool.tool_id, call_id)

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", str(err) if err else "unknown error")
                        code = getattr(err, "code", "")

                        if code == "conversation_already_has_active_response":
                            # response.create was rejected.  The sender worker
                            # is waiting on _response_done_event; when the active
                            # response finishes it will wake up and see this flag.
                            self._last_response_rejected = True
                            logger.debug("response.create rejected; worker will retry after active response finishes")
                        else:
                            logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                        # Only show user-facing errors, not internal state errors
                        if code not in ("input_audio_buffer_commit_empty",):
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                            )
            finally:
                # Stop the response sender worker.
                if response_sender_task is not None:
                    response_sender_task.cancel()
                    try:
                        await response_sender_task
                    except asyncio.CancelledError:
                        pass

                # Stop background tool manager tasks (listener + cleanup) in all paths.
                await self.tool_manager.shutdown()

    def _to_wake_rate(self, audio_frame: NDArray[Any], input_sample_rate: int) -> NDArray[np.int16]:
        """Convert a mono frame to the 16 kHz int16 format the wake backends expect."""
        mono = audio_frame.reshape(-1)
        if input_sample_rate != WAKE_SAMPLE_RATE:
            mono = resample(mono, int(len(mono) * WAKE_SAMPLE_RATE / input_sample_rate))
        if np.issubdtype(mono.dtype, np.integer):
            return mono.astype(np.int16, copy=False)
        converted: NDArray[np.int16] = audio_to_int16(np.asarray(mono, dtype=np.float32))
        return converted

    def _feed_wake_detector(self, audio_frame: NDArray[Any], input_sample_rate: int) -> None:
        """Forward mic audio to the local wake detector, restarting it if needed."""
        if self._wake_detector is None:
            return
        if not self._wake_detector.is_running():
            logger.warning("Wake detector thread not running; restarting")
            self._wake_detector.start()
        self._wake_detector.feed(self._to_wake_rate(audio_frame, input_sample_rate))

    def _make_chime(self, *, ascending: bool) -> NDArray[np.int16]:
        """Generate a short two-tone chime marking a wake/sleep transition."""
        sample_rate = self.output_sample_rate
        freqs = (660.0, 880.0) if ascending else (880.0, 660.0)
        fade = max(1, int(sample_rate * 0.01))
        tones = []
        for freq in freqs:
            t = np.arange(int(sample_rate * 0.12)) / sample_rate
            tone = 0.25 * np.sin(2 * np.pi * freq * t)
            tone[:fade] *= np.linspace(0.0, 1.0, fade)
            tone[-fade:] *= np.linspace(1.0, 0.0, fade)
            tones.append(tone)
        return (np.concatenate(tones) * 32767).astype(np.int16)

    async def _play_chime(self, *, ascending: bool) -> None:
        try:
            chime = self._make_chime(ascending=ascending)
            await self.output_queue.put((self.output_sample_rate, chime.reshape(1, -1)))
        except Exception:
            logger.debug("Chime skipped", exc_info=True)

    def _queue_antenna_cue(self, *, awake: bool) -> None:
        """Raise antennas while streaming; relax them and nod head down when asleep."""
        try:
            from reachy_mini.utils import create_head_pose
            from reachy_mini.utils.interpolation import compose_world_offset
            from bobe.dance_emotion_moves import GotoQueueMove

            robot = self.deps.reachy_mini
            head_pose = robot.get_current_head_pose()
            head_joints, antennas = robot.get_current_joint_positions()
            # Mirrored joints: (-, +) perks both antennas outward; (+, -) crosses them.
            target = (-0.5, 0.5) if awake else (0.0, 0.0)
            z_offset_mm = _SLEEP_HEAD_Z_OFFSET_MM if awake else -_SLEEP_HEAD_Z_OFFSET_MM
            head_offset = create_head_pose(0, 0, z_offset_mm, 0, 0, 0, degrees=True, mm=True)
            target_head_pose = compose_world_offset(head_pose, head_offset, reorthonormalize=True)
            move = GotoQueueMove(
                target_head_pose=target_head_pose,
                start_head_pose=head_pose,
                target_antennas=target,
                start_antennas=(float(antennas[0]), float(antennas[1])),
                target_body_yaw=float(head_joints[0]),
                start_body_yaw=float(head_joints[0]),
                duration=0.6,
            )
            self.deps.movement_manager.queue_move(move)
        except Exception:
            logger.debug("Antenna cue skipped", exc_info=True)

    async def _transition_to_awake(self) -> bool:
        """Open the streaming window after a local wake-word detection."""
        if not await self._ensure_openai_connection():
            logger.error("Wake ignored: OpenAI Realtime unavailable")
            return False

        # Clear any stale in-flight response state so server VAD is not suppressed.
        self._response_done_event.set()
        self._last_assistant_audio_at = 0.0

        self._sleep_pending = False
        self._listen_for_sleep_detector()
        self.wake_session.wake()
        logger.info("Wake word heard: streaming audio to OpenAI until timeout or sleep phrase")
        await self._play_chime(ascending=True)
        self._queue_antenna_cue(awake=True)

        # Flush the buffered audio that arrived with/after the wake phrase so a
        # one-breath request like "hey jarvis, what's the weather" is not lost.
        tail = self._wake_buffer.drain_tail(DEFAULT_FLUSH_SECONDS)
        if tail.size and self.connection:
            try:
                await self.connection.input_audio_buffer.append(
                    audio=base64.b64encode(tail.tobytes()).decode("utf-8")
                )
            except Exception as e:
                logger.warning("Could not flush pre-wake audio: %s", e)
        return True

    async def _transition_to_sleep(self, reason: str) -> None:
        """Close the streaming window; audio stays on the robot again."""
        self.wake_session.sleep()
        self._sleep_pending = False
        logger.info("Going to sleep (%s): audio stays local until the wake word", reason)

        # Stop any in-flight answer (e.g. the auto-response to "go to sleep").
        if self.connection:
            try:
                await self.connection.response.cancel()
            except Exception as e:
                logger.debug("No active response to cancel on sleep: %s", e)
            try:
                await self.connection.input_audio_buffer.clear()
            except Exception as e:
                logger.debug("Could not clear input buffer on sleep: %s", e)
        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
            self._clear_queue()

        await self._play_chime(ascending=False)
        self._queue_antenna_cue(awake=False)
        self._listen_for_wake_detector()

    def _listen_for_sleep_detector(self) -> None:
        detector = self._wake_detector
        if detector is not None and hasattr(detector, "listen_for_sleep"):
            detector.listen_for_sleep()

    def _listen_for_wake_detector(self) -> None:
        detector = self._wake_detector
        if detector is None:
            return
        if hasattr(detector, "listen_for_wake"):
            detector.listen_for_wake()
        if not detector.is_running():
            logger.warning("Wake detector thread not running; restarting")
            detector.start()

    def _pause_wake_detector(self) -> None:
        self._listen_for_sleep_detector()

    def _resume_wake_detector(self) -> None:
        self._listen_for_wake_detector()

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive a mic frame; keep it local while asleep, otherwise send upstream.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for OpenAI's API. Resamples if the input sample rate differs
        from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        upstream_frame = audio_frame
        if self.input_sample_rate != input_sample_rate:
            upstream_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        upstream_frame = audio_to_int16(upstream_frame)

        if self.wake_test_mode:
            if self.wake_session.consume_wake_request():
                self.wake_test_detections += 1
            self._feed_wake_detector(audio_frame, input_sample_rate)
            return

        if self.wake_gating_enabled:
            if self.wake_session.consume_wake_request():
                if not await self._transition_to_awake():
                    self.wake_session.request_wake()
            elif self.wake_session.consume_sleep_request():
                await self._transition_to_sleep("local sleep phrase")
            elif self.wake_session.expired():
                await self._transition_to_sleep("inactivity timeout")

            if not self.wake_session.awake:
                self._wake_buffer.append(upstream_frame.reshape(-1))
                self._feed_wake_detector(audio_frame, input_sample_rate)
                return

            self._feed_wake_detector(audio_frame, input_sample_rate)

        if not self.connection:
            now = time.monotonic()
            if now - self._last_reconnect_attempt_at >= 2.0:
                self._last_reconnect_attempt_at = now
                logger.warning("Awake without OpenAI connection; attempting reconnect")
                await self._ensure_openai_connection(timeout=2.0)
            if not self.connection:
                return

        # Send to OpenAI (guard against races during reconnect)
        try:
            audio_message = base64.b64encode(upstream_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.warning("Failed to send audio frame; will retry reconnect (%s)", e)
            conn = self.connection
            self.connection = None
            try:
                self._connected_event.clear()
            except Exception:
                pass
            if conn is not None:
                try:
                    await conn.close()
                except Exception:
                    pass
            now = time.monotonic()
            if now - self._last_reconnect_attempt_at >= 2.0:
                self._last_reconnect_attempt_at = now
                await self._restart_session()
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Stop the local wake-word detector thread
        if self._wake_detector is not None:
            self._wake_detector.stop()

        session_task = self._realtime_session_task
        if session_task is not None and not session_task.done():
            session_task.cancel()
            try:
                await session_task
            except (asyncio.CancelledError, Exception):
                pass
        self._realtime_session_task = None

        # Unblock the response sender worker so it can exit
        self._response_done_event.set()

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()

        await self._partial_debouncer.cancel()

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        # Conservative fallback list with default first
        fallback = [
            "cedar",
            "alloy",
            "aria",
            "ballad",
            "verse",
            "sage",
            "coral",
        ]
        try:
            # Best effort discovery; safe-guarded for unexpected shapes
            model = await self.client.models.retrieve(config.MODEL_NAME)
            # Try common serialization paths
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None
            # Scan for voice candidates
            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if "voice" in kl and isinstance(v, (list, tuple)):
                                for item in v:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                                        candidates.add(item["name"])
                            else:
                                _collect(v)
                    elif isinstance(obj, (list, tuple)):
                        for it in obj:
                            _collect(it)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)
            # Ensure default present and stable order
            voices = sorted(candidates) if candidates else fallback
            if "cedar" not in voices:
                voices = ["cedar", *[v for v in voices if v != "cedar"]]
            return voices
        except Exception:
            return fallback

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key into `.env` inside `instance_path/` when appropriate.

        - Only runs in Gradio mode when key came from the textbox and is non-empty.
        - Only saves if `self.instance_path` is not None.
        - Writes `.env` to `instance_path/.env` (does not overwrite if it already exists).
        - If `instance_path/.env.example` exists, copies its contents while overriding OPENAI_API_KEY.
        """
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided via textbox; skipping persistence.")
                return
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            # Update the current process environment for downstream consumers
            try:
                import os

                os.environ["OPENAI_API_KEY"] = key
            except Exception:  # best-effort
                pass

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                # Respect existing user configuration
                logger.info(".env already exists at %s; not overwriting.", env_path)
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example at %s: %s", example_path, e)

            # Replace or append the OPENAI_API_KEY line
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    content_lines[i] = f"OPENAI_API_KEY={key}"
                    replaced = True
                    break
            if not replaced:
                content_lines.append(f"OPENAI_API_KEY={key}")

            # Ensure file ends with newline
            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created %s and stored OPENAI_API_KEY for future runs.", env_path)
        except Exception as e:
            # Never crash the app for QoL persistence; just log.
            logger.warning("Could not persist OPENAI_API_KEY to .env: %s", e)
