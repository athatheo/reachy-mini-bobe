"""Hermes-backed voice handler: wake gating, LAN audio, and speech playback.

The robot's entire conversation loop runs against the Mac wake daemon:

- **Asleep**: mic PCM streams to the daemon, which listens for the wake phrase.
  Nothing ever leaves the LAN.
- **Awake**: the daemon's converse mode transcribes utterances and hands them
  to the Hermes agent (via the bobe platform plugin); Hermes' TTS replies come
  back as speech clips over the wake WebSocket and play through the speaker.
- **Sleep**: a sleep phrase (daemon-matched) or the inactivity timeout closes
  the window; audio stays on the robot again.

There is deliberately no cloud-model client in this module: OpenAI Realtime
was removed in favor of the Hermes pipeline (see README "Hermes voice
backend"). The fastrtc ``AsyncStreamHandler`` contract (receive/emit) is kept
so both the Gradio stream and the headless ``LocalStream`` drive it unchanged.
"""

import json
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional

import numpy as np
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from bobe.cues import play_chime, queue_antenna_cue
from bobe.wake_word import (
    WakeGate,
    WakeConfig,
    WakeSession,
    AudioRingBuffer,
    create_wake_detector,
)
from bobe.tools.core_tools import ToolDependencies, dispatch_tool_call


logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE: Final[Literal[24000]] = 24000


class BobeVoiceHandler(AsyncStreamHandler):
    """fastrtc stream handler wiring the robot's audio to the Hermes pipeline."""

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=AUDIO_SAMPLE_RATE,
            input_sample_rate=AUDIO_SAMPLE_RATE,
        )
        self.output_sample_rate: Literal[24000] = AUDIO_SAMPLE_RATE
        self.input_sample_rate: Literal[24000] = AUDIO_SAMPLE_RATE

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        # Never reassign this queue: console.clear_audio_queue drains it in place.
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self._shutdown_requested = False
        self._stopped: asyncio.Event = asyncio.Event()

        # Hermes announcements and speech clips run as background tasks so the
        # mic loop never blocks; content received while asleep parks here until
        # the wake transition plays it.
        self._announce_tasks: set[asyncio.Task[None]] = set()
        self._pending_announcements: list[str] = []
        self._pending_speech: list[Tuple[NDArray[np.int16], int]] = []
        self._pending_emotes: list[str] = []
        # A spoken "go to sleep" flips this off so announcements queue without
        # waking the robot; any real wake flips it back on.
        self._announce_wake_allowed: bool = True
        self._wake_transition_task: asyncio.Task[None] | None = None
        # Set by main.run() when a camera is present; surfaced in /status.
        self.presence_watcher: Any = None

        # Local wake-word gating. The factory is looked up in this module's
        # namespace at call time so tests can stub
        # bobe.voice_handler.create_wake_detector.
        self.wake_gate = WakeGate(
            input_sample_rate=self.input_sample_rate,
            detector_factory=create_wake_detector,
        )

    # --- Wake-gate aliases: settings_server and tests reach these directly ---

    @property
    def wake_config(self) -> WakeConfig:
        """Alias for the wake gate's config."""
        return self.wake_gate.config

    @wake_config.setter
    def wake_config(self, value: WakeConfig) -> None:
        """Replace the wake gate's config."""
        self.wake_gate.config = value

    @property
    def wake_session(self) -> WakeSession:
        """Alias for the wake gate's session."""
        return self.wake_gate.session

    @wake_session.setter
    def wake_session(self, value: WakeSession) -> None:
        """Replace the wake gate's session."""
        self.wake_gate.session = value

    @property
    def wake_gating_enabled(self) -> bool:
        """Alias for whether the wake gate is enabled."""
        return self.wake_gate.enabled

    @wake_gating_enabled.setter
    def wake_gating_enabled(self, value: bool) -> None:
        """Override whether the wake gate is enabled."""
        self.wake_gate.enabled = value

    @property
    def wake_error(self) -> str | None:
        """Alias for the wake gate's user-visible error."""
        return self.wake_gate.error

    @wake_error.setter
    def wake_error(self, value: str | None) -> None:
        """Replace the wake gate's user-visible error."""
        self.wake_gate.error = value

    @property
    def _wake_buffer(self) -> AudioRingBuffer:
        return self.wake_gate.buffer

    @_wake_buffer.setter
    def _wake_buffer(self, value: AudioRingBuffer) -> None:
        self.wake_gate.buffer = value

    @property
    def _wake_detector(self) -> Any:
        return self.wake_gate.detector

    @_wake_detector.setter
    def _wake_detector(self, value: Any) -> None:
        self.wake_gate.detector = value

    def copy(self) -> "BobeVoiceHandler":
        """Create a copy of the handler."""
        return BobeVoiceHandler(self.deps, self.gradio_mode, self.instance_path)

    async def start_up(self) -> None:
        """Start the wake detector and park until shutdown.

        All conversation intelligence lives on the Mac (daemon + Hermes); the
        handler's async work happens in receive()/emit() and background tasks.
        """
        logger.info("BoBe voice handler active: conversations run via the Mac wake daemon and Hermes")
        self._stopped.clear()
        self.wake_gate.start()
        await self._stopped.wait()

    # ------------------------------------------------------------------
    # Wake / sleep transitions
    # ------------------------------------------------------------------

    def _wake_transition_active(self) -> bool:
        """Return whether an awake transition task is currently running."""
        task = self._wake_transition_task
        return task is not None and not task.done()

    def _start_wake_transition(self) -> None:
        """Kick off the awake transition without stalling the mic loop."""
        if self._shutdown_requested or self._wake_transition_active():
            return
        self._wake_transition_task = asyncio.create_task(
            self._run_wake_transition(),
            name="wake-transition",
        )

    async def _run_wake_transition(self) -> None:
        try:
            await self._transition_to_awake()
        except asyncio.CancelledError:
            raise
        except Exception:
            # There is no upstream connection to fail on; anything here is a
            # local bug. Re-queue the wake so a later frame retries once.
            logger.exception("Wake transition failed; re-queuing wake")
            self.wake_gate.requeue_wake()

    async def _transition_to_awake(self) -> bool:
        """Open the conversation window after a wake-word detection."""
        self.wake_gate.enter_awake(converse=True)
        logger.info("Wake word heard: conversing via Hermes until timeout or sleep phrase")
        await self._play_chime(ascending=True)
        self._queue_antenna_cue(awake=True)

        # A real wake restores announce-wake behavior after a spoken sleep.
        self._announce_wake_allowed = True

        # Deliver announcements and speech clips that arrived while asleep.
        pending = self._pending_announcements
        self._pending_announcements = []
        for text in pending:
            await self._speak_announcement(text)
        pending_emotes = self._pending_emotes
        self._pending_emotes = []
        for emotion in pending_emotes:
            await self._handle_emote(emotion)
        pending_speech = self._pending_speech
        self._pending_speech = []
        for pcm, rate in pending_speech:
            await self._play_speech_clip(pcm, rate)
        return True

    async def _transition_to_sleep(self, reason: str) -> None:
        """Close the conversation window; audio stays on the robot again."""
        user_requested = reason in ("local sleep phrase", "sleep phrase")
        self.wake_gate.sleep()
        logger.info("Going to sleep (%s): audio stays local until the wake word", reason)

        # A spoken sleep command wins permanently: announcements queue but must
        # not wake the robot until the next real wake. Timeout sleeps keep the
        # ambient announce-wake behavior.
        if user_requested:
            self._announce_wake_allowed = False

        # Release any listening freeze and drop queued audio: sleep means
        # silence, including the tail of a playing speech clip.
        self.deps.movement_manager.set_listening(False)
        if self._clear_queue is not None:
            self._clear_queue()

        await self._play_chime(ascending=False)
        self._queue_antenna_cue(awake=False)
        self.wake_gate.listen_for_wake()

    # ------------------------------------------------------------------
    # Announcements (text) and speech clips (TTS audio) from the daemon
    # ------------------------------------------------------------------

    def _start_announcement(self, text: str) -> None:
        """Handle a Hermes announcement without stalling the mic loop."""
        if self._shutdown_requested:
            return
        task = asyncio.create_task(self._handle_announcement(text), name="announcement")
        self._announce_tasks.add(task)
        task.add_done_callback(self._announce_tasks.discard)

    async def _handle_announcement(self, text: str) -> None:
        """Surface an announcement now, or park it and wake the robot first."""
        try:
            if self.wake_gate.enabled and not self.wake_session.awake:
                self._pending_announcements.append(text)
                if self._announce_wake_allowed:
                    self.wake_session.request_wake()
                else:
                    logger.info("Holding announcement until the next wake (user asked for sleep): %r", text)
                return
            await self._speak_announcement(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Announcement handling failed")

    async def _speak_announcement(self, text: str) -> None:
        """Surface an announcement to the UI/log.

        Plain-text announces have no voice in the Hermes pipeline (the plugin
        normally delivers speech clips instead); the text still reaches the
        chat UI and log so nothing is silently dropped.
        """
        if self.wake_gate.enabled and not self.wake_session.awake:
            # Sleep landed between scheduling and speaking; hold the text.
            self._pending_announcements.append(text)
            return
        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text}))
        self.wake_session.touch()

    def _start_speech_clip(self, pcm: NDArray[np.int16], rate: int) -> None:
        """Handle a daemon-relayed speech clip without stalling the mic loop."""
        if self._shutdown_requested:
            return
        task = asyncio.create_task(self._handle_speech_clip(pcm, rate), name="speech-clip")
        self._announce_tasks.add(task)
        task.add_done_callback(self._announce_tasks.discard)

    async def _handle_speech_clip(self, pcm: NDArray[np.int16], rate: int) -> None:
        """Play a speech clip now, or park it and wake the robot first."""
        try:
            if self.wake_gate.enabled and not self.wake_session.awake:
                self._pending_speech.append((pcm, rate))
                if self._announce_wake_allowed:
                    self.wake_session.request_wake()
                else:
                    logger.info("Holding speech clip until the next wake (user asked for sleep)")
                return
            await self._play_speech_clip(pcm, rate)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Speech clip handling failed")

    def _start_emote(self, emotion: str) -> None:
        """Play a Hermes-requested emotion move without stalling the mic loop."""
        if self._shutdown_requested:
            return
        task = asyncio.create_task(self._handle_emote(emotion), name="emote")
        self._announce_tasks.add(task)
        task.add_done_callback(self._announce_tasks.discard)

    async def _handle_emote(self, emotion: str) -> None:
        """Queue the emotion move; asleep emotes park until the next wake.

        Emotes normally accompany a spoken reply, so the parked emote plays
        right when the reply's speech clip wakes the robot.
        """
        try:
            if self.wake_gate.enabled and not self.wake_session.awake:
                self._pending_emotes.append(emotion)
                return
            result = await dispatch_tool_call("play_emotion", json.dumps({"emotion": emotion}), self.deps)
            if isinstance(result, dict) and result.get("error"):
                logger.warning("Emote %r failed: %s", emotion, result["error"])
            else:
                logger.info("Emote queued: %r", emotion)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Emote handling failed")

    async def _play_speech_clip(self, pcm: NDArray[np.int16], rate: int) -> None:
        """Queue a PCM clip for the speaker at (near) real-time pace.

        A ~1 s primed buffer plus real-time pacing keeps the daemon-side echo
        guard aligned with actual playback and feeds the head wobbler at the
        cadence it expects, instead of dumping a whole minute of audio into
        the output queue at once.
        """
        if self.wake_gate.enabled and not self.wake_session.awake:
            # Sleep landed between scheduling and playing; hold the clip.
            self._pending_speech.append((pcm, rate))
            return
        chunk_samples = max(1, rate // 2)
        primed_s = 1.0
        elapsed_s = 0.0
        for start in range(0, pcm.size, chunk_samples):
            if self.wake_gate.enabled and not self.wake_session.awake:
                # Sleep interrupts playback; the remainder is dropped, not
                # parked — sleep means silence.
                logger.info("Speech clip playback stopped by sleep")
                return
            chunk = pcm[start : start + chunk_samples]
            self.wake_session.touch()
            if self.deps.head_wobbler is not None and rate == AUDIO_SAMPLE_RATE:
                self.deps.head_wobbler.feed(base64.b64encode(chunk.tobytes()).decode("utf-8"))
            await self.output_queue.put((rate, chunk.reshape(1, -1)))
            chunk_s = chunk.size / rate
            if elapsed_s >= primed_s:
                await asyncio.sleep(chunk_s)
            elapsed_s += chunk_s

    # ------------------------------------------------------------------
    # fastrtc stream contract
    # ------------------------------------------------------------------

    async def _play_chime(self, *, ascending: bool) -> None:
        await play_chime(self.output_queue, self.output_sample_rate, ascending=ascending)

    def _queue_antenna_cue(self, *, awake: bool) -> None:
        """Queue the wake/sleep antenna-and-head posture cue (see bobe.cues)."""
        queue_antenna_cue(self.deps.movement_manager, awake=awake)

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive a mic frame and keep it on the LAN.

        Frames are normalized to mono int16 at the handler rate, then fed to
        the wake gate, whose detector streams them to the Mac daemon. Asleep
        or awake, audio never goes further than the daemon.
        """
        input_sample_rate, audio_frame = frame

        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        normalized = audio_frame
        if self.input_sample_rate != input_sample_rate:
            normalized = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))
        normalized = audio_to_int16(normalized)

        if not self.wake_gate.enabled:
            return

        event = self.wake_gate.poll()
        if event == "wake":
            # Never await the transition here: receive() is called serially
            # per mic frame by the record loop.
            self._start_wake_transition()
        elif event == "sleep":
            await self._transition_to_sleep("local sleep phrase")
        elif event == "expired":
            await self._transition_to_sleep("inactivity timeout")

        for announcement in self.wake_gate.drain_announcements():
            self._start_announcement(announcement)
        for speech_pcm, speech_rate in self.wake_gate.drain_speech():
            self._start_speech_clip(speech_pcm, speech_rate)
        for emotion in self.wake_gate.drain_emotes():
            self._start_emote(emotion)

        self.wake_gate.feed(audio_frame, input_sample_rate)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit an audio frame (or UI payload) to be played by the speaker."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler: stop the detector and cancel background work."""
        self._shutdown_requested = True
        self._stopped.set()

        self.wake_gate.stop()

        background_tasks = [self._wake_transition_task, *self._announce_tasks]
        self._wake_transition_task = None
        self._announce_tasks.clear()
        for background_task in background_tasks:
            if background_task is None or background_task.done():
                continue
            background_task.cancel()
            try:
                await background_task
            except (asyncio.CancelledError, Exception):
                pass

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
