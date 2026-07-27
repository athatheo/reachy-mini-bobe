"""Bidirectional local audio stream for headless (no Gradio) mode.

If required API keys are not available via environment/.env, launch() waits
until non-technical users enter them through the settings page that
``bootstrap_settings_ui`` (called from the Reachy Mini Apps entrypoint in
``bobe.main``) mounts on the apps settings server before the voice handler
starts. Keys are persisted to the app instance's private ``.env`` file when
available.
"""

import os
import time
import asyncio
import logging
import threading
from typing import List, Optional

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from bobe.config import config
from bobe.env_file import is_plausible_openai_key, is_plausible_anthropic_key
from bobe.openai_realtime import OpenaiRealtimeHandler


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        robot: ReachyMini,
    ):
        """Initialize the stream with an OpenAI realtime handler and pipelines."""
        self.handler = handler
        self._robot = robot
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        # Set by close(). launch() checks it at every stage (start, key-wait
        # poll, pre-media, and right after the loop is registered in runner)
        # and the record/play loops poll it each iteration, so a stop that
        # lands before the asyncio loop exists is never lost.
        self._close_requested = threading.Event()

    def _required_api_keys_configured(self) -> bool:
        """Return whether all explicit user-provided keys are configured."""
        return is_plausible_openai_key(str(config.OPENAI_API_KEY or "")) and is_plausible_anthropic_key(
            os.getenv("ANTHROPIC_API_KEY")
        )

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the OpenAI key is missing, wait for the user to provide it via the
        settings UI (mounted by ``bobe.main``) before starting streams.
        """
        if self._close_requested.is_set():
            logger.info("Close already requested; not starting LocalStream.")
            return

        # The instance .env was already loaded by main.run(); config.OPENAI_API_KEY
        # is kept in sync there and by the settings endpoints.

        # Never auto-download shared/demo keys. Wait for explicit user-provided keys.
        if not self._required_api_keys_configured():
            logger.warning(
                "Required API keys missing. Open the app settings page to enter OpenAI and Anthropic keys."
            )
            warned_at = time.monotonic()
            try:
                while not self._required_api_keys_configured():
                    if self._close_requested.is_set():
                        logger.info("Close requested while waiting for API keys.")
                        return
                    if time.monotonic() - warned_at >= 30.0:
                        logger.warning(
                            "Still waiting for OpenAI and Anthropic API keys in settings (http://<robot>:7860/)."
                        )
                        warned_at = time.monotonic()
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for API keys.")
                return

        if self._close_requested.is_set():
            logger.info("Close requested before media startup; aborting launch.")
            return

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            self._asyncio_loop = asyncio.get_running_loop()
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            # A close() that landed before the loop was registered could not
            # be marshalled onto it; replay its cancellation path here, on the
            # loop thread, so the stop request is never lost.
            if self._close_requested.is_set():
                for task in self._tasks:
                    task.cancel()
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        try:
            asyncio.run(runner())
        finally:
            # close() may have landed in the pre-media window, where its media
            # stops were no-ops on pipelines we had not started yet. launch()
            # owns the pipelines it started, so stop them on every exit path
            # (idempotent; a second stop from close() is harmless).
            try:
                self._robot.media.stop_recording()
            except Exception as e:
                logger.debug(f"Error stopping recording on launch exit: {e}")
            try:
                self._robot.media.stop_playing()
            except Exception as e:
                logger.debug(f"Error stopping playback on launch exit: {e}")

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the close flag to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)

        Safe to call from any thread, at any point in the lifecycle (before
        launch(), while waiting for API keys, during media startup, or with
        the asyncio loop running), and safe to call more than once.
        """
        logger.info("Stopping LocalStream...")

        # Record the request FIRST: launch() checks this flag at every stage,
        # the record/play loops poll it each iteration, and runner() re-checks
        # it right after registering the loop, so a close that lands before
        # the loop exists still terminates launch().
        self._close_requested.set()

        # Stop media pipelines FIRST before cancelling async tasks
        # This ensures clean shutdown before PortAudio cleanup
        try:
            self._robot.media.stop_recording()
        except Exception as e:
            logger.debug(f"Error stopping recording (may already be stopped): {e}")

        try:
            self._robot.media.stop_playing()
        except Exception as e:
            logger.debug(f"Error stopping playback (may already be stopped): {e}")

        # Now cancel the running tasks. close() is invoked from a foreign
        # thread (e.g. the dashboard stop poller in main.py), and Task.cancel()
        # is not thread-safe: it must be marshalled onto the loop thread. If
        # the asyncio runner never started (or already finished), there is no
        # loop thread to race with; _close_requested (set above) makes
        # launch() abort or runner() cancel its tasks.
        loop = self._asyncio_loop
        if loop is not None and not loop.is_closed():
            try:
                for task in self._tasks:
                    if not task.done():
                        loop.call_soon_threadsafe(task.cancel)
            except RuntimeError as e:
                # The loop finished between the is_closed() check and the call.
                logger.debug(f"Event loop already closed while stopping: {e}")

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            # Directly flush gstreamer audio pipe
            self._robot.media.audio.clear_player()
        elif self._robot.media.backend == MediaBackend.DEFAULT or self._robot.media.backend == MediaBackend.DEFAULT_NO_VIDEO:
            self._robot.media.audio.clear_output_buffer()
        # Drain the handler's queue IN PLACE: replacing the object would leave
        # producers that captured the old reference (e.g. the partial-transcript
        # debouncer) feeding an orphaned queue nobody reads anymore.
        queue = self.handler.output_queue
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        input_sample_rate = self._robot.media.get_input_audio_samplerate()
        logger.debug(f"Audio recording started at {input_sample_rate} Hz")

        while not self._close_requested.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._close_requested.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_data = handler_output
                output_sample_rate = self._robot.media.get_output_audio_samplerate()

                # Reshape if needed
                if audio_data.ndim == 2:
                    # Scipy channels last convention
                    if audio_data.shape[1] > audio_data.shape[0]:
                        audio_data = audio_data.T
                    # Multiple channels -> Mono channel
                    if audio_data.shape[1] > 1:
                        audio_data = audio_data[:, 0]

                # Cast if needed
                audio_frame = audio_to_float32(audio_data)

                # Resample if needed
                if input_sample_rate != output_sample_rate:
                    audio_frame = resample(
                        audio_frame,
                        int(len(audio_frame) * output_sample_rate / input_sample_rate),
                    )

                self._robot.media.push_audio_sample(audio_frame)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
