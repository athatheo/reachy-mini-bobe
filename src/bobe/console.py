"""Bidirectional local audio stream with optional settings UI.

In headless mode, there is no Gradio UI. If required API keys are not
available via environment/.env, we expose a minimal settings page via the
Reachy Mini Apps settings server to let non-technical users enter them.

The settings UI is served from this package's ``static/`` folder and stores
``OPENAI_API_KEY`` for the realtime voice bridge plus ``ANTHROPIC_API_KEY`` for
Claude-backed answers. Keys are persisted to the app instance's private ``.env``
file when available.
"""

import os
import time
import asyncio
import logging
import threading
from typing import List, Optional

from fastapi import FastAPI
from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from bobe.config import config
from bobe.env_file import is_plausible_openai_key, is_plausible_anthropic_key
from bobe.instance import load_instance_env
from bobe.openai_realtime import OpenaiRealtimeHandler
from bobe.settings_server import get_settings_server, bootstrap_settings_ui


logger = logging.getLogger(__name__)


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
        app_stop_event: Optional[threading.Event] = None,
    ):
        """Initialize the stream with an OpenAI realtime handler and pipelines.

        - ``settings_app``: the Reachy Mini Apps FastAPI to attach settings endpoints.
        - ``instance_path``: directory where per-instance ``.env`` should be stored.
        """
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue
        self._settings_app: Optional[FastAPI] = settings_app
        self._instance_path: Optional[str] = instance_path
        self._app_stop_event = app_stop_event
        self._settings_initialized = False
        self._asyncio_loop: Optional[asyncio.AbstractEventLoop] = None
        # Set by close(). launch() checks it at every stage (start, key-wait
        # poll, pre-media, and right after the loop is registered in runner)
        # so a stop that lands before the asyncio loop exists is never lost.
        self._close_requested = threading.Event()

    # ---- Settings UI (only when API key is missing) ----
    def _required_api_keys_configured(self) -> bool:
        """Return whether all explicit user-provided keys are configured."""
        return is_plausible_openai_key(str(config.OPENAI_API_KEY or "")) and is_plausible_anthropic_key(
            os.getenv("ANTHROPIC_API_KEY")
        )

    def _init_settings_ui_if_needed(self) -> None:
        """Ensure settings routes are mounted on the Reachy settings app."""
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return
        if get_settings_server() is None:
            bootstrap_settings_ui(self._settings_app, self._instance_path, lambda: self.handler)
        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the OpenAI key is missing, expose a tiny settings UI via the
        Reachy Mini settings server to collect it before starting streams.
        """
        if self._close_requested.is_set():
            logger.info("Close already requested; not starting LocalStream.")
            return

        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs);
        # load_instance_env also syncs config.OPENAI_API_KEY.
        if self._instance_path:
            try:
                load_instance_env(self._instance_path)
            except Exception as exc:
                logger.warning("Could not load instance .env from %s: %s", self._instance_path, exc)

        # Always expose settings UI if a settings app is available.
        self._init_settings_ui_if_needed()

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
                    if self._app_stop_event is not None and self._app_stop_event.is_set():
                        logger.info("Stop requested while waiting for API keys.")
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
                self._stop_event.set()
                for task in self._tasks:
                    task.cancel()
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Stops audio recording and playback first
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)

        Safe to call from any thread, at any point in the lifecycle (before
        launch(), while waiting for API keys, during media startup, or with
        the asyncio loop running), and safe to call more than once.
        """
        logger.info("Stopping LocalStream...")

        # Record the request FIRST: launch() checks this flag at every stage,
        # and runner() re-checks it right after registering the loop, so a
        # close that lands before the loop exists still terminates launch().
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

        # Now signal async loops to stop and cancel the running tasks.
        # close() is invoked from a foreign thread (e.g. the dashboard stop
        # poller in main.py), and neither asyncio.Event.set() nor Task.cancel()
        # is thread-safe: both must be marshalled onto the loop thread.
        loop = self._asyncio_loop
        if loop is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(self._stop_event.set)
                for task in self._tasks:
                    if not task.done():
                        loop.call_soon_threadsafe(task.cancel)
            except RuntimeError as e:
                # The loop finished between the is_closed() check and the call.
                logger.debug(f"Event loop already closed while stopping: {e}")
        else:
            # The asyncio runner never started (or already finished); there is
            # no loop thread to race with. _close_requested (set above) makes
            # launch() abort or runner() cancel its tasks; setting the stop
            # event here additionally unblocks the record/play loops.
            self._stop_event.set()

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

        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                await self.handler.receive((input_sample_rate, audio_frame))
            await asyncio.sleep(0)  # avoid busy loop

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
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
