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
import sys
import time
import asyncio
import logging
from typing import List, Optional
from pathlib import Path

from fastrtc import AdditionalOutputs, audio_to_float32
from scipy.signal import resample

from reachy_mini import ReachyMini
from reachy_mini.media.media_manager import MediaBackend
from bobe.claude import DEFAULT_CLAUDE_MODEL
from bobe.config import LOCKED_PROFILE, config
from bobe.openai_realtime import OpenaiRealtimeHandler
from bobe.headless_personality_ui import mount_personality_routes


try:
    # FastAPI is provided by the Reachy Mini Apps runtime
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from fastapi.responses import FileResponse, JSONResponse
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover - only loaded when settings_app is used
    FastAPI = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    StaticFiles = object  # type: ignore
    BaseModel = object  # type: ignore


logger = logging.getLogger(__name__)


def _is_plausible_openai_key(value: str | None) -> bool:
    """Return whether a value looks like an OpenAI API key."""
    key = (value or "").strip()
    return key.startswith("sk-") and len(key) >= 20


def _is_plausible_anthropic_key(value: str | None) -> bool:
    """Return whether a value looks like an Anthropic API key."""
    key = (value or "").strip()
    return key.startswith("sk-ant-") and len(key) >= 20


class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(
        self,
        handler: OpenaiRealtimeHandler,
        robot: ReachyMini,
        *,
        settings_app: Optional[FastAPI] = None,
        instance_path: Optional[str] = None,
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
        self._settings_initialized = False
        self._asyncio_loop = None

    # ---- Settings UI (only when API key is missing) ----
    def _read_env_lines(self, env_path: Path) -> list[str]:
        """Load env file contents or a template as a list of lines."""
        inst = env_path.parent
        try:
            if env_path.exists():
                try:
                    return env_path.read_text(encoding="utf-8").splitlines()
                except Exception:
                    return []
            template_text = None
            ex = inst / ".env.example"
            if ex.exists():
                try:
                    template_text = ex.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                try:
                    cwd_example = Path.cwd() / ".env.example"
                    if cwd_example.exists():
                        template_text = cwd_example.read_text(encoding="utf-8")
                except Exception:
                    template_text = None
            if template_text is None:
                packaged = Path(__file__).parent / ".env.example"
                if packaged.exists():
                    try:
                        template_text = packaged.read_text(encoding="utf-8")
                    except Exception:
                        template_text = None
            return template_text.splitlines() if template_text else []
        except Exception:
            return []

    def _required_api_keys_configured(self) -> bool:
        """Return whether all explicit user-provided keys are configured."""
        return _is_plausible_openai_key(str(config.OPENAI_API_KEY or "")) and _is_plausible_anthropic_key(
            os.getenv("ANTHROPIC_API_KEY")
        )

    def _persist_api_settings(
        self,
        *,
        openai_api_key: str,
        anthropic_api_key: str,
        claude_model: str,
    ) -> None:
        """Persist explicit API settings to environment and instance ``.env``."""
        values = {
            "OPENAI_API_KEY": openai_api_key.strip(),
            "ANTHROPIC_API_KEY": anthropic_api_key.strip(),
            "CLAUDE_MODEL": (claude_model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL,
        }
        if not values["OPENAI_API_KEY"] or not values["ANTHROPIC_API_KEY"]:
            return

        os.environ.update(values)
        try:
            config.OPENAI_API_KEY = values["OPENAI_API_KEY"]
        except Exception:
            pass

        if not self._instance_path:
            return

        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            for key, value in values.items():
                replacement = f"{key}={value}"
                for index, line in enumerate(lines):
                    if line.strip().startswith(f"{key}="):
                        lines[index] = replacement
                        break
                else:
                    lines.append(replacement)

            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Persisted explicit API settings to %s", env_path)

            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist explicit API settings: %s", e)

    def _persist_personality(self, profile: Optional[str]) -> None:
        """Persist the startup personality to the instance .env and config."""
        if LOCKED_PROFILE is not None:
            return
        selection = (profile or "").strip() or None
        try:
            from bobe.config import set_custom_profile

            set_custom_profile(selection)
        except Exception:
            pass

        if not self._instance_path:
            return
        try:
            env_path = Path(self._instance_path) / ".env"
            lines = self._read_env_lines(env_path)
            replaced = False
            for i, ln in enumerate(list(lines)):
                if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                    if selection:
                        lines[i] = f"REACHY_MINI_CUSTOM_PROFILE={selection}"
                    else:
                        lines.pop(i)
                    replaced = True
                    break
            if selection and not replaced:
                lines.append(f"REACHY_MINI_CUSTOM_PROFILE={selection}")
            if selection is None and not env_path.exists():
                return
            final_text = "\n".join(lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Persisted startup personality to %s", env_path)
            try:
                from dotenv import load_dotenv

                load_dotenv(dotenv_path=str(env_path), override=True)
            except Exception:
                pass
        except Exception as e:
            logger.warning("Failed to persist REACHY_MINI_CUSTOM_PROFILE: %s", e)

    def _read_persisted_personality(self) -> Optional[str]:
        """Read persisted startup personality from instance .env (if any)."""
        if not self._instance_path:
            return None
        env_path = Path(self._instance_path) / ".env"
        try:
            if env_path.exists():
                for ln in env_path.read_text(encoding="utf-8").splitlines():
                    if ln.strip().startswith("REACHY_MINI_CUSTOM_PROFILE="):
                        _, _, val = ln.partition("=")
                        v = val.strip()
                        return v or None
        except Exception:
            pass
        return None

    def _init_settings_ui_if_needed(self) -> None:
        """Attach minimal settings UI to the settings app.

        Always mounts the UI when a settings_app is provided so that users
        see a confirmation message even if the API key is already configured.
        """
        if self._settings_initialized:
            return
        if self._settings_app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(self._settings_app, "mount"):
            try:
                # Serve /static/* assets
                self._settings_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        class ApiSettingsPayload(BaseModel):
            openai_api_key: str
            anthropic_api_key: str
            claude_model: str = DEFAULT_CLAUDE_MODEL

        # GET / -> index.html
        @self._settings_app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        # GET /favicon.ico -> optional, avoid noisy 404s on some browsers
        @self._settings_app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        # GET /status -> required keys plus live wake/streaming state
        @self._settings_app.get("/status")
        def _status() -> JSONResponse:
            has_openai_key = _is_plausible_openai_key(str(config.OPENAI_API_KEY or ""))
            has_anthropic_key = _is_plausible_anthropic_key(os.getenv("ANTHROPIC_API_KEY"))

            wake_config = getattr(self.handler, "wake_config", None)
            wake_session = getattr(self.handler, "wake_session", None)
            wake_enabled = wake_config is not None
            awake = bool(wake_session and wake_session.awake)

            wake_detector = getattr(self.handler, "_wake_detector", None)
            wake_debug = wake_detector.debug_state() if wake_detector is not None else None
            wake_remote_url = getattr(wake_config, "remote_url", None) if wake_config else None

            return JSONResponse(
                {
                    "has_key": has_openai_key and has_anthropic_key,
                    "has_openai_key": has_openai_key,
                    "has_anthropic_key": has_anthropic_key,
                    "claude_model": os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
                    "wake_enabled": wake_enabled,
                    "awake": awake,
                    "wake_backend": wake_config.backend if wake_config else None,
                    "wake_model": wake_config.model_name if wake_config else None,
                    "wake_phrase": getattr(wake_detector, "phrase", None) if wake_detector is not None else None,
                    "wake_remote_url": wake_remote_url,
                    "wake_timeout_s": wake_config.timeout_s if wake_config else None,
                    "wake_debug": wake_debug,
                    "wake_test_mode": bool(getattr(self.handler, "wake_test_mode", False)),
                    "wake_test_detections": int(getattr(self.handler, "wake_test_detections", 0)),
                }
            )

        class WakeTestPayload(BaseModel):
            enabled: bool

        # POST /wake-test -> toggle diagnostic mode (scores recorded, no waking/answers)
        @self._settings_app.post("/wake-test")
        def _wake_test(payload: WakeTestPayload) -> JSONResponse:
            handler = self.handler
            handler.wake_test_mode = payload.enabled
            if payload.enabled:
                handler.wake_test_detections = 0
                wake_session = getattr(handler, "wake_session", None)
                if wake_session is not None:
                    wake_session.sleep()
            return JSONResponse(
                {
                    "wake_test_mode": handler.wake_test_mode,
                    "wake_test_detections": handler.wake_test_detections,
                }
            )

        # GET /ready -> whether backend finished loading tools
        @self._settings_app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("bobe.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        # POST /api_keys -> set/persist explicit user-provided keys
        @self._settings_app.post("/api_keys")
        def _set_api_keys(payload: ApiSettingsPayload) -> JSONResponse:
            openai_key = (payload.openai_api_key or "").strip()
            anthropic_key = (payload.anthropic_api_key or "").strip()
            claude_model = (payload.claude_model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
            if not _is_plausible_openai_key(openai_key):
                return JSONResponse({"ok": False, "error": "invalid_openai_api_key"}, status_code=400)
            if not _is_plausible_anthropic_key(anthropic_key):
                return JSONResponse({"ok": False, "error": "invalid_anthropic_api_key"}, status_code=400)
            self._persist_api_settings(
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                claude_model=claude_model,
            )
            return JSONResponse({"ok": True})

        self._settings_initialized = True

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops.

        If the OpenAI key is missing, expose a tiny settings UI via the
        Reachy Mini settings server to collect it before starting streams.
        """
        self._stop_event.clear()

        # Try to load an existing instance .env first (covers subsequent runs)
        if self._instance_path:
            try:
                from dotenv import load_dotenv

                from bobe.config import set_custom_profile

                env_path = Path(self._instance_path) / ".env"
                if env_path.exists():
                    load_dotenv(dotenv_path=str(env_path), override=True)
                    # Update config with newly loaded values
                    new_key = os.getenv("OPENAI_API_KEY", "").strip()
                    if new_key:
                        try:
                            config.OPENAI_API_KEY = new_key
                        except Exception:
                            pass
                    if LOCKED_PROFILE is None:
                        new_profile = os.getenv("REACHY_MINI_CUSTOM_PROFILE")
                        if new_profile is not None:
                            try:
                                set_custom_profile(new_profile.strip() or None)
                            except Exception:
                                pass  # Best-effort profile update
            except Exception:
                pass  # Instance .env loading is optional; continue with defaults

        # Always expose settings UI if a settings app is available.
        self._init_settings_ui_if_needed()

        # Never auto-download shared/demo keys. Wait for explicit user-provided keys.
        if not self._required_api_keys_configured():
            logger.warning("Required API keys missing. Open the app settings page to enter OpenAI and Anthropic keys.")
            try:
                while not self._required_api_keys_configured():
                    time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info("Interrupted while waiting for API keys.")
                return

        # Start media after key is set/available
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            # Capture loop for cross-thread personality actions
            loop = asyncio.get_running_loop()
            self._asyncio_loop = loop  # type: ignore[assignment]
            # Mount personality routes now that loop and handler are available
            try:
                if self._settings_app is not None:
                    mount_personality_routes(
                        self._settings_app,
                        self.handler,
                        lambda: self._asyncio_loop,
                        persist_personality=self._persist_personality,
                        get_persisted_personality=self._read_persisted_personality,
                    )
            except Exception:
                pass
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
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
        """
        logger.info("Stopping LocalStream...")

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

        # Now signal async loops to stop
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

    def clear_audio_queue(self) -> None:
        """Flush the player's appsrc to drop any queued audio immediately."""
        logger.info("User intervention: flushing player queue")
        if self._robot.media.backend == MediaBackend.GSTREAMER:
            # Directly flush gstreamer audio pipe
            self._robot.media.audio.clear_player()
        elif self._robot.media.backend == MediaBackend.DEFAULT or self._robot.media.backend == MediaBackend.DEFAULT_NO_VIDEO:
            self._robot.media.audio.clear_output_buffer()
        self.handler.output_queue = asyncio.Queue()

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
