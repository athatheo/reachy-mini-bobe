"""HTTP settings UI routes for BoBe (mounted before the voice handler starts)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable

from bobe.claude import DEFAULT_CLAUDE_MODEL
from bobe.config import LOCKED_PROFILE, config
from bobe.headless_personality_ui import mount_personality_routes
from bobe.instance import load_instance_env
from bobe.openai_realtime import OpenaiRealtimeHandler
from bobe.wake_env import persist_wake_env


logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Response
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel
    from starlette.staticfiles import StaticFiles
except Exception:  # pragma: no cover
    FastAPI = object  # type: ignore
    Response = object  # type: ignore
    FileResponse = object  # type: ignore
    JSONResponse = object  # type: ignore
    BaseModel = object  # type: ignore
    StaticFiles = object  # type: ignore


def _is_plausible_openai_key(value: str | None) -> bool:
    key = (value or "").strip()
    return key.startswith("sk-") and len(key) >= 20


def _is_plausible_anthropic_key(value: str | None) -> bool:
    key = (value or "").strip()
    return key.startswith("sk-ant-") and len(key) >= 20


def _read_env_lines(env_path: Path) -> list[str]:
    if env_path.exists():
        try:
            return env_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
    example = Path(__file__).parent / ".env.example"
    if example.exists():
        try:
            return example.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
    return []


def _persist_api_settings(
    instance_path: str | None,
    *,
    openai_api_key: str,
    anthropic_api_key: str,
    claude_model: str,
) -> None:
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

    if not instance_path:
        return

    env_path = Path(instance_path) / ".env"
    lines = _read_env_lines(env_path)
    for key, value in values.items():
        replacement = f"{key}={value}"
        for index, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[index] = replacement
                break
        else:
            lines.append(replacement)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    load_instance_env(instance_path)


_settings_server: SettingsUIServer | None = None


class SettingsUIServer:
    """Registers BoBe settings routes on the Reachy Mini settings app."""

    def __init__(self, instance_path: str | None, get_handler: Callable[[], OpenaiRealtimeHandler | None]) -> None:
        self.instance_path = instance_path
        self._get_handler = get_handler
        self._mounted = False

    @property
    def handler(self) -> OpenaiRealtimeHandler | None:
        return self._get_handler()

    def mount(self, app: FastAPI) -> None:
        if self._mounted or app is None:
            return

        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if hasattr(app, "mount"):
            try:
                app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            except Exception:
                pass

        class ApiSettingsPayload(BaseModel):
            openai_api_key: str
            anthropic_api_key: str
            claude_model: str = DEFAULT_CLAUDE_MODEL

        class WakeTestPayload(BaseModel):
            enabled: bool

        class WakeConfigPayload(BaseModel):
            backend: str = "remote"
            remote_url: str
            token: str
            gain: float = 1.75

        @app.get("/")
        def _root() -> FileResponse:
            return FileResponse(str(index_file))

        @app.get("/favicon.ico")
        def _favicon() -> Response:
            return Response(status_code=204)

        @app.get("/status")
        def _status() -> JSONResponse:
            openai_key = os.getenv("OPENAI_API_KEY") or str(config.OPENAI_API_KEY or "")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY") or ""
            has_openai_key = _is_plausible_openai_key(openai_key)
            has_anthropic_key = _is_plausible_anthropic_key(anthropic_key)

            handler = self.handler
            wake_config = getattr(handler, "wake_config", None) if handler else None
            wake_session = getattr(handler, "wake_session", None) if handler else None
            wake_enabled = wake_config is not None
            awake = bool(wake_session and wake_session.awake)
            wake_detector = getattr(handler, "_wake_detector", None) if handler else None
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
                    "wake_test_mode": bool(getattr(handler, "wake_test_mode", False)) if handler else False,
                    "wake_test_detections": int(getattr(handler, "wake_test_detections", 0)) if handler else 0,
                }
            )

        @app.post("/wake-config")
        def _wake_config(payload: WakeConfigPayload) -> JSONResponse:
            remote_url = (payload.remote_url or "").strip()
            token = (payload.token or "").strip()
            backend = (payload.backend or "remote").strip().lower()
            if backend != "remote":
                return JSONResponse({"ok": False, "error": "unsupported_backend"}, status_code=400)
            if not remote_url.startswith("ws://") and not remote_url.startswith("wss://"):
                return JSONResponse({"ok": False, "error": "invalid_remote_url"}, status_code=400)
            if not token:
                return JSONResponse({"ok": False, "error": "missing_token"}, status_code=400)
            if self.instance_path is None:
                return JSONResponse({"ok": False, "error": "missing_instance_path"}, status_code=500)
            try:
                env_path = persist_wake_env(
                    self.instance_path,
                    backend=backend,
                    remote_url=remote_url,
                    token=token,
                    gain=max(1.0, float(payload.gain)),
                )
            except Exception as exc:
                logger.warning("Failed to persist wake config: %s", exc)
                return JSONResponse({"ok": False, "error": "persist_failed"}, status_code=500)
            return JSONResponse({"ok": True, "env_path": str(env_path), "restart_required": True})

        @app.post("/wake-test")
        def _wake_test(payload: WakeTestPayload) -> JSONResponse:
            handler = self.handler
            if handler is None:
                return JSONResponse({"ok": False, "error": "handler_not_ready"}, status_code=503)
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

        @app.get("/ready")
        def _ready() -> JSONResponse:
            try:
                mod = sys.modules.get("bobe.tools.core_tools")
                ready = bool(getattr(mod, "_TOOLS_INITIALIZED", False)) if mod else False
            except Exception:
                ready = False
            return JSONResponse({"ready": ready})

        @app.post("/api_keys")
        def _set_api_keys(payload: ApiSettingsPayload) -> JSONResponse:
            openai_key = (payload.openai_api_key or "").strip()
            anthropic_key = (payload.anthropic_api_key or "").strip()
            claude_model = (payload.claude_model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
            if not _is_plausible_openai_key(openai_key):
                return JSONResponse({"ok": False, "error": "invalid_openai_api_key"}, status_code=400)
            if not _is_plausible_anthropic_key(anthropic_key):
                return JSONResponse({"ok": False, "error": "invalid_anthropic_api_key"}, status_code=400)
            _persist_api_settings(
                self.instance_path,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                claude_model=claude_model,
            )
            return JSONResponse({"ok": True})

        self._mounted = True


def bootstrap_settings_ui(
    app: FastAPI | None,
    instance_path: str | None,
    get_handler: Callable[[], OpenaiRealtimeHandler | None],
) -> SettingsUIServer | None:
    """Mount settings routes immediately when the Reachy settings server starts."""
    global _settings_server
    if app is None:
        return None
    if _settings_server is None:
        _settings_server = SettingsUIServer(instance_path, get_handler)
        _settings_server.mount(app)
    else:
        _settings_server.instance_path = instance_path
        _settings_server._get_handler = get_handler
    return _settings_server


def get_settings_server() -> SettingsUIServer | None:
    return _settings_server


def mount_personality_on_settings(app: FastAPI | None, handler: OpenaiRealtimeHandler, get_loop) -> None:
    if app is None:
        return
    mount_personality_routes(
        app,
        handler,
        get_loop,
        persist_personality=None,
        get_persisted_personality=None,
    )
