"""HTTP settings UI routes for BoBe (mounted before the voice handler starts)."""

# ruff: noqa: D102,D103,D107

from __future__ import annotations
import os
import logging
from typing import Any, Callable, Annotated
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Body, FastAPI, Response
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from bobe.claude import DEFAULT_CLAUDE_MODEL
from bobe.config import config
from bobe.env_file import persist_api_settings, is_plausible_openai_key, is_plausible_anthropic_key
from bobe.wake_env import persist_wake_env, is_wake_remote_host_allowed
from bobe.wake.phrases import WAKE_PHRASE
from bobe.openai_realtime import OpenaiRealtimeHandler


logger = logging.getLogger(__name__)

WAKE_DEBUG_TRANSCRIPT_KEYS = (
    "transcript_last",
    "transcript_partial",
    "transcript_stream",
    "transcript_display",
)
WAKE_DEBUG_REMOTE_TRANSCRIPT_KEYS = ("transcript", "partial")


class ApiSettingsPayload(BaseModel):
    """POST /api_keys payload."""

    openai_api_key: str
    anthropic_api_key: str
    claude_model: str = DEFAULT_CLAUDE_MODEL


class WakeConfigPayload(BaseModel):
    """POST /wake-config payload."""

    backend: str = "remote"
    remote_url: str
    token: str
    gain: float = 1.75


_settings_server: SettingsUIServer | None = None


def _redact_wake_debug_for_public(
    wake_debug: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Strip transcript-bearing wake debug fields for unauthenticated status callers."""
    if wake_debug is None:
        return None
    redacted = dict(wake_debug)
    for key in WAKE_DEBUG_TRANSCRIPT_KEYS:
        redacted.pop(key, None)
    remote_stats = redacted.get("remote_stats")
    if isinstance(remote_stats, dict):
        redacted["remote_stats"] = {
            key: value
            for key, value in remote_stats.items()
            if key not in WAKE_DEBUG_REMOTE_TRANSCRIPT_KEYS
        }
    return redacted


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

        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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
            has_openai_key = is_plausible_openai_key(openai_key)
            has_anthropic_key = is_plausible_anthropic_key(anthropic_key)

            handler = self.handler
            wake_config = getattr(handler, "wake_config", None) if handler else None
            wake_session = getattr(handler, "wake_session", None) if handler else None
            wake_enabled = bool(getattr(handler, "wake_gating_enabled", False)) if handler else False
            wake_error = getattr(handler, "wake_error", None) if handler else None
            awake = bool(wake_session and wake_session.awake)
            wake_detector = getattr(handler, "_wake_detector", None) if handler else None
            wake_debug = wake_detector.debug_state() if wake_detector is not None else None
            wake_remote_url = getattr(wake_config, "remote_url", None) if wake_config else None
            openai_connected = bool(getattr(handler, "connection", None)) if handler else False
            authenticated = has_openai_key and has_anthropic_key
            if not authenticated:
                wake_debug = _redact_wake_debug_for_public(wake_debug)

            return JSONResponse(
                {
                    "has_key": has_openai_key and has_anthropic_key,
                    "has_openai_key": has_openai_key,
                    "has_anthropic_key": has_anthropic_key,
                    "openai_connected": openai_connected,
                    "claude_model": os.getenv("CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL),
                    "wake_enabled": wake_enabled,
                    "wake_error": wake_error,
                    "awake": awake,
                    "wake_backend": wake_config.backend if wake_config else None,
                    "wake_phrase": wake_config.phrase if wake_config else WAKE_PHRASE,
                    "wake_remote_url": wake_remote_url,
                    "wake_timeout_s": wake_config.timeout_s if wake_config else None,
                    "wake_debug": wake_debug,
                }
            )

        @app.post("/wake-config")
        def _wake_config(payload: Annotated[WakeConfigPayload, Body()]) -> JSONResponse:
            remote_url = (payload.remote_url or "").strip()
            token = (payload.token or "").strip()
            backend = (payload.backend or "remote").strip().lower()
            if backend != "remote":
                return JSONResponse({"ok": False, "error": "unsupported_backend"}, status_code=400)
            if not remote_url.startswith("ws://") and not remote_url.startswith("wss://"):
                return JSONResponse({"ok": False, "error": "invalid_remote_url"}, status_code=400)
            ws_scheme = "http://" if remote_url.startswith("ws://") else "https://"
            ws_path = remote_url.split("://", 1)[1]
            hostname = urlparse(ws_scheme + ws_path).hostname
            if not hostname or not is_wake_remote_host_allowed(hostname):
                return JSONResponse({"ok": False, "error": "remote_host_not_allowed"}, status_code=400)
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

        @app.post("/api_keys")
        def _set_api_keys(payload: Annotated[ApiSettingsPayload, Body()]) -> JSONResponse:
            openai_key = (payload.openai_api_key or "").strip()
            anthropic_key = (payload.anthropic_api_key or "").strip()
            claude_model = (payload.claude_model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL
            if not is_plausible_openai_key(openai_key):
                return JSONResponse({"ok": False, "error": "invalid_openai_api_key"}, status_code=400)
            if not is_plausible_anthropic_key(anthropic_key):
                return JSONResponse({"ok": False, "error": "invalid_anthropic_api_key"}, status_code=400)
            persist_api_settings(
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
