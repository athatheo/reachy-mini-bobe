"""Robot-side client and confirmation gate for Claude Code managed sessions."""

from __future__ import annotations

import asyncio
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from bobe.claude_code_client import (
    DEFAULT_CONFIRM_TTL_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    derive_daemon_http_url,
    request_daemon_json,
    transcript_matches_phrase,
)
from bobe.env_utils import clean_optional, parse_float

COMMAND_CONFIRMATION_PHRASE = "confirm claude command"
CONTROL_PATH = "/v1/claude-code"


@dataclass(frozen=True)
class ClaudeCodeSessionSettings:
    """Robot-side settings for the Mac Claude Code session API."""

    base_url: str | None
    token: str | None
    confirm_ttl_s: float = DEFAULT_CONFIRM_TTL_S
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S

    @property
    def is_configured(self) -> bool:
        """Return whether the robot has the Mac session API credentials."""
        return bool(self.base_url and self.token)


@dataclass
class PendingClaudeCodeCommand:
    """A pending, not-yet-confirmed Claude Code instruction."""

    command: str
    requested_at: float
    expires_at: float


class ClaudeCodeSessionController:
    """Owns pending voice commands and calls the Mac session API."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], ClaudeCodeSessionSettings] | None = None,
        clock: Callable[[], float] = time.monotonic,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        """Initialize the controller with injectable clock and HTTP opener."""
        self._settings_loader = settings_loader or load_claude_code_session_settings
        self._clock = clock
        self._opener = opener
        self._pending: PendingClaudeCodeCommand | None = None

    async def start(self) -> dict[str, Any]:
        """Start or reuse a daemon-owned Claude Code session."""
        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()
        return await asyncio.to_thread(self._post, settings, "/session/start", {})

    def request_send(self, command: str) -> dict[str, Any]:
        """Stage a command for exact spoken confirmation."""
        clean_command = command.strip()
        if not clean_command:
            return {"status": "error", "error": "command is required"}

        settings = self._settings_loader()
        if not settings.is_configured:
            self._pending = None
            return _missing_config()

        now = self._clock()
        ttl = max(1.0, settings.confirm_ttl_s)
        self._pending = PendingClaudeCodeCommand(
            command=clean_command,
            requested_at=now,
            expires_at=now + ttl,
        )
        return {
            "status": "pending_confirmation",
            "confirmation_phrase": COMMAND_CONFIRMATION_PHRASE,
            "command": clean_command,
            "expires_in_s": round(ttl, 1),
            "message": f"To send that to Claude Code, say exactly: {COMMAND_CONFIRMATION_PHRASE}.",
        }

    async def maybe_confirm_from_transcript(self, transcript: str | None) -> dict[str, Any] | None:
        """Send a pending command only after the exact confirmation phrase."""
        if not command_confirmation_phrase_matches(transcript):
            return None

        pending = self._pending
        if pending is None:
            return {"status": "no_pending_command", "message": "No Claude Code command is pending."}

        now = self._clock()
        self._pending = None
        if now > pending.expires_at:
            return {
                "status": "expired",
                "message": "Claude Code command confirmation expired. Tell me the command again.",
            }

        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()

        result = await asyncio.to_thread(self._post, settings, "/session/send", {"command": pending.command})
        if result.get("ok"):
            return {
                "status": "sent",
                "message": "I sent that command to Claude Code.",
                "result": result,
            }
        error = str(result.get("error") or "send_failed")
        return {
            "status": "error",
            "message": f"Claude Code command failed: {error}.",
            "result": result,
        }

    async def status(self) -> dict[str, Any]:
        """Fetch managed Claude Code session status."""
        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()
        return await asyncio.to_thread(self._request, settings, "GET", "/session/status", None)

    async def stop(self) -> dict[str, Any]:
        """Stop the managed Claude Code session."""
        self._pending = None
        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()
        return await asyncio.to_thread(self._post, settings, "/session/stop", {})

    def has_pending(self) -> bool:
        """Return whether a non-expired command confirmation is pending."""
        pending = self._pending
        if pending is None:
            return False
        if self._clock() > pending.expires_at:
            self._pending = None
            return False
        return True

    def _post(self, settings: ClaudeCodeSessionSettings, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request(settings, "POST", path, payload)

    def _request(
        self,
        settings: ClaudeCodeSessionSettings,
        method: str,
        path: str,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        assert settings.base_url is not None
        assert settings.token is not None
        return request_daemon_json(
            self._opener,
            url=urllib.parse.urljoin(settings.base_url.rstrip("/") + "/", path.lstrip("/")),
            token=settings.token,
            method=method,
            payload=payload,
            timeout_s=settings.request_timeout_s,
            log_label="Claude Code session",
        )


def load_claude_code_session_settings(env: dict[str, str] | None = None) -> ClaudeCodeSessionSettings:
    """Load robot-side Claude Code session settings."""
    source = os.environ if env is None else env
    base_url = clean_optional(source.get("BOBE_CLAUDE_CODE_CONTROL_URL"))
    if base_url is None:
        base_url = derive_control_url_from_wake_url(source.get("BOBE_WAKE_REMOTE_URL"))

    return ClaudeCodeSessionSettings(
        base_url=base_url,
        token=clean_optional(source.get("BOBE_CLAUDE_CODE_LAUNCH_TOKEN")),
        confirm_ttl_s=max(
            1.0, parse_float(source.get("BOBE_CLAUDE_CODE_COMMAND_CONFIRM_TTL_S"), DEFAULT_CONFIRM_TTL_S)
        ),
        request_timeout_s=max(
            1.0, parse_float(source.get("BOBE_CLAUDE_CODE_REQUEST_TIMEOUT_S"), DEFAULT_REQUEST_TIMEOUT_S)
        ),
    )


def derive_control_url_from_wake_url(wake_url: str | None) -> str | None:
    """Derive the Claude Code control base URL from the wake daemon URL."""
    return derive_daemon_http_url(wake_url, CONTROL_PATH)


def command_confirmation_phrase_matches(transcript: str | None) -> bool:
    """Return True only for the exact command confirmation phrase."""
    return transcript_matches_phrase(transcript, COMMAND_CONFIRMATION_PHRASE)


_controller = ClaudeCodeSessionController()


def get_claude_code_session_controller() -> ClaudeCodeSessionController:
    """Return the process-wide Claude Code session controller."""
    return _controller


def reset_claude_code_session_controller(controller: ClaudeCodeSessionController | None = None) -> None:
    """Reset the process-wide controller for tests."""
    global _controller
    _controller = controller or ClaudeCodeSessionController()


async def maybe_confirm_claude_code_command(transcript: str | None) -> dict[str, Any] | None:
    """Confirm a pending Claude Code command from a completed transcript."""
    return await _controller.maybe_confirm_from_transcript(transcript)


def _missing_config() -> dict[str, Any]:
    return {
        "status": "missing_config",
        "ok": False,
        "message": (
            "Claude Code session control is not configured. Set "
            "BOBE_CLAUDE_CODE_CONTROL_URL or BOBE_WAKE_REMOTE_URL, plus BOBE_CLAUDE_CODE_LAUNCH_TOKEN."
        ),
    }
