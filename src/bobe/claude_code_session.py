"""Robot-side client and confirmation gate for Claude Code managed sessions."""

from __future__ import annotations
import os
import time
import asyncio
import urllib.parse
import urllib.request
from typing import Any, Callable
from dataclasses import dataclass

from bobe.env_utils import parse_float, clean_optional
from bobe.claude_code_client import (
    DEFAULT_CONFIRM_TTL_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    ConfirmationGate,
    request_daemon_json,
    derive_daemon_http_url,
    transcript_matches_phrase,
)


COMMAND_CONFIRMATION_PHRASE = "confirm claude command"
COMMAND_CONFIRMATION_INSTRUCTION = f"To send the command, say exactly: {COMMAND_CONFIRMATION_PHRASE}."
CONTROL_PATH = "/v1/claude-code"

# After the daemon accepts a command (202-style), poll briefly for a fast
# result before reporting the command as still running.
SEND_POLL_INTERVAL_S = 1.0
SEND_POLL_WINDOW_S = 8.0


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


class ClaudeCodeSessionController:
    """Owns pending voice commands and calls the Mac session API."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], ClaudeCodeSessionSettings] | None = None,
        clock: Callable[[], float] = time.monotonic,
        opener: Callable[..., Any] = urllib.request.urlopen,
        sleeper: Callable[[float], Any] | None = None,
    ) -> None:
        """Initialize the controller with injectable clock, HTTP opener, and sleeper."""
        self._settings_loader = settings_loader or load_claude_code_session_settings
        self._clock = clock
        self._opener = opener
        self._sleep = sleeper or asyncio.sleep
        self._gate = ConfirmationGate(
            phrase=COMMAND_CONFIRMATION_PHRASE,
            instruction=COMMAND_CONFIRMATION_INSTRUCTION,
            no_pending_reply={
                "status": "no_pending_command",
                "message": "No Claude Code command is pending.",
            },
            expired_reply={
                "status": "expired",
                "message": "Claude Code command confirmation expired. Tell me the command again.",
            },
            clock=clock,
        )

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
            self._gate.clear()
            return _missing_config()

        ttl = self._gate.stage(clean_command, ttl_s=settings.confirm_ttl_s)
        return {
            "status": "pending_confirmation",
            "confirmation_phrase": COMMAND_CONFIRMATION_PHRASE,
            "command": clean_command,
            "expires_in_s": round(ttl, 1),
            "message": f"To send that to Claude Code, say exactly: {COMMAND_CONFIRMATION_PHRASE}.",
        }

    async def maybe_confirm_from_transcript(
        self,
        transcript: str | None,
        *,
        correct_mismatch: bool = True,
    ) -> dict[str, Any] | None:
        """Send a pending command only after the exact confirmation phrase.

        See :meth:`bobe.claude_code_client.ConfirmationGate.consume` for the
        ``correct_mismatch`` coordination contract used by orchestrators with
        several pending confirmations.
        """
        outcome = self._gate.consume(transcript, correct_mismatch=correct_mismatch)
        if outcome is None:
            return None
        if not outcome.confirmed:
            return outcome.reply

        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()

        result = await asyncio.to_thread(self._post, settings, "/session/send", {"command": outcome.payload})
        if result.get("ok"):
            if result.get("accepted"):
                # New daemons accept the command and run it in the background;
                # poll briefly so fast commands still get their result spoken.
                return await self._report_accepted_command(settings, result)
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

    async def _report_accepted_command(
        self,
        settings: ClaudeCodeSessionSettings,
        accepted: dict[str, Any],
    ) -> dict[str, Any]:
        """Poll session status briefly; long commands report as running, not failed."""
        deadline = self._clock() + SEND_POLL_WINDOW_S
        while self._clock() < deadline:
            await self._sleep(SEND_POLL_INTERVAL_S)
            status = await asyncio.to_thread(self._request, settings, "GET", "/session/status", None)
            if not status.get("ok"):
                break  # transient status hiccup: fall through to the running report
            if status.get("session_id") != accepted.get("session_id") or not status.get("active", True):
                return {
                    "status": "stopped",
                    "message": "The Claude Code session was stopped before that command finished.",
                    "result": status,
                }
            if status.get("running"):
                continue
            last_result = status.get("last_result")
            if isinstance(last_result, dict):
                if last_result.get("ok"):
                    return {
                        "status": "sent",
                        "message": "Claude Code finished that command.",
                        "result": last_result,
                    }
                error = str(last_result.get("error") or "send_failed")
                return {
                    "status": "error",
                    "message": f"Claude Code command failed: {error}.",
                    "result": last_result,
                }
            break
        return {
            "status": "running",
            "message": (
                "Claude Code is working on that command. "
                "Ask me for the Claude Code status in a little while."
            ),
            "result": accepted,
        }

    async def status(self) -> dict[str, Any]:
        """Fetch managed Claude Code session status."""
        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()
        return await asyncio.to_thread(self._request, settings, "GET", "/session/status", None)

    async def stop(self) -> dict[str, Any]:
        """Stop the managed Claude Code session."""
        self._gate.clear()
        settings = self._settings_loader()
        if not settings.is_configured:
            return _missing_config()
        return await asyncio.to_thread(self._post, settings, "/session/stop", {})

    def has_pending(self) -> bool:
        """Return whether a non-expired command confirmation is pending."""
        return self._gate.has_pending()

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


async def maybe_confirm_claude_code_command(
    transcript: str | None,
    *,
    correct_mismatch: bool = True,
) -> dict[str, Any] | None:
    """Confirm a pending Claude Code command from a completed transcript."""
    return await _controller.maybe_confirm_from_transcript(transcript, correct_mismatch=correct_mismatch)


def pending_command_confirmation_instruction() -> str | None:
    """Return the exact-phrase instruction while a command confirmation is pending."""
    return COMMAND_CONFIRMATION_INSTRUCTION if _controller.has_pending() else None


def _missing_config() -> dict[str, Any]:
    return {
        "status": "missing_config",
        "ok": False,
        "message": (
            "Claude Code session control is not configured. Set "
            "BOBE_CLAUDE_CODE_CONTROL_URL or BOBE_WAKE_REMOTE_URL, plus BOBE_CLAUDE_CODE_LAUNCH_TOKEN."
        ),
    }
