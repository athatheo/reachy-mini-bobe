"""Robot-side state and client for confirmed Claude Code launches."""

from __future__ import annotations
import os
import time
import asyncio
import logging
import urllib.request
from typing import Any, Callable
from dataclasses import dataclass

from bobe.env_utils import parse_float, clean_optional
from bobe.claude_code_client import (
    DEFAULT_CONFIRM_TTL_S,
    DEFAULT_REQUEST_TIMEOUT_S,
    request_daemon_json,
    derive_daemon_http_url,
    transcript_matches_phrase,
)


logger = logging.getLogger(__name__)

CONFIRMATION_PHRASE = "confirm launch claude code"
LAUNCH_PATH = "/v1/launch/claude-code"


@dataclass(frozen=True)
class ClaudeCodeLaunchSettings:
    """Robot-side settings for the Mac launch endpoint."""

    launch_url: str | None
    launch_token: str | None
    confirm_ttl_s: float = DEFAULT_CONFIRM_TTL_S
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S

    @property
    def is_configured(self) -> bool:
        """Return whether the robot has the Mac endpoint credentials."""
        return bool(self.launch_url and self.launch_token)


@dataclass
class PendingClaudeCodeLaunch:
    """A pending, not-yet-confirmed launch request."""

    requested_at: float
    expires_at: float


class ClaudeCodeLaunchController:
    """Owns pending launch state and the Mac endpoint call."""

    def __init__(
        self,
        *,
        settings_loader: Callable[[], ClaudeCodeLaunchSettings] | None = None,
        clock: Callable[[], float] = time.monotonic,
        opener: Callable[..., Any] = urllib.request.urlopen,
    ) -> None:
        """Initialize the controller with injectable clock and HTTP opener."""
        self._settings_loader = settings_loader or load_claude_code_launch_settings
        self._clock = clock
        self._opener = opener
        self._pending: PendingClaudeCodeLaunch | None = None

    def request(self) -> dict[str, Any]:
        """Create a pending launch request if the robot is configured."""
        settings = self._settings_loader()
        if not settings.is_configured:
            self._pending = None
            return {
                "status": "missing_config",
                "message": (
                    "Claude Code launching is not configured. Set "
                    "BOBE_CLAUDE_CODE_LAUNCH_URL and BOBE_CLAUDE_CODE_LAUNCH_TOKEN."
                ),
            }

        now = self._clock()
        self._pending = PendingClaudeCodeLaunch(
            requested_at=now,
            expires_at=now + max(1.0, settings.confirm_ttl_s),
        )
        return {
            "status": "pending_confirmation",
            "confirmation_phrase": CONFIRMATION_PHRASE,
            "expires_in_s": round(max(1.0, settings.confirm_ttl_s), 1),
            "message": f"To launch Claude Code, say exactly: {CONFIRMATION_PHRASE}.",
        }

    def cancel(self) -> dict[str, Any]:
        """Cancel any pending launch request."""
        had_pending = self._pending is not None
        self._pending = None
        return {
            "status": "cancelled" if had_pending else "nothing_pending",
            "message": "Claude Code launch cancelled." if had_pending else "No Claude Code launch was pending.",
        }

    def has_pending(self) -> bool:
        """Return whether a non-expired launch confirmation is pending."""
        pending = self._pending
        if pending is None:
            return False
        if self._clock() > pending.expires_at:
            self._pending = None
            return False
        return True

    async def maybe_confirm_from_transcript(self, transcript: str | None) -> dict[str, Any] | None:
        """Launch only when a completed transcript is the exact confirmation phrase."""
        if not confirmation_phrase_matches(transcript):
            return None

        pending = self._pending
        if pending is None:
            return {
                "status": "no_pending_launch",
                "message": "No Claude Code launch is pending.",
            }

        now = self._clock()
        self._pending = None
        if now > pending.expires_at:
            return {
                "status": "expired",
                "message": "Claude Code launch confirmation expired. Ask me to launch it again.",
            }

        settings = self._settings_loader()
        if not settings.is_configured:
            return {
                "status": "missing_config",
                "message": "Claude Code launching is not configured on the robot.",
            }

        result = await asyncio.to_thread(self._post_launch, settings)
        if result.get("ok"):
            return {
                "status": "launched",
                "message": "Claude Code is launching on the Mac mini.",
                "result": result,
            }

        error = str(result.get("error") or "launch_failed")
        if error == "cooldown":
            retry_after = result.get("retry_after_s")
            return {
                "status": "cooldown",
                "message": f"Claude Code was launched recently. Try again in {retry_after} seconds.",
                "result": result,
            }
        return {
            "status": "error",
            "message": f"Claude Code launch failed: {error}.",
            "result": result,
        }

    def _post_launch(self, settings: ClaudeCodeLaunchSettings) -> dict[str, Any]:
        assert settings.launch_url is not None
        assert settings.launch_token is not None
        return request_daemon_json(
            self._opener,
            url=settings.launch_url,
            token=settings.launch_token,
            method="POST",
            payload={"source": "bobe", "confirmed_phrase": CONFIRMATION_PHRASE},
            timeout_s=settings.request_timeout_s,
            log_label="Claude Code launch",
        )


def load_claude_code_launch_settings(env: dict[str, str] | None = None) -> ClaudeCodeLaunchSettings:
    """Load robot-side Claude Code launch settings."""
    source = os.environ if env is None else env
    launch_url = clean_optional(source.get("BOBE_CLAUDE_CODE_LAUNCH_URL"))
    if launch_url is None:
        launch_url = derive_launch_url_from_wake_url(source.get("BOBE_WAKE_REMOTE_URL"))

    return ClaudeCodeLaunchSettings(
        launch_url=launch_url,
        launch_token=clean_optional(source.get("BOBE_CLAUDE_CODE_LAUNCH_TOKEN")),
        confirm_ttl_s=max(1.0, parse_float(source.get("BOBE_CLAUDE_CODE_CONFIRM_TTL_S"), DEFAULT_CONFIRM_TTL_S)),
        request_timeout_s=max(
            1.0, parse_float(source.get("BOBE_CLAUDE_CODE_REQUEST_TIMEOUT_S"), DEFAULT_REQUEST_TIMEOUT_S)
        ),
    )


def derive_launch_url_from_wake_url(wake_url: str | None) -> str | None:
    """Derive the launch endpoint from the configured wake daemon URL."""
    return derive_daemon_http_url(wake_url, LAUNCH_PATH)


def confirmation_phrase_matches(transcript: str | None) -> bool:
    """Return True only for the exact confirmation phrase, allowing ASR punctuation."""
    return transcript_matches_phrase(transcript, CONFIRMATION_PHRASE)


_controller = ClaudeCodeLaunchController()


def get_claude_code_launch_controller() -> ClaudeCodeLaunchController:
    """Return the process-wide Claude Code launch controller."""
    return _controller


def reset_claude_code_launch_controller(controller: ClaudeCodeLaunchController | None = None) -> None:
    """Reset the process-wide controller for tests."""
    global _controller
    _controller = controller or ClaudeCodeLaunchController()


async def maybe_confirm_claude_code_launch(transcript: str | None) -> dict[str, Any] | None:
    """Confirm a pending launch from a completed transcript when possible."""
    return await _controller.maybe_confirm_from_transcript(transcript)
