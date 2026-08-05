"""BoBe (Reachy Mini) platform adapter — Hermes plugin.

Outbound-only channel: ``send()`` POSTs the message to the BoBe wake
daemon's ``/v1/announce`` endpoint on this Mac; the daemon relays it over
the robot's wake WebSocket and the robot speaks it aloud. There is no
inbound side — voice requests reach Hermes through the API server.

Install: copy or symlink this directory to ``~/.hermes/plugins/bobe/``,
set ``BOBE_WAKE_TOKEN`` in ``~/.hermes/.env`` (same value as the wake
daemon's token), enable the platform in ``~/.hermes/config.yaml``::

    platforms:
      bobe:
        enabled: true

then run ``hermes gateway restart``. After that, "send it to bobe",
cron ``deliver=bobe``, and webhook delivery all reach the robot's voice.
"""

import os
import logging
from typing import Any, Dict, List, Optional

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult


logger = logging.getLogger(__name__)

DEFAULT_ANNOUNCE_URL = "http://127.0.0.1:8765/v1/announce"
# Announcements are spoken aloud — anything longer than this is a bad fit.
MAX_MESSAGE_LENGTH = 2000
SEND_TIMEOUT_S = 15.0


def _announce_url(extra: Dict[str, Any]) -> str:
    return (extra.get("announce_url") or os.getenv("BOBE_ANNOUNCE_URL", DEFAULT_ANNOUNCE_URL)).strip()


def _wake_token(extra: Dict[str, Any]) -> str:
    return (extra.get("token") or os.getenv("BOBE_WAKE_TOKEN", "")).strip()


async def _post_announcement(url: str, token: str, message: str) -> SendResult:
    """POST one announcement to the wake daemon, mapping errors to SendResult."""
    if len(message) > MAX_MESSAGE_LENGTH:
        logger.warning("bobe: truncating announcement from %d to %d chars", len(message), MAX_MESSAGE_LENGTH)
        message = message[:MAX_MESSAGE_LENGTH]
    try:
        async with httpx.AsyncClient(timeout=SEND_TIMEOUT_S) as client:
            resp = await client.post(url, json={"message": message}, headers={"X-BoBe-Wake-Token": token})
    except httpx.TimeoutException:
        return SendResult(success=False, error="Timeout reaching the BoBe wake daemon")
    except Exception as exc:
        return SendResult(success=False, error=f"BoBe wake daemon unreachable: {exc}")

    if resp.status_code < 300:
        return SendResult(success=True, message_id="bobe-announce")
    try:
        error = resp.json().get("error", "")
    except Exception:
        error = resp.text[:200]
    if resp.status_code == 409:
        return SendResult(success=False, error="Robot is not connected to the wake daemon")
    return SendResult(success=False, error=f"HTTP {resp.status_code}: {error}")


def check_requirements() -> bool:
    """The adapter only needs httpx (a Hermes dependency) and the daemon token."""
    return bool(os.getenv("BOBE_WAKE_TOKEN", "").strip())


class BobeAdapter(BasePlatformAdapter):
    """Speak Hermes messages through the Reachy Mini robot."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("bobe"))
        extra = config.extra or {}
        self._url = _announce_url(extra)
        self._token = _wake_token(extra)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._token:
            logger.warning("[%s] BOBE_WAKE_TOKEN not configured", self.name)
            return False
        self._mark_connected()
        logger.info("[%s] Connected — announcements go to %s", self.name, self._url)
        return True

    async def disconnect(self) -> None:
        return None

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """POST the message to the wake daemon; the robot speaks it."""
        return await _post_announcement(self._url, self._token, content)


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process delivery for cron / send_message_tool fallbacks.

    ``thread_id`` / ``media_files`` are accepted for signature parity only —
    the robot speaks plain text.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    result = await _post_announcement(_announce_url(extra), _wake_token(extra), message)
    if result.success:
        return {"message_id": result.message_id}
    return {"error": result.error}


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="bobe",
        label="BoBe",
        adapter_factory=lambda cfg: BobeAdapter(cfg),
        check_fn=check_requirements,
        required_env=["BOBE_WAKE_TOKEN"],
        install_hint="Set BOBE_WAKE_TOKEN in ~/.hermes/.env (wake daemon's token)",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🤖",
        pii_safe=True,
        platform_hint=(
            "Messages to bobe are spoken aloud by a home robot. "
            "Write short, plain spoken sentences — no markdown, no links, "
            "no code, nothing sensitive."
        ),
    )
