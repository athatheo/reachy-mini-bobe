"""BoBe (Reachy Mini) platform adapter — Hermes plugin.

Two-way voice channel backed by the BoBe wake daemon on this Mac:

- **Outbound text**: ``send()`` POSTs to the daemon's ``/v1/announce``; the
  daemon relays it over the robot's wake WebSocket and the robot speaks it.
- **Outbound voice**: ``play_tts()`` / ``send_voice()`` POST synthesized audio
  to ``/v1/speak``; the daemon decodes it to PCM and the robot plays it.
- **Inbound voice**: a long-poll loop reads robot utterance transcripts from
  ``/v1/utterances`` and dispatches them as ``MessageType.VOICE`` events, so
  Hermes auto-TTS answers with audio (the chat is opted in on connect).

Authorization is upstream: the robot's stream is authenticated to the daemon
with ``BOBE_WAKE_TOKEN`` and this adapter authenticates with the same shared
secret, so utterances are treated as the owner speaking at home.

Install: copy or symlink this directory to ``~/.hermes/plugins/bobe/``,
set ``BOBE_WAKE_TOKEN`` in ``~/.hermes/.env`` (same value as the wake
daemon's token), enable the platform in ``~/.hermes/config.yaml``::

    platforms:
      bobe:
        enabled: true

then run ``hermes gateway restart``. After that, "send it to bobe",
cron ``deliver=bobe``, and webhook delivery all reach the robot's voice,
and anything the robot hears in converse mode reaches the agent.
"""

# ruff: noqa: D102, D107, D401 — runs inside Hermes, not this repo; keep it
# close to the upstream ntfy plugin's shape rather than this repo's docstyle.

import os
import re
import time
import asyncio
import logging
import mimetypes
from typing import Any, Dict, List, Optional

import httpx
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    SendResult,
    MessageType,
    MessageEvent,
    BasePlatformAdapter,
)


logger = logging.getLogger(__name__)

DEFAULT_ANNOUNCE_URL = "http://127.0.0.1:8765/v1/announce"
# Announcements are spoken aloud — anything longer than this is a bad fit.
MAX_MESSAGE_LENGTH = 2000
SEND_TIMEOUT_S = 15.0
# TTS clips are larger than announce payloads and the daemon decodes them
# before answering; give it real time.
SPEAK_TIMEOUT_S = 30.0
# Long-poll window for /v1/utterances; the HTTP timeout adds slack on top.
POLL_WAIT_S = 25.0
POLL_ERROR_BACKOFF_S = 3.0
POLL_AUTH_BACKOFF_S = 60.0
# After play_tts delivers a reply as audio, the gateway still sends the same
# reply as text; suppress that echo briefly so the robot doesn't speak twice.
TTS_TEXT_SUPPRESS_S = 5.0
# The robot is a single implicit channel.
ROBOT_CHAT_ID = "robot"
# Optional expression tag the agent may put at the START of a bobe reply, e.g.
# "[emotion:amazed1] That is wonderful news!". Stripped before TTS/announce
# and forwarded to the robot's motion system.
EMOTION_TAG_RE = re.compile(r"^\s*\[emotion:\s*([a-z0-9_-]+)\s*\]\s*", re.IGNORECASE)


def _announce_url(extra: Dict[str, Any]) -> str:
    return (extra.get("announce_url") or os.getenv("BOBE_ANNOUNCE_URL", DEFAULT_ANNOUNCE_URL)).strip()


def _daemon_endpoint(announce_url: str, endpoint: str) -> str:
    """Derive a sibling daemon endpoint from the configured announce URL."""
    base = announce_url.rsplit("/v1/", 1)[0].rstrip("/")
    return f"{base}/v1/{endpoint}"


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
    """Two-way voice bridge between Hermes and the Reachy Mini robot."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    # Utterances arrive over the daemon's token-authenticated link and the
    # robot mic belongs to the owner at home; no per-user allowlist applies.
    authorization_is_upstream = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("bobe"))
        extra = config.extra or {}
        self._url = _announce_url(extra)
        self._speak_url = _daemon_endpoint(self._url, "speak")
        self._utterances_url = _daemon_endpoint(self._url, "utterances")
        self._token = _wake_token(extra)
        self._poll_task: Optional[asyncio.Task] = None
        self._suppress_text_until = 0.0
        self._emote_url = _daemon_endpoint(self._url, "emote")
        self._pending_emotion: Optional[str] = None

    def _should_auto_tts_for_chat(self, chat_id: str) -> bool:
        """Robot conversations are voice-first: always answer VOICE with TTS.

        Overridden (rather than seeding ``_auto_tts_enabled_chats`` on
        connect) because the gateway re-syncs those sets from its persisted
        ``/voice`` store after connect, wiping any adapter-side opt-in.
        """
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self._token:
            logger.warning("[%s] BOBE_WAKE_TOKEN not configured", self.name)
            return False
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_utterances(), name="bobe-utterance-poll")
        self._mark_connected()
        logger.info("[%s] Connected — announcements to %s, utterances from %s", self.name, self._url, self._utterances_url)
        return True

    async def disconnect(self) -> None:
        task, self._poll_task = self._poll_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._mark_disconnected()

    # ------------------------------------------------------------------
    # Inbound: robot utterances -> agent
    # ------------------------------------------------------------------

    async def _poll_utterances(self) -> None:
        """Long-poll the wake daemon for robot utterances forever."""
        while True:
            try:
                async with httpx.AsyncClient(timeout=POLL_WAIT_S + 10.0) as client:
                    resp = await client.get(
                        self._utterances_url,
                        params={"wait": POLL_WAIT_S},
                        headers={"X-BoBe-Wake-Token": self._token},
                    )
                if resp.status_code == 401:
                    logger.error("[%s] Wake daemon rejected the token; utterance polling paused", self.name)
                    await asyncio.sleep(POLL_AUTH_BACKOFF_S)
                    continue
                resp.raise_for_status()
                events = resp.json().get("events", [])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("[%s] Utterance poll failed (%s); retrying", self.name, exc)
                await asyncio.sleep(POLL_ERROR_BACKOFF_S)
                continue

            for event in events:
                text = str(event.get("text") or "").strip() if isinstance(event, dict) else ""
                if not text:
                    continue
                try:
                    await self._dispatch_utterance(text)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("[%s] Failed to dispatch utterance", self.name)

    async def _dispatch_utterance(self, text: str) -> None:
        """Hand one robot utterance to the gateway as a VOICE message."""
        logger.info("[%s] Robot utterance: %r", self.name, text)
        source = self.build_source(
            chat_id=ROBOT_CHAT_ID,
            chat_name="BoBe robot",
            chat_type="dm",
            user_id="bobe-voice",
            user_name="BoBe",
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE,
            user_id="bobe-voice",
            user_name="BoBe",
            source=source,
        )
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Outbound: agent replies -> robot
    # ------------------------------------------------------------------

    @staticmethod
    def _split_emotion_tag(content: str) -> tuple[Optional[str], str]:
        """Return (emotion, text-without-tag) for a leading [emotion:NAME] tag."""
        match = EMOTION_TAG_RE.match(content or "")
        if not match:
            return None, content
        return match.group(1).lower(), content[match.end():]

    def prepare_tts_text(self, text: str) -> str:
        """Strip a leading emotion tag (stashing it for play_tts) before TTS."""
        emotion, stripped = self._split_emotion_tag(text)
        if emotion:
            self._pending_emotion = emotion
        return super().prepare_tts_text(stripped)

    async def _post_emote(self, emotion: str) -> None:
        """Best-effort relay of an emotion move to the robot."""
        try:
            async with httpx.AsyncClient(timeout=SEND_TIMEOUT_S) as client:
                resp = await client.post(
                    self._emote_url,
                    json={"emotion": emotion},
                    headers={"X-BoBe-Wake-Token": self._token},
                )
            if resp.status_code >= 300:
                logger.warning("[%s] Emote %r not delivered (HTTP %s)", self.name, emotion, resp.status_code)
        except Exception as exc:
            logger.warning("[%s] Emote %r not delivered: %s", self.name, emotion, exc)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Speak the message on the robot: synthesized TTS, or text announce.

        Skipped briefly after a successful TTS delivery: the gateway sends the
        reply text after ``play_tts`` and the robot must not speak it twice.
        TTS-first matters for pushes ("send it to bobe", cron): with the
        Hermes voice backend the robot has no realtime model to read text
        aloud, so a synthesized clip is the only spoken path. The text
        announce remains as fallback (and feeds the robot's UI/log).
        """
        if time.monotonic() < self._suppress_text_until:
            logger.debug("[%s] Suppressing text echo after TTS delivery", self.name)
            return SendResult(success=True, message_id="bobe-tts-audio")
        emotion, content = self._split_emotion_tag(content)
        if emotion:
            await self._post_emote(emotion)
        if not content.strip():
            return SendResult(success=True, message_id="bobe-emote-only")
        tts_result = await self._synthesize_and_speak(content)
        if tts_result is not None and tts_result.success:
            return tts_result
        return await _post_announcement(self._url, self._token, content)

    async def _synthesize_and_speak(self, content: str) -> Optional[SendResult]:
        """Synthesize ``content`` with the configured TTS and play it on the robot.

        Returns None when TTS is unavailable/failed so callers fall back to a
        plain text announce.
        """
        cleanup_paths: set = set()
        try:
            from tools.tts_tool import text_to_speech_tool, check_tts_requirements
            from gateway.platforms.base import build_auto_tts_output_path

            if not check_tts_requirements():
                return None
            speech_text = self.prepare_tts_text(content)
            if not speech_text:
                return None
            audio_path = build_auto_tts_output_path(self.platform)
            cleanup_paths.add(audio_path)
            import json as _json

            tts_result_str = await asyncio.to_thread(
                text_to_speech_tool, text=speech_text, output_path=audio_path
            )
            tts_data = _json.loads(tts_result_str)
            if not tts_data.get("success", True):
                return None
            produced = tts_data.get("file_path") or audio_path
            cleanup_paths.add(produced)
            result = await self._post_speech(produced)
            return result if result.success else None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("[%s] send() TTS synthesis failed (%s); announcing text", self.name, exc)
            return None
        finally:
            for path in cleanup_paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    async def _post_speech(self, audio_path: str) -> SendResult:
        """POST an audio file to the daemon's /v1/speak for robot playback."""
        try:
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
        except OSError as exc:
            return SendResult(success=False, error=f"Cannot read TTS audio: {exc}")
        content_type = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
        if not content_type.startswith("audio/"):
            content_type = "application/octet-stream"
        try:
            async with httpx.AsyncClient(timeout=SPEAK_TIMEOUT_S) as client:
                resp = await client.post(
                    self._speak_url,
                    content=audio_bytes,
                    headers={"X-BoBe-Wake-Token": self._token, "Content-Type": content_type},
                )
        except httpx.TimeoutException:
            return SendResult(success=False, error="Timeout sending speech to the BoBe wake daemon")
        except Exception as exc:
            return SendResult(success=False, error=f"BoBe wake daemon unreachable: {exc}")

        if resp.status_code < 300:
            return SendResult(success=True, message_id="bobe-speak")
        try:
            error = resp.json().get("error", "")
        except Exception:
            error = resp.text[:200]
        if resp.status_code == 409:
            return SendResult(success=False, error="Robot is not connected to the wake daemon")
        return SendResult(success=False, error=f"HTTP {resp.status_code}: {error}")

    async def play_tts(self, chat_id: str, audio_path: str, **kwargs) -> SendResult:
        """Deliver auto-TTS reply audio straight to the robot's speaker."""
        emotion, self._pending_emotion = self._pending_emotion, None
        if emotion:
            await self._post_emote(emotion)
        result = await self._post_speech(audio_path)
        if result.success:
            self._suppress_text_until = time.monotonic() + TTS_TEXT_SUPPRESS_S
        else:
            # Fall through silently: the gateway sends the reply text next and
            # send() will announce it, so the answer is spoken either way.
            logger.warning("[%s] TTS delivery failed (%s); falling back to text announce", self.name, result.error)
        return result

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        """Play an audio attachment through the robot's speaker."""
        result = await self._post_speech(audio_path)
        if not result.success and caption:
            return await _post_announcement(self._url, self._token, caption)
        return result

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """There is a single implicit channel: the robot's voice."""
        return {"name": "BoBe robot", "type": "dm"}


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
            "Messages to bobe are spoken aloud by a home robot, and voice "
            "questions heard by the robot arrive from this channel. "
            "Write short, plain spoken sentences — no markdown, no links, "
            "no code, nothing sensitive. OPTIONAL expression: you may start "
            "a reply with ONE tag like [emotion:amazed1] and the robot will "
            "perform that motion while speaking. Use it sparingly — only "
            "when the moment clearly calls for it (great news, a blunder, "
            "a warm greeting); most replies should have NO tag. Available: "
            "welcoming1 (greeting), loving1 (affection), amazed1 (wow), "
            "surprised1, thoughtful2 (pondering), understanding1 (nod), "
            "proud3 (success cheer), success1, oops1 (blunder), sad1, "
            "no1 (firm no), enthusiastic2 (good news)."
        ),
    )
