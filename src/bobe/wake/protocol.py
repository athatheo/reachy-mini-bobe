"""JSON control messages for the remote wake-word stream."""

from __future__ import annotations
import json
from typing import Any

from bobe.wake.phrases import WAKE_PHRASE


# Declared in both handshake messages so either side can key compat behavior
# off the peer's version instead of guessing. Peers ignore unknown JSON
# fields, so adding it is wire-compatible with older builds (which simply
# never send it).
PROTOCOL_VERSION = 1

MSG_HELLO = "hello"
MSG_READY = "ready"
MSG_WAKE = "wake"
MSG_SLEEP = "sleep"
MSG_STATS = "stats"
MSG_LISTEN = "listen"
MSG_ANNOUNCE = "announce"
MSG_SPEAK = "speak"
MSG_EMOTE = "emote"
MSG_PRESENCE = "presence"

# WebSocket close codes used by the daemon handshake (RFC 6455).
CLOSE_UNSUPPORTED_DATA = 1003
CLOSE_POLICY_VIOLATION = 1008


def hello_message(*, sample_rate: int, token: str | None, phrase: str = WAKE_PHRASE) -> dict[str, Any]:
    """Build the robot handshake payload."""
    payload: dict[str, Any] = {
        "type": MSG_HELLO,
        "client": "bobe",
        "version": PROTOCOL_VERSION,
        "sample_rate": sample_rate,
        "phrase": phrase,
    }
    if token:
        payload["token"] = token
    return payload


def ready_message(*, engine: str, phrase: str = WAKE_PHRASE) -> dict[str, Any]:
    """Build the daemon ready acknowledgement."""
    return {
        "type": MSG_READY,
        "version": PROTOCOL_VERSION,
        "engine": engine,
        "phrase": phrase,
    }


def wake_message(*, transcript: str, latency_ms: float, phrase: str = WAKE_PHRASE) -> dict[str, Any]:
    """Build a wake detection event."""
    return {
        "type": MSG_WAKE,
        "phrase": phrase,
        "transcript": transcript,
        "latency_ms": round(latency_ms, 1),
    }


def stats_message(**fields: Any) -> dict[str, Any]:
    """Build a periodic debug stats payload."""
    return {"type": MSG_STATS, **fields}


def sleep_message(*, transcript: str, latency_ms: float) -> dict[str, Any]:
    """Build a sleep detection event."""
    return {
        "type": MSG_SLEEP,
        "transcript": transcript,
        "latency_ms": round(latency_ms, 1),
    }


def announce_message(*, text: str) -> dict[str, Any]:
    """Build a daemon-to-robot announcement the robot should speak."""
    return {
        "type": MSG_ANNOUNCE,
        "text": text,
    }


def presence_message(*, jpeg_b64: str | None = None) -> dict[str, Any]:
    """Build a robot-to-daemon presence payload.

    With ``jpeg_b64`` the daemon runs face detection on the snapshot; without
    it the message is a bare (already-detected) sighting report.
    """
    payload: dict[str, Any] = {"type": MSG_PRESENCE}
    if jpeg_b64 is not None:
        payload["jpeg_b64"] = jpeg_b64
    return payload


def emote_message(*, emotion: str) -> dict[str, Any]:
    """Build a daemon-to-robot request to play a recorded emotion move."""
    return {
        "type": MSG_EMOTE,
        "emotion": emotion,
    }


def speak_message(*, clip_id: str, seq: int, pcm_b64: str, rate: int, last: bool) -> dict[str, Any]:
    """Build one chunk of daemon-to-robot speech audio (mono s16le PCM).

    Clips are chunked so a long reply never exceeds websocket message limits;
    the robot reassembles by ``id`` and plays the clip once ``last`` arrives.
    """
    return {
        "type": MSG_SPEAK,
        "id": clip_id,
        "seq": seq,
        "pcm_b64": pcm_b64,
        "rate": rate,
        "last": last,
    }


def listen_message(
    *,
    mode: str,
    sleep_phrases: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Tell the daemon which phrase class to listen for.

    ``sleep`` and ``converse`` both carry sleep phrases: converse mode keeps
    sleep-phrase preemption while additionally capturing utterances.
    """
    payload: dict[str, Any] = {"type": MSG_LISTEN, "mode": mode}
    if mode in ("sleep", "converse") and sleep_phrases:
        payload["sleep_phrases"] = list(sleep_phrases)
    return payload


def parse_json(raw: str) -> dict[str, Any] | None:
    """Parse a JSON control message, returning None on failure."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
