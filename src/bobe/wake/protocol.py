"""JSON control messages for the remote wake-word stream."""

from __future__ import annotations

import json
from typing import Any

from bobe.wake.phrases import WAKE_PHRASE


def hello_message(*, sample_rate: int, token: str | None) -> dict[str, Any]:
    """Build the robot handshake payload."""
    payload: dict[str, Any] = {
        "type": "hello",
        "client": "bobe",
        "sample_rate": sample_rate,
        "phrase": WAKE_PHRASE,
    }
    if token:
        payload["token"] = token
    return payload


def ready_message(*, engine: str, phrase: str = WAKE_PHRASE) -> dict[str, Any]:
    """Build the daemon ready acknowledgement."""
    return {
        "type": "ready",
        "engine": engine,
        "phrase": phrase,
    }


def wake_message(*, transcript: str, latency_ms: float, phrase: str = WAKE_PHRASE) -> dict[str, Any]:
    """Build a wake detection event."""
    return {
        "type": "wake",
        "phrase": phrase,
        "transcript": transcript,
        "latency_ms": round(latency_ms, 1),
    }


def stats_message(**fields: Any) -> dict[str, Any]:
    """Build a periodic debug stats payload."""
    return {"type": "stats", **fields}


def pause_message() -> dict[str, Any]:
    """Tell the daemon to ignore audio until resume."""
    return {"type": "pause"}


def resume_message() -> dict[str, Any]:
    """Tell the daemon to resume wake detection."""
    return {"type": "resume"}


def encode_json(message: dict[str, Any]) -> str:
    """Serialize a control message."""
    return json.dumps(message, separators=(",", ":"))


def parse_json(raw: str) -> dict[str, Any] | None:
    """Parse a JSON control message, returning None on failure."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
