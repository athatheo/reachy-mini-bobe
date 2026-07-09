"""JSON control messages for the remote wake-word stream."""

from __future__ import annotations

import json
from typing import Any

from bobe.wake.phrases import WAKE_PHRASE


def hello_message(*, sample_rate: int, token: str | None, phrase: str = WAKE_PHRASE) -> dict[str, Any]:
    """Build the robot handshake payload."""
    payload: dict[str, Any] = {
        "type": "hello",
        "client": "bobe",
        "sample_rate": sample_rate,
        "phrase": phrase,
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


def sleep_message(*, transcript: str, latency_ms: float) -> dict[str, Any]:
    """Build a sleep detection event."""
    return {
        "type": "sleep",
        "transcript": transcript,
        "latency_ms": round(latency_ms, 1),
    }


def listen_message(
    *,
    mode: str,
    sleep_phrases: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Tell the daemon which phrase class to listen for."""
    payload: dict[str, Any] = {"type": "listen", "mode": mode}
    if mode == "sleep" and sleep_phrases:
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
