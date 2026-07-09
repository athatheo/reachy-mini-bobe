"""Shared robot-side plumbing for talking to the Mac wake daemon's Claude Code API.

Used by :mod:`bobe.claude_code_launch` (one-shot Terminal launch) and
:mod:`bobe.claude_code_session` (managed ``claude -p`` sessions), which share
the same auth header, URL derivation, confirmation-phrase matching, and
JSON-over-HTTP error mapping.
"""

from __future__ import annotations
import re
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable


logger = logging.getLogger(__name__)

DEFAULT_CONFIRM_TTL_S = 45.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
AUTH_HEADER = "X-BoBe-Launch-Token"

_SPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCTUATION_RE = re.compile(r"^[\s\"'`.,!?;:]+|[\s\"'`.,!?;:]+$")


def transcript_matches_phrase(transcript: str | None, phrase: str) -> bool:
    """Return True only when the transcript is exactly ``phrase``, modulo ASR punctuation."""
    if transcript is None:
        return False
    normalized = _TRAILING_PUNCTUATION_RE.sub("", transcript.casefold())
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    return normalized == phrase


def derive_daemon_http_url(wake_url: str | None, path: str) -> str | None:
    """Derive an HTTP(S) daemon endpoint from the ws(s) wake stream URL."""
    if wake_url is None or not wake_url.strip():
        return None
    parsed = urllib.parse.urlparse(wake_url.strip())
    if parsed.scheme == "ws":
        scheme = "http"
    elif parsed.scheme == "wss":
        scheme = "https"
    else:
        return None
    if not parsed.netloc:
        return None
    return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", "", ""))


def request_daemon_json(
    opener: Callable[..., Any],
    *,
    url: str,
    token: str,
    method: str,
    payload: dict[str, Any] | None,
    timeout_s: float,
    log_label: str,
) -> dict[str, Any]:
    """Call a daemon endpoint and map transport failures to ``{"ok": False, ...}`` dicts."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            AUTH_HEADER: token,
        },
        method=method,
    )
    try:
        with opener(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return json_or_error(body, fallback={"ok": False, "error": f"http_{exc.code}"})
    except urllib.error.URLError as exc:
        logger.warning("%s endpoint unreachable: %s", log_label, exc)
        return {"ok": False, "error": "endpoint_unreachable"}
    except TimeoutError:
        return {"ok": False, "error": "endpoint_timeout"}
    return json_or_error(raw, fallback={"ok": False, "error": "bad_response"})


def json_or_error(raw: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON object body, returning ``fallback`` for anything else."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback
    return parsed if isinstance(parsed, dict) else fallback
