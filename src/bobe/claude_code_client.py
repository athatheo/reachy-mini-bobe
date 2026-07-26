"""Shared robot-side plumbing for talking to the Mac wake daemon's Claude Code API.

Used by :mod:`bobe.claude_code_launch` (one-shot Terminal launch) and
:mod:`bobe.claude_code_session` (managed ``claude -p`` sessions), which share
the same auth header, URL derivation, confirmation-phrase matching, and
JSON-over-HTTP error mapping.
"""

from __future__ import annotations
import json
import logging
import functools
import http.client
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from bobe.wake.phrases import normalize_transcript


logger = logging.getLogger(__name__)

DEFAULT_CONFIRM_TTL_S = 45.0
DEFAULT_REQUEST_TIMEOUT_S = 10.0
AUTH_HEADER = "X-BoBe-Launch-Token"

# Common Whisper/gpt-4o-transcribe mishears (keep tight, mirroring
# WAKE_PHRASE_ASR_VARIANTS in bobe.wake.phrases).
_CLAUDE_ASR_VARIANTS: tuple[str, ...] = ("cloud", "clod", "clawed", "claud")
_CONFIRM_ASR_VARIANTS: tuple[str, ...] = ("confirmed",)


@functools.lru_cache(maxsize=8)
def _phrase_variants(phrase: str) -> frozenset[str]:
    """Return the accepted normalized spellings of a confirmation phrase."""
    base = normalize_transcript(phrase)
    if not base:
        return frozenset()
    variants = {base}
    for mishear in _CLAUDE_ASR_VARIANTS:
        variants.add(base.replace("claude", mishear))
    for confirmed in _CONFIRM_ASR_VARIANTS:
        variants.update(
            f"{confirmed} {variant.removeprefix('confirm ')}"
            for variant in tuple(variants)
            if variant.startswith("confirm ")
        )
    return frozenset(variants)


def transcript_matches_phrase(transcript: str | None, phrase: str) -> bool:
    """Return True only when the transcript is exactly ``phrase``, modulo ASR noise.

    Both sides are normalized like the wake path (internal punctuation becomes
    spaces), and a small variant list absorbs the well-known ``claude`` ->
    ``cloud``/``clod`` and ``confirm`` -> ``confirmed`` mishears.
    """
    if transcript is None:
        return False
    normalized = normalize_transcript(transcript)
    if not normalized:
        return False
    return normalized in _phrase_variants(phrase)


def transcript_attempts_confirmation(transcript: str | None) -> bool:
    """Return True when a transcript looks like a (possibly garbled) confirmation attempt."""
    if transcript is None:
        return False
    normalized = normalize_transcript(transcript)
    if not normalized:
        return False
    return normalized.split(" ", 1)[0] in {"confirm", *_CONFIRM_ASR_VARIANTS}


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
    except (OSError, http.client.HTTPException) as exc:
        # Body-read failures (ConnectionResetError, IncompleteRead, ...) are not
        # wrapped in URLError by urllib; they must not escape into the realtime
        # event loop, so map them like the other transport errors.
        logger.warning("%s endpoint request failed: %s", log_label, exc)
        return {"ok": False, "error": "endpoint_error"}
    return json_or_error(raw, fallback={"ok": False, "error": "bad_response"})


def json_or_error(raw: str, *, fallback: dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON object body, returning ``fallback`` for anything else."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return fallback
    return parsed if isinstance(parsed, dict) else fallback
