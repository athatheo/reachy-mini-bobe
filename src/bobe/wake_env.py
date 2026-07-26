"""Helpers for persisting remote wake settings into the app instance .env."""

from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlparse

from bobe.env_file import (
    ENV_FILE_LOCK,
    read_env_lines,
    upsert_env_keys,
    write_env_lines,
    _read_lines_if_exists,
)


REMOTE_WAKE_KEYS = (
    "BOBE_WAKE_BACKEND",
    "BOBE_WAKE_REMOTE_URL",
    "BOBE_WAKE_TOKEN",
    "BOBE_WAKE_GAIN",
    "BOBE_WAKE_ALLOWED_HOSTS",
)

# BOBE_WAKE_ALLOWED_HOSTS is intentionally never seeded into the instance .env:
# a pinned copy goes stale when the Mac's IP changes and would override a
# republished .env.example forever. wake_allowed_hosts() instead unions the
# packaged defaults and the configured remote URL host at read time, so the
# value is only persisted when the user explicitly submits it via /wake-config.
_SEEDABLE_WAKE_KEYS = tuple(key for key in REMOTE_WAKE_KEYS if key != "BOBE_WAKE_ALLOWED_HOSTS")

_PACKAGED_ENV_EXAMPLE = Path(__file__).parent / ".env.example"


def _hostname_from_ws_url(url: str) -> str | None:
    normalized = (url or "").strip()
    if not normalized:
        return None
    if normalized.startswith("ws://"):
        normalized = "http://" + normalized[5:]
    elif normalized.startswith("wss://"):
        normalized = "https://" + normalized[6:]
    hostname = urlparse(normalized).hostname
    return hostname.casefold() if hostname else None


def default_wake_allowed_hosts() -> frozenset[str]:
    """Hostnames from the packaged .env.example, always part of the effective allowlist."""
    if not _PACKAGED_ENV_EXAMPLE.exists():
        return frozenset()
    hosts: set[str] = set()
    for line in _PACKAGED_ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        value = value.strip().strip('"').strip("'")
        if key == "BOBE_WAKE_REMOTE_URL":
            host = _hostname_from_ws_url(value)
            if host:
                hosts.add(host)
        elif key == "BOBE_WAKE_ALLOWED_HOSTS":
            for part in value.split(","):
                normalized = part.strip().casefold()
                if normalized:
                    hosts.add(normalized)
    return frozenset(hosts)


def wake_allowed_hosts() -> frozenset[str]:
    """Return the effective allowlist of remote wake daemon hostnames.

    The union of the configured BOBE_WAKE_ALLOWED_HOSTS, the packaged
    .env.example defaults, and the host of the currently configured
    BOBE_WAKE_REMOTE_URL, so a stale configured allowlist can never lock out
    either recovery path (a republished app or the active daemon URL).
    """
    hosts: set[str] = set(default_wake_allowed_hosts())
    raw = (os.getenv("BOBE_WAKE_ALLOWED_HOSTS") or "").strip()
    if raw:
        hosts.update(part.strip().casefold() for part in raw.split(",") if part.strip())
    url_host = _hostname_from_ws_url(os.getenv("BOBE_WAKE_REMOTE_URL") or "")
    if url_host:
        hosts.add(url_host)
    return frozenset(hosts)


def is_wake_remote_host_allowed(hostname: str) -> bool:
    """Return whether hostname is on the wake daemon allowlist."""
    normalized = (hostname or "").strip().casefold()
    if not normalized:
        return False
    allowed = wake_allowed_hosts()
    return bool(allowed) and normalized in allowed


def _allowed_hosts_from_lines(lines: list[str]) -> set[str]:
    """Parse the BOBE_WAKE_ALLOWED_HOSTS assignment from env file lines."""
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("BOBE_WAKE_ALLOWED_HOSTS="):
            continue
        value = stripped.partition("=")[2].strip().strip('"').strip("'")
        return {part.strip().casefold() for part in value.split(",") if part.strip()}
    return set()


def upsert_wake_env_lines(
    lines: list[str],
    *,
    backend: str | None = "remote",
    remote_url: str | None = None,
    token: str | None = None,
    gain: float | None = None,
    allowed_hosts: str | None = None,
) -> None:
    """Merge remote wake settings into env file lines.

    ``None`` values are left out entirely so existing entries are preserved.
    """
    values: dict[str, str] = {}
    if backend is not None:
        values["BOBE_WAKE_BACKEND"] = backend
    if remote_url is not None:
        values["BOBE_WAKE_REMOTE_URL"] = remote_url
    if token is not None:
        values["BOBE_WAKE_TOKEN"] = token
    if gain is not None:
        values["BOBE_WAKE_GAIN"] = str(gain)
    if allowed_hosts is not None:
        values["BOBE_WAKE_ALLOWED_HOSTS"] = allowed_hosts
    upsert_env_keys(lines, values)


def persist_wake_env(
    instance_path: str | Path,
    *,
    backend: str | None = "remote",
    remote_url: str | None = None,
    token: str | None = None,
    gain: float | None = None,
) -> Path:
    """Write remote wake settings to ``instance_path/.env``.

    ``None`` keeps whatever is currently configured (tuned gain, daemon URL)
    instead of resetting it to a default. When a new remote URL is supplied,
    its just-validated hostname is merged into the persisted allowlist so the
    stored BOBE_WAKE_ALLOWED_HOSTS never silently reverts.
    """
    env_path = Path(instance_path) / ".env"
    with ENV_FILE_LOCK:
        lines = read_env_lines(env_path)
        allowed_hosts: str | None = None
        if remote_url is not None:
            host = _hostname_from_ws_url(remote_url)
            if host:
                merged = set(wake_allowed_hosts()) | _allowed_hosts_from_lines(lines) | {host}
                allowed_hosts = ",".join(sorted(merged))
        upsert_wake_env_lines(
            lines,
            backend=backend,
            remote_url=remote_url,
            token=token,
            gain=gain,
            allowed_hosts=allowed_hosts,
        )
        write_env_lines(env_path, lines)

    env_updates = {
        "BOBE_WAKE_BACKEND": backend,
        "BOBE_WAKE_REMOTE_URL": remote_url,
        "BOBE_WAKE_TOKEN": token,
        "BOBE_WAKE_GAIN": str(gain) if gain is not None else None,
        "BOBE_WAKE_ALLOWED_HOSTS": allowed_hosts,
    }
    for key, value in env_updates.items():
        if value is not None:
            os.environ[key] = value
    return env_path


def merge_packaged_wake_defaults(instance_path: str | Path) -> bool:
    """Copy missing remote wake keys from packaged ``.env.example`` into instance ``.env``.

    BOBE_WAKE_ALLOWED_HOSTS is never seeded; see ``_SEEDABLE_WAKE_KEYS``.
    """
    example = Path(__file__).parent / ".env.example"
    if not example.exists():
        return False

    example_values: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key in _SEEDABLE_WAKE_KEYS and value.strip():
            example_values[key] = value.strip().strip('"').strip("'")

    if example_values.get("BOBE_WAKE_BACKEND") != "remote":
        return False
    if not example_values.get("BOBE_WAKE_REMOTE_URL"):
        return False

    env_path = Path(instance_path) / ".env"
    with ENV_FILE_LOCK:
        lines = _read_lines_if_exists(env_path) or []
        current: dict[str, str] = {}
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            current[key] = value.strip().strip('"').strip("'")

        # Only seed keys with no configured value anywhere: an already-tuned
        # value (instance .env or live environment) must never be reset to
        # the packaged example defaults.
        missing = {
            key: example_values[key]
            for key in _SEEDABLE_WAKE_KEYS
            if not current.get(key)
            and not (os.getenv(key) or "").strip()
            and example_values.get(key)
        }
        if not missing:
            return False

        upsert_env_keys(lines, missing)
        os.environ.update(missing)
        write_env_lines(env_path, lines)
    return True
