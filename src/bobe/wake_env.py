"""Helpers for persisting remote wake settings into the app instance .env."""

from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlparse

from bobe.env_file import read_env_lines, upsert_env_keys, _read_lines_if_exists


REMOTE_WAKE_KEYS = (
    "BOBE_WAKE_BACKEND",
    "BOBE_WAKE_REMOTE_URL",
    "BOBE_WAKE_TOKEN",
    "BOBE_WAKE_GAIN",
)

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
    """Hostnames allowed when BOBE_WAKE_ALLOWED_HOSTS is unset (from packaged .env.example)."""
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
    """Return the configured allowlist of remote wake daemon hostnames."""
    raw = (os.getenv("BOBE_WAKE_ALLOWED_HOSTS") or "").strip()
    if raw:
        return frozenset(part.strip().casefold() for part in raw.split(",") if part.strip())
    return default_wake_allowed_hosts()


def is_wake_remote_host_allowed(hostname: str) -> bool:
    """Return whether hostname is on the wake daemon allowlist."""
    normalized = (hostname or "").strip().casefold()
    if not normalized:
        return False
    allowed = wake_allowed_hosts()
    return bool(allowed) and normalized in allowed


def upsert_wake_env_lines(
    lines: list[str],
    *,
    backend: str = "remote",
    remote_url: str,
    token: str,
    gain: float = 1.75,
) -> None:
    """Merge remote wake settings into env file lines."""
    upsert_env_keys(
        lines,
        {
            "BOBE_WAKE_BACKEND": backend,
            "BOBE_WAKE_REMOTE_URL": remote_url,
            "BOBE_WAKE_TOKEN": token,
            "BOBE_WAKE_GAIN": str(gain),
        },
    )


def persist_wake_env(
    instance_path: str | Path,
    *,
    backend: str = "remote",
    remote_url: str,
    token: str,
    gain: float = 1.75,
) -> Path:
    """Write remote wake settings to ``instance_path/.env``."""
    env_path = Path(instance_path) / ".env"
    lines = read_env_lines(env_path)
    upsert_wake_env_lines(
        lines,
        backend=backend,
        remote_url=remote_url,
        token=token,
        gain=gain,
    )
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    os.environ["BOBE_WAKE_BACKEND"] = backend
    os.environ["BOBE_WAKE_REMOTE_URL"] = remote_url
    os.environ["BOBE_WAKE_TOKEN"] = token
    os.environ["BOBE_WAKE_GAIN"] = str(gain)
    return env_path


def merge_packaged_wake_defaults(instance_path: str | Path) -> bool:
    """Copy missing remote wake keys from packaged ``.env.example`` into instance ``.env``."""
    example = Path(__file__).parent / ".env.example"
    if not example.exists():
        return False

    example_values: dict[str, str] = {}
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key in REMOTE_WAKE_KEYS and value.strip():
            example_values[key] = value.strip().strip('"').strip("'")

    if example_values.get("BOBE_WAKE_BACKEND") != "remote":
        return False
    if not example_values.get("BOBE_WAKE_REMOTE_URL"):
        return False

    env_path = Path(instance_path) / ".env"
    lines = _read_lines_if_exists(env_path) or []
    current: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        current[key] = value.strip().strip('"').strip("'")

    missing = {
        key: example_values[key]
        for key in REMOTE_WAKE_KEYS
        if not current.get(key) and example_values.get(key)
    }
    if not missing:
        return False

    upsert_env_keys(lines, missing)
    os.environ.update(missing)
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return True
