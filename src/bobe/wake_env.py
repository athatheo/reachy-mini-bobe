"""Helpers for persisting remote wake settings into the app instance .env."""

from __future__ import annotations

import os
from pathlib import Path

REMOTE_WAKE_KEYS = (
    "BOBE_WAKE_BACKEND",
    "BOBE_WAKE_REMOTE_URL",
    "BOBE_WAKE_TOKEN",
    "BOBE_WAKE_GAIN",
)


def _upsert_line(lines: list[str], key: str, value: str) -> None:
    replacement = f"{key}={value}"
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[index] = replacement
            return
    lines.append(replacement)


def upsert_wake_env_lines(
    lines: list[str],
    *,
    backend: str = "remote",
    remote_url: str,
    token: str,
    gain: float = 1.75,
) -> None:
    """Merge remote wake settings into env file lines."""
    _upsert_line(lines, "BOBE_WAKE_BACKEND", backend)
    _upsert_line(lines, "BOBE_WAKE_REMOTE_URL", remote_url)
    _upsert_line(lines, "BOBE_WAKE_TOKEN", token)
    _upsert_line(lines, "BOBE_WAKE_GAIN", str(gain))


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
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        lines = []

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
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    current: dict[str, str] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        current[key] = value.strip().strip('"').strip("'")

    changed = False
    for key in REMOTE_WAKE_KEYS:
        if current.get(key):
            continue
        value = example_values.get(key)
        if not value:
            continue
        _upsert_line(lines, key, value)
        os.environ[key] = value
        changed = True

    if changed:
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return changed
