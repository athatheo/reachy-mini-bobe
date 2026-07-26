#!/usr/bin/env python3
"""Merge remote wake settings into a BoBe .env file."""

from __future__ import annotations
import socket
import argparse
from pathlib import Path


def _detect_host(default: str) -> str:
    try:
        return socket.gethostname().split(".")[0] or default
    except Exception:
        return default


def _read_token(daemon_env: Path) -> str:
    for line in daemon_env.read_text(encoding="utf-8").splitlines():
        if line.startswith("BOBE_WAKE_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"BOBE_WAKE_TOKEN not found in {daemon_env}")


def _upsert(lines: list[str], key: str, value: str) -> None:
    replacement = f"{key}={value}"
    for index, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[index] = replacement
            return
    lines.append(replacement)


def _existing_value(lines: list[str], key: str) -> str | None:
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            value = stripped.partition("=")[2].strip().strip('"').strip("'")
            return value or None
    return None


def _template_lines(example: Path) -> list[str]:
    """Template lines minus empty KEY= placeholders (never bake empty values)."""
    lines: list[str] = []
    for line in example.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            value = stripped.partition("=")[2].strip().strip('"').strip("'")
            if not value:
                continue
        lines.append(line)
    return lines


def _remote_url(mac_host: str, port: int) -> str:
    host = mac_host.strip()
    if "." in host and not host.endswith(".local"):
        return f"ws://{host}:{port}/v1/stream"
    if not host.endswith(".local"):
        host = f"{host}.local"
    return f"ws://{host}:{port}/v1/stream"


def configure_env(
    env_path: Path,
    *,
    mac_host: str | None,
    port: int,
    token: str,
    gain: float | None,
) -> None:
    """Merge remote wake settings into env_path; None keeps the current values."""
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        example = env_path.parent / ".env.example"
        if example.exists():
            lines = _template_lines(example)
        else:
            lines = []

    existing_url = _existing_value(lines, "BOBE_WAKE_REMOTE_URL")
    remote_url: str | None = None
    if mac_host:
        remote_url = _remote_url(mac_host, port)
    elif not existing_url:
        remote_url = _remote_url(_detect_host("Mac"), port)

    _upsert(lines, "BOBE_WAKE_BACKEND", "remote")
    if remote_url is not None:
        _upsert(lines, "BOBE_WAKE_REMOTE_URL", remote_url)
    _upsert(lines, "BOBE_WAKE_TOKEN", token)
    if gain is not None:
        _upsert(lines, "BOBE_WAKE_GAIN", str(gain))

    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Updated {env_path}")
    print(f"  BOBE_WAKE_REMOTE_URL={remote_url or existing_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Configure BoBe for Mac remote wake")
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Target .env file (default: ./ .env)",
    )
    parser.add_argument(
        "--daemon-env",
        type=Path,
        default=Path("config/wake-daemon.env"),
        help="Mac wake daemon env file with BOBE_WAKE_TOKEN",
    )
    parser.add_argument(
        "--mac-host",
        default=None,
        help="Mac Bonjour hostname (default: keep the existing BOBE_WAKE_REMOTE_URL, or detect)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--gain",
        type=float,
        default=None,
        help="Wake mic gain (default: keep the existing BOBE_WAKE_GAIN)",
    )
    args = parser.parse_args()

    token = _read_token(args.daemon_env)
    configure_env(
        args.env,
        mac_host=args.mac_host,
        port=args.port,
        token=token,
        gain=args.gain,
    )


if __name__ == "__main__":
    main()
