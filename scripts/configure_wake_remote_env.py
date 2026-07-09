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
    mac_host: str,
    port: int,
    token: str,
    gain: float,
) -> None:
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    else:
        example = env_path.parent / ".env.example"
        if example.exists():
            lines = example.read_text(encoding="utf-8").splitlines()
        else:
            lines = []

    remote_url = _remote_url(mac_host, port)
    _upsert(lines, "BOBE_WAKE_BACKEND", "remote")
    _upsert(lines, "BOBE_WAKE_REMOTE_URL", remote_url)
    _upsert(lines, "BOBE_WAKE_TOKEN", token)
    _upsert(lines, "BOBE_WAKE_GAIN", str(gain))

    env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Updated {env_path}")
    print(f"  BOBE_WAKE_REMOTE_URL={remote_url}")


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
    parser.add_argument("--mac-host", default=_detect_host("Mac"), help="Mac Bonjour hostname")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--gain", type=float, default=1.75)
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
