"""Stdlib-only helpers shared by the deploy/recover/configure scripts.

Keep this module free of third-party and bobe imports: the installer runs
configure_wake_remote_env.py with the system python3.
"""

from __future__ import annotations
import json
import urllib.request
from pathlib import Path


def _read_token(daemon_env: Path) -> str:
    for line in daemon_env.read_text(encoding="utf-8").splitlines():
        if line.startswith("BOBE_WAKE_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"BOBE_WAKE_TOKEN not found in {daemon_env}")


def _request(method: str, url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if payload is not None else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}
