"""Managed Claude Code sessions for the Mac wake daemon."""

from __future__ import annotations
import os
import json
import time
import uuid
import signal
import threading
import subprocess
from typing import Any, Callable
from dataclasses import dataclass

from bobe.wake_daemon.config import WakeDaemonConfig
from bobe.wake_daemon.launcher import ClaudeCodeLaunchError, resolve_binary, resolve_workdir


ALLOWED_PERMISSION_MODES = {"acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan"}


@dataclass(frozen=True)
class ClaudeCodeSessionSettings:
    """Validated settings for a managed Claude Code session."""

    workdir: str
    binary: str
    permission_mode: str
    command_timeout_s: float
    output_limit_chars: int


class ClaudeCodeSessionManager:
    """Run follow-up Claude Code commands under one daemon-owned session id."""

    def __init__(
        self,
        config: WakeDaemonConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        popen_factory: Callable[..., subprocess.Popen[str]] = subprocess.Popen,
    ) -> None:
        """Initialize the session manager with injectable clock/process factory."""
        self._config = config
        self._clock = clock
        self._popen_factory = popen_factory
        self._lock = threading.Lock()
        self._session_id: str | None = None
        self._started_at: float | None = None
        self._last_activity_at: float | None = None
        self._last_result: dict[str, Any] | None = None
        self._active_process: subprocess.Popen[str] | None = None

    def start(self) -> dict[str, Any]:
        """Create a daemon-owned Claude Code session id without sending a prompt."""
        try:
            settings = self._settings()
        except ClaudeCodeLaunchError as exc:
            return {"ok": False, "error": "invalid_config", "message": str(exc)}

        with self._lock:
            if self._session_id is None:
                now = self._clock()
                self._session_id = str(uuid.uuid4())
                self._started_at = now
                self._last_activity_at = now
                self._last_result = None
            return {
                "ok": True,
                "session_id": self._session_id,
                "workdir": settings.workdir,
                "running": self._active_process is not None,
            }

    def send(self, command: str) -> dict[str, Any]:
        """Send one prompt to Claude Code using the managed session id."""
        clean_command = command.strip()
        if not clean_command:
            return {"ok": False, "error": "empty_command"}

        start_result = self.start()
        if not start_result.get("ok"):
            return start_result
        settings = self._settings()

        with self._lock:
            if self._active_process is not None:
                return {"ok": False, "error": "busy"}
            session_id = self._session_id
            assert session_id is not None
            args = [
                settings.binary,
                "-p",
                "--session-id",
                session_id,
                "--output-format",
                "json",
                "--permission-mode",
                settings.permission_mode,
                clean_command,
            ]
            process = self._popen_factory(
                args,
                cwd=settings.workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            self._active_process = process
            self._last_activity_at = self._clock()

        try:
            stdout, stderr = process.communicate(timeout=settings.command_timeout_s)
            result = _result_from_process(
                process.returncode,
                stdout,
                stderr,
                limit=settings.output_limit_chars,
            )
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            stdout, stderr = process.communicate()
            result = {
                "ok": False,
                "error": "timeout",
                "stdout": _limit_text(stdout or "", settings.output_limit_chars),
                "stderr": _limit_text(stderr or "", settings.output_limit_chars),
            }
        finally:
            with self._lock:
                if self._active_process is process:
                    self._active_process = None
                self._last_activity_at = self._clock()

        with self._lock:
            self._last_result = result
            return {
                **result,
                "session_id": self._session_id,
                "workdir": settings.workdir,
            }

    def status(self) -> dict[str, Any]:
        """Return current managed session status."""
        with self._lock:
            return {
                "ok": True,
                "active": self._session_id is not None,
                "session_id": self._session_id,
                "running": self._active_process is not None,
                "started_at": self._started_at,
                "last_activity_at": self._last_activity_at,
                "last_result": self._last_result,
            }

    def stop(self) -> dict[str, Any]:
        """Terminate any active command and clear the managed session id."""
        with self._lock:
            process = self._active_process
            session_id = self._session_id
            self._session_id = None
            self._started_at = None
            self._last_activity_at = self._clock()
            self._last_result = None
            self._active_process = None

        if process is not None:
            _terminate_process_group(process)
        return {"ok": True, "stopped_session_id": session_id, "terminated_process": process is not None}

    def _settings(self) -> ClaudeCodeSessionSettings:
        workdir = resolve_workdir(self._config.claude_code_workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        binary = resolve_binary(self._config.claude_code_bin)
        permission_mode = self._config.claude_code_permission_mode
        if permission_mode not in ALLOWED_PERMISSION_MODES:
            raise ClaudeCodeLaunchError(f"invalid Claude Code permission mode: {permission_mode}")
        return ClaudeCodeSessionSettings(
            workdir=str(workdir),
            binary=binary,
            permission_mode=permission_mode,
            command_timeout_s=self._config.claude_code_command_timeout_s,
            output_limit_chars=self._config.claude_code_output_limit_chars,
        )


def _result_from_process(returncode: int | None, stdout: str, stderr: str, *, limit: int) -> dict[str, Any]:
    parsed = _parse_json(stdout)
    output = _extract_output(parsed, stdout)
    ok = returncode == 0
    return {
        "ok": ok,
        "returncode": returncode,
        "output": _limit_text(output, limit),
        "stdout": _limit_text(stdout, limit),
        "stderr": _limit_text(stderr, limit),
        "parsed_json": parsed is not None,
        **({} if ok else {"error": "claude_failed"}),
    }


def _parse_json(raw: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_output(parsed: dict[str, Any] | None, raw: str) -> str:
    if parsed is None:
        return raw.strip()
    for key in ("result", "response", "message", "text"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return raw.strip()


def _limit_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    try:
        if process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if process.pid:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
