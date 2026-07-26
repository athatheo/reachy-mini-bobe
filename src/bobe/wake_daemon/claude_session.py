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
        self._spawning = False
        self._waiter_thread: threading.Thread | None = None

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
                "running": self._active_process is not None or self._spawning,
            }

    def send(self, command: str) -> dict[str, Any]:
        """Accept one prompt for Claude Code and run it in the background.

        Returns promptly with ``{"ok": True, "accepted": True, ...}``; the
        command's result is published to :meth:`status` as ``last_result`` when
        the subprocess finishes, so callers never block behind a long command.
        """
        clean_command = command.strip()
        if not clean_command:
            return {"ok": False, "error": "empty_command"}

        try:
            settings = self._settings()
        except ClaudeCodeLaunchError as exc:
            return {"ok": False, "error": "invalid_config", "message": str(exc)}

        with self._lock:
            if self._active_process is not None or self._spawning:
                return {"ok": False, "error": "busy"}
            # Resolve the session id under the same lock acquisition as the
            # busy check: a concurrent stop() could otherwise clear it between
            # a start() call and this critical section (never assert on it).
            if self._session_id is None:
                now = self._clock()
                self._session_id = str(uuid.uuid4())
                self._started_at = now
                self._last_result = None
            session_id = self._session_id
            self._spawning = True
            self._last_activity_at = self._clock()

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
        # Spawn outside the lock: fork/exec of the claude CLI can take seconds,
        # and status() (served from the daemon event loop) must never wait on a
        # lock held across it.
        try:
            process = self._popen_factory(
                args,
                cwd=settings.workdir,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception:
            with self._lock:
                self._spawning = False
            raise

        waiter = threading.Thread(
            target=self._await_result,
            args=(process, session_id, settings),
            name="claude-code-waiter",
            daemon=True,
        )
        with self._lock:
            self._spawning = False
            stopped_while_spawning = self._session_id != session_id
            if not stopped_while_spawning:
                self._active_process = process
                self._waiter_thread = waiter
                self._last_activity_at = self._clock()
        if stopped_while_spawning:
            # stop() raced the spawn; don't run the command under a dead session.
            _terminate_process_group(process)
            return {"ok": False, "error": "no_session"}

        waiter.start()
        return {
            "ok": True,
            "accepted": True,
            "running": True,
            "session_id": session_id,
            "workdir": settings.workdir,
        }

    def _await_result(
        self,
        process: subprocess.Popen[str],
        session_id: str,
        settings: ClaudeCodeSessionSettings,
    ) -> None:
        """Wait for a spawned command and publish its result to status()."""
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
        except Exception as exc:  # communicate() can fail after a kill
            result = {"ok": False, "error": "command_failed", "message": str(exc)}

        with self._lock:
            if self._active_process is process:
                self._active_process = None
            self._last_activity_at = self._clock()
            # Generation check: stop()/restart may have replaced the session
            # while the command ran; never resurrect its result into the new
            # session's state (or attribute it to the wrong session id).
            if self._session_id == session_id:
                self._last_result = {
                    **result,
                    "session_id": session_id,
                    "workdir": settings.workdir,
                }

    def status(self) -> dict[str, Any]:
        """Return current managed session status (never blocks on a running command)."""
        with self._lock:
            return {
                "ok": True,
                "active": self._session_id is not None,
                "session_id": self._session_id,
                "running": self._active_process is not None or self._spawning,
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

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        """Terminate any running command and reap its waiter thread at daemon exit."""
        self.stop()
        with self._lock:
            waiter = self._waiter_thread
            self._waiter_thread = None
        if waiter is not None and waiter.is_alive():
            waiter.join(timeout=timeout_s)

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
        try:
            process.wait(timeout=3)  # reap: don't leave a zombie behind SIGKILL
        except subprocess.TimeoutExpired:
            pass
    except ProcessLookupError:
        pass
