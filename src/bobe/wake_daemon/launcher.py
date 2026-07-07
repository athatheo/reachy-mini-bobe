"""Mac-only launcher for confirmed Claude Code voice requests."""

from __future__ import annotations
import os
import time
import shlex
import shutil
import tempfile
import subprocess
from typing import Any, Callable
from pathlib import Path

from bobe.wake_daemon.config import DEFAULT_CLAUDE_CODE_WORKDIR, WakeDaemonConfig


class ClaudeCodeLaunchError(ValueError):
    """Raised when Claude Code launch settings are unsafe or invalid."""


class ClaudeCodeLauncher:
    """Launch Claude Code in Terminal with a narrow allowlist."""

    def __init__(
        self,
        config: WakeDaemonConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        """Initialize the launcher with injectable clock and command runner."""
        self._config = config
        self._clock = clock
        self._runner = runner
        self._last_launch_at: float | None = None

    def launch(self) -> dict[str, Any]:
        """Launch Claude Code in a Terminal window."""
        if not self._config.claude_code_launch_enabled:
            return {"ok": False, "error": "disabled"}

        cooldown = self._config.claude_code_launch_cooldown_s
        now = self._clock()
        if self._last_launch_at is not None and now - self._last_launch_at < cooldown:
            retry_after = max(0.0, cooldown - (now - self._last_launch_at))
            return {
                "ok": False,
                "error": "cooldown",
                "retry_after_s": round(retry_after, 1),
            }

        try:
            workdir = resolve_workdir(self._config.claude_code_workdir)
            binary = resolve_binary(self._config.claude_code_bin)
        except ClaudeCodeLaunchError as exc:
            return {"ok": False, "error": "invalid_config", "message": str(exc)}

        workdir.mkdir(parents=True, exist_ok=True)
        script_path = create_terminal_command_script(workdir=workdir, binary=binary)

        try:
            self._runner(
                [
                    "open",
                    "-a",
                    "Terminal",
                    str(script_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except FileNotFoundError:
            _remove_script(script_path)
            return {"ok": False, "error": "open_not_found"}
        except subprocess.CalledProcessError as exc:
            _remove_script(script_path)
            stderr = (exc.stderr or "").strip()
            return {"ok": False, "error": "launch_failed", "message": stderr or str(exc)}
        except subprocess.TimeoutExpired:
            _remove_script(script_path)
            return {"ok": False, "error": "launch_timeout"}

        self._last_launch_at = now
        return {
            "ok": True,
            "workdir": str(workdir),
            "binary": binary,
        }


def create_terminal_command_script(*, workdir: Path, binary: str) -> Path:
    """Create a temporary Terminal script for the allowlisted launch command."""
    fd, raw_path = tempfile.mkstemp(prefix="bobe-claude-code-", suffix=".command")
    script_path = Path(raw_path)
    script = f'#!/bin/zsh\nrm -f "$0"\ncd {shlex.quote(str(workdir))} || exit 1\nexec {shlex.quote(binary)}\n'
    with os.fdopen(fd, "w") as handle:
        handle.write(script)
    script_path.chmod(0o700)
    return script_path


def resolve_workdir(raw_workdir: str | None) -> Path:
    """Resolve and validate the configured Claude Code working directory."""
    workdir_value = (raw_workdir or DEFAULT_CLAUDE_CODE_WORKDIR).strip()
    if not workdir_value:
        raise ClaudeCodeLaunchError("BOBE_CLAUDE_CODE_WORKDIR must not be empty")

    raw_path = Path(os.path.expanduser(workdir_value))
    if not raw_path.is_absolute():
        raw_path = Path.home() / "repos" / raw_path
    return ensure_workdir_under_repos(raw_path)


def ensure_workdir_under_repos(path: Path) -> Path:
    """Return a resolved path only if it stays under ~/repos."""
    repos_root = (Path.home() / "repos").resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(repos_root)
    except ValueError as exc:
        raise ClaudeCodeLaunchError(f"workdir must stay under {repos_root}") from exc
    return resolved


def resolve_binary(raw_binary: str | None) -> str:
    """Validate the configured Claude Code binary without accepting shell text."""
    binary = (raw_binary or "claude").strip()
    if not binary:
        raise ClaudeCodeLaunchError("BOBE_CLAUDE_CODE_BIN must not be empty")
    if "\x00" in binary or "\n" in binary or "\r" in binary:
        raise ClaudeCodeLaunchError("BOBE_CLAUDE_CODE_BIN contains invalid control characters")

    binary_path = Path(os.path.expanduser(binary))
    if binary_path.is_absolute():
        resolved = binary_path.resolve(strict=False)
        if not resolved.exists():
            raise ClaudeCodeLaunchError(f"Claude Code binary not found: {resolved}")
        if not os.access(resolved, os.X_OK):
            raise ClaudeCodeLaunchError(f"Claude Code binary is not executable: {resolved}")
        return str(resolved)

    if "/" in binary:
        raise ClaudeCodeLaunchError("BOBE_CLAUDE_CODE_BIN must be a bare executable name or absolute path")
    resolved_binary = shutil.which(binary)
    if resolved_binary is None:
        raise ClaudeCodeLaunchError(f"Claude Code binary not found on PATH: {binary}")
    return str(Path(resolved_binary).resolve(strict=False))


def _remove_script(script_path: Path) -> None:
    script_path.unlink(missing_ok=True)
