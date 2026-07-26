# ruff: noqa: D103

import threading
import subprocess
from pathlib import Path

import pytest

from bobe.wake_daemon import launcher as launcher_module
from bobe.wake_daemon.config import WakeDaemonConfig
from bobe.wake_daemon.launcher import ClaudeCodeLauncher, ClaudeCodeLaunchError, resolve_workdir


def test_resolve_workdir_defaults_under_repos(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    workdir = resolve_workdir(None)

    assert workdir == tmp_path / "repos" / "bobe-claude-code-workspace"


def test_resolve_workdir_rejects_path_outside_repos(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(ClaudeCodeLaunchError):
        resolve_workdir(str(tmp_path / "outside"))


def test_launcher_returns_disabled_when_not_enabled():
    launcher = ClaudeCodeLauncher(WakeDaemonConfig(token="wake-token"))

    assert launcher.launch() == {"ok": False, "error": "disabled"}


def test_launcher_opens_terminal_with_valid_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(launcher_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    calls = []

    def fake_runner(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    config = WakeDaemonConfig(
        token="wake-token",
        claude_code_launch_enabled=True,
        claude_code_workdir="voice-work",
        claude_code_bin="claude",
    )
    launcher = ClaudeCodeLauncher(config, runner=fake_runner)

    result = launcher.launch()

    assert result["ok"] is True
    assert result["workdir"] == str(tmp_path / "repos" / "voice-work")
    assert calls
    args, kwargs = calls[0]
    assert args[:3] == ["open", "-a", "Terminal"]
    script_path = Path(args[-1])
    script = script_path.read_text()
    assert "cd" in script
    assert "voice-work" in script
    assert "exec /usr/bin/claude" in script
    script_path.unlink()
    assert result["binary"] == "/usr/bin/claude"
    assert kwargs["check"] is True


def test_launcher_enforces_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(launcher_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    now = {"value": 10.0}

    config = WakeDaemonConfig(
        token="wake-token",
        claude_code_launch_enabled=True,
        claude_code_launch_cooldown_s=30.0,
    )

    def fake_runner(args, **kwargs):
        Path(args[-1]).unlink()
        return subprocess.CompletedProcess(args=args, returncode=0)

    launcher = ClaudeCodeLauncher(config, clock=lambda: now["value"], runner=fake_runner)

    assert launcher.launch()["ok"] is True
    now["value"] = 20.0
    result = launcher.launch()

    assert result["ok"] is False
    assert result["error"] == "cooldown"
    assert result["retry_after_s"] == 20.0


def test_launcher_cooldown_engages_while_launch_is_in_flight(tmp_path, monkeypatch):
    """A concurrent launch during the (slow) spawn window must not open a second Terminal."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(launcher_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    spawn_started = threading.Event()
    release_spawn = threading.Event()
    calls = []

    def blocking_runner(args, **kwargs):
        calls.append(args)
        spawn_started.set()
        assert release_spawn.wait(timeout=5)
        Path(args[-1]).unlink()
        return subprocess.CompletedProcess(args=args, returncode=0)

    config = WakeDaemonConfig(
        token="wake-token",
        claude_code_launch_enabled=True,
        claude_code_launch_cooldown_s=30.0,
    )
    launcher = ClaudeCodeLauncher(config, runner=blocking_runner)
    results: list[dict] = []
    worker = threading.Thread(target=lambda: results.append(launcher.launch()))
    worker.start()
    try:
        assert spawn_started.wait(timeout=5)
        concurrent = launcher.launch()
    finally:
        release_spawn.set()
        worker.join(timeout=5)

    assert concurrent["ok"] is False
    assert concurrent["error"] == "cooldown"
    assert results[0]["ok"] is True
    assert len(calls) == 1


def test_launcher_failed_launch_does_not_consume_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(launcher_module.shutil, "which", lambda binary: f"/usr/bin/{binary}")
    outcomes = iter(["fail", "ok"])

    def flaky_runner(args, **kwargs):
        Path(args[-1]).unlink()
        if next(outcomes) == "fail":
            raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="boom")
        return subprocess.CompletedProcess(args=args, returncode=0)

    config = WakeDaemonConfig(
        token="wake-token",
        claude_code_launch_enabled=True,
        claude_code_launch_cooldown_s=30.0,
    )
    launcher = ClaudeCodeLauncher(config, clock=lambda: 10.0, runner=flaky_runner)

    failed = launcher.launch()
    retried = launcher.launch()

    assert failed["ok"] is False
    assert failed["error"] == "launch_failed"
    assert retried["ok"] is True
