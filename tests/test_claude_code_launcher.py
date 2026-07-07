# ruff: noqa: D103

import subprocess

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
    assert args[0] == "osascript"
    assert "cd" in args[-1]
    assert "voice-work" in args[-1]
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
    launcher = ClaudeCodeLauncher(
        config,
        clock=lambda: now["value"],
        runner=lambda args, **kwargs: subprocess.CompletedProcess(args=args, returncode=0),
    )

    assert launcher.launch()["ok"] is True
    now["value"] = 20.0
    result = launcher.launch()

    assert result["ok"] is False
    assert result["error"] == "cooldown"
    assert result["retry_after_s"] == 20.0
