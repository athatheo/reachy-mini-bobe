# ruff: noqa: D101,D102,D103,D107

import json
import subprocess

from bobe.wake_daemon import claude_session as session_module
from bobe.wake_daemon.config import WakeDaemonConfig
from bobe.wake_daemon.claude_session import ClaudeCodeSessionManager


class FakeProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.pid = 12345
        self.terminated = False

    def communicate(self, timeout=None):
        return self._stdout, self._stderr

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.returncode


def test_session_manager_starts_without_running_claude(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    calls = []
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = manager.start()

    assert result["ok"] is True
    assert result["session_id"]
    assert result["running"] is False
    assert calls == []


def test_session_manager_sends_command_with_session_id(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    calls = []

    def fake_popen(args, **kwargs):
        calls.append((args, kwargs))
        return FakeProcess(json.dumps({"result": "Tests passed"}))

    config = WakeDaemonConfig(
        token="wake-token",
        claude_code_workdir="voice-work",
        claude_code_permission_mode="plan",
    )
    manager = ClaudeCodeSessionManager(config, popen_factory=fake_popen)

    result = manager.send("run the focused tests")

    assert result["ok"] is True
    assert result["output"] == "Tests passed"
    args, kwargs = calls[0]
    assert args[0] == "/usr/local/bin/claude"
    assert "-p" in args
    assert "--session-id" in args
    assert "--output-format" in args
    assert "json" in args
    assert "--permission-mode" in args
    assert "plan" in args
    assert args[-1] == "run the focused tests"
    assert kwargs["cwd"] == str(tmp_path / "repos" / "voice-work")


def test_session_manager_reports_failed_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: FakeProcess("", "permission needed", returncode=1),
    )

    result = manager.send("edit a file")

    assert result["ok"] is False
    assert result["error"] == "claude_failed"
    assert result["stderr"] == "permission needed"


def test_session_manager_does_not_return_unbounded_json_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    stdout = json.dumps({"result": "x" * 50})
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token", claude_code_output_limit_chars=10),
        popen_factory=lambda *args, **kwargs: FakeProcess(stdout),
    )

    result = manager.send("summarize")

    assert result["ok"] is True
    assert result["output"] == "x" * 10
    assert len(result["stdout"]) == 10
    assert result["parsed_json"] is True
    assert "json" not in result


def test_session_manager_rejects_invalid_permission_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token", claude_code_permission_mode="invalid-mode")
    )

    result = manager.send("run tests")

    assert result["ok"] is False
    assert result["error"] == "invalid_config"
    assert "permission mode" in result["message"]


def test_session_manager_times_out_and_clears_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")

    class TimeoutProcess(FakeProcess):
        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
            return "partial", ""

    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token", claude_code_command_timeout_s=1.0),
        popen_factory=lambda *args, **kwargs: TimeoutProcess("partial"),
    )

    result = manager.send("long task")

    assert result["ok"] is False
    assert result["error"] == "timeout"
    assert manager.status()["running"] is False


def test_session_manager_stop_clears_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = ClaudeCodeSessionManager(WakeDaemonConfig(token="wake-token"))
    session_id = manager.start()["session_id"]

    result = manager.stop()

    assert result["ok"] is True
    assert result["stopped_session_id"] == session_id
    assert manager.status()["active"] is False
