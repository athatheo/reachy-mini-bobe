# ruff: noqa: D101,D102,D103,D107

import json
import time
import threading
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


class KillableProcess(FakeProcess):
    """Blocks in communicate() until the process-group terminate path runs."""

    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        super().__init__(stdout, stderr, returncode)
        self.pid = 0  # keep _terminate_process_group away from os.killpg
        self.done = threading.Event()

    def communicate(self, timeout=None):
        assert self.done.wait(timeout=5)
        return self._stdout, self._stderr

    def wait(self, timeout=None):
        self.done.set()
        return self.returncode


def _wait_for_result(manager: ClaudeCodeSessionManager, timeout: float = 5.0) -> dict:
    """Poll status() until the background command publishes last_result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status()
        if not status["running"] and status["last_result"] is not None:
            return status["last_result"]
        time.sleep(0.005)
    raise AssertionError("command result was never published to status()")


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


def test_session_manager_send_is_accepted_and_publishes_result(tmp_path, monkeypatch):
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

    accepted = manager.send("run the focused tests")

    assert accepted["ok"] is True
    assert accepted["accepted"] is True
    assert accepted["running"] is True
    assert accepted["session_id"]
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

    result = _wait_for_result(manager)
    assert result["ok"] is True
    assert result["output"] == "Tests passed"
    assert result["session_id"] == accepted["session_id"]


def test_session_manager_reports_failed_command(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: FakeProcess("", "permission needed", returncode=1),
    )

    assert manager.send("edit a file")["ok"] is True
    result = _wait_for_result(manager)

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

    assert manager.send("summarize")["ok"] is True
    result = _wait_for_result(manager)

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


def test_session_manager_rejects_second_command_while_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    process = KillableProcess(json.dumps({"result": "done"}))
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: process,
    )

    assert manager.send("long task")["ok"] is True
    busy = manager.send("second task")
    process.done.set()

    assert busy["ok"] is False
    assert busy["error"] == "busy"
    assert _wait_for_result(manager)["output"] == "done"


def test_session_manager_times_out_and_clears_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")

    class TimeoutProcess(FakeProcess):
        def communicate(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired(cmd="claude", timeout=timeout)
            return "partial", ""

        def wait(self, timeout=None):
            return self.returncode

    process = TimeoutProcess("partial")
    process.pid = 0  # keep the timeout path away from os.killpg
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token", claude_code_command_timeout_s=1.0),
        popen_factory=lambda *args, **kwargs: process,
    )

    assert manager.send("long task")["ok"] is True
    result = _wait_for_result(manager)

    assert result["ok"] is False
    assert result["error"] == "timeout"
    assert manager.status()["running"] is False


def test_session_manager_status_is_not_blocked_while_send_spawns(tmp_path, monkeypatch):
    """status() must never wait on a lock held across the slow claude spawn."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    spawn_started = threading.Event()
    release_spawn = threading.Event()

    def blocking_popen(args, **kwargs):
        spawn_started.set()
        assert release_spawn.wait(timeout=5)
        return FakeProcess(json.dumps({"result": "done"}))

    manager = ClaudeCodeSessionManager(WakeDaemonConfig(token="wake-token"), popen_factory=blocking_popen)
    worker = threading.Thread(target=lambda: manager.send("task"))
    worker.start()
    try:
        assert spawn_started.wait(timeout=5)
        started = time.monotonic()
        status = manager.status()
        elapsed = time.monotonic() - started
    finally:
        release_spawn.set()
        worker.join(timeout=5)

    assert status["ok"] is True
    assert status["running"] is True
    assert elapsed < 1.0


def test_session_manager_send_aborts_when_stopped_during_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    spawn_started = threading.Event()
    release_spawn = threading.Event()

    def blocking_popen(args, **kwargs):
        spawn_started.set()
        assert release_spawn.wait(timeout=5)
        process = FakeProcess(json.dumps({"result": "done"}))
        process.pid = 0  # keep the abort path away from os.killpg
        return process

    manager = ClaudeCodeSessionManager(WakeDaemonConfig(token="wake-token"), popen_factory=blocking_popen)
    results: list[dict] = []
    worker = threading.Thread(target=lambda: results.append(manager.send("task")))
    worker.start()
    assert spawn_started.wait(timeout=5)
    stop_result = manager.stop()
    release_spawn.set()
    worker.join(timeout=5)

    assert stop_result["ok"] is True
    assert results[0]["ok"] is False
    assert results[0]["error"] == "no_session"
    assert manager.status()["running"] is False


def test_session_manager_send_never_asserts_when_stop_races_session_id(tmp_path, monkeypatch):
    """A stop() landing around send() must yield a clean error/new session, not a 500."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: FakeProcess(json.dumps({"result": "done"})),
    )
    manager.start()
    manager.stop()  # simulates the stop() winning the race before send()'s critical section

    result = manager.send("task")

    assert result["ok"] is True
    assert result["session_id"]
    assert _wait_for_result(manager)["ok"] is True


def test_session_manager_drops_result_from_session_stopped_mid_command(tmp_path, monkeypatch):
    """stop() during a running command must not have its result resurrected (#36)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    process = KillableProcess(json.dumps({"result": "stale"}), returncode=1)
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: process,
    )

    assert manager.send("long task")["ok"] is True
    stop_result = manager.stop()  # kills the process; the waiter then reaps it
    waiter = manager._waiter_thread
    assert waiter is not None
    waiter.join(timeout=5)

    status = manager.status()
    assert stop_result["ok"] is True
    assert status["active"] is False
    assert status["running"] is False
    assert status["last_result"] is None


def test_session_manager_result_keeps_spawning_session_id_after_restart(tmp_path, monkeypatch):
    """A dead session's result must not be attributed to a brand-new session (#36)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    process = KillableProcess(json.dumps({"result": "stale"}))
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token"),
        popen_factory=lambda *args, **kwargs: process,
    )

    assert manager.send("long task")["ok"] is True
    manager.stop()
    new_session_id = manager.start()["session_id"]
    waiter = manager._waiter_thread
    assert waiter is not None
    waiter.join(timeout=5)

    status = manager.status()
    assert status["session_id"] == new_session_id
    assert status["last_result"] is None


def test_session_manager_shutdown_terminates_active_command(tmp_path, monkeypatch):
    """Daemon shutdown must kill the active claude process and reap its waiter (#25)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(session_module, "resolve_binary", lambda _binary: "/usr/local/bin/claude")
    process = KillableProcess(json.dumps({"result": "done"}))
    manager = ClaudeCodeSessionManager(
        WakeDaemonConfig(token="wake-token", claude_code_command_timeout_s=300.0),
        popen_factory=lambda *args, **kwargs: process,
    )
    assert manager.send("long task")["ok"] is True

    started = time.monotonic()
    manager.shutdown(timeout_s=5.0)
    elapsed = time.monotonic() - started

    assert process.done.is_set()
    assert elapsed < 5.0
    status = manager.status()
    assert status["running"] is False
    assert status["active"] is False


def test_session_manager_stop_clears_session(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    manager = ClaudeCodeSessionManager(WakeDaemonConfig(token="wake-token"))
    session_id = manager.start()["session_id"]

    result = manager.stop()

    assert result["ok"] is True
    assert result["stopped_session_id"] == session_id
    assert manager.status()["active"] is False
