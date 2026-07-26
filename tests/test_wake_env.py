import os
from pathlib import Path

from bobe.wake_env import (
    REMOTE_WAKE_KEYS,
    persist_wake_env,
    wake_allowed_hosts,
    upsert_wake_env_lines,
    default_wake_allowed_hosts,
    is_wake_remote_host_allowed,
    merge_packaged_wake_defaults,
)


def _clear_wake_env(monkeypatch) -> None:
    """Unset wake env vars, registering restoration for values set by code under test."""
    for key in REMOTE_WAKE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_upsert_wake_env_lines():
    lines = ["OPENAI_API_KEY=sk-test"]
    upsert_wake_env_lines(
        lines,
        backend="remote",
        remote_url="ws://192.168.1.114:8765/v1/stream",
        token="abc123",
        gain=1.75,
    )
    joined = "\n".join(lines)
    assert "BOBE_WAKE_BACKEND=remote" in joined
    assert "BOBE_WAKE_REMOTE_URL=ws://192.168.1.114:8765/v1/stream" in joined
    assert "BOBE_WAKE_TOKEN=abc123" in joined


def test_upsert_wake_env_lines_none_preserves_existing():
    lines = [
        "BOBE_WAKE_REMOTE_URL=ws://192.168.1.172:8765/v1/stream",
        "BOBE_WAKE_GAIN=1.1",
    ]
    upsert_wake_env_lines(lines, token="abc123")
    joined = "\n".join(lines)
    assert "BOBE_WAKE_REMOTE_URL=ws://192.168.1.172:8765/v1/stream" in joined
    assert "BOBE_WAKE_GAIN=1.1" in joined
    assert "BOBE_WAKE_TOKEN=abc123" in joined


def test_persist_wake_env(tmp_path: Path, monkeypatch):
    _clear_wake_env(monkeypatch)
    env_path = persist_wake_env(
        tmp_path,
        remote_url="ws://example:8765/v1/stream",
        token="secret",
    )
    text = env_path.read_text(encoding="utf-8")
    assert "BOBE_WAKE_BACKEND=remote" in text
    assert "secret" in text


def test_persist_wake_env_preserves_tuned_gain_and_url(tmp_path: Path, monkeypatch):
    """Omitted gain/url must keep the owner's tuned values, not reset them."""
    _clear_wake_env(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "BOBE_WAKE_REMOTE_URL=ws://192.168.1.172:8765/v1/stream\nBOBE_WAKE_GAIN=1.1\n",
        encoding="utf-8",
    )

    persist_wake_env(tmp_path, token="new-token")

    text = env_path.read_text(encoding="utf-8")
    assert "BOBE_WAKE_REMOTE_URL=ws://192.168.1.172:8765/v1/stream" in text
    assert "BOBE_WAKE_GAIN=1.1" in text
    assert "BOBE_WAKE_TOKEN=new-token" in text
    assert os.getenv("BOBE_WAKE_REMOTE_URL") is None
    assert os.getenv("BOBE_WAKE_GAIN") is None


def test_persist_wake_env_persists_allowed_hosts(tmp_path: Path, monkeypatch):
    """A newly accepted URL hostname is merged into the stored allowlist."""
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.172")
    env_path = persist_wake_env(
        tmp_path,
        remote_url="ws://192.168.1.114:8765/v1/stream",
        token="secret",
    )
    text = env_path.read_text(encoding="utf-8")
    hosts_line = next(
        line for line in text.splitlines() if line.startswith("BOBE_WAKE_ALLOWED_HOSTS=")
    )
    assert "192.168.1.114" in hosts_line
    assert "192.168.1.172" in hosts_line
    assert wake_allowed_hosts() >= {"192.168.1.114", "192.168.1.172"}


def test_merge_packaged_wake_defaults(tmp_path: Path, monkeypatch):
    _clear_wake_env(monkeypatch)
    example = tmp_path / "example.env"
    example.write_text(
        "\n".join(
            [
                "BOBE_WAKE_BACKEND=remote",
                "BOBE_WAKE_REMOTE_URL=ws://192.168.1.114:8765/v1/stream",
                "BOBE_WAKE_GAIN=1.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    real_path = Path

    def fake_path(*parts: str) -> Path:
        if parts and parts[-1] == ".env.example":
            return example
        return real_path(*parts)

    monkeypatch.setattr("bobe.wake_env.Path", fake_path)
    changed = merge_packaged_wake_defaults(tmp_path)
    assert changed is True
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BOBE_WAKE_BACKEND=remote" in env


def test_merge_packaged_wake_defaults_does_not_seed_allowed_hosts(tmp_path: Path, monkeypatch):
    """The allowlist must stay unpinned so republished packaged defaults keep applying."""
    _clear_wake_env(monkeypatch)
    changed = merge_packaged_wake_defaults(tmp_path)
    assert changed is True
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BOBE_WAKE_BACKEND=remote" in env
    assert "BOBE_WAKE_ALLOWED_HOSTS=" not in env
    assert os.getenv("BOBE_WAKE_ALLOWED_HOSTS") is None


def test_merge_packaged_wake_defaults_respects_live_env(tmp_path: Path, monkeypatch):
    """Example defaults must not overwrite a tuned value exported in the environment."""
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_GAIN", "1.1")

    merge_packaged_wake_defaults(tmp_path)

    env_file = tmp_path / ".env"
    text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    assert "BOBE_WAKE_GAIN=1.75" not in text
    assert os.environ["BOBE_WAKE_GAIN"] == "1.1"


def test_default_wake_allowed_hosts_from_packaged_example():
    hosts = default_wake_allowed_hosts()
    assert "mac.local" in hosts


def test_wake_allowed_hosts_env_extends_defaults(monkeypatch):
    """A configured allowlist adds casefolded hosts without hiding the packaged defaults."""
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "Custom-Host.local, robot.local")
    hosts = wake_allowed_hosts()
    assert {"custom-host.local", "robot.local"} <= hosts
    assert hosts >= default_wake_allowed_hosts()


def test_wake_allowed_hosts_always_includes_remote_url_host(monkeypatch):
    """The configured daemon URL host is allowed even when a stale allowlist omits it."""
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "10.0.0.5")
    monkeypatch.setenv("BOBE_WAKE_REMOTE_URL", "ws://New-Mac.local:8765/v1/stream")
    assert "new-mac.local" in wake_allowed_hosts()
    assert is_wake_remote_host_allowed("New-Mac.local")


def test_is_wake_remote_host_allowed(monkeypatch):
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.114")
    assert is_wake_remote_host_allowed("192.168.1.114")
    assert not is_wake_remote_host_allowed("evil.example")
