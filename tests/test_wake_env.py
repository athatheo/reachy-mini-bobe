from pathlib import Path

from bobe.wake_env import (
    persist_wake_env,
    wake_allowed_hosts,
    upsert_wake_env_lines,
    default_wake_allowed_hosts,
    is_wake_remote_host_allowed,
    merge_packaged_wake_defaults,
)


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


def test_persist_wake_env(tmp_path: Path):
    env_path = persist_wake_env(
        tmp_path,
        remote_url="ws://example:8765/v1/stream",
        token="secret",
    )
    text = env_path.read_text(encoding="utf-8")
    assert "BOBE_WAKE_BACKEND=remote" in text
    assert "secret" in text


def test_merge_packaged_wake_defaults(tmp_path: Path, monkeypatch):
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


def test_default_wake_allowed_hosts_from_packaged_example():
    hosts = default_wake_allowed_hosts()
    assert "mac.local" in hosts


def test_wake_allowed_hosts_env_override(monkeypatch):
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "Mac.local, robot.local")
    assert wake_allowed_hosts() == frozenset({"mac.local", "robot.local"})


def test_is_wake_remote_host_allowed(monkeypatch):
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.114")
    assert is_wake_remote_host_allowed("192.168.1.114")
    assert not is_wake_remote_host_allowed("evil.example")
