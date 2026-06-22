from pathlib import Path

from bobe.wake_env import merge_packaged_wake_defaults, persist_wake_env, upsert_wake_env_lines


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
    monkeypatch.setattr("bobe.wake_env.Path", lambda *parts: example if parts[-1] == ".env.example" else Path(*parts))
    changed = merge_packaged_wake_defaults(tmp_path)
    assert changed is True
    env = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BOBE_WAKE_BACKEND=remote" in env
