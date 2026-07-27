# ruff: noqa: D101,D102,D103,D107

import os
import threading
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bobe.config import config
from bobe.console import LocalStream
from bobe.env_file import (
    read_env_lines,
    parse_env_lines,
    upsert_env_keys,
    persist_api_settings,
    is_plausible_openai_key,
    is_plausible_anthropic_key,
)
from bobe.wake_env import REMOTE_WAKE_KEYS, persist_wake_env
from bobe.settings_server import SettingsUIServer, _redact_wake_debug_for_public


def _clear_wake_env(monkeypatch) -> None:
    """Unset wake env vars, registering restoration for values set by code under test."""
    for key in REMOTE_WAKE_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_persist_api_settings_writes_explicit_provider_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    persist_api_settings(
        str(tmp_path),
        openai_api_key=" sk-proj-test-openai-key ",
        anthropic_api_key=" sk-ant-test-anthropic-key ",
        claude_model=" claude-test ",
    )

    assert os.environ["OPENAI_API_KEY"] == "sk-proj-test-openai-key"
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-test-anthropic-key"
    assert os.environ["CLAUDE_MODEL"] == "claude-test"
    assert config.OPENAI_API_KEY == "sk-proj-test-openai-key"

    env_text = (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY=sk-proj-test-openai-key" in env_text
    assert "ANTHROPIC_API_KEY=sk-ant-test-anthropic-key" in env_text
    assert "CLAUDE_MODEL=claude-test" in env_text


def test_persist_api_settings_does_not_bake_template_lines(tmp_path, monkeypatch):
    """A missing instance .env must not be seeded from .env.example templates."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    (tmp_path / ".env.example").write_text(
        "BOBE_WAKE_TOKEN=\nBOBE_WAKE_GAIN=1.75\nHF_TOKEN=\n", encoding="utf-8"
    )

    persist_api_settings(
        str(tmp_path),
        openai_api_key="sk-proj-test-openai-key",
        anthropic_api_key="sk-ant-test-anthropic-key",
        claude_model="claude-test",
    )

    env_text = (tmp_path / ".env").read_text()
    assert "BOBE_WAKE_TOKEN" not in env_text
    assert "BOBE_WAKE_GAIN" not in env_text
    assert "HF_TOKEN" not in env_text
    assert "OPENAI_API_KEY=sk-proj-test-openai-key" in env_text


def test_read_env_lines_missing_file_ignores_templates(tmp_path):
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\nBOBE_WAKE_GAIN=1.75\n", encoding="utf-8")
    assert read_env_lines(tmp_path / ".env") == []


def test_parse_env_lines_strips_quotes_and_skips_comments():
    lines = [
        "# comment",
        "",
        "PLAIN=value",
        'QUOTED="ws://mac.local:8765/v1/stream"',
        "SINGLE='1.75'",
        "NOEQUALS",
        "EMPTY=",
    ]
    assert parse_env_lines(lines) == {
        "PLAIN": "value",
        "QUOTED": "ws://mac.local:8765/v1/stream",
        "SINGLE": "1.75",
        "EMPTY": "",
    }


def test_parse_env_lines_first_occurrence_wins():
    """Duplicate keys keep the first value, matching upsert_env_keys' first-line update."""
    assert parse_env_lines(["KEY=first", "KEY=second"]) == {"KEY": "first"}


def test_upsert_env_keys_skips_empty_values():
    lines = ["BOBE_WAKE_TOKEN=live-token"]
    upsert_env_keys(lines, {"BOBE_WAKE_TOKEN": "", "OPENAI_API_KEY": " "})
    assert lines == ["BOBE_WAKE_TOKEN=live-token"]


def test_concurrent_api_and_wake_persist_keep_both_key_sets(tmp_path, monkeypatch):
    """Serialized read-modify-write: neither writer may drop the other's keys."""
    _clear_wake_env(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    def write_api() -> None:
        for _ in range(20):
            persist_api_settings(
                str(tmp_path),
                openai_api_key="sk-proj-test-openai-key",
                anthropic_api_key="sk-ant-test-anthropic-key",
                claude_model="claude-test",
            )

    def write_wake() -> None:
        for _ in range(20):
            persist_wake_env(
                tmp_path,
                remote_url="ws://192.168.1.114:8765/v1/stream",
                token="secret-token",
            )

    threads = [threading.Thread(target=write_api), threading.Thread(target=write_wake)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-proj-test-openai-key" in env_text
    assert "ANTHROPIC_API_KEY=sk-ant-test-anthropic-key" in env_text
    assert "BOBE_WAKE_TOKEN=secret-token" in env_text
    assert "BOBE_WAKE_REMOTE_URL=ws://192.168.1.114:8765/v1/stream" in env_text


def test_required_api_keys_configured_requires_both_keys(monkeypatch):
    stream = LocalStream(SimpleNamespace(), SimpleNamespace())

    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not stream._required_api_keys_configured()

    monkeypatch.setattr(config, "OPENAI_API_KEY", "sk-proj-test-openai-key")
    assert not stream._required_api_keys_configured()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-anthropic-key")
    assert stream._required_api_keys_configured()


def test_api_key_shape_validation_rejects_dummy_values():
    assert not is_plausible_openai_key("x")
    assert not is_plausible_anthropic_key("y")
    assert is_plausible_openai_key("sk-proj-test-openai-key")
    assert is_plausible_anthropic_key("sk-ant-test-anthropic-key")


def test_redact_wake_debug_strips_transcript_fields():
    debug = {
        "connected": True,
        "transcript_last": "hey jarvis",
        "transcript_partial": "hey",
        "transcript_stream": [{"text": "hey jarvis"}],
        "transcript_display": ["[final] hey jarvis"],
        "rms_last": 512.0,
        "remote_stats": {"transcript": "hey jarvis", "partial": "hey", "rms": 512.0},
    }
    redacted = _redact_wake_debug_for_public(debug)
    assert redacted["connected"] is True
    assert redacted["rms_last"] == 512.0
    assert "transcript_last" not in redacted
    assert "transcript_stream" not in redacted
    assert "transcript" not in redacted["remote_stats"]
    assert redacted["remote_stats"]["rms"] == 512.0


def _settings_client(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    handler = SimpleNamespace(
        wake_config=None,
        wake_session=None,
        _wake_detector=SimpleNamespace(
            debug_state=lambda: {
                "connected": True,
                "transcript_last": "secret speech",
                "transcript_stream": [{"text": "secret speech"}],
                "transcript_display": ["[final] secret speech"],
                "remote_stats": {"transcript": "secret speech", "rms": 100.0},
            }
        ),
        connection=None,
    )
    app = FastAPI()
    SettingsUIServer(str(tmp_path), lambda: handler).mount(app)
    return TestClient(app), handler


def test_status_redacts_wake_debug_without_api_keys(tmp_path, monkeypatch):
    client, _ = _settings_client(tmp_path, monkeypatch)
    data = client.get("/status").json()
    wake_debug = data["wake_debug"]
    assert "transcript_last" not in wake_debug
    assert "transcript_stream" not in wake_debug
    assert "transcript" not in wake_debug["remote_stats"]
    assert wake_debug["remote_stats"]["rms"] == 100.0


def test_status_includes_wake_debug_with_api_keys(tmp_path, monkeypatch):
    client, _ = _settings_client(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-openai-key-long-enough")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-anthropic-key-long")
    data = client.get("/status").json()
    assert data["wake_debug"]["transcript_last"] == "secret speech"


def test_status_reports_wake_error_when_gating_disabled(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    from bobe.openai_realtime import OpenaiRealtimeHandler
    from bobe.tools.core_tools import ToolDependencies

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("BOBE_WAKE_REMOTE_URL", raising=False)
    monkeypatch.delenv("BOBE_WAKE_TOKEN", raising=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = OpenaiRealtimeHandler(deps)
    app = FastAPI()
    SettingsUIServer(str(tmp_path), lambda: handler).mount(app)
    client = TestClient(app)

    data = client.get("/status").json()
    assert data["wake_enabled"] is False
    assert data["wake_error"] is not None
    assert "BOBE_WAKE_REMOTE_URL" in data["wake_error"]


def test_wake_config_rejects_disallowed_host(tmp_path, monkeypatch):
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.114")
    client, _ = _settings_client(tmp_path, monkeypatch)
    resp = client.post(
        "/wake-config",
        json={
            "backend": "remote",
            "remote_url": "ws://evil.example:8765/v1/stream",
            "token": "secret-token",
            "gain": 1.75,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "remote_host_not_allowed"


def test_wake_config_accepts_allowed_host(tmp_path, monkeypatch):
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.114")
    client, _ = _settings_client(tmp_path, monkeypatch)
    resp = client.post(
        "/wake-config",
        json={
            "backend": "remote",
            "remote_url": "ws://192.168.1.114:8765/v1/stream",
            "token": "secret-token",
            "gain": 1.75,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_wake_config_omitted_gain_and_url_preserve_tuned_values(tmp_path, monkeypatch):
    """POST /wake-config without gain/remote_url must not reset the tuned config."""
    _clear_wake_env(monkeypatch)
    (tmp_path / ".env").write_text(
        "BOBE_WAKE_REMOTE_URL=ws://192.168.1.172:8765/v1/stream\nBOBE_WAKE_GAIN=1.1\n",
        encoding="utf-8",
    )
    client, _ = _settings_client(tmp_path, monkeypatch)
    resp = client.post("/wake-config", json={"backend": "remote", "token": "secret-token"})
    assert resp.status_code == 200
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "BOBE_WAKE_REMOTE_URL=ws://192.168.1.172:8765/v1/stream" in env_text
    assert "BOBE_WAKE_GAIN=1.1" in env_text
    assert "BOBE_WAKE_TOKEN=secret-token" in env_text


def test_wake_config_persists_allowlist_with_accepted_host(tmp_path, monkeypatch):
    """The just-validated hostname is persisted so the allowlist never reverts."""
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.114")
    client, _ = _settings_client(tmp_path, monkeypatch)
    resp = client.post(
        "/wake-config",
        json={
            "backend": "remote",
            "remote_url": "ws://192.168.1.114:8765/v1/stream",
            "token": "secret-token",
        },
    )
    assert resp.status_code == 200
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    hosts_line = next(
        line for line in env_text.splitlines() if line.startswith("BOBE_WAKE_ALLOWED_HOSTS=")
    )
    assert "192.168.1.114" in hosts_line


def test_wake_config_still_validates_provided_url(tmp_path, monkeypatch):
    _clear_wake_env(monkeypatch)
    monkeypatch.setenv("BOBE_WAKE_ALLOWED_HOSTS", "192.168.1.114")
    client, _ = _settings_client(tmp_path, monkeypatch)
    resp = client.post(
        "/wake-config",
        json={"backend": "remote", "remote_url": "http://192.168.1.114:8765", "token": "secret-token"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_remote_url"
