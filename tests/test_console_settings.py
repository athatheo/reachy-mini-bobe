# ruff: noqa: D101,D102,D103,D107

import os
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from bobe.config import config
from bobe.console import LocalStream
from bobe.env_file import is_plausible_anthropic_key, is_plausible_openai_key, persist_api_settings
from bobe.settings_server import SettingsUIServer, _redact_wake_debug_for_public


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
        wake_test_mode=False,
        wake_test_detections=0,
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
    from bobe.openai_realtime import OpenaiRealtimeHandler
    from unittest.mock import MagicMock
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
