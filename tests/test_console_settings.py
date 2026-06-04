# ruff: noqa: D101,D102,D103,D107

import os
from types import SimpleNamespace

from bobe.config import config
from bobe.console import LocalStream


def test_persist_api_settings_writes_explicit_provider_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)
    monkeypatch.setattr(config, "OPENAI_API_KEY", None)

    stream = LocalStream(SimpleNamespace(), SimpleNamespace(), instance_path=str(tmp_path))

    stream._persist_api_settings(
        openai_api_key=" openai-key ",
        anthropic_api_key=" anthropic-key ",
        claude_model=" claude-test ",
    )

    assert os.environ["OPENAI_API_KEY"] == "openai-key"
    assert os.environ["ANTHROPIC_API_KEY"] == "anthropic-key"
    assert os.environ["CLAUDE_MODEL"] == "claude-test"
    assert config.OPENAI_API_KEY == "openai-key"

    env_text = (tmp_path / ".env").read_text()
    assert "OPENAI_API_KEY=openai-key" in env_text
    assert "ANTHROPIC_API_KEY=anthropic-key" in env_text
    assert "CLAUDE_MODEL=claude-test" in env_text


def test_required_api_keys_configured_requires_both_keys(monkeypatch):
    stream = LocalStream(SimpleNamespace(), SimpleNamespace())

    monkeypatch.setattr(config, "OPENAI_API_KEY", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert not stream._required_api_keys_configured()

    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-key")
    assert not stream._required_api_keys_configured()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    assert stream._required_api_keys_configured()
