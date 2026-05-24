# ruff: noqa: D103

from types import SimpleNamespace

import pytest

from bobe.claude import ClaudeSettings
from bobe.profiles._bobe_locked_profile import ask_claude as tool_module


@pytest.mark.asyncio
async def test_ask_claude_tool_returns_emotion_metadata(monkeypatch):
    async def fake_ask_claude(question, *, settings):
        assert question == "What went well?"
        assert settings.model == "claude-test"
        return "Great, that worked nicely."

    monkeypatch.setattr(tool_module, "load_claude_settings", lambda: ClaudeSettings(api_key="key", model="claude-test"))
    monkeypatch.setattr(tool_module, "ask_claude", fake_ask_claude)

    result = await tool_module.AskClaude()(SimpleNamespace(), question="What went well?")

    assert result["status"] == "ok"
    assert result["answer"] == "Great, that worked nicely."
    assert result["emotion"] == "happy"
    assert result["should_play_emotion"] is True


@pytest.mark.asyncio
async def test_ask_claude_tool_returns_neutral_metadata(monkeypatch):
    async def fake_ask_claude(question, *, settings):
        return "The next event is at three."

    monkeypatch.setattr(tool_module, "load_claude_settings", lambda: ClaudeSettings(api_key="key"))
    monkeypatch.setattr(tool_module, "ask_claude", fake_ask_claude)

    result = await tool_module.AskClaude()(SimpleNamespace(), question="What is next?")

    assert result["status"] == "ok"
    assert result["emotion"] == "neutral"
    assert result["should_play_emotion"] is False
