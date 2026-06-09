# ruff: noqa: D101,D102,D103,D107
from types import SimpleNamespace

import pytest

from bobe.claude import (
    ClaudeSettings,
    ClaudeNotConfiguredError,
    ask_claude,
    extract_message_text,
    load_claude_settings,
)


class FakeMessages:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(text="Hello from Claude.")])


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


def test_load_claude_settings_defaults_to_current_sonnet():
    settings = load_claude_settings({})

    assert settings.model == "claude-sonnet-4-6"
    assert settings.max_tokens == 1024
    assert settings.web_search
    assert not settings.is_configured


def test_load_claude_settings_can_disable_web_search():
    settings = load_claude_settings({"BOBE_CLAUDE_WEB_SEARCH": "0"})

    assert not settings.web_search


def test_load_claude_settings_uses_environment_overrides():
    settings = load_claude_settings(
        {
            "ANTHROPIC_API_KEY": " sk-ant-test ",
            "CLAUDE_MODEL": "claude-opus-4-7",
            "CLAUDE_MAX_TOKENS": "123",
        }
    )

    assert settings.api_key == "sk-ant-test"
    assert settings.model == "claude-opus-4-7"
    assert settings.max_tokens == 123
    assert settings.is_configured


def test_extract_message_text_handles_objects_and_dicts():
    message = SimpleNamespace(content=[SimpleNamespace(text="Hello "), {"text": "there"}, {"type": "tool_use"}])

    assert extract_message_text(message) == "Hello there"


@pytest.mark.asyncio
async def test_ask_claude_requires_api_key():
    with pytest.raises(ClaudeNotConfiguredError):
        await ask_claude("hello", settings=ClaudeSettings(api_key=None))


@pytest.mark.asyncio
async def test_ask_claude_sends_spoken_friendly_request_with_web_search():
    client = FakeClient()

    answer = await ask_claude("What should I do today?", settings=ClaudeSettings(api_key="key"), client=client)

    assert answer == "Hello from Claude."
    assert client.messages.kwargs["model"] == "claude-sonnet-4-6"
    assert client.messages.kwargs["messages"] == [{"role": "user", "content": "What should I do today?"}]
    assert "Reachy Mini" in client.messages.kwargs["system"]

    (tool,) = client.messages.kwargs["tools"]
    assert tool["name"] == "web_search"
    assert tool["type"].startswith("web_search_")


@pytest.mark.asyncio
async def test_ask_claude_omits_tools_when_web_search_disabled():
    client = FakeClient()

    await ask_claude("hi", settings=ClaudeSettings(api_key="key", web_search=False), client=client)

    assert "tools" not in client.messages.kwargs
