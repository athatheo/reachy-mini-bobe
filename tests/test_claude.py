from types import SimpleNamespace

import pytest

from bobe.claude import (
    ClaudeNotConfiguredError,
    ClaudeSettings,
    ask_claude,
    extract_message_text,
    load_claude_settings,
    should_respond_to_wake_word,
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


def test_load_claude_settings_defaults_to_bob_and_current_sonnet():
    settings = load_claude_settings({})

    assert settings.wake_word == "Bob"
    assert settings.model == "claude-sonnet-4-6"
    assert settings.max_tokens == 512
    assert not settings.is_configured


def test_load_claude_settings_uses_environment_overrides():
    settings = load_claude_settings(
        {
            "ANTHROPIC_API_KEY": " sk-ant-test ",
            "CLAUDE_MODEL": "claude-opus-4-7",
            "CLAUDE_MAX_TOKENS": "123",
            "BOBE_WAKE_WORD": "Bobby",
        }
    )

    assert settings.api_key == "sk-ant-test"
    assert settings.model == "claude-opus-4-7"
    assert settings.max_tokens == 123
    assert settings.wake_word == "Bobby"
    assert settings.is_configured


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bob, what is next?", True),
        ("hey bob can you help", True),
        ("bobcat facts", False),
        ("this has no wake word", False),
    ],
)
def test_should_respond_to_wake_word_matches_standalone_word(text, expected):
    assert should_respond_to_wake_word(text) is expected


def test_extract_message_text_handles_objects_and_dicts():
    message = SimpleNamespace(content=[SimpleNamespace(text="Hello "), {"text": "there"}, {"type": "tool_use"}])

    assert extract_message_text(message) == "Hello there"


@pytest.mark.asyncio
async def test_ask_claude_requires_api_key():
    with pytest.raises(ClaudeNotConfiguredError):
        await ask_claude("hello", settings=ClaudeSettings(api_key=None))


@pytest.mark.asyncio
async def test_ask_claude_sends_spoken_friendly_request():
    client = FakeClient()

    answer = await ask_claude("What should I do today?", settings=ClaudeSettings(api_key="key"), client=client)

    assert answer == "Hello from Claude."
    assert client.messages.kwargs["model"] == "claude-sonnet-4-6"
    assert client.messages.kwargs["messages"] == [{"role": "user", "content": "What should I do today?"}]
    assert "Reachy Mini" in client.messages.kwargs["system"]
