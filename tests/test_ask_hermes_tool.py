# ruff: noqa: D103

import json
from types import SimpleNamespace

import httpx
import pytest

from bobe.profiles._bobe_locked_profile import ask_hermes as tool_module


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient capturing the request it is given."""

    calls: list[dict] = []
    response: _FakeResponse | Exception = _FakeResponse({})

    def __init__(self, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def post(self, url, *, json=None, headers=None):
        _FakeAsyncClient.calls.append({"url": url, "json": json, "headers": headers})
        if isinstance(_FakeAsyncClient.response, Exception):
            raise _FakeAsyncClient.response
        return _FakeAsyncClient.response


@pytest.fixture
def hermes_env(monkeypatch):
    monkeypatch.setenv("BOBE_HERMES_URL", "http://mac.test:8642/v1")
    monkeypatch.setenv("BOBE_HERMES_API_KEY", "hermes-test-key")
    monkeypatch.setattr(tool_module.httpx, "AsyncClient", _FakeAsyncClient)
    _FakeAsyncClient.calls = []
    _FakeAsyncClient.response = _FakeResponse(
        {"choices": [{"message": {"role": "assistant", "content": "Two tasks are open."}}]}
    )


@pytest.mark.asyncio
async def test_ask_hermes_returns_answer(hermes_env):
    result = await tool_module.AskHermes()(SimpleNamespace(), request="What's on my kanban?")

    assert result == {"status": "ok", "answer": "Two tasks are open."}
    call = _FakeAsyncClient.calls[0]
    assert call["url"] == "http://mac.test:8642/v1/chat/completions"
    assert call["json"]["messages"] == [{"role": "user", "content": "What's on my kanban?"}]
    assert call["headers"]["Authorization"] == "Bearer hermes-test-key"
    assert call["headers"]["X-Hermes-Session-Id"] == tool_module.HERMES_SESSION_ID


@pytest.mark.asyncio
async def test_ask_hermes_requires_configuration(monkeypatch):
    monkeypatch.delenv("BOBE_HERMES_URL", raising=False)
    monkeypatch.delenv("BOBE_HERMES_API_KEY", raising=False)

    result = await tool_module.AskHermes()(SimpleNamespace(), request="hello")

    assert result["status"] == "missing_config"


@pytest.mark.asyncio
async def test_ask_hermes_requires_request(hermes_env):
    result = await tool_module.AskHermes()(SimpleNamespace(), request="  ")

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_ask_hermes_reports_timeout(hermes_env):
    _FakeAsyncClient.response = httpx.TimeoutException("slow")

    result = await tool_module.AskHermes()(SimpleNamespace(), request="do a long thing")

    assert result["status"] == "error"
    assert "did not answer" in result["error"]


@pytest.mark.asyncio
async def test_ask_hermes_reports_http_errors(hermes_env):
    _FakeAsyncClient.response = _FakeResponse({}, status_code=500)

    result = await tool_module.AskHermes()(SimpleNamespace(), request="hello")

    assert result["status"] == "error"


@pytest.mark.asyncio
async def test_ask_hermes_rejects_empty_answer(hermes_env):
    _FakeAsyncClient.response = _FakeResponse({"choices": []})

    result = await tool_module.AskHermes()(SimpleNamespace(), request="hello")

    assert result["status"] == "error"
    assert "empty" in result["error"]


def test_extract_answer_handles_malformed_payloads():
    assert tool_module._extract_answer({}) == ""
    assert tool_module._extract_answer({"choices": [{}]}) == ""
    assert tool_module._extract_answer({"choices": [{"message": {"content": None}}]}) == ""
    assert tool_module._extract_answer(json.loads('{"choices": [{"message": {"content": " hi "}}]}')) == "hi"
