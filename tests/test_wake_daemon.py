# ruff: noqa: D103
import sys
import time
import types
import logging
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from bobe.wake_daemon import engine as engine_module
from bobe.wake_daemon.config import load_wake_daemon_config
from bobe.wake_daemon.engine import WhisperWakeEngine, warn_if_phrases_unsupported


_TEST_ENV = {"BOBE_WAKE_TOKEN": "test-token"}


def _claude_code_launch_config():
    return load_wake_daemon_config(
        {
            **_TEST_ENV,
            "BOBE_CLAUDE_CODE_LAUNCH_ENABLED": "1",
            "BOBE_CLAUDE_CODE_LAUNCH_TOKEN": "launch-token",
        }
    )


def test_load_wake_daemon_config_requires_token():
    with pytest.raises(ValueError, match="BOBE_WAKE_TOKEN"):
        load_wake_daemon_config({})


def _session(config=None, *, monkeypatch=None, transcribe=None):
    runtime = config or load_wake_daemon_config(_TEST_ENV)
    engine = WhisperWakeEngine(runtime)
    session = engine.session(runtime)
    if monkeypatch is not None and transcribe is not None:
        monkeypatch.setattr(engine, "transcribe", lambda pcm, *, config=None: transcribe(pcm))
    return session


def test_load_wake_daemon_config_defaults():
    config = load_wake_daemon_config(_TEST_ENV)

    assert config.phrase == "hey bobe"
    assert config.whisper_model == "distil-small.en"
    assert config.whisper_language is None
    assert config.whisper_initial_prompt is None
    assert config.whisper_hotwords is None
    assert config.port == 8765


def test_whisper_language_env_override():
    assert load_wake_daemon_config({**_TEST_ENV, "WHISPER_LANGUAGE": "EL"}).whisper_language == "el"
    assert load_wake_daemon_config({**_TEST_ENV, "WHISPER_LANGUAGE": ""}).whisper_language is None
    assert load_wake_daemon_config({**_TEST_ENV, "WHISPER_LANGUAGE": "auto"}).whisper_language is None


def test_whisper_prompt_helpers():
    from bobe.wake_daemon.config import whisper_initial_prompt_from_phrase

    assert whisper_initial_prompt_from_phrase("hey bobe") == "Bobe."


def test_whisper_prompt_env_override():
    config = load_wake_daemon_config(
        {
            **_TEST_ENV,
            "WHISPER_INITIAL_PROMPT": "Jarvis.",
            "WHISPER_HOTWORDS": "Jarvis",
        }
    )
    assert config.whisper_initial_prompt == "Jarvis."
    assert config.whisper_hotwords == "Jarvis"


def test_whisper_engine_detects_wake_phrase(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "hey bobe")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = session.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe

    assert event is not None
    assert event["type"] == "wake"
    assert event["phrase"] == "hey bobe"


def test_whisper_engine_wakes_on_partial_before_final(monkeypatch):
    """Wake must fire from a live partial — waiting for final often rewrites noise."""
    session = _session(
        monkeypatch=monkeypatch,
        transcribe=lambda _audio: "Hey Bobby, how are you?",
    )
    # Continuous speech only (no trailing silence), so finalize should not run.
    pcm = np.full(16000, 5000, dtype=np.int16)
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = session.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe
            break

    assert event is not None
    assert event["type"] == "wake"
    assert "bobby" in event["transcript"].casefold() or "bobe" in event["transcript"].casefold()


def test_whisper_engine_ignores_unrelated_speech(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "good morning")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    events = [session.feed(pcm[offset : offset + 1600]) for offset in range(0, pcm.size, 1600)]

    assert all(event is None for event in events)


def test_whisper_session_listen_modes_are_isolated():
    config = load_wake_daemon_config(_TEST_ENV)
    engine = WhisperWakeEngine(config)
    session_a = engine.session(config)
    session_b = engine.session(config)

    session_a.set_listen_mode("sleep")
    session_b.set_listen_mode("wake")

    assert session_a.debug_state()["listen_mode"] == "sleep"
    assert session_b.debug_state()["listen_mode"] == "wake"

    pcm = np.zeros(1600, dtype=np.int16)
    pcm[:] = 5000
    assert session_a.feed(pcm) is None
    assert session_b.feed(pcm) is None
    assert session_b.debug_state()["in_speech"] is True
    assert session_a.debug_state()["in_speech"] is True


def test_whisper_engine_detects_sleep_phrase(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "go to sleep")
    session.set_listen_mode("sleep")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = session.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe

    assert event is not None
    assert event["type"] == "sleep"


def test_transcribe_passes_configured_language():
    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return ([], None)

    engine = WhisperWakeEngine(load_wake_daemon_config({**_TEST_ENV, "WHISPER_LANGUAGE": "el"}))
    engine._model = FakeModel()

    engine.transcribe(np.zeros(1600, dtype=np.int16))

    assert captured["language"] == "el"


def test_transcribe_defaults_to_language_autodetect():
    captured = {}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return ([], None)

    engine = WhisperWakeEngine(load_wake_daemon_config(_TEST_ENV))
    engine._model = FakeModel()

    engine.transcribe(np.zeros(1600, dtype=np.int16))

    assert captured["language"] is None


def test_partial_throttle_measured_from_transcribe_completion(monkeypatch):
    """Chunks buffered behind a slow transcribe must not each re-transcribe the utterance."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(engine_module.time, "monotonic", lambda: clock["now"])
    calls = {"count": 0}

    def slow_transcribe(_audio):
        calls["count"] += 1
        clock["now"] += 0.6  # slower than PARTIAL_TRANSCRIBE_INTERVAL_S (0.45s)
        return "hello there"

    session = _session(monkeypatch=monkeypatch, transcribe=slow_transcribe)

    chunk = np.full(1600, 5000, dtype=np.int16)  # 0.1s of voiced audio at 16 kHz
    for _ in range(20):
        # No wall time passes between feeds: this models the backlog of frames
        # that buffered up while the slow transcribe blocked the handler.
        assert session.feed(chunk) is None

    assert calls["count"] == 1

    clock["now"] += 0.5  # real quiet time elapses -> the next partial may run
    session.feed(chunk)
    assert calls["count"] == 2


def test_whisper_model_load_is_single_flight_across_threads(monkeypatch):
    created = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)
            created.append(self)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    engine = WhisperWakeEngine(load_wake_daemon_config(_TEST_ENV))

    results: list[object] = []
    threads = [threading.Thread(target=lambda: results.append(engine._load_model())) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(created) == 1
    assert results == [created[0]] * 4


def test_engine_preload_loads_model_once(monkeypatch):
    created = []

    class FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            created.append(self)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    engine = WhisperWakeEngine(load_wake_daemon_config(_TEST_ENV))

    engine.preload()
    engine.preload()

    assert len(created) == 1


def test_whisper_engine_loads_model_once(monkeypatch):
    config = load_wake_daemon_config(_TEST_ENV)
    engine = WhisperWakeEngine(config)
    load_calls = {"count": 0}

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            return ([], None)

    def fake_load():
        if engine._model is not None:
            return engine._model
        load_calls["count"] += 1
        engine._model = FakeModel()
        return engine._model

    monkeypatch.setattr(engine, "_load_model", fake_load)

    pcm = np.zeros(1600, dtype=np.int16)
    engine.transcribe(pcm)
    engine.transcribe(pcm)

    assert load_calls["count"] == 1


def test_wake_daemon_app_starts_with_empty_engine_pool():
    from bobe.wake_daemon.server import create_app

    config = load_wake_daemon_config(_TEST_ENV)
    app = create_app(config)
    assert app.state.wake_engines == {}


def test_create_app_preloads_whisper_model_on_startup(monkeypatch):
    from bobe.wake_daemon.server import create_app

    preloaded = []
    monkeypatch.setattr(WhisperWakeEngine, "preload", lambda self: preloaded.append(self))
    app = create_app(load_wake_daemon_config(_TEST_ENV))

    with TestClient(app):
        app.state.whisper_preload_thread.join(timeout=5)

    assert len(preloaded) == 1
    assert list(app.state.wake_engines.values()) == preloaded


def test_create_app_warns_when_english_only_model_meets_non_ascii_phrase(caplog):
    from bobe.wake_daemon.server import create_app

    config = load_wake_daemon_config({**_TEST_ENV, "WHISPER_MODEL": "medium.en"})
    with caplog.at_level(logging.WARNING, logger="bobe.wake_daemon.engine"):
        create_app(config)

    assert any("English-only" in record.getMessage() for record in caplog.records)


def test_no_language_warning_with_multilingual_model(caplog):
    config = load_wake_daemon_config({**_TEST_ENV, "WHISPER_MODEL": "small"})
    with caplog.at_level(logging.WARNING, logger="bobe.wake_daemon.engine"):
        warn_if_phrases_unsupported(config)

    assert not [record for record in caplog.records if "never be detected" in record.getMessage()]


def test_warns_when_language_forced_english_with_non_ascii_phrase(caplog):
    config = load_wake_daemon_config({**_TEST_ENV, "WHISPER_MODEL": "small", "WHISPER_LANGUAGE": "en"})
    with caplog.at_level(logging.WARNING, logger="bobe.wake_daemon.engine"):
        warn_if_phrases_unsupported(config)

    assert any("WHISPER_LANGUAGE=en" in record.getMessage() for record in caplog.records)


def test_stream_rejects_invalid_hello_token():
    from bobe.wake_daemon.server import create_app

    client = TestClient(create_app(load_wake_daemon_config(_TEST_ENV)))

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "wrong-token", "sample_rate": 16000, "phrase": "hey bobe"})
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()

    assert excinfo.value.code == 1008


def test_stream_accepts_valid_hello_token():
    from bobe.wake_daemon.server import create_app

    client = TestClient(create_app(load_wake_daemon_config(_TEST_ENV)))

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        ready = ws.receive_json()

    assert ready["type"] == "ready"
    assert ready["phrase"] == "hey bobe"


def test_claude_code_launch_endpoint_disabled_by_default():
    from bobe.wake_daemon.server import create_app

    config = load_wake_daemon_config(_TEST_ENV)
    client = TestClient(create_app(config))

    response = client.post("/v1/launch/claude-code")

    assert response.status_code == 403
    assert response.json()["error"] == "disabled"


def test_claude_code_launch_endpoint_rejects_bad_token():
    from bobe.wake_daemon.server import create_app

    config = _claude_code_launch_config()
    client = TestClient(create_app(config))

    response = client.post("/v1/launch/claude-code", headers={"X-BoBe-Launch-Token": "bad-token"})

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_claude_code_launch_endpoint_requires_launch_token_when_enabled():
    from bobe.wake_daemon.server import create_app

    config = load_wake_daemon_config(
        {
            **_TEST_ENV,
            "BOBE_CLAUDE_CODE_LAUNCH_ENABLED": "1",
        }
    )
    client = TestClient(create_app(config))

    response = client.post("/v1/launch/claude-code", headers={"X-BoBe-Launch-Token": "launch-token"})

    assert response.status_code == 503
    assert response.json()["error"] == "missing_launch_token"


def test_claude_code_launch_endpoint_calls_launcher_when_enabled():
    from bobe.wake_daemon.server import create_app

    class FakeLauncher:
        def launch(self):
            return {"ok": True, "workdir": "/tmp/repos/bobe", "binary": "claude"}

    config = _claude_code_launch_config()
    app = create_app(config)
    app.state.claude_code_launcher = FakeLauncher()
    client = TestClient(app)

    response = client.post("/v1/launch/claude-code", headers={"X-BoBe-Launch-Token": "launch-token"})

    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_claude_code_launch_endpoint_returns_cooldown_status():
    from bobe.wake_daemon.server import create_app

    class FakeLauncher:
        def launch(self):
            return {"ok": False, "error": "cooldown", "retry_after_s": 12.0}

    config = _claude_code_launch_config()
    app = create_app(config)
    app.state.claude_code_launcher = FakeLauncher()
    client = TestClient(app)

    response = client.post("/v1/launch/claude-code", headers={"X-BoBe-Launch-Token": "launch-token"})

    assert response.status_code == 429
    assert response.json()["error"] == "cooldown"


def test_claude_code_launch_endpoint_maps_invalid_config():
    from bobe.wake_daemon.server import create_app

    class FakeLauncher:
        def launch(self):
            return {"ok": False, "error": "invalid_config", "message": "bad workdir"}

    config = _claude_code_launch_config()
    app = create_app(config)
    app.state.claude_code_launcher = FakeLauncher()
    client = TestClient(app)

    response = client.post("/v1/launch/claude-code", headers={"X-BoBe-Launch-Token": "launch-token"})

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_config"


def test_claude_code_session_start_endpoint_calls_manager():
    from bobe.wake_daemon.server import create_app

    class FakeManager:
        def start(self):
            return {"ok": True, "session_id": "session-1"}

    app = create_app(_claude_code_launch_config())
    app.state.claude_code_session_manager = FakeManager()
    client = TestClient(app)

    response = client.post("/v1/claude-code/session/start", headers={"X-BoBe-Launch-Token": "launch-token"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-1"


def test_claude_code_session_send_endpoint_passes_command():
    from bobe.wake_daemon.server import create_app

    class FakeManager:
        def __init__(self):
            self.commands = []

        def send(self, command):
            self.commands.append(command)
            return {"ok": True, "output": "done"}

    manager = FakeManager()
    app = create_app(_claude_code_launch_config())
    app.state.claude_code_session_manager = manager
    client = TestClient(app)

    response = client.post(
        "/v1/claude-code/session/send",
        headers={"X-BoBe-Launch-Token": "launch-token"},
        json={"command": "run tests"},
    )

    assert response.status_code == 200
    assert response.json()["output"] == "done"
    assert manager.commands == ["run tests"]


def test_claude_code_session_send_endpoint_returns_202_for_accepted_command():
    from bobe.wake_daemon.server import create_app

    class FakeManager:
        def send(self, command):
            return {"ok": True, "accepted": True, "running": True, "session_id": "session-1"}

    app = create_app(_claude_code_launch_config())
    app.state.claude_code_session_manager = FakeManager()
    client = TestClient(app)

    response = client.post(
        "/v1/claude-code/session/send",
        headers={"X-BoBe-Launch-Token": "launch-token"},
        json={"command": "run tests"},
    )

    assert response.status_code == 202
    assert response.json()["accepted"] is True


def test_wake_daemon_lifespan_shuts_down_claude_session_manager(monkeypatch):
    """Daemon shutdown must terminate any active claude command (finding #25)."""
    from bobe.wake_daemon.server import create_app

    monkeypatch.setattr(WhisperWakeEngine, "preload", lambda self: None)

    class FakeManager:
        def __init__(self):
            self.shutdowns = 0

        def shutdown(self):
            self.shutdowns += 1

    app = create_app(load_wake_daemon_config(_TEST_ENV))
    manager = FakeManager()
    app.state.claude_code_session_manager = manager

    with TestClient(app):
        pass

    assert manager.shutdowns == 1


def test_claude_code_session_send_endpoint_rejects_empty_command():
    from bobe.wake_daemon.server import create_app

    class FakeManager:
        def send(self, command):
            return {"ok": False, "error": "empty_command"}

    app = create_app(_claude_code_launch_config())
    app.state.claude_code_session_manager = FakeManager()
    client = TestClient(app)

    response = client.post(
        "/v1/claude-code/session/send",
        headers={"X-BoBe-Launch-Token": "launch-token"},
        json={},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "empty_command"


def test_claude_code_session_status_and_stop_endpoints():
    from bobe.wake_daemon.server import create_app

    class FakeManager:
        def status(self):
            return {"ok": True, "active": True}

        def stop(self):
            return {"ok": True, "stopped_session_id": "session-1"}

    app = create_app(_claude_code_launch_config())
    app.state.claude_code_session_manager = FakeManager()
    client = TestClient(app)

    status_response = client.get("/v1/claude-code/session/status", headers={"X-BoBe-Launch-Token": "launch-token"})
    stop_response = client.post("/v1/claude-code/session/stop", headers={"X-BoBe-Launch-Token": "launch-token"})

    assert status_response.status_code == 200
    assert status_response.json()["active"] is True
    assert stop_response.status_code == 200
    assert stop_response.json()["stopped_session_id"] == "session-1"


def test_claude_code_session_endpoints_require_token():
    from bobe.wake_daemon.server import create_app

    client = TestClient(create_app(_claude_code_launch_config()))

    response = client.post("/v1/claude-code/session/start", headers={"X-BoBe-Launch-Token": "bad-token"})

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"
