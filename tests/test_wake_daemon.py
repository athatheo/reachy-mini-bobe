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


def test_whisper_engine_detects_sleep_command_with_fillers(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "Okay Bobe, please go to sleep now.")
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


def test_whisper_engine_ignores_sleep_phrase_inside_conversation(monkeypatch):
    """Substring matching used to sleep on 'my toddler won't go to sleep'."""
    session = _session(
        monkeypatch=monkeypatch,
        transcribe=lambda _audio: "My toddler won't go to sleep, any tips?",
    )
    session.set_listen_mode("sleep")

    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    events = [session.feed(pcm[offset : offset + 1600]) for offset in range(0, pcm.size, 1600)]

    assert all(event is None for event in events)


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


def test_wake_daemon_app_starts_without_engine():
    from bobe.wake_daemon.server import create_app

    config = load_wake_daemon_config(_TEST_ENV)
    app = create_app(config)
    assert app.state.wake_engine is None


def test_create_app_preloads_whisper_model_on_startup(monkeypatch):
    from bobe.wake_daemon.server import create_app

    preloaded = []
    monkeypatch.setattr(WhisperWakeEngine, "preload", lambda self: preloaded.append(self))
    app = create_app(load_wake_daemon_config(_TEST_ENV))

    with TestClient(app):
        app.state.whisper_preload_thread.join(timeout=5)

    assert len(preloaded) == 1
    assert [app.state.wake_engine] == preloaded


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



def _announce_client():
    from bobe.wake_daemon.server import create_app

    return TestClient(create_app(load_wake_daemon_config(_TEST_ENV)))


def test_announce_rejects_bad_token():
    client = _announce_client()

    response = client.post(
        "/v1/announce",
        headers={"X-BoBe-Wake-Token": "wrong-token"},
        json={"message": "hello"},
    )

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized"


def test_announce_rejects_empty_message():
    client = _announce_client()

    response = client.post(
        "/v1/announce",
        headers={"X-BoBe-Wake-Token": "test-token"},
        json={},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "empty_message"


def test_announce_without_robot_returns_conflict():
    client = _announce_client()

    response = client.post(
        "/v1/announce",
        headers={"X-BoBe-Wake-Token": "test-token"},
        json={"message": "hello"},
    )

    assert response.status_code == 409
    assert response.json()["error"] == "no_robot_connected"


def test_announce_forwards_to_connected_robot_stream():
    client = _announce_client()

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"

        response = client.post(
            "/v1/announce",
            headers={"X-BoBe-Wake-Token": "test-token"},
            json={"message": "Build finished."},
        )

        assert response.status_code == 200
        assert response.json() == {"ok": True, "delivered": 1}
        frame = ws.receive_json()

    assert frame == {"type": "announce", "text": "Build finished."}


def test_announce_after_stream_disconnect_returns_conflict():
    client = _announce_client()

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"

    response = client.post(
        "/v1/announce",
        headers={"X-BoBe-Wake-Token": "test-token"},
        json={"message": "hello"},
    )

    assert response.status_code == 409


# ---- /v1/speak ----


def _wav_bytes(pcm, rate=24000):
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_speak_rejects_bad_token():
    client = _announce_client()

    response = client.post(
        "/v1/speak",
        headers={"X-BoBe-Wake-Token": "wrong-token", "Content-Type": "audio/wav"},
        content=_wav_bytes(np.zeros(100, dtype=np.int16)),
    )

    assert response.status_code == 401


def test_speak_rejects_missing_audio():
    client = _announce_client()

    response = client.post(
        "/v1/speak",
        headers={"X-BoBe-Wake-Token": "test-token"},
        json={},
    )

    assert response.status_code == 400
    assert response.json()["error"] == "missing_audio"


def test_speak_rejects_undecodable_audio():
    client = _announce_client()

    response = client.post(
        "/v1/speak",
        headers={"X-BoBe-Wake-Token": "test-token", "Content-Type": "audio/wav"},
        content=b"this is not audio at all",
    )

    assert response.status_code == 400
    assert response.json()["error"].startswith("undecodable_audio")


def test_speak_without_robot_returns_conflict():
    client = _announce_client()

    response = client.post(
        "/v1/speak",
        headers={"X-BoBe-Wake-Token": "test-token", "Content-Type": "audio/wav"},
        content=_wav_bytes(np.zeros(2400, dtype=np.int16)),
    )

    assert response.status_code == 409
    assert response.json()["error"] == "no_robot_connected"


def test_speak_relays_chunked_pcm_to_robot_stream():
    import base64 as b64

    client = _announce_client()
    rng = np.random.default_rng(7)
    pcm = rng.integers(-2000, 2000, size=60000, dtype=np.int16)  # 2.5 s @ 24 kHz

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"

        response = client.post(
            "/v1/speak",
            headers={"X-BoBe-Wake-Token": "test-token", "Content-Type": "audio/wav"},
            content=_wav_bytes(pcm),
        )

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["delivered"] == 1
        assert body["seconds"] == 2.5

        frames = []
        while True:
            frame = ws.receive_json()
            assert frame["type"] == "speak"
            frames.append(frame)
            if frame["last"]:
                break

    assert [frame["seq"] for frame in frames] == [0, 1, 2]
    assert {frame["id"] for frame in frames} == {frames[0]["id"]}
    assert all(frame["rate"] == 24000 for frame in frames)
    assert [frame["last"] for frame in frames] == [False, False, True]
    rebuilt = np.concatenate(
        [np.frombuffer(b64.b64decode(frame["pcm_b64"]), dtype=np.int16) for frame in frames]
    )
    assert rebuilt.size == pcm.size
    assert np.array_equal(rebuilt, pcm)


def test_speak_accepts_base64_json_payload():
    import base64 as b64

    client = _announce_client()
    pcm = np.ones(24000, dtype=np.int16) * 1000

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"

        response = client.post(
            "/v1/speak",
            headers={"X-BoBe-Wake-Token": "test-token"},
            json={"audio_b64": b64.b64encode(_wav_bytes(pcm)).decode("ascii")},
        )

        assert response.status_code == 200
        frame = ws.receive_json()

    assert frame["type"] == "speak"
    assert frame["last"] is True


def test_decode_audio_rejects_empty_and_oversized():
    from bobe.wake_daemon.audio import MAX_AUDIO_BYTES, AudioDecodeError, decode_audio_to_pcm

    with pytest.raises(AudioDecodeError):
        decode_audio_to_pcm(b"")
    with pytest.raises(AudioDecodeError):
        decode_audio_to_pcm(b"\0" * (MAX_AUDIO_BYTES + 1))


def test_decode_audio_wav_stdlib_fallback_resamples(monkeypatch):
    from bobe.wake_daemon import audio as audio_module

    monkeypatch.setattr(audio_module, "_find_ffmpeg", lambda: None)
    pcm = np.ones(16000, dtype=np.int16) * 2000  # 1 s @ 16 kHz
    decoded = audio_module.decode_audio_to_pcm(_wav_bytes(pcm, rate=16000), rate=24000)

    assert decoded.dtype == np.int16
    assert decoded.size == 24000


# ---- converse mode ----


def _feed_utterance(session, transcribe_result):
    """Push a voiced utterance plus trailing silence through the session."""
    pcm = np.zeros(16000, dtype=np.int16)
    pcm[:8000] = 5000
    event = None
    for offset in range(0, pcm.size, 1600):
        maybe = session.feed(pcm[offset : offset + 1600])
        if maybe is not None:
            event = maybe
    return event


def test_converse_mode_emits_utterance_event(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "what's the weather today")
    session.set_listen_mode("converse")

    event = _feed_utterance(session, "what's the weather today")

    assert event is not None
    assert event["type"] == "utterance"
    assert event["transcript"] == "what's the weather today"


def test_converse_mode_still_detects_sleep_phrase(monkeypatch):
    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: "go to sleep")
    session.set_listen_mode("converse")

    event = _feed_utterance(session, "go to sleep")

    assert event is not None
    assert event["type"] == "sleep"


def test_converse_mode_does_not_emit_utterance_on_partial(monkeypatch):
    calls = []

    def transcribe(_audio):
        calls.append(True)
        return "tell me a story"

    session = _session(monkeypatch=monkeypatch, transcribe=transcribe)
    session.set_listen_mode("converse")

    # Feed voiced audio WITHOUT trailing silence: partials run, no final.
    pcm = np.full(16000, 5000, dtype=np.int16)
    events = [session.feed(pcm[offset : offset + 1600]) for offset in range(0, pcm.size, 1600)]

    assert all(event is None for event in events)
    assert calls  # partial transcription did run


def test_stream_converse_mode_enqueues_utterances(monkeypatch):
    from bobe.wake_daemon.server import create_app

    app = create_app(load_wake_daemon_config(_TEST_ENV))
    client = TestClient(app)

    def fake_transcribe(self, pcm, *, config=None):
        return "hello agent"

    monkeypatch.setattr(engine_module.WhisperWakeEngine, "transcribe", fake_transcribe)

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "listen", "mode": "converse", "sleep_phrases": ["go to sleep"]})

        pcm = np.zeros(16000, dtype=np.int16)
        pcm[:8000] = 5000
        for offset in range(0, pcm.size, 1600):
            ws.send_bytes(pcm[offset : offset + 1600].tobytes())

        response = client.get(
            "/v1/utterances",
            params={"wait": 5},
            headers={"X-BoBe-Wake-Token": "test-token"},
        )

    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) >= 1
    assert events[0]["text"] == "hello agent"


def test_utterances_rejects_bad_token():
    client = _announce_client()

    response = client.get("/v1/utterances", headers={"X-BoBe-Wake-Token": "nope"})

    assert response.status_code == 401


def test_inject_utterance_roundtrip():
    client = _announce_client()

    post = client.post(
        "/v1/utterances",
        headers={"X-BoBe-Wake-Token": "test-token"},
        json={"text": "ping"},
    )
    assert post.status_code == 200

    got = client.get(
        "/v1/utterances",
        params={"wait": 0},
        headers={"X-BoBe-Wake-Token": "test-token"},
    )
    assert got.status_code == 200
    assert [event["text"] for event in got.json()["events"]] == ["ping"]


# ---- /v1/emote ----


def test_emote_rejects_bad_token():
    client = _announce_client()

    response = client.post(
        "/v1/emote",
        headers={"X-BoBe-Wake-Token": "wrong"},
        json={"emotion": "amazed1"},
    )

    assert response.status_code == 401


def test_emote_requires_emotion_and_robot():
    client = _announce_client()

    empty = client.post("/v1/emote", headers={"X-BoBe-Wake-Token": "test-token"}, json={})
    assert empty.status_code == 400

    no_robot = client.post(
        "/v1/emote", headers={"X-BoBe-Wake-Token": "test-token"}, json={"emotion": "amazed1"}
    )
    assert no_robot.status_code == 409


def test_emote_relays_to_robot_stream():
    client = _announce_client()

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"

        response = client.post(
            "/v1/emote",
            headers={"X-BoBe-Wake-Token": "test-token"},
            json={"emotion": "amazed1"},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": True, "delivered": 1}
        frame = ws.receive_json()

    assert frame == {"type": "emote", "emotion": "amazed1"}


def test_converse_mode_drops_junk_utterances(monkeypatch):
    from bobe.wake_daemon.engine import is_junk_utterance

    assert is_junk_utterance(". . . . . .")
    assert is_junk_utterance("You")
    assert is_junk_utterance("Thank you.")
    assert not is_junk_utterance("yes")
    assert not is_junk_utterance("no")
    assert not is_junk_utterance("stop the music")

    session = _session(monkeypatch=monkeypatch, transcribe=lambda _audio: ". . . . . .")
    session.set_listen_mode("converse")
    assert _feed_utterance(session, ". . . . . .") is None


# ---- presence + morning briefing ----


def _brief_env(tmp_path, hour=0):
    return {
        **_TEST_ENV,
        "BOBE_BRIEF_AFTER_HOUR": str(hour),
        "BOBE_BRIEF_STATE_FILE": str(tmp_path / "brief-state"),
    }


def test_presence_fires_briefing_once_per_day(tmp_path):
    from bobe.wake_daemon.server import MORNING_BRIEF_PROMPT, create_app

    app = create_app(load_wake_daemon_config(_brief_env(tmp_path, hour=0)))
    client = TestClient(app)

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "presence"})
        ws.send_json({"type": "presence"})

        got = client.get(
            "/v1/utterances", params={"wait": 2}, headers={"X-BoBe-Wake-Token": "test-token"}
        )
        status = client.get("/v1/presence", headers={"X-BoBe-Wake-Token": "test-token"})

    events = got.json()["events"]
    assert [e["text"] for e in events] == [MORNING_BRIEF_PROMPT]
    body = status.json()
    assert body["last_presence_at"] is not None
    assert body["brief_fired_on"] is not None


def test_presence_respects_after_hour_gate(tmp_path):
    from bobe.wake_daemon.server import create_app

    # Hour 25 can never be reached: the gate must hold all day.
    app = create_app(load_wake_daemon_config(_brief_env(tmp_path, hour=25)))
    client = TestClient(app)

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "presence"})

        got = client.get(
            "/v1/utterances", params={"wait": 0}, headers={"X-BoBe-Wake-Token": "test-token"}
        )

    assert got.json()["events"] == []


def test_presence_disabled_with_negative_hour(tmp_path):
    from bobe.wake_daemon.server import create_app

    app = create_app(load_wake_daemon_config(_brief_env(tmp_path, hour=-1)))
    client = TestClient(app)

    with client.websocket_connect("/v1/stream") as ws:
        ws.send_json({"type": "hello", "token": "test-token", "sample_rate": 16000, "phrase": "hey bobe"})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "presence"})
        got = client.get(
            "/v1/utterances", params={"wait": 0}, headers={"X-BoBe-Wake-Token": "test-token"}
        )

    assert got.json()["events"] == []
