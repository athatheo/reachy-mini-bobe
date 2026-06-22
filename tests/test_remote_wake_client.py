# ruff: noqa: D103
import json

from bobe.wake.remote_client import RemoteWakeClient


def test_remote_client_debug_state_is_json_serializable():
    client = RemoteWakeClient(lambda: None, url="ws://192.168.1.114:8765/v1/stream", token="secret")
    client._log_event("info", "Connected")
    client._apply_remote_stats(
        {
            "transcript": "hey jarvis",
            "rms": 512.0,
            "in_speech": True,
            "latency_ms_last": 82.5,
            "engine": "faster-whisper",
            "model": "tiny.en",
        }
    )
    state = client.debug_state()
    json.dumps(state)
    assert state["backend"] == "remote"
    assert state["remote_stats"]["transcript"] == "hey jarvis"
    assert len(state["events"]) == 1


def test_remote_client_wake_event_is_logged():
    client = RemoteWakeClient(lambda: None, url="ws://127.0.0.1:8765/v1/stream")
    client._log_event("wake", "Wake detected: 'hey jarvis'", latency_ms=75.0)
    events = client.debug_state()["events"]
    assert events[-1]["level"] == "wake"
    assert "hey jarvis" in str(events[-1]["message"])
