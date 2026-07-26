"""Tests for the vision processing module (processors, trackers, camera worker/tools)."""

import sys
import time
import types
import base64
import importlib
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytest

from bobe.tools.camera import Camera
from bobe.camera_worker import CameraWorker
from bobe.vision.processors import (
    LocalVision,
    VisionConfig,
    VisionProcessor,
    initialize_local_vision,
)
from bobe.tools.head_tracking import HeadTracking


def test_vision_config_defaults() -> None:
    """Test VisionConfig has sensible defaults."""
    config = VisionConfig()
    assert config.max_new_tokens == 64
    assert config.jpeg_quality == 85
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.device_preference == "auto"


def test_vision_config_custom_values() -> None:
    """Test VisionConfig accepts custom values."""
    config = VisionConfig(
        model_path="/custom/path",
        max_new_tokens=128,
        jpeg_quality=95,
        max_retries=5,
        retry_delay=2.0,
        device_preference="cpu",
    )
    assert config.model_path == "/custom/path"
    assert config.max_new_tokens == 128
    assert config.jpeg_quality == 95
    assert config.max_retries == 5
    assert config.retry_delay == 2.0
    assert config.device_preference == "cpu"



@pytest.fixture
def mock_torch() -> Any:
    """Mock torch module to avoid loading actual models."""
    with patch("bobe.vision.processors.torch") as mock:
        mock.cuda.is_available.return_value = False
        mock.backends.mps.is_available.return_value = False
        mock.float32 = "float32"
        mock.bfloat16 = "bfloat16"
        yield mock


@pytest.fixture
def mock_transformers() -> Any:
    """Mock transformers module."""
    with patch("bobe.vision.processors.AutoProcessor") as proc, \
         patch("bobe.vision.processors.AutoModelForImageTextToText") as model:

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.apply_chat_template.return_value = {
            "input_ids": MagicMock(to=lambda x: MagicMock()),
            "attention_mask": MagicMock(to=lambda x: MagicMock()),
            "pixel_values": MagicMock(to=lambda x: MagicMock()),
        }
        mock_processor.batch_decode.return_value = ["assistant\nThis is a test description."]
        mock_processor.tokenizer.eos_token_id = 2
        proc.from_pretrained.return_value = mock_processor

        # Mock model
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = None
        mock_model_instance.generate.return_value = [[1, 2, 3]]
        mock_model_instance.to.return_value = mock_model_instance
        model.from_pretrained.return_value = mock_model_instance

        yield {"processor": proc, "model": model}


def test_vision_processor_device_selection_cpu(mock_torch: Any) -> None:
    """Test VisionProcessor selects CPU when specified."""
    config = VisionConfig(device_preference="cpu")
    processor = VisionProcessor(config)
    assert processor.device == "cpu"


def test_vision_processor_device_selection_cuda_unavailable(mock_torch: Any) -> None:
    """Test VisionProcessor falls back to CPU when CUDA unavailable."""
    mock_torch.cuda.is_available.return_value = False
    config = VisionConfig(device_preference="cuda")
    processor = VisionProcessor(config)
    assert processor.device == "cpu"


def test_vision_processor_device_selection_cuda_available(mock_torch: Any) -> None:
    """Test VisionProcessor selects CUDA when available."""
    mock_torch.cuda.is_available.return_value = True
    config = VisionConfig(device_preference="cuda")
    processor = VisionProcessor(config)
    assert processor.device == "cuda"


def test_vision_processor_device_selection_mps_available(mock_torch: Any) -> None:
    """Test VisionProcessor selects MPS when available on Apple Silicon."""
    mock_torch.backends.mps.is_available.return_value = True
    config = VisionConfig(device_preference="mps")
    processor = VisionProcessor(config)
    assert processor.device == "mps"


def test_vision_processor_device_selection_auto_prefers_mps(mock_torch: Any) -> None:
    """Test VisionProcessor auto mode prefers MPS on Apple Silicon."""
    mock_torch.backends.mps.is_available.return_value = True
    mock_torch.cuda.is_available.return_value = False
    config = VisionConfig(device_preference="auto")
    processor = VisionProcessor(config)
    assert processor.device == "mps"


def test_vision_processor_device_selection_auto_prefers_cuda_over_cpu(mock_torch: Any) -> None:
    """Test VisionProcessor auto mode prefers CUDA over CPU."""
    mock_torch.backends.mps.is_available.return_value = False
    mock_torch.cuda.is_available.return_value = True
    config = VisionConfig(device_preference="auto")
    processor = VisionProcessor(config)
    assert processor.device == "cuda"


def test_vision_processor_initialization(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor initializes successfully."""
    config = VisionConfig(model_path="test/model")
    processor = VisionProcessor(config)

    assert not processor._initialized
    result = processor.initialize()

    assert result is True
    assert processor._initialized
    mock_transformers["processor"].from_pretrained.assert_called_once_with("test/model")
    mock_transformers["model"].from_pretrained.assert_called_once()


def test_vision_processor_initialization_failure(mock_torch: Any) -> None:
    """Test VisionProcessor handles initialization failure gracefully."""
    with patch("bobe.vision.processors.AutoProcessor") as mock_proc:
        mock_proc.from_pretrained.side_effect = Exception("Model not found")

        config = VisionConfig(model_path="invalid/model")
        processor = VisionProcessor(config)
        result = processor.initialize()

        assert result is False
        assert not processor._initialized


def test_vision_processor_process_image_not_initialized(mock_torch: Any) -> None:
    """Test process_image returns error when model not initialized."""
    processor = VisionProcessor()
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)

    result = processor.process_image(test_image)
    assert result == "Vision model not initialized"


def test_vision_processor_process_image_success(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image processes an image successfully."""
    with patch("bobe.vision.processors.cv2") as mock_cv2:
        # Mock cv2.imencode to return success
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image, "Describe this image.")

        assert isinstance(result, str)
        assert result == "This is a test description."


def test_vision_processor_process_image_encode_failure(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image handles image encoding failure."""
    with patch("bobe.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (False, None)
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert result == "Failed to encode image"


def test_vision_processor_process_image_with_retry(mock_torch: Any, mock_transformers: Any) -> None:
    """Test process_image retries on failure."""
    with patch("bobe.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        # Set up the OutOfMemoryError to be a proper exception
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})

        processor = VisionProcessor(VisionConfig(max_retries=3, retry_delay=0.01))
        processor.initialize()

        # Make the model generate fail twice, then succeed
        call_count = [0]
        assert processor.model is not None
        original_generate = processor.model.generate

        def failing_generate(*args: Any, **kwargs: Any) -> Any:
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return original_generate(*args, **kwargs)

        processor.model.generate = failing_generate

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert isinstance(result, str)
        assert call_count[0] == 3


def test_vision_processor_extract_response_variants() -> None:
    """Test _extract_response handles different response formats."""
    processor = VisionProcessor()

    # Test with "assistant\n" marker
    result = processor._extract_response("user prompt\nassistant\nThe response text")
    assert result == "The response text"

    # Test with "Assistant:" marker
    result = processor._extract_response("User: prompt\nAssistant: Another response")
    assert result == "Another response"

    # Test fallback to full text
    result = processor._extract_response("Just some text without markers")
    assert result == "Just some text without markers"


def test_vision_processor_get_model_info(mock_torch: Any, mock_transformers: Any) -> None:
    """Test get_model_info returns correct information."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_properties.return_value.total_memory = 8 * 1024**3

    processor = VisionProcessor(VisionConfig(model_path="test/model", device_preference="cpu"))
    processor.initialize()

    info = processor.get_model_info()

    assert info["initialized"] is True
    assert info["device"] == "cpu"
    assert info["model_path"] == "test/model"
    assert "cuda_available" in info


def test_initialize_local_vision_success(mock_torch: Any, mock_transformers: Any) -> None:
    """Test initialize_local_vision creates LocalVision successfully."""
    with patch("bobe.vision.processors.snapshot_download") as mock_download, \
         patch("bobe.vision.processors.os.makedirs"), \
         patch("bobe.vision.processors.config") as mock_config:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"

        result = initialize_local_vision()

        assert result is not None
        assert isinstance(result, LocalVision)
        assert result.processor._initialized
        mock_download.assert_called_once()


def test_initialize_local_vision_download_failure(mock_torch: Any) -> None:
    """Test initialize_local_vision handles download failure."""
    with patch("bobe.vision.processors.snapshot_download") as mock_download, \
         patch("bobe.vision.processors.os.makedirs"), \
         patch("bobe.vision.processors.config") as mock_config:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"
        mock_download.side_effect = Exception("Network error")

        result = initialize_local_vision()

        assert result is None


def test_initialize_local_vision_processor_failure(mock_torch: Any) -> None:
    """Test initialize_local_vision handles processor initialization failure."""
    with patch("bobe.vision.processors.snapshot_download"), \
         patch("bobe.vision.processors.os.makedirs"), \
         patch("bobe.vision.processors.config") as mock_config, \
         patch("bobe.vision.processors.AutoProcessor") as mock_proc:

        mock_config.LOCAL_VISION_MODEL = "test/model"
        mock_config.HF_HOME = "/tmp/hf_cache"
        mock_proc.from_pretrained.side_effect = Exception("Model load error")

        result = initialize_local_vision()

        assert result is None


def test_vision_processor_cuda_oom_recovery(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor recovers from CUDA OOM errors."""
    with patch("bobe.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor(VisionConfig(max_retries=2, retry_delay=0.01))
        processor.initialize()
        processor.device = "cuda"  # Force CUDA for this test

        # Make generate raise OOM error
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (Exception,), {})
        assert processor.model is not None
        processor.model.generate.side_effect = mock_torch.cuda.OutOfMemoryError("OOM")

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = processor.process_image(test_image)

        assert "GPU out of memory" in result
        mock_torch.cuda.empty_cache.assert_called()


def test_vision_processor_cache_cleanup_mps(mock_torch: Any, mock_transformers: Any) -> None:
    """Test VisionProcessor cleans up MPS cache after processing."""
    with patch("bobe.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()
        processor.device = "mps"  # Force MPS for this test

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        processor.process_image(test_image)

        # Should call mps empty_cache
        mock_torch.mps.empty_cache.assert_called()


def test_vision_processor_process_image_serializes_concurrent_calls(
    mock_torch: Any, mock_transformers: Any,
) -> None:
    """Concurrent process_image calls must never run model.generate in parallel."""
    with patch("bobe.vision.processors.cv2") as mock_cv2:
        mock_cv2.imencode.return_value = (True, np.array([1, 2, 3], dtype=np.uint8))
        mock_cv2.IMWRITE_JPEG_QUALITY = 1

        processor = VisionProcessor()
        processor.initialize()

        active = 0
        max_active = 0
        counter_lock = threading.Lock()

        def tracking_generate(*args: Any, **kwargs: Any) -> Any:
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
            return [[1, 2, 3]]

        assert processor.model is not None
        processor.model.generate = tracking_generate

        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(processor.process_image, test_image) for _ in range(2)]
            results = [f.result(timeout=5) for f in futures]

        assert max_active == 1  # generate was never entered concurrently
        assert results == ["This is a test description."] * 2


# ---------------------------------------------------------------------------
# YOLO head tracker (degrades to no-tracking when the model cannot load)
# ---------------------------------------------------------------------------


@pytest.fixture
def yolo_tracker_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import bobe.vision.yolo_head_tracker with stubbed ultralytics/supervision."""

    class _FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = path

        def to(self, device: str) -> "_FakeYOLO":
            return self

    ultralytics_stub = types.ModuleType("ultralytics")
    ultralytics_stub.YOLO = _FakeYOLO  # type: ignore[attr-defined]
    supervision_stub = types.ModuleType("supervision")
    supervision_stub.Detections = type("Detections", (), {})  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics_stub)
    monkeypatch.setitem(sys.modules, "supervision", supervision_stub)
    sys.modules.pop("bobe.vision.yolo_head_tracker", None)
    module = importlib.import_module("bobe.vision.yolo_head_tracker")
    yield module
    sys.modules.pop("bobe.vision.yolo_head_tracker", None)


def test_head_tracker_degrades_when_model_load_fails(
    yolo_tracker_module: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed HuggingFace download must not raise; tracking degrades to no-op."""

    def failing_download(**kwargs: Any) -> str:
        raise OSError("no internet and no cached model")

    monkeypatch.setattr(yolo_tracker_module, "hf_hub_download", failing_download)

    tracker = yolo_tracker_module.HeadTracker()  # must not raise

    assert tracker.model is None
    assert tracker.available is False

    eye_center, roll = tracker.get_head_position(np.zeros((32, 32, 3), dtype=np.uint8))
    assert eye_center is None
    assert roll is None


def test_head_tracker_available_when_model_loads(
    yolo_tracker_module: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful model load reports the tracker as available and BGR-native."""
    monkeypatch.setattr(yolo_tracker_module, "hf_hub_download", lambda **kwargs: "/tmp/model.pt")

    tracker = yolo_tracker_module.HeadTracker()

    assert tracker.available is True
    assert tracker.expects_rgb is False


# ---------------------------------------------------------------------------
# CameraWorker: frame staleness, return-to-neutral on camera death, RGB trackers
# ---------------------------------------------------------------------------


def _make_worker(head_tracker: Any = None) -> CameraWorker:
    return CameraWorker(MagicMock(), head_tracker=head_tracker)


def test_get_latest_frame_returns_none_before_first_capture() -> None:
    """No frame captured yet -> None."""
    worker = _make_worker()
    assert worker.get_latest_frame() is None


def test_get_latest_frame_returns_fresh_frame() -> None:
    """A recently captured frame is served as a copy."""
    worker = _make_worker()
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    with worker.frame_lock:
        worker.latest_frame = frame
        worker.latest_frame_time = time.monotonic()

    result = worker.get_latest_frame()
    assert result is not None
    assert np.array_equal(result, frame)
    assert result is not frame  # copy, not the buffer itself


def test_get_latest_frame_returns_none_when_stale() -> None:
    """Frames older than frame_max_age_s are treated as unavailable."""
    worker = _make_worker()
    with worker.frame_lock:
        worker.latest_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        worker.latest_frame_time = time.monotonic() - worker.frame_max_age_s - 1.0

    assert worker.get_latest_frame() is None


def _run_until_offsets_neutral(worker: CameraWorker, timeout: float = 3.0) -> None:
    worker.face_lost_delay = 0.0
    worker.interpolation_duration = 0.05
    with worker.face_tracking_lock:
        worker.face_tracking_offsets = [0.01, 0.02, 0.03, 0.1, 0.2, 0.3]
    worker.last_face_detected_time = time.time()

    worker.start()
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            offsets = worker.get_face_tracking_offsets()
            if all(abs(v) < 1e-6 for v in offsets):
                break
            time.sleep(0.02)
    finally:
        worker.stop()

    assert all(abs(v) < 1e-6 for v in worker.get_face_tracking_offsets())
    assert worker.last_face_detected_time is None


def test_offsets_return_to_neutral_when_camera_returns_none() -> None:
    """Face-tracking offsets must decay to neutral even if get_frame() yields None forever."""
    worker = _make_worker()
    worker.reachy_mini.media.get_frame.return_value = None
    _run_until_offsets_neutral(worker)


def test_offsets_return_to_neutral_when_camera_raises() -> None:
    """Face-tracking offsets must decay to neutral even if get_frame() raises forever."""
    worker = _make_worker()
    worker.reachy_mini.media.get_frame.side_effect = RuntimeError("camera stream died")
    _run_until_offsets_neutral(worker)


class _RecordingTracker:
    """Fake MediaPipe-style tracker that records the frames it receives."""

    expects_rgb = True

    def __init__(self) -> None:
        self.frames: list = []

    def get_head_position(self, img: Any) -> Any:
        self.frames.append(img)
        return None, None


def test_rgb_expecting_tracker_receives_converted_frame() -> None:
    """A tracker declaring expects_rgb=True must get an RGB frame, not raw BGR."""
    tracker = _RecordingTracker()
    worker = _make_worker(head_tracker=tracker)

    bgr_frame = np.zeros((8, 8, 3), dtype=np.uint8)
    bgr_frame[..., 0] = 10  # B
    bgr_frame[..., 1] = 20  # G
    bgr_frame[..., 2] = 30  # R
    worker.reachy_mini.media.get_frame.return_value = bgr_frame

    worker.start()
    try:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not tracker.frames:
            time.sleep(0.02)
    finally:
        worker.stop()

    assert tracker.frames, "tracker never received a frame"
    expected_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    assert np.array_equal(tracker.frames[0], expected_rgb)
    # The buffered frame stays in original BGR for tool consumers
    latest = worker.get_latest_frame()
    assert latest is not None
    assert np.array_equal(latest, bgr_frame)


def test_tracker_wants_rgb_detection() -> None:
    """RGB detection honors declared attribute and falls back to module heuristics."""
    assert CameraWorker._tracker_wants_rgb(None) is False
    assert CameraWorker._tracker_wants_rgb(_RecordingTracker()) is True

    class _DeclaredBgr:
        expects_rgb = False

    assert CameraWorker._tracker_wants_rgb(_DeclaredBgr()) is False

    toolbox_cls = type("HeadTracker", (), {})
    toolbox_cls.__module__ = "reachy_mini_toolbox.vision.head_tracker"
    assert CameraWorker._tracker_wants_rgb(toolbox_cls()) is True

    yolo_cls = type("HeadTracker", (), {})
    yolo_cls.__module__ = "bobe.vision.yolo_head_tracker"
    assert CameraWorker._tracker_wants_rgb(yolo_cls()) is False


# ---------------------------------------------------------------------------
# Camera tool: threaded JPEG encode, stale-frame error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_camera_tool_returns_b64_jpeg() -> None:
    """Without a vision manager the tool returns the stable {'b64_im': ...} shape."""
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = frame
    deps = SimpleNamespace(camera_worker=camera_worker, vision_manager=None)

    result = await Camera()(deps, question="what do you see?")

    assert set(result.keys()) == {"b64_im"}
    decoded = base64.b64decode(result["b64_im"])
    assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes


@pytest.mark.asyncio
async def test_camera_tool_errors_when_no_fresh_frame() -> None:
    """A stale/absent frame surfaces a clear error instead of stale imagery."""
    camera_worker = MagicMock()
    camera_worker.get_latest_frame.return_value = None
    deps = SimpleNamespace(camera_worker=camera_worker, vision_manager=None)

    result = await Camera()(deps, question="what do you see?")

    assert "error" in result
    assert "b64_im" not in result


# ---------------------------------------------------------------------------
# head_tracking tool: honest errors when no tracker is configured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_head_tracking_tool_errors_without_camera_worker() -> None:
    deps = SimpleNamespace(camera_worker=None)

    result = await HeadTracking()(deps, start=True)

    assert "error" in result
    assert "status" not in result


@pytest.mark.asyncio
async def test_head_tracking_tool_errors_without_tracker() -> None:
    camera_worker = MagicMock()
    camera_worker.head_tracker = None
    deps = SimpleNamespace(camera_worker=camera_worker)

    result = await HeadTracking()(deps, start=True)

    assert "error" in result
    camera_worker.set_head_tracking_enabled.assert_not_called()


@pytest.mark.asyncio
async def test_head_tracking_tool_errors_with_degraded_tracker() -> None:
    camera_worker = MagicMock()
    camera_worker.head_tracker = SimpleNamespace(available=False)
    deps = SimpleNamespace(camera_worker=camera_worker)

    result = await HeadTracking()(deps, start=True)

    assert "error" in result
    camera_worker.set_head_tracking_enabled.assert_not_called()


@pytest.mark.asyncio
async def test_head_tracking_tool_toggles_with_working_tracker() -> None:
    camera_worker = MagicMock()
    camera_worker.head_tracker = SimpleNamespace(available=True)
    deps = SimpleNamespace(camera_worker=camera_worker)

    result = await HeadTracking()(deps, start=True)
    assert result == {"status": "head tracking started"}
    camera_worker.set_head_tracking_enabled.assert_called_once_with(True)

    result = await HeadTracking()(deps, start=False)
    assert result == {"status": "head tracking stopped"}
    camera_worker.set_head_tracking_enabled.assert_called_with(False)

