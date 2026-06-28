import os
import time
import base64
import logging
from typing import Any, Dict
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from huggingface_hub import snapshot_download

from bobe.config import config


try:
    import torch as _torch
except ImportError:  # pragma: no cover - exercised through import behavior
    _torch = None

torch: Any = _torch

try:
    from transformers import AutoProcessor as _AutoProcessor
    from transformers import AutoModelForImageTextToText as _AutoModelForImageTextToText
except ImportError:  # pragma: no cover - exercised through import behavior
    _AutoProcessor = None
    _AutoModelForImageTextToText = None

AutoProcessor: Any = _AutoProcessor
AutoModelForImageTextToText: Any = _AutoModelForImageTextToText


logger = logging.getLogger(__name__)


def _local_vision_dependencies_available() -> bool:
    return torch is not None and AutoProcessor is not None and AutoModelForImageTextToText is not None


def _is_cuda_oom(error: Exception) -> bool:
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    oom_error = getattr(cuda, "OutOfMemoryError", None)
    return isinstance(oom_error, type) and isinstance(error, oom_error)


@dataclass
class VisionConfig:
    """Configuration for vision processing."""

    model_path: str = config.LOCAL_VISION_MODEL
    max_new_tokens: int = 64
    jpeg_quality: int = 85
    max_retries: int = 3
    retry_delay: float = 1.0
    device_preference: str = "auto"  # "auto", "cuda", "cpu"


class VisionProcessor:
    """Handles SmolVLM2 model loading and inference."""

    def __init__(self, vision_config: VisionConfig | None = None):
        """Initialize the vision processor."""
        self.vision_config = vision_config or VisionConfig()
        self.model_path = self.vision_config.model_path
        self.device = self._determine_device()
        self.processor = None
        self.model = None
        self._initialized = False

    def _determine_device(self) -> str:
        pref = self.vision_config.device_preference
        if torch is None:
            return "cpu"
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if pref == "mps":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        # auto: prefer mps on Apple, then cuda, else cpu
        if torch.backends.mps.is_available():
            return "mps"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def initialize(self) -> bool:
        """Load model and processor onto the selected device."""
        if not _local_vision_dependencies_available():
            logger.error("Local vision dependencies missing; install the local_vision extra to enable this feature")
            return False

        try:
            logger.info(f"Loading SmolVLM2 model on {self.device} (HF_HOME={config.HF_HOME})")
            self.processor = AutoProcessor.from_pretrained(self.model_path)

            # Select dtype depending on device
            if self.device == "cuda":
                dtype = torch.bfloat16
            elif self.device == "mps":
                dtype = torch.float32  # best for MPS
            else:
                dtype = torch.float32

            model_kwargs: Dict[str, Any] = {"dtype": dtype}

            # flash_attention_2 is CUDA-only; skip on MPS/CPU
            if self.device == "cuda":
                model_kwargs["_attn_implementation"] = "flash_attention_2"

            # Load model weights
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_path, **model_kwargs).to(self.device)

            if self.model is not None:
                self.model.eval()
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            return False

    def process_image(
        self,
        cv2_image: NDArray[np.uint8],
        prompt: str = "Briefly describe what you see in one sentence.",
    ) -> str:
        """Process CV2 image and return description with retry logic."""
        if not self._initialized or self.processor is None or self.model is None:
            return "Vision model not initialized"

        for attempt in range(self.vision_config.max_retries):
            try:
                # Convert to JPEG bytes
                success, jpeg_buffer = cv2.imencode(
                    ".jpg",
                    cv2_image,
                    [cv2.IMWRITE_JPEG_QUALITY, self.vision_config.jpeg_quality],
                )
                if not success:
                    return "Failed to encode image"

                # Convert to base64
                image_base64 = base64.b64encode(jpeg_buffer.tobytes()).decode("utf-8")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )

                # Move tensors to device WITHOUT forcing dtype (keeps input_ids as torch.long)
                inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self.vision_config.max_new_tokens,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                generated_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                # Extract just the response part
                full_text = generated_texts[0]
                response = self._extract_response(full_text)

                # Clean up GPU memory if using CUDA
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                elif self.device == "mps":
                    torch.mps.empty_cache()

                return response.replace(chr(10), " ").strip()

            except Exception as e:
                if _is_cuda_oom(e):
                    logger.error("CUDA OOM on attempt %d: %s", attempt + 1, e)
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    if attempt < self.vision_config.max_retries - 1:
                        time.sleep(self.vision_config.retry_delay * (attempt + 1))
                    else:
                        return "GPU out of memory - vision processing failed"
                    continue

                logger.error("Vision processing failed (attempt %d): %s", attempt + 1, e)
                if attempt < self.vision_config.max_retries - 1:
                    time.sleep(self.vision_config.retry_delay)
                else:
                    return f"Vision processing error after {self.vision_config.max_retries} attempts"

    def _extract_response(self, full_text: str) -> str:
        """Extract the assistant's response from the full generated text."""
        # Handle different response formats
        markers = ["assistant\n", "Assistant:", "Response:", "\n\n"]

        for marker in markers:
            if marker in full_text:
                response = full_text.split(marker)[-1].strip()
                if response:  # Ensure we got a meaningful response
                    return response

        # Fallback: return the full text cleaned up
        return full_text.strip()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        return {
            "initialized": self._initialized,
            "device": self.device,
            "model_path": self.model_path,
            "cuda_available": cuda_available,
            "gpu_memory": torch.cuda.get_device_properties(0).total_memory // (1024**3)
            if cuda_available
            else "N/A",
        }


@dataclass
class LocalVision:
    """Lazy-loaded local SmolVLM processor for the camera tool."""

    processor: VisionProcessor


def initialize_local_vision(camera_worker: Any) -> LocalVision | None:
    """Download and initialize the local vision model for on-demand camera tool use."""
    _ = camera_worker  # retained for call-site compatibility
    try:
        model_id = config.LOCAL_VISION_MODEL
        cache_dir = os.path.expanduser(config.HF_HOME)

        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir
        logger.info("HF_HOME set to %s", cache_dir)

        logger.info("Downloading vision model %s to cache...", model_id)
        snapshot_download(
            repo_id=model_id,
            repo_type="model",
            cache_dir=cache_dir,
        )
        logger.info("Model %s downloaded to %s", model_id, cache_dir)

        vision_config = VisionConfig(
            model_path=model_id,
            max_new_tokens=64,
            jpeg_quality=85,
            max_retries=3,
            retry_delay=1.0,
            device_preference="auto",
        )
        processor = VisionProcessor(vision_config)
        if not processor.initialize():
            return None

        device_info = processor.get_model_info()
        logger.info(
            "Local vision enabled: %s on %s",
            device_info.get("model_path"),
            device_info.get("device"),
        )
        return LocalVision(processor=processor)

    except Exception as e:
        logger.error("Failed to initialize local vision: %s", e)
        return None
