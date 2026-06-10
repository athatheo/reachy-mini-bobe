#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

uv run python scripts/seed_heed_training_data.py
uv run heed train wake_training/hey_jarvis --tts-pos 400 --model-size medium --epochs 35
uv run heed export wake_training/hey_jarvis --output src/bobe/wake_models/hey_jarvis
uv run python - <<'PY'
import base64
from pathlib import Path
src = Path('src/bobe/wake_models/hey_jarvis/wake.onnx')
dst = Path('src/bobe/wake_models/hey_jarvis/wake_model.b64')
dst.write_bytes(base64.b64encode(src.read_bytes()))
print(f"Wrote {dst} ({dst.stat().st_size} bytes) for HF Space deploy")
PY

echo "Exported Heed model to src/bobe/wake_models/hey_jarvis"
