"""Shared .env file helpers and API key validation."""

from __future__ import annotations
import os
import logging
import threading
from pathlib import Path

from bobe.claude import DEFAULT_CLAUDE_MODEL
from bobe.config import config
from bobe.instance import load_instance_env


logger = logging.getLogger(__name__)

# Serializes read-modify-write cycles on the shared instance .env so concurrent
# settings endpoints (/api_keys, /wake-config) cannot drop each other's keys.
ENV_FILE_LOCK = threading.Lock()


def is_plausible_openai_key(value: str | None) -> bool:
    """Return whether a value looks like an OpenAI API key."""
    key = (value or "").strip()
    return key.startswith("sk-") and len(key) >= 20


def is_plausible_anthropic_key(value: str | None) -> bool:
    """Return whether a value looks like an Anthropic API key."""
    key = (value or "").strip()
    return key.startswith("sk-ant-") and len(key) >= 20


def _read_lines_if_exists(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None


def read_env_lines(env_path: Path) -> list[str]:
    """Load env file contents as a list of lines (empty when missing).

    Never falls back to packaged ``.env.example`` templates: baking template
    defaults (empty API keys/token lines, example wake URL/gain) into the
    instance .env would silently override live tuned values on the next load.
    """
    try:
        return _read_lines_if_exists(env_path) or []
    except Exception:
        return []


def write_env_lines(env_path: Path, lines: list[str]) -> None:
    """Atomically write env file lines (temp file + rename) so readers never see a torn file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = env_path.with_name(env_path.name + ".tmp")
    tmp_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    os.replace(tmp_path, env_path)


def upsert_env_keys(lines: list[str], values: dict[str, str]) -> list[str]:
    """Update or append KEY=value entries in env file lines.

    Empty values are skipped: an empty assignment must never be persisted
    because it would later erase the live value when the file is loaded.
    """
    for key, value in values.items():
        if not str(value).strip():
            continue
        replacement = f"{key}={value}"
        for index, line in enumerate(lines):
            if line.strip().startswith(f"{key}="):
                lines[index] = replacement
                break
        else:
            lines.append(replacement)
    return lines


def persist_api_settings(
    instance_path: str | None,
    *,
    openai_api_key: str,
    anthropic_api_key: str,
    claude_model: str,
) -> None:
    """Persist explicit API settings to environment and instance ``.env``."""
    values = {
        "OPENAI_API_KEY": openai_api_key.strip(),
        "ANTHROPIC_API_KEY": anthropic_api_key.strip(),
        "CLAUDE_MODEL": (claude_model or DEFAULT_CLAUDE_MODEL).strip() or DEFAULT_CLAUDE_MODEL,
    }
    if not values["OPENAI_API_KEY"] or not values["ANTHROPIC_API_KEY"]:
        return

    os.environ.update(values)
    try:
        config.OPENAI_API_KEY = values["OPENAI_API_KEY"]
    except Exception:
        pass

    if not instance_path:
        return

    try:
        env_path = Path(instance_path) / ".env"
        with ENV_FILE_LOCK:
            lines = upsert_env_keys(read_env_lines(env_path), values)
            write_env_lines(env_path, lines)
        logger.info("Persisted explicit API settings to %s", env_path)
        load_instance_env(instance_path)
    except Exception as exc:
        logger.warning("Failed to persist explicit API settings: %s", exc)
