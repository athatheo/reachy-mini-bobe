"""Shared .env file helpers and API key validation."""

from __future__ import annotations
import os
import logging
from pathlib import Path

from bobe.claude import DEFAULT_CLAUDE_MODEL
from bobe.config import config
from bobe.instance import load_instance_env


logger = logging.getLogger(__name__)


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
    """Load env file contents or a template as a list of lines."""
    try:
        lines = _read_lines_if_exists(env_path)
        if lines is not None:
            return lines
        for candidate in (
            env_path.parent / ".env.example",
            Path.cwd() / ".env.example",
            Path(__file__).parent / ".env.example",
        ):
            lines = _read_lines_if_exists(candidate)
            if lines is not None:
                return lines
        return []
    except Exception:
        return []


def upsert_env_keys(lines: list[str], values: dict[str, str]) -> list[str]:
    """Update or append KEY=value entries in env file lines."""
    for key, value in values.items():
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
        lines = upsert_env_keys(read_env_lines(env_path), values)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        logger.info("Persisted explicit API settings to %s", env_path)
        load_instance_env(instance_path)
    except Exception as exc:
        logger.warning("Failed to persist explicit API settings: %s", exc)
