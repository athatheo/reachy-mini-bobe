"""Persistent per-robot BoBe instance directory (survives app reinstalls)."""

from __future__ import annotations
import os
import shutil
import logging
from pathlib import Path

from dotenv import dotenv_values

from bobe.config import config


logger = logging.getLogger(__name__)

APP_NAME = "bobe"


def default_instance_dir() -> Path:
    """Return the default persistent instance directory for BoBe."""
    override = (os.getenv("REACHY_MINI_APP_INSTANCE") or "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".reachy_mini" / "apps" / APP_NAME


def packaged_instance_dir() -> Path:
    """Legacy instance directory inside the installed Python package."""
    return Path(__file__).resolve().parent


def resolve_instance_path() -> Path:
    """Resolve and prepare the persistent instance directory."""
    instance_dir = default_instance_dir()
    instance_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_env(instance_dir)
    return instance_dir


def _migrate_legacy_env(instance_dir: Path) -> None:
    """Copy keys/settings from the old package-local .env if needed."""
    target = instance_dir / ".env"
    if target.exists():
        return

    legacy = packaged_instance_dir() / ".env"
    if not legacy.exists():
        return

    try:
        shutil.copy2(legacy, target)
        logger.info("Migrated BoBe instance .env from %s to %s", legacy, target)
    except Exception as exc:
        logger.warning("Could not migrate legacy .env from %s: %s", legacy, exc)


def load_instance_env(instance_path: Path | str | None) -> Path | None:
    """Load ``instance_path/.env`` into the process environment.

    Non-empty values override the process environment (persisted settings are
    explicit user intent), but empty ``KEY=`` placeholder lines are skipped so
    they can never erase live values.
    """
    if not instance_path:
        return None
    env_path = Path(instance_path) / ".env"
    if not env_path.exists():
        return None
    for key, value in dotenv_values(str(env_path)).items():
        if value is None or not value.strip():
            continue
        os.environ[key] = value
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        config.OPENAI_API_KEY = openai_key
    model_name = os.getenv("MODEL_NAME", "").strip()
    if model_name:
        config.MODEL_NAME = model_name
    return env_path
