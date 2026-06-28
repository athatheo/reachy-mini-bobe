"""Filesystem helpers for conversation personalities (profiles)."""

from __future__ import annotations

import re
from pathlib import Path

from bobe.config import DEFAULT_PROFILES_DIRECTORY, config


DEFAULT_OPTION = "(built-in default)"


def _profiles_root() -> Path:
    return config.PROFILES_DIRECTORY if config.PROFILES_DIRECTORY else DEFAULT_PROFILES_DIRECTORY


def _prompts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "prompts"


def _tools_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "tools"


def sanitize_name(name: str) -> str:
    """Normalize a user-provided profile name for filesystem storage."""
    s = name.strip()
    s = re.sub(r"\s+", "_", s)
    return re.sub(r"[^a-zA-Z0-9_-]", "", s)


def list_personalities() -> list[str]:
    """List available personality profile names."""
    names: list[str] = []
    root = _profiles_root()
    try:
        if root.exists():
            for profile in sorted(root.iterdir()):
                if profile.name == "user_personalities":
                    continue
                if profile.is_dir() and (profile / "instructions.txt").exists():
                    names.append(profile.name)
            user_dir = root / "user_personalities"
            if user_dir.exists():
                for profile in sorted(user_dir.iterdir()):
                    if profile.is_dir() and (profile / "instructions.txt").exists():
                        names.append(f"user_personalities/{profile.name}")
    except Exception:
        pass
    return names


def resolve_profile_dir(selection: str) -> Path:
    """Resolve the directory path for the given profile selection."""
    return _profiles_root() / selection


def read_instructions_for(name: str) -> str:
    """Read instructions for a profile, or the built-in default prompt."""
    try:
        if name == DEFAULT_OPTION:
            default_file = _prompts_dir() / "default_prompt.txt"
            return default_file.read_text(encoding="utf-8").strip() if default_file.exists() else ""
        target = resolve_profile_dir(name) / "instructions.txt"
        return target.read_text(encoding="utf-8").strip() if target.exists() else ""
    except Exception as exc:
        return f"Could not load instructions: {exc}"


def available_tools_for(selected: str) -> list[str]:
    """List tool module names available to the given profile selection."""
    shared: list[str] = []
    try:
        for module in _tools_dir().glob("*.py"):
            if module.stem in {"__init__", "core_tools"}:
                continue
            shared.append(module.stem)
    except Exception:
        pass
    local: list[str] = []
    try:
        if selected != DEFAULT_OPTION:
            for module in resolve_profile_dir(selected).glob("*.py"):
                local.append(module.stem)
    except Exception:
        pass
    return sorted(set(shared + local))


def write_profile(name: str, instructions: str, tools_text: str, voice: str = "cedar") -> str:
    """Write a user personality under ``user_personalities/`` and return its profile id."""
    name_s = sanitize_name(name)
    if not name_s:
        raise ValueError("invalid profile name")
    target_dir = _profiles_root() / "user_personalities" / name_s
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "instructions.txt").write_text(instructions.strip() + "\n", encoding="utf-8")
    (target_dir / "tools.txt").write_text((tools_text or "").strip() + "\n", encoding="utf-8")
    (target_dir / "voice.txt").write_text((voice or "cedar").strip() + "\n", encoding="utf-8")
    return f"user_personalities/{name_s}"
