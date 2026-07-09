"""Shared parsing helpers for environment-variable style string values.

Leaf module: must not import anything from ``bobe`` so every config module
(robot and wake daemon) can use it without cycles.
"""

from __future__ import annotations

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def clean_optional(raw: str | None) -> str | None:
    """Strip a raw env value, mapping missing/blank values to None."""
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def parse_float(raw: str | None, default: float) -> float:
    """Parse a float env value, falling back to ``default`` when unset/invalid."""
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def parse_int(raw: str | None, default: int) -> int:
    """Parse an int env value, falling back to ``default`` when unset/invalid."""
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_bool(raw: str | None) -> bool | None:
    """Parse a boolean env value; return None when unset or unrecognized."""
    if raw is None:
        return None
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return None
