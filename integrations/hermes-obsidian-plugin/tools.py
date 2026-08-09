"""Scoped Obsidian vault tools for Hermes: search, read, list, append.

The agent's vault access is deliberately one-directional for existing
knowledge: it can find and read anything, and it can ADD — append to any
existing note, or create new notes inside allowed folders (default
``Hermes/`` and ``Journal/``). Tools to delete, overwrite, truncate, or
rename do not exist, so the model has no way to destroy or rewrite notes.

Install: symlink this directory to ``~/.hermes/plugins/obsidian/``, set
``OBSIDIAN_VAULT`` (and optionally ``OBSIDIAN_CREATE_FOLDERS``) in
``~/.hermes/.env``, enable the plugin, grant the ``obsidian`` toolset in
``platform_toolsets``, and restart the gateway.
"""

# ruff: noqa: D103 — runs inside Hermes, matching its plugin conventions.

import os
import json
import logging
from typing import Any, Dict
from pathlib import Path
from datetime import datetime


logger = logging.getLogger(__name__)

_SKIP_DIRS = {".obsidian", ".git", ".trash", "node_modules"}
_MAX_READ_BYTES = 80_000
_MAX_APPEND_CHARS = 8_000
_MAX_SEARCH_RESULTS = 30


def _vault_root() -> Path:
    raw = os.path.expanduser(os.getenv("OBSIDIAN_VAULT", "")).strip()
    if not raw:
        raise RuntimeError("OBSIDIAN_VAULT is not set")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise RuntimeError(f"OBSIDIAN_VAULT does not exist: {root}")
    return root


def _create_folders() -> list:
    raw = os.getenv("OBSIDIAN_CREATE_FOLDERS", "Hermes,Journal")
    return [part.strip().strip("/") for part in raw.split(",") if part.strip()]


def _resolve_note(root: Path, rel_path: str, *, must_exist: bool) -> Path:
    """Resolve a vault-relative path with traversal protection."""
    rel = (rel_path or "").strip().lstrip("/")
    if not rel:
        raise ValueError("path is required")
    if not rel.endswith(".md"):
        rel += ".md"
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise ValueError("path escapes the vault")
    if any(part in _SKIP_DIRS for part in target.relative_to(root).parts):
        raise ValueError("path is inside a protected folder")
    if must_exist and not target.is_file():
        raise FileNotFoundError(f"note not found: {rel}")
    return target


def _iter_notes(root: Path, folder: str | None = None):
    base = root
    if folder:
        base = _resolve_note(root, folder.rstrip("/") + "/_", must_exist=False).parent
        if not base.is_dir():
            return
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for filename in sorted(filenames):
            if filename.endswith(".md"):
                yield Path(dirpath) / filename


def _handle_search(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        query = (params.get("query") or "").strip()
        if len(query) < 2:
            return json.dumps({"success": False, "error": "query must be at least 2 characters"})
        root = _vault_root()
        needle = query.casefold()
        matches = []
        for note in _iter_notes(root, params.get("folder")):
            rel = str(note.relative_to(root))
            try:
                text = note.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            title_hit = needle in note.stem.casefold()
            line_hits = []
            for number, line in enumerate(text.splitlines(), 1):
                if needle in line.casefold():
                    line_hits.append({"line": number, "text": line.strip()[:200]})
                    if len(line_hits) >= 3:
                        break
            if title_hit or line_hits:
                matches.append({"note": rel, "title_match": title_hit, "matches": line_hits})
                if len(matches) >= _MAX_SEARCH_RESULTS:
                    break
        return json.dumps({"success": True, "query": query, "results": matches, "count": len(matches)})
    except Exception as exc:
        logger.exception("obsidian_search failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def _handle_read(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        root = _vault_root()
        note = _resolve_note(root, params.get("path") or "", must_exist=True)
        raw = note.read_bytes()
        truncated = len(raw) > _MAX_READ_BYTES
        text = raw[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
        return json.dumps(
            {
                "success": True,
                "note": str(note.relative_to(root)),
                "content": text,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        logger.exception("obsidian_read failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def _handle_list(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        root = _vault_root()
        notes = [str(n.relative_to(root)) for n in _iter_notes(root, params.get("folder"))]
        return json.dumps({"success": True, "notes": notes[:300], "count": len(notes)})
    except Exception as exc:
        logger.exception("obsidian_list failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def _handle_append(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        text = (params.get("text") or "").rstrip()
        if not text:
            return json.dumps({"success": False, "error": "text is required"})
        if len(text) > _MAX_APPEND_CHARS:
            return json.dumps({"success": False, "error": f"text too long (max {_MAX_APPEND_CHARS} chars)"})
        root = _vault_root()
        note = _resolve_note(root, params.get("path") or "", must_exist=False)
        rel = str(note.relative_to(root))
        created = not note.exists()
        if created:
            allowed = _create_folders()
            top = note.relative_to(root).parts[0] if len(note.relative_to(root).parts) > 1 else ""
            if top not in allowed:
                return json.dumps(
                    {
                        "success": False,
                        "error": (
                            f"New notes may only be created inside: {', '.join(allowed)}/. "
                            "Appending to existing notes works anywhere."
                        ),
                    }
                )
            note.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        block = ("" if created else "\n") + f"\n{text}\n" if params.get("no_header") else (
            ("" if created else "\n") + f"\n###### {stamp} (Hermes)\n{text}\n"
        )
        # Append-only by construction: mode "a" cannot truncate or rewrite.
        with open(note, "a", encoding="utf-8") as handle:
            handle.write(block)
        return json.dumps(
            {
                "success": True,
                "note": rel,
                "created": created,
                "note_to_agent": "Appended. You cannot edit or delete notes — additions are permanent.",
            }
        )
    except Exception as exc:
        logger.exception("obsidian_append failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def register(ctx: Any) -> None:
    """Plugin entry point — read/search/list/append; nothing destructive."""
    ctx.register_tool(
        name="obsidian_search",
        toolset="obsidian",
        schema={
            "name": "obsidian_search",
            "description": (
                "Full-text search across the user's Obsidian vault (note titles and "
                "content). Use this to find relevant notes before answering questions "
                "about the user's projects, ideas, or journal."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text (case-insensitive)"},
                    "folder": {"type": "string", "description": "Optional folder to limit the search"},
                },
                "required": ["query"],
            },
        },
        handler=_handle_search,
        description="Search the Obsidian vault (read-only).",
    )
    ctx.register_tool(
        name="obsidian_read",
        toolset="obsidian",
        schema={
            "name": "obsidian_read",
            "description": "Read a note from the user's Obsidian vault by its relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault-relative path, e.g. 'BoBe/ideas.md'"},
                },
                "required": ["path"],
            },
        },
        handler=_handle_read,
        description="Read an Obsidian note (read-only).",
    )
    ctx.register_tool(
        name="obsidian_list",
        toolset="obsidian",
        schema={
            "name": "obsidian_list",
            "description": "List notes in the user's Obsidian vault (optionally within one folder).",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string", "description": "Optional folder to list"},
                },
            },
        },
        handler=_handle_list,
        description="List Obsidian notes (read-only).",
    )
    ctx.register_tool(
        name="obsidian_append",
        toolset="obsidian",
        schema={
            "name": "obsidian_append",
            "description": (
                "APPEND text to a note in the user's Obsidian vault (adds a timestamped "
                "block at the end; never modifies existing content). Appending works on "
                "any existing note; creating a NEW note is only allowed inside the "
                "configured folders (default Hermes/ and Journal/). There are no tools "
                "to edit, overwrite, or delete notes — if asked, explain the user must "
                "do that themselves in Obsidian."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Vault-relative note path"},
                    "text": {"type": "string", "description": "Markdown text to append"},
                    "no_header": {"type": "boolean", "description": "Skip the timestamp header line"},
                },
                "required": ["path", "text"],
            },
        },
        handler=_handle_append,
        description="Append to an Obsidian note (append-only; no edit/delete).",
    )
    logger.info("obsidian-scoped plugin registered: search/read/list/append only")
