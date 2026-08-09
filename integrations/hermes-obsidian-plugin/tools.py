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
    _register_semantic(ctx)
    logger.info("obsidian-scoped plugin registered: search/semantic/read/list/append only")


# ----------------------------------------------------------------------
# Layer 4: local semantic search (fastembed ONNX; index outside the vault)
# ----------------------------------------------------------------------

_INDEX_DIR = Path(os.path.expanduser("~/.hermes/obsidian-index"))
_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
_CHUNK_CHARS = 1200
_EMBEDDER = None


def _get_embedder() -> Any:
    global _EMBEDDER
    if _EMBEDDER is None:
        from fastembed import TextEmbedding

        _EMBEDDER = TextEmbedding(model_name=_EMBED_MODEL)
    return _EMBEDDER


def _chunk_note(rel: str, text: str) -> list:
    """Split a note into overlapping-ish chunks keyed by heading context."""
    chunks = []
    buf: list = []
    size = 0
    for line in text.splitlines():
        buf.append(line)
        size += len(line) + 1
        if size >= _CHUNK_CHARS:
            chunks.append("\n".join(buf).strip())
            buf, size = buf[-3:], sum(len(b) + 1 for b in buf[-3:])
    tail = "\n".join(buf).strip()
    if tail:
        chunks.append(tail)
    return [{"note": rel, "text": chunk} for chunk in chunks if len(chunk) > 40]


def _build_or_update_index(root: Path) -> dict:
    """Incrementally (re)embed changed notes; returns the loaded index."""
    import numpy as np

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = _INDEX_DIR / "manifest.json"
    chunks_path = _INDEX_DIR / "chunks.json"
    vectors_path = _INDEX_DIR / "vectors.npy"

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    chunks = json.loads(chunks_path.read_text()) if chunks_path.exists() else []
    vectors = np.load(vectors_path) if vectors_path.exists() and chunks else None

    current: dict = {}
    for note in _iter_notes(root):
        rel = str(note.relative_to(root))
        stat = note.stat()
        current[rel] = f"{stat.st_mtime_ns}:{stat.st_size}"

    stale = {rel for rel in manifest if manifest.get(rel) != current.get(rel)}
    fresh = {rel for rel in current if rel not in manifest}
    to_embed = sorted(stale | fresh)

    if to_embed or (set(manifest) - set(current)):
        keep_mask = [c["note"] not in stale and c["note"] in current for c in chunks]
        chunks = [c for c, keep in zip(chunks, keep_mask) if keep]
        if vectors is not None and len(keep_mask) == vectors.shape[0]:
            vectors = vectors[np.array(keep_mask, dtype=bool)] if chunks else None
        else:
            vectors, chunks = None, []
            to_embed = sorted(current)

        new_chunks = []
        for rel in to_embed:
            try:
                text = (root / rel).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            new_chunks.extend(_chunk_note(rel, text))
        if new_chunks:
            embedder = _get_embedder()
            new_vecs = np.array(list(embedder.embed([c["text"] for c in new_chunks])), dtype=np.float32)
            vectors = new_vecs if vectors is None else np.vstack([vectors, new_vecs])
            chunks = chunks + new_chunks

        manifest_path.write_text(json.dumps(current))
        chunks_path.write_text(json.dumps(chunks))
        if vectors is not None:
            np.save(vectors_path, vectors)

    return {"chunks": chunks, "vectors": vectors}


def _handle_semantic_search(params: Dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    try:
        import numpy as np

        query = (params.get("query") or "").strip()
        if len(query) < 3:
            return json.dumps({"success": False, "error": "query must be at least 3 characters"})
        max_results = min(int(params.get("max_results") or 8), 15)
        root = _vault_root()
        index = _build_or_update_index(root)
        if index["vectors"] is None or not index["chunks"]:
            return json.dumps({"success": True, "results": [], "note": "vault index is empty"})

        embedder = _get_embedder()
        query_vec = np.array(list(embedder.embed([query])), dtype=np.float32)[0]
        vectors = index["vectors"]
        scores = vectors @ query_vec / (
            (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec)) + 1e-9
        )
        order = np.argsort(-scores)

        seen_notes: dict = {}
        results = []
        for idx in order:
            chunk = index["chunks"][int(idx)]
            if seen_notes.get(chunk["note"], 0) >= 2:
                continue
            seen_notes[chunk["note"]] = seen_notes.get(chunk["note"], 0) + 1
            results.append(
                {
                    "note": chunk["note"],
                    "score": round(float(scores[int(idx)]), 3),
                    "snippet": chunk["text"][:300],
                }
            )
            if len(results) >= max_results:
                break
        return json.dumps({"success": True, "query": query, "results": results})
    except Exception as exc:
        logger.exception("obsidian_semantic_search failed")
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})


def _register_semantic(ctx: Any) -> None:
    ctx.register_tool(
        name="obsidian_semantic_search",
        toolset="obsidian",
        schema={
            "name": "obsidian_semantic_search",
            "description": (
                "Semantic (meaning-based) search over the user's Obsidian vault — finds "
                "relevant notes even when they use different words than the query. Use for "
                "fuzzy recall; use obsidian_search for exact terms. Follow up promising "
                "hits with obsidian_read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What you are looking for, phrased naturally"},
                    "max_results": {"type": "integer", "description": "Max passages (default 8)"},
                },
                "required": ["query"],
            },
        },
        handler=_handle_semantic_search,
        description="Semantic search over the Obsidian vault (read-only, fully local).",
    )
