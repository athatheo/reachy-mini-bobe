#!/usr/bin/env python3
"""Upload a clean BoBe snapshot to the private HF space."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


INCLUDE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "index.html",
    "src",
    "scripts",
    "tests",
    "docs",
    "config/wake-daemon.env.example",
    ".env.example",
    ".gitattributes",
    ".gitignore",
    ".huggingfaceignore",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload clean BoBe tree to HF space")
    parser.add_argument("--space", default="athatheo/bobe")
    parser.add_argument("--message", default="fix: clean space upload without cache artifacts")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        "hf",
        "upload",
        args.space,
        ".",
        "--repo-type",
        "space",
        "--commit-message",
        args.message,
        "--delete",
        ".mypy_cache/**",
        "--delete",
        ".pytest_cache/**",
        "--delete",
        ".ruff_cache/**",
        "--delete",
        "build/**",
        "--delete",
        "wake_training/**",
        "--delete",
        ".venv/**",
    ]
    for path in INCLUDE_PATHS:
        cmd.extend(["--include", path])

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=repo_root)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
