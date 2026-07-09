"""Guard against drift between the root and packaged .env.example files."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_packaged_env_example_matches_root():
    """src/bobe/.env.example seeds instance dirs; it must match the documented root file."""
    root = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    packaged = (REPO_ROOT / "src" / "bobe" / ".env.example").read_text(encoding="utf-8")
    assert packaged == root
