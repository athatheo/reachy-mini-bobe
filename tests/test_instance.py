import os

from bobe.config import config
from bobe.instance import load_instance_env, _migrate_legacy_env, resolve_instance_path


def test_resolve_instance_path_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("bobe.instance.default_instance_dir", lambda: tmp_path / "bobe")
    path = resolve_instance_path()
    assert path == tmp_path / "bobe"
    assert path.is_dir()


def test_migrate_legacy_env(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "pkg" / "bobe"
    legacy_dir.mkdir(parents=True)
    legacy_env = legacy_dir / ".env"
    legacy_env.write_text("OPENAI_API_KEY=sk-testkey123456789012345\n", encoding="utf-8")

    target_dir = tmp_path / "instance"
    target_dir.mkdir()
    monkeypatch.setattr("bobe.instance.packaged_instance_dir", lambda: legacy_dir)

    _migrate_legacy_env(target_dir)
    migrated = (target_dir / ".env").read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=sk-testkey123456789012345" in migrated


def test_load_instance_env_never_erases_live_values_with_empty_lines(tmp_path, monkeypatch):
    """Empty KEY= placeholders in the instance .env must not wipe live env values."""
    (tmp_path / ".env").write_text(
        "BOBE_WAKE_TOKEN=\nOPENAI_API_KEY=\nMODEL_NAME=test-model\n", encoding="utf-8"
    )
    monkeypatch.setenv("BOBE_WAKE_TOKEN", "live-token")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_NAME", "old-model")
    monkeypatch.setattr(config, "MODEL_NAME", "old-model")

    env_path = load_instance_env(tmp_path)

    assert env_path == tmp_path / ".env"
    assert os.environ["BOBE_WAKE_TOKEN"] == "live-token"
    assert "OPENAI_API_KEY" not in os.environ
    # Non-empty persisted values still override the process environment.
    assert os.environ["MODEL_NAME"] == "test-model"
    assert config.MODEL_NAME == "test-model"
