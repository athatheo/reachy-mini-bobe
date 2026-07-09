from bobe.instance import _migrate_legacy_env, resolve_instance_path


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
