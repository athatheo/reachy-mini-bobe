import os
from pathlib import Path

import pytest

import bobe.config as config_mod


def test_config_raises_on_external_profile_name_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should fail fast when external/built-in profile names collide."""
    builtin_profiles = config_mod._collect_profile_names(config_mod.DEFAULT_PROFILES_DIRECTORY) - {"__pycache__"}
    assert builtin_profiles, "expected at least one built-in profile"
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)
    (external_profiles / sorted(builtin_profiles)[0]).mkdir()

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Ambiguous profile names"):
        config_mod.Config()


def test_config_raises_on_external_tool_name_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should fail fast when external/built-in tool names collide."""
    external_tools = tmp_path / "external_tools"
    external_tools.mkdir(parents=True)
    (external_tools / "dance.py").write_text("# collision with built-in dance tool\n", encoding="utf-8")

    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", config_mod.DEFAULT_PROFILES_DIRECTORY)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", external_tools)

    with pytest.raises(RuntimeError, match="Ambiguous tool names"):
        config_mod.Config()


def test_config_raises_when_selected_external_profile_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should fail fast when selected profile is absent from external root."""
    external_profiles = tmp_path / "external_profiles"
    external_profiles.mkdir(parents=True)

    monkeypatch.setattr(config_mod.Config, "REACHY_MINI_CUSTOM_PROFILE", "missing_profile")
    monkeypatch.setattr(config_mod.Config, "PROFILES_DIRECTORY", external_profiles)
    monkeypatch.setattr(config_mod.Config, "TOOLS_DIRECTORY", None)

    with pytest.raises(RuntimeError, match="Selected profile 'missing_profile' was not found"):
        config_mod.Config()


def test_apply_dotenv_values_skips_empty_and_respects_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty KEY= lines never erase exported vars, and the process env wins."""
    env_file = tmp_path / ".env"
    env_file.write_text("EMPTY_KEY=\nFILE_ONLY=from-file\nBOTH=from-file\n", encoding="utf-8")
    monkeypatch.setenv("EMPTY_KEY", "live")
    monkeypatch.delenv("FILE_ONLY", raising=False)
    monkeypatch.setenv("BOTH", "from-env")

    config_mod._apply_dotenv_values(env_file)

    assert os.environ["EMPTY_KEY"] == "live"
    assert os.environ["FILE_ONLY"] == "from-file"
    assert os.environ["BOTH"] == "from-env"


def test_find_config_dotenv_ignores_launch_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config .env discovery is anchored to the package/repo, not the CWD."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=cwd-key\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    found = config_mod._find_config_dotenv()

    assert found != tmp_path / ".env"
    if found is not None:
        assert tmp_path not in found.parents


def _fake_repo_checkout(tmp_path: Path) -> tuple[Path, Path]:
    """Create a repo-checkout layout and return (repo_root, package_dir)."""
    repo_root = tmp_path.resolve() / "repo"
    package_dir = repo_root / "src" / "bobe"
    package_dir.mkdir(parents=True)
    return repo_root, package_dir


def test_find_config_dotenv_prefers_repo_root_over_package_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The README-documented repo-root .env must win over a legacy src/bobe/.env."""
    repo_root, package_dir = _fake_repo_checkout(tmp_path)
    (repo_root / ".env").write_text("OPENAI_API_KEY=root-key\n", encoding="utf-8")
    (package_dir / ".env").write_text("OPENAI_API_KEY=legacy-key\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "__file__", str(package_dir / "config.py"))

    assert config_mod._find_config_dotenv() == repo_root / ".env"


def test_find_config_dotenv_falls_back_to_package_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a repo-root .env, the legacy src/bobe/.env is still honored."""
    repo_root, package_dir = _fake_repo_checkout(tmp_path)
    (package_dir / ".env").write_text("OPENAI_API_KEY=legacy-key\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "__file__", str(package_dir / "config.py"))

    assert config_mod._find_config_dotenv() == package_dir / ".env"


def test_find_config_dotenv_returns_none_when_no_env_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No candidate .env files means no dotenv is loaded."""
    _, package_dir = _fake_repo_checkout(tmp_path)
    monkeypatch.setattr(config_mod, "__file__", str(package_dir / "config.py"))

    assert config_mod._find_config_dotenv() is None


def test_find_config_dotenv_installed_package_only_checks_package_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside a src/ checkout, only the packaged directory's .env is considered."""
    site_packages = tmp_path.resolve() / "site-packages"
    package_dir = site_packages / "bobe"
    package_dir.mkdir(parents=True)
    (site_packages.parent / ".env").write_text("OPENAI_API_KEY=outside\n", encoding="utf-8")
    (package_dir / ".env").write_text("OPENAI_API_KEY=packaged\n", encoding="utf-8")
    monkeypatch.setattr(config_mod, "__file__", str(package_dir / "config.py"))

    assert config_mod._find_config_dotenv() == package_dir / ".env"
