"""Tests for config loader."""

from pathlib import Path
from typing import Optional

import pytest
from pydantic import BaseModel

from snowflake_ml_template.core.config.loader import ConfigLoader


class DummyConfig(BaseModel):
    """Dummy config for testing."""

    key: str
    nested: Optional[dict] = None


def _write_yaml(path: Path, content: str) -> None:
    """Write YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_init_invalid_environment_raises():
    """Test init invalid environment raises."""
    with pytest.raises(ValueError):
        ConfigLoader(config_dir="irrelevant", environment="invalid")


def test_load_from_defaults_only(tmp_path: Path):
    """Test loading from defaults only."""
    cfg_dir = tmp_path / "config"
    defaults = cfg_dir / "defaults" / "test_config.yaml"
    _write_yaml(
        defaults,
        """
key: value
nested:
  a: 1
  b: 2
""".strip(),
    )

    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(DummyConfig, config_file="test_config.yaml", use_env_vars=False)

    assert cfg.key == "value"
    assert cfg.nested == {"a": 1, "b": 2}


def test_env_environment_overrides_section_merging(tmp_path: Path):
    """Test environment environment overrides section merging."""
    cfg_dir = tmp_path / "config"
    # defaults
    _write_yaml(cfg_dir / "defaults" / "test_config.yaml", "key: default")
    # environment file with namespaced section
    _write_yaml(
        cfg_dir / "environments" / "dev.yaml",
        """
test_config:
  key: from_env_file
  nested:
    x: 10
""".strip(),
    )

    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(DummyConfig, config_file="test_config.yaml", use_env_vars=False)

    assert cfg.key == "from_env_file"
    assert cfg.nested == {"x": 10}


def test_env_environment_no_named_section_merges_all(tmp_path: Path):
    """Test environment environment no named section merges all."""
    cfg_dir = tmp_path / "config"
    # defaults
    _write_yaml(cfg_dir / "defaults" / "test_config.yaml", "nested:\n  a: 1")
    # environment file without section name
    _write_yaml(
        cfg_dir / "environments" / "dev.yaml",
        """
key: from_env
nested:
  b: 2
""".strip(),
    )

    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(DummyConfig, config_file="test_config.yaml", use_env_vars=False)
    assert cfg.key == "from_env"
    assert cfg.nested == {"a": 1, "b": 2}


def test_explicit_overrides_take_precedence(tmp_path: Path):
    """Test explicit overrides take precedence."""
    cfg_dir = tmp_path / "config"
    _write_yaml(cfg_dir / "defaults" / "test_config.yaml", "key: default")

    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(
        DummyConfig,
        config_file="test_config.yaml",
        overrides={"key": "explicit"},
        use_env_vars=False,
    )

    assert cfg.key == "explicit"


def test_load_from_env_with_prefix(monkeypatch: pytest.MonkeyPatch):
    """Test loading from environment with prefix."""
    monkeypatch.setenv("APP_KEY", "from_env")
    loader = ConfigLoader(config_dir="config", environment="dev")
    cfg = loader.load(DummyConfig, use_env_vars=True, env_prefix="APP_")
    assert cfg.key == "from_env"


def test_env_var_substitution_in_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test environment variable substitution in values."""
    cfg_dir = tmp_path / "config"
    _write_yaml(cfg_dir / "defaults" / "test_config.yaml", "key: ${MY_VAR}")

    monkeypatch.setenv("MY_VAR", "substituted")
    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(DummyConfig, config_file="test_config.yaml", use_env_vars=False)

    assert cfg.key == "substituted"


def test_load_from_dict_simple():
    """Test loading from dict."""
    loader = ConfigLoader()
    cfg = loader.load_from_dict(DummyConfig, {"key": "dict_value", "nested": {"a": 1}})
    assert cfg.key == "dict_value"
    assert cfg.nested == {"a": 1}


def test_load_yaml_missing_file_raises(tmp_path: Path):
    """Test loading YAML missing file raises exception."""
    loader = ConfigLoader(config_dir=tmp_path / "config", environment="dev")
    with pytest.raises(FileNotFoundError):
        # Call private to trigger the specific branch
        loader._load_yaml(tmp_path / "nope.yaml")


def test_load_validation_error_raises(tmp_path: Path):
    """Test loading validation error raises exception."""
    cfg_dir = tmp_path / "config"
    # Missing required field 'key' to trigger ValidationError
    _write_yaml(cfg_dir / "defaults" / "test_config.yaml", "nested: {a: 1}")
    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    with pytest.raises(Exception):
        loader.load(DummyConfig, config_file="test_config.yaml", use_env_vars=False)


def test_deep_merge_merges_nested_dicts():
    """Test deep merge merges nested dicts."""
    loader = ConfigLoader()
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"b": 2, "nested": {"y": 20, "z": 3}}
    merged = loader._deep_merge(base, override)

    assert merged["a"] == 1
    assert merged["b"] == 2
    assert merged["nested"] == {"x": 1, "y": 20, "z": 3}


def test_parse_env_value_types():
    """Test parsing environment variable values."""
    loader = ConfigLoader()
    # booleans
    assert loader._parse_env_value("true") is True
    assert loader._parse_env_value("false") is False
    assert loader._parse_env_value("1") is True  # treated as boolean before int
    assert loader._parse_env_value("0") is False
    # int
    assert loader._parse_env_value("42") == 42
    # float
    assert loader._parse_env_value("3.14") == 3.14
    # json dict
    assert loader._parse_env_value('{"a": 1}') == {"a": 1}
    # json list
    assert loader._parse_env_value("[1,2,3]") == [1, 2, 3]
    # fallback string
    assert loader._parse_env_value("hello") == "hello"


def test_substitute_env_vars_handles_missing(monkeypatch: pytest.MonkeyPatch):
    """Test substitute environment variables handles missing."""
    loader = ConfigLoader()
    cfg = {"key": "${MISSING_VAR}", "other": "plain"}
    # Do not set env; missing should keep original template
    out = loader._substitute_env_vars(cfg)
    assert out["key"] == "${MISSING_VAR}"
    assert out["other"] == "plain"


def test_save_to_yaml_writes_file(tmp_path: Path):
    """Test saving to YAML file."""
    loader = ConfigLoader()
    cfg = DummyConfig(key="k", nested={"a": 1})
    out_file = tmp_path / "out.yaml"
    loader.save_to_yaml(cfg, out_file)
    content = out_file.read_text()
    assert "key: k" in content
    assert "nested:" in content


def test_load_yaml_invalid_yaml_raises(tmp_path: Path):
    """Test loading invalid YAML raises exception."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("key: [unclosed")
    loader = ConfigLoader()
    with pytest.raises(Exception):
        loader._load_yaml(bad)


def test_load_from_env_only(monkeypatch: pytest.MonkeyPatch):
    """Test loading from environment variables only."""
    monkeypatch.setenv("TST_KEY", "v")

    class EnvModel(BaseModel):
        key: str

    loader = ConfigLoader()
    cfg = loader.load_from_env(EnvModel, prefix="TST_")
    assert cfg.key == "v"


def test_deep_merge_overrides_non_dict():
    """Test deep merge overrides non-dict."""
    loader = ConfigLoader()
    base = {"a": {"x": 1}, "b": 1}
    override = {"a": 2, "b": {"y": 2}}
    merged = loader._deep_merge(base, override)
    assert merged["a"] == 2
    assert merged["b"] == {"y": 2}


def test_empty_yaml_returns_empty_dict_and_merge(tmp_path: Path):
    """Test empty YAML returns empty dict and merge."""
    cfg_dir = tmp_path / "config"
    # empty defaults file
    (cfg_dir / "defaults").mkdir(parents=True, exist_ok=True)
    (cfg_dir / "defaults" / "test_config.yaml").write_text("")
    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(
        DummyConfig,
        config_file="test_config.yaml",
        overrides={"key": "x"},
        use_env_vars=False,
    )
    assert cfg.key == "x"


def test_nested_env_var_substitution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test nested environment variable substitution."""
    cfg_dir = tmp_path / "config"
    _write_yaml(
        cfg_dir / "defaults" / "test_config.yaml",
        """
nested:
  inner: ${INNER}
key: ${OUTER}
""".strip(),
    )
    monkeypatch.setenv("INNER", "iv")
    monkeypatch.setenv("OUTER", "ov")
    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(DummyConfig, config_file="test_config.yaml", use_env_vars=False)
    assert cfg.key == "ov"
    assert cfg.nested == {"inner": "iv"}


def test_full_precedence_defaults_envfile_envvars_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Test full precedence of defaults, envfile, envvars, and overrides."""
    cfg_dir = tmp_path / "config"
    _write_yaml(cfg_dir / "defaults" / "test_config.yaml", "key: default")
    _write_yaml(cfg_dir / "environments" / "dev.yaml", "test_config:\n  key: envfile")
    monkeypatch.setenv("APP_KEY", "envvar")
    loader = ConfigLoader(config_dir=cfg_dir, environment="dev")
    cfg = loader.load(
        DummyConfig,
        config_file="test_config.yaml",
        use_env_vars=True,
        env_prefix="APP_",
        overrides={"key": "override"},
    )
    assert cfg.key == "override"
