from pathlib import Path

import yaml

from dite.config import load_config


def test_load_config_creates_global_config_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"

    assert not config_path.exists()

    cfg = load_config()

    assert config_path.exists()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["models"]["embedding"] == cfg.models.embedding
    assert data["cache"]["directory"] == str(tmp_path / ".cache" / "dite")
    assert ".markdown" in data["formats"]["documents"]


def test_load_config_supports_markdown_extension_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = load_config()

    assert ".markdown" in cfg.formats.documents


def test_load_config_backfills_markdown_alias_for_existing_md_configs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "formats:",
                "  documents: [.md, .txt]",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert ".md" in cfg.formats.documents
    assert ".markdown" in cfg.formats.documents
    assert ".txt" in cfg.formats.documents


def test_load_config_ignores_workspace_dite_yaml(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "i18n:",
                "  locale: en-US",
            ]
        ),
        encoding="utf-8",
    )

    (tmp_path / "dite.yaml").write_text(
        "\n".join(
            [
                "i18n:",
                "  locale: zh-CN",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()
    assert cfg.i18n.locale == "en-US"


def test_load_config_reads_cluster_naming_request_profile(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "request_profiles:",
                "  cluster_naming:",
                "    max_tokens: 64",
                "    reasoning_mode: off",
                "    thinking_budget: 256",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.request_profiles.cluster_naming.max_tokens == 64
    assert cfg.request_profiles.cluster_naming.reasoning_mode == "off"
    assert cfg.request_profiles.cluster_naming.thinking_budget == 256
