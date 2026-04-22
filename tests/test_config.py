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


def test_load_config_reads_cluster_naming_workers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "processing:",
                "  extract_workers: 4",
                "  docling_pdf_workers: 1",
                "  cluster_naming_workers: 3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.processing.extract_workers == 4
    assert cfg.processing.docling_pdf_workers == 1
    assert cfg.processing.cluster_naming_workers == 3


def test_load_config_reads_docling_device(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "processing:",
                "  docling_device: cpu",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.processing.docling_device == "cpu"


def test_load_config_reads_api_runtime_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "api:",
                "  connect_timeout_sec: 3.0",
                "  read_timeout_sec: 45.0",
                "  write_timeout_sec: 12.0",
                "  pool_timeout_sec: 4.0",
                "  max_retries: 5",
                "  max_connections: 20",
                "  max_keepalive_connections: 10",
                "  keepalive_expiry_sec: 9.5",
                "processing:",
                "  vlm_api_workers: 6",
                "  vlm_pages_per_document: 3",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config()

    assert cfg.api.connect_timeout_sec == 3.0
    assert cfg.api.read_timeout_sec == 45.0
    assert cfg.api.write_timeout_sec == 12.0
    assert cfg.api.pool_timeout_sec == 4.0
    assert cfg.api.max_retries == 5
    assert cfg.api.max_connections == 20
    assert cfg.api.max_keepalive_connections == 10
    assert cfg.api.keepalive_expiry_sec == 9.5
    assert cfg.processing.vlm_api_workers == 6
    assert cfg.processing.vlm_pages_per_document == 3


def test_load_config_rejects_keepalive_larger_than_max_connections(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    global_config_path = tmp_path / ".config" / "dite" / "config.yaml"
    global_config_path.parent.mkdir(parents=True, exist_ok=True)
    global_config_path.write_text(
        "\n".join(
            [
                "api:",
                "  max_connections: 8",
                "  max_keepalive_connections: 16",
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_config()
    except ValueError as exc:
        assert "max_keepalive_connections" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("load_config should reject invalid API pool settings")
