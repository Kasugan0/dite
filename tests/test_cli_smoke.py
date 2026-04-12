"""CLI smoke tests for Phase 0 baseline stability."""

from pathlib import Path

from typer.testing import CliRunner

from dite.cli import app
from dite.core.organizer import OrganizePreview


def _write_test_config(tmp_path: Path, monkeypatch, locale: str = "en") -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  base_url: https://api.example.com/v1",
                "  api_key: dummy-key",
                "cache:",
                f"  directory: {tmp_path / 'cache'}",
                "i18n:",
                f"  locale: {locale}",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_dite_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_dite_scan_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0


def test_dite_scan_help_with_en_us_locale(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch, locale="en-US")
    result = runner.invoke(app, ["scan", "--help"])
    assert result.exit_code == 0


def test_dite_organize_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["organize", "--help"])
    assert result.exit_code == 0


def test_dite_cache_status(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)

    result = runner.invoke(app, ["cache", "status"])
    assert result.exit_code == 0


def test_dite_setup_help(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)

    result = runner.invoke(app, ["setup", "--help"])
    assert result.exit_code == 0


def test_dite_setup_docling_pdf_runs_downloader(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    _write_test_config(tmp_path, monkeypatch)
    target_dir = tmp_path / ".cache" / "dite" / "docling" / "models"

    calls: dict[str, object] = {}

    def fake_download(output_dir: Path, *, force: bool, progress: bool) -> Path:
        calls["output_dir"] = output_dir
        calls["force"] = force
        calls["progress"] = progress
        for name in (
            "docling-project--docling-layout-heron",
            "docling-project--docling-models",
        ):
            (output_dir / name).mkdir(parents=True, exist_ok=True)
        return output_dir

    monkeypatch.setattr("dite.cli.download_docling_pdf_models", fake_download)

    result = runner.invoke(app, ["setup", "docling-pdf"])

    assert result.exit_code == 0
    assert calls == {
        "output_dir": target_dir,
        "force": False,
        "progress": False,
    }
    assert "Docling PDF models are ready" in result.output


def test_dite_cache_status_creates_global_config_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    assert not config_path.exists()

    runner = CliRunner()
    result = runner.invoke(app, ["cache", "status"])

    assert result.exit_code == 0
    assert config_path.exists()


def test_dite_rejects_removed_config_option(tmp_path: Path, monkeypatch) -> None:
    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["--config", str(tmp_path / "x.yaml"), "--help"])

    assert result.exit_code != 0
    assert "No such option: --config" in result.output


def test_organize_preview_sanitizes_cluster_name(tmp_path: Path) -> None:
    preview = OrganizePreview(source_folder=tmp_path, target_folder=tmp_path / "out")
    preview.add_cluster("finance/reports:*", [tmp_path / "a.pdf"])

    assert "finance-reports-" in preview.cluster_map


def test_organize_preview_avoids_destination_overwrite(tmp_path: Path) -> None:
    source_folder = tmp_path / "src"
    target_folder = tmp_path / "out"
    source_folder.mkdir()
    target_folder.mkdir()

    first = source_folder / "a" / "report.pdf"
    second = source_folder / "b" / "report.pdf"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("1", encoding="utf-8")
    second.write_text("2", encoding="utf-8")

    cluster_dir = target_folder / "cluster"
    cluster_dir.mkdir()
    (cluster_dir / "report.pdf").write_text("existing", encoding="utf-8")

    preview = OrganizePreview(source_folder=source_folder, target_folder=target_folder)
    preview.add_cluster("cluster", [first, second])

    assert preview.operations[0].destination.name == "report_1.pdf"
    assert preview.operations[1].destination.name == "report_2.pdf"


def test_organize_preview_execute_handles_partial_copy_failure(
    tmp_path: Path, monkeypatch
) -> None:
    source_folder = tmp_path / "src"
    target_folder = tmp_path / "out"
    source_folder.mkdir(parents=True)

    ok_file = source_folder / "ok.txt"
    bad_file = source_folder / "bad.txt"
    ok_file.write_text("ok", encoding="utf-8")
    bad_file.write_text("bad", encoding="utf-8")

    preview = OrganizePreview(source_folder=source_folder, target_folder=target_folder)
    preview.add_cluster("reports", [ok_file, bad_file])

    import shutil

    real_copy2 = shutil.copy2

    def flaky_copy2(src, dst, *args, **kwargs):
        if Path(src).name == "bad.txt":
            raise OSError("disk full")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr("dite.core.organizer.shutil.copy2", flaky_copy2)

    success, failed = preview.execute(dry_run=False)

    assert success == 1
    assert failed == 1
    assert preview.operations[0].status == "success"
    assert preview.operations[1].status == "failed"
    assert preview.operations[0].destination.exists()
    assert not preview.operations[1].destination.exists()
    assert not ok_file.exists()
    assert bad_file.exists()
