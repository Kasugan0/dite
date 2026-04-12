import time
from pathlib import Path

from dite.extractors.docling import (
    DoclingExtractor,
    _get_required_pdf_artifact_dirs,
    download_docling_pdf_models,
    get_docling_pdf_artifacts_path,
    has_docling_pdf_artifacts,
)
from dite.i18n import set_locale


def test_get_docling_pdf_artifacts_path_uses_dite_cache_root(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert get_docling_pdf_artifacts_path() == (
        tmp_path / ".cache" / "dite" / "docling" / "models"
    )


def test_has_docling_pdf_artifacts_requires_layout_and_tableformer(
    tmp_path: Path,
) -> None:
    required_dirs = _get_required_pdf_artifact_dirs()

    assert has_docling_pdf_artifacts(tmp_path) is False

    for name in required_dirs:
        (tmp_path / name).mkdir(parents=True)

    assert has_docling_pdf_artifacts(tmp_path) is True


def test_download_docling_pdf_models_downloads_minimal_set(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_download_models(output_dir: Path, **kwargs) -> None:
        captured["output_dir"] = output_dir
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "docling.utils.model_downloader.download_models",
        fake_download_models,
    )

    result = download_docling_pdf_models(tmp_path, force=True, progress=True)

    assert result == tmp_path
    assert captured["output_dir"] == tmp_path
    assert captured["kwargs"] == {
        "force": True,
        "progress": True,
        "with_layout": True,
        "with_tableformer": True,
        "with_code_formula": False,
        "with_picture_classifier": False,
        "with_smolvlm": False,
        "with_granitedocling": False,
        "with_granitedocling_mlx": False,
        "with_smoldocling": False,
        "with_smoldocling_mlx": False,
        "with_granite_vision": False,
        "with_rapidocr": False,
        "with_easyocr": False,
    }


def test_docling_pdf_returns_clear_error_when_models_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    result = DoclingExtractor().extract(pdf_path)

    assert result.success is False
    assert result.extractor == "docling"
    assert (
        result.error
        == "Docling PDF models are not installed. Run `uv run dite setup docling-pdf` first."
    )


def test_docling_pdf_times_out(tmp_path: Path, monkeypatch) -> None:
    set_locale("en")
    pdf_path = tmp_path / "slow.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    class _SlowConverter:
        def convert(self, _file_path: str):
            time.sleep(1)
            raise AssertionError("timeout did not interrupt conversion")

    monkeypatch.setattr(
        "dite.extractors.docling.has_docling_pdf_artifacts",
        lambda _artifacts_path: True,
    )
    monkeypatch.setattr(
        DoclingExtractor,
        "_get_converter",
        lambda _self: _SlowConverter(),
    )

    result = DoclingExtractor(pdf_timeout_sec=0.01).extract(pdf_path)

    assert result.success is False
    assert result.extractor == "docling"
    assert result.error == "Docling PDF extraction timed out after 0.01s"
