import time
from pathlib import Path

from dite.extractors.base import ExtractionResult
from dite.extractors.docling import (
    DoclingExtractor,
    _docling_pdf_extract_child,
    _get_required_pdf_artifact_dirs,
    download_docling_pdf_models,
    extract_docling_pdf_in_subprocess,
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
        result.error == "Docling PDF models are not installed. Run "
        "`uv run dite setup docling-pdf` first."
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


def test_docling_extractor_passes_configured_device_to_accelerator_options(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeAcceleratorOptions:
        def __init__(self, *, device: str) -> None:
            captured["device"] = device

    class _FakePdfPipelineOptions:
        def __init__(self) -> None:
            self.do_ocr = True
            self.artifacts_path = None
            self.accelerator_options = None

    class _FakePdfFormatOption:
        def __init__(self, *, pipeline_options) -> None:
            captured["pipeline_options"] = pipeline_options

    class _FakeDocumentConverter:
        def __init__(self, *, format_options) -> None:
            captured["format_options"] = format_options

    monkeypatch.setattr(
        "docling.datamodel.accelerator_options.AcceleratorOptions",
        _FakeAcceleratorOptions,
    )
    monkeypatch.setattr(
        "docling.datamodel.pipeline_options.PdfPipelineOptions",
        _FakePdfPipelineOptions,
    )
    monkeypatch.setattr(
        "docling.document_converter.PdfFormatOption",
        _FakePdfFormatOption,
    )
    monkeypatch.setattr(
        "docling.document_converter.DocumentConverter",
        _FakeDocumentConverter,
    )

    extractor = DoclingExtractor(
        artifacts_path=tmp_path,
        pdf_timeout_sec=60.0,
        device="cpu",
    )
    converter = extractor._get_converter()

    assert isinstance(converter, _FakeDocumentConverter)
    assert captured["device"] == "cpu"
    pipeline_options = captured["pipeline_options"]
    assert pipeline_options.do_ocr is False
    assert pipeline_options.artifacts_path == tmp_path
    assert pipeline_options.accelerator_options is not None


def test_extract_docling_pdf_in_subprocess_returns_child_result(
    tmp_path: Path, monkeypatch
) -> None:
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    expected = ExtractionResult(
        content="docling child result",
        success=True,
        extractor="docling",
    )
    captured: dict[str, object] = {}

    class _Conn:
        def __init__(self, result: ExtractionResult | None) -> None:
            self._result = result
            self.closed = False

        def poll(self) -> bool:
            return self._result is not None

        def recv(self) -> ExtractionResult:
            assert self._result is not None
            return self._result

        def close(self) -> None:
            self.closed = True

    class _Process:
        exitcode = 0

        def __init__(self, target, args) -> None:
            captured["target"] = target
            captured["args"] = args
            self._alive = False

        def start(self) -> None:
            captured["started"] = True

        def join(self, timeout=None) -> None:
            captured.setdefault("join_calls", []).append(timeout)

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            raise AssertionError("terminate should not run in success path")

        def kill(self) -> None:
            raise AssertionError("kill should not run in success path")

    class _Context:
        def Pipe(self, duplex=False):
            assert duplex is False
            return _Conn(expected), _Conn(None)

        def Process(self, target, args):
            return _Process(target, args)

    monkeypatch.setattr(
        "dite.extractors.docling.multiprocessing.get_context",
        lambda method: _Context(),
    )

    result = extract_docling_pdf_in_subprocess(
        pdf_path,
        timeout_sec=0.5,
        locale="en",
    )

    assert result == expected
    assert captured["started"] is True
    assert captured["join_calls"] == [0.5]


def test_docling_pdf_extract_child_silences_docling_logging_before_extract(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[str] = []
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")

    class _Conn:
        def __init__(self) -> None:
            self.payload = None
            self.closed = False

        def send(self, payload) -> None:
            self.payload = payload

        def close(self) -> None:
            self.closed = True

    def fake_silence_docling_logging() -> None:
        events.append("silence")

    class _FakeDoclingExtractor:
        def __init__(self, **kwargs) -> None:
            del kwargs
            events.append("init")

        def extract(self, file_path: Path) -> ExtractionResult:
            events.append(f"extract:{file_path.name}")
            return ExtractionResult(
                content="child result",
                success=True,
                extractor="docling",
            )

    monkeypatch.setattr(
        "dite.extractors.docling.silence_docling_logging",
        fake_silence_docling_logging,
    )
    monkeypatch.setattr(
        "dite.extractors.docling.DoclingExtractor",
        _FakeDoclingExtractor,
    )

    conn = _Conn()
    _docling_pdf_extract_child(
        str(pdf_path),
        False,
        None,
        "en",
        "auto",
        conn,
    )

    assert events == ["silence", "init", f"extract:{pdf_path.name}"]
    assert conn.closed is True
    assert conn.payload == ExtractionResult(
        content="child result",
        success=True,
        extractor="docling",
    )


def test_extract_docling_pdf_in_subprocess_times_out_and_terminates(
    tmp_path: Path, monkeypatch
) -> None:
    set_locale("en")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    lifecycle: list[str] = []

    class _Conn:
        def poll(self) -> bool:
            return False

        def recv(self) -> ExtractionResult:
            raise AssertionError("recv should not run on timeout")

        def close(self) -> None:
            lifecycle.append("close_conn")

    class _Process:
        exitcode = None

        def __init__(self, target, args) -> None:
            del target, args
            self._alive = True

        def start(self) -> None:
            lifecycle.append("start")

        def join(self, timeout=None) -> None:
            lifecycle.append(f"join:{timeout}")

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            lifecycle.append("terminate")
            self._alive = False

        def kill(self) -> None:
            lifecycle.append("kill")

    class _Context:
        def Pipe(self, duplex=False):
            assert duplex is False
            return _Conn(), _Conn()

        def Process(self, target, args):
            return _Process(target, args)

    monkeypatch.setattr(
        "dite.extractors.docling.multiprocessing.get_context",
        lambda method: _Context(),
    )

    result = extract_docling_pdf_in_subprocess(
        pdf_path,
        timeout_sec=0.25,
        locale="en",
    )

    assert result.success is False
    assert result.extractor == "docling"
    assert result.error == "Docling PDF extraction timed out after 0.25s"
    assert "terminate" in lifecycle
