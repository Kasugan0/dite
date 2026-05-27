from __future__ import annotations

from pathlib import Path

from dite.app.config import Config
from dite.app.i18n import set_locale
from dite.io.pdf.render import PDFRenderResult
from dite.io.pdf.vlm import extract_pdf_with_vlm_sampling
from dite.util.api import ChatCompletionResult
from dite.util.log import setup_logging


class _Image:
    width = 1000
    height = 1200

    def resize(self, new_size):
        resized = _Image()
        resized.width, resized.height = new_size
        return resized

    def save(self, buffer, _fmt):
        buffer.write(b"png")


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **_kwargs):
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response

        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = _Message(content)

        class _Response:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        return _Response(response)


class _FakeClient:
    def __init__(self, responses):
        class _Chat:
            def __init__(self, inner_responses) -> None:
                self.completions = _FakeCompletions(inner_responses)

        self.chat = _Chat(responses)


class _Runtime:
    def __init__(self, results: list[ChatCompletionResult]) -> None:
        self.results = results
        self.calls: list[object] = []

    def run_vlm_page_batch(self, requests, *, per_document_limit):
        self.calls.append((requests, per_document_limit))
        return self.results


def test_extract_pdf_with_vlm_sampling_propagates_render_error(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    set_locale("en")

    monkeypatch.setattr(
        "dite.io.pdf.vlm.render_pdf_pages",
        lambda path, *, max_pages: PDFRenderResult(
            pages=[],
            success=False,
            error="pdf2image is not installed, cannot use VLM fallback",
            sample_page_limit=max_pages,
        ),
    )

    sampling = extract_pdf_with_vlm_sampling(file_path, _FakeClient([]), config=cfg)

    assert sampling.result.success is False
    assert (
        sampling.result.error == "pdf2image is not installed, cannot use VLM fallback"
    )
    assert sampling.api_page_calls == 0
    assert sampling.sample_page_limit == 10


def test_extract_pdf_with_vlm_sampling_fails_when_all_api_pages_fail(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    set_locale("en")

    monkeypatch.setattr(
        "dite.io.pdf.vlm.render_pdf_pages",
        lambda path, *, max_pages: PDFRenderResult(
            pages=[_Image(), _Image()],
            success=True,
            error=None,
            sample_page_limit=max_pages,
        ),
    )

    sampling = extract_pdf_with_vlm_sampling(
        file_path,
        _FakeClient([RuntimeError("boom"), RuntimeError("boom again")]),
        config=cfg,
    )

    assert sampling.result.success is False
    assert sampling.result.error == "VLM fallback returned no usable content"
    assert sampling.api_page_calls == 2


def test_extract_pdf_with_vlm_sampling_fails_when_all_pages_are_empty(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    set_locale("en")

    monkeypatch.setattr(
        "dite.io.pdf.vlm.render_pdf_pages",
        lambda path, *, max_pages: PDFRenderResult(
            pages=[_Image(), _Image()],
            success=True,
            error=None,
            sample_page_limit=max_pages,
        ),
    )

    sampling = extract_pdf_with_vlm_sampling(
        file_path,
        _FakeClient(["   ", ""]),
        config=cfg,
    )

    assert sampling.result.success is False
    assert sampling.result.error == "VLM fallback returned no usable content"
    assert sampling.api_page_calls == 2


def test_extract_pdf_with_vlm_sampling_keeps_partial_runtime_success(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_pages_per_document = 3
    set_locale("en")

    monkeypatch.setattr(
        "dite.io.pdf.vlm.render_pdf_pages",
        lambda path, *, max_pages: PDFRenderResult(
            pages=[_Image(), _Image()],
            success=True,
            error=None,
            sample_page_limit=max_pages,
        ),
    )
    runtime = _Runtime(
        [
            ChatCompletionResult(
                content="",
                error="boom",
                queue_wait_sec=0.01,
                request_elapsed_sec=0.02,
            ),
            ChatCompletionResult(
                content="page 2",
                error=None,
                queue_wait_sec=0.01,
                request_elapsed_sec=0.03,
            ),
        ]
    )

    sampling = extract_pdf_with_vlm_sampling(
        file_path,
        _FakeClient([]),
        config=cfg,
        request_runtime=runtime,
    )

    assert sampling.result.success is True
    assert sampling.result.content == "[Page 2]\npage 2"
    assert sampling.api_page_calls == 2
    requests, per_document_limit = runtime.calls[0]
    assert len(requests) == 2
    assert per_document_limit == 3


def test_extract_pdf_with_vlm_sampling_runtime_logs_follow_locale(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_pages_per_document = 3
    setup_logging(verbose=True)
    set_locale("en")

    monkeypatch.setattr(
        "dite.io.pdf.vlm.render_pdf_pages",
        lambda path, *, max_pages: PDFRenderResult(
            pages=[_Image(), _Image()],
            success=True,
            error=None,
            sample_page_limit=max_pages,
        ),
    )
    runtime = _Runtime(
        [
            ChatCompletionResult(
                content="page 1",
                error=None,
                queue_wait_sec=0.01,
                request_elapsed_sec=0.02,
            ),
            ChatCompletionResult(
                content=None,
                error="boom",
                queue_wait_sec=0.03,
                request_elapsed_sec=0.04,
            ),
        ]
    )

    sampling = extract_pdf_with_vlm_sampling(
        file_path,
        _FakeClient([]),
        config=cfg,
        request_runtime=runtime,
    )

    output = capsys.readouterr().out

    assert sampling.result.success is True
    assert "Page 1 result: success" in output
    assert "Page 2 result: failed" in output
    assert "VLM batch summary: success=1, failed=1" in output
    assert "第 1 页结果" not in output
