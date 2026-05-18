from __future__ import annotations

from pathlib import Path

from dite.config import Config
from dite.extractors.base import BaseExtractor, ExtractionResult
from dite.extractors.router import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    ExtractorRegistry,
    VLMSamplingResult,
    _compute_effective_content_length,
    _content_quality_score,
    _count_pdf_glyph_noise_tokens,
    _detect_real_type,
    _should_prefer_vlm_content,
    classify_pdf_profile,
    extract_content,
    extract_document,
    get_extractor,
    needs_vlm_fallback,
    resolve_document_extraction,
)
from dite.extractors.text import TextExtractor
from dite.extractors.vlm import VLMExtractor
from dite.i18n import set_locale
from dite.utils.api_runtime import ChatCompletionResult


class _StubExtractor(BaseExtractor):
    def __init__(
        self,
        name: str,
        extensions: set[str],
        result: ExtractionResult | None = None,
    ) -> None:
        self.name = name
        self._extensions = extensions
        self._result = result or ExtractionResult(
            content=f"{name}-content",
            success=True,
            extractor=name,
        )

    @property
    def supported_extensions(self) -> set[str]:
        return self._extensions

    def extract(self, file_path: Path) -> ExtractionResult:
        return self._result


class _Registry:
    def __init__(self) -> None:
        self.docling = _StubExtractor("docling", {".pdf", ".docx", ".pptx"})
        self.markitdown = _StubExtractor("markitdown", {".doc", ".ppt", ".xls"})
        self.text = _StubExtractor("text", {".txt", ".md"})
        self.vlm = _StubExtractor("vlm", {".png", ".jpg"})

    def get_docling(self) -> BaseExtractor:
        return self.docling

    def get_markitdown(self) -> BaseExtractor:
        return self.markitdown

    def get_text(self) -> BaseExtractor:
        return self.text

    def get_vlm(self) -> BaseExtractor:
        return self.vlm


class _FakeCompletions:
    def __init__(self, content: str, calls: list[dict]) -> None:
        self._content = content
        self._calls = calls

    def create(self, **kwargs):
        self._calls.append(kwargs)

        class _Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class _Choice:
            def __init__(self, content: str) -> None:
                self.message = _Message(content)

        class _Response:
            def __init__(self, content: str) -> None:
                self.choices = [_Choice(content)]

        return _Response(self._content)


class _FakeClient:
    def __init__(self, content: str, calls: list[dict]) -> None:
        class _Chat:
            def __init__(self, content: str, calls: list[dict]) -> None:
                self.completions = _FakeCompletions(content, calls)

        self.chat = _Chat(content, calls)


class _Runtime:
    def __init__(self, results: list[ChatCompletionResult]) -> None:
        self._results = results
        self.calls: list[object] = []

    def run_vlm_page_batch(self, requests, *, per_document_limit):
        self.calls.append((requests, per_document_limit))
        return self._results

    def run_image_vlm(self, request):
        self.calls.append(request)
        return self._results[0]


def test_detect_real_type_recognizes_magic_headers(tmp_path: Path) -> None:
    cases = {
        "sample.docx": b"PK\x03\x04extra",
        "sample.doc": b"\xd0\xcf\x11\xe0rest",
        "sample.pdf": b"%PDF-1.7",
        "sample.bin": b"ABCDEFGH",
    }

    expected = {
        "sample.docx": "ooxml",
        "sample.doc": "ole",
        "sample.pdf": "pdf",
        "sample.bin": "unknown",
    }

    for name, header in cases.items():
        file_path = tmp_path / name
        file_path.write_bytes(header)
        assert _detect_real_type(file_path) == expected[name]


def test_get_extractor_routes_by_real_type_and_extension(
    tmp_path: Path, monkeypatch
) -> None:
    registry = _Registry()

    image = tmp_path / "image.png"
    image.write_bytes(b"img")
    assert get_extractor(image, config=Config(), registry=registry) is registry.vlm

    text = tmp_path / "notes.txt"
    text.write_text("hello", encoding="utf-8")
    assert get_extractor(text, config=Config(), registry=registry) is registry.text

    wrong_suffix_ooxml = tmp_path / "slides.ppt"
    wrong_suffix_ooxml.write_bytes(b"PK\x03\x04fake")
    assert (
        get_extractor(wrong_suffix_ooxml, config=Config(), registry=registry)
        is registry.docling
    )

    old_office = tmp_path / "legacy.docx"
    old_office.write_bytes(b"\xd0\xcf\x11\xe0fake")
    assert (
        get_extractor(old_office, config=Config(), registry=registry)
        is registry.markitdown
    )

    pdf = tmp_path / "paper.bin"
    pdf.write_bytes(b"%PDF-1.7")
    assert get_extractor(pdf, config=Config(), registry=registry) is registry.docling

    unknown = tmp_path / "archive.xyz"
    unknown.write_bytes(b"1234")
    monkeypatch.setattr(
        "dite.extractors.router._detect_real_type",
        lambda path: "unknown",
    )
    assert get_extractor(unknown, config=Config(), registry=registry) is None


def test_extract_document_returns_failure_for_unsupported_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("i18n:\n  locale: zh-CN\n", encoding="utf-8")

    file_path = tmp_path / "unsupported.xyz"
    file_path.write_text("payload", encoding="utf-8")

    cfg = Config()
    cfg.i18n.locale = "zh-CN"
    set_locale("zh-CN")

    result = extract_document(file_path, config=cfg, registry=_Registry())

    assert result.success is False
    assert result.extractor == "none"
    assert "不支持的文件格式" in (result.error or "")


def test_extract_document_error_follows_passed_config_locale(tmp_path: Path) -> None:
    file_path = tmp_path / "unsupported.xyz"
    file_path.write_text("payload", encoding="utf-8")
    cfg = Config()
    cfg.i18n.locale = "en"
    set_locale("en")

    result = extract_document(file_path, registry=_Registry(), config=cfg)

    assert result.success is False
    assert result.extractor == "none"
    assert result.error == "Unsupported file format: .xyz"


def test_extractor_registry_passes_docling_pdf_timeout() -> None:
    cfg = Config()
    cfg.processing.docling_pdf_timeout_sec = 12.5

    extractor = ExtractorRegistry(cfg).get_docling()

    assert extractor._pdf_timeout_sec == 12.5


def test_get_extractor_keeps_registry_bound_vlm_client(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"img")
    cfg = Config()

    calls_a: list[dict] = []
    calls_b: list[dict] = []
    client_a = _FakeClient("client-a", calls_a)
    client_b = _FakeClient("client-b", calls_b)
    registry = ExtractorRegistry(cfg, client=client_a)

    extractor = get_extractor(
        image,
        client=client_b,
        config=cfg,
        registry=registry,
    )

    assert isinstance(extractor, VLMExtractor)
    result = extractor.extract(image)

    assert result.content == "client-a"
    assert len(calls_a) == 1
    assert not calls_b


def test_needs_vlm_fallback_only_for_short_pdf() -> None:
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 20

    assert needs_vlm_fallback("short text", Path("scan.pdf"), config=cfg) is True
    assert needs_vlm_fallback("x" * 50, Path("scan.pdf"), config=cfg) is False
    assert needs_vlm_fallback("short text", Path("scan.txt"), config=cfg) is False


def test_effective_content_length_ignores_pdf_glyph_noise() -> None:
    content = "/G25/G26/G27/G28 /G21/G22\n真正内容"

    assert _compute_effective_content_length(content) == len("真正内容")
    assert _content_quality_score(content) == len("真正内容")
    assert _count_pdf_glyph_noise_tokens(content) == 6


def test_should_prefer_vlm_content_uses_quality_not_raw_length() -> None:
    noisy_doc = "/G25/G26/G27/G28 " * 40 + ("A" * 147)
    vlm_content = "这是 VLM 提取出的正常文本。" * 20

    assert _should_prefer_vlm_content(noisy_doc, vlm_content) is True


def test_needs_vlm_fallback_for_pdf_glyph_noise() -> None:
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 20
    noisy = "/G25/G26/G27/G28 " * 50 + "题目"

    assert needs_vlm_fallback(noisy, Path("scan.pdf"), config=cfg) is True


def test_needs_vlm_fallback_when_glyph_noise_dominates_long_content() -> None:
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 100
    noisy = "/G25/G26/G27/G28 " * 40 + ("A" * 147)

    assert needs_vlm_fallback(noisy, Path("scan.pdf"), config=cfg) is True


def test_classify_pdf_profile_for_native_text() -> None:
    profile = classify_pdf_profile(
        "This is a readable PDF text layer." * 10,
        Path("paper.pdf"),
        config=Config(),
        success=True,
        vlm_fallback_threshold=100,
    )

    assert profile is not None
    assert profile.kind == "native_text"
    assert profile.needs_vlm_fallback is False
    assert profile.reason == "usable_text_layer"


def test_classify_pdf_profile_for_scanned_image() -> None:
    profile = classify_pdf_profile(
        "",
        Path("scan.pdf"),
        config=Config(),
        success=True,
        vlm_fallback_threshold=100,
    )

    assert profile is not None
    assert profile.kind == "scanned_image"
    assert profile.needs_vlm_fallback is True
    assert profile.reason == "no_effective_text"


def test_classify_pdf_profile_for_weak_glyph_noise() -> None:
    profile = classify_pdf_profile(
        "/G25/G26/G27/G28 " * 20 + "题目",
        Path("noisy.pdf"),
        config=Config(),
        success=True,
        vlm_fallback_threshold=20,
    )

    assert profile is not None
    assert profile.kind == "weak_text"
    assert profile.needs_vlm_fallback is True
    assert profile.reason == "glyph_noise_dominates"


def test_classify_pdf_profile_for_mixed_pdf() -> None:
    profile = classify_pdf_profile(
        ("正常正文内容" * 80) + " /G25/G26",
        Path("mixed.pdf"),
        config=Config(),
        success=True,
        vlm_fallback_threshold=100,
    )

    assert profile is not None
    assert profile.kind == "mixed_pdf"
    assert profile.needs_vlm_fallback is False
    assert profile.reason == "text_with_glyph_noise"


def test_classify_pdf_profile_for_parser_failure() -> None:
    profile = classify_pdf_profile(
        "",
        Path("broken.pdf"),
        config=Config(),
        success=False,
        error="timeout>60s",
        vlm_fallback_threshold=100,
    )

    assert profile is not None
    assert profile.kind == "parser_timeout_or_broken"
    assert profile.needs_vlm_fallback is True
    assert profile.reason == "timeout>60s"


def test_extract_content_uses_vlm_fallback_when_better(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    cfg.processing.text_truncate_limit = 100
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(content="too short", success=True, extractor="docling"),
    )

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        lambda file_path, client, config: VLMSamplingResult(
            result=ExtractionResult(
                content="much better vlm content",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=2,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        ),
    )

    result = extract_content(
        file_path,
        client=object(),
        config=cfg,
        registry=registry,
    )

    assert result == "much better vlm content"


def test_extract_content_keeps_original_when_vlm_not_better(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    cfg.processing.text_truncate_limit = 10
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(
            content="sufficient source",
            success=True,
            extractor="docling",
        ),
    )

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        lambda file_path, client, config: VLMSamplingResult(
            result=ExtractionResult(
                content="tiny",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=1,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        ),
    )

    result = extract_content(
        file_path,
        client=object(),
        config=cfg,
        registry=registry,
    )

    assert result == "sufficient"


def test_extract_content_prefers_vlm_when_doc_is_long_glyph_noise(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 100
    cfg.processing.text_truncate_limit = 5000
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(
            content=("/G25/G26/G27/G28 " * 40) + ("A" * 147),
            success=True,
            extractor="docling",
        ),
    )

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        lambda file_path, client, config, request_runtime=None: VLMSamplingResult(
            result=ExtractionResult(
                content="这是 VLM 提取出的正常文本。" * 20,
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=3,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        ),
    )

    result = extract_content(
        file_path,
        client=object(),
        config=cfg,
        registry=registry,
    )

    assert result.startswith("这是 VLM 提取出的正常文本")


def test_resolve_document_extraction_reports_vlm_sampling_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(content="short text", success=True, extractor="docling"),
    )

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        lambda file_path, client, config, request_runtime=None: VLMSamplingResult(
            result=ExtractionResult(
                content="much better vlm content",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=4,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        ),
    )

    resolved = resolve_document_extraction(
        file_path,
        client=object(),
        registry=registry,
        config=cfg,
    )

    assert resolved.fallback_needed is True
    assert resolved.selected_source == "vlm_api"
    assert resolved.final_content == "much better vlm content"
    assert resolved.vlm_api_page_calls == 4
    assert resolved.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_resolve_document_extraction_prefers_cached_vlm_without_api_call(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "cached.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(content="short text", success=True, extractor="docling"),
    )

    def fail_extract(*args, **kwargs):
        raise AssertionError("VLM API should not be called when cached VLM exists")

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        fail_extract,
    )

    resolved = resolve_document_extraction(
        file_path,
        client=object(),
        registry=registry,
        config=cfg,
        cached_vlm_content="cached vlm output " * 10,
    )

    assert resolved.fallback_needed is True
    assert resolved.selected_source == "vlm_cache"
    assert resolved.vlm_source == "cache"
    assert resolved.vlm_api_success is False
    assert resolved.vlm_api_page_calls == 0
    assert resolved.final_content.startswith("cached vlm output")
    assert resolved.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_resolve_document_extraction_keeps_fallback_needed_when_api_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "no-api.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(content="short text", success=True, extractor="docling"),
    )

    def fail_extract(*args, **kwargs):
        raise AssertionError("VLM API should stay disabled")

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        fail_extract,
    )

    resolved = resolve_document_extraction(
        file_path,
        client=object(),
        registry=registry,
        config=cfg,
        allow_vlm_api=False,
    )

    assert resolved.fallback_needed is True
    assert resolved.selected_source == "primary"
    assert resolved.vlm_source == "none"
    assert resolved.vlm_content is None
    assert resolved.vlm_api_success is False
    assert resolved.vlm_api_page_calls == 0
    assert resolved.final_content == "short text"
    assert resolved.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_resolve_document_extraction_marks_cache_write_for_selected_fresh_vlm(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "fresh-vlm.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(content="short text", success=True, extractor="docling"),
    )

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        lambda file_path, client, config: VLMSamplingResult(
            result=ExtractionResult(
                content="fresh vlm content " * 10,
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=2,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        ),
    )

    resolved = resolve_document_extraction(
        file_path,
        client=object(),
        registry=registry,
        config=cfg,
    )

    assert resolved.selected_source == "vlm_api"
    assert resolved.cache_write_intent.should_write is True
    assert resolved.cache_write_intent.content == resolved.vlm_content


def test_resolve_document_extraction_skips_cache_write_for_rejected_fresh_vlm(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "reject-vlm.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    registry = _Registry()
    primary_content = "usable primary content"
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(
            content=primary_content,
            success=True,
            extractor="docling",
        ),
    )

    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        lambda file_path, client, config: VLMSamplingResult(
            result=ExtractionResult(
                content="tiny",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=1,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        ),
    )

    resolved = resolve_document_extraction(
        file_path,
        client=object(),
        registry=registry,
        config=cfg,
    )

    assert resolved.selected_source == "primary"
    assert resolved.vlm_source == "api"
    assert resolved.vlm_api_success is True
    assert resolved.cache_write_intent.should_write is False
    assert resolved.cache_write_intent.content is None


def test_text_extractor_reads_text_and_reports_missing_file(tmp_path: Path) -> None:
    extractor = TextExtractor()
    file_path = tmp_path / "notes.txt"
    file_path.write_text("plain text", encoding="utf-8")

    success = extractor.extract(file_path)
    failure = extractor.extract(tmp_path / "missing.txt")

    assert success.success is True
    assert success.content == "plain text"
    assert failure.success is False
    assert failure.error


def test_vlm_extractor_handles_missing_client_and_success(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-really-a-png")

    cfg = Config()
    cfg.i18n.locale = "en"
    set_locale("en")
    no_client = VLMExtractor(cfg, client=None).extract(image_path)
    assert no_client.success is False
    assert no_client.error == "VLM client is not initialized"

    calls: list[dict] = []
    set_locale("en")
    extractor = VLMExtractor(cfg, client=_FakeClient("image summary", calls))
    result = extractor.extract(image_path)

    assert result.success is True
    assert result.content == "image summary"
    assert calls
    assert calls[0]["model"] == Config().models.vlm
    message = calls[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert message.startswith("data:image/png;base64,")
    prompt = calls[0]["messages"][0]["content"][1]["text"]
    assert "Describe this image in detail." in prompt


def test_vlm_extractor_uses_zh_prompt_when_locale_is_zh(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-really-a-png")

    cfg = Config()
    cfg.i18n.locale = "zh-CN"
    set_locale("zh-CN")

    calls: list[dict] = []
    extractor = VLMExtractor(cfg, client=_FakeClient("图像摘要", calls))
    result = extractor.extract(image_path)

    assert result.success is True
    prompt = calls[0]["messages"][0]["content"][1]["text"]
    assert "请详细描述这张图片的内容。" in prompt


def test_extract_document_uses_request_runtime_for_images(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image-bytes")

    runtime = _Runtime(
        [
            ChatCompletionResult(
                content="runtime image summary",
                error=None,
                queue_wait_sec=0.01,
                request_elapsed_sec=0.02,
            )
        ]
    )

    result = extract_document(
        image_path,
        client=object(),
        config=Config(),
        request_runtime=runtime,
    )

    assert result.success is True
    assert result.content == "runtime image summary"
    assert len(runtime.calls) == 1
    request = runtime.calls[0]
    assert request.kwargs["model"] == Config().models.vlm
    assert request.kwargs["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_resolve_document_extraction_uses_runtime_for_pdf_vlm_sampling(
    tmp_path: Path, monkeypatch
) -> None:
    file_path = tmp_path / "scan.pdf"
    file_path.write_bytes(b"%PDF-1.7")
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50
    cfg.processing.vlm_pages_per_document = 3
    registry = _Registry()
    registry.docling = _StubExtractor(
        "docling",
        {".pdf"},
        ExtractionResult(content="short text", success=True, extractor="docling"),
    )

    class _Image:
        width = 1000
        height = 1200

        def resize(self, new_size):
            resized = _Image()
            resized.width, resized.height = new_size
            return resized

        def save(self, buffer, _fmt):
            buffer.write(b"png")

    monkeypatch.setattr(
        "pdf2image.convert_from_path",
        lambda *_args, **_kwargs: [_Image(), _Image()],
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
                content="page 2",
                error=None,
                queue_wait_sec=0.01,
                request_elapsed_sec=0.03,
            ),
        ]
    )
    set_locale("en")

    resolved = resolve_document_extraction(
        file_path,
        client=object(),
        registry=registry,
        config=cfg,
        request_runtime=runtime,
    )

    assert resolved.selected_source == "vlm_api"
    assert resolved.vlm_api_page_calls == 2
    requests, per_document_limit = runtime.calls[0]
    assert len(requests) == 2
    assert per_document_limit == 3
    assert resolved.final_content == "[Page 1]\npage 1\n\n[Page 2]\npage 2"
