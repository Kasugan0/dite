"""提取器路由 - 根据文件类型选择合适的提取器"""

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI

from dite.config import Config
from dite.i18n import t
from dite.utils.api_runtime import AsyncRequestRuntime, ChatCompletionRequest
from dite.utils.logging import get_logger

from .base import BaseExtractor, ExtractionResult
from .docling import DoclingExtractor
from .markitdown import MarkItDownExtractor
from .pdf_policy import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    PDFDecision,
    PDFProfile,
    PDFProfileKind,
    build_pdf_decision,
    classify_pdf_profile,
    compute_effective_content_length,
    content_quality_score,
    count_pdf_glyph_noise_tokens,
    needs_vlm_fallback,
    should_prefer_vlm_content,
)
from .pdf_vlm import VLMSamplingResult, extract_pdf_with_vlm_sampling
from .text import TextExtractor
from .vlm import VLMExtractor, _build_vlm_prompt

ResolvedSource = Literal["primary", "vlm_cache", "vlm_api"]
VLMSource = Literal["none", "cache", "api"]


@dataclass(frozen=True)
class ResolvedExtraction:
    """Final extraction decision after optional PDF VLM fallback resolution."""

    primary_result: ExtractionResult
    primary_effective_length: int
    pdf_profile: PDFProfile | None
    fallback_needed: bool
    selected_source: ResolvedSource
    final_content: str
    final_effective_length: int
    vlm_content: str | None
    vlm_source: VLMSource
    vlm_api_success: bool
    vlm_api_page_calls: int
    sample_page_limit: int | None


_compute_effective_content_length = compute_effective_content_length
_count_pdf_glyph_noise_tokens = count_pdf_glyph_noise_tokens
_content_quality_score = content_quality_score
_should_prefer_vlm_content = should_prefer_vlm_content
_extract_pdf_with_vlm_sampling = extract_pdf_with_vlm_sampling

__all__ = [
    "PDFProfileKind",
    "PDFProfile",
    "PDF_VLM_SAMPLE_PAGE_LIMIT",
    "VLMSamplingResult",
    "ResolvedExtraction",
    "ExtractorRegistry",
    "get_extractor",
    "classify_pdf_profile",
    "needs_vlm_fallback",
    "extract_document",
    "resolve_document_extraction",
    "extract_with_vlm_fallback",
    "extract_content",
]


class ExtractorRegistry:
    """可替换的提取器注册表（按实例持有状态，不使用进程级全局单例）。"""

    def __init__(self, config: Config, client: OpenAI | None = None) -> None:
        self.config = config
        self._client = client
        self._docling: DoclingExtractor | None = None
        self._markitdown: MarkItDownExtractor | None = None
        self._text: TextExtractor | None = None
        self._vlm: VLMExtractor | None = None

    def get_docling(self) -> DoclingExtractor:
        if self._docling is None:
            self._docling = DoclingExtractor(
                pdf_timeout_sec=self.config.processing.docling_pdf_timeout_sec,
                device=self.config.processing.docling_device,
            )
        return self._docling

    def get_markitdown(self) -> MarkItDownExtractor:
        if self._markitdown is None:
            self._markitdown = MarkItDownExtractor()
        return self._markitdown

    def get_text(self) -> TextExtractor:
        if self._text is None:
            self._text = TextExtractor()
        return self._text

    def get_vlm(self) -> VLMExtractor:
        if self._vlm is None:
            self._vlm = VLMExtractor(config=self.config, client=self._client)
        return self._vlm


def _build_vlm_chat_request(
    *,
    model: str,
    image_url: str,
    prompt_text: str,
    timeout: float | None = None,
) -> ChatCompletionRequest:
    kwargs = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return ChatCompletionRequest(kwargs=kwargs)


def _extract_image_with_vlm_runtime(
    file_path: Path,
    *,
    config: Config,
    request_runtime: AsyncRequestRuntime,
) -> ExtractionResult:
    mime_type = VLMExtractor.MIME_TYPES.get(file_path.suffix.lower(), "image/jpeg")
    with open(file_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")
    request = _build_vlm_chat_request(
        model=config.models.vlm,
        image_url=f"data:{mime_type};base64,{image_data}",
        prompt_text=_build_vlm_prompt(),
    )
    result = request_runtime.run_image_vlm(request)
    if result.error is not None:
        return ExtractionResult(
            content="",
            success=False,
            extractor="vlm",
            error=str(result.error),
        )
    return ExtractionResult(
        content=result.content or "",
        success=True,
        extractor="vlm",
    )


def _detect_real_type(file_path: Path) -> str:
    """
    检测文件的真实类型（基于魔数而非扩展名）

    Returns:
        "ooxml" - DOCX/PPTX/XLSX (ZIP-based OOXML)
        "ole" - DOC/PPT/XLS (OLE compound)
        "pdf" - PDF
        "unknown" - 无法识别
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)

        # ZIP (OOXML: DOCX/PPTX/XLSX) - 开头 PK
        if header[:2] == b"PK":
            return "ooxml"

        # OLE Compound (旧版 Office) - D0 CF 11 E0
        if header[:4] == b"\xd0\xcf\x11\xe0":
            return "ole"

        # PDF - %PDF
        if header[:4] == b"%PDF":
            return "pdf"

        return "unknown"
    except Exception:
        return "unknown"


def get_extractor(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    registry: ExtractorRegistry | None = None,
) -> BaseExtractor | None:
    """
    根据文件类型获取合适的提取器

    路由策略：
    1. 图片 → VLM
    2. 纯文本 → Text
    3. 检测真实文件类型（魔数）
       - OOXML (含错误扩展名的 .ppt) → Docling
       - OLE (真正的旧版 Office) → MarkItDown
    4. 扩展名回退
    """
    reg = registry or ExtractorRegistry(config=config, client=client)
    suffix = file_path.suffix.lower()

    # 图片 → VLM
    if suffix in VLMExtractor.MIME_TYPES:
        return reg.get_vlm()

    # 纯文本 → Text
    text_ext = reg.get_text()
    if text_ext.can_handle(file_path):
        return text_ext

    # 检测真实文件类型
    real_type = _detect_real_type(file_path)

    # OOXML 格式（含扩展名错误的 .ppt/.doc/.xls）→ Docling
    if real_type == "ooxml":
        return reg.get_docling()

    # OLE 格式（真正的旧版 Office）→ MarkItDown
    if real_type == "ole":
        return reg.get_markitdown()

    # PDF → Docling
    if real_type == "pdf" or suffix == ".pdf":
        return reg.get_docling()

    # 扩展名回退
    docling_ext = reg.get_docling()
    if docling_ext.can_handle(file_path):
        return docling_ext

    markitdown_ext = reg.get_markitdown()
    if markitdown_ext.can_handle(file_path):
        return markitdown_ext

    return None


def extract_document(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    registry: ExtractorRegistry | None = None,
    request_runtime: AsyncRequestRuntime | None = None,
) -> ExtractionResult:
    """
    提取文档内容（仅文档转换，不含 VLM 回退）

    这是文档转换的核心函数，结果应该被长期缓存。

    Args:
        file_path: 文件路径
        client: OpenAI 客户端（用于纯图片文件的 VLM）

    Returns:
        ExtractionResult
    """
    logger = get_logger()
    extractor = get_extractor(
        file_path,
        client,
        config=config,
        registry=registry,
    )

    if extractor is None:
        logger.warning(t("warning_unsupported_file_format", suffix=file_path.suffix))
        return ExtractionResult(
            content="",
            success=False,
            extractor="none",
            error=t("warning_unsupported_file_format", suffix=file_path.suffix),
        )

    if isinstance(extractor, VLMExtractor) and request_runtime is not None:
        return _extract_image_with_vlm_runtime(
            file_path,
            config=config,
            request_runtime=request_runtime,
        )

    return extractor.extract(file_path)


def _resolve_pdf_vlm_fallback(
    file_path: Path,
    client: OpenAI | None,
    *,
    config: Config,
    primary_effective_length: int,
    decision: PDFDecision,
    cached_vlm_content: str | None,
    request_runtime: AsyncRequestRuntime | None = None,
) -> tuple[str | None, VLMSource, bool, int]:
    """Resolve PDF-only VLM fallback state without changing the public API."""
    logger = get_logger()
    pdf_profile = decision.profile
    assert pdf_profile is not None

    logger.debug(
        t(
            "debug_extract_vlm_check",
            suffix=file_path.suffix.lower(),
            effective_length=primary_effective_length,
            threshold=config.processing.vlm_fallback_threshold,
            needed=decision.fallback_needed,
        )
    )
    logger.debug(
        t(
            "debug_pdf_profile",
            kind=pdf_profile.kind,
            reason=pdf_profile.reason,
            effective_length=pdf_profile.effective_length,
            glyph_noise_tokens=pdf_profile.glyph_noise_tokens,
            needs_vlm_fallback=pdf_profile.needs_vlm_fallback,
        )
    )

    if decision.fallback_source == "cache":
        return cached_vlm_content, "cache", False, 0

    if decision.fallback_source != "api":
        return None, "none", False, 0

    logger.debug(t("debug_extract_vlm_api_call"))
    sampling_kwargs = {"config": config}
    if request_runtime is not None:
        sampling_kwargs["request_runtime"] = request_runtime
    sampling = _extract_pdf_with_vlm_sampling(
        file_path,
        client,
        **sampling_kwargs,
    )
    vlm_result = sampling.result
    logger.debug(
        t(
            "debug_extract_vlm_result",
            success=vlm_result.success,
            length=len(vlm_result.content),
            error=vlm_result.error or "-",
        )
    )
    if not vlm_result.success:
        return None, "none", False, sampling.api_page_calls

    return vlm_result.content, "api", True, sampling.api_page_calls


def _select_pdf_final_content(
    primary_content: str,
    primary_effective_length: int,
    vlm_content: str | None,
    vlm_source: VLMSource,
) -> tuple[ResolvedSource, str, int]:
    """Pick the best PDF content source after primary extraction and fallback."""
    logger = get_logger()
    if not vlm_content:
        return "primary", primary_content, primary_effective_length

    vlm_effective_length = _content_quality_score(vlm_content)
    if _should_prefer_vlm_content(primary_content, vlm_content):
        logger.debug(
            t(
                "debug_extract_vlm_selected",
                vlm_length=vlm_effective_length,
                doc_length=primary_effective_length,
            )
        )
        selected_source: ResolvedSource = (
            "vlm_cache" if vlm_source == "cache" else "vlm_api"
        )
        return selected_source, vlm_content, vlm_effective_length

    logger.debug(
        t(
            "debug_extract_vlm_skipped",
            doc_length=primary_effective_length,
            vlm_length=vlm_effective_length,
        )
    )
    return "primary", primary_content, primary_effective_length


def resolve_document_extraction(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    enable_vlm_fallback: bool = True,
    allow_vlm_api: bool = True,
    cached_vlm_content: str | None = None,
    primary_result: ExtractionResult | None = None,
    registry: ExtractorRegistry | None = None,
    request_runtime: AsyncRequestRuntime | None = None,
) -> ResolvedExtraction:
    """Resolve primary extraction, PDF fallback, and final content selection."""
    logger = get_logger()

    resolved_primary = primary_result
    if resolved_primary is None:
        extract_kwargs = {
            "config": config,
            "registry": registry,
        }
        if request_runtime is not None:
            extract_kwargs["request_runtime"] = request_runtime
        resolved_primary = extract_document(
            file_path,
            client,
            **extract_kwargs,
        )
        logger.debug(
            t(
                "debug_extract_doc_result",
                extractor=resolved_primary.extractor,
                success=resolved_primary.success,
                length=len(resolved_primary.content if resolved_primary.success else ""),
                error=resolved_primary.error or "-",
            )
        )

    primary_content = resolved_primary.content if resolved_primary.success else ""
    normalized_primary = ExtractionResult(
        content=primary_content,
        success=resolved_primary.success,
        extractor=resolved_primary.extractor,
        error=resolved_primary.error,
    )
    primary_effective_length = _compute_effective_content_length(primary_content)
    decision = build_pdf_decision(
        file_path,
        config=config,
        primary_result=normalized_primary,
        cached_vlm_content=cached_vlm_content,
        enable_vlm_fallback=enable_vlm_fallback,
        allow_vlm_api=allow_vlm_api,
        has_client=client is not None,
    )
    pdf_profile = decision.profile
    vlm_content = cached_vlm_content
    vlm_source: VLMSource = "cache" if cached_vlm_content is not None else "none"
    if pdf_profile is None:
        return ResolvedExtraction(
            primary_result=normalized_primary,
            primary_effective_length=primary_effective_length,
            pdf_profile=None,
            fallback_needed=False,
            selected_source="primary",
            final_content=primary_content,
            final_effective_length=primary_effective_length,
            vlm_content=vlm_content,
            vlm_source=vlm_source,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    vlm_content, vlm_source, vlm_api_success, vlm_api_page_calls = (
        _resolve_pdf_vlm_fallback(
            file_path,
            client,
            config=config,
            primary_effective_length=primary_effective_length,
            decision=decision,
            cached_vlm_content=cached_vlm_content,
            request_runtime=request_runtime,
        )
    )
    selected_source, final_content, final_effective_length = (
        _select_pdf_final_content(
            primary_content,
            primary_effective_length,
            vlm_content,
            vlm_source,
        )
    )

    return ResolvedExtraction(
        primary_result=normalized_primary,
        primary_effective_length=primary_effective_length,
        pdf_profile=pdf_profile,
        fallback_needed=decision.fallback_needed,
        selected_source=selected_source,
        final_content=final_content,
        final_effective_length=final_effective_length,
        vlm_content=vlm_content,
        vlm_source=vlm_source,
        vlm_api_success=vlm_api_success,
        vlm_api_page_calls=vlm_api_page_calls,
        sample_page_limit=decision.sample_page_limit,
    )


def extract_with_vlm_fallback(
    file_path: Path,
    client: OpenAI,
    *,
    config: Config,
    request_runtime: AsyncRequestRuntime | None = None,
) -> ExtractionResult:
    """
    使用 VLM 提取 PDF 内容（多页）

    Args:
        file_path: PDF 文件路径
        client: OpenAI 客户端

    Returns:
        ExtractionResult
    """
    sampling_kwargs = {"config": config}
    if request_runtime is not None:
        sampling_kwargs["request_runtime"] = request_runtime
    return _extract_pdf_with_vlm_sampling(
        file_path,
        client,
        **sampling_kwargs,
    ).result


def extract_content(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    truncate_limit: int | None = None,
    enable_vlm_fallback: bool = True,
    registry: ExtractorRegistry | None = None,
    request_runtime: AsyncRequestRuntime | None = None,
) -> str:
    """
    提取文件内容，返回截断后的文本

    Args:
        file_path: 文件路径
        client: OpenAI 客户端（用于 VLM）
        truncate_limit: 截断限制
        enable_vlm_fallback: 是否启用 VLM 回退（用于扫描件 PDF）

    Returns:
        提取的文本内容
    """
    if truncate_limit is None:
        truncate_limit = config.processing.text_truncate_limit

    resolve_kwargs = {
        "config": config,
        "enable_vlm_fallback": enable_vlm_fallback,
        "allow_vlm_api": True,
        "registry": registry,
    }
    if request_runtime is not None:
        resolve_kwargs["request_runtime"] = request_runtime
    resolved = resolve_document_extraction(
        file_path,
        client,
        **resolve_kwargs,
    )
    return resolved.final_content[:truncate_limit]
