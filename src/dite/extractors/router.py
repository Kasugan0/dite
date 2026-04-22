"""提取器路由 - 根据文件类型选择合适的提取器"""

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI

from dite.config import Config
from dite.i18n import get_locale, t
from dite.utils.logging import get_logger

from .base import BaseExtractor, ExtractionResult
from .docling import DoclingExtractor
from .markitdown import MarkItDownExtractor
from .text import TextExtractor
from .vlm import VLMExtractor

PDFProfileKind = Literal[
    "native_text",
    "weak_text",
    "scanned_image",
    "mixed_pdf",
    "parser_timeout_or_broken",
]
ResolvedSource = Literal["primary", "vlm_cache", "vlm_api"]
VLMSource = Literal["none", "cache", "api"]

PDF_VLM_SAMPLE_PAGE_LIMIT = 10


@dataclass(frozen=True)
class PDFProfile:
    """PDF extraction profile used to pick and explain the processing path."""

    kind: PDFProfileKind
    effective_length: int
    glyph_noise_tokens: int
    glyph_noise_ratio: float
    needs_vlm_fallback: bool
    success: bool
    reason: str


@dataclass(frozen=True)
class VLMSamplingResult:
    """VLM PDF sampling result plus runtime-only metrics."""

    result: ExtractionResult
    api_page_calls: int
    sample_page_limit: int


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


def _build_vlm_fallback_prompt(page_num: int) -> str:
    """Build a locale-aware fallback prompt for scanned PDF pages."""
    if get_locale() == "zh-CN":
        if page_num == 1:
            return (
                "这是一份扫描文档的第一页。请详细描述这份文档的内容，"
                "包括：文档类型、标题、核心主题、关键信息。"
                "如果有表格或图表，请描述其内容。"
                "直接输出描述，不要啰嗦。"
            )
        return (
            f"这是文档的第 {page_num} 页。请描述此页的主要内容，"
            "包括重要段落、公式、图表等信息。"
            "直接输出描述，不要啰嗦。"
        )

    if page_num == 1:
        return (
            "This is the first page of a scanned document. "
            "Describe the document in detail, including its type, title, core topic, "
            "and key information. If there are tables or charts, describe them too. "
            "Return only the description."
        )
    return (
        f"This is page {page_num} of the document. "
        "Describe the main content of this page, including important paragraphs, "
        "formulas, charts, and other key information. Return only the description."
    )


def _format_vlm_page_content(page_num: int, content: str) -> str:
    """Format page content markers in the active locale."""
    if get_locale() == "zh-CN":
        return f"【第{page_num}页】\n{content}"
    return f"[Page {page_num}]\n{content}"


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


def _extract_pdf_with_vlm_sampling(
    file_path: Path,
    client: OpenAI,
    config: Config,
    max_pages: int = PDF_VLM_SAMPLE_PAGE_LIMIT,
) -> VLMSamplingResult:
    """
    使用 VLM 处理 PDF 多页（用于扫描件回退）

    Args:
        file_path: PDF 文件路径
        client: OpenAI 客户端
        max_pages: 最多提取的页数（默认10页）

    Returns:
        ExtractionResult
    """
    logger = get_logger()
    try:
        # 尝试导入 pdf2image
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning(t("warning_pdf2image_missing"))
        return VLMSamplingResult(
            result=ExtractionResult(
                content="",
                success=False,
                extractor="vlm_fallback",
                error=t("warning_pdf2image_missing"),
            ),
            api_page_calls=0,
            sample_page_limit=max_pages,
        )

    try:
        # 渲染 PDF 多页为图片
        images = convert_from_path(
            file_path,
            first_page=1,
            last_page=max_pages,
            dpi=150,
        )

        if not images:
            return VLMSamplingResult(
                result=ExtractionResult(
                    content="",
                    success=False,
                    extractor="vlm_fallback",
                    error=t("error_pdf_render_failed"),
                ),
                api_page_calls=0,
                sample_page_limit=max_pages,
            )

        # 收集所有页面的描述
        all_contents = []
        api_page_calls = 0

        for page_num, image in enumerate(images, start=1):
            logger.debug(
                t(
                    "debug_vlm_page_processing",
                    page=page_num,
                    total=len(images),
                    width=image.width,
                    height=image.height,
                )
            )

            # 限制图片尺寸：长边最大 1920，短边最大 1080
            max_long = 1920
            max_short = 1080
            width, height = image.width, image.height

            if width > height:  # 横向
                if width > max_long or height > max_short:
                    ratio = min(max_long / width, max_short / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    image = image.resize(new_size)
                    logger.debug(
                        t(
                            "debug_vlm_page_resized",
                            old_width=width,
                            old_height=height,
                            new_width=new_size[0],
                            new_height=new_size[1],
                        )
                    )
            else:  # 纵向
                if height > max_long or width > max_short:
                    ratio = min(max_short / width, max_long / height)
                    new_size = (int(width * ratio), int(height * ratio))
                    image = image.resize(new_size)
                    logger.debug(
                        t(
                            "debug_vlm_page_resized",
                            old_width=width,
                            old_height=height,
                            new_width=new_size[0],
                            new_height=new_size[1],
                        )
                    )

            # 保存为临时 PNG 文件
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                image.save(tmp.name, "PNG")
                temp_path = Path(tmp.name)

            try:
                # 读取图片并转为 base64
                with open(temp_path, "rb") as f:
                    image_data = base64.b64encode(f.read()).decode("utf-8")

                file_size_kb = temp_path.stat().st_size / 1024
                logger.debug(t("debug_vlm_image_size", size_kb=file_size_kb))

                # 根据页数调整 prompt
                prompt_text = _build_vlm_fallback_prompt(page_num)

                # 调用 VLM（设置超时）
                logger.debug(t("debug_vlm_api_call"))
                api_page_calls += 1
                response = client.chat.completions.create(
                    model=config.models.vlm,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{image_data}"
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt_text,
                                },
                            ],
                        }
                    ],
                    max_tokens=500,
                    timeout=60.0,  # 60 秒超时
                )

                page_content = response.choices[0].message.content or ""
                if page_content.strip():
                    all_contents.append(
                        _format_vlm_page_content(page_num, page_content.strip())
                    )
                    logger.debug(
                        t(
                            "debug_vlm_page_done",
                            page=page_num,
                            length=len(page_content),
                        )
                    )

            except Exception as api_error:
                logger.debug(
                    t(
                        "debug_vlm_page_failed",
                        page=page_num,
                        error=api_error,
                    )
                )

            finally:
                # 清理临时文件
                temp_path.unlink(missing_ok=True)

        # 合并所有页面内容
        combined_content = "\n\n".join(all_contents)

        return VLMSamplingResult(
            result=ExtractionResult(
                content=combined_content,
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=api_page_calls,
            sample_page_limit=max_pages,
        )

    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        logger.warning(t("warning_vlm_fallback_failed", error=error_msg))
        return VLMSamplingResult(
            result=ExtractionResult(
                content="",
                success=False,
                extractor="vlm_fallback",
                error=error_msg,
            ),
            api_page_calls=0,
            sample_page_limit=max_pages,
        )


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


def _compute_effective_content_length(content: str) -> int:
    """
    计算有效内容长度（排除标记和无意义内容）

    Args:
        content: 原始内容

    Returns:
        有效内容的字符数
    """
    import re

    # 移除 HTML 注释标记（如 <!-- image -->）
    clean = re.sub(r"<!--[^>]*-->", "", content)

    # 移除 PDF 字形编码噪音（如 /G25/G26），这类内容很长但几乎没有语义。
    clean = re.sub(r"(?:/G[0-9A-F]{2})+", " ", clean)

    # 移除连续的数字行号（如 000, 001, 002...）
    clean = re.sub(r"\b\d{3}\b\s*\n?", "", clean)

    # 移除多余空白
    clean = re.sub(r"\s+", " ", clean).strip()

    return len(clean)


def _count_pdf_glyph_noise_tokens(content: str) -> int:
    """统计 PDF 提取结果中的字形编码噪音数量。"""
    import re

    return len(re.findall(r"/G[0-9A-F]{2}", content))


def _content_quality_score(content: str) -> int:
    """返回提取内容的有效质量分数。"""
    return _compute_effective_content_length(content)


def _should_prefer_vlm_content(doc_content: str, vlm_content: str) -> bool:
    """统一使用有效内容质量来决定是否采用 VLM 结果。"""
    return _content_quality_score(vlm_content) > _content_quality_score(doc_content)


def classify_pdf_profile(
    content: str,
    file_path: Path,
    *,
    config: Config,
    success: bool,
    error: str | None = None,
    vlm_fallback_threshold: int | None = None,
) -> PDFProfile | None:
    """Classify a PDF by the processing path DITE should use."""
    if file_path.suffix.lower() != ".pdf":
        return None

    threshold = vlm_fallback_threshold
    if threshold is None:
        threshold = config.processing.vlm_fallback_threshold

    effective_length = _compute_effective_content_length(content)
    glyph_noise_tokens = _count_pdf_glyph_noise_tokens(content)
    glyph_noise_ratio = glyph_noise_tokens / max(effective_length, 1)

    if not success:
        return PDFProfile(
            kind="parser_timeout_or_broken",
            effective_length=effective_length,
            glyph_noise_tokens=glyph_noise_tokens,
            glyph_noise_ratio=glyph_noise_ratio,
            needs_vlm_fallback=True,
            success=False,
            reason=error or "extractor_failed",
        )

    needs_fallback = needs_vlm_fallback(
        content,
        file_path,
        config=config,
        vlm_fallback_threshold=vlm_fallback_threshold,
    )
    if effective_length == 0:
        return PDFProfile(
            kind="scanned_image",
            effective_length=effective_length,
            glyph_noise_tokens=glyph_noise_tokens,
            glyph_noise_ratio=glyph_noise_ratio,
            needs_vlm_fallback=True,
            success=True,
            reason="no_effective_text",
        )

    if glyph_noise_tokens and glyph_noise_tokens * 4 >= max(effective_length, 1):
        return PDFProfile(
            kind="weak_text",
            effective_length=effective_length,
            glyph_noise_tokens=glyph_noise_tokens,
            glyph_noise_ratio=glyph_noise_ratio,
            needs_vlm_fallback=True,
            success=True,
            reason="glyph_noise_dominates",
        )

    if effective_length < threshold:
        return PDFProfile(
            kind="weak_text",
            effective_length=effective_length,
            glyph_noise_tokens=glyph_noise_tokens,
            glyph_noise_ratio=glyph_noise_ratio,
            needs_vlm_fallback=needs_fallback,
            success=True,
            reason="effective_text_below_threshold",
        )

    if glyph_noise_tokens:
        return PDFProfile(
            kind="mixed_pdf",
            effective_length=effective_length,
            glyph_noise_tokens=glyph_noise_tokens,
            glyph_noise_ratio=glyph_noise_ratio,
            needs_vlm_fallback=needs_fallback,
            success=True,
            reason="text_with_glyph_noise",
        )

    return PDFProfile(
        kind="native_text",
        effective_length=effective_length,
        glyph_noise_tokens=glyph_noise_tokens,
        glyph_noise_ratio=glyph_noise_ratio,
        needs_vlm_fallback=needs_fallback,
        success=True,
        reason="usable_text_layer",
    )


def extract_document(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    registry: ExtractorRegistry | None = None,
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

    return extractor.extract(file_path)


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
) -> ResolvedExtraction:
    """Resolve primary extraction, PDF fallback, and final content selection."""
    logger = get_logger()

    resolved_primary = primary_result
    if resolved_primary is None:
        resolved_primary = extract_document(
            file_path,
            client,
            config=config,
            registry=registry,
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
    pdf_profile = classify_pdf_profile(
        primary_content,
        file_path,
        config=config,
        success=normalized_primary.success,
        error=normalized_primary.error,
        vlm_fallback_threshold=config.processing.vlm_fallback_threshold,
    )
    fallback_needed = False
    final_content = primary_content
    final_effective_length = primary_effective_length
    selected_source: ResolvedSource = "primary"
    vlm_content = cached_vlm_content
    vlm_source: VLMSource = "cache" if cached_vlm_content is not None else "none"
    vlm_api_success = False
    vlm_api_page_calls = 0
    sample_page_limit = (
        PDF_VLM_SAMPLE_PAGE_LIMIT if file_path.suffix.lower() == ".pdf" else None
    )

    if file_path.suffix.lower() == ".pdf":
        fallback_needed = (
            enable_vlm_fallback
            and client is not None
            and normalized_primary.extractor == "docling"
            and pdf_profile is not None
            and pdf_profile.needs_vlm_fallback
        )
        logger.debug(
            t(
                "debug_extract_vlm_check",
                suffix=file_path.suffix.lower(),
                effective_length=primary_effective_length,
                threshold=config.processing.vlm_fallback_threshold,
                needed=fallback_needed,
            )
        )
        if pdf_profile is not None:
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

        if fallback_needed and vlm_content is None and allow_vlm_api:
            logger.debug(t("debug_extract_vlm_api_call"))
            sampling = _extract_pdf_with_vlm_sampling(file_path, client, config=config)
            vlm_result = sampling.result
            vlm_api_page_calls = sampling.api_page_calls
            logger.debug(
                t(
                    "debug_extract_vlm_result",
                    success=vlm_result.success,
                    length=len(vlm_result.content),
                    error=vlm_result.error or "-",
                )
            )
            if vlm_result.success:
                vlm_content = vlm_result.content
                vlm_source = "api"
                vlm_api_success = True

        if vlm_content and _should_prefer_vlm_content(primary_content, vlm_content):
            final_content = vlm_content
            final_effective_length = _content_quality_score(vlm_content)
            selected_source = "vlm_cache" if vlm_source == "cache" else "vlm_api"
            logger.debug(
                t(
                    "debug_extract_vlm_selected",
                    vlm_length=final_effective_length,
                    doc_length=primary_effective_length,
                )
            )
        elif vlm_content:
            logger.debug(
                t(
                    "debug_extract_vlm_skipped",
                    doc_length=primary_effective_length,
                    vlm_length=_content_quality_score(vlm_content),
                )
            )

    return ResolvedExtraction(
        primary_result=normalized_primary,
        primary_effective_length=primary_effective_length,
        pdf_profile=pdf_profile,
        fallback_needed=fallback_needed,
        selected_source=selected_source,
        final_content=final_content,
        final_effective_length=final_effective_length,
        vlm_content=vlm_content,
        vlm_source=vlm_source,
        vlm_api_success=vlm_api_success,
        vlm_api_page_calls=vlm_api_page_calls,
        sample_page_limit=sample_page_limit,
    )


def needs_vlm_fallback(
    content: str,
    file_path: Path,
    *,
    config: Config,
    vlm_fallback_threshold: int | None = None,
) -> bool:
    """
    判断是否需要 VLM 回退

    Args:
        content: docling/markitdown 提取的内容
        file_path: 文件路径

    Returns:
        是否需要 VLM 回退
    """
    threshold = vlm_fallback_threshold
    if threshold is None:
        threshold = config.processing.vlm_fallback_threshold

    # 仅对 PDF 生效
    if file_path.suffix.lower() != ".pdf":
        return False

    effective_len = _compute_effective_content_length(content)
    glyph_noise = _count_pdf_glyph_noise_tokens(content)
    if glyph_noise and glyph_noise * 4 >= max(effective_len, 1):
        return True

    return effective_len < threshold


def extract_with_vlm_fallback(
    file_path: Path,
    client: OpenAI,
    *,
    config: Config,
) -> ExtractionResult:
    """
    使用 VLM 提取 PDF 内容（多页）

    Args:
        file_path: PDF 文件路径
        client: OpenAI 客户端

    Returns:
        ExtractionResult
    """
    return _extract_pdf_with_vlm_sampling(file_path, client, config=config).result


def extract_content(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    truncate_limit: int | None = None,
    enable_vlm_fallback: bool = True,
    registry: ExtractorRegistry | None = None,
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

    resolved = resolve_document_extraction(
        file_path,
        client,
        config=config,
        enable_vlm_fallback=enable_vlm_fallback,
        allow_vlm_api=True,
        registry=registry,
    )
    return resolved.final_content[:truncate_limit]
