"""Document extraction backends and routing."""

from . import route as router
from .base import BaseExtractor, ExtractionResult
from .docling import DoclingExtractor, extract_docling_pdf_in_subprocess
from .pdf import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    PDFCacheWriteIntent,
    PDFDecision,
    PDFProfile,
    PDFProfileKind,
    PDFRenderResult,
    ResolvedSource,
    VLMSamplingResult,
    VLMSource,
    build_pdf_decision,
    classify_pdf_profile,
    compute_effective_content_length,
    content_quality_score,
    count_pdf_glyph_noise_tokens,
    extract_pdf_with_vlm_sampling,
    render_pdf_pages,
    resolve_pdf_vlm_fallback,
    select_pdf_final_content,
    should_prefer_vlm_content,
)
from .pdf import (
    needs_vlm_fallback as needs_pdf_vlm_fallback,
)
from .route import (
    extract_content,
    extract_document,
    extract_with_vlm_fallback,
    get_extractor,
    needs_vlm_fallback,
    resolve_document_extraction,
)
from .types import ExtractorRegistry, ResolvedExtraction

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "router",
    "DoclingExtractor",
    "extract_docling_pdf_in_subprocess",
    "ExtractorRegistry",
    "ResolvedExtraction",
    "get_extractor",
    "extract_document",
    "resolve_document_extraction",
    "extract_with_vlm_fallback",
    "extract_content",
    "needs_vlm_fallback",
    "PDFCacheWriteIntent",
    "PDFDecision",
    "PDFProfile",
    "PDFProfileKind",
    "PDFRenderResult",
    "PDF_VLM_SAMPLE_PAGE_LIMIT",
    "ResolvedSource",
    "VLMSamplingResult",
    "VLMSource",
    "build_pdf_decision",
    "classify_pdf_profile",
    "compute_effective_content_length",
    "content_quality_score",
    "count_pdf_glyph_noise_tokens",
    "extract_pdf_with_vlm_sampling",
    "needs_pdf_vlm_fallback",
    "render_pdf_pages",
    "resolve_pdf_vlm_fallback",
    "select_pdf_final_content",
    "should_prefer_vlm_content",
]
