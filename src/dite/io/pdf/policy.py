"""PDF extraction policy and quality heuristics."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dite.app.config import Config

from ..base import ExtractionResult

PDFProfileKind = Literal[
    "native_text",
    "weak_text",
    "scanned_image",
    "mixed_pdf",
    "parser_timeout_or_broken",
]
PDFFallbackSource = Literal["none", "cache", "api"]

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
class PDFDecision:
    """Control-plane decision for a PDF after primary extraction."""

    profile: PDFProfile | None
    fallback_needed: bool
    fallback_source: PDFFallbackSource
    sample_page_limit: int | None
    reason: str


def compute_effective_content_length(content: str) -> int:
    """Compute useful content length after stripping known PDF noise."""
    import re

    clean = re.sub(r"<!--[^>]*-->", "", content)
    clean = re.sub(r"(?:/G[0-9A-F]{2})+", " ", clean)
    clean = re.sub(r"\b\d{3}\b\s*\n?", "", clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return len(clean)


def count_pdf_glyph_noise_tokens(content: str) -> int:
    """Count glyph-encoded PDF noise tokens."""
    import re

    return len(re.findall(r"/G[0-9A-F]{2}", content))


def content_quality_score(content: str) -> int:
    """Return a stable score used to compare extracted content quality."""
    return compute_effective_content_length(content)


def should_prefer_vlm_content(doc_content: str, vlm_content: str) -> bool:
    """Prefer the source with higher effective content quality."""
    return content_quality_score(vlm_content) > content_quality_score(doc_content)


def needs_vlm_fallback(
    content: str,
    file_path: Path,
    *,
    config: Config,
    vlm_fallback_threshold: int | None = None,
) -> bool:
    """Return whether PDF content is weak enough to justify VLM fallback."""
    threshold = vlm_fallback_threshold
    if threshold is None:
        threshold = config.processing.vlm_fallback_threshold

    if file_path.suffix.lower() != ".pdf":
        return False

    effective_len = compute_effective_content_length(content)
    glyph_noise = count_pdf_glyph_noise_tokens(content)
    if glyph_noise and glyph_noise * 4 >= max(effective_len, 1):
        return True

    return effective_len < threshold


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

    effective_length = compute_effective_content_length(content)
    glyph_noise_tokens = count_pdf_glyph_noise_tokens(content)
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

    fallback_needed = needs_vlm_fallback(
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
            needs_vlm_fallback=fallback_needed,
            success=True,
            reason="effective_text_below_threshold",
        )

    if glyph_noise_tokens:
        return PDFProfile(
            kind="mixed_pdf",
            effective_length=effective_length,
            glyph_noise_tokens=glyph_noise_tokens,
            glyph_noise_ratio=glyph_noise_ratio,
            needs_vlm_fallback=fallback_needed,
            success=True,
            reason="text_with_glyph_noise",
        )

    return PDFProfile(
        kind="native_text",
        effective_length=effective_length,
        glyph_noise_tokens=glyph_noise_tokens,
        glyph_noise_ratio=glyph_noise_ratio,
        needs_vlm_fallback=fallback_needed,
        success=True,
        reason="usable_text_layer",
    )


def build_pdf_decision(
    file_path: Path,
    *,
    config: Config,
    primary_result: ExtractionResult,
    cached_vlm_content: str | None,
    enable_vlm_fallback: bool,
    allow_vlm_api: bool,
    has_client: bool,
) -> PDFDecision:
    """Resolve how PDF fallback should proceed without side effects."""
    profile = classify_pdf_profile(
        primary_result.content if primary_result.success else "",
        file_path,
        config=config,
        success=primary_result.success,
        error=primary_result.error,
        vlm_fallback_threshold=config.processing.vlm_fallback_threshold,
    )
    if profile is None:
        return PDFDecision(
            profile=None,
            fallback_needed=False,
            fallback_source="none",
            sample_page_limit=None,
            reason="not_pdf",
        )

    fallback_needed = (
        enable_vlm_fallback
        and primary_result.extractor == "docling"
        and profile.needs_vlm_fallback
    )
    if not fallback_needed:
        return PDFDecision(
            profile=profile,
            fallback_needed=False,
            fallback_source="none",
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            reason=profile.reason,
        )

    if cached_vlm_content:
        return PDFDecision(
            profile=profile,
            fallback_needed=True,
            fallback_source="cache",
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            reason="cached_vlm_available",
        )

    if allow_vlm_api and has_client:
        return PDFDecision(
            profile=profile,
            fallback_needed=True,
            fallback_source="api",
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
            reason="vlm_api_allowed",
        )

    return PDFDecision(
        profile=profile,
        fallback_needed=True,
        fallback_source="none",
        sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        reason="vlm_fallback_unavailable",
    )
