"""Explicit data models and helpers for PDF fallback and final selection."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from openai import OpenAI

from dite.config import Config
from dite.i18n import t
from dite.utils.api_runtime import AsyncRequestRuntime
from dite.utils.logging import get_logger

from .pdf_policy import PDFDecision, content_quality_score, should_prefer_vlm_content

if TYPE_CHECKING:
    from .pdf_vlm import VLMSamplingResult

ResolvedSource = Literal["primary", "vlm_cache", "vlm_api"]
VLMSource = Literal["none", "cache", "api"]


@dataclass(frozen=True)
class PDFFallbackResolution:
    """Structured result of resolving PDF-only VLM fallback."""

    vlm_content: str | None
    vlm_source: VLMSource
    vlm_api_success: bool
    vlm_api_page_calls: int
    sample_page_limit: int | None


@dataclass(frozen=True)
class PDFCacheWriteIntent:
    """Whether fresh VLM content should be persisted to cache."""

    should_write: bool
    content: str | None


@dataclass(frozen=True)
class PDFContentSelection:
    """Final content choice after comparing primary and VLM PDF outputs."""

    selected_source: ResolvedSource
    final_content: str
    final_effective_length: int
    cache_write_intent: PDFCacheWriteIntent


def resolve_pdf_vlm_fallback(
    file_path: Path,
    client: OpenAI | None,
    *,
    config: Config,
    primary_effective_length: int,
    decision: PDFDecision,
    cached_vlm_content: str | None,
    sample_with_vlm: Callable[..., "VLMSamplingResult"],
    request_runtime: AsyncRequestRuntime | None = None,
) -> PDFFallbackResolution:
    """Resolve PDF-only VLM fallback state without changing outer APIs."""
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
        return PDFFallbackResolution(
            vlm_content=cached_vlm_content,
            vlm_source="cache",
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=decision.sample_page_limit,
        )

    if decision.fallback_source != "api":
        return PDFFallbackResolution(
            vlm_content=None,
            vlm_source="none",
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=decision.sample_page_limit,
        )

    logger.debug(t("debug_extract_vlm_api_call"))
    sampling_kwargs = {"config": config}
    if request_runtime is not None:
        sampling_kwargs["request_runtime"] = request_runtime
    sampling = sample_with_vlm(
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
        return PDFFallbackResolution(
            vlm_content=None,
            vlm_source="none",
            vlm_api_success=False,
            vlm_api_page_calls=sampling.api_page_calls,
            sample_page_limit=sampling.sample_page_limit,
        )

    return PDFFallbackResolution(
        vlm_content=vlm_result.content,
        vlm_source="api",
        vlm_api_success=True,
        vlm_api_page_calls=sampling.api_page_calls,
        sample_page_limit=sampling.sample_page_limit,
    )


def select_pdf_final_content(
    primary_content: str,
    primary_effective_length: int,
    vlm_content: str | None,
    vlm_source: VLMSource,
) -> PDFContentSelection:
    """Pick the best final PDF content and derive cache intent."""
    logger = get_logger()
    if not vlm_content:
        return PDFContentSelection(
            selected_source="primary",
            final_content=primary_content,
            final_effective_length=primary_effective_length,
            cache_write_intent=PDFCacheWriteIntent(
                should_write=False,
                content=None,
            ),
        )

    vlm_effective_length = content_quality_score(vlm_content)
    if should_prefer_vlm_content(primary_content, vlm_content):
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
        return PDFContentSelection(
            selected_source=selected_source,
            final_content=vlm_content,
            final_effective_length=vlm_effective_length,
            cache_write_intent=PDFCacheWriteIntent(
                should_write=vlm_source == "api",
                content=vlm_content if vlm_source == "api" else None,
            ),
        )

    logger.debug(
        t(
            "debug_extract_vlm_skipped",
            doc_length=primary_effective_length,
            vlm_length=vlm_effective_length,
        )
    )
    return PDFContentSelection(
        selected_source="primary",
        final_content=primary_content,
        final_effective_length=primary_effective_length,
        cache_write_intent=PDFCacheWriteIntent(
            should_write=False,
            content=None,
        ),
    )
