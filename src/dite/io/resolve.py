"""Final extraction resolution and content selection helpers."""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from dite.app.config import Config
from dite.util.api import AsyncRequestRuntime
from dite.util.log import get_logger

from .base import ExtractionResult
from .pdf.final import (
    PDFContentSelection,
    PDFFallbackResolution,
    VLMSource,
    resolve_pdf_vlm_fallback,
    select_pdf_final_content,
)
from .pdf.policy import build_pdf_decision, compute_effective_content_length
from .types import ExtractorRegistry, ResolvedExtraction


def resolve_document_extraction(
    file_path: Path,
    client: OpenAI | None = None,
    *,
    config: Config,
    extract_document,
    sample_with_vlm,
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
            "Extractor result: "
            f"{resolved_primary.extractor} "
            f"success={resolved_primary.success} "
            "length="
            f"{len(resolved_primary.content if resolved_primary.success else '')} "
            f"error={resolved_primary.error or '-'}"
        )

    primary_content = resolved_primary.content if resolved_primary.success else ""
    normalized_primary = ExtractionResult(
        content=primary_content,
        success=resolved_primary.success,
        extractor=resolved_primary.extractor,
        error=resolved_primary.error,
    )
    primary_effective_length = compute_effective_content_length(primary_content)
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

    fallback_resolution: PDFFallbackResolution = resolve_pdf_vlm_fallback(
        file_path,
        client,
        config=config,
        primary_effective_length=primary_effective_length,
        decision=decision,
        cached_vlm_content=cached_vlm_content,
        sample_with_vlm=sample_with_vlm,
        request_runtime=request_runtime,
    )
    vlm_content = fallback_resolution.vlm_content
    vlm_source = fallback_resolution.vlm_source
    vlm_api_success = fallback_resolution.vlm_api_success
    vlm_api_page_calls = fallback_resolution.vlm_api_page_calls
    selection: PDFContentSelection = select_pdf_final_content(
        primary_content,
        primary_effective_length,
        vlm_content,
        vlm_source,
    )

    return ResolvedExtraction(
        primary_result=normalized_primary,
        primary_effective_length=primary_effective_length,
        pdf_profile=pdf_profile,
        fallback_needed=decision.fallback_needed,
        selected_source=selection.selected_source,
        final_content=selection.final_content,
        final_effective_length=selection.final_effective_length,
        vlm_content=vlm_content,
        vlm_source=vlm_source,
        vlm_api_success=vlm_api_success,
        vlm_api_page_calls=vlm_api_page_calls,
        sample_page_limit=fallback_resolution.sample_page_limit,
        cache_write_intent=selection.cache_write_intent,
    )
