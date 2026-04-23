from pathlib import Path

from dite.config import Config
from dite.extractors.base import ExtractionResult
from dite.extractors.pdf_policy import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    build_pdf_decision,
)


def test_build_pdf_decision_ignores_non_pdf() -> None:
    decision = build_pdf_decision(
        Path("notes.txt"),
        config=Config(),
        primary_result=ExtractionResult(
            content="payload",
            success=True,
            extractor="text",
        ),
        cached_vlm_content=None,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        has_client=True,
    )

    assert decision.profile is None
    assert decision.fallback_needed is False
    assert decision.fallback_source == "none"
    assert decision.sample_page_limit is None


def test_build_pdf_decision_keeps_native_text_on_primary() -> None:
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50

    decision = build_pdf_decision(
        Path("paper.pdf"),
        config=cfg,
        primary_result=ExtractionResult(
            content="This is a readable PDF text layer. " * 10,
            success=True,
            extractor="docling",
        ),
        cached_vlm_content=None,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        has_client=True,
    )

    assert decision.profile is not None
    assert decision.profile.kind == "native_text"
    assert decision.fallback_needed is False
    assert decision.fallback_source == "none"
    assert decision.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_build_pdf_decision_uses_cached_vlm_for_weak_pdf() -> None:
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50

    decision = build_pdf_decision(
        Path("scan.pdf"),
        config=cfg,
        primary_result=ExtractionResult(
            content="too short",
            success=True,
            extractor="docling",
        ),
        cached_vlm_content="cached fallback content",
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        has_client=False,
    )

    assert decision.profile is not None
    assert decision.profile.kind == "weak_text"
    assert decision.fallback_needed is True
    assert decision.fallback_source == "cache"
    assert decision.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_build_pdf_decision_reports_unavailable_api_without_client() -> None:
    cfg = Config()
    cfg.processing.vlm_fallback_threshold = 50

    decision = build_pdf_decision(
        Path("scan.pdf"),
        config=cfg,
        primary_result=ExtractionResult(
            content="too short",
            success=True,
            extractor="docling",
        ),
        cached_vlm_content=None,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        has_client=False,
    )

    assert decision.profile is not None
    assert decision.fallback_needed is True
    assert decision.fallback_source == "none"
    assert decision.reason == "vlm_fallback_unavailable"
