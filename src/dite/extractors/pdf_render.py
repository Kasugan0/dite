"""PDF page rendering helpers used by PDF VLM fallback."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dite.i18n import t
from dite.utils.logging import get_logger


@dataclass(frozen=True)
class PDFRenderResult:
    """Rendered PDF pages plus status used by higher layers."""

    pages: list[Any]
    success: bool
    error: str | None
    sample_page_limit: int


def render_pdf_pages(file_path: Path, *, max_pages: int) -> PDFRenderResult:
    """Render at most ``max_pages`` pages from a PDF."""
    logger = get_logger()
    try:
        from pdf2image import convert_from_path
    except ImportError:
        logger.warning(t("warning_pdf2image_missing"))
        return PDFRenderResult(
            pages=[],
            success=False,
            error=t("warning_pdf2image_missing"),
            sample_page_limit=max_pages,
        )

    try:
        pages = convert_from_path(
            file_path,
            first_page=1,
            last_page=max_pages,
            dpi=150,
        )
    except Exception as exc:
        error_msg = str(exc)
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        return PDFRenderResult(
            pages=[],
            success=False,
            error=error_msg,
            sample_page_limit=max_pages,
        )

    if not pages:
        return PDFRenderResult(
            pages=[],
            success=False,
            error=t("error_pdf_render_failed"),
            sample_page_limit=max_pages,
        )

    return PDFRenderResult(
        pages=pages,
        success=True,
        error=None,
        sample_page_limit=max_pages,
    )
