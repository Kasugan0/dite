"""Explicit data models for PDF fallback resolution and final selection."""

from dataclasses import dataclass
from typing import Literal

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
