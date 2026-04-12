"""内容提取器模块"""

from .base import BaseExtractor, ExtractionResult
from .router import (
    ExtractorRegistry,
    extract_content,
    extract_document,
    extract_with_vlm_fallback,
    get_extractor,
    needs_vlm_fallback,
)

__all__ = [
    "BaseExtractor",
    "ExtractionResult",
    "ExtractorRegistry",
    "get_extractor",
    "extract_content",
    "extract_document",
    "needs_vlm_fallback",
    "extract_with_vlm_fallback",
]
