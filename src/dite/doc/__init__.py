"""Document analysis, embedding, and data models."""

from .analyze import (
    ContentInfo,
    DocumentAnalysis,
    LayoutInfo,
    analyze_and_build_payload,
    analyze_document,
    build_weighted_payload,
)
from .build import build_document_features
from .embed import (
    ContentTruncator,
    EmbeddingInputMode,
    get_embedding_cache_version,
    get_embeddings,
    normalize_embeddings,
)
from .model import (
    DocumentFeatures,
    EntityFeatures,
    LayoutHints,
    MetadataFeatures,
    QualityFlags,
)

__all__ = [
    "ContentInfo",
    "DocumentAnalysis",
    "LayoutInfo",
    "analyze_and_build_payload",
    "analyze_document",
    "build_weighted_payload",
    "build_document_features",
    "ContentTruncator",
    "EmbeddingInputMode",
    "get_embedding_cache_version",
    "get_embeddings",
    "normalize_embeddings",
    "DocumentFeatures",
    "EntityFeatures",
    "LayoutHints",
    "MetadataFeatures",
    "QualityFlags",
]
