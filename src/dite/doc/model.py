"""Core feature-layer models for clustering V2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class QualityFlags:
    """Quality signals that influence downstream clustering decisions."""

    extraction_failed: bool = False
    short_text: bool = False
    filename_dominant: bool = False
    ocr_noisy: bool = False
    layout_sparse: bool = False
    language_uncertain: bool = False
    low_confidence_analysis: bool = False


@dataclass
class LayoutHints:
    """Lightweight structural and layout hints about a file."""

    document_type: str = ""
    columns: str = "single"
    has_table: bool = False
    has_image_heavy_layout: bool = False
    template_signals: list[str] = field(default_factory=list)
    page_count: int | None = None


@dataclass
class MetadataFeatures:
    """Normalized metadata-derived signals for file grouping."""

    file_name_tokens: list[str] = field(default_factory=list)
    parent_path_tokens: list[str] = field(default_factory=list)
    title_candidates: list[str] = field(default_factory=list)


@dataclass
class EntityFeatures:
    """Entity- and term-level grouping signals."""

    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    topic: str = ""
    domain: str = ""


@dataclass
class DocumentFeatures:
    """Structured multi-view document representation for clustering V2."""

    file_id: str
    path: Path
    name: str
    stem: str
    extension: str
    content_text: str = ""
    content_excerpt: str = ""
    language: str = ""
    token_count_estimate: int = 0
    summary: str = ""
    metadata: MetadataFeatures = field(default_factory=MetadataFeatures)
    entities: EntityFeatures = field(default_factory=EntityFeatures)
    layout: LayoutHints = field(default_factory=LayoutHints)
    quality_flags: QualityFlags = field(default_factory=QualityFlags)
    selected_source: str = ""
    extraction_trace: str = ""
    content_embedding: np.ndarray | None = None
    metadata_vector: np.ndarray | None = None
    entity_vector: np.ndarray | None = None
    layout_vector: np.ndarray | None = None

    @property
    def title_candidates(self) -> list[str]:
        """Expose title candidates at the top level for ergonomic access."""
        return self.metadata.title_candidates

    @property
    def file_name_tokens(self) -> list[str]:
        """Expose filename tokens at the top level for ergonomic access."""
        return self.metadata.file_name_tokens

    @property
    def parent_path_tokens(self) -> list[str]:
        """Expose parent-path tokens at the top level for ergonomic access."""
        return self.metadata.parent_path_tokens

    @property
    def keywords(self) -> list[str]:
        """Expose keywords at the top level for ergonomic access."""
        return self.entities.keywords

    @property
    def topic(self) -> str:
        """Expose topic at the top level for ergonomic access."""
        return self.entities.topic

    @property
    def domain(self) -> str:
        """Expose domain at the top level for ergonomic access."""
        return self.entities.domain
