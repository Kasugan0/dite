"""Shared pipeline data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dite.cluster.model import (
    AdjudicationDecision,
    AdjudicationRequest,
    CandidateComponent,
    CandidateEdge,
    ClusterMetrics,
    ClusterRepresentation,
)
from dite.doc import DocumentFeatures
from dite.io.pdf.final import ResolvedSource


@dataclass
class PipelineOptions:
    """Pipeline execution options."""

    use_cache: bool = True
    use_embedding_cache: bool = True
    repair_noise: bool = True
    merge_same_name: bool = False
    allow_vlm_api: bool = True
    embedding_input_mode: str = "with_filename"
    cluster_allow_single_cluster: bool = False
    cluster_pca_components: int | None = None
    cluster_mode: str = "density"
    exclude_paths: list[Path] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Pipeline output and runtime metrics."""

    files: list[Path]
    contents: list[str]
    embeddings: np.ndarray
    labels: np.ndarray
    cluster_names: dict[int, str]
    cluster_representations: dict[int, ClusterRepresentation] = field(
        default_factory=dict
    )
    document_features: list[DocumentFeatures] = field(default_factory=list)
    candidate_edges: list[CandidateEdge] = field(default_factory=list)
    candidate_components: list[CandidateComponent] = field(default_factory=list)
    adjudication_requests: list[AdjudicationRequest] = field(default_factory=list)
    adjudication_decisions: list[AdjudicationDecision] = field(default_factory=list)
    noise_repaired: int = 0
    clusters_merged: int = 0
    cluster_metrics: ClusterMetrics = field(default_factory=lambda: ClusterMetrics())
    extraction: ExtractionSummary = field(default_factory=lambda: ExtractionSummary())
    file_reports: list[ExtractionFileReport] = field(default_factory=list)


@dataclass
class ExtractionFileReport:
    """Per-file extraction diagnostics used by CLI/reporting layers."""

    file: Path
    primary_extractor: str
    primary_success: bool
    primary_error: str | None
    source_profile: str | None
    source_effective_length: int
    selected_source: ResolvedSource
    final_effective_length: int
    excerpt_was_truncated: bool
    vlm_api_page_calls: int
    sample_page_limit: int | None
    file_hash: str
    source_reason: str | None = None
    fallback_needed: bool = False


@dataclass
class ExtractionSummary:
    """Aggregated extraction metrics separated from core pipeline output."""

    doc_cache_hits: int = 0
    vlm_cache_hits: int = 0
    primary_failures: int = 0
    source_fallback_needed: int = 0
    selected_vlm_files: int = 0
    vlm_api_page_calls: int = 0
    duplicate_count: int = 0
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractionSummaryDelta:
    doc_cache_hits: int = 0
    vlm_cache_hits: int = 0
    primary_failures: int = 0
    source_fallback_needed: int = 0
    selected_vlm_files: int = 0
    vlm_api_page_calls: int = 0


@dataclass(frozen=True)
class ExtractionWorkItem:
    index: int
    file: Path
    file_hash: str


@dataclass(frozen=True)
class ExtractionWorkResult:
    index: int
    content: str
    file_hash: str
    report: ExtractionFileReport
    summary_delta: ExtractionSummaryDelta
