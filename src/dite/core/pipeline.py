"""Shared document processing pipeline for scan and organize commands."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI

from dite.cache import VLM_CACHE_VERSION, FileCache
from dite.config import Config
from dite.extractors import (
    ExtractorRegistry,
    get_extractor,
)
from dite.extractors.base import ExtractionResult
from dite.extractors.router import (
    ResolvedSource,
    resolve_document_extraction,
)
from dite.i18n import t
from dite.utils.hashing import compute_file_hash
from dite.utils.logging import get_logger

from .clusterer import cluster_documents, generate_all_cluster_names
from .embedder import ContentTruncator, get_embedding_cache_version, get_embeddings
from .scanner import scan_files


@dataclass
class PipelineOptions:
    """Pipeline execution options."""

    use_cache: bool = True
    use_embedding_cache: bool = True
    repair_noise: bool = True
    merge_same_name: bool = False
    allow_vlm_api: bool = True
    exclude_paths: list[Path] = field(default_factory=list)


@dataclass
class PipelineResult:
    """Pipeline output and runtime metrics."""

    files: list[Path]
    contents: list[str]
    embeddings: np.ndarray
    labels: np.ndarray
    cluster_names: dict[int, str]
    noise_repaired: int = 0
    clusters_merged: int = 0
    extraction: "ExtractionSummary" = field(default_factory=lambda: ExtractionSummary())
    file_reports: list["ExtractionFileReport"] = field(default_factory=list)


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


class PipelineService:
    """Unified scan/extract/embed/cluster/name pipeline."""

    def __init__(
        self,
        client: OpenAI,
        *,
        config: Config,
        cache: FileCache | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.cache = cache
        self.logger = get_logger()

    def _make_extractor_registry(self) -> ExtractorRegistry:
        return ExtractorRegistry(config=self.config, client=self.client)

    def run(self, folder: Path, options: PipelineOptions) -> PipelineResult:
        files = scan_files(
            folder,
            config=self.config,
            extensions=self.config.formats.all_extensions,
            exclude_paths=options.exclude_paths,
        )
        if not files:
            return PipelineResult(
                files=[],
                contents=[],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
                extraction=ExtractionSummary(),
                file_reports=[],
            )

        contents, file_hashes, extraction, file_reports = self._extract_contents(
            files, options
        )
        embeddings = self._vectorize(files, file_hashes, contents, options)
        labels, noise_repaired = cluster_documents(
            embeddings,
            config=self.config,
            repair_noise=options.repair_noise,
            clustering=self.config.clustering,
            item_names=[file.name for file in files],
        )
        labels, cluster_names, clusters_merged = generate_all_cluster_names(
            self.client,
            labels,
            contents,
            files,
            config=self.config,
            embeddings=embeddings,
            merge_same_name=options.merge_same_name,
            llm_model=self.config.models.llm,
        )

        return PipelineResult(
            files=files,
            contents=contents,
            embeddings=embeddings,
            labels=labels,
            cluster_names=cluster_names,
            noise_repaired=noise_repaired,
            clusters_merged=clusters_merged,
            extraction=extraction,
            file_reports=file_reports,
        )

    def extract_files(
        self, files: list[Path], options: PipelineOptions
    ) -> PipelineResult:
        """Extract file contents without embedding, clustering, or naming."""
        if not files:
            return PipelineResult(
                files=[],
                contents=[],
                embeddings=np.array([]),
                labels=np.array([]),
                cluster_names={},
                extraction=ExtractionSummary(),
                file_reports=[],
            )

        contents, _file_hashes, extraction, file_reports = self._extract_contents(
            files, options
        )
        return PipelineResult(
            files=files,
            contents=contents,
            embeddings=np.array([]),
            labels=np.array([]),
            cluster_names={},
            extraction=extraction,
            file_reports=file_reports,
        )

    def _extract_contents(
        self,
        files: list[Path],
        options: PipelineOptions,
    ) -> tuple[list[str], list[str], ExtractionSummary, list[ExtractionFileReport]]:
        contents: list[str] = []
        file_hashes: list[str] = []
        file_reports: list[ExtractionFileReport] = []
        summary = ExtractionSummary()
        hash_to_files: dict[str, list[str]] = {}
        truncate_limit = self.config.processing.text_truncate_limit
        extractor_registry = self._make_extractor_registry()

        for file in files:
            self.logger.debug(t("debug_extract_processing_file", path=file))
            file_hash = compute_file_hash(file)
            file_hashes.append(file_hash)
            if file_hash:
                self.logger.debug(t("debug_extract_hash", hash=file_hash[:12]))
                hash_to_files.setdefault(file_hash, []).append(str(file))

            primary_result = None
            cached_vlm_content = None
            if options.use_cache and self.cache:
                cached_primary_content, source_file = self.cache.get_content(
                    file, file_hash
                )
                if cached_primary_content:
                    extractor = get_extractor(
                        file,
                        self.client,
                        config=self.config,
                        registry=extractor_registry,
                    )
                    primary_result = self._build_cached_primary_result(
                        file,
                        cached_primary_content,
                        extractor.name if extractor is not None else "cache",
                    )
                    summary.doc_cache_hits += 1
                    self.logger.debug(t("debug_extract_doc_cache_hit"))
                    if source_file:
                        self.logger.debug(
                            t(
                                "debug_extract_doc_cache_duplicate_source",
                                source=Path(source_file).name,
                            )
                        )
                else:
                    self.logger.debug(t("debug_extract_doc_cache_miss"))

            if file.suffix.lower() == ".pdf" and options.use_cache and self.cache:
                cached_vlm_content = self.cache.get_vlm_content(
                    file, file_hash, VLM_CACHE_VERSION
                )
                if cached_vlm_content:
                    summary.vlm_cache_hits += 1
                    self.logger.debug(
                        t(
                            "debug_extract_vlm_cache_hit",
                            length=len(cached_vlm_content),
                        )
                    )

            resolved = resolve_document_extraction(
                file,
                self.client,
                config=self.config,
                enable_vlm_fallback=True,
                allow_vlm_api=options.allow_vlm_api,
                cached_vlm_content=cached_vlm_content,
                primary_result=primary_result,
                registry=extractor_registry,
            )

            if (
                primary_result is None
                and options.use_cache
                and self.cache
                and resolved.primary_result.success
            ):
                self.cache.save(
                    file_path=file,
                    file_hash=file_hash,
                    file_mtime=file.stat().st_mtime,
                    content_md=resolved.primary_result.content,
                    model_version=self.config.models.embedding,
                )

            if resolved.primary_result.success is False:
                summary.primary_failures += 1
            if resolved.fallback_needed:
                summary.source_fallback_needed += 1
            if resolved.selected_source != "primary":
                summary.selected_vlm_files += 1
            summary.vlm_api_page_calls += resolved.vlm_api_page_calls

            if (
                resolved.vlm_source == "api"
                and resolved.vlm_api_success
                and resolved.vlm_content
                and options.use_cache
                and self.cache
            ):
                self.cache.update_vlm_content(
                    file,
                    file_hash,
                    resolved.vlm_content,
                    VLM_CACHE_VERSION,
                )

            if len(resolved.final_content) > truncate_limit:
                self.logger.debug(
                    t(
                        "debug_extract_truncated",
                        original=len(resolved.final_content),
                        limit=truncate_limit,
                    )
                )
            contents.append(
                ContentTruncator.truncate_smart(
                    resolved.final_content,
                    max_chars=truncate_limit,
                )
            )
            file_reports.append(
                ExtractionFileReport(
                    file=file,
                    primary_extractor=resolved.primary_result.extractor,
                    primary_success=resolved.primary_result.success,
                    primary_error=resolved.primary_result.error,
                    source_profile=(
                        resolved.pdf_profile.kind if resolved.pdf_profile else None
                    ),
                    source_effective_length=resolved.primary_effective_length,
                    selected_source=resolved.selected_source,
                    final_effective_length=resolved.final_effective_length,
                    excerpt_was_truncated=len(resolved.final_content) > truncate_limit,
                    vlm_api_page_calls=resolved.vlm_api_page_calls,
                    sample_page_limit=resolved.sample_page_limit,
                    file_hash=file_hash,
                )
            )

        duplicate_groups = {
            file_hash: file_list
            for file_hash, file_list in hash_to_files.items()
            if file_hash and len(file_list) > 1
        }
        summary.duplicate_groups = duplicate_groups
        summary.duplicate_count = sum(
            len(file_list) - 1 for file_list in duplicate_groups.values()
        )

        self.logger.debug(
            t(
                "debug_extract_summary",
                doc_cache_hits=summary.doc_cache_hits,
                vlm_cache_hits=summary.vlm_cache_hits,
                primary_failures=summary.primary_failures,
                source_fallback_needed=summary.source_fallback_needed,
                selected_vlm_files=summary.selected_vlm_files,
                vlm_api_page_calls=summary.vlm_api_page_calls,
                duplicates=summary.duplicate_count,
            )
        )

        return contents, file_hashes, summary, file_reports

    @staticmethod
    def _build_cached_primary_result(
        _file_path: Path,
        content: str,
        extractor_name: str,
    ) -> ExtractionResult:
        return ExtractionResult(
            content=content,
            success=True,
            extractor=extractor_name,
        )

    def _vectorize(
        self,
        files: list[Path],
        file_hashes: list[str],
        contents: list[str],
        options: PipelineOptions,
    ) -> np.ndarray:
        if options.use_cache and options.use_embedding_cache and self.cache:
            embeddings_list: list[tuple[int, np.ndarray]] = []
            need_embedding_indices: list[int] = []
            need_embedding_contents: list[str] = []
            embedding_model = self.config.models.embedding
            cache_model_version = get_embedding_cache_version(embedding_model)

            for i, (file, content, file_hash) in enumerate(
                zip(files, contents, file_hashes, strict=False)
            ):
                cached_embedding = self.cache.get_embedding(
                    file,
                    file_hash,
                    required_model_version=cache_model_version,
                )
                if cached_embedding is not None:
                    embeddings_list.append((i, cached_embedding))
                    continue
                need_embedding_indices.append(i)
                need_embedding_contents.append(content)

            self.logger.debug(
                t(
                    "debug_vector_cache_summary",
                    hits=len(embeddings_list),
                    misses=len(need_embedding_indices),
                )
            )

            if need_embedding_contents:
                file_names = [files[i].name for i in need_embedding_indices]
                new_embeddings = get_embeddings(
                    self.client,
                    need_embedding_contents,
                    config=self.config,
                    file_names=file_names,
                    embedding_model=embedding_model,
                )

                for idx, embedding in zip(
                    need_embedding_indices, new_embeddings, strict=False
                ):
                    embeddings_list.append((idx, embedding))
                    file = files[idx]
                    file_hash = file_hashes[idx]
                    self.cache.update_embedding(
                        file_path=file,
                        file_hash=file_hash,
                        embedding=embedding,
                        model_version=cache_model_version,
                    )

            embeddings_list.sort(key=lambda item: item[0])
            return np.array([embedding for _, embedding in embeddings_list])

        file_names = [file.name for file in files]
        return get_embeddings(
            self.client,
            contents,
            config=self.config,
            file_names=file_names,
            embedding_model=self.config.models.embedding,
        )
