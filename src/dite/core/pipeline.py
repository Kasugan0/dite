"""Shared document processing pipeline for scan and organize commands."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI

from dite.cache import VLM_CACHE_VERSION, FileCache
from dite.config import Config, load_config
from dite.extractors import (
    ExtractorRegistry,
    extract_document,
    extract_with_vlm_fallback,
    needs_vlm_fallback,
)
from dite.extractors.router import (
    _compute_effective_content_length,
    _content_quality_score,
    _should_prefer_vlm_content,
    classify_pdf_profile,
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
    docling_cache_hits: int = 0
    vlm_cache_hits: int = 0
    vlm_fallback_count: int = 0
    duplicate_count: int = 0
    duplicate_groups: dict[str, list[str]] = field(default_factory=dict)


class PipelineService:
    """Unified scan/extract/embed/cluster/name pipeline."""

    def __init__(
        self,
        client: OpenAI,
        config: Config | None = None,
        cache: FileCache | None = None,
        extractor_registry: ExtractorRegistry | None = None,
    ) -> None:
        self.client = client
        self.config = config or load_config()
        self.cache = cache
        self.extractor_registry = extractor_registry or ExtractorRegistry(
            config=self.config,
            client=client,
        )
        self.logger = get_logger()

    def run(self, folder: Path, options: PipelineOptions) -> PipelineResult:
        files = scan_files(
            folder,
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
            )

        contents, file_hashes, metrics = self._extract_contents(files, options)
        embeddings = self._vectorize(files, file_hashes, contents, options)
        labels, noise_repaired = cluster_documents(
            embeddings,
            repair_noise=options.repair_noise,
            clustering=self.config.clustering,
            item_names=[file.name for file in files],
        )
        labels, cluster_names, clusters_merged = generate_all_cluster_names(
            self.client,
            labels,
            contents,
            files,
            embeddings,
            merge_same_name=options.merge_same_name,
            llm_model=self.config.models.llm,
            config=self.config,
        )

        return PipelineResult(
            files=files,
            contents=contents,
            embeddings=embeddings,
            labels=labels,
            cluster_names=cluster_names,
            noise_repaired=noise_repaired,
            clusters_merged=clusters_merged,
            **metrics,
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
            )

        contents, _file_hashes, metrics = self._extract_contents(files, options)
        return PipelineResult(
            files=files,
            contents=contents,
            embeddings=np.array([]),
            labels=np.array([]),
            cluster_names={},
            **metrics,
        )

    def _extract_contents(
        self,
        files: list[Path],
        options: PipelineOptions,
    ) -> tuple[list[str], list[str], dict[str, int | dict[str, list[str]]]]:
        contents: list[str] = []
        file_hashes: list[str] = []
        docling_cache_hits = 0
        vlm_cache_hits = 0
        vlm_fallback_count = 0
        duplicate_count = 0
        failed_count = 0
        duplicate_groups: dict[str, list[str]] = {}
        hash_to_files: dict[str, list[str]] = {}
        vlm_threshold = self.config.processing.vlm_fallback_threshold
        truncate_limit = self.config.processing.text_truncate_limit

        for file in files:
            self.logger.debug(t("debug_extract_processing_file", path=file))
            file_hash = compute_file_hash(file) if options.use_cache and self.cache else ""
            file_hashes.append(file_hash)
            if file_hash:
                self.logger.debug(t("debug_extract_hash", hash=file_hash[:12]))
                hash_to_files.setdefault(file_hash, []).append(str(file))

            docling_content = None
            cached_vlm_content = None
            source_file = None
            doc_extraction_success = False
            doc_extraction_error = None
            if options.use_cache and self.cache:
                docling_content, source_file = self.cache.get_content(file, file_hash)
                if docling_content:
                    doc_extraction_success = True
                    docling_cache_hits += 1
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

            if (
                docling_content is None
                and file.suffix.lower() == ".pdf"
                and options.use_cache
                and self.cache
            ):
                cached_vlm_content = self.cache.get_vlm_content(
                    file, file_hash, VLM_CACHE_VERSION
                )
                if cached_vlm_content:
                    vlm_cache_hits += 1
                    docling_content = ""
                    self.logger.debug(
                        t(
                            "debug_extract_vlm_cache_hit",
                            length=len(cached_vlm_content),
                        )
                    )

            if docling_content is None:
                result = extract_document(
                    file,
                    self.client,
                    registry=self.extractor_registry,
                    config=self.config,
                )
                docling_content = result.content if result.success else ""
                doc_extraction_success = result.success
                doc_extraction_error = result.error
                if not result.success:
                    failed_count += 1
                self.logger.debug(
                    t(
                        "debug_extract_doc_result",
                        extractor=result.extractor,
                        success=result.success,
                        length=len(docling_content),
                        error=result.error or "-",
                    )
                )
                if options.use_cache and self.cache and result.success:
                    self.cache.save(
                        file_path=file,
                        file_hash=file_hash,
                        file_mtime=file.stat().st_mtime,
                        content_md=docling_content,
                        model_version=self.config.models.embedding,
                    )

            final_content = docling_content
            effective_length = _compute_effective_content_length(docling_content)
            needs_fallback = needs_vlm_fallback(
                docling_content,
                file,
                vlm_fallback_threshold=vlm_threshold,
            )
            if file.suffix.lower() == ".pdf":
                self.logger.debug(
                    t(
                        "debug_extract_vlm_check",
                        suffix=file.suffix.lower(),
                        effective_length=effective_length,
                        threshold=vlm_threshold,
                        needed=needs_fallback,
                    )
                )
                pdf_profile = classify_pdf_profile(
                    docling_content,
                    file,
                    success=doc_extraction_success,
                    error=doc_extraction_error,
                    vlm_fallback_threshold=vlm_threshold,
                    config=self.config,
                )
                if pdf_profile is not None:
                    self.logger.debug(
                        t(
                            "debug_pdf_profile",
                            kind=pdf_profile.kind,
                            reason=pdf_profile.reason,
                            effective_length=pdf_profile.effective_length,
                            glyph_noise_tokens=pdf_profile.glyph_noise_tokens,
                            needs_vlm_fallback=pdf_profile.needs_vlm_fallback,
                        )
                    )

            if needs_fallback:
                vlm_content = cached_vlm_content
                if options.use_cache and self.cache:
                    if vlm_content is None:
                        vlm_content = self.cache.get_vlm_content(
                            file, file_hash, VLM_CACHE_VERSION
                        )
                    if vlm_content and cached_vlm_content is None:
                        vlm_cache_hits += 1
                        self.logger.debug(
                            t(
                                "debug_extract_vlm_cache_hit",
                                length=len(vlm_content),
                            )
                        )

                if vlm_content is None and options.allow_vlm_api:
                    self.logger.debug(t("debug_extract_vlm_api_call"))
                    vlm_result = extract_with_vlm_fallback(
                        file,
                        self.client,
                        config=self.config,
                    )
                    self.logger.debug(
                        t(
                            "debug_extract_vlm_result",
                            success=vlm_result.success,
                            length=len(vlm_result.content),
                            error=vlm_result.error or "-",
                        )
                    )
                    if vlm_result.success:
                        vlm_content = vlm_result.content
                        vlm_fallback_count += 1
                        if options.use_cache and self.cache:
                            self.cache.update_vlm_content(
                                file,
                                file_hash,
                                vlm_content,
                                VLM_CACHE_VERSION,
                            )

                if vlm_content and _should_prefer_vlm_content(
                    docling_content, vlm_content
                ):
                    final_content = vlm_content
                    self.logger.debug(
                        t(
                            "debug_extract_vlm_selected",
                            vlm_length=_content_quality_score(vlm_content),
                            doc_length=effective_length,
                        )
                    )
                elif vlm_content:
                    self.logger.debug(
                        t(
                            "debug_extract_vlm_skipped",
                            doc_length=effective_length,
                            vlm_length=_content_quality_score(vlm_content),
                        )
                    )

            if len(final_content) > truncate_limit:
                self.logger.debug(
                    t(
                        "debug_extract_truncated",
                        original=len(final_content),
                        limit=truncate_limit,
                    )
                )
            contents.append(
                ContentTruncator.truncate_smart(final_content, max_chars=truncate_limit)
            )

        duplicate_groups = {
            file_hash: file_list
            for file_hash, file_list in hash_to_files.items()
            if len(file_list) > 1
        }
        duplicate_count = sum(len(file_list) - 1 for file_list in duplicate_groups.values())

        self.logger.debug(
            t(
                "debug_extract_summary",
                doc_cache_hits=docling_cache_hits,
                vlm_cache_hits=vlm_cache_hits,
                vlm_fallback_calls=vlm_fallback_count,
                duplicates=duplicate_count,
                failed=failed_count,
            )
        )

        return contents, file_hashes, {
            "docling_cache_hits": docling_cache_hits,
            "vlm_cache_hits": vlm_cache_hits,
            "vlm_fallback_count": vlm_fallback_count,
            "duplicate_count": duplicate_count,
            "duplicate_groups": duplicate_groups,
        }

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
                    file_names,
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
            file_names,
            embedding_model=self.config.models.embedding,
        )
