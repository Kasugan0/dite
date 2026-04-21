import concurrent.futures
import shutil
import threading
import time
from pathlib import Path

import httpx
import numpy as np
from openai import APIStatusError

from dite.cache import FileCache
from dite.config import (
    ChatCompletionProfileConfig,
    Config,
    RequestProfilesConfig,
)
from dite.core.clusterer import repair_noise_with_knn
from dite.core.embedder import get_embedding_cache_version
from dite.core.pipeline import (
    ExtractionFileReport,
    ExtractionSummaryDelta,
    ExtractionWorkResult,
    PipelineOptions,
    PipelineService,
)
from dite.core.scanner import scan_files
from dite.extractors.base import ExtractionResult
from dite.extractors.docling import DoclingExtractor
from dite.extractors.router import (
    PDF_VLM_SAMPLE_PAGE_LIMIT,
    PDFProfile,
    ResolvedExtraction,
    VLMSamplingResult,
)
from dite.i18n import set_locale
from dite.utils.hashing import compute_file_hash
from dite.utils.logging import setup_logging


def test_pipeline_reuses_vlm_cache_across_runs(tmp_path: Path, monkeypatch) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)

    call_count = {"doc": 0, "vlm": 0, "emb": 0}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, extractor, semaphore
        call_count["doc"] += 1
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_needs_vlm_fallback(
        content: str,
        file_path: Path,
        vlm_fallback_threshold=None,
        config=None,
    ) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        call_count["vlm"] += 1
        return VLMSamplingResult(
            result=ExtractionResult(
                content="this is vlm content",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=3,
            sample_page_limit=max_pages,
        )

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        call_count["emb"] += 1
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.extractors.router.needs_vlm_fallback", fake_needs_vlm_fallback
    )
    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    options = PipelineOptions(
        use_cache=True,
        use_embedding_cache=True,
        repair_noise=True,
        merge_same_name=True,
    )

    first = service.run(tmp_path, options)
    second = service.run(tmp_path, options)

    assert first.extraction.selected_vlm_files == 1
    assert first.extraction.vlm_cache_hits == 0
    assert first.extraction.vlm_api_page_calls == 3
    assert second.extraction.selected_vlm_files == 1
    assert second.extraction.vlm_cache_hits == 1
    assert call_count == {"doc": 2, "vlm": 1, "emb": 1}

    cache.close()


def test_extract_files_stops_before_embedding_and_clustering(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(
            content="usable extracted content" * 20,
            success=True,
            extractor="docling",
        )

    def fail_get_embeddings(*args, **kwargs) -> np.ndarray:
        raise AssertionError("embedding should not run in PDF extraction check")

    def fail_cluster_documents(*args, **kwargs):
        raise AssertionError("clustering should not run in PDF extraction check")

    def fail_generate_all_cluster_names(*args, **kwargs):
        raise AssertionError("naming should not run in PDF extraction check")

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fail_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fail_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names",
        fail_generate_all_cluster_names,
    )

    result = service.extract_files(
        [doc],
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert result.files == [doc]
    assert result.contents == ["usable extracted content" * 20]
    assert result.embeddings.size == 0
    assert result.labels.size == 0
    assert result.cluster_names == {}


def test_extract_files_can_disable_vlm_api(tmp_path: Path, monkeypatch) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        raise AssertionError("VLM API should not run when disabled")

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.extractors.router.needs_vlm_fallback",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )

    result = service.extract_files(
        [doc],
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            allow_vlm_api=False,
        ),
    )

    assert result.contents == [""]
    assert result.extraction.selected_vlm_files == 0


def test_pipeline_service_creates_fresh_registry_per_extraction_batch(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    created_registries: list[object] = []
    seen_registries: list[object] = []

    class _Registry:
        pass

    def fake_make_registry(self):
        registry = _Registry()
        created_registries.append(registry)
        return registry

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="prefetched", success=True, extractor="text")

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        config,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
    ) -> ResolvedExtraction:
        del (
            client,
            config,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
        )
        seen_registries.append(registry)
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=file_path.name,
                success=True,
                extractor="text",
            ),
            primary_effective_length=len(file_path.name),
            pdf_profile=None,
            fallback_needed=False,
            selected_source="primary",
            final_content=file_path.name,
            final_effective_length=len(file_path.name),
            vlm_content=None,
            vlm_source="none",
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    monkeypatch.setattr(
        PipelineService,
        "_make_extractor_registry",
        fake_make_registry,
    )
    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.core.pipeline.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    service = PipelineService(client=object(), config=Config(), cache=None)
    options = PipelineOptions(use_cache=False, use_embedding_cache=False)

    service.extract_files([first], options)
    service.extract_files([second], options)

    assert len(created_registries) == 2
    assert created_registries[0] is not created_registries[1]
    assert seen_registries == created_registries


def test_extract_files_defers_cache_size_enforcement_until_batch_end(
    tmp_path: Path, monkeypatch
) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    service = PipelineService(client=object(), config=Config(), cache=cache)
    save_flags: list[bool] = []
    enforce_calls = {"count": 0}

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
        del client, registry, config
        return ExtractionResult(
            content=f"{file_path.stem} content long enough",
            success=True,
            extractor="text",
        )

    def fake_save(
        self,
        file_path,
        file_hash,
        file_mtime,
        content_md,
        vlm_content=None,
        vlm_version=None,
        embedding=None,
        model_version="",
        enforce_size_limit=True,
    ) -> None:
        del (
            self,
            file_path,
            file_hash,
            file_mtime,
            content_md,
            vlm_content,
            vlm_version,
            embedding,
            model_version,
        )
        save_flags.append(enforce_size_limit)

    def fake_enforce(self) -> int:
        del self
        enforce_calls["count"] += 1
        return 0

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
    monkeypatch.setattr(FileCache, "save", fake_save)
    monkeypatch.setattr(FileCache, "enforce_size_limit", fake_enforce)

    result = service.extract_files(
        [first, second],
        PipelineOptions(use_cache=True, use_embedding_cache=False),
    )

    assert len(result.contents) == 2
    assert save_flags == [False, False]
    assert enforce_calls["count"] == 1


def test_extract_contents_preserves_input_order_when_workers_finish_out_of_order(
    tmp_path: Path, monkeypatch
) -> None:
    files = [tmp_path / "first.txt", tmp_path / "second.txt", tmp_path / "third.txt"]
    for file in files:
        file.write_text(file.stem, encoding="utf-8")

    service = PipelineService(client=object(), config=Config(), cache=None)
    delays = {"first.txt": 0.20, "second.txt": 0.05, "third.txt": 0.10}

    def fake_work_item(
        self,
        item,
        options,
        truncate_limit,
        docling_pdf_semaphore,
    ) -> object:
        del self, options, truncate_limit, docling_pdf_semaphore
        time.sleep(delays[item.file.name])
        return ExtractionWorkResult(
            index=item.index,
            content=f"content::{item.file.name}",
            file_hash=f"hash::{item.file.name}",
            report=ExtractionFileReport(
                file=item.file,
                primary_extractor="text",
                primary_success=True,
                primary_error=None,
                source_profile=None,
                source_effective_length=len(item.file.name),
                selected_source="primary",
                final_effective_length=len(item.file.name),
                excerpt_was_truncated=False,
                vlm_api_page_calls=0,
                sample_page_limit=None,
                file_hash=f"hash::{item.file.name}",
            ),
            summary_delta=ExtractionSummaryDelta(
                primary_failures=0,
                source_fallback_needed=0,
                selected_vlm_files=0,
                vlm_api_page_calls=0,
            ),
        )

    monkeypatch.setattr(
        PipelineService,
        "_extract_content_work_item",
        fake_work_item,
    )

    contents, file_hashes, summary, reports = service._extract_contents(
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert contents == [
        "content::first.txt",
        "content::second.txt",
        "content::third.txt",
    ]
    assert file_hashes == [
        "hash::first.txt",
        "hash::second.txt",
        "hash::third.txt",
    ]
    assert [report.file.name for report in reports] == [
        "first.txt",
        "second.txt",
        "third.txt",
    ]
    assert summary.duplicate_count == 0


def test_docling_pdf_workers_limit_concurrent_docling_subprocesses(monkeypatch) -> None:
    service = PipelineService(client=object(), config=Config(), cache=None)
    service.config.processing.docling_pdf_workers = 1
    semaphore = threading.BoundedSemaphore(service.config.processing.docling_pdf_workers)
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    class _Extractor:
        _enable_ocr = False
        _artifacts_path = None
        _pdf_timeout_sec = 0.5

    def fake_subprocess(
        file_path,
        *,
        enable_ocr,
        artifacts_path,
        timeout_sec,
        locale,
    ) -> ExtractionResult:
        del file_path, enable_ocr, artifacts_path, timeout_sec, locale
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1
        return ExtractionResult(content="ok", success=True, extractor="docling")

    monkeypatch.setattr(
        "dite.core.pipeline.extract_docling_pdf_in_subprocess",
        fake_subprocess,
    )

    files = [Path("one.pdf"), Path("two.pdf"), Path("three.pdf")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                service._extract_docling_pdf_primary_result,
                file_path,
                _Extractor(),
                semaphore,
            )
            for file_path in files
        ]
        results = [future.result() for future in futures]

    assert all(result.success for result in results)
    assert active["max"] == 1


def test_extract_primary_result_routes_pdf_docling_to_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.7")
    service = PipelineService(client=object(), config=Config(), cache=None)
    calls: dict[str, object] = {}

    extractor = DoclingExtractor(pdf_timeout_sec=1.0)

    monkeypatch.setattr(
        "dite.core.pipeline.get_extractor",
        lambda *args, **kwargs: extractor,
    )

    def fake_subprocess(self, file_path: Path, extractor, semaphore) -> ExtractionResult:
        calls["file_path"] = file_path
        calls["extractor"] = extractor
        calls["semaphore"] = semaphore
        return ExtractionResult(content="pdf-result", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_subprocess,
    )

    result = service._extract_primary_result(
        pdf_path,
        registry=object(),
        docling_pdf_semaphore=threading.BoundedSemaphore(1),
    )

    assert result.content == "pdf-result"
    assert calls["file_path"] == pdf_path
    assert calls["extractor"] is extractor


def test_extract_primary_result_routes_non_pdf_through_router(
    tmp_path: Path, monkeypatch
) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("payload", encoding="utf-8")
    service = PipelineService(client=object(), config=Config(), cache=None)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "dite.core.pipeline.get_extractor",
        lambda *args, **kwargs: object(),
    )

    def fake_extract_document(file_path: Path, client, *, config, registry=None):
        calls["file_path"] = file_path
        calls["client"] = client
        calls["config"] = config
        calls["registry"] = registry
        return ExtractionResult(content="text-result", success=True, extractor="text")

    monkeypatch.setattr(
        "dite.extractors.router.extract_document",
        fake_extract_document,
    )

    result = service._extract_primary_result(
        txt_path,
        registry="worker-registry",
        docling_pdf_semaphore=threading.BoundedSemaphore(1),
    )

    assert result.content == "text-result"
    assert calls == {
        "file_path": txt_path,
        "client": service.client,
        "config": service.config,
        "registry": "worker-registry",
    }


def test_pipeline_reuses_vlm_cache_by_hash_across_paths(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    cached_source = tmp_path / "cached-source.pdf"
    alias = docs / "alias.pdf"
    cached_source.write_text("same-pdf-bytes", encoding="utf-8")
    alias.write_text("same-pdf-bytes", encoding="utf-8")
    file_hash = compute_file_hash(alias)

    cache = FileCache(db_path=tmp_path / "cache.db")
    cache.update_vlm_content(
        file_path=cached_source,
        file_hash=file_hash,
        vlm_content="cached vlm content from another path",
        vlm_version=2,
    )
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)

    call_count = {"doc": 0, "vlm": 0, "emb": 0}
    captured: dict[str, list[str]] = {}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, extractor, semaphore
        call_count["doc"] += 1
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        call_count["vlm"] += 1
        raise AssertionError("VLM should not run when VLM hash cache exists")

    def fake_needs_vlm_fallback(
        content: str,
        file_path: Path,
        vlm_fallback_threshold=None,
        config=None,
    ) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        call_count["emb"] += 1
        captured["texts"] = texts
        captured["file_names"] = file_names
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr(
        "dite.extractors.router.needs_vlm_fallback", fake_needs_vlm_fallback
    )
    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        docs,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert result.extraction.vlm_cache_hits == 1
    assert result.extraction.selected_vlm_files == 1
    assert call_count == {"doc": 1, "vlm": 0, "emb": 1}
    assert result.contents == ["cached vlm content from another path"]
    assert captured["texts"] == ["cached vlm content from another path"]
    assert captured["file_names"] == ["alias.pdf"]
    cache.close()


def test_pipeline_prefers_vlm_when_docling_content_is_glyph_noise(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.processing.text_truncate_limit = 5000
    service = PipelineService(client=object(), config=config, cache=cache)

    captured: dict[str, list[str]] = {}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(
            content=("/G25/G26/G27/G28 " * 40) + ("A" * 147),
            success=True,
            extractor="docling",
        )

    def fake_needs_vlm_fallback(
        content: str,
        file_path: Path,
        vlm_fallback_threshold=None,
        config=None,
    ) -> bool:
        return True

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        return VLMSamplingResult(
            result=ExtractionResult(
                content="这是 VLM 提取出的正常文本。" * 20,
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=4,
            sample_page_limit=max_pages,
        )

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        captured["texts"] = texts
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.extractors.router.needs_vlm_fallback", fake_needs_vlm_fallback
    )
    monkeypatch.setattr(
        "dite.extractors.router._extract_pdf_with_vlm_sampling",
        fake_extract_with_vlm_sampling,
    )
    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert len(result.files) == 1
    assert captured["texts"][0].startswith("这是 VLM 提取出的正常文本")
    cache.close()


def test_pipeline_real_scan_extract_and_cache_hits(tmp_path: Path, monkeypatch) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("same content", encoding="utf-8")
    b.write_text("same content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    service = PipelineService(client=object(), config=config, cache=cache)

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        return np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0, 0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    options = PipelineOptions(
        use_cache=True,
        use_embedding_cache=True,
        repair_noise=True,
        merge_same_name=True,
    )

    first = service.run(tmp_path, options)
    second = service.run(tmp_path, options)

    assert len(first.files) == 2
    assert first.extraction.doc_cache_hits == 1
    assert first.extraction.duplicate_count == 1
    assert second.extraction.doc_cache_hits == 2
    assert second.extraction.duplicate_count == 1
    assert len(second.extraction.duplicate_groups) == 1
    assert first.extraction.selected_vlm_files == 0
    assert second.extraction.selected_vlm_files == 0

    cache.close()


def test_extract_files_locks_down_failure_corpus_classification_baseline(
    tmp_path: Path, monkeypatch
) -> None:
    files: list[Path] = []
    names = [*(f"failed-{index}.pdf" for index in range(1, 8)), "weak-1.pdf", "weak-2.pdf"]
    for name in names:
        path = tmp_path / name
        path.write_bytes(f"%PDF-1.7 {name}".encode())
        files.append(path)

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        is_failed = file_path.name.startswith("failed-")
        primary_content = "" if is_failed else ("/G25/G26/G27/G28 " * 20 + "题目")
        primary_success = not is_failed
        primary_error = "parser failed" if is_failed else None
        reason = "parser_timeout_or_broken" if is_failed else "glyph_noise_dominates"
        profile_kind = "parser_timeout_or_broken" if is_failed else "weak_text"
        page_calls = 10 if is_failed else 6
        final_content = f"{file_path.stem} final content " * 20

        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=primary_content,
                success=primary_success,
                extractor="docling",
                error=primary_error,
            ),
            primary_effective_length=0 if is_failed else 2,
            pdf_profile=PDFProfile(
                kind=profile_kind,
                effective_length=0 if is_failed else 2,
                glyph_noise_tokens=0 if is_failed else 20,
                glyph_noise_ratio=0.0 if is_failed else 0.9,
                needs_vlm_fallback=True,
                success=primary_success,
                reason=reason,
            ),
            fallback_needed=True,
            selected_source="vlm_api",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=final_content,
            vlm_source="api",
            vlm_api_success=True,
            vlm_api_page_calls=page_calls,
            sample_page_limit=PDF_VLM_SAMPLE_PAGE_LIMIT,
        )

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.core.pipeline.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = service.extract_files(
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert len(result.file_reports) == 9
    assert result.extraction.primary_failures == 7
    assert result.extraction.source_fallback_needed == 9
    assert result.extraction.selected_vlm_files == 9
    assert result.extraction.vlm_api_page_calls == 82
    assert result.extraction.duplicate_count == 0
    assert {report.source_profile for report in result.file_reports} == {
        "parser_timeout_or_broken",
        "weak_text",
    }
    assert all(report.selected_source == "vlm_api" for report in result.file_reports)
    assert all(
        report.sample_page_limit == PDF_VLM_SAMPLE_PAGE_LIMIT
        for report in result.file_reports
    )
    assert all(report.final_effective_length >= 100 for report in result.file_reports)


def test_extract_files_detects_real_duplicate_group_without_cache(
    tmp_path: Path, monkeypatch
) -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "docs" / "test"
    duplicate_names = [
        "2506.12116v3.pdf",
        "2506.12116v3 (1).pdf",
        "2506.12116v3 (2).pdf",
    ]
    files: list[Path] = []
    for name in duplicate_names:
        copied = tmp_path / name
        shutil.copy2(fixture_dir / name, copied)
        files.append(copied)

    service = PipelineService(client=object(), config=Config(), cache=None)

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        final_content = f"{file_path.name} usable content " * 20
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=final_content,
                success=True,
                extractor="docling",
            ),
            primary_effective_length=len(final_content),
            pdf_profile=PDFProfile(
                kind="native_text",
                effective_length=len(final_content),
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=False,
                success=True,
                reason="usable_text_layer",
            ),
            fallback_needed=False,
            selected_source="primary",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=None,
            vlm_source=None,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.core.pipeline.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = service.extract_files(
        files,
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert result.extraction.duplicate_count == 2
    assert len(result.extraction.duplicate_groups) == 1
    duplicate_group = next(iter(result.extraction.duplicate_groups.values()))
    assert {Path(path).name for path in duplicate_group} == set(duplicate_names)
    assert len({report.file_hash for report in result.file_reports}) == 1


def test_pipeline_recomputes_embedding_when_model_changes(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("same content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.models.embedding = "embed-v2"
    cache.save(
        file_path=doc,
        file_hash="hash-1",
        file_mtime=doc.stat().st_mtime,
        content_md="same content",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        model_version="embed-v1",
    )
    service = PipelineService(client=object(), config=config, cache=cache)

    call_count = {"emb": 0}

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        call_count["emb"] += 1
        assert embedding_model == "embed-v2"
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert len(result.files) == 1
    assert call_count["emb"] == 1
    np.testing.assert_allclose(result.embeddings, np.array([[0.9, 0.8]]))
    cache.close()


def test_pipeline_recomputes_embedding_when_input_version_changes(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("same content", encoding="utf-8")

    cache = FileCache(db_path=tmp_path / "cache.db")
    config = Config()
    config.models.embedding = "embed-v1"
    file_hash = compute_file_hash(doc)
    cache.save(
        file_path=doc,
        file_hash=file_hash,
        file_mtime=doc.stat().st_mtime,
        content_md="same content",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
        model_version="embed-v1",
    )
    service = PipelineService(client=object(), config=config, cache=cache)

    captured: dict[str, object] = {}

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        captured["texts"] = texts
        captured["file_names"] = file_names
        captured["embedding_model"] = embedding_model
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=True,
            use_embedding_cache=True,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    assert captured == {
        "texts": ["same content"],
        "file_names": ["sample.txt"],
        "embedding_model": "embed-v1",
    }
    np.testing.assert_allclose(result.embeddings, np.array([[0.9, 0.8]]))
    entry = cache.get_by_path(doc)
    assert entry is not None
    assert entry.model_version == get_embedding_cache_version("embed-v1")
    cache.close()


def test_pipeline_uses_smart_content_excerpt_for_embedding(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "book.pdf"
    doc.write_text("fake-pdf-content", encoding="utf-8")

    config = Config()
    config.processing.text_truncate_limit = 160
    service = PipelineService(client=object(), config=config, cache=None)
    long_content = ("COVER " * 80) + ("RUST OWNERSHIP " * 20) + ("INDEX " * 80)
    captured: dict[str, object] = {}

    def fake_extract_docling_pdf_primary_result(
        self, file_path: Path, extractor, semaphore
    ) -> ExtractionResult:
        del self, file_path, extractor, semaphore
        return ExtractionResult(content=long_content, success=True, extractor="docling")

    def fake_get_embeddings(
        client, texts, *, config=None, file_names=None, embedding_model=None
    ) -> np.ndarray:
        captured["texts"] = texts
        captured["file_names"] = file_names
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr(
        PipelineService,
        "_extract_docling_pdf_primary_result",
        fake_extract_docling_pdf_primary_result,
    )
    monkeypatch.setattr(
        "dite.extractors.router.needs_vlm_fallback", lambda *args, **kwargs: False
    )
    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    excerpt = captured["texts"][0]
    assert isinstance(excerpt, str)
    assert len(excerpt) <= config.processing.text_truncate_limit
    assert "COVER" in excerpt
    assert "RUST" in excerpt
    assert "OWNERSHIP" in excerpt
    assert "INDEX" in excerpt
    assert captured["file_names"] == ["book.pdf"]


def test_extract_files_preserves_final_length_separately_from_truncated_contents(
    tmp_path: Path, monkeypatch
) -> None:
    doc = tmp_path / "long.pdf"
    doc.write_bytes(b"%PDF-1.7 long")

    config = Config()
    config.processing.text_truncate_limit = 80
    service = PipelineService(client=object(), config=config, cache=None)
    final_content = ("COVER " * 40) + ("RUST OWNERSHIP " * 20) + ("INDEX " * 40)

    def fake_resolve_document_extraction(
        file_path: Path,
        client,
        *,
        enable_vlm_fallback=True,
        allow_vlm_api=True,
        cached_vlm_content=None,
        primary_result=None,
        registry=None,
        config=None,
    ) -> ResolvedExtraction:
        del (
            file_path,
            client,
            enable_vlm_fallback,
            allow_vlm_api,
            cached_vlm_content,
            primary_result,
            registry,
            config,
        )
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=final_content,
                success=True,
                extractor="docling",
            ),
            primary_effective_length=len(final_content),
            pdf_profile=PDFProfile(
                kind="native_text",
                effective_length=len(final_content),
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=False,
                success=True,
                reason="usable_text_layer",
            ),
            fallback_needed=False,
            selected_source="primary",
            final_content=final_content,
            final_effective_length=len(final_content),
            vlm_content=None,
            vlm_source=None,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    def fake_extract_primary_result(
        self, file_path: Path, registry, docling_pdf_semaphore
    ) -> ExtractionResult:
        del self, file_path, registry, docling_pdf_semaphore
        return ExtractionResult(content="", success=True, extractor="docling")

    monkeypatch.setattr(
        PipelineService,
        "_extract_primary_result",
        fake_extract_primary_result,
    )
    monkeypatch.setattr(
        "dite.core.pipeline.resolve_document_extraction",
        fake_resolve_document_extraction,
    )

    result = service.extract_files(
        [doc],
        PipelineOptions(use_cache=False, use_embedding_cache=False),
    )

    assert len(result.contents) == 1
    assert len(result.contents[0]) <= config.processing.text_truncate_limit
    assert result.file_reports[0].final_effective_length == len(final_content)
    assert result.file_reports[0].excerpt_was_truncated is True


def test_knn_repair_keeps_far_noise_with_threshold() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],  # near core cluster
            [-1.0, 0.0],   # far from core cluster
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, -1, -1], dtype=int)

    repaired_labels, repaired_count = repair_noise_with_knn(
        embeddings,
        labels,
        k=1,
        distance_threshold=0.2,
    )

    assert repaired_count == 1
    assert repaired_labels[3] == 0
    assert repaired_labels[4] == -1


def test_knn_repair_uses_dynamic_threshold() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.03],  # near core cluster
            [-1.0, 0.0],   # far from core cluster
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, -1, -1], dtype=int)

    repaired_labels, repaired_count = repair_noise_with_knn(
        embeddings,
        labels,
        k=1,
        distance_threshold=None,
    )

    assert repaired_count == 1
    assert repaired_labels[3] == 0
    assert repaired_labels[4] == -1


def test_scan_files_excludes_target_directory(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")

    organized = root / "organized"
    organized.mkdir()
    (organized / "b.txt").write_text("beta", encoding="utf-8")

    files = scan_files(
        root,
        config=Config(),
        extensions={".txt"},
        exclude_paths=[organized],
    )

    assert [p.name for p in files] == ["a.txt"]


def test_scan_files_real_docs_fixture_uses_supported_extensions() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "docs" / "test"
    assert fixture_dir.exists()

    extensions = Config().formats.all_extensions
    files = scan_files(fixture_dir, config=Config(), extensions=extensions)

    assert files
    assert all(path.suffix.lower() in extensions for path in files)


def test_scan_files_verbose_logs_follow_locale(
    tmp_path: Path,
    capsys,
) -> None:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.bin").write_text("beta", encoding="utf-8")

    setup_logging(verbose=True)
    set_locale("en")

    files = scan_files(root, config=Config(), extensions={".txt"})

    output = capsys.readouterr().out
    assert [path.name for path in files] == ["a.txt"]
    assert "Scan folder:" in output
    assert "Scan completed:" in output
    assert "Skipped 1 unsupported files" in output
    assert "扫描目录" not in output


def test_pipeline_verbose_logs_include_extraction_summary(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    doc = tmp_path / "sample.txt"
    doc.write_text("payload", encoding="utf-8")

    calls: list[dict] = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)

            class _Item:
                embedding = [0.1, 0.2]

            class _Usage:
                total_tokens = 12

            class _Response:
                data = [_Item()]
                usage = _Usage()

            return _Response()

    class _Client:
        embeddings = _Embeddings()

    config = Config()
    service = PipelineService(client=_Client(), config=config, cache=None)

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
        return ExtractionResult(
            content="example content that is long enough",
            success=True,
            extractor="text",
        )

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        item_names=None,
    ):
        return np.array([0]), 0

    def fake_generate_all_cluster_names(
        client,
        labels: np.ndarray,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config: Config | None = None,
    ):
        return labels, {0: "Notes"}, 0

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
    monkeypatch.setattr(
        "dite.extractors.router.needs_vlm_fallback", lambda *args, **kwargs: False
    )
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    setup_logging(verbose=True)
    set_locale("en")

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    output = capsys.readouterr().out
    assert len(result.files) == 1
    assert "Processing file:" in output
    assert "Document extraction: extractor=text, success=True" in output
    assert "Extraction summary:" in output
    assert "Vectorizing 1 documents" in output
    assert calls[0]["input"] == [
        "File name: sample.txt\n\nContent:\nexample content that is long enough"
    ]


def test_generate_all_cluster_names_debug_uses_letter_labels(capsys) -> None:
    from dite.core.clusterer import generate_all_cluster_names

    class _Completions:
        def create(self, **kwargs):
            name = "Alpha" if "a.txt" in kwargs["messages"][0]["content"] else "Beta"

            class _Message:
                def __init__(self, content: str) -> None:
                    self.content = content

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Message(content)

            class _Response:
                choices = [_Choice(name)]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    setup_logging(verbose=True)
    set_locale("en")

    labels = np.array([0, 2, 2], dtype=int)
    generate_all_cluster_names(
        client=_Client(),
        labels=labels,
        contents=["doc a", "doc b", "doc c"],
        files=[Path("a.txt"), Path("b.txt"), Path("c.txt")],
        config=Config(),
        embeddings=None,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    output = capsys.readouterr().out
    assert "Cluster A named Alpha" in output
    assert "Cluster B named Beta" in output
    assert "Cluster 0 named" not in output
    assert "Cluster 2 named" not in output


def test_merge_clusters_by_name_merges_duplicate_names() -> None:
    from dite.core.clusterer import merge_clusters_by_name

    labels = np.array([0, 1, 1, 2, -1], dtype=int)
    cluster_names = {0: "机器学习", 1: "机器学习", 2: "财务"}

    merged_labels, merged_names, merged_count = merge_clusters_by_name(
        labels, cluster_names
    )

    assert merged_count == 1
    assert np.array_equal(merged_labels, np.array([0, 0, 0, 2, -1], dtype=int))
    assert merged_names == {0: "机器学习", 2: "财务"}


def test_merge_clusters_by_name_does_not_merge_unnamed_fallback() -> None:
    from dite.core.clusterer import merge_clusters_by_name

    labels = np.array([0, 1, 2, 2], dtype=int)
    cluster_names = {0: "未命名", 1: "未命名", 2: "线性代数"}

    merged_labels, merged_names, merged_count = merge_clusters_by_name(
        labels, cluster_names
    )

    assert merged_count == 0
    assert np.array_equal(merged_labels, labels)
    assert merged_names == cluster_names


def test_generate_all_cluster_names_falls_back_to_unnamed_on_api_error() -> None:
    from dite.core.clusterer import generate_all_cluster_names

    class _FailingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("api down")

    class _FailingChat:
        completions = _FailingCompletions()

    class _FailingClient:
        chat = _FailingChat()

    labels = np.array([0, 1], dtype=int)
    contents = ["机器学习导论\nA" * 80, "财务报表分析\nB" * 80]
    files = [Path("ml.txt"), Path("finance.txt")]
    embeddings = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)

    new_labels, cluster_names, merged_count = generate_all_cluster_names(
        client=_FailingClient(),
        labels=labels,
        contents=contents,
        files=files,
        config=Config(),
        embeddings=embeddings,
        merge_same_name=False,
        llm_model="dummy-model",
    )

    assert np.array_equal(new_labels, labels)
    assert cluster_names == {0: "机器学习导论", 1: "财务报表分析"}
    assert merged_count == 0


def test_generate_cluster_name_truncates_long_contents_before_api_call() -> None:
    from dite.core.clusterer import (
        CLUSTER_NAME_CONTENT_LIMIT,
        CLUSTER_NAME_EXCERPT_LIMIT,
        generate_cluster_name,
    )

    captured: list[dict] = []

    class _Completions:
        def create(self, **kwargs):
            captured.append(kwargs)

            class _Message:
                content = "数学教材"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()
        base_url = "https://api.siliconflow.cn/v1"

    cfg = Config(
        request_profiles=RequestProfilesConfig(
            cluster_naming=ChatCompletionProfileConfig(
                max_tokens=64,
                reasoning_mode="off",
            )
        )
    )
    set_locale("en")
    long_text = "A" * (CLUSTER_NAME_CONTENT_LIMIT + 500)
    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=[long_text],
        sample_names=["long.txt"],
        llm_model="Qwen/Qwen3.5-27B",
        config=cfg,
    )

    assert name == "数学教材"
    prompt = captured[0]["messages"][0]["content"]
    assert "File name: long.txt" in prompt
    assert "Title candidate: long" in prompt
    assert "Name this category in 2-4 English words." in prompt
    assert "A" * (CLUSTER_NAME_EXCERPT_LIMIT + 100) not in prompt
    assert captured[0]["max_tokens"] == 64
    assert captured[0]["extra_body"] == {"enable_thinking": False}


def test_generate_cluster_name_uses_heuristic_when_response_is_empty() -> None:
    from dite.core.clusterer import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            class _Message:
                content = ""

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["线性代数导论\n这是一本教材的内容摘要。"],
        sample_names=["linear-algebra.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "线性代数导论"


def test_generate_cluster_name_skips_placeholder_and_author_lines() -> None:
    from dite.core.clusterer import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            class _Message:
                content = ""

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=[
            "<!-- image -->\nShuo Wang * Chunlong Xia\nDocument Intelligence\n摘要",
        ],
        sample_names=["document-intelligence.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "Document Intelligence"


def test_generate_cluster_name_uses_heuristic_for_invalid_model_output() -> None:
    from dite.core.clusterer import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            class _Message:
                content = "<!-- image -->"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["操作系统原理\n这是一本教材的内容摘要。"],
        sample_names=["os.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "操作系统原理"


def test_generate_cluster_name_retries_provider_server_error_then_succeeds() -> None:
    from dite.core.clusterer import generate_cluster_name

    calls = {"count": 0}

    class _Completions:
        def create(self, **kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                request = httpx.Request(
                    "POST",
                    "https://api.example.com/v1/chat/completions",
                )
                response = httpx.Response(
                    500,
                    request=request,
                    headers={"x-request-id": "rid-retry"},
                )
                raise APIStatusError(
                    "provider boom",
                    response=response,
                    body={
                        "code": 50507,
                        "message": "Request processing failed due to an unknown error.",
                        "data": None,
                    },
                )

            class _Message:
                content = "线性代数教材"

            class _Choice:
                message = _Message()

            class _Response:
                choices = [_Choice()]

            return _Response()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["线性代数导论\n教材内容摘要"],
        sample_names=["linear-algebra.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    assert name == "线性代数教材"
    assert calls["count"] == 3


def test_generate_cluster_name_formats_final_api_error_in_debug(capsys) -> None:
    from dite.core.clusterer import generate_cluster_name

    class _Completions:
        def create(self, **kwargs):
            request = httpx.Request(
                "POST",
                "https://api.example.com/v1/chat/completions",
            )
            response = httpx.Response(
                500,
                request=request,
                headers={"x-request-id": "rid-final"},
            )
            raise APIStatusError(
                "provider boom",
                response=response,
                body={
                    "code": 50507,
                    "message": "Request processing failed due to an unknown error.",
                    "data": None,
                },
            )

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    setup_logging(verbose=True)
    set_locale("en")

    name = generate_cluster_name(
        client=_Client(),
        cluster_embeddings=None,
        sample_contents=["操作系统原理\n这是一本教材的内容摘要。"],
        sample_names=["os.pdf"],
        config=Config(),
        llm_model="dummy-model",
    )

    output = capsys.readouterr().out
    assert name == "操作系统原理"
    assert "Request processing failed" in output
    assert "unknown error." in output
    assert "status=500" in output
    assert "code=50507" in output
    assert "{'code': 50507" not in output


def test_extract_with_vlm_fallback_uses_passed_config(
    tmp_path: Path, monkeypatch
) -> None:
    from dite.config import Config
    from dite.extractors import router

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_text("fake", encoding="utf-8")
    cfg = Config()
    expected = ExtractionResult(
        content="vlm content",
        success=True,
        extractor="vlm_fallback",
    )
    captured: dict[str, object] = {}

    def fake_extract(
        file_path, client, config, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        captured["file_path"] = file_path
        captured["client"] = client
        captured["config"] = config
        captured["max_pages"] = max_pages
        return VLMSamplingResult(
            result=expected,
            api_page_calls=2,
            sample_page_limit=max_pages,
        )

    monkeypatch.setattr(router, "_extract_pdf_with_vlm_sampling", fake_extract)

    client = object()
    result = router.extract_with_vlm_fallback(pdf_path, client=client, config=cfg)

    assert result == expected
    assert captured["file_path"] == pdf_path
    assert captured["client"] is client
    assert captured["config"] is cfg
    assert captured["max_pages"] == PDF_VLM_SAMPLE_PAGE_LIMIT


def test_extract_with_vlm_fallback_requires_explicit_config(tmp_path: Path) -> None:
    from dite.extractors import router

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_text("fake", encoding="utf-8")

    try:
        router.extract_with_vlm_fallback(pdf_path, client=object())  # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("extract_with_vlm_fallback should require config")
