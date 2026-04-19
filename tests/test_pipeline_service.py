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
from dite.core.pipeline import PipelineOptions, PipelineService
from dite.core.scanner import scan_files
from dite.extractors.base import ExtractionResult
from dite.extractors.router import PDF_VLM_SAMPLE_PAGE_LIMIT, VLMSamplingResult
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

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
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
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        call_count["emb"] += 1
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
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

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
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

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
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

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
        return ExtractionResult(content="", success=False, extractor="docling")

    def fake_extract_with_vlm_sampling(
        file_path: Path, client, config=None, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        raise AssertionError("VLM API should not run when disabled")

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
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

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
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
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        call_count["emb"] += 1
        captured["texts"] = texts
        captured["file_names"] = file_names
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
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

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
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
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        captured["texts"] = texts
        return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
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
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        return np.array([[0.1, 0.2], [0.1, 0.2]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        call_count["emb"] += 1
        assert embedding_model == "embed-v2"
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        captured["texts"] = texts
        captured["file_names"] = file_names
        captured["embedding_model"] = embedding_model
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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

    def fake_extract_document(
        file_path: Path, client, registry=None, config=None
    ) -> ExtractionResult:
        return ExtractionResult(content=long_content, success=True, extractor="docling")

    def fake_get_embeddings(
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        captured["texts"] = texts
        captured["file_names"] = file_names
        return np.array([[0.9, 0.8]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
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

    monkeypatch.setattr("dite.extractors.router.extract_document", fake_extract_document)
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

    files = scan_files(root, extensions={".txt"}, exclude_paths=[organized])

    assert [p.name for p in files] == ["a.txt"]


def test_scan_files_real_docs_fixture_uses_supported_extensions() -> None:
    fixture_dir = Path(__file__).resolve().parents[1] / "docs" / "test"
    assert fixture_dir.exists()

    extensions = Config().formats.all_extensions
    files = scan_files(fixture_dir, extensions=extensions)

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

    files = scan_files(root, extensions={".txt"})

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


def test_extract_with_vlm_fallback_loads_config_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    from dite.config import Config
    from dite.extractors import router

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_text("fake", encoding="utf-8")
    loaded_cfg = Config()
    calls = {"load": 0}

    def fake_load_config():
        calls["load"] += 1
        return loaded_cfg

    def fake_extract(
        file_path, client, config, max_pages=PDF_VLM_SAMPLE_PAGE_LIMIT
    ) -> VLMSamplingResult:
        return VLMSamplingResult(
            result=ExtractionResult(
                content="loaded" if config is loaded_cfg else "wrong",
                success=True,
                extractor="vlm_fallback",
            ),
            api_page_calls=1,
            sample_page_limit=max_pages,
        )

    monkeypatch.setattr(router, "load_config", fake_load_config)
    monkeypatch.setattr(router, "_extract_pdf_with_vlm_sampling", fake_extract)

    result = router.extract_with_vlm_fallback(pdf_path, client=object(), config=None)

    assert calls["load"] == 1
    assert result.content == "loaded"
