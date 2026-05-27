import json
import shutil
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from dite.app.cli import app
from dite.cache import FileCache
from dite.cluster.api import ClusterMetrics, ClusterResult
from dite.io.base import ExtractionResult
from dite.io.route import PDFProfile, ResolvedExtraction

TEST_CORPUS_DIR = Path(__file__).resolve().parents[1] / "docs" / "test" / "valid"


def make_cluster_result(
    labels: list[int] | np.ndarray | ClusterResult,
    *,
    cluster_names: dict[int, str] | None = None,
    repaired_mask: list[bool] | np.ndarray | None = None,
    metrics: ClusterMetrics | None = None,
) -> ClusterResult:
    if isinstance(labels, ClusterResult):
        base = labels
        return ClusterResult(
            labels=base.labels.copy(),
            cluster_names=cluster_names or base.cluster_names.copy(),
            repaired_mask=(
                np.array(repaired_mask, dtype=bool)
                if repaired_mask is not None
                else base.repaired_mask.copy()
            ),
            metrics=metrics or ClusterMetrics(),
        )
    labels_array = np.array(labels, dtype=int)
    if repaired_mask is None:
        repaired_mask_array = np.zeros(labels_array.shape, dtype=bool)
    else:
        repaired_mask_array = np.array(repaired_mask, dtype=bool)
    return ClusterResult(
        labels=labels_array,
        cluster_names=cluster_names or {},
        repaired_mask=repaired_mask_array,
        metrics=metrics or ClusterMetrics(),
    )


def _write_test_config(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                "  base_url: https://api.example.com/v1",
                "  api_key: dummy-key",
                "cache:",
                f"  directory: {tmp_path / 'cache'}",
                "i18n:",
                "  locale: en",
            ]
        ),
        encoding="utf-8",
    )


def test_scan_cli_end_to_end_with_cache_fast(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha content", encoding="utf-8")
    (docs / "b.txt").write_text("beta content", encoding="utf-8")

    report_path = tmp_path / "report.json"
    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del client, texts, config, file_names, embedding_model, input_mode
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        del (
            embeddings,
            repair_noise,
            config,
            clustering,
            allow_single_cluster,
            item_names,
        )
        return make_cluster_result([0, 0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config=None,
    ):
        del client, contents, files, embeddings, merge_same_name, llm_model, config
        return make_cluster_result(result, cluster_names={0: "Cluster_A"})

    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    first = runner.invoke(app, ["scan", str(docs), "--output", str(report_path)])
    second = runner.invoke(app, ["scan", str(docs), "--output", str(report_path)])

    assert first.exit_code == 0
    assert second.exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_files"] == 2
    assert report["summary"]["num_clusters"] == 1
    assert report["summary"]["num_noise"] == 0

    cache = FileCache(db_path=tmp_path / "cache" / "cache.db")
    stats = cache.get_stats()
    assert stats["total_entries"] == 2
    assert stats["with_embedding"] == 2
    assert stats["with_vlm"] == 0
    cache.close()


def test_scan_cli_duplicate_groups_verbose_fast(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    duplicate_names = [
        "2506.12116v3.pdf",
        "2506.12116v3 (1).pdf",
        "2506.12116v3 (2).pdf",
    ]
    for name in duplicate_names:
        shutil.copy2(TEST_CORPUS_DIR / name, docs / name)

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

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
        content = f"{file_path.name} usable text layer " * 10
        return ResolvedExtraction(
            primary_result=ExtractionResult(
                content=content,
                success=True,
                extractor="docling",
            ),
            primary_effective_length=len(content),
            pdf_profile=PDFProfile(
                kind="native_text",
                effective_length=len(content),
                glyph_noise_tokens=0,
                glyph_noise_ratio=0.0,
                needs_vlm_fallback=False,
                success=True,
                reason="usable_text_layer",
            ),
            fallback_needed=False,
            selected_source="primary",
            final_content=content,
            final_effective_length=len(content),
            vlm_content=None,
            vlm_source=None,
            vlm_api_success=False,
            vlm_api_page_calls=0,
            sample_page_limit=None,
        )

    def fake_get_embeddings(
        client,
        texts,
        *,
        config=None,
        file_names=None,
        embedding_model=None,
        input_mode=None,
    ) -> np.ndarray:
        del client, texts, file_names, embedding_model, input_mode, config
        return np.array([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]], dtype=np.float32)

    def fake_cluster_documents(
        embeddings: np.ndarray,
        repair_noise: bool,
        *,
        config=None,
        clustering=None,
        allow_single_cluster=False,
        item_names=None,
    ):
        del (
            embeddings,
            repair_noise,
            clustering,
            allow_single_cluster,
            item_names,
            config,
        )
        return make_cluster_result([0, 0, 0])

    def fake_generate_all_cluster_names(
        client,
        result,
        contents: list[str],
        files: list[Path],
        embeddings: np.ndarray | None = None,
        merge_same_name: bool = True,
        llm_model: str | None = None,
        config=None,
    ):
        del client, contents, files, embeddings, merge_same_name, llm_model, config
        return make_cluster_result(result, cluster_names={0: "Duplicate Papers"})

    monkeypatch.setattr(
        "dite.flow.api.resolve_document_extraction",
        fake_resolve_document_extraction,
    )
    monkeypatch.setattr(
        "dite.flow.api.get_extractor",
        lambda *args, **kwargs: type("_Extractor", (), {"name": "docling"})(),
    )
    monkeypatch.setattr(
        "dite.flow.api.PipelineService._extract_primary_result",
        lambda self, file_path, registry, docling_pdf_semaphore: ExtractionResult(
            content=f"{file_path.name} usable text layer " * 10,
            success=True,
            extractor="docling",
        ),
    )
    monkeypatch.setattr("dite.flow.api.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.flow.api.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.flow.api.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    result = runner.invoke(app, ["scan", str(docs), "--no-cache"])
    verbose_result = runner.invoke(app, ["scan", str(docs), "--no-cache", "--verbose"])

    assert result.exit_code == 0
    assert "Duplicates: 2" in result.output
    assert "Duplicate file groups detected:" not in result.output
    assert verbose_result.exit_code == 0
    assert "Duplicates: 2" in verbose_result.output
    assert "DEBUG: Duplicate file groups detected:" in verbose_result.output
    assert "2506.12116v3.pdf" in verbose_result.output
    assert "2506.12116v3 (1).pdf" in verbose_result.output
    assert "2506.12116v3 (2).pdf" in verbose_result.output
