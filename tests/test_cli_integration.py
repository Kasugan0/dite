import json
from pathlib import Path

import httpx
import numpy as np
from openai import APIStatusError
from typer.testing import CliRunner

from dite.cache import FileCache
from dite.cli import app
from dite.core.pipeline import PipelineResult
from dite.utils.llm import format_api_error


def _write_test_config(tmp_path: Path, monkeypatch) -> Path:
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
    return config_path


def test_scan_cli_end_to_end_with_cache(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha content", encoding="utf-8")
    (docs / "b.txt").write_text("beta content", encoding="utf-8")

    report_path = tmp_path / "report.json"
    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_get_embeddings(
        client, texts, file_names, embedding_model=None
    ) -> np.ndarray:
        return np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

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
        config=None,
    ):
        return labels, {0: "Cluster_A"}, 0

    monkeypatch.setattr("dite.core.pipeline.get_embeddings", fake_get_embeddings)
    monkeypatch.setattr("dite.core.pipeline.cluster_documents", fake_cluster_documents)
    monkeypatch.setattr(
        "dite.core.pipeline.generate_all_cluster_names", fake_generate_all_cluster_names
    )

    first = runner.invoke(
        app,
        [
            "scan",
            str(docs),
            "--output",
            str(report_path),
        ],
    )
    second = runner.invoke(
        app,
        [
            "scan",
            str(docs),
            "--output",
            str(report_path),
        ],
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert report_path.exists()

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


def test_scan_cli_handles_api_connection_error(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha content", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        import httpx
        from openai import APIConnectionError

        raise APIConnectionError(
            message="Connection error.",
            request=httpx.Request("POST", "https://api.example.com/v1/embeddings"),
        )

    monkeypatch.setattr("dite.core.pipeline.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        [
            "scan",
            str(docs),
        ],
    )

    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "API connection failed" in result.output
    assert "Traceback" not in result.output


def test_format_api_error_extracts_provider_fields() -> None:
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(
        500,
        request=request,
        headers={"x-request-id": "rid-123"},
    )
    exc = APIStatusError(
        "provider boom",
        response=response,
        body={
            "code": 50507,
            "message": "Request processing failed due to an unknown error.",
            "data": None,
        },
    )

    formatted = format_api_error(exc)

    assert (
        formatted
        == "Request processing failed due to an unknown error. "
        "(status=500, code=50507, request_id=rid-123)"
    )


def test_organize_cli_execute_avoids_overwrite_and_moves_files(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    first = docs / "a" / "report.txt"
    second = docs / "b" / "report.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    target = tmp_path / "organized"
    existing_cluster = target / "finance-reports-"
    existing_cluster.mkdir(parents=True)
    (existing_cluster / "report.txt").write_text("existing", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        return PipelineResult(
            files=[first, second],
            contents=["first", "second"],
            embeddings=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            labels=np.array([0, 0], dtype=int),
            cluster_names={0: "finance/reports:*"},
        )

    monkeypatch.setattr("dite.core.pipeline.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        [
            "organize",
            str(docs),
            "--target",
            str(target),
            "--execute",
        ],
        input="y\n",
    )

    assert result.exit_code == 0
    assert (existing_cluster / "report.txt").read_text(encoding="utf-8") == "existing"
    assert (existing_cluster / "report_1.txt").read_text(encoding="utf-8") == "first"
    assert (existing_cluster / "report_2.txt").read_text(encoding="utf-8") == "second"
    assert not first.exists()
    assert not second.exists()


def test_organize_cli_passes_target_to_pipeline_exclude_paths(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}

    def fake_run(self, folder: Path, options):
        captured["exclude_paths"] = options.exclude_paths
        captured["merge_same_name"] = options.merge_same_name
        return PipelineResult(
            files=[source],
            contents=["alpha"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([0], dtype=int),
            cluster_names={0: "Cluster_A"},
        )

    monkeypatch.setattr("dite.core.pipeline.PipelineService.run", fake_run)

    result = runner.invoke(app, ["organize", str(docs), "--dry-run"])

    assert result.exit_code == 0
    assert captured["exclude_paths"] == [(docs / "organized").resolve()]
    assert captured["merge_same_name"] is False


def test_scan_cli_disables_same_name_merge_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "a.txt"
    source.write_text("alpha", encoding="utf-8")

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()
    captured: dict[str, bool] = {}

    def fake_run(self, folder: Path, options):
        captured["merge_same_name"] = options.merge_same_name
        return PipelineResult(
            files=[source],
            contents=["alpha"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([0], dtype=int),
            cluster_names={0: "Cluster_A"},
        )

    monkeypatch.setattr("dite.core.pipeline.PipelineService.run", fake_run)

    result = runner.invoke(app, ["scan", str(docs)])

    assert result.exit_code == 0
    assert captured["merge_same_name"] is False


def test_organize_cli_output_script_generates_shell_script(
    tmp_path: Path, monkeypatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    source = docs / "paper.txt"
    source.write_text("paper", encoding="utf-8")
    script_path = tmp_path / "organize.sh"

    _write_test_config(tmp_path, monkeypatch)
    runner = CliRunner()

    def fake_run(self, folder: Path, options):
        return PipelineResult(
            files=[source],
            contents=["paper"],
            embeddings=np.array([[0.1, 0.2]], dtype=np.float32),
            labels=np.array([0]),
            cluster_names={0: "Research Notes"},
            noise_repaired=0,
            clusters_merged=0,
        )

    monkeypatch.setattr("dite.core.pipeline.PipelineService.run", fake_run)

    result = runner.invoke(
        app,
        ["organize", str(docs), "--output-script", str(script_path)],
    )

    assert result.exit_code == 0
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    assert 'mkdir -p "' in script
    assert 'cp -p "' in script
    assert str(source) in script
    assert source.exists()


def test_cache_clear_vlm_cli_clears_only_vlm_entries(
    tmp_path: Path, monkeypatch
) -> None:
    _write_test_config(tmp_path, monkeypatch)

    file_a = tmp_path / "a.pdf"
    file_b = tmp_path / "b.pdf"

    cache = FileCache(cache_dir=tmp_path / "cache")
    cache.save(
        file_path=file_a,
        file_hash="hash-a",
        file_mtime=1.0,
        content_md="doc-a",
        vlm_content="vlm-a",
        vlm_version=2,
    )
    cache.save(
        file_path=file_b,
        file_hash="hash-b",
        file_mtime=1.0,
        content_md="doc-b",
    )
    cache.close()

    runner = CliRunner()
    result = runner.invoke(app, ["cache", "clear-vlm"])

    assert result.exit_code == 0

    cache = FileCache(cache_dir=tmp_path / "cache")
    assert cache.get_vlm_content(file_a, "hash-a", required_version=2) is None

    entry_a = cache.get_by_path(file_a)
    entry_b = cache.get_by_path(file_b)
    assert entry_a is not None
    assert entry_b is not None
    assert entry_a.content_md == "doc-a"
    assert entry_b.content_md == "doc-b"
    cache.close()
