import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
from openai import NotFoundError, OpenAI
from sklearn.metrics.pairwise import cosine_distances
from typer.testing import CliRunner

from dite.cli import app
from dite.config import load_config
from dite.core.clusterer import ClusterMetrics, ClusterResult, generate_cluster_name
from dite.core.embedder import get_embeddings
from dite.core.pipeline import PipelineOptions, PipelineService
from dite.extractors.docling import (
    get_docling_pdf_artifacts_path,
    has_docling_pdf_artifacts,
)

TEST_CORPUS_DIR = Path(__file__).resolve().parents[1] / "docs" / "test" / "valid"


def _real_api_config_values() -> tuple[str, str, str, str, str, str]:
    """Return real API configuration values from the global config file."""
    if os.getenv("DITE_REAL_API") != "1":
        pytest.skip("Set DITE_REAL_API=1 to run real API tests.")

    config = load_config()
    api_key = config.api.api_key
    base_url = config.api.base_url

    if not api_key or not base_url:
        pytest.skip(
            "Missing API config. Set api.api_key/api.base_url in "
            "~/.config/dite/config.yaml."
        )

    return (
        api_key,
        base_url,
        config.models.embedding,
        config.models.llm,
        config.models.vlm,
        config.i18n.locale,
    )


def _real_api_settings() -> tuple[OpenAI, str, str]:
    """Return a real API client and model names, or skip if not configured."""
    api_key, base_url, embed_model, llm_model, _vlm_model, _locale = (
        _real_api_config_values()
    )
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, embed_model, llm_model


def _require_real_pdf_corpus_enabled() -> Path:
    if not has_docling_pdf_artifacts():
        pytest.skip(
            "Docling PDF models are not ready. Run "
            "`uv run dite setup docling-pdf` first."
        )

    corpus_dir = TEST_CORPUS_DIR
    if not corpus_dir.exists():
        pytest.skip("Missing docs/test/valid corpus.")
    return corpus_dir


def _real_failure_corpus_files(corpus_dir: Path) -> list[Path]:
    names = [
        "Linux-UNIX系统编程手册.pdf",
        "Rust 程序设计.pdf",
        "The-Art-of-Linear-Algebra-zh-CN.pdf",
        "信息安全数学基础.pdf",
        "抽象代数 - Algebra.pdf",
        "线性代数 中文第5版【Gilbert Strang】.pdf",
        "线性代数及其应用（第六版） - Linear Algebra and Its Applications.pdf",
        "物理考后再练.pdf",
        "线性代数及其应用 (David C. Lay Steven R. Lay Judi J. McDonald) "
        "(Z-Library).pdf",
    ]
    return [corpus_dir / name for name in names]


def _real_pdf_check_fixture_files(corpus_dir: Path) -> list[Path]:
    names = [
        *[path.name for path in _real_failure_corpus_files(corpus_dir)],
        "2506.12116v3.pdf",
        "2506.12116v3 (1).pdf",
        "2506.12116v3 (2).pdf",
    ]
    return [corpus_dir / name for name in names]


def _average_pairwise_distance(vectors: np.ndarray, indices: list[int]) -> float:
    pairs: list[float] = []
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            pairs.append(
                float(cosine_distances(vectors[[left]], vectors[[right]])[0][0])
            )
    return float(sum(pairs) / len(pairs))


def _write_topic_docs(root: Path) -> dict[str, list[Path]]:
    groups = {
        "ml": [
            (
                "ml_regression.txt",
                "机器学习课程讲义：线性回归、梯度下降、特征工程、交叉验证、监督学习。",
            ),
            (
                "ml_deep_learning.md",
                "深度学习笔记：神经网络、反向传播、卷积网络、训练损失、验证集。",
            ),
            (
                "ml_classification.markdown",
                "数据挖掘作业：分类模型、随机森林、模型评估、召回率、准确率。",
            ),
        ],
        "finance": [
            (
                "finance_statement.txt",
                "财务报表分析：资产负债表、利润表、现金流量表、营收、毛利率。",
            ),
            (
                "finance_valuation.md",
                "公司估值备忘录：市盈率、折现现金流、自由现金流、净利润、资本开支。",
            ),
            (
                "finance_accounting.markdown",
                "会计复习资料：应收账款、存货、折旧、审计、记账凭证、成本核算。",
            ),
        ],
    }

    written: dict[str, list[Path]] = {}
    for topic, files in groups.items():
        written[topic] = []
        for file_name, content in files:
            path = root / file_name
            path.write_text(content, encoding="utf-8")
            written[topic].append(path)
    return written


def _write_real_api_config(
    home_dir: Path,
    config_values: tuple[str, str, str, str, str, str],
    *,
    docling_pdf_timeout_sec: int = 60,
) -> None:
    api_key, base_url, embed_model, llm_model, vlm_model, locale = config_values
    config_path = home_dir / ".config" / "dite" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "api:",
                f"  base_url: {base_url}",
                f"  api_key: {api_key}",
                "models:",
                f"  embedding: {embed_model}",
                f"  vlm: {vlm_model}",
                f"  llm: {llm_model}",
                "processing:",
                f"  docling_pdf_timeout_sec: {docling_pdf_timeout_sec}",
                "cache:",
                f"  directory: {home_dir / '.cache' / 'dite'}",
                "i18n:",
                f"  locale: {locale}",
            ]
        ),
        encoding="utf-8",
    )


def _link_real_docling_models_into_home(
    home_dir: Path, source_models_dir: Path
) -> None:
    target_models_dir = home_dir / ".cache" / "dite" / "docling" / "models"
    target_models_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_models_dir.exists() or target_models_dir.is_symlink():
        target_models_dir.unlink()
    target_models_dir.symlink_to(source_models_dir)


def test_real_api_embeddings_smoke() -> None:
    client, embed_model, _ = _real_api_settings()
    config = load_config()
    try:
        vectors = get_embeddings(
            client=client,
            texts=[
                "机器学习课程讲义，内容包括线性回归和梯度下降。",
                "财务报表分析，包含利润表、资产负债表和现金流量表。",
            ],
            config=config,
            file_names=["ml-notes.txt", "finance-notes.txt"],
            embedding_model=embed_model,
        )
    except NotFoundError as exc:
        pytest.fail(
            "Embedding endpoint/model not found. "
            f"Check api.base_url and models.embedding. Current embedding model: "
            f"{embed_model!r}. Original error: {exc}"
        )

    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    assert np.isfinite(vectors).all()


def test_real_api_embeddings_keep_same_topic_closer_than_cross_topic() -> None:
    client, embed_model, _ = _real_api_settings()
    config = load_config()
    texts = [
        "机器学习课程讲义，内容包括线性回归、梯度下降和交叉验证。",
        "深度学习笔记，讨论神经网络、反向传播和卷积网络。",
        "财务报表分析，包含利润表、资产负债表和现金流量表。",
        "公司估值备忘录，讨论自由现金流、折现现金流和市盈率。",
    ]
    vectors = get_embeddings(
        client=client,
        texts=texts,
        config=config,
        file_names=[
            "ml_regression.txt",
            "ml_deep_learning.md",
            "finance_statement.txt",
            "finance_valuation.md",
        ],
        embedding_model=embed_model,
    )

    ml_distance = _average_pairwise_distance(vectors, [0, 1])
    finance_distance = _average_pairwise_distance(vectors, [2, 3])
    cross_distance = float(
        np.mean(
            [
                cosine_distances(vectors[[0]], vectors[[2]])[0][0],
                cosine_distances(vectors[[0]], vectors[[3]])[0][0],
                cosine_distances(vectors[[1]], vectors[[2]])[0][0],
                cosine_distances(vectors[[1]], vectors[[3]])[0][0],
            ]
        )
    )

    assert ml_distance < cross_distance
    assert finance_distance < cross_distance


def test_real_api_cluster_name_smoke() -> None:
    client, _, llm_model = _real_api_settings()
    config = load_config()
    name = generate_cluster_name(
        client=client,
        cluster_embeddings=None,
        sample_contents=["这批文档主要讨论机器学习中的监督学习、模型评估和特征工程。"],
        sample_names=["ml_overview.md"],
        config=config,
        llm_model=llm_model,
    )

    assert isinstance(name, str)
    assert name.strip() != ""
    assert "\n" not in name


def test_real_api_cluster_names_differ_for_distinct_topics() -> None:
    client, embed_model, llm_model = _real_api_settings()
    config = load_config()
    texts = [
        "机器学习课程讲义，内容包括线性回归、梯度下降和交叉验证。",
        "深度学习笔记，讨论神经网络、反向传播和卷积网络。",
        "财务报表分析，包含利润表、资产负债表和现金流量表。",
        "公司估值备忘录，讨论自由现金流、折现现金流和市盈率。",
    ]
    vectors = get_embeddings(
        client=client,
        texts=texts,
        config=config,
        file_names=[
            "ml_regression.txt",
            "ml_deep_learning.md",
            "finance_statement.txt",
            "finance_valuation.md",
        ],
        embedding_model=embed_model,
    )

    ml_name = generate_cluster_name(
        client=client,
        cluster_embeddings=vectors[:2],
        sample_contents=texts[:2],
        sample_names=["ml_regression.txt", "ml_deep_learning.md"],
        config=config,
        llm_model=llm_model,
    )
    finance_name = generate_cluster_name(
        client=client,
        cluster_embeddings=vectors[2:],
        sample_contents=texts[2:],
        sample_names=["finance_statement.txt", "finance_valuation.md"],
        config=config,
        llm_model=llm_model,
    )

    assert ml_name.strip()
    assert finance_name.strip()
    assert ml_name != finance_name
    assert ml_name not in {"未命名", "Unnamed"}
    assert finance_name not in {"未命名", "Unnamed"}


def test_real_api_generate_all_cluster_names_smoke() -> None:
    from dite.core.clusterer import generate_all_cluster_names

    client, embed_model, llm_model = _real_api_settings()
    config = load_config()
    config.processing.cluster_naming_workers = 2

    texts = [
        "机器学习课程讲义，内容包括线性回归、梯度下降和交叉验证。",
        "深度学习笔记，讨论神经网络、反向传播和卷积网络。",
        "财务报表分析，包含利润表、资产负债表和现金流量表。",
        "公司估值备忘录，讨论自由现金流、折现现金流和市盈率。",
        "线性代数讲义，讨论矩阵、向量空间、特征值与奇异值分解。",
        "数值分析笔记，讨论迭代法、误差估计、插值与数值积分。",
    ]
    files = [
        Path("ml_regression.txt"),
        Path("ml_deep_learning.md"),
        Path("finance_statement.txt"),
        Path("finance_valuation.md"),
        Path("linear_algebra.txt"),
        Path("numerical_analysis.md"),
    ]
    labels = np.array([10, 10, 20, 20, 30, 30], dtype=int)
    embeddings = get_embeddings(
        client=client,
        texts=texts,
        config=config,
        file_names=[path.name for path in files],
        embedding_model=embed_model,
    )

    result = ClusterResult(
        labels=labels,
        cluster_names={},
        repaired_mask=np.zeros(labels.shape, dtype=bool),
        metrics=ClusterMetrics(),
    )
    named_result = generate_all_cluster_names(
        client=client,
        result=result,
        contents=texts,
        files=files,
        config=config,
        embeddings=embeddings,
        merge_same_name=False,
        llm_model=llm_model,
    )

    assert named_result.name_clusters_merged == 0
    assert np.array_equal(named_result.labels, labels)
    assert set(named_result.cluster_names) == {10, 20, 30}
    assert all(name.strip() for name in named_result.cluster_names.values())
    assert all("\n" not in name for name in named_result.cluster_names.values())
    assert all(
        name not in {"未命名", "Unnamed"}
        for name in named_result.cluster_names.values()
    )


def test_real_api_cluster_name_file_name_only_smoke() -> None:
    client, _, llm_model = _real_api_settings()
    config = load_config()
    name = generate_cluster_name(
        client=client,
        cluster_embeddings=None,
        sample_contents=["   ", "\n\n"],
        sample_names=["machine-learning-overview.pdf", "supervised-learning-notes.md"],
        config=config,
        llm_model=llm_model,
    )

    assert isinstance(name, str)
    assert name.strip() != ""
    assert "\n" not in name
    assert name not in {"未命名", "Unnamed"}


def test_real_api_pipeline_end_to_end_supports_markdown(tmp_path: Path) -> None:
    client, _, _ = _real_api_settings()
    written = _write_topic_docs(tmp_path)

    config = load_config()
    config.clustering.min_cluster_size = 2
    config.clustering.min_samples = 1
    service = PipelineService(client=client, config=config, cache=None)

    result = service.run(
        tmp_path,
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=True,
            merge_same_name=False,
        ),
    )

    label_by_name = {
        file.name: int(label)
        for file, label in zip(result.files, result.labels, strict=True)
    }
    ml_labels = {label_by_name[path.name] for path in written["ml"]}
    finance_labels = {label_by_name[path.name] for path in written["finance"]}

    assert len(result.files) == 6
    assert result.embeddings.shape[0] == 6
    assert np.isfinite(result.embeddings).all()
    assert all(
        path.suffix.lower() in {".txt", ".md", ".markdown"} for path in result.files
    )
    assert -1 not in ml_labels
    assert -1 not in finance_labels
    assert len(ml_labels) == 1
    assert len(finance_labels) == 1
    assert ml_labels != finance_labels
    assert result.cluster_names
    assert all(name.strip() for name in result.cluster_names.values())


def test_real_api_cli_scan_reports_markdown_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home_dir = tmp_path / "home"
    docs_dir = tmp_path / "docs"
    report_path = tmp_path / "report.json"
    config_values = _real_api_config_values()

    monkeypatch.setenv("HOME", str(home_dir))
    _write_real_api_config(home_dir, config_values)
    docs_dir.mkdir()
    _write_topic_docs(docs_dir)

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "scan",
            str(docs_dir),
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    reported_names = {
        entry["name"] for cluster in report["clusters"] for entry in cluster["files"]
    } | {entry["name"] for entry in report["noise"]}

    assert report["summary"]["total_files"] == 6
    assert report["summary"]["num_extraction_failed"] == 0
    assert report["summary"]["num_clusters"] >= 2
    assert "ml_classification.markdown" in reported_names
    assert "finance_accounting.markdown" in reported_names


def test_real_api_extract_files_failure_corpus_matches_current_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = _require_real_pdf_corpus_enabled()
    source_models_dir = get_docling_pdf_artifacts_path()
    config_values = _real_api_config_values()
    home_dir = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home_dir))
    _write_real_api_config(home_dir, config_values, docling_pdf_timeout_sec=15)
    _link_real_docling_models_into_home(home_dir, source_models_dir)
    client = OpenAI(api_key=config_values[0], base_url=config_values[1])
    config = load_config()
    service = PipelineService(client=client, config=config, cache=None)

    result = service.extract_files(
        _real_failure_corpus_files(corpus_dir),
        PipelineOptions(
            use_cache=False,
            use_embedding_cache=False,
            repair_noise=False,
            merge_same_name=False,
            allow_vlm_api=True,
        ),
    )

    threshold = config.processing.vlm_fallback_threshold
    weak_reports = [
        report
        for report in result.file_reports
        if report.final_effective_length < threshold
    ]
    profile_counts = {
        profile: sum(
            1 for report in result.file_reports if report.source_profile == profile
        )
        for profile in {"parser_timeout_or_broken", "weak_text"}
    }

    assert len(result.files) == 9
    assert result.extraction.primary_failures == 8
    assert result.extraction.source_fallback_needed == 9
    assert result.extraction.selected_vlm_files == 9
    assert result.extraction.vlm_api_page_calls == 82
    assert profile_counts == {
        "parser_timeout_or_broken": 8,
        "weak_text": 1,
    }
    assert not weak_reports
    assert all(report.selected_source != "primary" for report in result.file_reports)
    assert all(report.sample_page_limit == 10 for report in result.file_reports)


def test_real_api_pdf_check_fixture_summary_matches_current_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_dir = _require_real_pdf_corpus_enabled()
    source_models_dir = get_docling_pdf_artifacts_path()
    config_values = _real_api_config_values()
    home_dir = tmp_path / "home"
    fixture_dir = tmp_path / "fixture"
    fixture_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("COLUMNS", "200")
    _write_real_api_config(
        home_dir,
        (
            config_values[0],
            config_values[1],
            config_values[2],
            config_values[3],
            config_values[4],
            "en",
        ),
        docling_pdf_timeout_sec=15,
    )
    _link_real_docling_models_into_home(home_dir, source_models_dir)

    for source in _real_pdf_check_fixture_files(corpus_dir):
        target = fixture_dir / source.name
        target.symlink_to(source.resolve())

    runner = CliRunner()
    result = runner.invoke(app, ["pdf-check", str(fixture_dir), "-v", "--no-cache"])
    normalized_output = re.sub(r"\s+", " ", result.output)

    assert result.exit_code == 0
    assert "Found 12 PDF files" in normalized_output
    assert "primary failures:" in normalized_output
    assert "fallback needed:" in normalized_output
    assert "selected VLM:" in normalized_output
    assert "VLM page calls:" in normalized_output
    assert "duplicates:" in normalized_output
    assert "weak:" in normalized_output
    assert "empty:" in normalized_output
    assert "VLM samples only the first 10 pages." in normalized_output
    assert "Extraction details" in normalized_output
    assert "Primary parser failed" in normalized_output
    assert "VLM sampling" in normalized_output
    assert "PDF outputs passed the smoke check" in normalized_output
    assert re.search(r"SUCCESS: 12 .*smoke check", normalized_output)
