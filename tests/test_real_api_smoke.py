import json
import os
from pathlib import Path

import numpy as np
import pytest
from openai import NotFoundError, OpenAI
from sklearn.metrics.pairwise import cosine_distances
from typer.testing import CliRunner

from dite.cli import app
from dite.config import load_config
from dite.core.clusterer import generate_cluster_name
from dite.core.embedder import get_embeddings
from dite.core.pipeline import PipelineOptions, PipelineService


def _load_dotenv(path: Path) -> dict[str, str]:
    """Load simple KEY=VALUE pairs from a .env file."""
    if not path.exists():
        return {}

    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip("\"'")
    return data


def _real_api_config_values() -> tuple[str, str, str, str]:
    """Return real API configuration values, or skip if not configured."""
    if os.getenv("DITE_REAL_API") != "1":
        pytest.skip("Set DITE_REAL_API=1 to run real API tests.")

    config = load_config()
    dotenv = _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    api_key = os.getenv("DITE_API_KEY") or config.api.api_key or dotenv.get("DITE_API_KEY")
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY") or dotenv.get("OPENAI_API_KEY")

    base_url = (
        os.getenv("DITE_API_BASE_URL")
        or config.api.base_url
        or dotenv.get("DITE_API_BASE_URL")
    )
    if not base_url:
        base_url = os.getenv("OPENAI_BASE_URL") or dotenv.get("OPENAI_BASE_URL")

    if not api_key or not base_url:
        pytest.skip(
            "Missing API config. Set api.api_key/api.base_url in "
            "~/.config/dite/config.yaml, or use env vars/.env as fallback."
        )

    embed_model = (
        os.getenv("DITE_EMBED_MODEL")
        or config.models.embedding
        or dotenv.get("DITE_EMBED_MODEL")
        or "Qwen/Qwen3-Embedding-8B"
    )
    llm_model = (
        os.getenv("DITE_LLM_MODEL")
        or config.models.llm
        or dotenv.get("DITE_LLM_MODEL")
        or "Qwen/Qwen3-32B"
    )
    return api_key, base_url, embed_model, llm_model


def _real_api_settings() -> tuple[OpenAI, str, str]:
    """Return a real API client and model names, or skip if not configured."""
    api_key, base_url, embed_model, llm_model = _real_api_config_values()
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client, embed_model, llm_model


def _average_pairwise_distance(vectors: np.ndarray, indices: list[int]) -> float:
    pairs: list[float] = []
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            pairs.append(float(cosine_distances(vectors[[left]], vectors[[right]])[0][0]))
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


def _write_real_api_config(home_dir: Path) -> None:
    api_key, base_url, embed_model, llm_model = _real_api_config_values()
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
                f"  llm: {llm_model}",
                "cache:",
                f"  directory: {home_dir / '.cache' / 'dite'}",
                "i18n:",
                "  locale: en",
            ]
        ),
        encoding="utf-8",
    )


def test_real_api_embeddings_smoke() -> None:
    client, embed_model, _ = _real_api_settings()
    try:
        vectors = get_embeddings(
            client=client,
            texts=[
                "机器学习课程讲义，内容包括线性回归和梯度下降。",
                "财务报表分析，包含利润表、资产负债表和现金流量表。",
            ],
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
    texts = [
        "机器学习课程讲义，内容包括线性回归、梯度下降和交叉验证。",
        "深度学习笔记，讨论神经网络、反向传播和卷积网络。",
        "财务报表分析，包含利润表、资产负债表和现金流量表。",
        "公司估值备忘录，讨论自由现金流、折现现金流和市盈率。",
    ]
    vectors = get_embeddings(
        client=client,
        texts=texts,
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
    name = generate_cluster_name(
        client=client,
        cluster_embeddings=None,
        sample_contents=[
            "这批文档主要讨论机器学习中的监督学习、模型评估和特征工程。"
        ],
        sample_names=["ml_overview.md"],
        llm_model=llm_model,
    )

    assert isinstance(name, str)
    assert name.strip() != ""
    assert "\n" not in name


def test_real_api_cluster_names_differ_for_distinct_topics() -> None:
    client, embed_model, llm_model = _real_api_settings()
    texts = [
        "机器学习课程讲义，内容包括线性回归、梯度下降和交叉验证。",
        "深度学习笔记，讨论神经网络、反向传播和卷积网络。",
        "财务报表分析，包含利润表、资产负债表和现金流量表。",
        "公司估值备忘录，讨论自由现金流、折现现金流和市盈率。",
    ]
    vectors = get_embeddings(
        client=client,
        texts=texts,
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
        llm_model=llm_model,
    )
    finance_name = generate_cluster_name(
        client=client,
        cluster_embeddings=vectors[2:],
        sample_contents=texts[2:],
        sample_names=["finance_statement.txt", "finance_valuation.md"],
        llm_model=llm_model,
    )

    assert ml_name.strip()
    assert finance_name.strip()
    assert ml_name != finance_name
    assert ml_name not in {"未命名", "Unnamed"}
    assert finance_name not in {"未命名", "Unnamed"}


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
        file.name: int(label) for file, label in zip(result.files, result.labels, strict=True)
    }
    ml_labels = {label_by_name[path.name] for path in written["ml"]}
    finance_labels = {label_by_name[path.name] for path in written["finance"]}

    assert len(result.files) == 6
    assert result.embeddings.shape[0] == 6
    assert np.isfinite(result.embeddings).all()
    assert all(path.suffix.lower() in {".txt", ".md", ".markdown"} for path in result.files)
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

    monkeypatch.setenv("HOME", str(home_dir))
    _write_real_api_config(home_dir)
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
        entry["name"]
        for cluster in report["clusters"]
        for entry in cluster["files"]
    } | {entry["name"] for entry in report["noise"]}

    assert report["summary"]["total_files"] == 6
    assert report["summary"]["num_extraction_failed"] == 0
    assert report["summary"]["num_clusters"] >= 2
    assert "ml_classification.markdown" in reported_names
    assert "finance_accounting.markdown" in reported_names
