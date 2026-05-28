from __future__ import annotations

import hashlib
import logging
import runpy
from pathlib import Path

import numpy as np

from dite.app.cli import _build_report, _print_report
from dite.app.config import Config
from dite.app.i18n import set_locale
from dite.cluster.model import CandidateComponent, CandidateEdge
from dite.doc import DocumentFeatures, MetadataFeatures, QualityFlags
from dite.doc.analyze import (
    DocumentAnalysis,
    analyze_and_build_payload,
    analyze_document,
    build_weighted_payload,
)
from dite.util.hash import compute_file_hash
from dite.util.log import LogLevel, get_logger, setup_logging


class _AnalyzerClient:
    def __init__(self, responses: list[str], calls: list[dict]) -> None:
        self._responses = responses
        self._calls = calls

        class _Completions:
            def __init__(self, outer: _AnalyzerClient) -> None:
                self._outer = outer

            def create(self, **kwargs):
                self._outer._calls.append(kwargs)
                content = self._outer._responses.pop(0)

                class _Message:
                    def __init__(self, content: str) -> None:
                        self.content = content

                class _Choice:
                    def __init__(self, content: str) -> None:
                        self.message = _Message(content)

                class _Response:
                    def __init__(self, content: str) -> None:
                        self.choices = [_Choice(content)]

                return _Response(content)

        class _Chat:
            def __init__(self, outer: _AnalyzerClient) -> None:
                self.completions = _Completions(outer)

        self.chat = _Chat(self)


def test_document_analysis_from_dict_uses_defaults() -> None:
    analysis = DocumentAnalysis.from_dict({"content": {"topic": "机器学习"}})

    assert analysis.content.topic == "机器学习"
    assert analysis.layout.type == "其他"
    assert analysis.layout.columns == "single"
    assert analysis.content.language == "zh"
    assert analysis.confidence == 0.0


def test_build_weighted_payload_repeats_keywords_and_truncates_excerpt() -> None:
    analysis = DocumentAnalysis.from_dict(
        {
            "layout": {"type": "论文", "columns": "double", "has_table": True},
            "content": {
                "topic": "机器学习",
                "keywords": ["回归", "梯度下降"],
            },
            "summary": "概述",
        }
    )
    set_locale("zh-CN")

    payload = build_weighted_payload(
        analysis,
        raw_content="abcdefghijk",
        max_excerpt_length=5,
    )

    assert "文档类型: 论文" in payload
    assert "布局特征: double栏, 表格:True" in payload
    assert payload.count("回归 梯度下降") == 3
    assert "内容节选: abcde" in payload


def test_analyze_document_returns_default_after_invalid_json() -> None:
    calls: list[dict] = []
    client = _AnalyzerClient(["not-json", "still-not-json"], calls)

    result = analyze_document(
        client,
        "example content",
        config=Config(),
        max_retries=2,
    )

    assert result == DocumentAnalysis()
    assert len(calls) == 2


def test_analyze_document_uses_model_and_truncates_content() -> None:
    calls: list[dict] = []
    client = _AnalyzerClient(
        [
            (
                '{"layout":{"type":"报告"},"content":{"topic":"财务","keywords":["利润"]},'
                '"summary":"摘要","confidence":0.9}'
            )
        ],
        calls,
    )
    set_locale("en")

    result = analyze_document(
        client,
        "x" * 20,
        config=Config(),
        max_content_length=5,
        llm_model="custom-llm",
    )

    assert result.layout.type == "报告"
    assert result.content.topic == "财务"
    assert calls[0]["model"] == "custom-llm"
    prompt = calls[0]["messages"][0]["content"]
    assert "[Content truncated...]" in prompt
    assert "Analyze the following document" in prompt
    assert calls[0]["response_format"] == {"type": "json_object"}


def test_build_weighted_payload_follows_locale() -> None:
    analysis = DocumentAnalysis.from_dict(
        {
            "layout": {"type": "report", "columns": "double", "has_table": True},
            "content": {
                "topic": "finance",
                "keywords": ["profit", "cashflow"],
            },
            "summary": "overview",
        }
    )

    set_locale("en")
    payload = build_weighted_payload(
        analysis,
        raw_content="abcdefghijk",
        max_excerpt_length=5,
    )

    assert "Document type: report" in payload
    assert "Layout: double columns, table:True" in payload
    assert payload.count("profit cashflow") == 3
    assert "Excerpt: abcde" in payload


def test_analyze_and_build_payload_combines_steps(monkeypatch) -> None:
    expected_analysis = DocumentAnalysis.from_dict(
        {"content": {"topic": "课程", "keywords": ["线代"]}}
    )

    monkeypatch.setattr(
        "dite.doc.analyze.analyze_document",
        lambda client, content, *, config: expected_analysis,
    )
    monkeypatch.setattr(
        "dite.doc.analyze.build_weighted_payload",
        lambda analysis,
        raw_content: f"payload::{analysis.content.topic}::{raw_content}",
    )

    analysis, payload = analyze_and_build_payload(
        object(),
        "raw body",
        config=Config(),
    )

    assert analysis is expected_analysis
    assert payload == "payload::课程::raw body"


def test_compute_file_hash_matches_sha256(tmp_path: Path) -> None:
    file_path = tmp_path / "data.bin"
    payload = (b"abc123" * 1000) + b"tail"
    file_path.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()

    assert compute_file_hash(file_path, chunk_size=7) == expected
    assert compute_file_hash(file_path, chunk_size=1024) == expected


def test_setup_logging_respects_verbose_and_quiet() -> None:
    verbose_logger = setup_logging(verbose=True)
    quiet_logger = setup_logging(quiet=True)

    assert verbose_logger.min_level == LogLevel.DEBUG
    assert verbose_logger._should_log(LogLevel.DEBUG) is True
    assert quiet_logger.min_level == LogLevel.ERROR
    assert quiet_logger._should_log(LogLevel.INFO) is False
    assert get_logger() is quiet_logger


def test_setup_logging_silences_docling_error_logs(capsys) -> None:
    setup_logging()

    logging.getLogger("docling.pipeline.standard_pdf_pipeline").error("docling-noise")
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "docling-noise" not in captured.out


def test_setup_logging_silences_docling_error_logs_in_verbose_mode(capsys) -> None:
    setup_logging(verbose=True)

    logging.getLogger("docling.pipeline.standard_pdf_pipeline").error(
        "docling-verbose-noise"
    )
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "docling-verbose-noise" not in captured.out


def test_python_m_entrypoint_calls_cli_app(monkeypatch) -> None:
    calls = {"count": 0}

    def fake_app() -> None:
        calls["count"] += 1

    monkeypatch.setattr("dite.app.cli.app", fake_app)
    runpy.run_module("dite", run_name="__main__")

    assert calls["count"] == 1


def test_print_report_localizes_extraction_failed_marker(capsys) -> None:
    report = {
        "summary": {
            "total_files": 1,
            "num_clusters": 1,
            "num_noise": 0,
            "num_extraction_failed": 1,
        },
        "clusters": [
            {
                "name": "Cluster",
                "files": [
                    {
                        "name": "a.txt",
                        "extraction_failed": True,
                    }
                ],
            }
        ],
        "noise": [],
    }

    setup_logging()
    set_locale("zh-CN")
    _print_report(report)
    zh_output = capsys.readouterr().out

    set_locale("en")
    _print_report(report)
    en_output = capsys.readouterr().out

    assert "(提取失败)" in zh_output
    assert "(extraction failed)" in en_output


def test_print_report_localizes_summary_labels(capsys) -> None:
    report = {
        "summary": {
            "total_files": 2,
            "num_clusters": 1,
            "num_noise": 0,
            "noise_repaired": 1,
            "small_clusters_merged": 1,
            "name_clusters_merged": 1,
            "total_clusters_merged": 2,
        },
        "clusters": [{"name": "Cluster", "files": [{"name": "a.txt"}]}],
        "noise": [],
    }

    setup_logging()
    set_locale("zh-CN")
    _print_report(report)
    zh_output = capsys.readouterr().out

    set_locale("en")
    _print_report(report)
    en_output = capsys.readouterr().out

    assert "合并:" in zh_output
    assert "Merge:" in en_output
    assert "个小簇已合并" in zh_output
    assert "个同名簇已合并" in zh_output
    assert "clusters merged in total" in en_output
    assert "same-name clusters merged" in en_output


def test_logger_prefixes_follow_locale(capsys) -> None:
    setup_logging(verbose=True)
    logger = get_logger()

    set_locale("zh-CN")
    logger.debug("调试")
    logger.warning("测试")
    zh_output = capsys.readouterr().out

    set_locale("en")
    logger.debug("debug")
    logger.warning("test")
    en_output = capsys.readouterr().out

    assert "调试: 调试" in zh_output
    assert "警告: 测试" in zh_output
    assert "DEBUG: debug" in en_output
    assert "WARNING: test" in en_output


def test_analyze_document_logs_follow_locale(capsys) -> None:
    setup_logging(verbose=True)
    set_locale("en")

    calls: list[dict] = []
    client = _AnalyzerClient(["not-json"], calls)

    result = analyze_document(
        client,
        "example content",
        config=Config(),
        max_retries=1,
    )

    output = capsys.readouterr().out
    assert result == DocumentAnalysis()
    assert "JSON parsing failed (attempt 1/1)" in output
    assert "Document analysis failed, using defaults" in output
    assert "JSON 解析失败" not in output


def test_verbose_report_uses_stable_cluster_debug_labels(capsys) -> None:
    files = [Path("b.txt"), Path("a.txt")]
    contents = ["beta content", "alpha content"]
    labels = [2, 0]
    report = _build_report(
        files=files,
        contents=contents,
        embeddings=None,
        labels=np.array(labels, dtype=int),
        cluster_names={0: "Alpha", 2: "Beta"},
    )

    setup_logging(verbose=True)
    set_locale("en")
    _print_report(report)
    output = capsys.readouterr().out

    assert "【A | Alpha】" in output
    assert "【B | Beta】" in output
    assert output.index("【A | Alpha】") < output.index("【B | Beta】")


def test_build_report_includes_feature_diagnostics() -> None:
    report = _build_report(
        files=[Path("a.txt")],
        contents=["content"],
        labels=np.array([0], dtype=int),
        cluster_names={0: "Topic A"},
        embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
        document_features=[
            DocumentFeatures(
                file_id="a",
                path=Path("a.txt"),
                name="a.txt",
                stem="a",
                extension=".txt",
                metadata=MetadataFeatures(file_name_tokens=["a"]),
                quality_flags=QualityFlags(
                    short_text=True,
                    filename_dominant=True,
                ),
            )
        ],
        candidate_edges=[
            CandidateEdge(
                source_id="a",
                target_id="b",
                edge_type="filename_similarity",
                score=0.9,
                evidence=["shared tokens"],
            )
        ],
        candidate_components=[
            CandidateComponent(
                component_id="component-1",
                member_file_ids=["a", "b"],
                component_type="strong_semantic_group",
                formation_evidence=["shared tokens"],
                confidence=1.0,
            )
        ],
        cluster_drafts=[
            type(
                "_Draft",
                (),
                {
                    "draft_cluster_id": 0,
                    "member_file_ids": ["a"],
                    "origin": "density",
                    "noise_members": [],
                    "merge_candidates": [],
                },
            )()
        ],
    )

    assert report["summary"]["num_document_features"] == 1
    assert report["summary"]["num_candidate_edges"] == 1
    assert report["summary"]["num_candidate_components"] == 1
    assert report["summary"]["num_cluster_drafts"] == 1
    assert report["summary"]["num_adjudication_requests"] == 0
    assert report["summary"]["num_adjudication_decisions"] == 0
    assert report["summary"]["num_cluster_representations"] == 0
    assert report["feature_diagnostics"]["filename_dominant_count"] == 1
    assert report["feature_diagnostics"]["short_text_count"] == 1
    assert "density_validation_score" in report["structure_metrics"]
    assert "filename_bias_rate" in report["structure_metrics"]
    assert report["feature_diagnostics"]["cluster_drafts"][0]["origin"] == "density"
    assert report["feature_diagnostics"]["candidate_edges"][0]["edge_type"] == (
        "filename_similarity"
    )
