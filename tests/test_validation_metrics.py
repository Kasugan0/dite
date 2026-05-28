from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dite.flow.api import PipelineResult
from dite.validation import (
    build_constraint_metrics,
    build_structure_metrics,
    load_validation_corpus,
)


def _write_manifest(root: Path, payload: dict) -> None:
    (root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pipeline_result(root: Path, labels: list[int]) -> PipelineResult:
    files = [root / "a.txt", root / "b.txt", root / "c.txt"]
    return PipelineResult(
        files=files,
        contents=["alpha", "beta", "gamma"],
        document_features=[],
        embeddings=np.array(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0]],
            dtype=np.float32,
        ),
        labels=np.array(labels, dtype=int),
        cluster_names={0: "A", 1: "B"},
    )


def test_load_validation_corpus_builds_constraints(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {
                    "path": "a.txt",
                    "cluster_id": "group_1",
                    "must_link": ["b.txt"],
                },
                {
                    "path": "b.txt",
                    "cluster_id": "group_1",
                },
                {
                    "path": "c.txt",
                    "cluster_id": "group_2",
                    "must_not_link": ["a.txt"],
                },
            ],
        },
    )

    corpus = load_validation_corpus(tmp_path)

    assert corpus.name == "sample"
    assert corpus.constraints.must_link_pairs == frozenset({("a.txt", "b.txt")})
    assert corpus.constraints.must_not_link_pairs == frozenset({("a.txt", "c.txt")})


def test_load_validation_corpus_rejects_missing_targets(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "regression",
            "description": "demo",
            "owner": "team",
            "files": [
                {
                    "path": "a.txt",
                    "cluster_id": "group_1",
                    "must_link": ["missing.txt"],
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="invalid must_link target"):
        load_validation_corpus(tmp_path)


def test_build_constraint_metrics_uses_manifest_pairs(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {
                    "path": "a.txt",
                    "cluster_id": "group_1",
                    "must_link": ["b.txt"],
                },
                {
                    "path": "b.txt",
                    "cluster_id": "group_1",
                },
                {
                    "path": "c.txt",
                    "cluster_id": "group_2",
                    "must_not_link": ["a.txt"],
                },
            ],
        },
    )
    corpus = load_validation_corpus(tmp_path)
    result = _pipeline_result(tmp_path, [0, 0, 0])

    metrics = build_constraint_metrics(result, corpus)

    assert metrics.must_link_total == 1
    assert metrics.must_link_recall == 1.0
    assert metrics.must_not_link_total == 1
    assert metrics.must_not_link_violations == 1
    assert metrics.must_not_link_violation_rate == 1.0
    assert metrics.cluster_id_fragmentation_total == 0
    assert metrics.cluster_id_purity_by_id["group_1"] == 1.0


def test_build_constraint_metrics_counts_fragmentation(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    _write_manifest(
        tmp_path,
        {
            "name": "sample",
            "tier": "representative",
            "description": "demo",
            "owner": "team",
            "files": [
                {"path": "a.txt", "cluster_id": "group_1"},
                {"path": "b.txt", "cluster_id": "group_1"},
                {"path": "c.txt", "cluster_id": "group_1"},
            ],
        },
    )
    corpus = load_validation_corpus(tmp_path)
    result = _pipeline_result(tmp_path, [0, 1, -1])

    metrics = build_constraint_metrics(result, corpus)

    assert metrics.cluster_id_fragmentation_total == 1
    assert metrics.cluster_id_fragmentation_by_id["group_1"] == 1
    assert metrics.cluster_id_purity_by_id["group_1"] == pytest.approx(1 / 3)


def test_build_structure_metrics_returns_none_without_feature_edges(tmp_path: Path) -> None:
    result = _pipeline_result(tmp_path, [0, 0, -1])

    metrics = build_structure_metrics(result)

    assert metrics.filename_bias_rate is None
