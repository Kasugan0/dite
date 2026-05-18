from pathlib import Path

import numpy as np
import pytest

from dite.core.analyzer import analyze_and_build_payload, analyze_document
from dite.core.clusterer import (
    cluster_documents,
    generate_all_cluster_names,
    generate_cluster_name,
)
from dite.core.embedder import get_embeddings
from dite.core.extractor import extract_content as core_extract_content
from dite.core.pipeline import PipelineService
from dite.core.scanner import scan_files
from dite.extractors.router import (
    classify_pdf_profile,
    extract_content,
    extract_document,
    extract_with_vlm_fallback,
    get_extractor,
    needs_vlm_fallback,
    resolve_document_extraction,
)


def call_invalid_signature(func, *args, **kwargs):
    return func(*args, **kwargs)


def test_pipeline_service_requires_explicit_config() -> None:
    with pytest.raises(TypeError):
        call_invalid_signature(PipelineService, client=object())


def test_core_public_functions_require_explicit_config(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("payload", encoding="utf-8")
    labels = np.array([0], dtype=int)
    embeddings = np.array([[0.1, 0.2]], dtype=np.float32)

    with pytest.raises(TypeError):
        call_invalid_signature(scan_files, tmp_path)

    with pytest.raises(TypeError):
        call_invalid_signature(get_embeddings, object(), ["payload"])

    with pytest.raises(TypeError):
        call_invalid_signature(analyze_document, object(), "payload")

    with pytest.raises(TypeError):
        call_invalid_signature(analyze_and_build_payload, object(), "payload")

    with pytest.raises(TypeError):
        call_invalid_signature(cluster_documents, embeddings)

    with pytest.raises(TypeError):
        call_invalid_signature(
            generate_cluster_name,
            object(),
            None,
            ["payload"],
            ["sample.txt"],
        )

    with pytest.raises(TypeError):
        call_invalid_signature(
            generate_all_cluster_names,
            object(),
            labels,
            ["payload"],
            [sample],
        )

    with pytest.raises(TypeError):
        call_invalid_signature(core_extract_content, sample)


def test_extractor_public_functions_require_explicit_config(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("payload", encoding="utf-8")

    with pytest.raises(TypeError):
        call_invalid_signature(get_extractor, sample)

    with pytest.raises(TypeError):
        call_invalid_signature(extract_document, sample)

    with pytest.raises(TypeError):
        call_invalid_signature(resolve_document_extraction, sample)

    with pytest.raises(TypeError):
        call_invalid_signature(extract_content, sample)

    with pytest.raises(TypeError):
        call_invalid_signature(extract_with_vlm_fallback, sample, client=object())

    with pytest.raises(TypeError):
        call_invalid_signature(needs_vlm_fallback, "payload", sample)

    with pytest.raises(TypeError):
        call_invalid_signature(classify_pdf_profile, "payload", sample, success=True)
