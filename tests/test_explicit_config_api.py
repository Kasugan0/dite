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


def test_pipeline_service_requires_explicit_config() -> None:
    with pytest.raises(TypeError):
        PipelineService(client=object())  # type: ignore[call-arg]


def test_core_public_functions_require_explicit_config(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("payload", encoding="utf-8")
    labels = np.array([0], dtype=int)
    embeddings = np.array([[0.1, 0.2]], dtype=np.float32)

    with pytest.raises(TypeError):
        scan_files(tmp_path)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        get_embeddings(object(), ["payload"])  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        analyze_document(object(), "payload")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        analyze_and_build_payload(object(), "payload")  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        cluster_documents(embeddings)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        generate_cluster_name(  # type: ignore[call-arg]
            object(),
            None,
            ["payload"],
            ["sample.txt"],
        )

    with pytest.raises(TypeError):
        generate_all_cluster_names(  # type: ignore[call-arg]
            object(),
            labels,
            ["payload"],
            [sample],
        )

    with pytest.raises(TypeError):
        core_extract_content(sample)  # type: ignore[call-arg]


def test_extractor_public_functions_require_explicit_config(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("payload", encoding="utf-8")

    with pytest.raises(TypeError):
        get_extractor(sample)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        extract_document(sample)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        resolve_document_extraction(sample)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        extract_content(sample)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        extract_with_vlm_fallback(sample, client=object())  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        needs_vlm_fallback("payload", sample)  # type: ignore[call-arg]

    with pytest.raises(TypeError):
        classify_pdf_profile("payload", sample, success=True)  # type: ignore[call-arg]
