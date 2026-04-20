from dite.config import Config
from dite.core.embedder import (
    ContentTruncator,
    get_embedding_cache_version,
    get_embeddings,
)


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class _Item:
            def __init__(self) -> None:
                self.embedding = [0.1, 0.2, 0.3]

        class _Usage:
            total_tokens = 3

        class _Response:
            data = [_Item()]
            usage = _Usage()

        return _Response()


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()


def test_get_embeddings_includes_file_name_for_non_empty_text() -> None:
    client = _FakeClient()

    result = get_embeddings(
        client,
        ["This document explains Rust ownership and borrowing."],
        config=Config(),
        file_names=["Rust 程序设计.pdf"],
        embedding_model="embed-model",
    )

    assert result.shape == (1, 3)
    request_input = client.embeddings.calls[0]["input"][0]
    assert request_input.startswith("File name: Rust 程序设计.pdf")
    assert "Content:\nThis document explains Rust ownership" in request_input
    assert client.embeddings.calls[0]["model"] == "embed-model"


def test_get_embedding_cache_version_tracks_input_format() -> None:
    assert get_embedding_cache_version("embed-model") == (
        "embed-model|input=filename-smart-content-v1"
    )


def test_content_truncator_keeps_head_middle_tail_within_limit() -> None:
    content = ("HEAD-" * 30) + ("MIDDLE-" * 30) + ("TAIL-" * 30)

    truncated = ContentTruncator.truncate_smart(content, max_chars=120)

    assert len(truncated) <= 120
    assert truncated.startswith("HEAD-")
    assert "MIDDLE-" in truncated
    assert truncated.endswith("TAIL-")
    assert "middle omitted" in truncated
