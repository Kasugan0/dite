"""Feature builders for clustering V2."""

from __future__ import annotations

import re
from pathlib import Path

from dite.doc.model import (
    DocumentFeatures,
    EntityFeatures,
    LayoutHints,
    MetadataFeatures,
    QualityFlags,
)

_EXCERPT_LIMIT = 200
_SHORT_TEXT_THRESHOLD = 100
_FAILED_TEXT_THRESHOLD = 10
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "into",
    "notes",
    "guide",
    "report",
    "intro",
    "introduction",
    "study",
    "document",
    "content",
}


def _tokenize_text(text: str) -> list[str]:
    """Return normalized coarse tokens from a file name or path fragment."""
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


def _title_candidates(file_path: Path, content: str) -> list[str]:
    """Build conservative title candidates from extracted content and file name."""
    titles: list[str] = []
    for line in content.splitlines():
        stripped = re.sub(r"\s+", " ", line.strip())
        if 4 <= len(stripped) <= 120:
            titles.append(stripped)
            break
    stem_title = re.sub(r"[_-]+", " ", file_path.stem).strip()
    if stem_title and stem_title not in titles:
        titles.append(stem_title)
    return titles


def _infer_keywords_and_topic(
    content: str,
    title_candidates: list[str],
) -> tuple[list[str], str]:
    """Infer lightweight keywords/topic without invoking a model."""
    tokens = [token for token in _tokenize_text(content) if token not in _STOPWORDS]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    keywords = [token for token, _count in ranked[:5]]
    if title_candidates:
        title_tokens = [
            token
            for token in _tokenize_text(title_candidates[0])
            if token not in _STOPWORDS
        ]
        if title_tokens:
            return keywords, " ".join(title_tokens[:3])
    if keywords:
        return keywords, " ".join(keywords[:3])
    return [], ""


def _infer_domain(keywords: list[str], path_tokens: list[str]) -> str:
    joined = set(keywords) | set(path_tokens)
    if {"matrix", "vector", "algebra", "math"} & joined:
        return "education"
    if {"minecraft", "server", "mod", "fabric"} & joined:
        return "gaming"
    if {"invoice", "bank", "finance", "revenue"} & joined:
        return "finance"
    if {"python", "rust", "api", "code"} & joined:
        return "tech"
    return ""


def build_document_features(
    file_path: Path,
    content: str,
    *,
    file_report=None,
) -> DocumentFeatures:
    """Build a conservative V2 feature object from current pipeline artifacts."""
    stripped = content.strip()
    parent_tokens = []
    for part in file_path.parent.parts:
        parent_tokens.extend(_tokenize_text(part))

    primary_success = getattr(file_report, "primary_success", True)
    selected_source = getattr(file_report, "selected_source", "")
    primary_extractor = getattr(file_report, "primary_extractor", "")
    extraction_trace = ""
    if primary_extractor or selected_source:
        extraction_trace = f"{primary_extractor}->{selected_source}".strip("->")

    quality_flags = QualityFlags(
        extraction_failed=(
            (not primary_success) or len(stripped) < _FAILED_TEXT_THRESHOLD
        ),
        short_text=len(stripped) < _SHORT_TEXT_THRESHOLD,
        filename_dominant=(
            len(stripped) < _SHORT_TEXT_THRESHOLD and bool(file_path.stem)
        ),
    )
    title_candidates = _title_candidates(file_path, content)
    keywords, topic = _infer_keywords_and_topic(content, title_candidates)
    domain = _infer_domain(keywords, parent_tokens)

    return DocumentFeatures(
        file_id=str(file_path),
        path=file_path,
        name=file_path.name,
        stem=file_path.stem,
        extension=file_path.suffix.lower(),
        content_text=content,
        content_excerpt=(
            content[:_EXCERPT_LIMIT] + "..."
            if len(content) > _EXCERPT_LIMIT
            else content
        ),
        language="",
        token_count_estimate=len(_TOKEN_PATTERN.findall(content)),
        summary=title_candidates[0] if title_candidates else "",
        metadata=MetadataFeatures(
            file_name_tokens=_tokenize_text(file_path.stem),
            parent_path_tokens=parent_tokens,
            title_candidates=title_candidates,
        ),
        entities=EntityFeatures(
            keywords=keywords,
            topic=topic,
            domain=domain,
        ),
        layout=LayoutHints(),
        quality_flags=quality_flags,
        selected_source=str(selected_source),
        extraction_trace=extraction_trace,
    )
