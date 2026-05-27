"""Candidate edge and component builders for clustering V2."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dite.cluster.model import CandidateComponent, CandidateEdge
from dite.doc import DocumentFeatures


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.size == 0 or right_array.size == 0:
        return 0.0
    left_norm = float(np.linalg.norm(left_array))
    right_norm = float(np.linalg.norm(right_array))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / (left_norm * right_norm))


def build_candidate_edges(
    document_features: list[DocumentFeatures],
    *,
    min_filename_token_overlap: float = 0.8,
    min_content_similarity: float = 0.92,
) -> list[CandidateEdge]:
    """Build a conservative first-pass candidate edge set."""
    edges: list[CandidateEdge] = []

    for index, left in enumerate(document_features):
        left_tokens = set(left.file_name_tokens)
        left_titles = {title.casefold() for title in left.title_candidates}

        for right in document_features[index + 1 :]:
            right_tokens = set(right.file_name_tokens)
            right_titles = {title.casefold() for title in right.title_candidates}
            shared_titles = left_titles & right_titles
            similarity = _jaccard(left_tokens, right_tokens)
            content_similarity = _cosine_similarity(
                left.content_embedding, right.content_embedding
            )

            evidence: list[str] = []
            edge_type = None
            score = 0.0

            if shared_titles:
                edge_type = "title_match"
                score = 1.0
                evidence.append(f"title_match:{sorted(shared_titles)[0]}")
            elif content_similarity >= min_content_similarity:
                edge_type = "content_similarity"
                score = content_similarity
                evidence.append(f"content_similarity:{content_similarity:.3f}")
            elif (
                similarity >= min_filename_token_overlap
                and left_tokens
                and right_tokens
            ):
                edge_type = "filename_similarity"
                score = similarity
                evidence.append(
                    f"filename_token_overlap:{sorted(left_tokens & right_tokens)}"
                )

            if edge_type is None:
                continue

            quality_guard = []
            if left.quality_flags.short_text or right.quality_flags.short_text:
                quality_guard.append("short_text")
            if (
                left.quality_flags.filename_dominant
                or right.quality_flags.filename_dominant
            ):
                quality_guard.append("filename_dominant")

            edges.append(
                CandidateEdge(
                    source_id=left.file_id,
                    target_id=right.file_id,
                    edge_type=edge_type,
                    score=score,
                    evidence=evidence,
                    quality_guard=quality_guard,
                )
            )

    return edges


def build_candidate_components(
    document_features: list[DocumentFeatures],
    edges: list[CandidateEdge],
    *,
    min_edge_score: float = 0.9,
) -> list[CandidateComponent]:
    """Build conservative connected components from strong edges."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        if edge.score < min_edge_score:
            continue
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)

    visited: set[str] = set()
    file_ids = [feature.file_id for feature in document_features]
    components: list[CandidateComponent] = []
    component_index = 0

    for file_id in file_ids:
        if file_id in visited or file_id not in adjacency:
            continue
        stack = [file_id]
        members: list[str] = []
        visited.add(file_id)
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                stack.append(neighbor)

        if len(members) < 2:
            continue
        component_index += 1
        components.append(
            CandidateComponent(
                component_id=f"component-{component_index}",
                member_file_ids=sorted(members),
                component_type="strong_semantic_group",
                formation_evidence=["strong_candidate_edges"],
                confidence=1.0,
            )
        )

    return components
