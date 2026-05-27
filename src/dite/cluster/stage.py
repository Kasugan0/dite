"""Canonical clustering stage helpers for the shared pipeline."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import PCA

from dite.app.config import Config
from dite.cluster import apply_rule_adjudication, build_adjudication_requests
from dite.cluster.link import build_candidate_components, build_candidate_edges
from dite.cluster.model import (
    AdjudicationDecision,
    AdjudicationRequest,
    CandidateComponent,
    CandidateEdge,
)
from dite.doc import DocumentFeatures, build_document_features
from dite.doc.embed import normalize_embeddings

from .model import ClusterMetrics, ClusterResult

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class CanonicalClusterStageResult:
    """Canonical clustering stage output before duplicate expansion."""

    canonical_embeddings: np.ndarray
    cluster_result: ClusterResult
    noise_repaired: int
    document_features: list[DocumentFeatures]
    candidate_edges: list[CandidateEdge]
    candidate_components: list[CandidateComponent]
    adjudication_requests: list[AdjudicationRequest]
    adjudication_decisions: list[AdjudicationDecision]


def apply_candidate_component_links(
    result: ClusterResult,
    canonical_document_features: list[DocumentFeatures],
    candidate_components: list[CandidateComponent],
) -> ClusterResult:
    """Apply conservative must-link merging from strong candidate components."""
    if not candidate_components:
        return result

    index_by_file_id = {
        feature.file_id: index
        for index, feature in enumerate(canonical_document_features)
    }
    labels = result.labels.copy()
    merged_any = False

    for component in candidate_components:
        member_indices = [
            index_by_file_id[file_id]
            for file_id in component.member_file_ids
            if file_id in index_by_file_id
        ]
        if len(member_indices) < 2:
            continue
        component_labels = [int(labels[index]) for index in member_indices]
        non_noise_labels = sorted({label for label in component_labels if label != -1})
        if not non_noise_labels:
            continue
        target_label = non_noise_labels[0]
        for index in member_indices:
            if labels[index] != target_label:
                labels[index] = target_label
                merged_any = True

    if not merged_any:
        return result

    return ClusterResult(
        labels=labels,
        cluster_names=result.cluster_names.copy(),
        repaired_mask=result.repaired_mask.copy(),
        metrics=dataclasses.replace(result.metrics),
    )


def apply_adjudication_links(
    result: ClusterResult,
    canonical_document_features: list[DocumentFeatures],
    adjudication_requests: list[AdjudicationRequest],
    adjudication_decisions: list[AdjudicationDecision],
) -> ClusterResult:
    """Apply merge decisions from the adjudication layer to cluster labels."""
    if not adjudication_decisions:
        return result

    request_by_id = {request.request_id: request for request in adjudication_requests}
    index_by_file_id = {
        feature.file_id: index
        for index, feature in enumerate(canonical_document_features)
    }
    labels = result.labels.copy()
    merged_any = False

    for decision in adjudication_decisions:
        if decision.decision != "merge_edge":
            continue
        request = request_by_id.get(decision.request_id)
        if request is None or len(request.subjects) != 2:
            continue
        left_id, right_id = request.subjects
        if left_id not in index_by_file_id or right_id not in index_by_file_id:
            continue
        left_index = index_by_file_id[left_id]
        right_index = index_by_file_id[right_id]
        left_label = int(labels[left_index])
        right_label = int(labels[right_index])
        non_noise_labels = [label for label in (left_label, right_label) if label != -1]
        if not non_noise_labels:
            continue
        target_label = min(non_noise_labels)
        if labels[left_index] != target_label:
            labels[left_index] = target_label
            merged_any = True
        if labels[right_index] != target_label:
            labels[right_index] = target_label
            merged_any = True

    if not merged_any:
        return result

    return ClusterResult(
        labels=labels,
        cluster_names=result.cluster_names.copy(),
        repaired_mask=result.repaired_mask.copy(),
        metrics=dataclasses.replace(result.metrics),
    )


def cluster_canonical_documents(
    embeddings: np.ndarray,
    *,
    config: Config,
    options: Any,
    item_names: list[str],
    canonical_document_features: list[DocumentFeatures],
    candidate_edges: list[CandidateEdge],
    candidate_components: list[CandidateComponent],
    cluster_documents_fn,
) -> ClusterResult:
    """Run clustering on canonical embeddings and apply conservative links."""
    if options.cluster_mode == "graph":
        labels = np.full(len(canonical_document_features), -1, dtype=int)
        file_id_to_index = {
            feature.file_id: index
            for index, feature in enumerate(canonical_document_features)
        }
        adjacency: dict[int, set[int]] = {
            index: set() for index in range(len(canonical_document_features))
        }
        min_edge_score = 0.9
        for edge in candidate_edges:
            if edge.score < min_edge_score:
                continue
            if (
                edge.source_id not in file_id_to_index
                or edge.target_id not in file_id_to_index
            ):
                continue
            left = file_id_to_index[edge.source_id]
            right = file_id_to_index[edge.target_id]
            adjacency[left].add(right)
            adjacency[right].add(left)

        visited: set[int] = set()
        next_label = 0
        for index in range(len(canonical_document_features)):
            if index in visited or not adjacency[index]:
                continue
            stack = [index]
            component_indices: list[int] = []
            visited.add(index)
            while stack:
                current = stack.pop()
                component_indices.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)
            if len(component_indices) < 2:
                continue
            for component_index in component_indices:
                labels[component_index] = next_label
            next_label += 1
        unassigned_indices = [
            index for index, label in enumerate(labels) if label == -1
        ]
        if len(unassigned_indices) >= max(1, config.clustering.min_cluster_size):
            residual_result = cluster_documents_fn(
                embeddings[unassigned_indices],
                config=config,
                repair_noise=options.repair_noise,
                clustering=config.clustering,
                allow_single_cluster=options.cluster_allow_single_cluster,
                item_names=[item_names[index] for index in unassigned_indices],
            )
            residual_labels = residual_result.labels.copy()
            non_noise_labels = sorted(
                label for label in set(residual_labels) if int(label) != -1
            )
            label_map = {
                int(label): next_label + offset
                for offset, label in enumerate(non_noise_labels)
            }
            for offset, original_index in enumerate(unassigned_indices):
                residual_label = int(residual_labels[offset])
                if residual_label == -1:
                    continue
                labels[original_index] = label_map[residual_label]
            next_label += len(non_noise_labels)
            residual_noise_repaired = residual_result.metrics.noise_repaired
            residual_small_merged = residual_result.metrics.small_clusters_merged
            residual_name_merged = residual_result.metrics.name_clusters_merged
        else:
            residual_noise_repaired = 0
            residual_small_merged = 0
            residual_name_merged = 0

        return ClusterResult(
            labels=labels,
            cluster_names={},
            repaired_mask=np.zeros(labels.shape, dtype=bool),
            metrics=ClusterMetrics(
                initial_clusters=next_label,
                initial_noise=int(np.sum(labels == -1)),
                noise_repaired=residual_noise_repaired,
                small_clusters_merged=residual_small_merged,
                name_clusters_merged=residual_name_merged,
            ),
        )

    cluster_inputs = embeddings
    if options.cluster_pca_components is not None:
        pca_components = min(
            options.cluster_pca_components,
            embeddings.shape[0],
            embeddings.shape[1],
        )
        if pca_components >= 1:
            cluster_inputs = PCA(
                n_components=pca_components,
                random_state=0,
            ).fit_transform(embeddings)

    result = cluster_documents_fn(
        cluster_inputs,
        config=config,
        repair_noise=options.repair_noise,
        clustering=config.clustering,
        allow_single_cluster=options.cluster_allow_single_cluster,
        item_names=item_names,
    )
    return apply_candidate_component_links(
        result,
        canonical_document_features,
        candidate_components,
    )


def build_canonical_cluster_stage(
    *,
    files: list[Path],
    contents: list[str],
    file_hashes: list[str],
    file_reports: list[Any],
    canonical_indices: list[int],
    options: Any,
    config: Config,
    vectorize,
    expand_noise_repaired_count,
    cluster_documents_fn,
) -> CanonicalClusterStageResult:
    """Build all V2 stage objects and canonical clustering outputs."""
    document_features = [
        build_document_features(file, content, file_report=report)
        for file, content, report in zip(files, contents, file_reports, strict=True)
    ]
    canonical_files = [files[index] for index in canonical_indices]
    canonical_hashes = [file_hashes[index] for index in canonical_indices]
    canonical_contents = [contents[index] for index in canonical_indices]
    canonical_document_features = [
        document_features[index] for index in canonical_indices
    ]
    canonical_embeddings = vectorize(
        canonical_files,
        canonical_hashes,
        canonical_contents,
        options,
    )
    canonical_embeddings = normalize_embeddings(canonical_embeddings)
    canonical_count = len(canonical_document_features)
    embedding_count = int(canonical_embeddings.shape[0])
    if embedding_count < canonical_count:
        raise ValueError(
            "canonical embedding count is smaller than canonical document count: "
            f"{embedding_count} < {canonical_count}"
        )
    if embedding_count > canonical_count:
        canonical_embeddings = canonical_embeddings[:canonical_count]
    for canonical_feature, embedding in zip(
        canonical_document_features, canonical_embeddings, strict=True
    ):
        canonical_feature.content_embedding = embedding

    candidate_edges = build_candidate_edges(canonical_document_features)
    candidate_components = build_candidate_components(
        canonical_document_features, candidate_edges
    )
    adjudication_requests = build_adjudication_requests(
        candidate_edges, candidate_components
    )
    adjudication_decisions = apply_rule_adjudication(adjudication_requests)

    cluster_result = cluster_canonical_documents(
        canonical_embeddings,
        config=config,
        options=options,
        item_names=[file.name for file in canonical_files],
        canonical_document_features=canonical_document_features,
        candidate_edges=candidate_edges,
        candidate_components=candidate_components,
        cluster_documents_fn=cluster_documents_fn,
    )
    cluster_result = apply_adjudication_links(
        cluster_result,
        canonical_document_features,
        adjudication_requests,
        adjudication_decisions,
    )
    noise_repaired = expand_noise_repaired_count(
        canonical_indices,
        cluster_result.repaired_mask,
        file_hashes,
    )
    return CanonicalClusterStageResult(
        canonical_embeddings=canonical_embeddings,
        cluster_result=cluster_result,
        noise_repaired=noise_repaired,
        document_features=canonical_document_features,
        candidate_edges=candidate_edges,
        candidate_components=candidate_components,
        adjudication_requests=adjudication_requests,
        adjudication_decisions=adjudication_decisions,
    )
