"""Core clustering algorithm implementation."""

from __future__ import annotations

import hdbscan
import numpy as np

from dite.app.config import ClusteringConfig, Config
from dite.app.i18n import t
from dite.cluster.name import _build_cluster_debug_labels
from dite.cluster.post import cluster_sizes
from dite.doc.embed import normalize_embeddings
from dite.util.log import get_logger

from .model import ClusterMetrics, ClusterResult


def cluster_documents(
    embeddings: np.ndarray,
    *,
    config: Config,
    repair_noise: bool = True,
    knn_k: int | None = None,
    knn_distance_threshold: float | None = None,
    clustering: ClusteringConfig | None = None,
    allow_single_cluster: bool = False,
    item_names: list[str] | None = None,
    hdbscan_module=hdbscan,
    repair_noise_with_knn_fn=None,
    repair_all_noise_with_similarity_fn=None,
    merge_small_clusters_by_similarity_fn=None,
) -> ClusterResult:
    """Cluster documents with HDBSCAN plus conservative post-processing."""
    from . import api as clusterer_module

    if repair_noise_with_knn_fn is None:
        repair_noise_with_knn_fn = clusterer_module._repair_noise_with_knn_for_algorithm
    if repair_all_noise_with_similarity_fn is None:
        repair_all_noise_with_similarity_fn = (
            clusterer_module._repair_all_noise_with_similarity_for_algorithm
        )
    if merge_small_clusters_by_similarity_fn is None:
        merge_small_clusters_by_similarity_fn = (
            clusterer_module._merge_small_clusters_by_similarity_for_algorithm
        )

    logger = get_logger()
    clustering_cfg = clustering or config.clustering
    embeddings = normalize_embeddings(embeddings)
    effective_knn_k = knn_k if knn_k is not None else clustering_cfg.knn_k
    effective_distance_threshold = (
        knn_distance_threshold
        if knn_distance_threshold is not None
        else clustering_cfg.knn_distance_threshold
    )

    logger.debug(t("debug_cluster_hdbscan_header"))
    logger.debug(
        t(
            "debug_cluster_hdbscan_min_cluster_size",
            value=clustering_cfg.min_cluster_size,
        )
    )
    logger.debug(
        t("debug_cluster_hdbscan_min_samples", value=clustering_cfg.min_samples)
    )
    logger.debug(
        t(
            "debug_cluster_hdbscan_epsilon",
            value=clustering_cfg.cluster_selection_epsilon,
        )
    )
    logger.debug(
        t(
            "debug_cluster_hdbscan_method",
            value=clustering_cfg.cluster_selection_method,
        )
    )
    logger.debug(
        t(
            "debug_cluster_input_vectors",
            count=embeddings.shape[0],
            dimension=embeddings.shape[1],
        )
    )

    if embeddings.shape[0] < clustering_cfg.min_cluster_size:
        labels = np.full(embeddings.shape[0], -1, dtype=int)
        return ClusterResult(
            labels=labels,
            cluster_names={},
            repaired_mask=np.zeros(labels.shape, dtype=bool),
            metrics=ClusterMetrics(initial_noise=int(labels.size)),
        )

    clusterer = hdbscan_module.HDBSCAN(
        min_cluster_size=clustering_cfg.min_cluster_size,
        min_samples=clustering_cfg.min_samples,
        cluster_selection_epsilon=clustering_cfg.cluster_selection_epsilon,
        metric="euclidean",
        cluster_selection_method=clustering_cfg.cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = int(np.sum(labels == -1))
    logger.debug(t("debug_cluster_initial_result", clusters=n_clusters, noise=n_noise))
    metrics = ClusterMetrics(initial_clusters=n_clusters, initial_noise=n_noise)

    debug_labels = _build_cluster_debug_labels(labels)
    if n_clusters > 0:
        cluster_size_labels = []
        for label in sorted(label for label in unique_labels if label != -1):
            size = int(np.sum(labels == label))
            cluster_size_labels.append(f"{debug_labels[int(label)]}:{size}")
        logger.debug(t("debug_cluster_sizes", sizes=", ".join(cluster_size_labels)))

    repaired_count = 0
    repaired_mask = np.zeros(labels.shape, dtype=bool)
    if repair_noise:
        before_repair = labels.copy()
        if n_clusters == 0 and n_noise > 0:
            labels, repaired_count = repair_all_noise_with_similarity_fn(
                embeddings,
                min_cluster_size=clustering_cfg.min_cluster_size,
                distance_threshold=effective_distance_threshold,
            )
        else:
            labels, repaired_count = repair_noise_with_knn_fn(
                embeddings,
                labels,
                k=effective_knn_k,
                distance_threshold=effective_distance_threshold,
                item_names=item_names,
            )
        repaired_mask = (before_repair == -1) & (labels != -1)
    metrics.noise_repaired = repaired_count

    if clustering_cfg.small_cluster_merge_enabled:
        labels_before_small_merge = labels.copy()
        (
            labels,
            metrics.small_clusters_merged,
            metrics.small_cluster_merge_events,
            metrics.small_cluster_skip_events,
            metrics.small_cluster_merge_max_similarity,
        ) = merge_small_clusters_by_similarity_fn(
            embeddings,
            labels,
            max_size=clustering_cfg.small_cluster_merge_max_size,
            cosine_threshold=clustering_cfg.small_cluster_merge_cosine_threshold,
        )
        metrics.small_cluster_merge_candidates = int(
            sum(
                1
                for size in cluster_sizes(labels_before_small_merge).values()
                if size <= clustering_cfg.small_cluster_merge_max_size
            )
        )
        metrics.small_cluster_merge_skipped = len(metrics.small_cluster_skip_events)

    return ClusterResult(
        labels=labels,
        cluster_names={},
        repaired_mask=repaired_mask,
        metrics=metrics,
    )
