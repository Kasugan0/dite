"""Cluster representation builders for clustering V2."""

from __future__ import annotations

from collections import Counter

from .model import ClusterRepresentation


def build_cluster_representations(
    *,
    labels,
    cluster_names,
    document_features,
) -> dict[int, ClusterRepresentation]:
    """Build conservative cluster representations from current document features."""
    cluster_indices: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if int(label) == -1:
            continue
        cluster_indices.setdefault(int(label), []).append(index)

    representations: dict[int, ClusterRepresentation] = {}
    for label, indices in cluster_indices.items():
        features = [document_features[index] for index in indices]
        summaries = [feature.summary for feature in features if feature.summary]
        topics = [feature.topic for feature in features if feature.topic]
        domains = [feature.domain for feature in features if feature.domain]
        keywords = [
            keyword
            for feature in features
            for keyword in feature.keywords
            if keyword
        ]
        top_keywords = [
            keyword for keyword, _count in Counter(keywords).most_common(5)
        ]
        topic = Counter(topics).most_common(1)[0][0] if topics else ""
        domain = Counter(domains).most_common(1)[0][0] if domains else ""
        summary = summaries[0] if summaries else ""
        evidence_parts = []
        if topic:
            evidence_parts.append(f"topic={topic}")
        if domain:
            evidence_parts.append(f"domain={domain}")
        if top_keywords:
            evidence_parts.append(f"keywords={', '.join(top_keywords[:3])}")

        representations[label] = ClusterRepresentation(
            cluster_id=label,
            name=cluster_names.get(label, ""),
            summary=summary,
            keywords=top_keywords,
            topic=topic,
            domain=domain,
            representative_file_ids=[features[0].file_id] if features else [],
            evidence_summary="; ".join(evidence_parts),
        )

    return representations
