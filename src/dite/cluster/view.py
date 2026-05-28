"""Cluster representation builders for clustering V2."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from openai import OpenAI

from dite.app.config import Config
from dite.util.api import ChatCompletionRequest
from dite.util.llm import build_chat_completion_kwargs

from .model import ClusterRepresentation


def build_cluster_representations(
    *,
    labels,
    cluster_names,
    document_features,
    config: Config | None = None,
    client: OpenAI | None = None,
    request_runtime=None,
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

        if (
            config is not None
            and client is not None
            and request_runtime is not None
            and config.cluster_representation.mode == "llm_enhanced"
        ):
            prompt = _cluster_representation_prompt(
                cluster_name=cluster_names.get(label, ""),
                summaries=summaries,
                topics=topics,
                domains=domains,
                keywords=top_keywords,
                file_names=[Path(feature.file_id).name for feature in features[:5]],
            )
            request = ChatCompletionRequest(
                kwargs=build_chat_completion_kwargs(
                    client=client,
                    model=config.models.llm,
                    messages=[{"role": "user", "content": prompt}],
                    profile=config.request_profiles.cluster_naming,
                    response_format={"type": "json_object"},
                )
            )
            response = request_runtime.run_cluster_naming_batch([request])[0]
            if response.error is None and response.content:
                llm_representation = _parse_llm_cluster_representation(response.content)
                if llm_representation is not None:
                    summary = llm_representation["summary"] or summary
                    topic = llm_representation["topic"] or topic
                    domain = llm_representation["domain"] or domain
                    if llm_representation["keywords"]:
                        top_keywords = llm_representation["keywords"]
                    if llm_representation["evidence_summary"]:
                        evidence_parts = [llm_representation["evidence_summary"]]

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


def _cluster_representation_prompt(
    *,
    cluster_name: str,
    summaries: list[str],
    topics: list[str],
    domains: list[str],
    keywords: list[str],
    file_names: list[str],
) -> str:
    joined_summaries = "\n".join(f"- {summary}" for summary in summaries[:3]) or "-"
    joined_topics = ", ".join(topics[:5]) or "-"
    joined_domains = ", ".join(domains[:5]) or "-"
    joined_keywords = ", ".join(keywords[:8]) or "-"
    joined_files = ", ".join(file_names[:5]) or "-"
    return (
        "Return strict JSON with keys summary, topic, domain, keywords, "
        "evidence_summary.\n"
        f"Cluster name: {cluster_name or '-'}\n"
        f"Representative files: {joined_files}\n"
        f"Summaries:\n{joined_summaries}\n"
        f"Topics: {joined_topics}\n"
        f"Domains: {joined_domains}\n"
        f"Keywords: {joined_keywords}\n"
    )


def _parse_llm_cluster_representation(content: str) -> dict[str, object] | None:
    import json

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "topic": str(payload.get("topic", "")).strip(),
        "domain": str(payload.get("domain", "")).strip(),
        "keywords": [
            str(item).strip()
            for item in (payload.get("keywords") or [])
            if str(item).strip()
        ],
        "evidence_summary": str(payload.get("evidence_summary", "")).strip(),
    }
