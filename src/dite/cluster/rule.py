"""Conservative rule-based adjudication for clustering V2."""

from __future__ import annotations

from dite.cluster.model import (
    AdjudicationDecision,
    AdjudicationRequest,
    CandidateComponent,
    CandidateEdge,
)


def build_adjudication_requests(
    candidate_edges: list[CandidateEdge],
    candidate_components: list[CandidateComponent],
) -> list[AdjudicationRequest]:
    """Build minimal adjudication requests from currently ambiguous edge cases."""
    requests: list[AdjudicationRequest] = []
    for component in candidate_components:
        if len(component.member_file_ids) < 2:
            continue
        requests.append(
            AdjudicationRequest(
                request_id=f"component:{component.component_id}",
                request_type="component_review",
                subjects=component.member_file_ids,
                evidence_bundle=component.formation_evidence,
                trigger_reason="strong_component_candidate",
            )
        )
    for edge in candidate_edges:
        if edge.score < 0.85:
            continue
        requests.append(
            AdjudicationRequest(
                request_id=f"edge:{edge.source_id}->{edge.target_id}",
                request_type="edge_review",
                subjects=[edge.source_id, edge.target_id],
                evidence_bundle=edge.evidence,
                trigger_reason=edge.edge_type,
                score=edge.score,
                quality_guard=edge.quality_guard,
            )
        )
    return requests


def apply_rule_adjudication(
    requests: list[AdjudicationRequest],
) -> list[AdjudicationDecision]:
    """Apply conservative rule-only adjudication to current requests."""
    decisions: list[AdjudicationDecision] = []
    for request in requests:
        if request.request_type == "component_review":
            decisions.append(
                AdjudicationDecision(
                    request_id=request.request_id,
                    decision="keep_component",
                    confidence=1.0,
                    reason=(
                        "Strong candidate component preserved for downstream "
                        "clustering."
                    ),
                    supporting_evidence=request.evidence_bundle,
                )
            )
            continue
        if (
            request.request_type == "edge_review"
            and request.score is not None
            and request.score >= 0.92
            and "filename_dominant" not in request.quality_guard
        ):
            decisions.append(
                AdjudicationDecision(
                    request_id=request.request_id,
                    decision="merge_edge",
                    confidence=min(request.score, 1.0),
                    reason="High-confidence semantic edge promoted to merge decision.",
                    supporting_evidence=request.evidence_bundle,
                )
            )
            continue
        decisions.append(
            AdjudicationDecision(
                request_id=request.request_id,
                decision="review_edge",
                confidence=0.5,
                reason=(
                    "Strong edge retained as evidence for later merge or "
                    "assignment review."
                ),
                supporting_evidence=request.evidence_bundle,
            )
        )
    return decisions
