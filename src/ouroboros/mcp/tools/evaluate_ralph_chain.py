"""Deterministic bridge from formal evaluation results to Ralph lineage state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from ouroboros.core.lineage import ACResult, EvaluationSummary
from ouroboros.core.seed import Seed, ac_text
from ouroboros.events.lineage import lineage_created, lineage_generation_completed
from ouroboros.mcp.types import ContentType, MCPContentItem, MCPToolResult
from ouroboros.persistence.event_store import EventStore


def _checklist_by_text(checklist: object) -> dict[str, list[Mapping[str, Any]]]:
    indexed: dict[str, list[Mapping[str, Any]]] = {}
    if not isinstance(checklist, Sequence) or isinstance(checklist, str | bytes):
        return indexed
    for raw_item in checklist:
        if not isinstance(raw_item, Mapping):
            continue
        raw_text = raw_item.get("ac_text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
        indexed.setdefault(raw_text.strip(), []).append(raw_item)
    return indexed


def _evidence_text(item: Mapping[str, Any]) -> str:
    parts: list[str] = []
    evidence = item.get("evidence")
    if isinstance(evidence, Sequence) and not isinstance(evidence, str | bytes):
        parts.extend(str(value).strip() for value in evidence if str(value).strip())
    for key in ("reasoning", "failure_reason"):
        value = item.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return "; ".join(parts)


def ac_results_from_checklist(seed: Seed, checklist: object) -> tuple[ACResult, ...]:
    """Map evaluator checklist rows onto the Seed's stable AC order."""

    # Single-AC evaluations do not emit the checklist contract. Ralph still
    # converges via its full-graph fallback, so do not fabricate a verdict.
    if not isinstance(checklist, Sequence) or isinstance(checklist, str | bytes):
        return ()
    rows = _checklist_by_text(checklist)
    results: list[ACResult] = []
    for index, criterion in enumerate(seed.acceptance_criteria):
        content = ac_text(criterion)
        matching = rows.get(content.strip(), [])
        item = matching.pop(0) if matching else None
        if item is None:
            results.append(
                ACResult(
                    ac_index=index,
                    ac_content=content,
                    semantic_ac_key=criterion.semantic_ac_key,
                    passed=False,
                    score=0.0,
                    evidence="No formal evaluation checklist result was produced for this AC.",
                    verification_method="formal_evaluation",
                    ac_verdict_state="not_evaluated",
                    final_verdict="fail",
                    rendered_verdict="NOT_EVALUATED",
                )
            )
            continue
        passed = item.get("passed") is True
        results.append(
            ACResult(
                ac_index=index,
                ac_content=content,
                semantic_ac_key=criterion.semantic_ac_key,
                passed=passed,
                score=1.0 if passed else 0.0,
                evidence=_evidence_text(item) or "Formal evaluation produced no evidence details.",
                verification_method="formal_evaluation",
                ac_verdict_state="evaluated",
                final_verdict="pass" if passed else "fail",
                rendered_verdict="PASS" if passed else "FAIL",
            )
        )
    return tuple(results)


def evaluation_summary_from_eval_meta(seed: Seed, meta: Mapping[str, Any]) -> EvaluationSummary:
    """Build the generation-1 evaluation snapshot consumed by focused evolve."""

    approved = meta.get("final_approved") is True
    score_value = meta.get("pass_rate")
    score = float(score_value) if isinstance(score_value, int | float) else None
    feedback = meta.get("run_feedback")
    failure_reason = None
    if not approved and isinstance(feedback, Sequence) and not isinstance(feedback, str | bytes):
        rendered = [str(item).strip() for item in feedback if str(item).strip()]
        if rendered:
            failure_reason = "; ".join(rendered)
    if not approved and failure_reason is None:
        failure_reason = "formal evaluation rejected the run"
    return EvaluationSummary(
        final_approved=approved,
        highest_stage_passed=3 if approved else 2,
        score=score,
        failure_reason=failure_reason,
        ac_results=ac_results_from_checklist(seed, meta.get("checklist")),
        execution_completion_status="completed",
        approval_status="approved" if approved else "rejected",
    )


def mint_chain_lineage_id(seed_id: str, session_id: str) -> str:
    """Mint one deterministic lineage per Seed and originating run session."""

    return f"ralph-{seed_id}-{session_id[:8]}"


async def seed_gen1_lineage(
    event_store: EventStore,
    *,
    lineage_id: str,
    seed: Seed,
    summary: EvaluationSummary,
) -> bool:
    """Project a completed generation 1 once; return whether events were appended."""

    if await event_store.replay_lineage(lineage_id):
        return False
    await event_store.append(lineage_created(lineage_id, seed.goal))
    await event_store.append(
        lineage_generation_completed(
            lineage_id,
            generation_number=1,
            seed_id=seed.metadata.seed_id,
            ontology_snapshot=seed.ontology_schema.model_dump(mode="json"),
            evaluation_summary=summary.model_dump(mode="json"),
            seed_json=json.dumps(seed.to_dict(), ensure_ascii=False),
        )
    )
    return True


def append_result_text(
    result: MCPToolResult,
    text: str,
    *,
    meta: Mapping[str, Any],
) -> MCPToolResult:
    """Return a result with additive chain status while preserving the verdict."""

    return MCPToolResult(
        content=(*result.content, MCPContentItem(type=ContentType.TEXT, text=text)),
        is_error=result.is_error,
        meta=dict(meta),
        structured_content=result.structured_content,
    )
