"""Deterministic bridge from formal evaluation results to Ralph lineage state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

from pydantic import ValidationError as PydanticValidationError
import yaml

from ouroboros.config import get_auto_evolve_max_generations
from ouroboros.core.errors import ValidationError
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

    identity = f"{seed_id}\0{session_id}".encode()
    return f"ralph-{seed_id}-{hashlib.sha256(identity).hexdigest()[:16]}"


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


async def enqueue_chained_ralph(
    evaluation_result: MCPToolResult,
    *,
    session_id: str,
    arguments: Mapping[str, Any],
    event_store: EventStore,
    job_manager: Any,
    start_ralph_handler: Any | None,
) -> MCPToolResult:
    """Project a rejection and start the configured pollable Ralph successor."""

    seed_content = arguments.get("seed_content")
    try:
        seed_data = yaml.safe_load(seed_content) if isinstance(seed_content, str) else None
        if not isinstance(seed_data, dict):
            raise ValueError("seed_content is unavailable or not a mapping")
        seed = Seed.from_dict(seed_data)
    except (ValueError, yaml.YAMLError, ValidationError, PydanticValidationError) as exc:
        return append_result_text(
            evaluation_result,
            "\nAutomatic Ralph: skipped because the original Seed could not be loaded.\n",
            meta={
                **evaluation_result.meta,
                "chained_ralph_skipped": "seed_unavailable",
                "chained_ralph_skip_detail": str(exc)[:1000],
            },
        )

    lineage_id = mint_chain_lineage_id(seed.metadata.seed_id, session_id)
    max_generations = get_auto_evolve_max_generations()
    try:
        await seed_gen1_lineage(
            event_store,
            lineage_id=lineage_id,
            seed=seed,
            summary=evaluation_summary_from_eval_meta(seed, evaluation_result.meta),
        )
        active = await job_manager.find_active_job_by_lineage(lineage_id, job_type="ralph")
        if active is not None:
            ralph_job_id = active.job_id
        else:
            if start_ralph_handler is None:
                raise RuntimeError(
                    "Automatic Ralph chaining is not configured: "
                    "StartRalphHandler dependency is missing"
                )
            started = await start_ralph_handler.handle(
                {
                    "lineage_id": lineage_id,
                    "execute": True,
                    "parallel": True,
                    "skip_qa": False,
                    "project_dir": arguments.get("working_dir"),
                    "max_generations": max_generations,
                }
            )
            if started.is_err:
                raise RuntimeError(started.error.message)
            ralph_job_id = started.value.meta.get("job_id")
            if not isinstance(ralph_job_id, str) or not ralph_job_id:
                raise RuntimeError("StartRalphHandler did not return a pollable Ralph job_id")
    except Exception as exc:  # noqa: BLE001 - chaining must never flip rejection.
        return append_result_text(
            evaluation_result,
            "\nAutomatic Ralph: enqueue failed; the evaluation verdict is unchanged.\n",
            meta={
                **evaluation_result.meta,
                "chained_ralph_error": str(exc)[:1000],
                "chained_ralph_lineage_id": lineage_id,
            },
        )

    return append_result_text(
        evaluation_result,
        (
            "\nAutomatic Ralph: queued bounded convergence loop\n"
            f"Ralph Job ID: {ralph_job_id}\nLineage ID: {lineage_id}\n"
        ),
        meta={
            **evaluation_result.meta,
            "chained_ralph_job_id": ralph_job_id,
            "chained_ralph_lineage_id": lineage_id,
            "chained_ralph_max_generations": max_generations,
        },
    )


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
