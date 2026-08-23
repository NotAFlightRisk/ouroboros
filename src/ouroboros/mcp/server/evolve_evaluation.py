"""Semantic fallback evaluation for evolve generations."""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

from ouroboros.core.lineage import ACResult, EvaluationSummary
from ouroboros.core.seed import AcceptanceCriterionSpec, ac_text
from ouroboros.evaluation.models import EvaluationContext, EvaluationResult, SemanticResult

log = structlog.get_logger(__name__)


def warn_if_seed_has_no_success_contract(seed: Any) -> None:
    """Expose when evolve has no fixed, structured verification anchor."""
    criteria = tuple(getattr(seed, "acceptance_criteria", ()) or ())

    structured_indices: list[int] = []
    unanchored_indices: list[int] = []
    for i, criterion in enumerate(criteria):
        if isinstance(criterion, AcceptanceCriterionSpec) and criterion.has_success_contract:
            structured_indices.append(i)
        else:
            unanchored_indices.append(i)

    metadata = getattr(seed, "metadata", None)
    seed_id = getattr(metadata, "seed_id", None)

    if not criteria:
        log.warning(
            "evolution.acceptance_contract.unstructured",
            seed_id=seed_id,
            acceptance_criteria=0,
            consequence=(
                "the Seed declares no acceptance criteria; evolve cannot establish coverage"
            ),
        )
        return

    if not structured_indices:
        # Wholly prose-only: no criterion has a structured contract.
        log.warning(
            "evolution.acceptance_contract.unstructured",
            seed_id=seed_id,
            acceptance_criteria=len(criteria),
            consequence=(
                "no AC declares verify_command, expected_artifacts, or output_assertion; "
                "evolve has no fixed structured verification anchor"
            ),
        )
        return

    if unanchored_indices:
        # Mixed: some structured, some prose-only.
        log.warning(
            "evolution.acceptance_contract.mixed",
            seed_id=seed_id,
            acceptance_criteria=len(criteria),
            structured_count=len(structured_indices),
            unanchored_count=len(unanchored_indices),
            unanchored_indices=unanchored_indices,
            consequence=(
                f"{len(unanchored_indices)}/{len(criteria)} AC(s) at indices "
                f"{unanchored_indices} have no structured verification anchor; "
                "evolve cannot mechanically verify those criteria"
            ),
        )


def _has_meaningful_evidence(stage2: SemanticResult) -> bool:
    """Require non-trivial verification questions AND evidence for authority.

    The semantic prompt contract requires the evaluator to show its work:
    criterion-specific questions it asked and concrete evidence it observed.
    Without both, the verdict is advisory, not authoritative.
    """
    has_questions = any(q.strip() for q in stage2.questions_used)
    has_evidence = any(e.strip() for e in stage2.evidence)
    return has_questions and has_evidence


def _rejected_summary(reason: str) -> EvaluationSummary:
    return EvaluationSummary(
        final_approved=False,
        highest_stage_passed=1,
        score=0.0,
        drift_score=1.0,
        failure_reason=reason,
        approval_status="rejected",
        execution_completion_status="completed",
    )


def _render_evidence(result: EvaluationResult) -> str:
    stage2 = result.stage2_result
    if stage2 is None:
        return result.failure_reason or "Evaluation did not produce a semantic verdict."
    parts = [*stage2.evidence]
    if stage2.reasoning and stage2.reasoning not in parts:
        parts.append(stage2.reasoning)
    if result.failure_reason and result.failure_reason not in parts:
        parts.append(result.failure_reason)
    return "; ".join(parts) or "Semantic evaluation produced no evidence details."


def _failure_reason(
    ac_results: tuple[ACResult, ...],
    results: tuple[EvaluationResult, ...],
) -> str | None:
    """Build human-readable failure reason from per-AC verdicts."""
    failed = [
        (
            ac.ac_index + 1,
            ac.ac_content,
            (
                result.failure_reason
                if result.stage2_result is not None
                else "semantic evaluation did not run"
            )
            if not ac.passed
            else None,
        )
        for ac, result in zip(ac_results, results, strict=True)
        if not ac.authoritative_pass
    ]
    if not failed:
        return None
    total = len(ac_results)
    details = "; ".join(
        f"AC {index} ({content}): {reason or 'insufficient evidence for authoritative pass'}"
        for index, content, reason in failed
    )
    return f"{len(failed)}/{total} acceptance criteria failed: {details}"


def _build_ac_result(index: int, criterion: Any, result: EvaluationResult) -> ACResult:
    """Construct an ACResult enforcing evidence requirements for authority.

    A semantic verdict becomes authoritative only when the evaluator supplied
    meaningful criterion-specific verification questions AND concrete evidence.
    Without both, the verdict is recorded but remains non-authoritative so that
    downstream focus selection, convergence, and lineage consumers do not
    consume an unproven PASS.
    """
    stage2 = result.stage2_result
    has_stage2 = stage2 is not None
    evidence_sufficient = has_stage2 and _has_meaningful_evidence(stage2)

    # Authority requires both Stage 2 presence and meaningful evidence.
    # Without evidence, the verdict is advisory — the AC remains unresolved.
    if has_stage2 and result.final_approved and evidence_sufficient:
        passed = True
        ac_verdict_state = "evaluated"
        final_verdict = "pass"
        rendered_verdict = "PASS"
    elif has_stage2 and not result.final_approved:
        # Explicit rejection is always authoritative (fail-closed is safe).
        passed = False
        ac_verdict_state = "evaluated"
        final_verdict = "fail"
        rendered_verdict = "FAIL"
    elif has_stage2 and result.final_approved and not evidence_sufficient:
        # Model said pass but provided no proof — non-authoritative.
        passed = False
        ac_verdict_state = "not_evaluated"
        final_verdict = "fail"
        rendered_verdict = "INSUFFICIENT_EVIDENCE"
    else:
        # No Stage 2 at all.
        passed = False
        ac_verdict_state = "not_evaluated"
        final_verdict = "fail"
        rendered_verdict = "NOT_EVALUATED"

    return ACResult(
        ac_index=index,
        ac_content=ac_text(criterion),
        semantic_ac_key=getattr(criterion, "semantic_ac_key", None),
        passed=passed,
        score=stage2.score if stage2 else 0.0,
        evidence=_render_evidence(result),
        verification_method=("semantic_evaluation" if has_stage2 else "unknown"),
        ac_verdict_state=ac_verdict_state,
        final_verdict=final_verdict,
        rendered_verdict=rendered_verdict,
    )


async def evaluate_seed_criteria(
    *,
    seed: Any,
    artifact: str,
    artifact_bundle: Any,
    pipeline: Any,
) -> EvaluationSummary:
    """Evaluate evolve's semantic fallback once per acceptance criterion."""
    criteria = tuple(getattr(seed, "acceptance_criteria", ()) or ())
    metadata = getattr(seed, "metadata", None)
    seed_id = str(getattr(metadata, "seed_id", "unknown"))
    goal = str(getattr(seed, "goal", ""))
    constraints = tuple(getattr(seed, "constraints", ()) or ())

    if not criteria:
        return _rejected_summary(
            "Seed has no acceptance criteria; semantic evaluation cannot establish coverage"
        )

    def context_for(index: int, criterion: Any) -> EvaluationContext:
        return EvaluationContext(
            execution_id=f"eval_{seed_id}_ac_{index + 1}",
            seed_id=seed_id,
            current_ac=ac_text(criterion),
            current_ac_spec=(criterion if isinstance(criterion, AcceptanceCriterionSpec) else None),
            artifact=artifact,
            artifact_type="code",
            goal=goal,
            constraints=constraints,
            artifact_bundle=artifact_bundle,
        )

    try:
        first = await pipeline.evaluate(context_for(0, criteria[0]))
    except Exception as exc:
        return _rejected_summary(f"AC 1 evaluation raised {type(exc).__name__}: {exc}")
    if first.is_err:
        return _rejected_summary(f"AC 1 evaluation failed: {first.error}")

    shared_stage1 = first.value.stage1_result

    async def evaluate_one(index: int, criterion: Any) -> Any:
        return await pipeline.evaluate(
            context_for(index, criterion),
            stage1_result=shared_stage1,
        )

    remaining = await asyncio.gather(
        *(evaluate_one(index, criterion) for index, criterion in enumerate(criteria[1:], start=1)),
        return_exceptions=True,
    )
    gathered = (first, *remaining)
    for index, entry in enumerate(gathered, start=1):
        if isinstance(entry, BaseException):
            return _rejected_summary(
                f"AC {index} evaluation raised {type(entry).__name__}: {entry}"
            )
        if entry.is_err:
            return _rejected_summary(f"AC {index} evaluation failed: {entry.error}")

    results = tuple(entry.value for entry in gathered)
    stage2_results = tuple(
        result.stage2_result for result in results if result.stage2_result is not None
    )
    scores = tuple(result.score for result in stage2_results)
    drift_scores = tuple(result.drift_score for result in stage2_results)
    reward_hacking_risks = tuple(result.reward_hacking_risk for result in stage2_results)
    ac_results = tuple(
        _build_ac_result(index, criterion, result)
        for index, (criterion, result) in enumerate(zip(criteria, results, strict=True))
    )

    # Aggregate approval derives from per-AC authority: every criterion must
    # have passed with sufficient evidence for the aggregate to approve.
    approved = all(ac.authoritative_pass for ac in ac_results)

    return EvaluationSummary(
        final_approved=approved,
        highest_stage_passed=min(max(1, result.highest_stage_completed) for result in results),
        score=sum(scores) / len(scores) if scores else 0.0,
        drift_score=max(drift_scores) if drift_scores else None,
        reward_hacking_risk=max(reward_hacking_risks) if reward_hacking_risks else None,
        failure_reason=_failure_reason(ac_results, results) if not approved else None,
        ac_results=ac_results,
        approval_status="approved" if approved else "rejected",
        execution_completion_status="completed",
    )


__all__ = ["evaluate_seed_criteria", "warn_if_seed_has_no_success_contract"]
