"""Regression tests for evolve's semantic fallback."""

from typing import Any

import pytest
from structlog.testing import capture_logs

from ouroboros.core.seed import (
    AcceptanceCriterionSpec,
    OntologySchema,
    Seed,
    SeedMetadata,
)
from ouroboros.core.types import Result
from ouroboros.evaluation.models import EvaluationResult, MechanicalResult, SemanticResult
from ouroboros.mcp.server.evolve_evaluation import (
    evaluate_seed_criteria,
    warn_if_seed_has_no_success_contract,
)


class _Pipeline:
    """Pipeline that supplies both questions_used and evidence (valid authority)."""

    def __init__(self, verdicts: tuple[bool, ...]) -> None:
        self.verdicts = verdicts
        self.contexts: list[Any] = []

    async def evaluate(self, context: Any, *, stage1_result: Any = None) -> Any:
        del stage1_result
        self.contexts.append(context)
        index = int(context.execution_id.rsplit("_", 1)[-1]) - 1
        passed = self.verdicts[index]
        semantic = SemanticResult(
            score=0.9 if passed else 0.4,
            ac_compliance=passed,
            goal_alignment=0.9,
            drift_score=0.1 if passed else 0.6,
            uncertainty=0.1,
            reasoning=f"criterion {index + 1} {'passed' if passed else 'failed'}",
            reward_hacking_risk=0.0,
            questions_used=(f"Does AC {index + 1} hold?",),
            evidence=(f"evidence-{index + 1}",),
        )
        return Result.ok(
            EvaluationResult(
                execution_id=context.execution_id,
                stage2_result=semantic,
                final_approved=passed,
            )
        )


class _EvidenceFreePipeline:
    """Pipeline that returns a semantic PASS with no questions or evidence."""

    def __init__(
        self,
        *,
        questions_used: tuple[str, ...] = (),
        evidence: tuple[str, ...] = (),
    ) -> None:
        self.questions_used = questions_used
        self.evidence = evidence
        self.contexts: list[Any] = []

    async def evaluate(self, context: Any, *, stage1_result: Any = None) -> Any:
        del stage1_result
        self.contexts.append(context)
        semantic = SemanticResult(
            score=0.95,
            ac_compliance=True,
            goal_alignment=0.95,
            drift_score=0.05,
            uncertainty=0.05,
            reasoning="looks good",
            reward_hacking_risk=0.0,
            questions_used=self.questions_used,
            evidence=self.evidence,
        )
        return Result.ok(
            EvaluationResult(
                execution_id=context.execution_id,
                stage2_result=semantic,
                final_approved=True,
            )
        )


class _NoSemanticPipeline:
    def __init__(
        self,
        *,
        final_approved: bool,
        stage1_result: MechanicalResult | None = None,
    ) -> None:
        self.final_approved = final_approved
        self.stage1_result = stage1_result
        self.contexts: list[Any] = []
        self.injected_stage1: list[MechanicalResult | None] = []

    async def evaluate(self, context: Any, *, stage1_result: Any = None) -> Any:
        self.contexts.append(context)
        self.injected_stage1.append(stage1_result)
        effective_stage1 = self.stage1_result if stage1_result is None else stage1_result
        return Result.ok(
            EvaluationResult(
                execution_id=context.execution_id,
                stage1_result=effective_stage1,
                final_approved=self.final_approved,
            )
        )


class _ExceptionPipeline:
    """Pipeline that raises on the Nth criterion (0-indexed)."""

    def __init__(self, *, fail_at: int, error: Exception) -> None:
        self._fail_at = fail_at
        self._error = error
        self.contexts: list[Any] = []

    async def evaluate(self, context: Any, *, stage1_result: Any = None) -> Any:
        del stage1_result
        self.contexts.append(context)
        index = int(context.execution_id.rsplit("_", 1)[-1]) - 1
        if index == self._fail_at:
            raise self._error
        semantic = SemanticResult(
            score=0.9,
            ac_compliance=True,
            goal_alignment=0.9,
            drift_score=0.1,
            uncertainty=0.1,
            reasoning=f"criterion {index + 1} passed",
            reward_hacking_risk=0.0,
            questions_used=(f"Does AC {index + 1} hold?",),
            evidence=(f"evidence-{index + 1}",),
        )
        return Result.ok(
            EvaluationResult(
                execution_id=context.execution_id,
                stage2_result=semantic,
                final_approved=True,
            )
        )


def _seed(*criteria: AcceptanceCriterionSpec) -> Seed:
    return Seed(
        metadata=SeedMetadata(seed_id="seed_regression"),
        acceptance_criteria=criteria,
        goal="Ship the requested behavior",
        constraints=("Preserve compatibility",),
        ontology_schema=OntologySchema(
            name="Regression",
            description="Regression-test ontology",
        ),
    )


@pytest.mark.asyncio
async def test_fallback_evaluates_each_acceptance_criterion_independently() -> None:
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior"),
        AcceptanceCriterionSpec(description="third behavior"),
    )
    pipeline = _Pipeline((True, False, True))

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report without legacy task markers",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert [context.current_ac for context in pipeline.contexts] == [
        "first behavior",
        "second behavior",
        "third behavior",
    ]
    assert [context.current_ac_spec for context in pipeline.contexts] == list(
        seed.acceptance_criteria
    )
    assert [result.passed for result in summary.ac_results] == [True, False, True]
    assert summary.final_approved is False
    assert "1/3 acceptance criteria failed" in (summary.failure_reason or "")
    assert "AC 2 (second behavior)" in (summary.failure_reason or "")
    assert "first behavior" not in (summary.failure_reason or "")


@pytest.mark.asyncio
async def test_fallback_approves_only_when_every_criterion_passes() -> None:
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior"),
    )

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report without legacy task markers",
        artifact_bundle=None,
        pipeline=_Pipeline((True, True)),
    )

    assert summary.final_approved is True
    assert summary.approval_status == "approved"
    assert summary.failure_reason is None
    assert all(result.verdict_is_authoritative for result in summary.ac_results)


def test_prose_only_acceptance_contract_emits_structural_warning() -> None:
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior"),
    )

    with capture_logs() as logs:
        warn_if_seed_has_no_success_contract(seed)

    assert logs == [
        {
            "event": "evolution.acceptance_contract.unstructured",
            "log_level": "warning",
            "seed_id": "seed_regression",
            "acceptance_criteria": 2,
            "consequence": (
                "no AC declares verify_command, expected_artifacts, or output_assertion; "
                "evolve has no fixed structured verification anchor"
            ),
        }
    ]


def test_any_structured_contract_suppresses_structural_warning() -> None:
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior", verify_command="pytest -q"),
    )

    with capture_logs() as logs:
        warn_if_seed_has_no_success_contract(seed)

    # With the mixed-contract diagnostic, an unanchored criterion is reported.
    assert len(logs) == 1
    assert logs[0]["event"] == "evolution.acceptance_contract.mixed"
    assert logs[0]["structured_count"] == 1
    assert logs[0]["unanchored_count"] == 1
    assert logs[0]["unanchored_indices"] == [0]


def test_fully_structured_contract_suppresses_all_warnings() -> None:
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior", verify_command="pytest -q"),
        AcceptanceCriterionSpec(description="second behavior", verify_command="make test"),
    )

    with capture_logs() as logs:
        warn_if_seed_has_no_success_contract(seed)

    assert logs == []


@pytest.mark.asyncio
async def test_stage1_failure_never_becomes_authoritative_semantic_verdict() -> None:
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior"),
    )
    stage1_failure = MechanicalResult(passed=False, checks=())
    pipeline = _NoSemanticPipeline(final_approved=False, stage1_result=stage1_failure)

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report without legacy task markers",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert pipeline.injected_stage1 == [None, stage1_failure]
    assert summary.final_approved is False
    assert [result.ac_verdict_state for result in summary.ac_results] == [
        "not_evaluated",
        "not_evaluated",
    ]
    assert [result.rendered_verdict for result in summary.ac_results] == [
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert not any(result.verdict_is_authoritative for result in summary.ac_results)
    assert "AC 1 (first behavior): semantic evaluation did not run" in (
        summary.failure_reason or ""
    )


@pytest.mark.asyncio
async def test_no_stage2_result_cannot_mint_authoritative_pass() -> None:
    seed = _seed(AcceptanceCriterionSpec(description="first behavior"))

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report without legacy task markers",
        artifact_bundle=None,
        pipeline=_NoSemanticPipeline(final_approved=True),
    )

    assert summary.final_approved is False
    assert summary.approval_status == "rejected"
    assert summary.ac_results[0].passed is False
    assert summary.ac_results[0].ac_verdict_state == "not_evaluated"
    assert summary.ac_results[0].rendered_verdict == "NOT_EVALUATED"
    assert summary.ac_results[0].verdict_is_authoritative is False


@pytest.mark.asyncio
async def test_zero_ac_seed_fails_closed_without_synthetic_coverage() -> None:
    seed = _seed()
    pipeline = _Pipeline(())

    with capture_logs() as logs:
        warn_if_seed_has_no_success_contract(seed)
    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report without legacy task markers",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert pipeline.contexts == []
    assert summary.final_approved is False
    assert summary.ac_results == ()
    assert summary.approval_status == "rejected"
    assert summary.failure_reason == (
        "Seed has no acceptance criteria; semantic evaluation cannot establish coverage"
    )
    assert logs[0]["event"] == "evolution.acceptance_contract.unstructured"
    assert logs[0]["acceptance_criteria"] == 0
    assert logs[0]["consequence"] == (
        "the Seed declares no acceptance criteria; evolve cannot establish coverage"
    )


# ────────────────────────────────────────────────────────────────────────────
# Regressions: evidence-free semantic PASS must never become authoritative
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evidence_free_semantic_pass_cannot_become_authoritative() -> None:
    """A semantic PASS with empty questions and evidence must not mint authority."""
    seed = _seed(AcceptanceCriterionSpec(description="first behavior"))
    pipeline = _EvidenceFreePipeline(questions_used=(), evidence=())

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is False
    assert summary.approval_status == "rejected"
    ac = summary.ac_results[0]
    assert ac.passed is False
    assert ac.verdict_is_authoritative is False
    assert ac.authoritative_pass is False
    assert ac.ac_verdict_state == "not_evaluated"
    assert ac.rendered_verdict == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_whitespace_only_evidence_cannot_mint_authoritative_pass() -> None:
    """Whitespace-only strings in questions_used and evidence are not meaningful."""
    seed = _seed(AcceptanceCriterionSpec(description="first behavior"))
    pipeline = _EvidenceFreePipeline(
        questions_used=("   ", "\t", "\n"),
        evidence=("  ", ""),
    )

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is False
    ac = summary.ac_results[0]
    assert ac.passed is False
    assert ac.verdict_is_authoritative is False
    assert ac.authoritative_pass is False
    assert ac.rendered_verdict == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_questions_only_without_evidence_is_non_authoritative() -> None:
    """Having questions but no evidence is one-sided and non-authoritative."""
    seed = _seed(AcceptanceCriterionSpec(description="first behavior"))
    pipeline = _EvidenceFreePipeline(
        questions_used=("Does the function handle edge cases?",),
        evidence=(),
    )

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is False
    ac = summary.ac_results[0]
    assert ac.passed is False
    assert ac.verdict_is_authoritative is False
    assert ac.authoritative_pass is False
    assert ac.rendered_verdict == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_evidence_only_without_questions_is_non_authoritative() -> None:
    """Having evidence but no verification questions is one-sided and non-authoritative."""
    seed = _seed(AcceptanceCriterionSpec(description="first behavior"))
    pipeline = _EvidenceFreePipeline(
        questions_used=(),
        evidence=("file src/main.py contains handler",),
    )

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is False
    ac = summary.ac_results[0]
    assert ac.passed is False
    assert ac.verdict_is_authoritative is False
    assert ac.authoritative_pass is False
    assert ac.rendered_verdict == "INSUFFICIENT_EVIDENCE"


@pytest.mark.asyncio
async def test_full_evidence_and_questions_grants_authority() -> None:
    """When both questions and evidence are present, authority is granted."""
    seed = _seed(AcceptanceCriterionSpec(description="first behavior"))
    pipeline = _EvidenceFreePipeline(
        questions_used=("Does the handler validate input?",),
        evidence=("src/handler.py:42 performs input validation",),
    )

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is True
    assert summary.approval_status == "approved"
    ac = summary.ac_results[0]
    assert ac.passed is True
    assert ac.verdict_is_authoritative is True
    assert ac.authoritative_pass is True
    assert ac.ac_verdict_state == "evaluated"
    assert ac.rendered_verdict == "PASS"


# ────────────────────────────────────────────────────────────────────────────
# Regressions: exception aggregation / first-error fail-closed policy
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_first_criterion_exception_returns_rejected_summary() -> None:
    """An exception on the first criterion fails closed immediately."""
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior"),
    )
    pipeline = _ExceptionPipeline(fail_at=0, error=RuntimeError("LLM timeout"))

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is False
    assert summary.approval_status == "rejected"
    assert "AC 1 evaluation raised RuntimeError" in (summary.failure_reason or "")
    assert "LLM timeout" in (summary.failure_reason or "")
    assert summary.ac_results == ()


@pytest.mark.asyncio
async def test_later_criterion_exception_returns_rejected_summary() -> None:
    """An exception on a gathered criterion fails closed with diagnostics."""
    seed = _seed(
        AcceptanceCriterionSpec(description="first behavior"),
        AcceptanceCriterionSpec(description="second behavior"),
        AcceptanceCriterionSpec(description="third behavior"),
    )
    pipeline = _ExceptionPipeline(fail_at=2, error=ValueError("parse failed"))

    summary = await evaluate_seed_criteria(
        seed=seed,
        artifact="worker report",
        artifact_bundle=None,
        pipeline=pipeline,
    )

    assert summary.final_approved is False
    assert summary.approval_status == "rejected"
    assert "AC 3 evaluation raised ValueError" in (summary.failure_reason or "")
    assert "parse failed" in (summary.failure_reason or "")


# ────────────────────────────────────────────────────────────────────────────
# Regressions: mixed structured/prose warning diagnostic
# ────────────────────────────────────────────────────────────────────────────


def test_mixed_structured_prose_seed_emits_mixed_warning() -> None:
    """A Seed with some structured and some prose-only criteria emits mixed warning."""
    seed = _seed(
        AcceptanceCriterionSpec(description="prose criterion A"),
        AcceptanceCriterionSpec(description="structured criterion", verify_command="pytest -q"),
        AcceptanceCriterionSpec(description="prose criterion B"),
    )

    with capture_logs() as logs:
        warn_if_seed_has_no_success_contract(seed)

    assert len(logs) == 1
    assert logs[0]["event"] == "evolution.acceptance_contract.mixed"
    assert logs[0]["structured_count"] == 1
    assert logs[0]["unanchored_count"] == 2
    assert logs[0]["unanchored_indices"] == [0, 2]
    assert "2/3 AC(s)" in logs[0]["consequence"]
