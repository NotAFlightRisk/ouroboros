"""Formal-evaluation to Ralph convergence chaining tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from ouroboros.core.lineage import ACResult, EvaluationSummary
from ouroboros.core.seed import OntologySchema, Seed, SeedMetadata
from ouroboros.core.types import Result
from ouroboros.evolution.evaluation_coverage import validate_seed_ac_coverage
from ouroboros.evolution.loop import EvolutionaryLoop, EvolutionaryLoopConfig
from ouroboros.evolution.loop_support import planned_evolve_generation
from ouroboros.evolution.reflect import ReflectOutput
from ouroboros.evolution.wonder import WonderOutput
from ouroboros.mcp.job_manager import JobManager, JobStatus
from ouroboros.mcp.tools import evaluation_handlers
from ouroboros.mcp.tools.evaluate_ralph_chain import (
    ac_results_from_checklist,
    evaluation_summary_from_eval_meta,
    mint_chain_lineage_id,
)
from ouroboros.mcp.tools.evaluation_handlers import StartEvaluateHandler
from ouroboros.mcp.tools.evolution_handlers import EvolveStepHandler
from ouroboros.mcp.tools.qa import QAHandler
from ouroboros.mcp.tools.ralph_handlers import StartRalphHandler
from ouroboros.mcp.types import ContentType, MCPContentItem, MCPToolResult
from ouroboros.persistence.event_store import EventStore


@pytest.fixture
async def event_store():
    store = EventStore("sqlite+aiosqlite:///:memory:")
    await store.initialize()
    yield store
    await store.close()


def _seed() -> Seed:
    return Seed(
        goal="Build a convergent artifact",
        acceptance_criteria=("passing AC", "failing AC"),
        ontology_schema=OntologySchema(name="Artifact", description="Artifact domain"),
        metadata=SeedMetadata(seed_id="seed-chain", ambiguity_score=0.1),
    )


def _seed_yaml(seed: Seed | None = None) -> str:
    return yaml.safe_dump((seed or _seed()).to_dict(), sort_keys=False)


def _rejected_result() -> MCPToolResult:
    return MCPToolResult(
        content=(MCPContentItem(type=ContentType.TEXT, text="REJECTED"),),
        meta={
            "session_id": "orch-chain-1234",
            "final_approved": False,
            "multi_ac": True,
            "pass_rate": 0.5,
            "run_feedback": ["failing AC: output is missing"],
            "checklist": [
                {
                    "ac_text": "passing AC",
                    "passed": True,
                    "reasoning": "implemented",
                    "evidence": ["artifact.txt exists"],
                    "questions_used": [],
                    "failure_reason": None,
                },
                {
                    "ac_text": "failing AC",
                    "passed": False,
                    "reasoning": "output missing",
                    "evidence": [],
                    "questions_used": ["Does output exist?"],
                    "failure_reason": "not found",
                },
            ],
        },
    )


async def _wait_terminal(manager: JobManager, job_id: str):
    for _ in range(200):
        snapshot = await manager.get_snapshot(job_id)
        if snapshot.is_terminal:
            return snapshot
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish")


class _StaticEvaluateHandler:
    def __init__(self, result: MCPToolResult) -> None:
        self.result = result

    async def handle(self, _: dict[str, Any]) -> Result[MCPToolResult, Any]:
        return Result.ok(self.result)


async def test_rejected_evaluation_seeds_gen1_and_enqueues_ralph(
    event_store: EventStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class FakeStartRalphHandler:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, arguments: dict[str, Any]) -> Result[MCPToolResult, Any]:
            calls.append(arguments)
            return Result.ok(MCPToolResult(meta={"job_id": "job_ralph_chain"}))

    monkeypatch.setattr(evaluation_handlers, "get_auto_evolve_enabled", lambda: True)
    monkeypatch.setattr(evaluation_handlers, "get_auto_evolve_max_generations", lambda: 3)
    manager = JobManager(event_store)
    handler = StartEvaluateHandler(
        evaluate_handler=_StaticEvaluateHandler(_rejected_result()),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=manager,
        start_ralph_handler=FakeStartRalphHandler(),  # type: ignore[arg-type]
    )
    arguments = {
        "session_id": "orch-chain-1234",
        "artifact": "partial artifact",
        "seed_content": _seed_yaml(),
        "working_dir": str(tmp_path),
    }

    started = await handler.handle(arguments)
    assert started.is_ok
    snapshot = await _wait_terminal(manager, started.value.meta["job_id"])

    assert snapshot.status == JobStatus.COMPLETED
    assert snapshot.result_meta["final_approved"] is False
    assert snapshot.result_meta["chained_ralph_job_id"] == "job_ralph_chain"
    lineage_id = snapshot.result_meta["chained_ralph_lineage_id"]
    assert lineage_id == "ralph-seed-chain-orch-cha"
    assert snapshot.result_meta["chained_ralph_max_generations"] == 3
    assert calls == [
        {
            "lineage_id": lineage_id,
            "execute": True,
            "parallel": True,
            "skip_qa": False,
            "project_dir": str(tmp_path),
            "max_generations": 3,
        }
    ]
    events = await event_store.replay_lineage(lineage_id)
    assert [event.type for event in events] == [
        "lineage.created",
        "lineage.generation.completed",
    ]
    summary = events[-1].data["evaluation_summary"]
    assert summary["ac_results"][0]["rendered_verdict"] == "PASS"
    assert summary["ac_results"][1]["rendered_verdict"] == "FAIL"
    assert await planned_evolve_generation(event_store, lineage_id, execute=True) == 2


async def test_real_rejection_to_ralph_completes_generation_two(
    event_store: EventStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real production boundary that constructor-mocked tests missed."""

    seed = _seed()
    generated_seed = Seed.from_dict(
        {
            **seed.to_dict(),
            "metadata": {
                **seed.metadata.model_dump(mode="json"),
                "seed_id": "seed-chain-gen2",
                "parent_seed_id": seed.metadata.seed_id,
            },
        }
    )
    wonder_engine = MagicMock()
    wonder_engine.wonder = AsyncMock(
        return_value=Result.ok(
            WonderOutput(
                questions=("What output repairs failing AC?",),
                grounded_questions=(),
                should_continue=True,
            )
        )
    )
    reflect_engine = MagicMock()
    reflect_engine.reflect = AsyncMock(
        return_value=Result.ok(
            ReflectOutput(
                refined_goal=seed.goal,
                refined_constraints=seed.constraints,
                refined_acs=seed.acceptance_criteria,
                settled_ac_indices=(0,),
                ontology_mutations=(),
                reasoning="Repair only the failed AC",
            )
        )
    )
    seed_generator = MagicMock()
    seed_generator.generate_from_reflect = MagicMock(return_value=Result.ok(generated_seed))
    executor_calls: list[dict[str, Any]] = []
    evaluator_calls: list[str | None] = []

    async def executor(_seed: Seed, **kwargs: Any) -> str:
        executor_calls.append(kwargs)
        return "generation 2 repaired output"

    async def evaluator(_seed: Seed, output: str | None, **_: Any) -> EvaluationSummary:
        evaluator_calls.append(output)
        return EvaluationSummary(
            final_approved=True,
            highest_stage_passed=3,
            score=1.0,
            approval_status="approved",
            execution_completion_status="completed",
            ac_results=(
                ACResult(
                    ac_index=0,
                    ac_content="passing AC",
                    passed=True,
                    score=1.0,
                    evidence="boundary reverified",
                    verification_method="integration_test",
                    final_verdict="pass",
                    rendered_verdict="PASS",
                ),
                ACResult(
                    ac_index=1,
                    ac_content="failing AC",
                    passed=True,
                    score=1.0,
                    evidence="repaired output exists",
                    verification_method="integration_test",
                    final_verdict="pass",
                    rendered_verdict="PASS",
                ),
            ),
        )

    executor.frugality_provider_tracking = True  # type: ignore[attr-defined]
    evaluator.frugality_provider_tracking = True  # type: ignore[attr-defined]
    loop = EvolutionaryLoop(
        event_store=event_store,
        config=EvolutionaryLoopConfig(min_generations=1, outcome_gate_enabled=False),
        wonder_engine=wonder_engine,
        reflect_engine=reflect_engine,
        seed_generator=seed_generator,
        executor=executor,
        evaluator=evaluator,
    )
    manager = JobManager(event_store)
    evolve_step = EvolveStepHandler(evolutionary_loop=loop, event_store=event_store)
    start_ralph = StartRalphHandler(
        evolve_handler=evolve_step,
        event_store=event_store,
        job_manager=manager,
    )
    handler = StartEvaluateHandler(
        event_store=event_store,
        job_manager=manager,
        start_ralph_handler=start_ralph,
    )

    async def passing_qa(
        _self: QAHandler,
        _arguments: dict[str, Any],
    ) -> Result[MCPToolResult, Any]:
        return Result.ok(
            MCPToolResult(
                content=(MCPContentItem(type=ContentType.TEXT, text="QA PASS"),),
                meta={
                    "verdict": "pass",
                    "passed": True,
                    "score": 1.0,
                    "pass_threshold": 0.8,
                },
            )
        )

    monkeypatch.setattr(QAHandler, "handle", passing_qa)
    monkeypatch.setattr(evaluation_handlers, "get_auto_evolve_max_generations", lambda: 2)

    chained = await handler._enqueue_chained_ralph(
        _rejected_result(),
        session_id="orch-real-boundary",
        arguments={"seed_content": _seed_yaml(seed), "working_dir": str(tmp_path)},
    )
    ralph_job_id = chained.meta["chained_ralph_job_id"]
    terminal = await _wait_terminal(manager, ralph_job_id)

    assert terminal.status == JobStatus.COMPLETED
    assert "EvolutionaryLoop not configured" not in (terminal.result_text or "")
    assert terminal.result_meta["stop_reason"] == "qa passed"
    assert terminal.result_meta["generations"] == [2]
    assert executor_calls
    assert evaluator_calls == ["generation 2 repaired output"]
    events = await event_store.replay_lineage(chained.meta["chained_ralph_lineage_id"])
    completed = [event for event in events if event.type == "lineage.generation.completed"]
    assert [event.data["generation_number"] for event in completed] == [1, 2]
    assert completed[-1].data["active_ac_indices"] == [1]
    assert completed[-1].data["frozen_ac_indices"] == [0]


async def test_server_composition_reuses_configured_chain_handlers(
    event_store: EventStore,
    tmp_path: Path,
) -> None:
    from ouroboros.mcp.server.adapter import create_ouroboros_server

    with (
        patch("ouroboros.orchestrator.create_agent_runtime", return_value=MagicMock()),
        patch("ouroboros.providers.create_llm_adapter", return_value=MagicMock()),
    ):
        server = create_ouroboros_server(
            event_store=event_store,
            state_dir=tmp_path / "state",
            runtime_backend="codex",
            llm_backend="claude_code",
        )

    start_execute = server._tool_handlers["ouroboros_start_execute_seed"]
    start_evaluate = server._tool_handlers["ouroboros_start_evaluate"]
    start_ralph = server._tool_handlers["ouroboros_start_ralph"]
    evolve_step = server._tool_handlers["ouroboros_evolve_step"]

    assert start_execute.start_evaluate_handler is start_evaluate
    assert start_evaluate.start_ralph_handler is start_ralph
    assert start_ralph._evolve_handler is evolve_step
    assert evolve_step.evolutionary_loop is not None


@pytest.mark.parametrize("meta", [{"final_approved": True}, {}])
async def test_approved_or_unjudged_evaluation_does_not_chain(
    event_store: EventStore,
    monkeypatch: pytest.MonkeyPatch,
    meta: dict[str, Any],
) -> None:
    manager = JobManager(event_store)
    handler = StartEvaluateHandler(
        evaluate_handler=_StaticEvaluateHandler(MCPToolResult(meta=meta)),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=manager,
    )
    started = await handler.handle(
        {"session_id": "orch-approved", "artifact": "artifact", "seed_content": _seed_yaml()}
    )
    snapshot = await _wait_terminal(manager, started.value.meta["job_id"])
    assert "chained_ralph_job_id" not in snapshot.result_meta


@pytest.mark.parametrize(
    ("config_enabled", "override"),
    [(False, None), (True, False)],
)
async def test_auto_evolve_opt_out_matrix(
    event_store: EventStore,
    monkeypatch: pytest.MonkeyPatch,
    config_enabled: bool,
    override: bool | None,
) -> None:
    monkeypatch.setattr(
        evaluation_handlers,
        "get_auto_evolve_enabled",
        lambda: config_enabled,
    )
    manager = JobManager(event_store)
    handler = StartEvaluateHandler(
        evaluate_handler=_StaticEvaluateHandler(_rejected_result()),  # type: ignore[arg-type]
        event_store=event_store,
        job_manager=manager,
    )
    arguments: dict[str, Any] = {
        "session_id": "orch-optout",
        "artifact": "artifact",
        "seed_content": _seed_yaml(),
    }
    if override is not None:
        arguments["auto_evolve"] = override
    started = await handler.handle(arguments)
    snapshot = await _wait_terminal(manager, started.value.meta["job_id"])
    assert "chained_ralph_job_id" not in snapshot.result_meta
    assert await event_store.replay_lineage("ralph-seed-chain-orch-opt") == []


async def test_seed_unavailable_skips_chain_fail_open(event_store: EventStore) -> None:
    manager = JobManager(event_store)
    handler = StartEvaluateHandler(event_store=event_store, job_manager=manager)

    result = await handler._enqueue_chained_ralph(
        _rejected_result(),
        session_id="orch-no-seed",
        arguments={"seed_content": "not: [valid"},
    )

    assert result.meta["final_approved"] is False
    assert result.meta["chained_ralph_skipped"] == "seed_unavailable"
    assert "chained_ralph_job_id" not in result.meta


async def test_enqueue_error_keeps_rejection_and_gen1_snapshot(
    event_store: EventStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingRalph:
        def __init__(self, **_: Any) -> None:
            pass

        async def handle(self, _: dict[str, Any]) -> Result[MCPToolResult, Any]:
            raise RuntimeError("ralph enqueue exploded")

    handler = StartEvaluateHandler(
        event_store=event_store,
        job_manager=JobManager(event_store),
        start_ralph_handler=FailingRalph(),  # type: ignore[arg-type]
    )
    result = await handler._enqueue_chained_ralph(
        _rejected_result(),
        session_id="orch-error",
        arguments={"seed_content": _seed_yaml()},
    )

    assert result.meta["final_approved"] is False
    assert result.meta["chained_ralph_error"] == "ralph enqueue exploded"
    assert len(await event_store.replay_lineage("ralph-seed-chain-orch-err")) == 2


async def test_gen1_seed_is_idempotent_and_active_job_is_reused(
    event_store: EventStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = JobManager(event_store)
    monkeypatch.setattr(
        manager,
        "find_active_job_by_lineage",
        lambda *_args, **_kwargs: _async_value(SimpleNamespace(job_id="job_existing")),
    )
    handler = StartEvaluateHandler(event_store=event_store, job_manager=manager)
    arguments = {"seed_content": _seed_yaml()}

    first = await handler._enqueue_chained_ralph(
        _rejected_result(), session_id="orch-reuse", arguments=arguments
    )
    second = await handler._enqueue_chained_ralph(
        _rejected_result(), session_id="orch-reuse", arguments=arguments
    )

    assert first.meta["chained_ralph_job_id"] == "job_existing"
    assert second.meta["chained_ralph_job_id"] == "job_existing"
    events = await event_store.replay_lineage("ralph-seed-chain-orch-reu")
    assert [event.type for event in events] == [
        "lineage.created",
        "lineage.generation.completed",
    ]


async def _async_value(value: Any) -> Any:
    return value


def test_checklist_conversion_preserves_seed_coverage_and_not_evaluated_shape() -> None:
    seed = _seed()
    checklist = _rejected_result().meta["checklist"][:1]
    results = ac_results_from_checklist(seed, checklist)
    summary = evaluation_summary_from_eval_meta(seed, {"final_approved": False, "checklist": checklist})

    assert len(results) == 2
    assert results[1].ac_content == "failing AC"
    assert results[1].ac_verdict_state == "not_evaluated"
    assert results[1].rendered_verdict == "NOT_EVALUATED"
    assert validate_seed_ac_coverage(seed, summary, require_authoritative=False).complete is True


def test_single_ac_without_checklist_degrades_to_empty_ac_results() -> None:
    seed = Seed(
        goal="single",
        acceptance_criteria=("only AC",),
        ontology_schema=OntologySchema(name="Single", description="Single domain"),
        metadata=SeedMetadata(seed_id="single", ambiguity_score=0.1),
    )
    assert ac_results_from_checklist(seed, None) == ()
    assert mint_chain_lineage_id("single", "orch-123456") == "ralph-single-orch-123"
