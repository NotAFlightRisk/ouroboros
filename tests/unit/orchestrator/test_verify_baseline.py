"""Stage 1 negative control: a contract must fail before the work exists.

The falsification cases from the evidential-force RFC (#2179), as unit-level
analogues: (a) a contract that already passes on the pristine workspace loses
the discriminating stamp and its recovery authority, and (b) a contract that
fails at baseline and passes after the work stamps ``discriminating_pass`` and
keeps recovery. Everything unknown — probe off, snapshot rejected, timeout —
grants nothing and revokes nothing.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ouroboros.core.seed import (
    AcceptanceCriterionSpec,
    OntologySchema,
    Seed,
    SeedMetadata,
)
from ouroboros.orchestrator.parallel_executor import (
    ACExecutionOutcome,
    ACExecutionResult,
    ParallelACExecutor,
    _deserialize_verify_gate_outcome,
    _serialize_verify_gate_outcome,
    _VerifyGateOutcome,
)
from ouroboros.orchestrator.verify_baseline import (
    BASELINE_ALL_PASS_EVENT,
    BASELINE_PROBE_EVENT,
    NON_DISCRIMINATING_EVENT,
    VERDICT_TIER_DISCRIMINATING_PASS,
    VERDICT_TIER_PASS,
    ACBaselineRecord,
    VerifyBaseline,
    baseline_snapshot,
    establish_verify_baseline,
    restore_verify_baseline,
    verify_baseline_checkpoint_state,
)
from ouroboros.orchestrator.verify_gate_outcome import (
    _revalidate_cached_verify_gate_outcome,
)


class _StubAdapter:
    def __init__(self, working_directory: str) -> None:
        self.runtime_backend = "claude"
        self.self_governs_rate_limit = True
        self.working_directory = working_directory
        self.permission_mode = "acceptEdits"


def _make_executor(
    *,
    working_directory: str,
    verify_baseline_probe: str = "observe",
    run_verify_commands: bool = True,
    verify_command_timeout_seconds: int = 30,
) -> ParallelACExecutor:
    return ParallelACExecutor(
        adapter=_StubAdapter(working_directory),
        event_store=AsyncMock(),
        console=MagicMock(),
        enable_decomposition=False,
        run_verify_commands=run_verify_commands,
        verify_command_timeout_seconds=verify_command_timeout_seconds,
        verify_baseline_probe=verify_baseline_probe,
    )


def _seed_with_specs(*specs: AcceptanceCriterionSpec | str) -> Seed:
    return Seed(
        goal="baseline negative control",
        acceptance_criteria=specs,
        ontology_schema=OntologySchema(name="n", description="d"),
        metadata=SeedMetadata(ambiguity_score=0.05),
    )


def _emitted(executor: ParallelACExecutor, event_type: str) -> list[Any]:
    return [
        call.args[0]
        for call in executor._event_store.append.await_args_list
        if call.args[0].type == event_type
    ]


# ---------------------------------------------------------------------------
# establish_verify_baseline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preexisting_pass_is_recorded_as_baseline_true(tmp_path: Any) -> None:
    """Falsification (a): a contract the pristine workspace already satisfies."""
    (tmp_path / "pre.txt").write_text("already here")
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(
            description="ac",
            verify_command="test -f pre.txt",
            expected_artifacts=("pre.txt",),
        )
    )

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    record = baseline.records[0]
    assert record.probed is True
    assert record.baseline_verdict is True
    assert record.preexisting_artifacts == ("pre.txt",)
    probe_events = _emitted(executor, BASELINE_PROBE_EVENT)
    assert len(probe_events) == 1
    assert probe_events[0].data["baseline_verdict"] is True
    assert probe_events[0].data["preexisting_artifacts"] == ["pre.txt"]
    # Every probed contract passed -> the broken-seed finding fires.
    assert baseline.all_contracts_baseline_pass is True
    assert len(_emitted(executor, BASELINE_ALL_PASS_EVENT)) == 1
    assert executor._console.print.called


@pytest.mark.asyncio
async def test_absent_target_is_recorded_as_baseline_false(tmp_path: Any) -> None:
    """Falsification (b): the contract fails before the work exists."""
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(description="ac", verify_command="test -f made.txt")
    )

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    record = baseline.records[0]
    assert record.probed is True
    assert record.baseline_verdict is False
    assert baseline.all_contracts_baseline_pass is False
    assert _emitted(executor, BASELINE_ALL_PASS_EVENT) == []


@pytest.mark.asyncio
async def test_destructive_contract_cannot_reach_siblings_or_live_tree(tmp_path: Any) -> None:
    """rm-bearing contracts get a fresh copy each; the live tree is untouched."""
    (tmp_path / "shared.txt").write_text("payload")
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(description="destroyer", verify_command="rm shared.txt"),
        AcceptanceCriterionSpec(description="reader", verify_command="test -f shared.txt"),
    )

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    # AC1 still sees shared.txt: AC0's rm ran in its own disposable copy.
    assert baseline.records[1].baseline_verdict is True
    # And the live workspace was never touched.
    assert (tmp_path / "shared.txt").exists()


@pytest.mark.asyncio
async def test_timeout_probe_is_unknown_not_a_fail(tmp_path: Any) -> None:
    """An infrastructure hiccup must never mint a discrimination certificate."""
    executor = _make_executor(working_directory=str(tmp_path), verify_command_timeout_seconds=1)
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(
            description="hung",
            verify_command='python3 -c "import time; time.sleep(10)"',
        )
    )

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    record = baseline.records[0]
    assert record.probed is False
    assert record.baseline_verdict is None
    assert baseline.all_contracts_baseline_pass is False


@pytest.mark.asyncio
async def test_probe_off_probes_nothing(tmp_path: Any) -> None:
    executor = _make_executor(working_directory=str(tmp_path), verify_baseline_probe="off")
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is None
    assert _emitted(executor, BASELINE_PROBE_EVENT) == []


def test_snapshot_rejects_escaping_symlink(tmp_path: Any) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("live filesystem")
    os.symlink(outside, workspace / "leak")

    with baseline_snapshot(str(workspace)) as snapshot:
        assert snapshot is None


def test_snapshot_excludes_git_and_keeps_untracked(tmp_path: Any) -> None:
    workspace = tmp_path / "ws"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (workspace / "untracked.txt").write_text("kept")

    with baseline_snapshot(str(workspace)) as snapshot:
        assert snapshot is not None
        assert not (os.path.exists(os.path.join(snapshot, ".git")))
        assert os.path.exists(os.path.join(snapshot, "untracked.txt"))


# ---------------------------------------------------------------------------
# settle via _apply_verify_gate — tier stamping, withhold, revocation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_baseline_pass_withholds_stamp_but_never_fails_the_ac(tmp_path: Any) -> None:
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = VerifyBaseline(
        records={0: ACBaselineRecord(probed=True, baseline_verdict=True)},
        all_contracts_baseline_pass=True,
    )
    result = ACExecutionResult(
        ac_index=0, ac_content="ac", success=True, outcome=ACExecutionOutcome.SUCCEEDED
    )

    settled = await executor._apply_verify_gate(
        seed=seed, ac_index=0, result=result, session_id="s", execution_id="e"
    )

    # Withhold-only: the pass stands, only the discriminating tier is withheld.
    assert settled.success is True
    assert settled.verify_gate_outcome.verdict_tier == VERDICT_TIER_PASS
    withholds = _emitted(executor, NON_DISCRIMINATING_EVENT)
    assert len(withholds) == 1
    assert withholds[0].data["recovery_revoked"] is False
    assert withholds[0].data["mode"] == "observe"
    assert withholds[0].data["evidence_exemption_changed"] is False


@pytest.mark.asyncio
async def test_baseline_pass_revokes_recovery_authority(tmp_path: Any) -> None:
    """Falsification (a), third clause: execution.verify.recovered unreachable."""
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = VerifyBaseline(
        records={0: ACBaselineRecord(probed=True, baseline_verdict=True)},
        all_contracts_baseline_pass=True,
    )
    result = ACExecutionResult(
        ac_index=0,
        ac_content="ac",
        success=False,
        error="worker reported failure",
        outcome=ACExecutionOutcome.FAILED,
    )

    settled = await executor._apply_verify_gate(
        seed=seed, ac_index=0, result=result, session_id="s", execution_id="e"
    )

    assert settled.success is False
    assert settled.error == "worker reported failure"
    assert _emitted(executor, "execution.verify.recovered") == []
    withholds = _emitted(executor, NON_DISCRIMINATING_EVENT)
    assert len(withholds) == 1
    assert withholds[0].data["recovery_revoked"] is True
    assert withholds[0].data["prior_error"] == "worker reported failure"


@pytest.mark.asyncio
async def test_discriminating_contract_stamps_and_keeps_recovery(tmp_path: Any) -> None:
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = VerifyBaseline(
        records={0: ACBaselineRecord(probed=True, baseline_verdict=False)},
        all_contracts_baseline_pass=False,
    )
    failed = ACExecutionResult(
        ac_index=0,
        ac_content="ac",
        success=False,
        error="runtime false negative",
        outcome=ACExecutionOutcome.FAILED,
    )

    recovered = await executor._apply_verify_gate(
        seed=seed, ac_index=0, result=failed, session_id="s", execution_id="e"
    )

    assert recovered.success is True
    assert recovered.verify_gate_outcome.verdict_tier == VERDICT_TIER_DISCRIMINATING_PASS
    assert len(_emitted(executor, "execution.verify.recovered")) == 1
    assert _emitted(executor, NON_DISCRIMINATING_EVENT) == []


@pytest.mark.asyncio
async def test_unknown_baseline_behaves_exactly_like_before(tmp_path: Any) -> None:
    """Probe off / not probed -> plain pass tier, recovery preserved."""
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = VerifyBaseline(
        records={0: ACBaselineRecord(probed=False, baseline_verdict=None)},
        all_contracts_baseline_pass=False,
    )
    failed = ACExecutionResult(
        ac_index=0,
        ac_content="ac",
        success=False,
        error="runtime false negative",
        outcome=ACExecutionOutcome.FAILED,
    )

    recovered = await executor._apply_verify_gate(
        seed=seed, ac_index=0, result=failed, session_id="s", execution_id="e"
    )

    assert recovered.success is True
    assert recovered.verify_gate_outcome.verdict_tier == VERDICT_TIER_PASS
    assert len(_emitted(executor, "execution.verify.recovered")) == 1
    assert _emitted(executor, NON_DISCRIMINATING_EVENT) == []


# ---------------------------------------------------------------------------
# verdict_tier serialization
# ---------------------------------------------------------------------------


def test_verdict_tier_round_trips_and_legacy_defaults_to_none() -> None:
    for tier in (VERDICT_TIER_PASS, VERDICT_TIER_DISCRIMINATING_PASS):
        outcome = _VerifyGateOutcome(passed=True, reason=None, output_tail="", verdict_tier=tier)
        payload = _serialize_verify_gate_outcome(outcome)
        assert payload is not None
        restored = _deserialize_verify_gate_outcome(payload)
        assert restored is not None
        assert restored.verdict_tier == tier

    legacy = {
        "passed": True,
        "reason": None,
        "output_tail": "",
        "missing_artifacts": [],
        "workspace_mutated": False,
        "workspace_digest": None,
        "environment_unverifiable": False,
    }
    restored = _deserialize_verify_gate_outcome(legacy)
    assert restored is not None
    assert restored.verdict_tier is None


def test_verdict_tier_rejects_invalid_vocabulary_and_failed_outcomes() -> None:
    with pytest.raises(RuntimeError):
        _serialize_verify_gate_outcome(
            _VerifyGateOutcome(passed=True, reason=None, output_tail="", verdict_tier="great")
        )
    with pytest.raises(RuntimeError):
        _serialize_verify_gate_outcome(
            _VerifyGateOutcome(
                passed=False, reason="r", output_tail="", verdict_tier=VERDICT_TIER_PASS
            )
        )
    payload = {
        "passed": True,
        "reason": None,
        "output_tail": "",
        "missing_artifacts": [],
        "workspace_mutated": False,
        "workspace_digest": None,
        "environment_unverifiable": False,
        "verdict_tier": "great",
    }
    assert _deserialize_verify_gate_outcome(payload) is None


def test_revalidation_preserves_tier_on_pass_and_clears_it_on_demotion(tmp_path: Any) -> None:
    spec = AcceptanceCriterionSpec(
        description="ac", verify_command="test -d .", expected_artifacts=("out.txt",)
    )
    outcome = _VerifyGateOutcome(
        passed=True,
        reason=None,
        output_tail="",
        verdict_tier=VERDICT_TIER_DISCRIMINATING_PASS,
    )

    (tmp_path / "out.txt").write_text("present")
    unchanged = _revalidate_cached_verify_gate_outcome(
        spec=spec, cwd=str(tmp_path), outcome=outcome
    )
    assert unchanged.verdict_tier == VERDICT_TIER_DISCRIMINATING_PASS

    (tmp_path / "out.txt").unlink()
    demoted = _revalidate_cached_verify_gate_outcome(spec=spec, cwd=str(tmp_path), outcome=outcome)
    assert demoted.passed is False
    assert demoted.verdict_tier is None


# ---------------------------------------------------------------------------
# checkpoint round-trip + resume
# ---------------------------------------------------------------------------


def test_baseline_checkpoint_round_trip() -> None:
    baseline = VerifyBaseline(
        records={
            0: ACBaselineRecord(
                probed=True,
                baseline_verdict=True,
                preexisting_artifacts=("docs/out.md",),
                reason=None,
            ),
            2: ACBaselineRecord(probed=False, baseline_verdict=None, reason="timed out after 1s"),
        },
        all_contracts_baseline_pass=False,
    )

    state = verify_baseline_checkpoint_state(baseline)
    assert state is not None
    restored = restore_verify_baseline(state)

    assert restored == baseline
    assert verify_baseline_checkpoint_state(None) is None


@pytest.mark.parametrize(
    "payload",
    [
        "not a mapping",
        {},
        {"all_contracts_baseline_pass": False, "records": {"x": {}}},
        {"all_contracts_baseline_pass": False, "records": {"0": {"probed": True}}},
        {
            "all_contracts_baseline_pass": False,
            # probed=True must carry a verdict; a mismatch is corruption.
            "records": {
                "0": {
                    "probed": True,
                    "baseline_verdict": None,
                    "preexisting_artifacts": [],
                    "reason": None,
                }
            },
        },
        {
            "all_contracts_baseline_pass": False,
            "records": {
                "0": {
                    "probed": True,
                    "baseline_verdict": True,
                    "preexisting_artifacts": [],
                    "reason": "x" * 501,
                }
            },
        },
    ],
)
def test_malformed_baseline_checkpoint_becomes_unknown(payload: Any) -> None:
    assert restore_verify_baseline(payload) is None


@pytest.mark.asyncio
async def test_restored_records_still_revoke_recovery(tmp_path: Any) -> None:
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    state = verify_baseline_checkpoint_state(
        VerifyBaseline(
            records={0: ACBaselineRecord(probed=True, baseline_verdict=True)},
            all_contracts_baseline_pass=True,
        )
    )
    executor._verify_baseline = restore_verify_baseline(state)
    failed = ACExecutionResult(
        ac_index=0,
        ac_content="ac",
        success=False,
        error="worker reported failure",
        outcome=ACExecutionOutcome.FAILED,
    )

    settled = await executor._apply_verify_gate(
        seed=seed, ac_index=0, result=failed, session_id="s", execution_id="e"
    )

    assert settled.success is False
    assert _emitted(executor, "execution.verify.recovered") == []


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_execution_config_defaults_to_observe_and_rejects_unknown_modes() -> None:
    from pydantic import ValidationError

    from ouroboros.config.models import ExecutionConfig

    assert ExecutionConfig().verify_baseline_probe == "observe"
    assert ExecutionConfig(verify_baseline_probe="off").verify_baseline_probe == "off"
    with pytest.raises(ValidationError):
        ExecutionConfig(verify_baseline_probe="enforce")
