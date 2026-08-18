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
from pathlib import Path
from types import SimpleNamespace
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

_UNSET: Any = object()


class _StubAdapter:
    def __init__(self, working_directory: str) -> None:
        self.runtime_backend = "claude"
        self.self_governs_rate_limit = True
        self.working_directory = working_directory
        self.permission_mode = "acceptEdits"


class _StubCheckpointStore:
    """A store whose saves land, unless told otherwise."""

    def __init__(self, *, saves: bool = True) -> None:
        self.saves = saves
        self.saved: list[Any] = []

    def save(self, checkpoint: Any) -> Any:
        self.saved.append(checkpoint)
        return SimpleNamespace(is_ok=self.saves, error=None if self.saves else "disk full")

    def load(self, seed_id: str) -> Any:
        return SimpleNamespace(is_ok=False, value=None, error="not found")


def _make_executor(
    *,
    working_directory: str,
    verify_baseline_probe: str = "observe",
    run_verify_commands: bool = True,
    verify_command_timeout_seconds: int = 30,
    checkpoint_store: Any | None = _UNSET,
) -> ParallelACExecutor:
    return ParallelACExecutor(
        adapter=_StubAdapter(working_directory),
        event_store=AsyncMock(),
        console=MagicMock(),
        enable_decomposition=False,
        run_verify_commands=run_verify_commands,
        verify_command_timeout_seconds=verify_command_timeout_seconds,
        verify_baseline_probe=verify_baseline_probe,
        checkpoint_store=(
            _StubCheckpointStore() if checkpoint_store is _UNSET else checkpoint_store
        ),
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


def test_snapshot_rejects_symlink_back_into_the_live_workspace(tmp_path: Any) -> None:
    """Through such a link a probed `rm` would reach the tree the snapshot
    exists to protect."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "real.txt").write_text("live")
    os.symlink(workspace / "real.txt", workspace / "leak")

    with baseline_snapshot(str(workspace)) as snapshot:
        assert snapshot is None


def test_snapshot_keeps_a_usable_interpreter_without_a_write_path_out(tmp_path: Any) -> None:
    """A `.venv/bin/python -> /usr/...` link is the standard virtualenv shape,
    so the snapshot must keep the interpreter — but as a copy. Left as a link,
    a probed contract writing through it would edit the real installation."""
    external = tmp_path / "outside" / "python3"
    external.parent.mkdir(parents=True)
    external.write_text("#!/bin/sh\necho REAL\n")
    external.chmod(0o755)

    workspace = tmp_path / "ws"
    bin_dir = workspace / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    os.symlink(external, bin_dir / "python")
    (workspace / "kept.txt").write_text("kept")

    with baseline_snapshot(str(workspace)) as snapshot:
        assert snapshot is not None
        copied = Path(snapshot) / ".venv" / "bin" / "python"
        assert copied.is_file() and not copied.is_symlink()
        assert copied.read_text() == external.read_text()
        assert os.access(copied, os.X_OK)
        assert os.path.exists(os.path.join(snapshot, "kept.txt"))

        copied.write_text("MUTATED BY PROBE")

    assert external.read_text() == "#!/bin/sh\necho REAL\n"


def test_snapshot_refuses_a_link_to_an_external_directory(tmp_path: Any) -> None:
    """No containment preserves its meaning, so the baseline stays unknown
    rather than leaving a probe a writable path into that directory."""
    outside = tmp_path / "outside"
    (outside / "data").mkdir(parents=True)
    (outside / "data" / "keep.txt").write_text("precious")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    os.symlink(outside / "data", workspace / "data")

    with baseline_snapshot(str(workspace)) as snapshot:
        assert snapshot is None

    assert (outside / "data" / "keep.txt").read_text() == "precious"


def test_snapshot_keeps_links_that_stay_inside_it(tmp_path: Any) -> None:
    workspace = tmp_path / "ws"
    (workspace / "pkg").mkdir(parents=True)
    (workspace / "pkg" / "cli.js").write_text("run")
    (workspace / "node_modules" / ".bin").mkdir(parents=True)
    os.symlink("../../pkg/cli.js", workspace / "node_modules" / ".bin" / "tool")

    with baseline_snapshot(str(workspace)) as snapshot:
        assert snapshot is not None
        link = Path(snapshot) / "node_modules" / ".bin" / "tool"
        assert link.is_symlink()
        assert link.resolve().is_relative_to(Path(snapshot).resolve())


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


@pytest.mark.asyncio
async def test_git_dependent_contract_is_unknown_not_judged(tmp_path: Any) -> None:
    """The snapshot has no `.git`, so a git-consulting contract would be judged
    in an environment with different semantics — unknown, never a verdict."""
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(description="ac", verify_command="git diff --check"),
        AcceptanceCriterionSpec(description="not git", verify_command="test -d ."),
    )

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    assert baseline.records[0].probed is False
    assert "git" in (baseline.records[0].reason or "")
    assert baseline.records[1].probed is True
    # A name merely containing "git" must not trip the detector.
    assert not any(record.probed is False for record in [baseline.records[1]])


@pytest.mark.asyncio
async def test_git_word_detector_ignores_lookalike_names(tmp_path: Any) -> None:
    script = tmp_path / "legit.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="./legit.sh"))

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    assert baseline.records[0].probed is True


@pytest.mark.asyncio
async def test_all_pass_claim_requires_full_probe_coverage(tmp_path: Any) -> None:
    """One unknown contract forbids the 'every contract already passes' claim."""
    (tmp_path / "pre.txt").write_text("here")
    executor = _make_executor(working_directory=str(tmp_path))
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(description="pass", verify_command="test -f pre.txt"),
        AcceptanceCriterionSpec(description="unknown", verify_command="git status"),
    )

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is not None
    assert baseline.all_contracts_baseline_pass is False
    assert _emitted(executor, BASELINE_ALL_PASS_EVENT) == []


@pytest.mark.asyncio
async def test_baseline_is_persisted_before_any_worker_effect(tmp_path: Any) -> None:
    from ouroboros.orchestrator.verify_baseline import persist_verify_baseline_checkpoint

    executor = _make_executor(working_directory=str(tmp_path))
    store = MagicMock()
    executor._checkpoint_store = store
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    await persist_verify_baseline_checkpoint(executor, seed=seed, session_id="s", execution_id="e")

    assert store.save.called
    saved_state = store.save.call_args.args[0].state
    assert saved_state["completed_levels"] == 0
    assert saved_state["session_id"] == "s"
    restored = restore_verify_baseline(saved_state["verify_baseline"])
    assert restored is not None
    assert restored.records[0].probed is True


def test_semantics_contract_carries_probe_field_and_climbs_from_v4() -> None:
    from ouroboros.orchestrator.execution_semantics import (
        migrated_legacy_execution_semantics,
    )

    v4 = {
        "version": 4,
        "run_verify_commands": True,
        "verify_command_timeout_seconds": 600,
        "ac_retry_attempts": 2,
        "cross_harness_redispatch": True,
        "enable_decomposition": True,
        "decomposition_mode": "bounce_only",
        "max_decomposition_depth": 2,
        "max_parallel_workers": 3,
        "effective_parallel_workers": 3,
        "adaptive_concurrency_policy": None,  # filled below
        "fat_harness_mode": False,
        "shadow_replay_enabled": False,
        "checkpoint_store_enabled": True,
        "session_signal_hub_enabled": True,
        "context_pack_enabled": True,
        "backend_limits_backend": "claude",
        "backend_max_concurrency": None,
        "backend_requests_per_minute": None,
        "backend_tokens_per_minute": None,
        "backend_self_governs_rate_limit": True,
        "usage_limit_pause_seconds": 18000,
        "runtime_effect_capabilities": None,  # filled below
    }
    from ouroboros.orchestrator.adaptive_concurrency import adaptive_concurrency_policy

    v4["adaptive_concurrency_policy"] = adaptive_concurrency_policy(initial_limit=3, max_limit=3)
    from ouroboros.orchestrator.execution_authority import (
        runtime_effect_capabilities_contract,
    )
    from ouroboros.orchestrator.execution_semantics import (
        valid_execution_semantics_contract,
    )

    class _Adapter:
        runtime_backend = "claude"

    v4["runtime_effect_capabilities"] = runtime_effect_capabilities_contract(_Adapter())

    # A v4 contract climbs every rung added since, each restoring what that
    # session actually ran under rather than this build's default.
    climbed_v4 = migrated_legacy_execution_semantics(v4)
    assert climbed_v4 is not None
    assert climbed_v4["version"] == 6
    assert climbed_v4["vacuous_contract_evidence"] == "honored"
    assert climbed_v4["verify_baseline_probe"] == "observe"
    assert valid_execution_semantics_contract(climbed_v4) is True

    # A v5 contract climbs only the rung it is missing.
    v5 = dict(v4)
    v5["version"] = 5
    v5["vacuous_contract_evidence"] = "revoked"
    climbed_v5 = migrated_legacy_execution_semantics(v5)
    assert climbed_v5 is not None
    assert climbed_v5["version"] == 6
    assert climbed_v5["vacuous_contract_evidence"] == "revoked"
    assert climbed_v5["verify_baseline_probe"] == "observe"

    # The current schema requires both fields, and a payload already carrying a
    # field it claims to predate has disproved its own version.
    current_missing = dict(climbed_v4)
    del current_missing["verify_baseline_probe"]
    assert valid_execution_semantics_contract(current_missing) is False
    assert migrated_legacy_execution_semantics(climbed_v4) is None
    v4_with = dict(v4)
    v4_with["verify_baseline_probe"] = "observe"
    assert migrated_legacy_execution_semantics(v4_with) is None


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


@pytest.mark.parametrize(
    "records",
    [
        # "0" and "00" both normalize to index 0: the later record would
        # silently replace the earlier authoritative one.
        {
            "0": {
                "probed": False,
                "baseline_verdict": None,
                "preexisting_artifacts": [],
                "reason": None,
            },
            "00": {
                "probed": True,
                "baseline_verdict": True,
                "preexisting_artifacts": [],
                "reason": None,
            },
        },
        # Non-canonical spellings on their own are not this run's keys either.
        {
            "007": {
                "probed": True,
                "baseline_verdict": True,
                "preexisting_artifacts": [],
                "reason": None,
            }
        },
        # `str.isdigit` accepts other numeric scripts; `int` reads some and
        # raises on others.
        {
            "٣": {
                "probed": True,
                "baseline_verdict": True,
                "preexisting_artifacts": [],
                "reason": None,
            }
        },
        {
            "²": {
                "probed": True,
                "baseline_verdict": True,
                "preexisting_artifacts": [],
                "reason": None,
            }
        },
    ],
)
def test_non_canonical_record_keys_are_wholly_unknown(records: Any) -> None:
    """Salvaging part of a malformed payload is how a crafted record replaces
    an authoritative one; the whole baseline becomes unknown instead."""
    assert (
        restore_verify_baseline(
            {"all_contracts_baseline_pass": False, "records": records},
        )
        is None
    )


@pytest.mark.asyncio
async def test_a_baseline_that_cannot_be_persisted_is_reported_as_such(tmp_path: Any) -> None:
    """`CheckpointStore.save` reports ordinary failures as `Result.err` rather
    than raising, so the returned value is the only place they are visible."""
    from ouroboros.orchestrator.verify_baseline import persist_verify_baseline_checkpoint

    executor = _make_executor(
        working_directory=str(tmp_path),
        checkpoint_store=_StubCheckpointStore(saves=False),
    )
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )
    assert executor._verify_baseline is not None

    durable = await persist_verify_baseline_checkpoint(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert durable is False


@pytest.mark.asyncio
async def test_a_raising_checkpoint_store_is_also_not_durable(tmp_path: Any) -> None:
    from ouroboros.orchestrator.verify_baseline import persist_verify_baseline_checkpoint

    store = MagicMock()
    store.save.side_effect = OSError("disk full")
    executor = _make_executor(working_directory=str(tmp_path), checkpoint_store=store)
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))
    executor._verify_baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert (
        await persist_verify_baseline_checkpoint(
            executor, seed=seed, session_id="s", execution_id="e"
        )
        is False
    )


@pytest.mark.asyncio
async def test_no_checkpoint_store_means_no_baseline_at_all(tmp_path: Any) -> None:
    """A baseline the next process cannot see is authority it cannot check:
    without a store the probe does not run, and everything reads unknown."""
    executor = _make_executor(working_directory=str(tmp_path), checkpoint_store=None)
    seed = _seed_with_specs(AcceptanceCriterionSpec(description="ac", verify_command="test -d ."))

    baseline = await establish_verify_baseline(
        executor, seed=seed, session_id="s", execution_id="e"
    )

    assert baseline is None
    assert _emitted(executor, BASELINE_PROBE_EVENT) == []


@pytest.mark.asyncio
async def test_workers_do_not_dispatch_under_an_unpersistable_baseline(tmp_path: Any) -> None:
    """The failed save is the whole point: without it a crash before the first
    level checkpoint re-enters with nothing recognized and probes a
    worker-modified tree as pristine. Nothing has been dispatched yet, so
    refusing here leaves the workspace exactly as the operator left it."""
    from ouroboros.orchestrator.dependency_analyzer import ACNode, DependencyGraph

    (tmp_path / "pre.txt").write_text("here")
    executor = _make_executor(
        working_directory=str(tmp_path),
        checkpoint_store=_StubCheckpointStore(saves=False),
    )
    dispatched = AsyncMock(return_value=[])
    executor._execute_ac_batch = dispatched  # type: ignore[method-assign]
    seed = _seed_with_specs(
        AcceptanceCriterionSpec(description="ac", verify_command="test -f pre.txt")
    )
    graph = DependencyGraph(
        nodes=(ACNode(index=0, content="ac", depends_on=()),),
        execution_levels=((0,),),
    )

    with pytest.raises(RuntimeError, match="verify baseline could not be persisted"):
        await executor.execute_parallel(
            seed=seed,
            execution_plan=graph.to_execution_plan(),
            session_id="s",
            execution_id="e",
            tools=["Read"],
            tool_catalog=None,
            system_prompt="sys",
        )

    assert dispatched.await_count == 0
    assert executor._verify_baseline is None
