"""Stage 0/1 of evidential force: baseline negative control for verify gates.

A contract earns evidential force exactly when its verdict is a non-constant
function of the worker's contribution. The static proof
(``core/verify_command_plan.constant_verdict``) catches independence when it is
provable from the command text alone; this module measures it: before any
worker is dispatched, every contract is run against a disposable copy of the
pristine workspace. A contract that already passes there discriminated nothing
this run — its later pass carries no information about the work.

Three rules, inherited from the verify gate's founding invariant, keep every
consumer honest (RFC ``docs/rfc/verify-gate-evidential-force.md``, #2179):

* **A probe's only authority is to withhold a stamp.** A baseline-passing
  contract never fails an AC and never stops a run; it loses the
  ``discriminating_pass`` tier and — because a contract that passes on the
  empty workspace proves nothing about the work — its authority to resurrect
  a worker-reported failure.
* **Unknown grants nothing and revokes nothing.** A probe that could not run
  honestly (no snapshot, no shell, start error, timeout) records
  ``probed=False``; the AC behaves exactly as it would with the probe off.
  Only a genuine judgment — exit status, output assertion, missing artifact,
  workspace mutation — may count as a baseline FAIL, so an infrastructure
  hiccup can never mint a discrimination certificate.
* **The live workspace is untouchable.** Probes run in per-AC disposable
  copies (measured contract corpus contains ``rm``-bearing commands), captured
  by plain ``copytree`` — never a sibling git worktree, and never sharing git
  metadata with the live repository.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import contextlib
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import TYPE_CHECKING, Any

from rich.text import Text

from ouroboros.core.seed import AcceptanceCriterionSpec, ac_text
from ouroboros.events.base import BaseEvent
from ouroboros.observability.logging import get_logger
from ouroboros.orchestrator.verify_gate_outcome import (
    _missing_expected_artifacts,
    _VerifyGateOutcome,
)

if TYPE_CHECKING:
    from ouroboros.core.seed import Seed
    from ouroboros.orchestrator.parallel_executor import ParallelACExecutor

log = get_logger(__name__)

BASELINE_PROBE_EVENT = "execution.verify.baseline"
NON_DISCRIMINATING_EVENT = "execution.verify.non_discriminating"
BASELINE_ALL_PASS_EVENT = "execution.verify.baseline_all_pass"

VERDICT_TIER_PASS = "pass"
VERDICT_TIER_DISCRIMINATING_PASS = "discriminating_pass"

# Bounded audit detail carried into the checkpoint, mirroring the output-tail
# discipline of the gate outcome itself.
_BASELINE_REASON_CHARS = 500

# The snapshot deliberately KEEPS environment directories (.venv, node_modules)
# that shadow_replay._COPY_IGNORE drops: shadow replay re-runs an LLM there,
# while the baseline probe runs the verify command itself, which needs a
# runnable environment (`uv run pytest -q` on a green repo must baseline-PASS,
# not fail for want of its interpreter). Only git metadata and pure caches are
# excluded. A contract that invokes git itself will fail at baseline for an
# incidental reason; that is reported as probed (an honest FAIL) only when the
# failure is a genuine judgment — see _classify_probe_outcome.
_BASELINE_COPY_IGNORE = shutil.ignore_patterns(
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)

# Gate reasons that mean nothing was judged. Kept in sync by hand with the
# reason strings _run_ac_verify_gate produces; matching is prefix/substring so
# parameterized tails (status codes, timeouts) do not defeat it. Anything
# matched here yields probed=False — the safe direction is always "unknown".
_UNJUDGED_REASON_PREFIX = "verify_command could not start"
_UNJUDGED_REASON_SUBSTRING = "timed out after"

# The snapshot carries no `.git`, so any contract that consults git would be
# judged against an environment with different semantics than the live tree —
# `git diff --check` exits 129 there while passing on the real workspace, a
# discrimination certificate without worker contribution. Word-boundary match;
# `legit.sh` or `digit` must not trip it.
_GIT_WORD = re.compile(r"(?<![\w/.-])git(?![\w.-])")

_RECORD_KEYS = frozenset({"probed", "baseline_verdict", "preexisting_artifacts", "reason"})
_BASELINE_KEYS = frozenset({"all_contracts_baseline_pass", "records"})


@dataclass(frozen=True)
class ACBaselineRecord:
    """What the negative control learned about one AC's contract."""

    probed: bool
    baseline_verdict: bool | None
    preexisting_artifacts: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class VerifyBaseline:
    """Per-AC baseline records for one run, keyed by AC index."""

    records: dict[int, ACBaselineRecord]
    all_contracts_baseline_pass: bool


@contextlib.contextmanager
def baseline_snapshot(cwd: str) -> Iterator[str | None]:
    """Yield a frozen master copy of ``cwd``, or ``None`` when unsafe.

    ``None`` means the probe is skipped for the whole run — never that the run
    stops. Escaping symlinks reject the snapshot for the same reason as shadow
    replay: a probed command must not be able to reach the live filesystem.
    """
    base: str | None = None
    try:
        try:
            base = tempfile.mkdtemp(prefix="ooo-verify-baseline-")
            target = Path(base) / "workspace"
            shutil.copytree(cwd, target, ignore=_BASELINE_COPY_IGNORE, symlinks=True)
            has_link_into_live_tree = _snapshot_links_into_live_workspace(target, Path(cwd))
        except Exception as exc:
            log.warning(
                "parallel_executor.verify_baseline.snapshot_error",
                cwd=cwd,
                error=str(exc),
            )
            yield None
            return
        if has_link_into_live_tree:
            log.warning("parallel_executor.verify_baseline.unsafe_symlink", cwd=cwd)
            yield None
            return
        yield str(target)
    finally:
        if base is not None:
            shutil.rmtree(base, ignore_errors=True)


def _snapshot_links_into_live_workspace(root: Path, live_root: Path) -> bool:
    """Return whether any copied symlink resolves inside the live workspace.

    Only links back into the live tree are rejected: through one, a probed
    contract could read or mutate the very workspace the snapshot exists to
    protect. Links to the wider system — a `.venv/bin/python` pointing at an
    installed interpreter is the standard shape — are exactly what the same
    command would use when the gate later runs it against the live tree, so
    rejecting them would silently disable the baseline for every ordinary
    Python workspace.
    """
    resolved_live = live_root.resolve()
    for current_root, dirnames, filenames in os.walk(root, followlinks=False):
        for name in (*dirnames, *filenames):
            candidate = Path(current_root) / name
            if not candidate.is_symlink():
                continue
            try:
                resolved = candidate.resolve(strict=False)
            except (OSError, RuntimeError):
                return True
            if resolved.is_relative_to(resolved_live):
                return True
    return False


def _classify_probe_outcome(outcome: _VerifyGateOutcome) -> ACBaselineRecord:
    """Turn a gate outcome on the pristine copy into a baseline record."""
    reason = outcome.reason
    bounded_reason = reason[:_BASELINE_REASON_CHARS] if reason is not None else None
    unjudged = outcome.environment_unverifiable or (
        reason is not None
        and (reason.startswith(_UNJUDGED_REASON_PREFIX) or _UNJUDGED_REASON_SUBSTRING in reason)
    )
    if unjudged:
        return ACBaselineRecord(probed=False, baseline_verdict=None, reason=bounded_reason)
    return ACBaselineRecord(
        probed=True,
        baseline_verdict=outcome.passed,
        reason=bounded_reason,
    )


def render_verify_baseline_all_pass_warning(display_indices: Sequence[int]) -> str:
    """Render the run-start finding that no contract can discriminate."""
    rendered = ", ".join(str(index) for index in display_indices)
    return (
        f"Every verify contract already passes on the pristine workspace "
        f"(AC {rendered}). None of them can distinguish this run's work from "
        "no work at all; their passes will not be counted as discriminating. "
        "Strengthen the contracts so they fail before the work exists."
    )


async def _emit_probe_event(
    executor: ParallelACExecutor,
    *,
    session_id: str,
    execution_id: str,
    ac_index: int,
    spec: AcceptanceCriterionSpec,
    record: ACBaselineRecord,
    duration_seconds: float,
) -> None:
    await executor._safe_emit_event(
        BaseEvent(
            type=BASELINE_PROBE_EVENT,
            aggregate_type="execution",
            aggregate_id=execution_id or session_id,
            data={
                "session_id": session_id,
                "execution_id": execution_id,
                "ac_index": ac_index,
                "ac_content": ac_text(spec),
                "verify_command": spec.verify_command,
                "probed": record.probed,
                "baseline_verdict": record.baseline_verdict,
                "preexisting_artifacts": list(record.preexisting_artifacts),
                "reason": record.reason,
                "duration_seconds": duration_seconds,
            },
        )
    )


async def establish_verify_baseline(
    executor: ParallelACExecutor,
    *,
    seed: Seed,
    session_id: str,
    execution_id: str,
) -> VerifyBaseline | None:
    """Run the negative control for every contract-carrying AC.

    Returns ``None`` when the probe is off or no snapshot could be captured —
    both indistinguishable, downstream, from "baseline unknown".
    """
    if executor._verify_baseline_probe == "off" or not executor._run_verify_commands:
        return None
    contract_indices = [
        index
        for index, criterion in enumerate(seed.acceptance_criteria)
        if isinstance(criterion, AcceptanceCriterionSpec)
        and criterion.has_evidential_success_contract
    ]
    if not contract_indices:
        return None

    from ouroboros.orchestrator.parallel_executor import (
        _FOUNDATION_A_ENTRY_RUN_AC_VERIFY_GATE,
        _invoke_execution_authority_entry,
    )

    cwd = executor._task_cwd or executor._adapter.working_directory or os.getcwd()
    records: dict[int, ACBaselineRecord] = {}
    with baseline_snapshot(cwd) as master:
        if master is None:
            return None
        for index in contract_indices:
            spec = seed.acceptance_criteria[index]
            assert isinstance(spec, AcceptanceCriterionSpec)
            started = time.monotonic()
            if spec.verify_command and _GIT_WORD.search(spec.verify_command):
                record = ACBaselineRecord(
                    probed=False,
                    baseline_verdict=None,
                    reason=(
                        "git-dependent contract cannot be probed against a "
                        "gitless snapshot; baseline unknown"
                    ),
                )
                records[index] = record
                await _emit_probe_event(
                    executor,
                    session_id=session_id,
                    execution_id=execution_id,
                    ac_index=index,
                    spec=spec,
                    record=record,
                    duration_seconds=0.0,
                )
                continue
            probe_dir: str | None = None
            try:
                # Fresh copy per AC: the measured contract corpus contains
                # rm-bearing commands, so probes must never share a tree.
                probe_base = tempfile.mkdtemp(prefix="ooo-verify-baseline-ac-")
                probe_dir = str(Path(probe_base) / "workspace")
                shutil.copytree(master, probe_dir, symlinks=True)
                missing = _missing_expected_artifacts(spec.expected_artifacts, probe_dir)
                preexisting = tuple(
                    artifact for artifact in spec.expected_artifacts if artifact not in missing
                )
                outcome = await _invoke_execution_authority_entry(
                    executor,
                    _FOUNDATION_A_ENTRY_RUN_AC_VERIFY_GATE,
                    spec=spec,
                    cwd=probe_dir,
                )
                record = replace(
                    _classify_probe_outcome(outcome), preexisting_artifacts=preexisting
                )
            except Exception as exc:
                log.warning(
                    "parallel_executor.verify_baseline.probe_error",
                    session_id=session_id,
                    ac_index=index,
                    error=str(exc),
                )
                record = ACBaselineRecord(
                    probed=False,
                    baseline_verdict=None,
                    reason=str(exc)[:_BASELINE_REASON_CHARS],
                )
            finally:
                if probe_dir is not None:
                    shutil.rmtree(Path(probe_dir).parent, ignore_errors=True)
            records[index] = record
            await _emit_probe_event(
                executor,
                session_id=session_id,
                execution_id=execution_id,
                ac_index=index,
                spec=spec,
                record=record,
                duration_seconds=round(time.monotonic() - started, 3),
            )

    # The broken-seed claim requires full coverage: one unknown contract means
    # "every contract already passes" is not a statement this run can make.
    all_pass = bool(records) and all(
        record.probed and record.baseline_verdict is True for record in records.values()
    )
    baseline = VerifyBaseline(records=records, all_contracts_baseline_pass=all_pass)
    if all_pass:
        display_indices = [index + 1 for index in sorted(records)]
        warning = render_verify_baseline_all_pass_warning(display_indices)
        await executor._safe_emit_event(
            BaseEvent(
                type=BASELINE_ALL_PASS_EVENT,
                aggregate_type="execution",
                aggregate_id=execution_id or session_id,
                data={
                    "session_id": session_id,
                    "execution_id": execution_id,
                    "ac_indices": sorted(records),
                    "count": len(records),
                    "guidance": warning,
                },
            )
        )
        log.warning(
            "parallel_executor.verify_baseline.all_contracts_pass",
            session_id=session_id,
            ac_indices=sorted(records),
        )
        executor._console.print(Text(warning, style="yellow"))
    return baseline


def verify_baseline_checkpoint_state(
    baseline: VerifyBaseline | None,
) -> dict[str, object] | None:
    """Encode the baseline into JSON-safe checkpoint state."""
    if baseline is None:
        return None
    return {
        "all_contracts_baseline_pass": baseline.all_contracts_baseline_pass,
        "records": {
            str(index): {
                "probed": record.probed,
                "baseline_verdict": record.baseline_verdict,
                "preexisting_artifacts": list(record.preexisting_artifacts),
                "reason": record.reason,
            }
            for index, record in baseline.records.items()
        },
    }


def restore_verify_baseline(value: object) -> VerifyBaseline | None:
    """Decode checkpointed baseline state; malformed payloads become unknown.

    Unknown grants nothing and revokes nothing, so rejecting a corrupt payload
    wholesale is strictly safer than salvaging part of it.
    """
    if not isinstance(value, Mapping) or set(value.keys()) != _BASELINE_KEYS:
        return None
    all_pass = value.get("all_contracts_baseline_pass")
    raw_records = value.get("records")
    if not isinstance(all_pass, bool) or not isinstance(raw_records, Mapping):
        return None
    records: dict[int, ACBaselineRecord] = {}
    for raw_index, raw_record in raw_records.items():
        if not isinstance(raw_index, str) or not raw_index.isdigit():
            return None
        if not isinstance(raw_record, Mapping) or set(raw_record.keys()) != _RECORD_KEYS:
            return None
        probed = raw_record.get("probed")
        verdict = raw_record.get("baseline_verdict")
        raw_preexisting = raw_record.get("preexisting_artifacts")
        reason = raw_record.get("reason")
        if not isinstance(probed, bool):
            return None
        if verdict is not None and not isinstance(verdict, bool):
            return None
        if probed != (verdict is not None):
            return None
        if not isinstance(raw_preexisting, list) or not all(
            isinstance(item, str) for item in raw_preexisting
        ):
            return None
        if reason is not None and (
            not isinstance(reason, str) or len(reason) > _BASELINE_REASON_CHARS
        ):
            return None
        records[int(raw_index)] = ACBaselineRecord(
            probed=probed,
            baseline_verdict=verdict,
            preexisting_artifacts=tuple(raw_preexisting),
            reason=reason,
        )
    return VerifyBaseline(records=records, all_contracts_baseline_pass=all_pass)


async def persist_verify_baseline_checkpoint(
    executor: ParallelACExecutor,
    *,
    seed: Seed,
    session_id: str,
    execution_id: str,
) -> None:
    """Durably record the baseline before the first worker can touch the tree.

    Levels checkpoint only on completion, so a crash between the first worker
    effect and the first level save would otherwise re-enter with no recognized
    checkpoint and re-probe a partially modified workspace as "pristine" — an
    artifact created before the crash would become a false baseline PASS. This
    zero-levels checkpoint closes that window: re-entry restores the probed
    records (or, if the payload is unreadable, fails closed to unknown).
    """
    if executor._checkpoint_store is None or executor._verify_baseline is None:
        # No baseline, nothing to protect: contract-less and probe-off runs
        # keep their exact pre-existing checkpoint cadence.
        return
    from ouroboros.orchestrator.execution_authority import canonical_workspace_authority
    from ouroboros.persistence.checkpoint import CheckpointData

    try:
        checkpoint = CheckpointData.create(
            seed_id=getattr(seed, "id", session_id),
            phase="parallel_execution",
            state={
                "session_id": session_id,
                "execution_id": execution_id,
                "workspace_identity": canonical_workspace_authority(
                    executor._task_cwd
                    or getattr(executor._adapter, "working_directory", None)
                    or os.getcwd()
                ),
                "completed_levels": 0,
                "verify_baseline": verify_baseline_checkpoint_state(executor._verify_baseline),
            },
        )
        executor._checkpoint_store.save(checkpoint)
    except Exception as exc:
        # Best-effort: losing the early save reopens only the pre-first-level
        # crash window it exists to close; it must never fail the run.
        log.warning(
            "parallel_executor.verify_baseline.early_checkpoint_error",
            session_id=session_id,
            error=str(exc),
        )


async def settle_passing_verify_outcome(
    executor: ParallelACExecutor,
    *,
    spec: AcceptanceCriterionSpec,
    ac_index: int,
    result: Any,
    outcome: _VerifyGateOutcome,
    cached_outcome: object,
    session_id: str,
    execution_id: str,
) -> Any:
    """Settle a passing contract gate under the baseline's verdict tiers.

    Owns everything the gate does once ``outcome.passed`` is true: tier
    stamping, the recovery of a worker-reported failure, and — for a
    baseline-passing contract — the withhold of both. Extracted from
    ``_apply_verify_gate`` so the withhold-only invariant lives next to the
    probe that justifies it.
    """
    from dataclasses import replace as dc_replace

    from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome

    baseline = executor._verify_baseline
    record = baseline.records.get(ac_index) if baseline is not None else None
    baseline_pass = record is not None and record.probed and record.baseline_verdict is True
    discriminating = record is not None and record.probed and record.baseline_verdict is False
    tier = VERDICT_TIER_DISCRIMINATING_PASS if discriminating else VERDICT_TIER_PASS
    stamped = outcome if outcome.verdict_tier == tier else dc_replace(outcome, verdict_tier=tier)

    async def emit_non_discriminating(*, recovery_revoked: bool, prior_error: str | None) -> None:
        await executor._safe_emit_event(
            BaseEvent(
                type=NON_DISCRIMINATING_EVENT,
                aggregate_type="execution",
                aggregate_id=execution_id or session_id,
                data={
                    "session_id": session_id,
                    "execution_id": execution_id,
                    "ac_index": ac_index,
                    "ac_content": ac_text(spec),
                    "verify_command": spec.verify_command,
                    "expected_artifacts": list(spec.expected_artifacts),
                    "preexisting_artifacts": list(
                        record.preexisting_artifacts if record is not None else ()
                    ),
                    "baseline_verdict": True,
                    "output_tail": outcome.output_tail,
                    "mode": "observe",
                    "recovery_revoked": recovery_revoked,
                    "evidence_exemption_changed": False,
                    "prior_error": prior_error,
                },
            )
        )

    if not result.success and not result.is_blocked and not result.is_invalid:
        if baseline_pass:
            # Revocation: a contract that passed on the pristine workspace
            # proves nothing about the work, so it must never resurrect a
            # failed AC. The worker's own report stands.
            await emit_non_discriminating(recovery_revoked=True, prior_error=result.error)
            log.warning(
                "parallel_executor.ac.verify_gate_recovery_revoked",
                session_id=session_id,
                ac_index=ac_index,
                prior_error=result.error,
            )
            return dc_replace(result, verify_gate_outcome=stamped)
        recovery_message = (
            "Runtime reported failure, but the AC success contract passed: "
            "expected_artifacts/verify_command satisfied."
        )
        await executor._safe_emit_event(
            BaseEvent(
                type="execution.verify.recovered",
                aggregate_type="execution",
                aggregate_id=execution_id or session_id,
                data={
                    "session_id": session_id,
                    "execution_id": execution_id,
                    "ac_index": ac_index,
                    "ac_content": ac_text(spec),
                    "verify_command": spec.verify_command,
                    "expected_artifacts": list(spec.expected_artifacts),
                    "prior_error": result.error,
                    "output_tail": outcome.output_tail,
                },
            )
        )
        log.info(
            "parallel_executor.ac.verify_gate_recovered",
            session_id=session_id,
            ac_index=ac_index,
            prior_error=result.error,
        )
        return dc_replace(
            result,
            success=True,
            error=None,
            final_message=(result.final_message or recovery_message),
            outcome=ACExecutionOutcome.SUCCEEDED,
            verify_gate_outcome=stamped,
        )

    if baseline_pass and outcome.verdict_tier != tier:
        # Withhold-only: the pass stands, the evidence obligations stand, and
        # only the discriminating stamp is withheld. Emitted once, when the
        # tier is first stamped, so finalization re-application stays quiet.
        await emit_non_discriminating(recovery_revoked=False, prior_error=None)
        log.info(
            "parallel_executor.ac.verify_gate_non_discriminating",
            session_id=session_id,
            ac_index=ac_index,
        )
    if cached_outcome is outcome and stamped is outcome:
        return result
    return dc_replace(result, verify_gate_outcome=stamped)
