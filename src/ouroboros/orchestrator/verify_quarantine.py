"""Handling for acceptance criteria this machine cannot verify right now.

When ``verify_command`` has no POSIX interpreter to run in, the orchestrator
judged nothing — it did not observe a failure, and it certainly did not observe
a pass. Three rules follow, and this module exists so they stay in one place:

* The run does not stop. Other criteria keep going.
* The criterion is never counted as verified, and is surfaced in the report as
  owed manual confirmation.
* The criterion keeps its retries. Missing bash is a fact about the machine,
  not about the leaf, and it is one an operator can fix while the run is still
  going: installing Git Bash or setting ``OUROBOROS_VERIFY_BASH`` makes the
  next attempt resolve a real shell (``resolve_verify_shell`` deliberately does
  not cache a failed resolution).

This is why the outcome is ``FAILED`` with
``FailureClass.VERIFY_ENVIRONMENT_UNAVAILABLE`` rather than ``BLOCKED``.
``BLOCKED`` reads as a hard precondition: ``route_escalation`` refuses to
escalate it and raises ``HUMAN_HANDOFF_REQUIRED``, and
``retry_hints.is_retryable_failure`` drops it. That is the correct shape for
"a human must intervene before anything can proceed", and the wrong shape for
"this machine is missing a shell, and the next attempt may find one."

"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ouroboros.events.base import BaseEvent
from ouroboros.observability.logging import get_logger
from ouroboros.orchestrator.failure_taxonomy import FailureClass
from ouroboros.orchestrator.parallel_executor_models import ACExecutionOutcome, ACExecutionResult
from ouroboros.orchestrator.verifier import VerifierVerdict

if TYPE_CHECKING:
    from ouroboros.core.seed import AcceptanceCriterionSpec

VERIFY_UNVERIFIABLE_EVENT = "execution.verify.unverifiable"

log = get_logger(__name__)


def quarantine_reason(gate_reason: str | None) -> str:
    """Explain the outcome without implying a verdict on the work itself."""
    detail = f"{gate_reason} " if gate_reason else ""
    return (
        f"AC unverifiable on this machine: {detail}"
        "The work was not judged. Fixing the shell makes the next attempt "
        "verify it; until then this AC is never counted as passed."
    )


def render_unverifiable_summary(display_indices: list[int]) -> str:
    """Render the report line that keeps these criteria out of the pass count."""
    rendered = ", ".join(str(index) for index in display_indices)
    return (
        f"Unverifiable on this machine ({len(display_indices)}), "
        f"needs manual confirmation: AC {rendered}"
    )


async def quarantine_unverifiable_result(
    *,
    result: ACExecutionResult,
    spec: AcceptanceCriterionSpec,
    ac_content: str,
    ac_index: int,
    gate_reason: str | None,
    verify_gate_outcome: object,
    session_id: str,
    execution_id: str,
    emit_event: Callable[[BaseEvent], Awaitable[Any]],
) -> ACExecutionResult:
    """Record that one AC could not be judged, and let the run continue."""
    reason = quarantine_reason(gate_reason)
    await emit_event(
        BaseEvent(
            type=VERIFY_UNVERIFIABLE_EVENT,
            aggregate_type="execution",
            aggregate_id=execution_id or session_id,
            data={
                "session_id": session_id,
                "execution_id": execution_id,
                "ac_index": ac_index,
                "ac_content": ac_content,
                "verify_command": spec.verify_command,
                "reason": gate_reason,
                "failure_class": FailureClass.VERIFY_ENVIRONMENT_UNAVAILABLE.value,
            },
        )
    )
    log.warning(
        "parallel_executor.ac.verify_gate_unverifiable",
        session_id=session_id,
        ac_index=ac_index,
        reason=gate_reason,
    )
    return replace(
        result,
        success=False,
        error=reason,
        final_message=reason,
        outcome=ACExecutionOutcome.FAILED,
        atomic_verifier_verdict=VerifierVerdict(
            passed=False,
            reasons=(reason,),
            failure_class=FailureClass.VERIFY_ENVIRONMENT_UNAVAILABLE.value,
        ),
        verify_gate_outcome=verify_gate_outcome,
    )
