"""Follow a run job's ``run → evaluate → ralph`` successor chain to terminal.

``ooo run`` already owns the successor policy: an evaluable terminal execution
enqueues a formal evaluation, and a rejected verdict enqueues a bounded Ralph
convergence loop. ``ooo auto`` used to re-implement that policy inside its own
phase machine (``RALPH_HANDOFF`` → ``EVALUATE``) with a *different* judge, which
is how the product ended up with several incompatible definitions of "done".

This module lets Auto reuse the run chain instead of owning a second one: it
walks the successor job ids the chain publishes on each job's result meta
(``chained_evaluate_job_id`` → ``chained_ralph_job_id``) and reports the last
terminal it reached. Auto then maps that single terminal onto its own phases.

Nothing here starts work — the chain is started by the run job itself. This only
observes, so a caller that does not wait (the default, non-``complete_product``
Auto) simply never invokes it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol

# Successor handles published by ``run_evaluate_chain`` / ``evaluate_ralph_chain``
# on the *preceding* job's result meta, in the order the chain creates them.
SUCCESSOR_META_KEYS: tuple[str, ...] = ("chained_evaluate_job_id", "chained_ralph_job_id")

_DEFAULT_POLL_SECONDS = 2.0

# The single Ralph stop reason that means the loop's own QA verdict passed
# (``ralph_loop.py:305``). Kept as a constant so the approval test never drifts
# into matching a budget terminal such as ``max_generations reached``.
RALPH_APPROVED_STOP_REASON = "qa passed"

# A verdict only counts from a job that actually finished cleanly. `failed`,
# `cancelled`, `interrupted` and the follower's own `timed_out` never approve,
# whatever their (possibly inherited) result meta says.
_APPROVING_STATUSES = frozenset({"completed"})


class _Snapshot(Protocol):
    job_id: str
    status: Any
    result_text: str | None
    result_meta: dict[str, Any]

    @property
    def is_terminal(self) -> bool: ...


class _JobManager(Protocol):
    async def get_snapshot(self, job_id: str) -> _Snapshot: ...


@dataclass(frozen=True)
class ChainTerminal:
    """Terminal state of the deepest successor the chain reached."""

    job_id: str
    job_kind: str
    """``run`` / ``evaluate`` / ``ralph`` — which link produced this terminal."""
    status: str
    result_text: str = ""
    result_meta: dict[str, Any] = field(default_factory=dict)
    followed_job_ids: tuple[str, ...] = ()

    @property
    def approved(self) -> bool:
        """True only when a link explicitly approved the product.

        A ``completed`` run job is *not* approval — it means execution finished,
        which is exactly the conflation this chain exists to remove. Approval is
        the formal evaluation's ``final_approved``, or a Ralph loop that stopped
        on :data:`RALPH_APPROVED_STOP_REASON` (every other Ralph stop reason is a
        budget or regression terminal, not a passing verdict).

        Both signals are read **only from the link that owns them**. Result meta
        is merged and forwarded along the chain, so an unqualified
        ``final_approved`` lookup would let a run or a *failed* Ralph inherit an
        approval it never issued — the same "someone else's verdict counts as
        mine" defect at a different layer. A non-terminal status never approves.
        """
        if self.status not in _APPROVING_STATUSES:
            return False
        if self.job_kind == "evaluate":
            return self.result_meta.get("final_approved") is True
        if self.job_kind == "ralph":
            return self.result_meta.get("stop_reason") == RALPH_APPROVED_STOP_REASON
        return False


_JOB_KINDS = ("run", "evaluate", "ralph")


async def follow_run_chain(
    job_manager: _JobManager,
    run_job_id: str,
    *,
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
    deadline_seconds: float | None = None,
) -> ChainTerminal:
    """Await ``run_job_id`` and every successor it hands off to.

    Returns the terminal of the deepest link reached. A link that finishes
    without publishing a successor handle ends the walk — that is the normal
    shape when evaluation approves the product (no Ralph) or when the chain is
    disabled by configuration (no evaluation).

    ``deadline_seconds`` bounds the whole walk; on expiry the last observed
    terminal is returned with ``status="timed_out"`` so the caller keeps the
    handles it needs to resume rather than losing the chain.
    """
    loop = asyncio.get_running_loop()
    expiry = None if deadline_seconds is None else loop.time() + deadline_seconds
    followed: list[str] = []
    job_id = run_job_id
    depth = 0
    terminal: ChainTerminal | None = None

    while True:
        kind = _JOB_KINDS[min(depth, len(_JOB_KINDS) - 1)]
        snapshot = await _await_terminal(job_manager, job_id, poll_seconds, expiry)
        followed.append(job_id)
        if snapshot is None:
            return ChainTerminal(
                job_id=job_id,
                job_kind=kind,
                status="timed_out",
                followed_job_ids=tuple(followed),
            )
        meta = dict(snapshot.result_meta or {})
        terminal = ChainTerminal(
            job_id=job_id,
            job_kind=kind,
            status=str(getattr(snapshot.status, "value", snapshot.status)),
            result_text=snapshot.result_text or "",
            result_meta=meta,
            followed_job_ids=tuple(followed),
        )
        successor = _next_job_id(meta, depth)
        if successor is None:
            return terminal
        job_id = successor
        depth += 1


def _next_job_id(meta: dict[str, Any], depth: int) -> str | None:
    """Return the successor handle published for the link at ``depth``."""
    if depth >= len(SUCCESSOR_META_KEYS):
        return None
    value = meta.get(SUCCESSOR_META_KEYS[depth])
    return value.strip() if isinstance(value, str) and value.strip() else None


async def _await_terminal(
    job_manager: _JobManager,
    job_id: str,
    poll_seconds: float,
    expiry: float | None,
) -> _Snapshot | None:
    """Poll one job until terminal; None when the deadline trips first."""
    loop = asyncio.get_running_loop()
    while True:
        snapshot = await job_manager.get_snapshot(job_id)
        if snapshot.is_terminal:
            return snapshot
        if expiry is not None and loop.time() >= expiry:
            return None
        remaining = poll_seconds if expiry is None else min(poll_seconds, expiry - loop.time())
        await asyncio.sleep(max(0.0, remaining))
