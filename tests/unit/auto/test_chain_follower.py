"""Auto follows the run job's own successor chain instead of owning a second one."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from ouroboros.auto.chain_follower import (
    MISMATCHED_LINK_STATUS,
    RALPH_APPROVED_STOP_REASON,
    ChainTerminal,
    follow_run_chain,
)

# Durable job types for the scripted chain, so tests exercise the same
# link-identity check production reads from persistence.
_DEFAULT_JOB_TYPES = {"job_run": "execute_seed", "job_ev": "evaluate", "job_ralph": "ralph"}


@dataclass
class _FakeSnapshot:
    job_id: str
    status: str
    result_text: str | None = None
    result_meta: dict[str, Any] = field(default_factory=dict)
    terminal: bool = True
    job_type: str = ""

    def __post_init__(self) -> None:
        if not self.job_type:
            self.job_type = _DEFAULT_JOB_TYPES.get(self.job_id, "execute_seed")

    @property
    def is_terminal(self) -> bool:
        return self.terminal


class _FakeJobManager:
    """Serves a scripted snapshot sequence per job id, optionally slowly."""

    def __init__(
        self,
        timelines: dict[str, list[_FakeSnapshot]],
        *,
        read_delay_seconds: dict[str, float] | None = None,
    ) -> None:
        self._timelines = {job_id: list(items) for job_id, items in timelines.items()}
        self._delays = read_delay_seconds or {}
        self.calls: list[str] = []

    async def get_snapshot(self, job_id: str) -> _FakeSnapshot:
        self.calls.append(job_id)
        delay = self._delays.get(job_id)
        if delay:
            await asyncio.sleep(delay)
        timeline = self._timelines[job_id]
        return timeline.pop(0) if len(timeline) > 1 else timeline[0]


@pytest.mark.asyncio
async def test_follows_run_to_evaluate_to_ralph() -> None:
    manager = _FakeJobManager(
        {
            "job_run": [
                _FakeSnapshot("job_run", "completed", "ran", {"chained_evaluate_job_id": "job_ev"})
            ],
            "job_ev": [
                _FakeSnapshot(
                    "job_ev",
                    "completed",
                    "judged",
                    {"final_approved": False, "chained_ralph_job_id": "job_ralph"},
                )
            ],
            "job_ralph": [
                _FakeSnapshot(
                    "job_ralph",
                    "completed",
                    "converged",
                    {"stop_reason": RALPH_APPROVED_STOP_REASON},
                )
            ],
        }
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0)

    assert terminal.job_kind == "ralph"
    assert terminal.job_id == "job_ralph"
    assert terminal.approved is True
    assert terminal.followed_job_ids == ("job_run", "job_ev", "job_ralph")


@pytest.mark.asyncio
async def test_stops_where_the_chain_stops() -> None:
    """An approving evaluation publishes no Ralph handle, so the walk ends there."""
    manager = _FakeJobManager(
        {
            "job_run": [
                _FakeSnapshot("job_run", "completed", "ran", {"chained_evaluate_job_id": "job_ev"})
            ],
            "job_ev": [_FakeSnapshot("job_ev", "completed", "judged", {"final_approved": True})],
        }
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0)

    assert terminal.job_kind == "evaluate"
    assert terminal.approved is True


@pytest.mark.asyncio
async def test_finished_run_alone_is_not_approval() -> None:
    """Execution finishing is not a verdict — that conflation is what the chain removes."""
    manager = _FakeJobManager({"job_run": [_FakeSnapshot("job_run", "completed", "ran", {})]})

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0)

    assert terminal.job_kind == "run"
    assert terminal.approved is False


@pytest.mark.asyncio
async def test_budget_terminal_is_not_approval() -> None:
    manager = _FakeJobManager(
        {
            "job_run": [
                _FakeSnapshot("job_run", "completed", "ran", {"chained_evaluate_job_id": "job_ev"})
            ],
            "job_ev": [
                _FakeSnapshot(
                    "job_ev",
                    "completed",
                    "judged",
                    {"final_approved": False, "chained_ralph_job_id": "job_ralph"},
                )
            ],
            "job_ralph": [
                _FakeSnapshot(
                    "job_ralph", "failed", "gave up", {"stop_reason": "max_generations reached"}
                )
            ],
        }
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0)

    assert terminal.status == "failed"
    assert terminal.approved is False


@pytest.mark.parametrize(
    ("job_kind", "status", "meta"),
    [
        # Result meta is merged and forwarded along the chain, so a link must
        # never inherit a verdict issued by a different link.
        ("run", "completed", {"final_approved": True}),
        ("ralph", "completed", {"final_approved": True}),
        # A link that did not finish cleanly never approves, whatever it carries.
        ("evaluate", "failed", {"final_approved": True}),
        ("ralph", "failed", {"stop_reason": RALPH_APPROVED_STOP_REASON}),
    ],
)
def test_only_the_owning_link_can_approve(job_kind: str, status: str, meta: dict[str, Any]) -> None:
    terminal = ChainTerminal(job_id="job_x", job_kind=job_kind, status=status, result_meta=meta)

    assert terminal.approved is False


@pytest.mark.asyncio
async def test_deadline_preserves_the_deepest_handles_reached() -> None:
    """A timeout deep in the chain must keep every handle needed to resume."""
    manager = _FakeJobManager(
        {
            "job_run": [
                _FakeSnapshot("job_run", "completed", "ran", {"chained_evaluate_job_id": "job_ev"})
            ],
            "job_ev": [
                _FakeSnapshot(
                    "job_ev",
                    "completed",
                    "judged",
                    {"final_approved": False, "chained_ralph_job_id": "job_ralph"},
                )
            ],
            "job_ralph": [_FakeSnapshot("job_ralph", "running", terminal=False)],
        }
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0, deadline_seconds=0.05)

    assert terminal.status == "timed_out"
    assert terminal.job_kind == "ralph"
    assert terminal.followed_job_ids == ("job_run", "job_ev", "job_ralph")
    assert terminal.approved is False


@pytest.mark.asyncio
async def test_a_slow_snapshot_read_cannot_outlive_the_deadline() -> None:
    """The read itself is bounded, not just the interval between reads.

    A snapshot served from persistence or a reconciliation path can block; a
    deadline enforced only after the await would let one read outlive the whole
    budget, or hang forever.
    """
    manager = _FakeJobManager(
        {"job_run": [_FakeSnapshot("job_run", "completed", "ran", {})]},
        read_delay_seconds={"job_run": 5.0},
    )

    terminal = await asyncio.wait_for(
        follow_run_chain(manager, "job_run", poll_seconds=0.0, deadline_seconds=0.05),
        timeout=2.0,
    )

    assert terminal.status == "timed_out"
    assert terminal.followed_job_ids == ("job_run",)


@pytest.mark.asyncio
async def test_a_successor_handle_pointing_at_the_wrong_job_is_refused() -> None:
    """A successor handle is just a string on someone else's result meta.

    A stale or misdirected id can name any completed job; identifying the link
    by traversal depth alone would let that job's ``final_approved`` be read as
    this chain's evaluation verdict.
    """
    manager = _FakeJobManager(
        {
            "job_run": [
                _FakeSnapshot("job_run", "completed", "ran", {"chained_evaluate_job_id": "job_ev"})
            ],
            # The handle resolves to another execution, not an evaluation.
            "job_ev": [
                _FakeSnapshot(
                    "job_ev",
                    "completed",
                    "someone else's run",
                    {"final_approved": True},
                    job_type="execute_seed",
                )
            ],
        }
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0)

    assert terminal.status == MISMATCHED_LINK_STATUS
    assert terminal.approved is False
    assert terminal.result_meta["observed_job_type"] == "execute_seed"


@pytest.mark.asyncio
async def test_deadline_returns_handles_instead_of_losing_the_chain() -> None:
    manager = _FakeJobManager(
        {"job_run": [_FakeSnapshot("job_run", "running", terminal=False)]},
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0, deadline_seconds=0.0)

    assert terminal.status == "timed_out"
    assert terminal.followed_job_ids == ("job_run",)
    assert isinstance(terminal, ChainTerminal)
