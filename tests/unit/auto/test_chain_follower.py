"""Auto follows the run job's own successor chain instead of owning a second one."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ouroboros.auto.chain_follower import (
    RALPH_APPROVED_STOP_REASON,
    ChainTerminal,
    follow_run_chain,
)


@dataclass
class _FakeSnapshot:
    job_id: str
    status: str
    result_text: str | None = None
    result_meta: dict[str, Any] = field(default_factory=dict)
    terminal: bool = True

    @property
    def is_terminal(self) -> bool:
        return self.terminal


class _FakeJobManager:
    """Serves a scripted snapshot sequence per job id."""

    def __init__(self, timelines: dict[str, list[_FakeSnapshot]]) -> None:
        self._timelines = {job_id: list(items) for job_id, items in timelines.items()}
        self.calls: list[str] = []

    async def get_snapshot(self, job_id: str) -> _FakeSnapshot:
        self.calls.append(job_id)
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


@pytest.mark.asyncio
async def test_deadline_returns_handles_instead_of_losing_the_chain() -> None:
    manager = _FakeJobManager(
        {"job_run": [_FakeSnapshot("job_run", "running", terminal=False)]},
    )

    terminal = await follow_run_chain(manager, "job_run", poll_seconds=0.0, deadline_seconds=0.0)

    assert terminal.status == "timed_out"
    assert terminal.followed_job_ids == ("job_run",)
    assert isinstance(terminal, ChainTerminal)
