"""Finite budgets for one atomic acceptance-criterion attempt.

The executor cannot rely on every agent runtime exposing the same native
``max_turns`` knob. It therefore enforces two runtime-neutral boundaries at
the stream owner: a wall-clock deadline and cancellation at the first observed
over-budget tool-bearing turn. Some runtimes expose a tool request before its
effect and others report it afterward, so the portable guarantee is that no
later turn is admitted. Text-only progress remains bounded by the deadline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

DEFAULT_MAX_ITERATIONS_PER_AC = 10
DEFAULT_AC_ATTEMPT_TIMEOUT_SECONDS = 900.0


class AttemptBudgetKind(StrEnum):
    """The finite resource that ended an atomic attempt."""

    AGENTIC_STEPS = "agentic_steps"
    WALL_CLOCK = "wall_clock"


@dataclass(frozen=True, slots=True)
class AttemptBudgetExhaustion:
    """Measured boundary that stopped one provider attempt."""

    kind: AttemptBudgetKind
    limit: float
    observed: float

    def __post_init__(self) -> None:
        for name, value in (("limit", self.limit), ("observed", self.observed)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"attempt budget {name} must be numeric")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"attempt budget {name} must be finite and non-negative")
        if self.limit <= 0:
            raise ValueError("attempt budget limit must be greater than zero")


def validate_attempt_budget(
    *,
    max_iterations_per_ac: int,
    timeout_seconds: float,
) -> tuple[int, float]:
    """Validate direct-constructor budget inputs without silently widening them."""

    if type(max_iterations_per_ac) is not int or max_iterations_per_ac < 1:
        raise ValueError("max_iterations_per_ac must be an integer >= 1")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError("ac_attempt_timeout_seconds must be finite and > 0")
    return max_iterations_per_ac, float(timeout_seconds)


__all__ = [
    "AttemptBudgetExhaustion",
    "AttemptBudgetKind",
    "DEFAULT_AC_ATTEMPT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_ITERATIONS_PER_AC",
    "validate_attempt_budget",
]
