"""Finite budgets for one atomic acceptance-criterion attempt.

The executor cannot rely on every agent runtime exposing the same native
``max_turns`` knob. It therefore enforces two runtime-neutral boundaries at
the stream owner: a wall-clock deadline and cancellation at the first observed
over-budget tool-bearing turn. Some runtimes expose a tool request before its
effect and others report it afterward, so the portable guarantee is that no
later turn is admitted. Text-only progress remains bounded by the deadline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
import math

DEFAULT_MAX_ITERATIONS_PER_AC = 10
DEFAULT_AC_ATTEMPT_TIMEOUT_SECONDS = 900.0
ATTEMPT_BUDGET_PROGRESS_SCHEMA_VERSION = 1
_MICROSECONDS_PER_SECOND = 1_000_000
_ATTEMPT_BUDGET_PROGRESS_KEYS = (
    "schema_version",
    "max_agentic_steps",
    "timeout_microseconds",
    "agentic_steps_consumed",
    "remaining_timeout_microseconds",
)
_MISSING_PROGRESS_FIELD = object()


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


@dataclass(frozen=True, slots=True)
class AttemptBudgetProgress:
    """Durable finite-resource state for one paused provider attempt."""

    max_agentic_steps: int
    timeout_microseconds: int
    agentic_steps_consumed: int
    remaining_timeout_microseconds: int

    def __post_init__(self) -> None:
        if type(self.max_agentic_steps) is not int or self.max_agentic_steps < 1:
            raise ValueError("maximum agentic steps must be an integer >= 1")
        if type(self.timeout_microseconds) is not int or self.timeout_microseconds < 1:
            raise ValueError("attempt timeout must be an integer >= 1 microsecond")
        if (
            type(self.agentic_steps_consumed) is not int
            or not 0 <= self.agentic_steps_consumed <= self.max_agentic_steps
        ):
            raise ValueError("consumed agentic steps must be an integer >= 0")
        if (
            type(self.remaining_timeout_microseconds) is not int
            or not 0 <= self.remaining_timeout_microseconds <= self.timeout_microseconds
        ):
            raise ValueError("remaining timeout must be an integer >= 0 microseconds")

    @property
    def remaining_timeout_seconds(self) -> float:
        return self.remaining_timeout_microseconds / _MICROSECONDS_PER_SECOND

    def elapsed_timeout_seconds(self) -> float:
        """Recover a conservative cumulative elapsed value from remaining time."""

        return max(
            0.0,
            (self.timeout_microseconds - self.remaining_timeout_microseconds)
            / _MICROSECONDS_PER_SECOND,
        )

    def to_contract_data(self) -> dict[str, int]:
        return {
            "schema_version": ATTEMPT_BUDGET_PROGRESS_SCHEMA_VERSION,
            "max_agentic_steps": self.max_agentic_steps,
            "timeout_microseconds": self.timeout_microseconds,
            "agentic_steps_consumed": self.agentic_steps_consumed,
            "remaining_timeout_microseconds": self.remaining_timeout_microseconds,
        }

    @classmethod
    def capture(
        cls,
        *,
        agentic_steps_consumed: int,
        elapsed_timeout_seconds: float,
        max_agentic_steps: int,
        timeout_seconds: float,
    ) -> AttemptBudgetProgress:
        """Measure current state without rounding remaining allowance upward."""

        max_agentic_steps, timeout_seconds = validate_attempt_budget(
            max_iterations_per_ac=max_agentic_steps,
            timeout_seconds=timeout_seconds,
        )
        if (
            type(agentic_steps_consumed) is not int
            or not 0 <= agentic_steps_consumed <= max_agentic_steps
        ):
            raise ValueError("consumed agentic steps exceed the configured attempt budget")
        if (
            isinstance(elapsed_timeout_seconds, bool)
            or not isinstance(elapsed_timeout_seconds, (int, float))
            or not math.isfinite(float(elapsed_timeout_seconds))
            or elapsed_timeout_seconds < 0
        ):
            raise ValueError("elapsed attempt time must be finite and non-negative")
        remaining = max(0.0, timeout_seconds - float(elapsed_timeout_seconds))
        timeout_microseconds = math.floor(timeout_seconds * _MICROSECONDS_PER_SECOND)
        if timeout_microseconds < 1:
            raise ValueError("attempt timeout must be at least one microsecond")
        return cls(
            max_agentic_steps=max_agentic_steps,
            timeout_microseconds=timeout_microseconds,
            agentic_steps_consumed=agentic_steps_consumed,
            remaining_timeout_microseconds=math.floor(remaining * _MICROSECONDS_PER_SECOND),
        )

    @classmethod
    def from_contract_data(
        cls,
        value: object,
        *,
        max_agentic_steps: int | None = None,
        timeout_seconds: float | None = None,
    ) -> AttemptBudgetProgress:
        """Validate an exact pause snapshot against its configured ceiling."""

        if not isinstance(value, Mapping):
            raise ValueError("attempt budget progress has an invalid schema")
        try:
            if len(value) != len(_ATTEMPT_BUDGET_PROGRESS_KEYS):
                raise ValueError("attempt budget progress has an invalid schema")
            projected = tuple(
                value.get(key, _MISSING_PROGRESS_FIELD) for key in _ATTEMPT_BUDGET_PROGRESS_KEYS
            )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("attempt budget progress has an invalid schema") from exc
        if any(item is _MISSING_PROGRESS_FIELD for item in projected):
            raise ValueError("attempt budget progress has an invalid schema")
        (
            schema_version,
            persisted_max_steps,
            persisted_timeout,
            steps,
            remaining,
        ) = projected
        if (
            type(schema_version) is not int
            or schema_version != ATTEMPT_BUDGET_PROGRESS_SCHEMA_VERSION
            or type(persisted_max_steps) is not int
            or persisted_max_steps < 1
            or type(persisted_timeout) is not int
            or persisted_timeout < 1
            or type(steps) is not int
            or not 0 <= steps <= persisted_max_steps
            or type(remaining) is not int
            or not 0 <= remaining <= persisted_timeout
        ):
            raise ValueError("attempt budget progress exceeds its configured boundary")
        if max_agentic_steps is not None and max_agentic_steps != persisted_max_steps:
            raise ValueError("attempt budget progress has a different step ceiling")
        if timeout_seconds is not None:
            _validated_steps, validated_timeout = validate_attempt_budget(
                max_iterations_per_ac=persisted_max_steps,
                timeout_seconds=timeout_seconds,
            )
            if math.floor(validated_timeout * _MICROSECONDS_PER_SECOND) != persisted_timeout:
                raise ValueError("attempt budget progress has a different timeout ceiling")
        return cls(
            max_agentic_steps=persisted_max_steps,
            timeout_microseconds=persisted_timeout,
            agentic_steps_consumed=steps,
            remaining_timeout_microseconds=remaining,
        )


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
    "ATTEMPT_BUDGET_PROGRESS_SCHEMA_VERSION",
    "AttemptBudgetExhaustion",
    "AttemptBudgetKind",
    "AttemptBudgetProgress",
    "DEFAULT_AC_ATTEMPT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_ITERATIONS_PER_AC",
    "validate_attempt_budget",
]
