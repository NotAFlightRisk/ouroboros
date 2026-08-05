"""Finite atomic-attempt budget contracts."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import math

import pytest

from ouroboros.core.attempt_budget import (
    AttemptBudgetExhaustion,
    AttemptBudgetKind,
    AttemptBudgetProgress,
    validate_attempt_budget,
)


def test_validate_attempt_budget_preserves_exact_values() -> None:
    assert validate_attempt_budget(
        max_iterations_per_ac=7,
        timeout_seconds=12.5,
    ) == (7, 12.5)


@pytest.mark.parametrize(
    ("steps", "timeout"),
    ((0, 1.0), (True, 1.0), (1, 0.0), (1, math.inf), (1, True)),
)
def test_validate_attempt_budget_rejects_unbounded_or_ambiguous_values(
    steps: object,
    timeout: object,
) -> None:
    with pytest.raises(ValueError):
        validate_attempt_budget(  # type: ignore[arg-type]
            max_iterations_per_ac=steps,
            timeout_seconds=timeout,
        )


def test_attempt_budget_exhaustion_rejects_invalid_measurement() -> None:
    with pytest.raises(ValueError):
        AttemptBudgetExhaustion(
            kind=AttemptBudgetKind.WALL_CLOCK,
            limit=1,
            observed=math.nan,
        )


class _NoIterationProgressMapping(Mapping[str, object]):
    def __init__(self, values: dict[str, object], *, reported_length: int = 5) -> None:
        self._values = values
        self._reported_length = reported_length

    def __getitem__(self, key: str) -> object:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("durable progress parser must not iterate hostile mappings")

    def __len__(self) -> int:
        return self._reported_length


def test_attempt_budget_progress_reads_only_bounded_exact_fields() -> None:
    progress = AttemptBudgetProgress.from_contract_data(
        _NoIterationProgressMapping(
            AttemptBudgetProgress(
                max_agentic_steps=3,
                timeout_microseconds=10_000_000,
                agentic_steps_consumed=2,
                remaining_timeout_microseconds=4_000_000,
            ).to_contract_data()
        ),
        max_agentic_steps=3,
        timeout_seconds=10,
    )

    assert progress.agentic_steps_consumed == 2
    assert progress.remaining_timeout_microseconds == 4_000_000


def test_attempt_budget_progress_rejects_oversized_mapping_without_iteration() -> None:
    with pytest.raises(ValueError, match="invalid schema"):
        AttemptBudgetProgress.from_contract_data(_NoIterationProgressMapping({}, reported_length=6))


def test_attempt_budget_progress_never_rounds_submicrosecond_timeout_up() -> None:
    with pytest.raises(ValueError, match="at least one microsecond"):
        AttemptBudgetProgress.capture(
            agentic_steps_consumed=0,
            elapsed_timeout_seconds=0,
            max_agentic_steps=1,
            timeout_seconds=0.0000005,
        )
